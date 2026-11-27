import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from src.gates.preview_diagnostics import (
    _followed_rest_holes, _hole_records, _interior_transparency,
    inspect_dynamic, inspect_exposure_stress,
)
class DynamicChecksTests(unittest.TestCase):
    def test_silhouette_motion_is_not_an_internal_hole(self):
        class Renderer:
            build_report={'moduleContract':{'modules':[]}}
            def pose(self,t):pass
        base=np.zeros((96,96,4),np.uint8);base[16:80,16:80]=255
        moved=np.zeros_like(base);moved[16:80,24:88]=255
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]/'.studio-work') as tmp:
            report=inspect_dynamic(Renderer(),{0.:base,.5:moved},base,tmp)
            self.assertEqual(report['new_gap']['area_px'],0)
            hole=base.copy();hole[36:60,36:60]=0
            report=inspect_dynamic(Renderer(),{0.:base,.5:hole},base,tmp)
            self.assertGreater(report['new_gap']['area_px'],0)
            self.assertEqual(report['new_gap']['time'],.5)

    def test_scans_both_soft_and_local_rigid_edges_without_role_or_coordinates(self):
        class Renderer:
            build_report={'moduleContract':{'modules':[]}}
            def pose(self,time): pass
        base=np.zeros((180,180,4),np.uint8)
        moved=base.copy()
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]/'.studio-work') as tmp:
            root=Path(tmp)
            parts=[]
            for index,(name,kind,local,x,y,color) in enumerate([
                ('silk','soft',False,12,15,(210,70,30,255)),('shoe','rigid',False,42,15,(30,56,150,255)),
                ('inner','rigid',True,92,105,(70,200,50,255)),('outer','rigid',True,122,105,(200,66,56,255)),
                ('hidden','soft',True,42,15,(46,225,8,255))]):
                image=np.full((36,30,4),color,np.uint8)
                Image.fromarray(image).save(root/(name+'.png'))
                parts.append({'name':name,'kind':kind,'enabled':name!='hidden','drawOrder':index,'bbox':[x,y,30,36],
                    'image':name+'.png','canvasLayer':{'provenance':{'local_derivation':{}} if local else {}}})
                if name!='hidden':
                    base[y:y+36,x:x+30]=image
                    offset=4 if name in ('silk','inner') else 0
                    moved[y:y+36,x-offset:x+30-offset]=image
            report=inspect_dynamic(Renderer(),{0.:base,.5:moved,1.:base},base,root,parts_dir=root,material={'parts':parts})
            by_pair={tuple(row['pair']):row for row in report['material_seams']}
            self.assertEqual(by_pair['silk','shoe']['basis'],['soft_layer'])
            self.assertEqual(by_pair['inner','outer']['basis'],['local_split'])
            self.assertTrue(all(row['time']==.5 for row in by_pair.values()))
            self.assertTrue(all(row['bbox'][0] < row['bbox'][2] < 180 for row in by_pair.values()))
            self.assertTrue((root/'seam-worst-crops.jpg').is_file())

    def test_stress_flags_an_open_contact_notch_at_150_percent(self):
        base=np.zeros((110,110,4),np.uint8);base[12:100,12:98]=255
        sheared=base.copy();sheared[12:64,49:58]=0
        calls=[]
        class Renderer:
            def frame(self,time,gain=1.,*,slot_names=None):
                calls.append((gain,slot_names))
                return sheared if time and gain>1 else base
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]/'.studio-work') as tmp:
            directory=Path(tmp)
            items=[]
            for order,(name,role,x) in enumerate([('fabric','cape',12),('leg','lower_leg',55),('staff','weapon',75)]):
                Image.fromarray(np.full((88,43,4),255,np.uint8)).save(directory/(name+'.png'))
                items.append({'name':name,'moduleId':name,'bbox':[x,12,43,88],'image':name+'.png',
                              'enabled':True,'drawOrder':order,'kind':'soft' if name=='fabric' else 'rigid'})
            report,worst=inspect_exposure_stress(Renderer(),directory,{'parts':items},
                [{'id':'fabric','role':'cape'},{'id':'leg','role':'lower_leg'},{'id':'staff','role':'weapon'}],[0.,.5],threshold=120)
        self.assertEqual(report['levels']['1.0x']['peak_exposure_px'],0)
        self.assertGreater(report['levels']['1.5x']['largest_region_px'],120)
        self.assertEqual(report['warnings'][0]['code'],'max_slider_exposure')
        self.assertEqual(set(worst[0]['layers']),{'fabric','leg'})
        self.assertTrue(all('staff' not in names for _,names in calls))

    def test_stress_ignores_pure_outer_silhouette_translation_while_catching_notch(self):
        base = np.zeros((160, 160, 4), np.uint8)
        base[20:140, 20:110] = 255
        # Entire solid body translates 20px right: zero cracks, zero internal gaps!
        shifted = np.zeros_like(base)
        shifted[20:140, 40:130] = 255
        class Renderer:
            def frame(self, time, gain=1., *, slot_names=None):
                return shifted if time else base
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2] / '.studio-work') as tmp:
            directory = Path(tmp)
            items = []
            for order, (name, role, x) in enumerate([('thigh', 'upper_leg', 20), ('shin', 'lower_leg', 65)]):
                Image.fromarray(np.full((120, 45, 4), 255, np.uint8)).save(directory / (name + '.png'))
                items.append({'name': name, 'moduleId': name, 'bbox': [x, 20, 45, 120], 'image': name + '.png',
                              'enabled': True, 'drawOrder': order, 'kind': 'rigid'})
            report, _worst = inspect_exposure_stress(
                Renderer(), directory, {'parts': items},
                [{'id': 'thigh', 'role': 'upper_leg'}, {'id': 'shin', 'role': 'lower_leg'}],
                [0., .5], threshold=120,
            )
        self.assertEqual(report['status'], 'pass')
        self.assertEqual(report['levels']['1.0x']['peak_exposure_px'], 0)
        self.assertEqual(report['levels']['1.5x']['peak_exposure_px'], 0)

    def test_stress_ignores_a_preexisting_transparent_hole_that_moves_with_the_texture(self):
        base = np.zeros((160, 160, 4), np.uint8)
        base[20:140, 20:120] = 255
        base[68:88, 68:88] = 0
        shifted = np.zeros_like(base)
        shifted[20:140, 28:128] = 255
        shifted[68:88, 76:96] = 0

        class Renderer:
            def frame(self, time, gain=1., *, slot_names=None):
                return shifted if time else base

        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2] / '.studio-work') as tmp:
            directory = Path(tmp)
            items = []
            for order, (name, x) in enumerate([('upper', 20), ('lower', 80)]):
                Image.fromarray(np.full((120, 40, 4), 255, np.uint8)).save(directory / (name + '.png'))
                items.append({'name': name, 'moduleId': name, 'bbox': [x, 20, 40, 120], 'image': name + '.png',
                              'enabled': True, 'drawOrder': order, 'kind': 'rigid'})
            report, _worst = inspect_exposure_stress(
                Renderer(), directory, {'parts': items},
                [{'id': 'upper', 'role': 'upper_leg'}, {'id': 'lower', 'role': 'lower_leg'}],
                [0., .5], threshold=1,
            )
        self.assertEqual(report['status'], 'pass')
        self.assertEqual(report['levels']['1.0x']['peak_exposure_px'], 0)
        self.assertEqual(report['levels']['1.5x']['peak_exposure_px'], 0)

    def test_stress_still_flags_a_new_crack_near_a_preexisting_hole(self):
        base = np.zeros((160, 160, 4), np.uint8)
        base[20:140, 20:120] = 255
        base[68:88, 68:88] = 0
        cracked = base.copy()
        cracked[58:78, 76:96] = 0

        class Renderer:
            def frame(self, time, gain=1., *, slot_names=None):
                return cracked if time else base

        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2] / '.studio-work') as tmp:
            directory = Path(tmp)
            items = []
            for order, (name, x) in enumerate([('upper', 20), ('lower', 80)]):
                Image.fromarray(np.full((120, 40, 4), 255, np.uint8)).save(directory / (name + '.png'))
                items.append({'name': name, 'moduleId': name, 'bbox': [x, 20, 40, 120], 'image': name + '.png',
                              'enabled': True, 'drawOrder': order, 'kind': 'rigid'})
            report, _worst = inspect_exposure_stress(
                Renderer(), directory, {'parts': items},
                [{'id': 'upper', 'role': 'upper_leg'}, {'id': 'lower', 'role': 'lower_leg'}],
                [0., .5], threshold=1,
            )
        self.assertEqual(report['status'], 'fail')
        self.assertGreater(report['levels']['1.0x']['largest_region_px'], 1)

    def test_rotated_original_hole_does_not_exempt_an_adjacent_tear(self):
        original = np.full((150, 160), 255, np.uint8)
        original[50:62, 41:101] = 0
        original[86:98, 41:101] = 0
        moved = np.asarray(Image.fromarray(original).rotate(9, fillcolor=255)).copy()
        moved[56:76, 82:102] = 0
        records = _hole_records(*_interior_transparency(original))
        holes, labels = _interior_transparency(moved)
        matched = _followed_rest_holes(records, holes, labels, max_shift=40)
        self.assertGreater(int(matched[75:].sum()), 650)
        self.assertGreater(int((holes & ~matched)[57:76, 82:102].sum()), 300)

    def test_stress_anchor_registration_ignores_translated_original_hole_but_catches_real_crack(self):
        from types import SimpleNamespace

        # 200x200 solid body with an ORIGINAL 24x24 artwork hole inside the torso [70:94, 70:94].
        base = np.zeros((200, 200, 4), np.uint8)
        base[30:170, 30:150] = 255
        base[70:94, 70:94] = 0

        # At t=0.5, the entire anchor bone translates rigidly (+16px right, +10px down).
        shifted = np.zeros_like(base)
        shifted[40:180, 46:166] = 255
        shifted[80:104, 86:110] = 0

        # Same translation PLUS a genuine 22x22 newly torn crack between torso and arm!
        cracked = shifted.copy()
        cracked[110:132, 118:140] = 0

        class AnchorRenderer:
            def __init__(self, frame_t):
                self._frame_t = frame_t
                self.origin_x = 100.0
                self.origin_y = 100.0
                self._canvas_sx = 1.0
                self._canvas_sy = 1.0
                self._bone = SimpleNamespace(a=1.0, b=0.0, c=0.0, d=1.0, worldX=0.0, worldY=0.0)
                self.sk = SimpleNamespace(by_name={'module_torso': self._bone})

            def pose(self, time, gain=1.0):
                if time > 0:
                    # Spine Y is up, canvas Y is down: +10px down => worldY = -10
                    self._bone.worldX = 16.0
                    self._bone.worldY = -10.0
                else:
                    self._bone.worldX = 0.0
                    self._bone.worldY = 0.0

            def frame(self, time, gain=1.0, *, slot_names=None):
                self.pose(time, gain=gain)
                return self._frame_t if time > 0 else base

        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2] / '.studio-work') as tmp:
            directory = Path(tmp)
            items = []
            for order, (name, role, x, w) in enumerate([('torso', 'torso', 30, 80), ('arm', 'arm', 110, 40)]):
                arr = np.full((140, w, 4), 255, np.uint8)
                Image.fromarray(arr).save(directory / (name + '.png'))
                items.append({
                    'name': name, 'moduleId': name, 'bbox': [x, 30, w, 140], 'image': name + '.png',
                    'enabled': True, 'drawOrder': order, 'kind': 'rigid',
                })
            modules = [
                {'id': 'torso', 'role': 'torso', 'parent': None, 'bone': 'module_torso'},
                {'id': 'arm', 'role': 'arm', 'parent': 'torso', 'bone': 'module_arm'},
            ]
            # 1) Pure rigid body translation of an original artwork hole MUST pass with 0 exposure!
            clean_rep, _ = inspect_exposure_stress(
                AnchorRenderer(shifted), directory, {'parts': items}, modules, [0., .5], threshold=120
            )
            self.assertTrue(clean_rep['anchor_registered'])
            self.assertEqual(clean_rep['status'], 'pass')
            self.assertEqual(clean_rep['levels']['1.5x']['largest_region_px'], 0)

            # 2) Real newly torn crack during the SAME rigid translation MUST STILL FAIL!
            bad_rep, _ = inspect_exposure_stress(
                AnchorRenderer(cracked), directory, {'parts': items}, modules, [0., .5], threshold=120
            )
            self.assertEqual(bad_rep['status'], 'fail')
            self.assertGreater(bad_rep['levels']['1.5x']['largest_region_px'], 300)

    def test_stress_filters_one_sided_vacated_air_while_catching_two_sided_seam_notch(self):
        from types import SimpleNamespace

        # Two overlapping soft strips at x=50..110 (left strip x=50..85, right strip x=75..110)
        # with a solid body bar at y=20..40 above them.
        h, w = 160, 180
        base_a = np.zeros((h, w, 4), np.uint8)
        base_b = np.zeros((h, w, 4), np.uint8)
        base_a[20:130, 50:85] = 255
        base_b[20:130, 75:110] = 255

        # Case 1 (one-sided vacated air): BOTH strips sway right (+24px and +50px),
        # leaving vacated exterior air on the left that ONLY strip_a borders (strip_b is far away!).
        sway_a = np.zeros_like(base_a); sway_a[20:130, 74:109] = 255
        sway_b = np.zeros_like(base_b); sway_b[20:130, 125:160] = 255
        sway_a[20:42, 40:130] = 255  # top bridge so closed_pose could bridge exterior wedge

        # Case 2 (real two-sided tear): strip_a stays left [50:74], strip_b moves right [94:129]
        # creating an open notch x=74..94 bordered on BOTH sides by strip_a and strip_b!
        tear_a = np.zeros_like(base_a); tear_a[20:130, 50:74] = 255
        tear_b = np.zeros_like(base_b); tear_b[20:130, 94:129] = 255

        class TwoLayerRenderer:
            def __init__(self, pa, pb):
                self.pa, self.pb = pa, pb
                self.origin_x, self.origin_y = w / 2.0, h / 2.0
                self._canvas_sx, self._canvas_sy = 1.0, 1.0
                self._bone = SimpleNamespace(a=1.0, b=0.0, c=0.0, d=1.0, worldX=0.0, worldY=0.0)
                self.sk = SimpleNamespace(by_name={'root': self._bone})

            def pose(self, time, gain=1.0):
                pass

            def frame(self, time, gain=1.0, *, slot_names=None):
                slots = set(slot_names) if slot_names is not None else {'strip_a', 'strip_b'}
                out = np.zeros((h, w, 4), np.uint8)
                src_a = base_a if time == 0 else self.pa
                src_b = base_b if time == 0 else self.pb
                if 'strip_a' in slots:
                    out = np.maximum(out, src_a)
                if 'strip_b' in slots:
                    out = np.maximum(out, src_b)
                return out

        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2] / '.studio-work') as tmp:
            directory = Path(tmp)
            items = []
            for order, (name, x) in enumerate([('strip_a', 50), ('strip_b', 75)]):
                Image.fromarray(np.full((110, 35, 4), 255, np.uint8)).save(directory / (name + '.png'))
                items.append({
                    'name': name, 'moduleId': name, 'bbox': [x, 20, 35, 110], 'image': name + '.png',
                    'enabled': True, 'drawOrder': order, 'kind': 'soft',
                })
            modules = [
                {'id': 'strip_a', 'role': 'cloth', 'parent': None, 'bone': 'root'},
                {'id': 'strip_b', 'role': 'cloth', 'parent': 'strip_a', 'bone': 'root'},
            ]
            # 1) One-sided vacated exterior air MUST NOT fail!
            sway_rep, _ = inspect_exposure_stress(
                TwoLayerRenderer(sway_a, sway_b), directory, {'parts': items}, modules, [0., .5], threshold=120
            )
            self.assertEqual(sway_rep['status'], 'pass')

            # 2) Real two-sided tear between strip_a and strip_b MUST FAIL!
            tear_rep, _ = inspect_exposure_stress(
                TwoLayerRenderer(tear_a, tear_b), directory, {'parts': items}, modules, [0., .5], threshold=120
            )
            self.assertEqual(tear_rep['status'], 'fail')
            self.assertGreater(tear_rep['levels']['1.5x']['largest_region_px'], 120)

    def test_stress_flags_a_small_expansion_of_a_preexisting_hole(self):
        base = np.zeros((1000, 1000, 4), np.uint8)
        base[100:900, 100:800] = 255
        base[400:500, 400:500] = 0
        expanded = base.copy()
        expanded[398:500, 420:520] = 0

        class Renderer:
            def frame(self, time, gain=1., *, slot_names=None):
                return expanded if time else base

        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2] / '.studio-work') as tmp:
            directory = Path(tmp)
            items = []
            for order, (name, x) in enumerate([('upper', 100), ('lower', 500)]):
                Image.fromarray(np.full((800, 400, 4), 255, np.uint8)).save(directory / (name + '.png'))
                items.append({'name': name, 'moduleId': name, 'bbox': [x, 100, 400, 800], 'image': name + '.png',
                              'enabled': True, 'drawOrder': order, 'kind': 'rigid'})
            report, _worst = inspect_exposure_stress(
                Renderer(), directory, {'parts': items},
                [{'id': 'upper', 'role': 'upper_leg'}, {'id': 'lower', 'role': 'lower_leg'}],
                [0., .5], threshold=120,
            )
        self.assertEqual(report['status'], 'fail')
        self.assertGreater(report['levels']['1.0x']['largest_region_px'], 120)

    def test_anchor_unwarper_prefers_body_over_leading_floating_prop_or_vfx(self):
        from types import SimpleNamespace
        from src.gates.preview_diagnostics import _build_anchor_unwarper

        def fake_bone(wx=0.0, wy=0.0):
            return SimpleNamespace(a=1.0, b=0.0, c=0.0, d=1.0, worldX=wx, worldY=wy)

        sk = SimpleNamespace(by_name={
            "module_orb": fake_bone(10.0, 20.0),
            "module_vortex": fake_bone(-15.0, 30.0),
            "module_foot": fake_bone(0.0, 0.0),
            "module_torso": fake_bone(0.0, 5.0),
        })
        ren = SimpleNamespace(
            sk=sk,
            pose=lambda _t, gain=1.0: None,
            origin_x=100.0, origin_y=100.0,
            _canvas_sx=1.0, _canvas_sy=1.0,
        )
        modules = [
            {"id": "orb", "role": "prop", "parent": None, "bone": "module_orb"},
            {"id": "vortex", "role": "effect_group", "parent": None, "bone": "module_vortex"},
            {"id": "foot", "role": "foot", "parent": None, "bone": "module_foot"},
            {"id": "torso", "role": "torso", "parent": None, "bone": "module_torso"},
        ]
        anchor_bone, _ = _build_anchor_unwarper(ren, modules)
        self.assertEqual(anchor_bone, "module_torso")
