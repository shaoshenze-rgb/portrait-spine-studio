import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from src.gates.preview_diagnostics import inspect_dynamic, inspect_exposure_stress
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
