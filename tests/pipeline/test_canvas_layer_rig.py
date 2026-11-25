"""Canvas binding candidates reuse typed motion and cannot become approved exports."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.pipeline.canvas_layer_parts import materialize_canvas_layer_parts
from src.pipeline.canvas_layer_rig import compile_canvas_layer_bindings
from src.pipeline.build_rig import build_rig
from src.render.renderer import SpineRenderer
from src.gates.review_binding import ensure_export_approved, ExportApprovalError

ROOT = Path(__file__).resolve().parents[2]


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


class CanvasLayerRigTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(dir=ROOT/'.studio-work', prefix='canvas-binding-test-')
        self.addCleanup(tmp.cleanup); self.root = Path(tmp.name)
        source = self.root/'original.png'; Image.new('RGBA', (64, 64), (30, 40, 50, 255)).save(source)
        material = {'version': 1, 'material_mode': 'canvas_layers', 'source_image': str(source),
                    'source_sha256': sha(source), 'image_size': [64, 64], 'pieces': []}
        for i, (name, rect, color, kind) in enumerate([
                ('body', (8, 8, 40, 54), (30, 110, 180, 255), 'rigid'),
                ('flame', (20, 28, 52, 60), (250, 90, 20, 128), 'soft'),
                ('head', (24, 0, 40, 18), (150, 100, 90, 255), 'rigid')]):
            rgba = np.zeros((64, 64, 4), np.uint8); x0, y0, x1, y1 = rect
            rgba[y0:y1, x0:x1] = color; p = self.root/(name+'.png'); Image.fromarray(rgba).save(p)
            material['pieces'].append({'id': name, 'module_id': name, 'kind': kind,
                'parent': None if i == 0 else 'body', 'pivot': [32, 12] if name == 'head' else [32, 32], 'draw_order': i,
                'image': str(p), 'sha256': sha(p), 'provenance': {'scene_id': 'test',
                    'node_id': name, 'task_id': 'test', 'operation': 'split-layers',
                    'source_asset_sha256': sha(source), 'result_asset_sha256': sha(p),
                    'transform': [[1, 0, 0], [0, 1, 0]]}})
        self.parts = self.root/'parts'; materialize_canvas_layer_parts(material, self.parts, evidence_root=self.root)
        self.plan = {'pose_type': 'airborne', 'anatomy': 'humanoid_bust', 'uncertainties': [], 'modules': [
            {'id': 'body', 'role': 'torso', 'motion_type': 'rigid', 'parent': None,
             'pivot': [.5, .5], 'motion': 'breath', 'confidence': .95},
            {'id': 'flame', 'role': 'effect_group', 'motion_type': 'soft', 'parent': 'body',
             'pivot': [.5, .5], 'centerline': [[.5, .5], [.6, .65], [.7, .8]],
             'motion': 'soft_bend', 'locked_root': True, 'confidence': .95},
            {'id': 'head', 'role': 'head', 'motion_type': 'rigid', 'parent': 'body',
             'pivot': [.5, 12/64], 'motion': 'sway', 'confidence': .95}],
            'connections': [{'id': 'attach', 'type': 'attachment', 'a': 'body', 'b': 'flame', 'point': [.5, .5]},
                            {'id': 'neck', 'type': 'joint', 'a': 'body', 'b': 'head', 'point': [.5, 12/64]}]}

    def compile(self, plan=None):
        return compile_canvas_layer_bindings(self.parts, plan or self.plan, self.root/'binding', evidence_root=self.root)

    def test_typed_rigid_soft_binding_real_renderer_and_export_block(self):
        spec = self.compile()
        made = build_rig(self.parts, spec, self.root/'rig')
        self.assertEqual(made['report']['generationMode'], 'canvas_layers_binding_candidate')
        renderer = SpineRenderer(self.root/'rig')
        np.testing.assert_array_equal(renderer.frame(0), renderer.frame(8))
        renderer.pose(2)
        matrix = renderer.sk.by_name['module_body'].matrix()[:, :2]
        np.testing.assert_allclose(matrix.T @ matrix, np.eye(2), atol=1e-8)
        self.assertIn('bend_flame1', renderer.sk.by_name); self.assertIn('bend_flame2', renderer.sk.by_name)
        projection = spec['moduleContract']
        self.assertFalse(projection['productionIntakeReady']); self.assertFalse(projection['exportAllowed'])
        with self.assertRaisesRegex(ExportApprovalError, 'Canvas独立层绑定候选'):
            ensure_export_approved(self.root/'rig')

    def test_pivot_move_and_grounded_downgrade_are_not_accepted(self):
        for case in ('pivot', 'grounded'):
            plan = copy.deepcopy(self.plan)
            if case == 'pivot': plan['modules'][1]['pivot'] = [.6, .5]
            else: plan['pose_type'] = 'grounded'
            with self.subTest(case=case), self.assertRaises(ValueError): self.compile(plan)
            self.assertFalse((self.root/'binding').exists())

    def test_matching_but_outside_support_pivot_hits_existing_placement_gate(self):
        path = self.parts/'parts.json'; material = json.loads(path.read_text())
        material['parts'][0]['pivot'] = [50, 32]; path.write_text(json.dumps(material))
        plan = copy.deepcopy(self.plan); plan['modules'][0]['pivot'] = [50/64, .5]
        with self.assertRaisesRegex(ValueError, 'pivot.*掩码外'): self.compile(plan)
        self.assertFalse((self.root/'binding').exists())

    def test_changed_crop_cannot_be_rebound_as_original_layer(self):
        path = self.parts/'parts/body.png'; rgba = np.asarray(Image.open(path)).copy()
        rgba[2, 2, 0] ^= 255; Image.fromarray(rgba).save(path)
        manifest = self.parts/'parts.json'; material = json.loads(manifest.read_text())
        material['parts'][0]['imageSHA256'] = sha(path); manifest.write_text(json.dumps(material))
        with self.assertRaisesRegex(ValueError, 'differs from registered'): self.compile()
        self.assertFalse((self.root/'binding').exists())

    def test_declared_occlusions_issues_and_refinements_are_never_dropped(self):
        polygon = [[.4, .4], [.6, .4], [.6, .6], [.4, .6]]
        declarations = {
            'occlusions': [{'front': 'flame', 'back': 'body', 'polygons': [polygon], 'completion': 'none'}],
            'input_issues': [{'id': 'issue', 'kind': 'source_damage', 'polygons': [polygon], 'notes': 'observed missing structure'}],
            'boundary_refinements': [{'id': 'edge', 'a': 'body', 'b': 'flame', 'axis': 'x', 'polygons': [polygon], 'notes': 'declared boundary correction'}],
        }
        for field, declaration in declarations.items():
            plan = copy.deepcopy(self.plan); plan[field] = declaration; before = copy.deepcopy(plan)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field): self.compile(plan)
            self.assertEqual(plan, before)
            self.assertFalse((self.root/'binding').exists())
        for field in declarations:
            plan = copy.deepcopy(self.plan); plan[field] = None
            with self.subTest(malformed=field), self.assertRaisesRegex(ValueError, field): self.compile(plan)
        plan = copy.deepcopy(self.plan); plan['unrecognized_declaration'] = True
        with self.assertRaisesRegex(ValueError, 'unsupported fields'): self.compile(plan)

    def test_explicit_empty_declarations_keep_supported_binding_path(self):
        plan = copy.deepcopy(self.plan)
        plan.update(occlusions=[], input_issues=[], boundary_refinements=[])
        result = self.compile(plan)
        self.assertEqual(result['moduleContract']['materialMode'], 'canvas_layers')
        self.assertFalse(result['moduleContract']['exportAllowed'])

    def test_preview_mode_preserves_point_and_lists_placement_warning(self):
        path = self.parts/'parts.json'; material = json.loads(path.read_text())
        material['parts'][0]['pivot'] = [50, 32]; path.write_text(json.dumps(material))
        plan = copy.deepcopy(self.plan); plan['modules'][0]['pivot'] = [50/64, .5]
        result = compile_canvas_layer_bindings(self.parts, plan, self.root/'preview',
            evidence_root=self.root, character='preview_test', strict_placement=False)
        self.assertEqual(result['character'], 'preview_test')
        self.assertEqual(result['bones'][1]['pos'], [50., 32.])
        self.assertTrue(result['placementWarnings'])
        build_rig(self.parts, result, self.root/'preview-rig')
        renderer = SpineRenderer(self.root/'preview-rig')
        np.testing.assert_array_equal(renderer.frame(0), renderer.frame(8))

    def test_preview_mode_keeps_input_issues_and_completion_as_advisories(self):
        plan = copy.deepcopy(self.plan)
        plan['input_issues'] = [{'notes': 'visible flame needs review'}]
        plan['occlusions'] = [{'front': 'flame', 'back': 'body', 'completion': 'local',
            'polygons': [[[.4, .4], [.6, .4], [.6, .6], [.4, .6]]]}]
        result = compile_canvas_layer_bindings(self.parts, plan, self.root/'preview',
            evidence_root=self.root, strict_placement=False)
        notes = result['moduleContract']['advisoryNotes']
        self.assertEqual(notes['input_issues'], plan['input_issues'])
        self.assertEqual(notes['occlusions'], plan['occlusions'])
        self.assertFalse(result['moduleContract']['exportAllowed'])

    def test_preview_grounded_bust_is_not_silently_changed_to_airborne(self):
        plan = copy.deepcopy(self.plan); plan['pose_type'] = 'grounded'
        result = compile_canvas_layer_bindings(self.parts, plan, self.root/'preview',
            evidence_root=self.root, strict_placement=False)
        self.assertEqual(result['pose_type'], 'grounded')
        build_rig(self.parts, result, self.root/'preview-rig')

    def test_preview_grounded_full_keeps_foot_anchor_and_rigid_ik(self):
        from src.studio.vision_schema import rasterize_polygons
        root = self.root/'grounded'; root.mkdir(); w, h = 128, 160
        shapes = {'body': [(50,40),(78,40),(78,78),(50,78)],
                  'head': [(54,10),(74,10),(74,40),(54,40)],
                  'upper': [(56,78),(67,78),(52,112),(41,108)],
                  'lower': [(41,108),(52,112),(68,145),(56,145)],
                  'foot': [(54,145),(74,145),(74,156),(54,156)]}
        roles = dict(body='torso', head='head', upper='upper_leg', lower='lower_leg', foot='foot')
        parents = dict(body=None, head='body', upper='body', lower='upper', foot=None)
        pivots = dict(body=[64,75], head=[64,40], upper=[61,78], lower=[47,109], foot=[62,145])
        source = root/'source.png'; Image.new('RGBA', (w,h)).save(source)
        material = {'version':1, 'material_mode':'canvas_layers', 'source_image':str(source),
                    'source_sha256':sha(source), 'image_size':[w,h], 'pieces':[]}
        modules = []
        for order, (name, polygon) in enumerate(shapes.items()):
            mask = rasterize_polygons([[[x/w,y/h] for x,y in polygon]], w, h)
            rgba = np.zeros((h,w,4),np.uint8); rgba[mask] = [30+order*35,70,140,255]
            path = root/(name+'.png'); Image.fromarray(rgba).save(path)
            kind = 'static' if name == 'foot' else 'rigid'
            material['pieces'].append({'id':name, 'module_id':name, 'kind':'rigid', 'parent':parents[name],
                'pivot':pivots[name], 'draw_order':order, 'image':str(path), 'sha256':sha(path),
                'provenance':{'scene_id':'test','node_id':name,'task_id':'test','operation':'split-layers',
                    'source_asset_sha256':sha(source),'result_asset_sha256':sha(path),'transform':[[1,0,0],[0,1,0]]}})
            m = {'id':name,'role':roles[name],'motion_type':kind,'parent':parents[name],
                 'pivot':[pivots[name][0]/w,pivots[name][1]/h],'confidence':.95}
            if name == 'foot': m['static_reason'] = 'declared ground contact'
            modules.append(m)
        parts = root/'parts'; materialize_canvas_layer_parts(material, parts, evidence_root=root)
        connections = [{'id':n, 'type':'joint','a':parents[n],'b':n,'point':m['pivot']}
                       for n,m in zip(shapes,modules) if parents[n]]
        connections.append({'id':'anchor','type':'foot_anchor','a':'upper','b':'foot',
                            'point':[62/w,145/h],'chain':['upper','lower']})
        plan = {'pose_type':'grounded','anatomy':'humanoid_full','modules':modules,'connections':connections}
        spec = compile_canvas_layer_bindings(parts, plan, root/'binding', evidence_root=root, strict_placement=False)
        made = build_rig(parts, spec, root/'rig')
        self.assertEqual(spec['moduleContract']['contacts'][-1]['type'], 'foot')
        self.assertEqual(len(spec['ik']), 1)
        self.assertLess(made['report']['moduleMotionPreflight']['rigidAnchorMaxDriftPx'], .001)
        renderer = SpineRenderer(root/'rig'); positions = []
        for t in np.linspace(0,8,17):
            renderer.pose(float(t)); foot = renderer.sk.by_name['module_foot']
            positions.append([foot.worldX, foot.worldY])
        self.assertLess(np.ptp(positions, axis=0).max(), 1e-8)


if __name__ == '__main__': unittest.main()
