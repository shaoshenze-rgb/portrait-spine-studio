"""Independent overlapping materials reach the existing renderer, not an export gate."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.pipeline.canvas_layer_parts import materialize_canvas_layer_parts
from src.pipeline.build_rig import build_rig
from src.render.renderer import SpineRenderer
from src.gates.review_binding import ensure_export_approved, ExportApprovalError

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CanvasLayerPartsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT / '.studio-work', prefix='layer-parts-test-')
        self.addCleanup(self.tmp.cleanup); self.root = Path(self.tmp.name)
        self.source = self.root / 'source.png'
        Image.new('RGBA', (40, 40), (40, 50, 60, 255)).save(self.source)
        self.arrays = []
        self.plan = {'version': 1, 'material_mode': 'canvas_layers', 'image_size': [40, 40],
                     'source_image': str(self.source), 'source_sha256': sha(self.source), 'pieces': []}
        for i, (name, rect, color) in enumerate([
                ('body', (4, 4, 30, 30), (25, 110, 180, 255)),
                ('effect', (14, 10, 35, 34), (240, 90, 20, 128))]):
            arr = np.zeros((40, 40, 4), np.uint8)
            x0, y0, x1, y1 = rect; arr[y0:y1, x0:x1] = color
            path = self.root / (name + '.png'); Image.fromarray(arr).save(path); self.arrays.append(arr)
            self.plan['pieces'].append({'id': name, 'module_id': name, 'kind': 'rigid',
                'parent': None if i == 0 else 'body', 'pivot': [20, 20], 'draw_order': i,
                'image': str(path), 'sha256': sha(path), 'provenance': {
                    'scene_id': 'synthetic', 'node_id': name, 'task_id': 'synthetic-test',
                    'operation': 'split-layers', 'source_asset_sha256': sha(self.source),
                    'result_asset_sha256': sha(path), 'transform': [[1, 0, 0], [0, 1, 0]]}})

    def emit(self, plan=None):
        return materialize_canvas_layer_parts(plan or self.plan, self.root / 'parts', evidence_root=self.root)

    def test_overlap_and_alpha_are_preserved_without_source_slicing(self):
        manifest = self.emit()
        expected = Image.new('RGBA', (40, 40))
        for arr, part in zip(self.arrays, manifest['parts']):
            expected = Image.alpha_composite(expected, Image.fromarray(arr))
            x, y, w, h = part['bbox']
            np.testing.assert_array_equal(np.asarray(Image.open(self.root / 'parts' / part['image'])), arr[y:y+h, x:x+w])
        np.testing.assert_array_equal(np.asarray(Image.open(manifest['source'])), np.asarray(expected))
        self.assertEqual(manifest['originalSourceSHA256'], sha(self.source))
        self.assertNotEqual(manifest['sourceSHA256'], manifest['originalSourceSHA256'])
        self.assertFalse(manifest['productionIntakeReady']); self.assertFalse(manifest['exportAllowed'])
        self.assertFalse((self.root / 'parts/labels.npy').exists())
        self.assertEqual(json.loads((self.root / 'parts/source_comparison.json').read_text())['status'], 'pending')

    def test_existing_rig_renderer_consumes_standard_parts_but_export_stays_blocked(self):
        manifest = self.emit()
        rig = {'character': 'synthetic_layers', 'duration': 8, 'fps': 30, 'bones': [
            {'name': 'root', 'pos': [20, 40]}, {'name': 'body', 'parent': 'root', 'pos': [20, 20]},
            {'name': 'effect', 'parent': 'body', 'pos': [20, 20]}],
            'skin': {name: {'type': 'region', 'bone': name} for name in ('body', 'effect')}}
        build_rig(self.root / 'parts', rig, self.root / 'rig', include_animation=False)
        actual = SpineRenderer(self.root / 'rig').frame(None)
        expected = np.asarray(Image.open(manifest['source'])).astype(float)
        expected[..., :3] *= expected[..., 3:4] / 255
        self.assertLessEqual(np.abs(actual.astype(float)-np.rint(expected)).max(), 2)
        with self.assertRaises(ExportApprovalError):
            ensure_export_approved(self.root / 'rig')

    def test_bad_identity_format_or_size_fails_before_publication(self):
        for case in ('source_hash', 'layer_hash', 'rgb', 'size'):
            plan = copy.deepcopy(self.plan)
            if case == 'source_hash': plan['source_sha256'] = '0' * 64
            elif case == 'layer_hash': plan['pieces'][0]['sha256'] = '0' * 64
            else:
                path = self.root / (case + '.png')
                Image.new('RGB' if case == 'rgb' else 'RGBA', (40, 40) if case == 'rgb' else (8, 8)).save(path)
                plan['pieces'][0].update(image=str(path), sha256=sha(path))
            with self.subTest(case=case), self.assertRaises((ValueError, ExportApprovalError)):
                self.emit(plan)
            self.assertFalse((self.root / 'parts').exists())

    def test_existing_output_is_not_overwritten_by_retry(self):
        self.emit(); path = self.root / 'parts/parts.json'; before = path.read_bytes()
        with self.assertRaisesRegex(ValueError, 'new output'): self.emit()
        self.assertEqual(path.read_bytes(), before)

    def test_missing_parent_and_nonrigid_registration_fail(self):
        for case in ('parent', 'cycle', 'reflection', 'shear'):
            plan = copy.deepcopy(self.plan)
            if case == 'parent': plan['pieces'][1]['parent'] = 'absent'
            elif case == 'cycle': plan['pieces'][0]['parent'] = 'effect'
            elif case == 'reflection': plan['pieces'][0]['provenance']['transform'] = [[-1, 0, 0], [0, 1, 0]]
            else: plan['pieces'][0]['provenance']['transform'] = [[1, .2, 0], [0, 1, 0]]
            with self.subTest(case=case), self.assertRaises(ValueError): self.emit(plan)
            self.assertFalse((self.root / 'parts').exists())


if __name__ == '__main__':
    unittest.main()
