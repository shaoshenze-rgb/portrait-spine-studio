import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from src.pipeline.local_body_split import BodyLayer, refine_mask_to_edges, snap_contour_band, split_local_body


class LocalBodySplitTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((32, 40, 4), dtype=np.uint8)
        self.image[5:26, 3:20] = (45, 100, 196, 255)
        self.image[5:26, 20:37] = (235, 146, 24, 255)
        self.back = np.zeros(self.image.shape[:2], dtype=bool)
        self.front = self.back.copy()
        self.back[5:26, 3:24] = True
        self.front[5:26, 20:37] = True

    def split(self, **options):
        return split_local_body(self.image, [BodyLayer('front', self.front, 20),
                                             BodyLayer('back', self.back, 10)], **options)

    def test_front_is_sharp_and_depth_order_not_input_order(self):
        split = self.split()
        np.testing.assert_array_equal(split.ownership['front'], self.front)
        self.assertFalse((split.ownership['back'] & split.ownership['front']).any())
        self.assertFalse(split.padding['front'].any())
        self.assertFalse(split.rgba['front'][~self.front].any())
        np.testing.assert_array_equal(split.rgba['front'][self.front], self.image[self.front])
        self.assertEqual(split.stats['proposed_overlap_px'], 84)

    def test_padding_only_uses_lower_layer_rgb_and_stays_under_opaque_front(self):
        self.image[5:8, 20:30, 3] = 200
        split = self.split(padding_px=8)
        pad = split.padding['back']
        self.assertTrue(pad[15, 27])
        self.assertFalse(pad[15, 28])
        self.assertFalse(pad[:8].any())
        self.assertFalse((pad & ~split.ownership['front']).any())
        self.assertLess(split.rgba['back'][15, 27, 3], 255)
        np.testing.assert_array_equal(split.rgba['back'][pad, :3],
                                      np.tile((45, 100, 196), (pad.sum(), 1)))

    def test_noise_removed_and_hole_filled_without_taking_other_claim(self):
        self.image[1, 1] = (1, 2, 3, 255)
        self.front[1, 1] = True
        self.front[12:14, 30:32] = False
        self.front[14:16, 33:35] = False
        self.back[13:16, 32:35] = True
        split = self.split(min_island_px=4, max_hole_px=8)
        self.assertFalse(split.ownership['front'][1, 1])
        self.assertFalse(split.rgba['front'][1, 1].any())
        self.assertTrue(split.ownership['front'][12:14, 30:32].all())
        self.assertTrue(split.ownership['back'][14, 33])
        self.assertEqual(split.stats['layers']['front']['discarded_island_px'], 1)
        self.assertEqual(split.stats['unclaimed_px'], 1)

    def test_reports_distant_soft_islands_and_exact_overlap_counts(self):
        mask = np.zeros((32, 40), dtype=bool)
        mask[8:14, 5:9] = True
        mask[18:24, 30:34] = True
        split = split_local_body(self.image, [BodyLayer('cloth', mask, 3, soft=True),
                                             BodyLayer('body', self.image[..., 3] > 0, 2)],
                                 soft_island_px=12, soft_island_gap_px=10)
        self.assertEqual(split.stats['warnings'][0]['code'], 'separated_soft_islands')
        self.assertEqual(split.stats['layers']['cloth']['owned_px'], 48)
        underlap = split.stats['layers']['body']['padding_under']
        self.assertEqual(underlap['cloth'], split.padding['body'].sum())

    def test_small_diagonally_connected_run_counts_as_one_island(self):
        mask = np.eye(5, dtype=bool)
        source = np.zeros((5, 5, 4), dtype=np.uint8)
        source[mask] = 255
        split = split_local_body(source, [BodyLayer('edge', mask, 1)], min_island_px=5)
        np.testing.assert_array_equal(split.ownership['edge'], mask)

    def test_hidden_bridge_connects_behind_front_without_changing_rest_pixels(self):
        image = np.zeros((60,110,4), np.uint8)
        image[12:48,8:102] = (195,93,31,255)
        image[12:48,30:80] = (62,90,190,255)
        image[12:48,28:30] = (12,9,7,255)
        image[12:48,80:82] = (12,9,7,255)
        front = np.zeros(image.shape[:2], bool); front[12:48,30:80] = True
        back = (image[...,3]>0) & ~front
        inputs = [BodyLayer('back',back,1,soft=True),BodyLayer('front',front,2)]
        with patch.dict(sys.modules, {'cv2':None}):
            split = split_local_body(image, inputs, padding_px=30, soft_island_px=500)
        self.assertGreater(split.stats['layers']['back']['occluded_bridge_px'], 1400)
        self.assertEqual(split.stats['warnings'], [])
        self.assertFalse((split.padding['back'] & ~front).any())
        np.testing.assert_array_equal(split.rgba['back'][back],image[back])
        np.testing.assert_array_equal(split.rgba['back'][25:38,42:67,:3], np.broadcast_to([195,93,31],(13,25,3)))
        combined = Image.alpha_composite(Image.fromarray(split.rgba['back']),Image.fromarray(split.rgba['front']))
        np.testing.assert_array_equal(np.asarray(combined),image)


class EdgeRefinementTests(unittest.TestCase):
    def setUp(self):
        self.image = np.full((68, 84, 4), (31, 37, 55, 255), dtype=np.uint8)
        self.image[10:60, 20:61] = (180, 135, 41, 255)
        self.image[:3, :, 3] = 0
        self.seed = np.zeros(self.image.shape[:2], dtype=bool)
        self.seed[10:60, 25:66] = True

    def assert_grabcut_refinement(self, refined):
        self.assertGreater(refined[16:54, 22:25].mean(), .85)
        self.assertLess(refined[16:54, 63:66].mean(), .15)
        self.assertTrue(refined[20:46, 32:56].all())
        self.assertFalse(refined[:3].any())

    def test_seed_refinement_reaches_color_boundary(self):
        self.assert_grabcut_refinement(refine_mask_to_edges(self.image, self.seed, band_px=9))

    def test_seed_refinement_without_cv2(self):
        with patch.dict(sys.modules, {'cv2': None}):
            self.assert_grabcut_refinement(refine_mask_to_edges(self.image, self.seed, band_px=9))

    def test_contour_follows_perceptual_edges_instead_of_guide_center(self):
        mask = snap_contour_band(self.image, [(25, 14), (25, 55)], [(65, 14), (65, 55)], radius=8)
        self.assertTrue(mask[14:56, 22:59].all())
        self.assertFalse(mask[:, :18].any())
        self.assertFalse(mask[:, 64:].any())
        self.assertFalse(mask[:14].any())
        with self.assertRaisesRegex(ValueError, 'identical rows'):
            snap_contour_band(self.image, [(25, 14), (25, 55)], [(65, 14), (65, 56)])

    def test_cli_writes_refined_rgba_and_checked_ownership(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=repo / '.studio-work') as tmp:
            root = Path(tmp)
            Image.fromarray(self.image).save(root / 'body.png')
            Image.fromarray(self.seed.astype(np.uint8) * 255).save(root / 'seed.png')
            spec = {'version': 1, 'layers': [{'id': 'flap', 'kind': 'soft', 'draw_order': 5,
                    'seed_mask': 'seed.png', 'band_px': 9}]}
            (root / 'spec.json').write_text(json.dumps(spec))
            result = subprocess.run([sys.executable, str(repo / 'scripts/split_local_body.py'), '--body', str(root / 'body.png'),
                                     '--spec', str(root / 'spec.json'), '--out-dir', str(root / 'out')],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((root / 'out/split-stats.json').read_text())
            layer = report['layers']['flap']
            pixels = (root / 'out' / layer['image']).read_bytes()
            self.assertEqual(layer['sha256'], hashlib.sha256(pixels).hexdigest())
            ownership = (root / 'out' / layer['owner_mask']).read_bytes()
            self.assertEqual(layer['owner_mask_sha256'], hashlib.sha256(ownership).hexdigest())
            self.assertEqual(report['parent_aligned_sha256'], hashlib.sha256((root / 'body.png').read_bytes()).hexdigest())


if __name__ == '__main__':
    unittest.main()
