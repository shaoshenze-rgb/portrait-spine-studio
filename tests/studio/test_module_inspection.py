"""Review artifact generation preserves source and groups visible pieces by module."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.inspect_module_plan import inspect_plan
from tests.pipeline.test_complete_modules import fixture

ROOT = Path(__file__).resolve().parents[2]


class TestModuleInspection(unittest.TestCase):
    def setUp(self):
        work = ROOT / '.studio-work'; work.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=work, prefix='module-inspect-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.image = self.root / 'source.png'
        self.plan = fixture(self.image); self.analysis = self.root / 'analysis.json'
        self.analysis.write_text(json.dumps(self.plan))

    def test_all_disconnected_pieces_share_one_unreviewed_module_image(self):
        before = hashlib.sha256(self.image.read_bytes()).hexdigest()
        result = inspect_plan(self.image, self.analysis, self.root / 'inspect')
        self.assertEqual(result['review_status'], 'unreviewed')
        self.assertEqual(result['module_count'], len(self.plan['modules']))
        blade = next(m for m in result['modules'] if m['id'] == 'blade')
        self.assertEqual(len(blade['pieces']), 2)
        with Image.open(blade['isolated_image']) as image:
            pixels = np.asarray(image)
            self.assertEqual(int((pixels[..., 3] > 0).sum()), blade['visible_pixels'])
            self.assertGreater(blade['visible_pixels'], 0)
        self.assertTrue(Path(blade['comparison_image']).is_file())
        self.assertEqual(hashlib.sha256(self.image.read_bytes()).hexdigest(), before)
        self.assertTrue(all(m['review_status'] == 'unreviewed' for m in result['modules']))

    def test_wrong_source_or_legacy_does_not_generate_review_as_if_valid(self):
        self.plan['input_sha256'] = '0' * 64
        self.analysis.write_text(json.dumps(self.plan))
        with self.assertRaises(ValueError): inspect_plan(self.image, self.analysis, self.root / 'wrong')
        self.assertFalse((self.root / 'wrong/module-review.json').exists())
        self.plan['version'] = 1; self.analysis.write_text(json.dumps(self.plan))
        with self.assertRaises(ValueError): inspect_plan(self.image, self.analysis, self.root / 'legacy')


if __name__ == '__main__': unittest.main()
