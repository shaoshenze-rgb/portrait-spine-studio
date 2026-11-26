import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.pipeline.local_matte import matte_portrait_file, remove_background_local


class TestLocalMatte(unittest.TestCase):
    def _make_solid_bg_portrait(self, bg_rgb=(250, 250, 250)):
        h, w = 160, 120
        img = np.zeros((h, w, 4), dtype=np.uint8)
        img[..., :3] = bg_rgb
        img[..., 3] = 255
        # Character body in center (dark coat + bright white shirt/eyes inside!)
        img[25:145, 32:88, :3] = (55, 68, 96)
        # Interior white highlight (must NOT be punched out!)
        img[45:65, 50:70, :3] = (252, 252, 252)
        return img

    def test_transparent_input_passes_through_losslessly(self):
        img = np.zeros((100, 80, 4), dtype=np.uint8)
        img[20:80, 20:60] = (180, 90, 60, 255)
        out, report = remove_background_local(img, prefer_sam=False)
        self.assertTrue(report["already_transparent"])
        self.assertEqual(report["method"], "transparent_passthrough")
        self.assertTrue(np.array_equal(out, img))

    def test_solid_white_background_removed_preserving_interior_white_highlights(self):
        src = self._make_solid_bg_portrait((252, 252, 252))
        out, report = remove_background_local(src, prefer_sam=False)
        self.assertFalse(report["already_transparent"])
        self.assertGreater(report["transparent_ratio"], 0.35)
        # Outer corner must be transparent
        self.assertEqual(int(out[2, 2, 3]), 0)
        self.assertEqual(int(out[155, 115, 3]), 0)
        # Interior white highlight inside the character must remain opaque!
        self.assertGreaterEqual(int(out[55, 60, 3]), 200)

    def test_dark_background_removed_cleanly(self):
        src = self._make_solid_bg_portrait((18, 20, 26))
        src[25:145, 32:88, :3] = (210, 150, 70)
        out, report = remove_background_local(src, prefer_sam=False)
        self.assertEqual(int(out[2, 2, 3]), 0)
        self.assertGreaterEqual(int(out[80, 60, 3]), 200)

    def test_cli_outputs_valid_rgba_and_three_panel_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src_path = root / "opaque.jpg"
            out_path = root / "matted.png"
            chk_path = root / "check.jpg"
            Image.fromarray(self._make_solid_bg_portrait((245, 245, 248))[..., :3], mode="RGB").save(src_path, quality=95)
            proc = subprocess.run(
                [
                    sys.executable,
                    "scripts/local_matte.py",
                    "--input",
                    str(src_path),
                    "--out",
                    str(out_path),
                    "--check-out",
                    str(chk_path),
                    "--no-sam",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            data = json.loads(proc.stdout)
            self.assertTrue(out_path.is_file())
            self.assertTrue(chk_path.is_file())
            self.assertEqual(data["size"], [120, 160])
            self.assertGreater(data["transparent_ratio"], 0.3)
            with Image.open(out_path) as im:
                self.assertEqual(im.mode, "RGBA")
                self.assertEqual(im.size, (120, 160))


if __name__ == "__main__":
    unittest.main()
