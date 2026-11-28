import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.pipeline.canvas_edit_patch import compose_canvas_edit_patch
from src.pipeline.canvas_layer_cleanup import check_canvas_fallback_policy


def _make_checkerboard(h: int, w: int, tile: int = 16) -> np.ndarray:
    yy, xx = np.indices((h, w))
    even = ((yy // tile) + (xx // tile)) % 2 == 0
    bg = np.where(even[..., None], 246, 202).astype(np.uint8)
    return np.repeat(bg, 3, axis=2)


def _paint_textured_body(h: int, w: int) -> np.ndarray:
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rng = np.random.default_rng(42)
    yy, xx = np.indices((h, w))
    body = (xx >= 80) & (xx < 240) & (yy >= 50) & (yy < 270)
    r = (95 + 45 * np.sin(xx * 0.14) + 30 * np.cos(yy * 0.11) + rng.integers(-12, 13, (h, w))).clip(20, 235)
    g = (70 + 35 * np.cos((xx + yy) * 0.09) + rng.integers(-10, 11, (h, w))).clip(20, 235)
    b = (120 + 50 * np.sin((xx - yy) * 0.12) + rng.integers(-12, 13, (h, w))).clip(20, 235)
    rgba[body, 0] = r[body].astype(np.uint8)
    rgba[body, 1] = g[body].astype(np.uint8)
    rgba[body, 2] = b[body].astype(np.uint8)
    rgba[body, 3] = 255
    for gy in range(70, 250, 30):
        stripe = body & (yy >= gy) & (yy < gy + 6) & ((xx // 12) % 2 == 0)
        rgba[stripe] = [230, 185, 60, 255]
    return rgba


class CanvasEditPatchTests(unittest.TestCase):
    def test_opaque_checkerboard_edit_strips_bg_and_fills_occluded_backing_losslessly(self):
        h, w = 320, 320
        full_body = _paint_textured_body(h, w)
        cloak_mask = np.zeros((h, w), dtype=bool)
        cloak_mask[110:210, 50:135] = True

        original = full_body.copy()
        original[cloak_mask] = [160, 35, 190, 255]

        base_layer = full_body.copy()
        base_layer[cloak_mask] = 0

        checker = _make_checkerboard(h, w, tile=16)
        edit_rgb = checker.copy()
        body_mask = full_body[..., 3] > 0
        shifted = np.clip(full_body[..., :3].astype(np.int16) + np.array([16, 12, -10]), 0, 255).astype(np.uint8)
        edit_rgb[body_mask] = shifted[body_mask]
        edit_rgb[70:110, 190:230] = [255, 20, 20]
        edit_pil = Image.fromarray(edit_rgb).resize((240, 240), Image.Resampling.BILINEAR)

        res = compose_canvas_edit_patch(
            base_layer,
            original,
            cloak_mask,
            edit_pil,
            alpha_mode="auto",
            feather_px=4,
            color_match=True,
        )

        self.assertTrue(res.stats["rest_pose_preserved"])
        self.assertEqual(res.stats["max_outside_diff"], 0)
        np.testing.assert_array_equal(res.rgba[~cloak_mask], base_layer[~cloak_mask])
        self.assertEqual(res.stats["alpha_source"], "exterior_checkerboard_strip")
        self.assertEqual(int(res.rgba[150, 60, 3]), 0)
        self.assertGreater(int(res.rgba[150, 110, 3]), 240)
        self.assertGreater(res.stats["patched_px"], 4000)
        mae_patched = float(
            np.abs(res.rgba[125:195, 95:125, :3].astype(np.float32) - full_body[125:195, 95:125, :3].astype(np.float32)).mean()
        )
        self.assertLess(mae_patched, 12.0, f"Color-matched backing MAE too high: {mae_patched}")

    def test_translucent_fx_inpaint_alpha_fills_bottle_hole_seamlessly(self):
        h, w = 240, 240
        yy, xx = np.indices((h, w))
        dist = np.hypot(xx - 120, yy - 120)
        glow_a = np.clip((1.0 - dist / 95.0) * 210.0, 0, 210).astype(np.uint8)
        full_glow = np.zeros((h, w, 4), dtype=np.uint8)
        full_glow[..., 0] = 45
        full_glow[..., 1] = 235
        full_glow[..., 2] = 110
        full_glow[..., 3] = glow_a

        bottle_mask = (xx >= 102) & (xx < 138) & (yy >= 102) & (yy < 138)
        base_glow = full_glow.copy()
        base_glow[bottle_mask] = 0

        edit_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        edit_rgb[..., 0] = 52
        edit_rgb[..., 1] = 230
        edit_rgb[..., 2] = 115

        res = compose_canvas_edit_patch(
            base_glow,
            full_glow,
            bottle_mask,
            edit_rgb,
            crop_bbox=(0, 0, w, h),
            alpha_mode="inpaint_alpha",
            feather_px=4,
            color_match=True,
        )

        np.testing.assert_array_equal(res.rgba[~bottle_mask], base_glow[~bottle_mask])
        center_a = int(res.rgba[120, 120, 3])
        self.assertGreater(center_a, 140)
        self.assertLess(center_a, 230)
        self.assertEqual(res.stats["alpha_source"], "inpaint_alpha")

    def test_cli_apply_edit_patch_produces_valid_canvas_provenance(self):
        h, w = 256, 256
        full_body = _paint_textured_body(h, w)
        front_mask = np.zeros((h, w), dtype=bool)
        front_mask[100:180, 90:150] = True
        front_layer = np.zeros((h, w, 4), dtype=np.uint8)
        front_layer[front_mask] = [180, 40, 200, 255]
        original = full_body.copy()
        original[front_mask] = front_layer[front_mask]
        base_layer = full_body.copy()
        base_layer[front_mask] = 0

        edit_rgb = np.full((h, w, 3), 40, dtype=np.uint8)
        edit_rgb[full_body[..., 3] > 0] = full_body[full_body[..., 3] > 0, :3]

        with tempfile.TemporaryDirectory() as tmp:
            td = Path(tmp)
            Image.fromarray(base_layer).save(td / "base.png")
            Image.fromarray(original).save(td / "orig.png")
            Image.fromarray(front_layer).save(td / "front.png")
            Image.fromarray(edit_rgb).save(td / "edit.png")

            proc = subprocess.run(
                [
                    sys.executable,
                    "scripts/canvas_handoff.py",
                    "apply-edit-patch",
                    "--base-layer", str(td / "base.png"),
                    "--original", str(td / "orig.png"),
                    "--front-layer", str(td / "front.png"),
                    "--edit-image", str(td / "edit.png"),
                    "--out", str(td / "patched.png"),
                    "--check-out", str(td / "check.jpg"),
                    "--report-out", str(td / "report.json"),
                    "--scene-id", "character-flow",
                    "--node-id", "image-slot-demo",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            rep = json.loads(proc.stdout)
            self.assertTrue(rep["ok"])
            self.assertTrue((td / "patched.png").exists())
            self.assertTrue((td / "check.jpg").exists())
            self.assertGreater(rep["stats"]["patched_px"], 3000)

            pol = check_canvas_fallback_policy([{"name": "torso", "enabled": True, "provenance": rep["provenance"]}])
            self.assertTrue(pol["passed"], pol)
            self.assertFalse(pol["bypassed"], pol)
            self.assertEqual(pol["canvas_layers"], ["torso"])


if __name__ == "__main__":
    unittest.main()
