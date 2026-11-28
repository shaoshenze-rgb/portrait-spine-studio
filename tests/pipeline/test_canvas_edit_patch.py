import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.pipeline.canvas_edit_patch import (
    clean_open_air_halo,
    compose_canvas_edit_patch,
    plan_occluded_edit_crop,
)
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


    def test_opaque_checkerboard_with_warp_oob_border_still_strips_checkerboard(self):
        # Simulates Hero 17: opaque checkerboard edit slightly smaller than target canvas after warp,
        # so cv2.warpAffine adds >10% transparent out-of-bounds margin; MUST NOT mistake OOB margin for native RGBA!
        h, w = 320, 320
        full_body = _paint_textured_body(h, w)
        cloak_mask = np.zeros((h, w), dtype=bool)
        cloak_mask[110:210, 50:135] = True
        original = full_body.copy()
        original[cloak_mask] = [160, 35, 190, 255]
        base_layer = full_body.copy()
        base_layer[cloak_mask] = 0

        # Crop edit tightly around body (x=40..260, y=35..285) so warp leaves >20% OOB border around full 320x320 canvas!
        checker = _make_checkerboard(h, w, tile=16)
        edit_full = checker.copy()
        edit_full[full_body[..., 3] > 0] = full_body[full_body[..., 3] > 0, :3]
        cropped_edit = edit_full[35:285, 40:260]

        res = compose_canvas_edit_patch(base_layer, original, cloak_mask, cropped_edit, alpha_mode="auto")
        self.assertEqual(res.stats["alpha_source"], "exterior_checkerboard_strip")
        self.assertEqual(int(res.rgba[150, 60, 3]), 0)
        self.assertGreater(int(res.rgba[150, 110, 3]), 240)
        self.assertEqual(res.stats["backing_diagnosis"], "usable_backing")

    def test_replace_occluded_overwrites_opaque_and_translucent_ghost_losslessly(self):
        # Case 1: opaque uncut torso base under an overlapping wing/scarf; replace_occluded=False writes 0,
        # while replace_occluded=True cleanly replaces the occluded zone with outside_diff == 0.
        h, w = 256, 256
        full_body = _paint_textured_body(h, w)
        wing_mask = np.zeros((h, w), dtype=bool)
        wing_mask[100:180, 95:165] = True

        original = full_body.copy()
        original[wing_mask] = [240, 240, 240, 255]  # bright white wing painted over torso

        # Base is uncut torso (100% opaque inside wing_mask with wing pixels still baked in)
        uncut_base = original.copy()
        edit_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        edit_rgb[full_body[..., 3] > 0] = full_body[full_body[..., 3] > 0, :3]

        res_default = compose_canvas_edit_patch(
            uncut_base, original, wing_mask, edit_rgb, crop_bbox=(0, 0, w, h), replace_occluded=False
        )
        self.assertEqual(res_default.stats["patched_px"], 0)

        res_replace = compose_canvas_edit_patch(
            uncut_base, original, wing_mask, edit_rgb, crop_bbox=(0, 0, w, h), replace_occluded=True, feather_px=3
        )
        self.assertEqual(res_replace.stats["max_outside_diff"], 0)
        np.testing.assert_array_equal(res_replace.rgba[~wing_mask], uncut_base[~wing_mask])
        self.assertGreater(res_replace.stats["patched_px"], 4000)
        # Center of wing_mask now shows torso color, not white wing color!
        mae_center = float(
            np.abs(res_replace.rgba[115:165, 110:150, :3].astype(np.float32) - full_body[115:165, 110:150, :3].astype(np.float32)).mean()
        )
        self.assertLess(mae_center, 8.0)

        # Case 2: translucent glow with semi-transparent bottle imprint under mask + inpaint_alpha + replace_occluded
        yy, xx = np.indices((h, w))
        dist = np.hypot(xx - 128, yy - 128)
        glow_a = np.clip((1.0 - dist / 95.0) * 200.0, 0, 200).astype(np.uint8)
        base_glow = np.zeros((h, w, 4), dtype=np.uint8)
        base_glow[..., :3] = [45, 220, 95]
        base_glow[..., 3] = glow_a
        bottle = (xx >= 110) & (xx < 146) & (yy >= 110) & (yy < 146)
        base_glow[bottle] = [20, 90, 20, 140]  # dark semi-transparent bottle ghost

        edit_glow = np.full((h, w, 3), [48, 222, 98], dtype=np.uint8)
        res_fx = compose_canvas_edit_patch(
            base_glow, base_glow, bottle, edit_glow,
            crop_bbox=(0, 0, w, h), alpha_mode="inpaint_alpha", replace_occluded=True, feather_px=3
        )
        self.assertEqual(res_fx.stats["max_outside_diff"], 0)
        self.assertGreater(int(res_fx.rgba[128, 128, 1]), 180)  # replaced dark ghost with bright green glow

    def test_open_air_no_backing_diagnosis_and_halo_cleanup(self):
        h, w = 256, 256
        full_body = _paint_textured_body(h, w)
        # Floating orb in open air next to the body (x=25..75, y=90..150), with soft purple glow halo bleeding onto x=76..86
        orb_mask = np.zeros((h, w), dtype=bool)
        orb_mask[95:145, 25:75] = True
        front_orb = np.zeros((h, w, 4), dtype=np.uint8)
        front_orb[orb_mask] = [190, 60, 240, 255]

        base_with_halo = full_body.copy()
        halo_region = np.zeros((h, w), dtype=bool)
        halo_region[95:145, 68:80] = True
        base_with_halo[halo_region] = [180, 70, 230, 120]  # soft translucent glow halo stuck to base

        original = base_with_halo.copy()
        original[orb_mask] = front_orb[orb_mask]

        # Canvas edit removed the orb cleanly -> pure black background inside orb_mask
        edit_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        edit_rgb[full_body[..., 3] > 0] = full_body[full_body[..., 3] > 0, :3]

        res = compose_canvas_edit_patch(base_with_halo, original, orb_mask, edit_rgb, crop_bbox=(0, 0, w, h))
        self.assertEqual(res.stats["backing_diagnosis"], "open_air_no_backing")
        self.assertEqual(res.stats["recommended_next_action"], "clean_open_air_halo")

        cleaned = clean_open_air_halo(base_with_halo, front_orb, radius_px=12, max_halo_alpha=215)
        self.assertTrue(cleaned.stats["solid_core_preserved"])
        self.assertEqual(cleaned.stats["max_solid_core_diff"], 0)
        self.assertGreater(cleaned.stats["stripped_halo_px"], 400)
        self.assertEqual(int(cleaned.rgba[120, 74, 3]), 0)
        # Solid torso core pixels right next to the halo remain untouched
        self.assertEqual(int(cleaned.rgba[120, 95, 3]), 255)
        np.testing.assert_array_equal(cleaned.rgba[100:200, 90:220], full_body[100:200, 90:220])

    def test_auto_prepare_edit_crop_and_cli_roundtrip_with_crop_meta(self):
        h, w = 320, 320
        full_body = _paint_textured_body(h, w)
        front_mask = np.zeros((h, w), dtype=bool)
        front_mask[120:170, 110:155] = True
        front_layer = np.zeros((h, w, 4), dtype=np.uint8)
        front_layer[front_mask] = [210, 50, 180, 255]
        original = full_body.copy()
        original[front_mask] = front_layer[front_mask]
        base_layer = full_body.copy()
        base_layer[front_mask] = 0

        diagnostics = {
            "exposure_stress": {
                "status": "fail",
                "threshold_px": 50,
                "levels": {
                    "1.0x": {
                        "frames": [
                            {"time": 1.0, "area_px": 400, "regions": [{"area_px": 400, "bbox": [112, 125, 150, 165], "layers": ["scarf", "torso"]}]}
                        ]
                    }
                },
            }
        }

        with tempfile.TemporaryDirectory() as tmp:
            td = Path(tmp)
            Image.fromarray(original).save(td / "orig.png")
            Image.fromarray(front_layer).save(td / "front.png")
            Image.fromarray(base_layer).save(td / "base.png")
            (td / "diag.json").write_text(json.dumps(diagnostics))

            p1 = subprocess.run(
                [
                    sys.executable, "scripts/canvas_handoff.py", "prepare-edit-crop",
                    "--original", str(td / "orig.png"),
                    "--front-layer", str(td / "front.png"),
                    "--base-layer", str(td / "base.png"),
                    "--diagnostics", str(td / "diag.json"),
                    "--layer-pair", "scarf,torso",
                    "--min-pad-px", "40",
                    "--min-crop-edge", "140",
                    "--out-crop", str(td / "crop.png"),
                    "--out-mask", str(td / "mask.png"),
                    "--out-meta", str(td / "meta.json"),
                ],
                capture_output=True, text=True, check=True,
            )
            meta = json.loads(p1.stdout)
            self.assertTrue(meta["ok"])
            self.assertEqual(meta["source"], "diagnostics_exposure_regions")
            cx, cy, cw, ch = meta["crop_bbox"]
            self.assertGreaterEqual(cw, 140)
            self.assertGreaterEqual(ch, 140)

            # Simulate Canvas editing only that local crop (cleanly removing scarf and showing body)
            edit_crop_rgb = np.zeros((ch, cw, 3), dtype=np.uint8)
            sub_body = full_body[cy:cy + ch, cx:cx + cw]
            edit_crop_rgb[sub_body[..., 3] > 0] = sub_body[sub_body[..., 3] > 0, :3]
            Image.fromarray(edit_crop_rgb).save(td / "edit_crop.png")

            p2 = subprocess.run(
                [
                    sys.executable, "scripts/canvas_handoff.py", "apply-edit-patch",
                    "--base-layer", str(td / "base.png"),
                    "--original", str(td / "orig.png"),
                    "--edit-image", str(td / "edit_crop.png"),
                    "--crop-meta", str(td / "meta.json"),
                    "--out", str(td / "patched.png"),
                    "--report-out", str(td / "patch_report.json"),
                ],
                capture_output=True, text=True, check=True,
            )
            rep = json.loads(p2.stdout)
            self.assertTrue(rep["ok"])
            self.assertEqual(rep["stats"]["max_outside_diff"], 0)
            self.assertEqual(rep["stats"]["backing_diagnosis"], "usable_backing")
            self.assertGreater(rep["stats"]["patched_px"], 1500)


if __name__ == "__main__":
    unittest.main()
