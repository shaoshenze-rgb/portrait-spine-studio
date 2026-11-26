import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from src.export.export_profile import (
    DEFAULT_EXPORT_PROFILE,
    compute_canvas_transform,
    export_job_with_profile,
    fit_frame_to_profile,
    normalize_export_profile,
    scale_spine_rig_to_profile,
)
from src.pipeline.build_rig import build_rig
from src.pipeline.preprocess import MAX_LONG_EDGE_4K, preprocess_portrait
from src.render.renderer import SpineRenderer, find_ffmpeg, straight_alpha


def _make_minimal_rig(root: Path, *, width: int = 120, height: int = 160, character: str = "hero_test") -> Path:
    parts_dir = root / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    body = np.zeros((height, width, 4), dtype=np.uint8)
    x0, y0, x1, y1 = width // 5, height // 8, width - width // 5, height - height // 10
    body[y0:y1, x0:x1] = (210, 145, 70, 255)
    body_path = parts_dir / "torso.png"
    Image.fromarray(body[y0:y1, x0:x1]).save(body_path)
    src_path = root / "matted.png"
    Image.fromarray(body).save(src_path)

    parts_meta = {
        "canvas": [width, height],
        "source": str(src_path),
        "parts": [
            {
                "name": "torso",
                "moduleId": "torso",
                "pieceId": "torso",
                "drawOrder": 0,
                "kind": "rigid",
                "parent": None,
                "pivot": [width // 2, int(height * 0.72)],
                "enabled": True,
                "bbox": [x0, y0, x1 - x0, y1 - y0],
                "image": "torso.png",
            }
        ],
    }
    (parts_dir / "parts.json").write_text(json.dumps(parts_meta), encoding="utf-8")

    rig_spec = {
        "version": 1,
        "character": character,
        "canvas": [width, height],
        "origin": [width / 2.0, float(height)],
        "body_height_px": y1 - y0,
        "bones": [
            {"name": "root", "parent": None, "pos": [width / 2.0, float(height)], "aim": None},
            {"name": "hip", "parent": "root", "pos": [width / 2.0, float(height * 0.72)], "aim": [width / 2.0, float(height * 0.45)], "role": "pelvis"},
            {"name": "torso", "parent": "hip", "pos": [width / 2.0, float(height * 0.55)], "aim": [width / 2.0, float(height * 0.25)], "role": "torso"},
        ],
        "slots": [
            {"name": "torso", "bone": "torso", "attachment": "torso", "part": "torso"},
        ],
        "ik": [],
        "chains": {}, "motion": {"hip": {"ty": [[4.0, 2, 0.0]]}},
    }
    rig_dir = root / "rig"
    build_rig(parts_dir, rig_spec, rig_dir, preset="agile_full", gain=1.0)
    return rig_dir


class TestExportProfile4K(unittest.TestCase):
    def test_default_profile_is_4k_portrait_and_transparent_web(self):
        prof = normalize_export_profile()
        self.assertEqual(prof["size_preset"], "4k_portrait")
        self.assertEqual(prof["width"], 2160)
        self.assertEqual(prof["height"], 3840)
        self.assertEqual(prof["align"], "bottom_center")
        self.assertEqual(prof["formats"], ["transparent_web"])
        self.assertEqual(prof["fps"], 30)
        self.assertEqual(DEFAULT_EXPORT_PROFILE["size_preset"], "4k_portrait")

    def test_4k_presets_and_upscale_policy(self):
        tr_4k_v = compute_canvas_transform(1000, 1800, {"size_preset": "4k_portrait", "upscale_policy": "allow_upscale"})
        self.assertEqual((tr_4k_v["out_w"], tr_4k_v["out_h"]), (2160, 3840))
        self.assertTrue(tr_4k_v["upscaled"])
        self.assertGreater(tr_4k_v["scale"], 1.9)
        self.assertAlmostEqual(
            tr_4k_v["offset_y"] + tr_4k_v["scaled_h"],
            round(3840 * (1.0 - 0.03)),
            delta=2,
        )

        tr_no_up = compute_canvas_transform(1000, 1800, {"size_preset": "4k_portrait", "upscale_policy": "no_upscale"})
        self.assertEqual((tr_no_up["out_w"], tr_no_up["out_h"]), (2160, 3840))
        self.assertEqual(tr_no_up["scale"], 1.0)
        self.assertFalse(tr_no_up["upscaled"])

        tr_4k_h = compute_canvas_transform(1920, 1080, {"size_preset": "4k_landscape"})
        self.assertEqual((tr_4k_h["out_w"], tr_4k_h["out_h"]), (3840, 2160))

        tr_edge = compute_canvas_transform(1500, 2000, {"size_preset": "4k_long_edge"})
        self.assertEqual(max(tr_edge["out_w"], tr_edge["out_h"]), 4096)
        self.assertEqual(tr_edge["out_w"] % 2, 0)
        self.assertEqual(tr_edge["out_h"] % 2, 0)

    def test_fit_frame_premultiplied_lanczos_has_no_dark_edge_halo(self):
        frame = np.zeros((100, 100, 4), dtype=np.uint8)
        frame[20:80, 20:80] = (230, 150, 50, 255)
        fitted, tr = fit_frame_to_profile(frame, {"size_preset": "custom", "width": 240, "height": 240, "align": "center"})
        self.assertEqual(fitted.shape, (240, 240, 4))
        self.assertTrue(tr["upscaled"])
        self.assertTrue((fitted[..., :3] <= fitted[..., 3:4]).all())
        st = straight_alpha(fitted)
        semi = (st[..., 3] > 30) & (st[..., 3] < 245)
        if semi.any():
            self.assertGreater(float(st[semi, 0].mean()), 190.0)
            self.assertGreater(float(st[semi, 1].mean()), 115.0)

    def test_preprocess_preserves_4k_resolution_with_4096_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "portrait_4k.png"
            dst = root / "out.png"
            img = Image.new("RGBA", (2160, 3840), (120, 140, 180, 255))
            img.save(src)
            res = preprocess_portrait(src, dst, max_long_edge=MAX_LONG_EDGE_4K)
            self.assertFalse(res["resized"])
            self.assertEqual(res["new_size"], [2160, 3840])
            with Image.open(dst) as saved:
                self.assertEqual(saved.size, (2160, 3840))

    def test_build_rig_4k_height_scale_not_clamped_at_135(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rig_1k = _make_minimal_rig(root / "r1", width=600, height=1080, character="c1k")
            rig_4k = _make_minimal_rig(root / "r4", width=1800, height=3240, character="c4k")
            d1 = json.loads((rig_1k / "c1k.spine.json").read_text(encoding="utf-8"))
            d4 = json.loads((rig_4k / "c4k.spine.json").read_text(encoding="utf-8"))
            t1 = (d1["animations"]["idle"]["bones"].get("hip") or d1["animations"]["idle"]["bones"]["torso"]).get("translate") or []
            t4 = (d4["animations"]["idle"]["bones"].get("hip") or d4["animations"]["idle"]["bones"]["torso"]).get("translate") or []
            max_y1 = max((abs(float(k.get("y", 0.0))) for k in t1), default=0.0)
            max_y4 = max((abs(float(k.get("y", 0.0))) for k in t4), default=0.0)
            self.assertGreater(max_y1, 0.1)
            self.assertGreater(max_y4 / max_y1, 2.2)

    def test_export_job_with_profile_4k_transparent_web_and_scaled_spine_zip(self):
        ff = find_ffmpeg()
        if not ff:
            self.skipTest("ffmpeg binary unavailable")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rig_dir = _make_minimal_rig(root / "work", width=120, height=160, character="hero4k")
            signoff = root / "user-signoff.json"
            signoff.write_text(json.dumps({"decision": "pass", "reviewer": "tester"}), encoding="utf-8")

            out_dir = root / "export_4k"
            res = export_job_with_profile(
                rig_dir,
                out_dir,
                {
                    "size_preset": "4k_portrait",
                    "formats": ["transparent_web", "spine_package"],
                    "fps": 30,
                },
                character="hero4k",
                user_signoff_path=signoff,
                ffmpeg_bin=ff,
                max_frames=2,
                require_approved=True,
            )
            self.assertEqual(res["target_size"], [2160, 3840])
            self.assertTrue((out_dir / "hero4k-poster.webp").is_file())
            self.assertTrue((out_dir / "hero4k-alpha-vp9.webm").is_file())
            self.assertTrue((out_dir / "hero4k-packed-h264.mp4").is_file())
            self.assertTrue((out_dir / "hero4k-spine42.zip").is_file())
            self.assertGreater((out_dir / "hero4k-alpha-vp9.webm").stat().st_size, 500)
            self.assertGreater((out_dir / "hero4k-packed-h264.mp4").stat().st_size, 500)

            with Image.open(out_dir / "hero4k-poster.webp") as poster:
                self.assertEqual(poster.size, (2160, 3840))
                self.assertEqual(poster.mode, "RGBA")

            with zipfile.ZipFile(out_dir / "hero4k-spine42.zip", "r") as zf:
                spine_json = json.loads(zf.read("hero4k.spine.json").decode("utf-8"))
                self.assertEqual(spine_json["skeleton"]["width"], 2160)
                self.assertEqual(spine_json["skeleton"]["height"], 3840)

            scaled_dir = root / "scaled_1080"
            scale_spine_rig_to_profile(rig_dir, scaled_dir, {"size_preset": "1080p_portrait"})
            ren = SpineRenderer(scaled_dir)
            self.assertEqual((ren.width, ren.height), (1080, 1920))
            f0 = ren.frame(0.0)
            self.assertEqual(f0.shape, (1920, 1080, 4))
            self.assertTrue((f0[..., 3] > 200).any())


if __name__ == "__main__":
    unittest.main()
