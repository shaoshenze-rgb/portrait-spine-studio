"""Wave 1 mask contract, preprocess RGB freeze, locked_root, standalone HTML."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.export.exporter import export_standalone_html
from src.gates.module_quality import ModuleQualityAuditor
from src.pipeline.module_rig import LOCKED_ROOT_DEG, resolve_module_motion, soft_bend_deg_for_link
from src.pipeline.preprocess import preprocess_portrait
from src.studio.module_contract import materialize_module_contract, validate_module_contract
from src.studio.vision_schema import VisionValidationError, VisionReviewRequired
from tests.pipeline.test_complete_modules import fixture

ROOT = Path(__file__).resolve().parents[2]


def _png_mask(mask: np.ndarray) -> tuple[str, str]:
    im = Image.fromarray(np.where(mask, 255, 0).astype(np.uint8))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    raw = buf.getvalue()
    return base64.b64encode(raw).decode("ascii"), hashlib.sha256(raw).hexdigest()


class TestMaskContractAndStandalone(unittest.TestCase):
    def setUp(self):
        work = ROOT / ".studio-work"
        work.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=work, prefix="mask-contract-test-")
        self.addCleanup(self.tmp.cleanup)
        self.d = Path(self.tmp.name)
        self.image = self.d / "src.png"
        self.contract = fixture(self.image)

    def test_mask_png_parses_and_sha_or_size_mismatch_is_rejected(self):
        with Image.open(self.image) as im:
            rgba = np.asarray(im.convert("RGBA"))
        h, w = rgba.shape[:2]
        body = (rgba[..., 1] > 140) & (rgba[..., 3] > 0)
        b64, digest = _png_mask(body)
        c = json.loads(json.dumps(self.contract))
        piece = c["modules"][0]["pieces"][0]
        piece["mask_png_base64"] = b64
        piece["mask_sha256"] = digest
        validate_module_contract(c, input_sha256=c["input_sha256"], image_size=(w, h))
        bad = json.loads(json.dumps(c))
        bad["modules"][0]["pieces"][0]["mask_sha256"] = "0" * 64
        with self.assertRaises(VisionValidationError):
            validate_module_contract(bad, input_sha256=c["input_sha256"], image_size=(w, h))
        tiny = np.zeros((4, 4), dtype=bool)
        tiny[1:3, 1:3] = True
        tb64, tdig = _png_mask(tiny)
        bad = json.loads(json.dumps(c))
        bad["modules"][0]["pieces"][0]["mask_png_base64"] = tb64
        bad["modules"][0]["pieces"][0]["mask_sha256"] = tdig
        with self.assertRaises(VisionValidationError):
            validate_module_contract(bad, input_sha256=c["input_sha256"], image_size=(w, h))

    def test_mask_derive_band_rest_diff_is_zero(self):
        from src.studio.vision_schema import vision_result_to_semantic_hints
        from src.studio.auto_spec import build_default_specs
        from src.pipeline.split_parts import split_parts
        from src.pipeline.build_rig import build_rig

        self.contract = fixture(self.image, local_fill=True)
        with Image.open(self.image) as im:
            rgba = np.asarray(im.convert("RGBA")).copy()
        h, w = rgba.shape[:2]
        # Exclusive masks from the painted rectangles: later draw_order owns overlaps.
        c = json.loads(json.dumps(self.contract))
        claimed = np.zeros((h, w), dtype=np.int32)
        entries = []
        for mi, module in enumerate(c["modules"]):
            for piece in module["pieces"]:
                entries.append((piece["draw_order"], mi, piece))
        entries.sort()
        for order, mi, piece in entries:
            poly = piece["polygons"][0]
            from src.studio.vision_schema import rasterize_polygons
            mask = rasterize_polygons([poly], w, h) & (rgba[..., 3] > 0)
            claimed[mask] = order + 1
        for order, mi, piece in entries:
            mask = claimed == (order + 1)
            b64, digest = _png_mask(mask)
            piece["mask_png_base64"] = b64
            piece["mask_sha256"] = digest
        c["occlusions"][0]["derive_band"] = True
        hints = materialize_module_contract(c, image_path=self.image, mask_dir=self.d / "masks")
        self.assertIn("bandDerivation", hints["module_projection"]["occlusions"][0])
        ps, rs = build_default_specs(self.image, character="maskband", semantic_hints=hints)
        split = split_parts(self.image, ps["parts"], self.d / "parts", min_island_px=0, feather_px=0,
                            module_contract=ps["module_contract"])
        self.assertEqual(split["rest_diff"]["maxRGBA"], 0)
        build_rig(self.d / "parts", rs, self.d / "rig")

    def test_preprocess_zeros_transparent_rgb_and_keeps_visible_bytes(self):
        src = self.d / "dirty.png"
        arr = np.zeros((32, 24, 4), dtype=np.uint8)
        arr[4:12, 6:18] = [17, 34, 51, 200]
        arr[0, 0] = [9, 8, 7, 0]
        arr[31, 23] = [255, 1, 2, 0]
        Image.fromarray(arr).save(src)
        out = self.d / "clean.png"
        preprocess_portrait(src, out)
        got = np.asarray(Image.open(out).convert("RGBA"))
        np.testing.assert_array_equal(got[4:12, 6:18], arr[4:12, 6:18])
        np.testing.assert_array_equal(got[0, 0], [0, 0, 0, 0])
        np.testing.assert_array_equal(got[31, 23], [0, 0, 0, 0])

    def test_locked_root_first_link_is_quarter_degree_and_tail_bends(self):
        from src.studio.vision_schema import vision_result_to_semantic_hints
        from src.studio.auto_spec import build_default_specs
        from src.pipeline.split_parts import split_parts
        from src.pipeline.build_rig import build_rig
        from src.render.renderer import SpineRenderer

        w, h = 128, 160
        with Image.open(self.image) as im:
            rgba = np.asarray(im.convert("RGBA")).copy()
        rgba[74:98, 80:96] = [255, 90, 40, 180]
        Image.fromarray(rgba).save(self.image)
        c = json.loads(json.dumps(self.contract))
        c["input_sha256"] = hashlib.sha256(self.image.read_bytes()).hexdigest()
        flame = {
            "id": "flame",
            "role": "effect_group",
            "motion_type": "soft",
            "parent": "hand",
            "pivot": [88 / w, 74 / h],
            "motion": "soft_bend",
            "locked_root": True,
            "centerline": [[88 / w, 74 / h], [88 / w, 86 / h], [88 / w, 97 / h]],
            "confidence": 0.95,
            "pieces": [{"id": "flame_p0", "polygons": [[[80 / w, 74 / h], [96 / w, 74 / h],
                                                       [96 / w, 98 / h], [80 / w, 98 / h]]],
                        "draw_order": 60}],
        }
        c["modules"].append(flame)
        c["connections"].append({"id": "palm_flame", "type": "attachment", "a": "hand", "b": "flame",
                                 "point": [88 / w, 74 / h]})
        hints = vision_result_to_semantic_hints(c, image_path=self.image, mask_dir=self.d / "flame-masks")
        ps, rs = build_default_specs(self.image, character="flame", semantic_hints=hints)
        self.assertTrue(any(m.get("lockedRoot") for m in rs["moduleContract"]["modules"]))
        motion = resolve_module_motion(rs, "agile_full")
        flame_mod = next(m for m in rs["moduleContract"]["modules"] if m["id"] == "flame")
        driver = motion[flame_mod["bone"]]["rotate"][0][0]
        first = motion[flame_mod["softBones"][0]]["rotate"][0][0]
        tail = [motion[b]["rotate"][0][0] for b in flame_mod["softBones"][1:]]
        self.assertLessEqual(abs(driver), 0.3)
        self.assertLessEqual(abs(first), 0.3)
        self.assertGreater(abs(first), 0.0)
        self.assertTrue(tail)
        self.assertEqual(soft_bend_deg_for_link(0, locked_root=True), LOCKED_ROOT_DEG)
        self.assertGreater(soft_bend_deg_for_link(1, locked_root=True, scale=1.0), LOCKED_ROOT_DEG)
        self.assertGreater(soft_bend_deg_for_link(2, locked_root=True, scale=1.0),
                           soft_bend_deg_for_link(1, locked_root=True, scale=1.0))
        split_parts(self.image, ps["parts"], self.d / "flame-parts", min_island_px=0, feather_px=0,
                    module_contract=ps["module_contract"])
        made = build_rig(self.d / "flame-parts", rs, self.d / "flame-rig")
        renderer = SpineRenderer(self.d / "flame-rig")
        qc = ModuleQualityAuditor(renderer, self.d / "flame-parts", made["report"], renderer.frame(None))
        for t in (0, 2, 4, 6, 8):
            qc.observe(t, 1.0, renderer.frame(t))
        quality = qc.finish()
        self.assertNotEqual(quality["bindingMotion"]["status"], "fail")
        self.assertNotEqual(quality["machineQC"]["status"], "fail")
        self.assertEqual(quality["bindingMotion"]["checks"]["lockedRoot"]["status"], "pass")
        self.assertLessEqual(quality["bindingMotion"]["checks"]["lockedRoot"]["metrics"]["flame"]["normalPtpDegrees"], 0.6)

    def test_standalone_html_inlines_scripts_and_pngs(self):
        from src.studio.vision_schema import vision_result_to_semantic_hints
        from src.studio.auto_spec import build_default_specs
        from src.pipeline.split_parts import split_parts
        from src.pipeline.build_rig import build_rig

        hints = vision_result_to_semantic_hints(self.contract, image_path=self.image, mask_dir=self.d / "html-masks")
        ps, rs = build_default_specs(self.image, character="htmlcase", semantic_hints=hints)
        split_parts(self.image, ps["parts"], self.d / "html-parts", min_island_px=0, feather_px=0,
                    module_contract=ps["module_contract"])
        build_rig(self.d / "html-parts", rs, self.d / "html-rig")
        out = self.d / "preview.html"
        export_standalone_html(self.d / "html-rig", out, require_approved=False)
        html = out.read_text(encoding="utf-8")
        self.assertIn("<script type=\"module\">", html)
        self.assertIn("class Spine42Skeleton", html)
        self.assertIn("class PortraitSpinePlayer", html)
        self.assertIn("data:image/png;base64,", html)
        self.assertNotRegex(html, r'src=["\'](?:\./)?images/')
        self.assertGreater(html.count("data:image/png;base64,"), 3)


    def test_mask_rejects_non_png_rgb_and_non_binary_gray(self):
        with Image.open(self.image) as im:
            w, h = im.size
        c = json.loads(json.dumps(self.contract))
        piece = c["modules"][0]["pieces"][0]
        huge = json.loads(json.dumps(c))
        huge["modules"][0]["pieces"][0]["mask_png_base64"] = "A" * (16 * 1024 * 1024 + 8)
        huge["modules"][0]["pieces"][0]["mask_sha256"] = "0" * 64
        with self.assertRaises(VisionValidationError):
            validate_module_contract(huge, input_sha256=c["input_sha256"], image_size=(w, h))
        jpeg = io.BytesIO()
        Image.fromarray(np.zeros((h, w), np.uint8)).save(jpeg, format="JPEG")
        raw = jpeg.getvalue()
        piece["mask_png_base64"] = base64.b64encode(raw).decode("ascii")
        piece["mask_sha256"] = hashlib.sha256(raw).hexdigest()
        with self.assertRaises(VisionValidationError):
            validate_module_contract(c, input_sha256=c["input_sha256"], image_size=(w, h))
        rgb = np.zeros((h, w, 3), np.uint8)
        buf = io.BytesIO(); Image.fromarray(rgb).save(buf, format="PNG"); raw = buf.getvalue()
        piece["mask_png_base64"] = base64.b64encode(raw).decode("ascii")
        piece["mask_sha256"] = hashlib.sha256(raw).hexdigest()
        with self.assertRaises(VisionValidationError):
            validate_module_contract(c, input_sha256=c["input_sha256"], image_size=(w, h))
        gray = np.zeros((h, w), np.uint8); gray[50:80, 50:70] = 128
        buf = io.BytesIO(); Image.fromarray(gray).save(buf, format="PNG"); raw = buf.getvalue()
        piece["mask_png_base64"] = base64.b64encode(raw).decode("ascii")
        piece["mask_sha256"] = hashlib.sha256(raw).hexdigest()
        with self.assertRaises(VisionValidationError):
            validate_module_contract(c, input_sha256=c["input_sha256"], image_size=(w, h))

    def test_mask_outside_foreground_is_review_required(self):
        with Image.open(self.image) as im:
            rgba = np.asarray(im.convert("RGBA"))
        h, w = rgba.shape[:2]
        mask = np.zeros((h, w), dtype=bool)
        mask[0, 0] = True
        self.assertEqual(int(rgba[0, 0, 3]), 0)
        b64, digest = _png_mask(mask)
        c = json.loads(json.dumps(self.contract))
        c["modules"][0]["pieces"][0]["mask_png_base64"] = b64
        c["modules"][0]["pieces"][0]["mask_sha256"] = digest
        validate_module_contract(c, input_sha256=c["input_sha256"], image_size=(w, h))
        with self.assertRaises(VisionReviewRequired):
            materialize_module_contract(c, image_path=self.image, mask_dir=self.d / "bad-fg")

    def test_segment_refuses_copied_masks_for_multi_piece_modules(self):
        from scripts.segment_modules_sam import expand_segment_targets, _encode_mask
        modules = [{"id": "staff", "draw_order": 3, "box": [1, 2, 3, 4], "pieces": [
            {"id": "staff_a", "draw_order": 3}, {"id": "staff_b", "draw_order": 4}]}]
        with self.assertRaises(ValueError):
            expand_segment_targets(modules)
        ok = [{"id": "staff", "draw_order": 3, "pieces": [
            {"id": "staff_a", "draw_order": 3, "box": [10, 10, 40, 40], "pos": [[20, 20]]},
            {"id": "staff_b", "draw_order": 4, "box": [40, 40, 80, 80], "pos": [[50, 50]]}]}]
        targets = expand_segment_targets(ok)
        self.assertEqual([t["id"] for t in targets], ["staff_a", "staff_b"])
        self.assertNotEqual(targets[0]["box"], targets[1]["box"])
        encoded = _encode_mask(np.array([[True, False], [False, True]]))
        raw = base64.b64decode(encoded["mask_png_base64"])
        with Image.open(io.BytesIO(raw)) as im:
            self.assertEqual(im.mode, "L")
            arr = np.asarray(im)
            self.assertTrue(set(np.unique(arr).tolist()).issubset({0, 255}))

    def test_locked_root_and_armor_motion_are_closed(self):
        c = json.loads(json.dumps(self.contract))
        cape = next(m for m in c["modules"] if m["id"] == "cape")
        cape["locked_root"] = True
        with self.assertRaises(VisionValidationError):
            validate_module_contract(c, input_sha256=c['input_sha256'], image_size=(128,160))
        c = json.loads(json.dumps(self.contract))
        spark = next(m for m in c["modules"] if m["id"] == "spark")
        spark.update({"motion_type": "soft", "motion": "soft_bend", "locked_root": True,
                      "parent": None, "centerline": [[7 / 128, 7 / 160], [10 / 128, 12 / 160]]})
        with self.assertRaises(VisionReviewRequired):
            validate_module_contract(c, input_sha256=c['input_sha256'], image_size=(128,160))
        c = json.loads(json.dumps(self.contract))
        w, h = 128, 160
        flame = {"id": "flame", "role": "effect_group", "motion_type": "soft", "parent": "spark",
                 "pivot": [7 / w, 7 / h], "motion": "soft_bend", "locked_root": True,
                 "centerline": [[7 / w, 7 / h], [8 / w, 10 / h]], "confidence": 0.95,
                 "pieces": [{"id": "flame_p0", "polygons": [[[5 / w, 5 / h], [8 / w, 5 / h], [8 / w, 8 / h], [5 / w, 8 / h]]],
                             "draw_order": 61}]}
        c["modules"].append(flame)
        c["connections"].append({"id": "spark_flame", "type": "attachment", "a": "spark", "b": "flame",
                                 "point": [7 / w, 7 / h]})
        with self.assertRaises(VisionReviewRequired):
            validate_module_contract(c, input_sha256=c['input_sha256'], image_size=(128,160))
        c = json.loads(json.dumps(self.contract))
        blade = next(m for m in c["modules"] if m["id"] == "blade")
        blade["role"] = "armor"; blade["motion"] = "breath"
        with self.assertRaises(VisionReviewRequired):
            validate_module_contract(c, input_sha256=c['input_sha256'], image_size=(128,160))
        c = json.loads(json.dumps(self.contract))
        blade = next(m for m in c["modules"] if m["id"] == "blade")
        blade["role"] = "armor"; blade["motion"] = "sway"
        c["connections"][2]["type"] = "attachment"
        validate_module_contract(c, input_sha256=c['input_sha256'], image_size=(128,160))

    def test_root_breath_translate_is_linear_height_scale_not_squared(self):
        from src.studio.vision_schema import vision_result_to_semantic_hints
        from src.studio.auto_spec import build_default_specs
        from src.pipeline.split_parts import split_parts
        from src.pipeline.build_rig import build_rig

        hints = vision_result_to_semantic_hints(self.contract, image_path=self.image, mask_dir=self.d / "breath-masks")
        ps, rs = build_default_specs(self.image, character="breathcase", semantic_hints=hints)
        split_parts(self.image, ps["parts"], self.d / "breath-parts", min_island_px=0, feather_px=0,
                    module_contract=ps["module_contract"])
        made = build_rig(self.d / "breath-parts", rs, self.d / "breath-rig")
        spine = json.loads((self.d / "breath-rig" / "breathcase.spine.json").read_text())
        body = spine["animations"]["idle"]["bones"]["module_body"]["translate"]
        xs = [float(k.get("x", 0.0)) for k in body]
        ys = [float(k.get("y", 0.0)) for k in body]
        body_h = float(rs["body_height_px"])
        height_scale = max(0.35, min(1.35, body_h / 1080.0))
        peak = max(max(abs(v) for v in xs), max(abs(v) for v in ys))
        from src.pipeline.module_rig import AIRBORNE_BREATH_TY
        ty_primary, ty_secondary = AIRBORNE_BREATH_TY
        self.assertGreater(peak, ty_primary * height_scale * 0.45)
        self.assertLess(peak, (ty_primary + ty_secondary) * height_scale * 1.2 + 0.05)
        squared = 1.8 * (body_h / 1080.0) * height_scale
        self.assertGreater(peak, squared * 3)


    def test_locked_root_motion_and_softbones_cannot_be_forged(self):
        from src.studio.vision_schema import vision_result_to_semantic_hints
        from src.studio.auto_spec import build_default_specs
        from src.pipeline.split_parts import split_parts
        from src.pipeline.build_rig import build_rig
        from src.pipeline.module_rig import validate_module_rig_spec, resolve_module_motion
        from src.render.renderer import SpineRenderer
        import copy

        w, h = 128, 160
        with Image.open(self.image) as im:
            rgba = np.asarray(im.convert("RGBA")).copy()
        rgba[74:98, 80:96] = [255, 90, 40, 180]
        Image.fromarray(rgba).save(self.image)
        c = json.loads(json.dumps(self.contract))
        c["input_sha256"] = hashlib.sha256(self.image.read_bytes()).hexdigest()
        flame = {
            "id": "flame", "role": "effect_group", "motion_type": "soft", "parent": "hand",
            "pivot": [88 / w, 74 / h], "motion": "soft_bend", "locked_root": True,
            "centerline": [[88 / w, 74 / h], [88 / w, 86 / h], [88 / w, 97 / h]],
            "confidence": 0.95,
            "pieces": [{"id": "flame_p0", "polygons": [[[80 / w, 74 / h], [96 / w, 74 / h],
                                                       [96 / w, 98 / h], [80 / w, 98 / h]]],
                        "draw_order": 60}],
        }
        c["modules"].append(flame)
        c["connections"].append({"id": "palm_flame", "type": "attachment", "a": "hand", "b": "flame",
                                 "point": [88 / w, 74 / h]})
        hints = vision_result_to_semantic_hints(c, image_path=self.image, mask_dir=self.d / "forge-masks")
        ps, rs = build_default_specs(self.image, character="forge", semantic_hints=hints)
        split_parts(self.image, ps["parts"], self.d / "forge-parts", min_island_px=0, feather_px=0,
                    module_contract=ps["module_contract"])
        made = build_rig(self.d / "forge-parts", rs, self.d / "forge-rig")
        flame_mod = next(m for m in rs["moduleContract"]["modules"] if m["id"] == "flame")
        self.assertEqual(flame_mod["motion"], "required")
        self.assertTrue(flame_mod["softBones"])
        for forged in ("static", "optional"):
            bad = copy.deepcopy(rs)
            next(m for m in bad["moduleContract"]["modules"] if m["id"] == "flame")["motion"] = forged
            with self.assertRaises(VisionReviewRequired):
                validate_module_rig_spec(bad)
            with self.assertRaises(VisionReviewRequired):
                resolve_module_motion(bad, "agile_full")
        empty = copy.deepcopy(rs)
        next(m for m in empty["moduleContract"]["modules"] if m["id"] == "flame")["softBones"] = []
        with self.assertRaises(VisionReviewRequired):
            validate_module_rig_spec(empty)
        with self.assertRaises(VisionReviewRequired):
            resolve_module_motion(empty, "agile_full")
        renderer = SpineRenderer(self.d / "forge-rig")
        report = copy.deepcopy(made["report"])
        next(m for m in report["moduleContract"]["modules"] if m["id"] == "flame")["motion"] = "optional"
        qc = ModuleQualityAuditor(renderer, self.d / "forge-parts", report, renderer.frame(None))
        for t in (0, 2, 4, 8):
            qc.observe(t, 1.0, renderer.frame(t))
        quality = qc.finish()
        self.assertEqual(quality["bindingMotion"]["checks"]["lockedRoot"]["status"], "fail")
        report2 = copy.deepcopy(made["report"])
        next(m for m in report2["moduleContract"]["modules"] if m["id"] == "flame")["softBones"] = []
        qc2 = ModuleQualityAuditor(renderer, self.d / "forge-parts", report2, renderer.frame(None))
        for t in (0, 2, 8):
            qc2.observe(t, 1.0, renderer.frame(t))
        self.assertEqual(qc2.finish()["bindingMotion"]["checks"]["lockedRoot"]["status"], "fail")

    def test_segment_ids_reject_path_escape(self):
        from scripts.segment_modules_sam import expand_segment_targets
        with self.assertRaises(ValueError):
            expand_segment_targets([{"id": "../../escape", "box": [1, 2, 3, 4]}])
        with self.assertRaises(ValueError):
            expand_segment_targets([{"id": "ok", "pieces": [
                {"id": "a/b", "draw_order": 1, "box": [1, 2, 3, 4]},
                {"id": "ok_b", "draw_order": 2, "box": [2, 3, 4, 5]}]}])
        with self.assertRaises(ValueError):
            expand_segment_targets([{"id": "ok", "pieces": [
                {"id": "dup", "draw_order": 1, "box": [1, 2, 3, 4]},
                {"id": "dup", "draw_order": 2, "box": [2, 3, 4, 5]}]}])




    def _flame_contract(self):
        w, h = 128, 160
        with Image.open(self.image) as im:
            rgba = np.asarray(im.convert("RGBA")).copy()
        rgba[74:98, 80:96] = [255, 90, 40, 180]
        Image.fromarray(rgba).save(self.image)
        c = json.loads(json.dumps(self.contract))
        c["input_sha256"] = hashlib.sha256(self.image.read_bytes()).hexdigest()
        flame = {
            "id": "flame", "role": "effect_group", "motion_type": "soft", "parent": "hand",
            "pivot": [88 / w, 74 / h], "motion": "soft_bend", "locked_root": True,
            "centerline": [[88 / w, 74 / h], [88 / w, 86 / h], [88 / w, 97 / h]],
            "confidence": 0.95,
            "pieces": [{"id": "flame_p0", "polygons": [[[80 / w, 74 / h], [96 / w, 74 / h],
                                                       [96 / w, 98 / h], [80 / w, 98 / h]]],
                        "draw_order": 60}],
        }
        c["modules"].append(flame)
        c["connections"].append({"id": "palm_flame", "type": "attachment", "a": "hand", "b": "flame",
                                 "point": [88 / w, 74 / h]})
        return c

    def test_soft_chain_graph_requires_real_bones(self):
        from src.studio.vision_schema import vision_result_to_semantic_hints
        from src.studio.auto_spec import build_default_specs
        from src.pipeline.split_parts import split_parts
        from src.pipeline.build_rig import build_rig
        from src.pipeline.module_rig import (
            validate_module_rig_spec, resolve_module_motion, validate_expanded_soft_chains,
        )
        from src.render.renderer import SpineRenderer
        import copy

        c = self._flame_contract()
        hints = vision_result_to_semantic_hints(c, image_path=self.image, mask_dir=self.d / "chain-masks")
        ps, rs = build_default_specs(self.image, character="chaincase", semantic_hints=hints)
        split_parts(self.image, ps["parts"], self.d / "chain-parts", min_island_px=0, feather_px=0,
                    module_contract=ps["module_contract"])
        validate_module_rig_spec(rs)
        resolve_module_motion(rs, "agile_full")
        made = build_rig(self.d / "chain-parts", rs, self.d / "chain-rig")
        flame = next(m for m in rs["moduleContract"]["modules"] if m["id"] == "flame")
        self.assertEqual(len(flame["softBones"]), 2)
        renderer = SpineRenderer(self.d / "chain-rig")
        by_name = renderer.sk.by_name
        first, last = flame["softBones"]
        self.assertEqual(by_name[first].parent.name, flame["bone"])
        self.assertEqual(by_name[last].parent.name, first)
        live = {b.name: (None if b.parent is None else b.parent.name) for b in renderer.sk.bones}
        validate_expanded_soft_chains(rs["moduleContract"], live)
        qc = ModuleQualityAuditor(renderer, self.d / "chain-parts", made["report"], renderer.frame(None))
        for t in (0, 2, 4, 8):
            qc.observe(t, 1.0, renderer.frame(t))
        self.assertEqual(qc.finish()["bindingMotion"]["checks"]["lockedRoot"]["status"], "pass")

        spine = json.loads((self.d / "chain-rig" / "chaincase.spine.json").read_text())
        expanded = copy.deepcopy(rs)
        have = {b["name"] for b in expanded["bones"]}
        for bone in spine["bones"]:
            if bone["name"] not in have:
                expanded["bones"].append({"name": bone["name"], "parent": bone.get("parent"), "pos": [0.0, 0.0]})
        validate_module_rig_spec(expanded)
        resolve_module_motion(expanded, "agile_full")

        def drop(name):
            bad = copy.deepcopy(expanded)
            bad["bones"] = [b for b in bad["bones"] if b["name"] != name]
            return bad

        for missing in (first, last):
            with self.assertRaises(VisionReviewRequired):
                validate_module_rig_spec(drop(missing))
            with self.assertRaises(VisionReviewRequired):
                resolve_module_motion(drop(missing), "agile_full")

        reparent = copy.deepcopy(expanded)
        for bone in reparent["bones"]:
            if bone["name"] == last:
                bone["parent"] = flame["bone"]
        with self.assertRaises(VisionReviewRequired):
            validate_module_rig_spec(reparent)
        with self.assertRaises(VisionReviewRequired):
            resolve_module_motion(reparent, "agile_full")

        setup = renderer.frame(None)
        report = made["report"]
        for missing in (first, last):
            live_ren = SpineRenderer(self.d / "chain-rig")
            live_ren.sk.bones = [b for b in live_ren.sk.bones if b.name != missing]
            live_ren.sk.by_name.pop(missing, None)
            broken = ModuleQualityAuditor(live_ren, self.d / "chain-parts", report, setup)
            for t in (0, 4, 8):
                broken.observe(t, 1.0, setup)
            self.assertEqual(broken.finish()["bindingMotion"]["checks"]["lockedRoot"]["status"], "fail")
        wrong = SpineRenderer(self.d / "chain-rig")
        wrong.sk.by_name[last].parent = wrong.sk.by_name[flame["bone"]]
        qc_wrong = ModuleQualityAuditor(wrong, self.d / "chain-parts", report, setup)
        for t in (0, 8):
            qc_wrong.observe(t, 1.0, setup)
        self.assertEqual(qc_wrong.finish()["bindingMotion"]["checks"]["lockedRoot"]["status"], "fail")


    def test_lively_idle_floating_prop_and_counterphase_vfx_and_grounded_ik(self):
        from src.pipeline.module_rig import resolve_module_motion, LOCKED_ROOT_DEG
        from src.pipeline.build_rig import harmonic

        def mod(mid, role, profile, shape, parent, pivot, extra=None):
            item = {
                "id": mid, "bone": "module_" + mid, "role": role, "motionProfile": profile,
                "motion": "required", "shape": shape, "parent": parent, "pivot": pivot,
                "pieces": ["piece_" + mid],
            }
            if extra:
                item.update(extra)
            return item

        modules = [
            mod("body", "body", "breath", "rigid", None, [500.0, 520.0]),
            mod("thigh", "upper_leg", "sway", "rigid", "body", [460.0, 620.0]),
            mod("shin", "lower_leg", "sway", "rigid", "thigh", [440.0, 820.0]),
            mod("foot", "foot", "static", "rigid", None, [450.0, 1020.0], {"motion": "static"}),
            mod("head", "head", "sway", "rigid", "body", [500.0, 320.0]),
            mod("book", "prop", "sway", "rigid", None, [260.0, 420.0]),
            mod("held_orb", "prop", "sway", "rigid", "body", [640.0, 460.0]),
            mod("vortex_back", "effect_group", "sway", "rigid", None, [420.0, 400.0]),
            mod("vortex_front", "effect_group", "sway", "rigid", None, [580.0, 400.0]),
            mod("palm_flame", "effect_group", "soft_bend", "soft", "body", [660.0, 450.0],
                {"lockedRoot": True, "centerline": [[660.0, 450.0], [660.0, 390.0]], "softBones": ["bend_palm_flame1"]}),
        ]
        bones = [
            {"name": "root", "parent": None, "pos": [500.0, 1020.0]},
            {"name": "module_body", "parent": "root", "pos": [500.0, 520.0]},
            {"name": "module_thigh", "parent": "module_body", "pos": [460.0, 620.0], "aim": [440.0, 820.0]},
            {"name": "module_shin", "parent": "module_thigh", "pos": [440.0, 820.0], "aim": [450.0, 1020.0]},
            {"name": "module_foot", "parent": "root", "pos": [450.0, 1020.0]},
            {"name": "module_head", "parent": "module_body", "pos": [500.0, 320.0]},
            {"name": "module_book", "parent": "root", "pos": [260.0, 420.0]},
            {"name": "module_held_orb", "parent": "module_body", "pos": [640.0, 460.0]},
            {"name": "module_vortex_back", "parent": "root", "pos": [420.0, 400.0]},
            {"name": "module_vortex_front", "parent": "root", "pos": [580.0, 400.0]},
            {"name": "module_palm_flame", "parent": "module_body", "pos": [660.0, 450.0]},
            {"name": "bend_palm_flame1", "parent": "module_palm_flame", "pos": [660.0, 420.0]},
        ]
        skin = {}
        chains = {}
        for m in modules:
            if m["shape"] == "soft":
                cname = "bend_" + m["id"]
                chains[cname] = {"parent": m["bone"], "count": 1, "pts": m["centerline"]}
                skin[m["pieces"][0]] = {"type": "mesh", "recipe": "chain", "chain": cname, "module_id": m["id"]}
            else:
                skin[m["pieces"][0]] = {"type": "region", "bone": m["bone"], "module_id": m["id"]}
        rig = {
            "duration": 8.0, "body_height_px": 1080.0, "bones": bones,
            "ik": [{"name": "contact_anchor_foot", "bones": ["module_thigh", "module_shin"], "target": "module_foot"}],
            "skin": skin, "chains": chains,
            "moduleContract": {"sourceVersion": 2, "modules": modules},
        }
        mot = resolve_module_motion(rig, "agile_full", duration=8.0)
        # 1. Grounded root breathes deeper than old 1.8 cap while keeping pure vertical IK crouch
        self.assertGreaterEqual(abs(mot["module_body"]["ty"][0][0]), 1.85)
        self.assertNotIn("rotate", mot["module_body"])
        # 2. Independent floating prop levitates (tx + ty + rotate); parented prop stays rotate-only
        self.assertIn("ty", mot["module_book"])
        self.assertIn("tx", mot["module_book"])
        self.assertNotIn("ty", mot["module_held_orb"])
        self.assertNotIn("tx", mot["module_held_orb"])
        # 3. Independent front/back VFX float & swirl in opposite directions; lockedRoot stays <= LOCKED_ROOT_DEG
        self.assertIn("ty", mot["module_vortex_back"])
        self.assertIn("ty", mot["module_vortex_front"])
        self.assertLess(mot["module_vortex_back"]["rotate"][0][0] * mot["module_vortex_front"]["rotate"][0][0], 0.0)
        self.assertAlmostEqual(abs(mot["module_palm_flame"]["rotate"][0][0]), LOCKED_ROOT_DEG, places=6)
        # 4. Seamless 8s zero loop across all channels
        for bname, ch in mot.items():
            for key in ("rotate", "tx", "ty"):
                spec = ch.get(key)
                if not spec:
                    continue
                self.assertAlmostEqual(harmonic(spec, 0.0, 8.0), 0.0, places=9, msg=f"{bname} {key} t=0")
                self.assertAlmostEqual(harmonic(spec, 8.0, 8.0), 0.0, places=9, msg=f"{bname} {key} t=8")

if __name__ == "__main__":
    unittest.main()


class SoftCenterlineGuardrailTests(unittest.TestCase):
    def test_smooth_curve_untouched_and_hairpin_kink_relaxed(self):
        from src.pipeline.module_rig import _centerline_max_turn_deg, smooth_sharp_centerline

        smooth = [[100.0, 100.0], [160.0, 115.0], [220.0, 140.0]]
        out_s, warn_s = smooth_sharp_centerline(smooth)
        self.assertIsNone(warn_s)
        self.assertEqual(out_s, smooth)

        # Real 04/cape_r ~83-deg hairpin V-bend
        hairpin = [[1095.0, 2831.0], [698.0, 2596.0], [546.0, 2790.0]]
        self.assertGreater(_centerline_max_turn_deg(hairpin), 75.0)
        out_h, warn_h = smooth_sharp_centerline(hairpin)
        self.assertIsNotNone(warn_h)
        self.assertEqual(warn_h["code"], "soft_centerline_sharp_kink_smoothed")
        self.assertEqual(out_h[0], hairpin[0])
        self.assertEqual(out_h[-1], hairpin[-1])
        self.assertLessEqual(_centerline_max_turn_deg(out_h), 42.0)


class GroundedStabilityPhysicsTests(unittest.TestCase):
    def test_grounded_static_lower_body_keeps_legs_nailed_and_breathes_from_waist(self):
        from src.pipeline.module_rig import GROUNDED_WAIST_BREATH_TY, resolve_module_motion
        from src.studio.module_contract import MODULE_RULE_VERSION, validate_module_contract
        import hashlib, tempfile
        from PIL import Image

        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2] / '.studio-work') as tmp:
            img = Path(tmp) / 'standing.png'
            Image.new('RGBA', (128, 160), (80, 120, 180, 255)).save(img)
            h = hashlib.sha256(img.read_bytes()).hexdigest()
            contract = {
                'version': 2, 'coord_mode': 'normalized', 'input_sha256': h, 'image_size': [128, 160],
                'pose_type': 'grounded', 'anatomy': 'humanoid_full', 'uncertainties': [],
                'modules': [
                    {'id': 'legs', 'role': 'lower_body', 'motion_type': 'static', 'motion': 'static',
                     'static_reason': 'standing grounded lower anchor', 'parent': None,
                     'pivot': [0.5, 0.85], 'confidence': 0.95,
                     'pieces': [{'id': 'legs_p0', 'polygons': [[[0.3, 0.55], [0.7, 0.55], [0.7, 0.95], [0.3, 0.95]]], 'draw_order': 10}]},
                    {'id': 'torso', 'role': 'torso', 'motion_type': 'rigid', 'motion': 'breath',
                     'parent': 'legs', 'pivot': [0.5, 0.55], 'confidence': 0.95,
                     'pieces': [{'id': 'torso_p0', 'polygons': [[[0.3, 0.25], [0.7, 0.25], [0.7, 0.58], [0.3, 0.58]]], 'draw_order': 20}]},
                    {'id': 'head', 'role': 'head', 'motion_type': 'rigid', 'motion': 'sway',
                     'parent': 'torso', 'pivot': [0.5, 0.25], 'confidence': 0.95,
                     'pieces': [{'id': 'head_p0', 'polygons': [[[0.38, 0.05], [0.62, 0.05], [0.62, 0.28], [0.38, 0.28]]], 'draw_order': 30}]},
                ],
                'connections': [
                    {'id': 'waist', 'type': 'joint', 'a': 'legs', 'b': 'torso', 'point': [0.5, 0.55]},
                    {'id': 'neck', 'type': 'joint', 'a': 'torso', 'b': 'head', 'point': [0.5, 0.25]},
                ],
                'occlusions': [],
            }
            validate_module_contract(contract, input_sha256=h, image_size=(128, 160))

            rig = {
                'pose_type': 'grounded', 'body_height_px': 2400.0, 'ik': [], 'skin': {},
                'bones': [
                    {'name': 'root', 'parent': None, 'pos': [64, 160]},
                    {'name': 'module_legs', 'parent': 'root', 'pos': [64, 136]},
                    {'name': 'module_torso', 'parent': 'module_legs', 'pos': [64, 88]},
                    {'name': 'module_head', 'parent': 'module_torso', 'pos': [64, 40]},
                ],
                'moduleContract': {
                    'sourceVersion': 2, 'version': 1, 'ruleVersion': MODULE_RULE_VERSION,
                    'modules': [
                        {'id': 'legs', 'role': 'lower_body', 'bone': 'module_legs', 'parent': None,
                         'shape': 'rigid', 'motion': 'static', 'motionProfile': 'static', 'pivot': [64, 136], 'pieces': []},
                        {'id': 'torso', 'role': 'torso', 'bone': 'module_torso', 'parent': 'legs',
                         'shape': 'rigid', 'motion': 'required', 'motionProfile': 'breath', 'pivot': [64, 88], 'pieces': []},
                        {'id': 'head', 'role': 'head', 'bone': 'module_head', 'parent': 'torso',
                         'shape': 'rigid', 'motion': 'required', 'motionProfile': 'sway', 'pivot': [64, 40], 'pieces': []},
                    ],
                },
            }
            mot = resolve_module_motion(rig, 'default')
            self.assertNotIn('module_legs', mot)
            self.assertIn('module_torso', mot)
            self.assertNotIn('tx', mot['module_torso'])
            self.assertAlmostEqual(mot['module_torso']['ty'][0][0], GROUNDED_WAIST_BREATH_TY[0], places=3)
            self.assertLessEqual(sum(abs(t[0]) for t in mot['module_torso']['rotate']), 0.75)
            self.assertIn('module_head', mot)

    def test_grounded_root_torso_never_gets_airborne_float_and_airborne_tilt_is_poised(self):
        from src.pipeline.module_rig import AIRBORNE_BREATH_TX, resolve_module_motion
        from src.studio.module_contract import MODULE_RULE_VERSION
        base_rig = {
            'body_height_px': 2400.0, 'ik': [], 'skin': {},
            'bones': [
                {'name': 'root', 'parent': None, 'pos': [64, 160]},
                {'name': 'module_torso', 'parent': 'root', 'pos': [64, 80]},
            ],
            'moduleContract': {
                'sourceVersion': 2, 'version': 1, 'ruleVersion': MODULE_RULE_VERSION,
                'modules': [
                    {'id': 'torso', 'role': 'torso', 'bone': 'module_torso', 'parent': None,
                     'shape': 'rigid', 'motion': 'required', 'motionProfile': 'breath', 'pivot': [64, 80], 'pieces': []},
                ],
            },
        }
        g_mot = resolve_module_motion({**base_rig, 'pose_type': 'grounded'}, 'default')
        self.assertNotIn('tx', g_mot['module_torso'])
        self.assertNotIn('rotate', g_mot['module_torso'])
        self.assertIn('ty', g_mot['module_torso'])

        a_mot = resolve_module_motion({**base_rig, 'pose_type': 'airborne'}, 'default')
        self.assertLessEqual(sum(abs(t[0]) for t in a_mot['module_torso']['rotate']), 1.7)
        self.assertLessEqual(sum(abs(t[0]) for t in a_mot['module_torso']['tx']), sum(AIRBORNE_BREATH_TX) + 0.01)

    def test_unified_upper_body_plan_mode_supports_grounded_and_airborne_without_neck_cut(self):
        from src.studio.module_contract import validate_module_contract
        from src.studio.vision_schema import VisionReviewRequired
        import hashlib, tempfile
        from PIL import Image

        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2] / '.studio-work') as tmp:
            img = Path(tmp) / 'hero.png'
            Image.new('RGBA', (128, 160), (90, 110, 160, 255)).save(img)
            h = hashlib.sha256(img.read_bytes()).hexdigest()
            base_grounded = {
                'version': 2, 'coord_mode': 'normalized', 'input_sha256': h, 'image_size': [128, 160],
                'pose_type': 'grounded', 'anatomy': 'humanoid_full', 'uncertainties': [],
                'plan_mode': 'unified_upper_body',
                'modules': [
                    {'id': 'legs', 'role': 'lower_body', 'motion_type': 'static', 'motion': 'static',
                     'static_reason': 'standing grounded lower anchor', 'parent': None,
                     'pivot': [0.5, 0.85], 'confidence': 0.95,
                     'pieces': [{'id': 'legs_p0', 'polygons': [[[0.3, 0.55], [0.7, 0.55], [0.7, 0.95], [0.3, 0.95]]], 'draw_order': 10}]},
                    {'id': 'torso', 'role': 'torso', 'motion_type': 'rigid', 'motion': 'breath',
                     'parent': 'legs', 'pivot': [0.5, 0.55], 'confidence': 0.95,
                     'pieces': [{'id': 'torso_p0', 'polygons': [[[0.3, 0.10], [0.7, 0.10], [0.7, 0.58], [0.3, 0.58]]], 'draw_order': 20}]},
                ],
                'connections': [
                    {'id': 'waist', 'type': 'joint', 'a': 'legs', 'b': 'torso', 'point': [0.5, 0.55]},
                ],
                'occlusions': [],
            }
            # Valid under canvas_layers + unified_upper_body
            c = validate_module_contract(base_grounded, input_sha256=h, image_size=(128, 160), material_mode='canvas_layers')
            self.assertEqual(c['plan_mode'], 'unified_upper_body')

            # Standard mode STILL rejects missing head / single active hard body
            std = dict(base_grounded, plan_mode='standard')
            with self.assertRaises(VisionReviewRequired):
                validate_module_contract(std, input_sha256=h, image_size=(128, 160), material_mode='canvas_layers')

            # Grounded unified_upper_body STILL rejects missing lower_body
            no_legs = {
                **base_grounded,
                'modules': [
                    {**base_grounded['modules'][1], 'parent': None},
                    {'id': 'fx', 'role': 'effect_group', 'motion_type': 'rigid', 'motion': 'sway',
                     'parent': None, 'pivot': [0.2, 0.3], 'confidence': 0.95,
                     'pieces': [{'id': 'fx_p0', 'polygons': [[[0.1, 0.1], [0.25, 0.1], [0.25, 0.4], [0.1, 0.4]]], 'draw_order': 30}]},
                ],
                'connections': [],
            }
            with self.assertRaises(VisionReviewRequired):
                validate_module_contract(no_legs, input_sha256=h, image_size=(128, 160), material_mode='canvas_layers')

            # Airborne unified_upper_body with torso + active effect_group passes
            air_ok = {**no_legs, 'pose_type': 'airborne'}
            c_air = validate_module_contract(air_ok, input_sha256=h, image_size=(128, 160), material_mode='canvas_layers')
            self.assertEqual(c_air['plan_mode'], 'unified_upper_body')

            # Airborne unified_upper_body with ONLY torso + static ground (single active module) is rejected
            air_single = {
                **air_ok,
                'modules': [
                    air_ok['modules'][0],
                    {'id': 'ground', 'role': 'ground', 'motion_type': 'static', 'motion': 'static',
                     'static_reason': 'static rock base', 'parent': None, 'pivot': [0.5, 0.9], 'confidence': 0.95,
                     'pieces': [{'id': 'g_p0', 'polygons': [[[0.2, 0.85], [0.8, 0.85], [0.8, 0.98], [0.2, 0.98]]], 'draw_order': 5}]},
                ],
            }
            with self.assertRaises(VisionReviewRequired):
                validate_module_contract(air_single, input_sha256=h, image_size=(128, 160), material_mode='canvas_layers')
