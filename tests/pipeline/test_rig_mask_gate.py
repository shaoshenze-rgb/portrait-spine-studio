"""ISSUE-02: rig placement hard gate and three-layer overlay."""
from __future__ import annotations

import base64
import hashlib
import os
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.pipeline.build_rig import build_rig
from src.pipeline.mask_geometry import derive_contract_geometry
from src.pipeline.module_rig import (
    _placement_tol,
    _validate_rig_mask_placement,
    build_module_specs,
    render_rig_mask_overlay,
)
from src.pipeline.split_parts import split_parts
from src.studio.auto_spec import build_default_specs
from src.studio.vision_schema import VisionReviewRequired, vision_result_to_semantic_hints

ROOT = Path(__file__).resolve().parents[2]


def _poly(w, h, x0, y0, x1, y1):
    return [[x0 / w, y0 / h], [x1 / w, y0 / h], [x1 / w, y1 / h], [x0 / w, y1 / h]]


def _png_mask(mask: np.ndarray):
    im = Image.fromarray(np.where(mask, 255, 0).astype(np.uint8))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    raw = buf.getvalue()
    return base64.b64encode(raw).decode("ascii"), hashlib.sha256(raw).hexdigest()


def _piece(pid, mask, order):
    b64, digest = _png_mask(mask)
    return {"id": pid, "draw_order": order, "mask_png_base64": b64, "mask_sha256": digest}


class RigMaskGateTests(unittest.TestCase):
    def setUp(self):
        work = ROOT / ".studio-work"
        work.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=work, prefix="rig-mask-gate-")
        self.addCleanup(self.tmp.cleanup)
        self.d = Path(self.tmp.name)

    def _paint_scene(self):
        w, h = 160, 200
        rgba = np.zeros((h, w, 4), np.uint8)
        rgba[20:50, 60:95] = (200, 90, 100, 255)  # head
        rgba[50:150, 50:100] = (30, 160, 70, 255)  # torso
        rgba[70:110, 100:145] = (160, 90, 40, 255)  # arm
        for y in range(55, 170):
            x0 = 34 + int(6 * math.sin((y - 55) / 28.0))
            x1 = min(50, x0 + 16)
            if x1 > x0:
                rgba[y, x0:x1] = (40, 80, 200, 255)
        path = self.d / "scene.png"
        Image.fromarray(rgba).save(path)
        return path, w, h, rgba

    def _contract_and_hints(self, *, derive=True, mutate=None):
        path, w, h, rgba = self._paint_scene()
        fg = rgba[..., 3] > 0
        head = (rgba[..., 0] > 180) & fg
        torso = (rgba[..., 1] > 120) & fg & ~head
        arm = (rgba[..., 0] > 120) & (rgba[..., 0] < 180) & fg
        ribbon = (rgba[..., 2] > 150) & fg & ~torso & ~arm & ~head
        torso = torso & ~arm
        modules = [
            {
                "id": "face",
                "role": "head",
                "motion_type": "rigid",
                "parent": "body",
                "pivot": [77 / w, 35 / h],
                "confidence": 0.95,
                "pieces": [_piece("face_p0", head, 30)],
            },
            {
                "id": "body",
                "role": "torso",
                "motion_type": "rigid",
                "parent": None,
                "pivot": [75 / w, 100 / h],
                "confidence": 0.95,
                "pieces": [_piece("body_p0", torso, 10)],
            },
            {
                "id": "arm",
                "role": "arm",
                "motion_type": "rigid",
                "parent": "body",
                "pivot": [110 / w, 90 / h],
                "confidence": 0.95,
                "representation": "mesh",
                "pieces": [_piece("arm_p0", arm, 20)],
            },
            {
                "id": "cape",
                "role": "cape",
                "motion_type": "soft",
                "parent": "body",
                "pivot": [25 / w, 60 / h],
                "centerline": [[25 / w, 60 / h], [24 / w, 90 / h], [22 / w, 120 / h], [20 / w, 145 / h], [18 / w, 165 / h]],
                "confidence": 0.95,
                "pieces": [_piece("cape_p0", ribbon, 0)],
            },
        ]
        if derive:
            modules = derive_contract_geometry(
                (w, h), modules, {"face_p0": head, "body_p0": torso, "arm_p0": arm, "cape_p0": ribbon}
            )
            for m in modules:
                if m.get("motion_type") == "soft" and m.get("centerline"):
                    m["pivot"] = list(m["centerline"][0])
                    if len(m["centerline"]) > 5:
                        m["centerline"] = m["centerline"][:5]
        by_id = {m["id"]: m for m in modules}
        connections = [
            {"id": "neck", "type": "joint", "a": "body", "b": "face", "point": list(by_id["face"]["pivot"])},
            {"id": "shoulder", "type": "joint", "a": "body", "b": "arm", "point": list(by_id["arm"]["pivot"])},
            {"id": "cape_root", "type": "attachment", "a": "body", "b": "cape", "point": list(by_id["cape"]["pivot"])},
        ]
        contract = {
            "version": 2,
            "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "image_size": [w, h],
            "coord_mode": "normalized",
            "pose_type": "airborne",
            "anatomy": "humanoid_bust",
            "modules": modules,
            "connections": connections,
            "occlusions": [],
            "uncertainties": [],
        }
        if mutate:
            mutate(contract)
            contract["input_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        hints = vision_result_to_semantic_hints(contract, image_path=path, mask_dir=self.d / "masks")
        return path, contract, hints, w, h

    def test_derived_geometry_passes_build_module_specs(self):
        path, contract, hints, w, h = self._contract_and_hints(derive=True)
        ps, rig = build_module_specs(path, character="gate", preset="agile_full", semantic_hints=hints)
        self.assertIn("moduleContract", rig)
        self.assertEqual({m["id"] for m in rig["moduleContract"]["modules"]}, {"body", "face", "arm", "cape"})
        self.assertTrue(any(c["id"] == "shoulder" for c in rig["moduleContract"]["contacts"]))

    def test_soft_centerline_outside_raises_finite_distance(self):
        def mutate(c):
            cape = next(m for m in c["modules"] if m["id"]=="cape")
            line = list(cape["centerline"])
            x, y = line[2]
            line[2] = [x - 15 / 160.0, y]
            cape["centerline"] = line
            cape["pivot"] = list(line[0])
        path, contract, hints, w, h = self._contract_and_hints(derive=True, mutate=mutate)
        with self.assertRaises(VisionReviewRequired) as ctx:
            build_module_specs(path, character="gate", preset="agile_full", semantic_hints=hints)
        msg = str(ctx.exception)
        self.assertIn("cape", msg)
        self.assertIn("2", msg)
        self.assertNotIn("inf", msg.lower())
        self.assertRegex(msg, r"\d+\.\d+px")

    def test_root_pivot_outside_raises_finite_distance(self):
        def mutate(c):
            body = next(m for m in c["modules"] if m["id"]=="body")
            body["pivot"] = [75 / 160.0, 8 / 200.0]
        path, contract, hints, w, h = self._contract_and_hints(derive=True, mutate=mutate)
        with self.assertRaises(VisionReviewRequired) as ctx:
            build_module_specs(path, character="gate", preset="agile_full", semantic_hints=hints)
        msg = str(ctx.exception)
        self.assertIn("body", msg)
        self.assertIn("pivot", msg)
        self.assertNotIn("inf", msg.lower())
        self.assertRegex(msg, r"\d+\.\d+px")

    def test_connection_outside_raises(self):
        def mutate(c):
            arm = next(m for m in c["modules"] if m["id"] == "arm")
            arm["pivot"] = [143 / 160.0, 90 / 200.0]
            conn = next(x for x in c["connections"] if x["id"] == "shoulder")
            conn["point"] = list(arm["pivot"])
        path, contract, hints, w, h = self._contract_and_hints(derive=True, mutate=mutate)
        with self.assertRaises(VisionReviewRequired) as ctx:
            build_module_specs(path, character="gate", preset="agile_full", semantic_hints=hints)
        msg = str(ctx.exception)
        self.assertIn("shoulder", msg)
        self.assertNotIn("inf", msg.lower())
        self.assertRegex(msg, r"\d+\.\d+px")

    def test_tolerance_boundaries(self):
        path, contract, hints, w, h = self._contract_and_hints(derive=True)
        # body_h from torso y=50..149 -> 100px, tol = max(2, min(6, round(100/300))) = 2
        self.assertEqual(_placement_tol(100), 2)

        def shift_soft(dx_px, dy_px=0.0):
            def mutate(c):
                cape = next(m for m in c["modules"] if m["id"] == "cape")
                line = [list(p) for p in cape["centerline"]]
                # at y=55 ribbon starts at x=34
                line[1] = [(34.0 - dx_px) / w, 55.0 / h]
                cape["centerline"] = line
                cape["pivot"] = list(line[0])
            return mutate

        path, contract, hints, w, h = self._contract_and_hints(derive=True, mutate=shift_soft(1.0))
        build_module_specs(path, character="gate", preset="agile_full", semantic_hints=hints)
        path, contract, hints, w, h = self._contract_and_hints(derive=True, mutate=shift_soft(5.0))
        with self.assertRaises(VisionReviewRequired):
            build_module_specs(path, character="gate", preset="agile_full", semantic_hints=hints)

        def shift_pivot(dy_px):
            def mutate(c):
                body = next(m for m in c["modules"] if m["id"] == "body")
                # torso top is y=50; negative dy_px is outside above the mask
                body["pivot"] = [75 / w, (50.0 + dy_px) / h]
            return mutate

        path, contract, hints, w, h = self._contract_and_hints(derive=True, mutate=shift_pivot(-2.0))
        build_module_specs(path, character="gate", preset="agile_full", semantic_hints=hints)
        path, contract, hints, w, h = self._contract_and_hints(derive=True, mutate=shift_pivot(-7.0))
        with self.assertRaises(VisionReviewRequired):
            build_module_specs(path, character="gate", preset="agile_full", semantic_hints=hints)

    def _compiled(self):
        path, contract, hints, w, h = self._contract_and_hints(derive=True)
        ps, rs = build_module_specs(path, character="gate", preset="agile_full", semantic_hints=hints)
        split = split_parts(
            path, ps["parts"], self.d / "parts", min_island_px=0, feather_px=0, module_contract=ps["module_contract"]
        )
        made = build_rig(self.d / "parts", rs, self.d / "rig")
        return path, rs, made

    def test_overlay_png_and_mismatched_mask_raise(self):
        path, rs, made = self._compiled()
        overlay = Path(made["report"]["rigMaskOverlay"]) if not Path(made["report"]["rigMaskOverlay"]).is_absolute() else Path(made["report"]["rigMaskOverlay"])
        overlay = self.d / "rig" / made["report"]["rigMaskOverlay"]
        self.assertTrue(overlay.is_file())
        self.assertGreater(overlay.stat().st_size, 32)
        with Image.open(overlay) as im:
            ow, oh = im.size
            overlay_arr = np.asarray(im.convert("RGB"))
        self.assertGreater(min(ow, oh), 0)
        with Image.open(path) as src:
            src_rgba = np.asarray(src.convert("RGBA"))
        gray = np.full(src_rgba.shape[:2] + (3,), 125, np.uint8)
        a = src_rgba[..., 3:4] / 255.0
        plain = (gray * (1 - a) + src_rgba[..., :3] * a).astype(np.uint8)
        if overlay_arr.shape[:2] != plain.shape[:2]:
            plain_im = Image.fromarray(plain).resize(overlay_arr.shape[:2][::-1], Image.BILINEAR)
            plain = np.asarray(plain_im)
        self.assertGreater(int(np.abs(overlay_arr.astype(int) - plain.astype(int)).max()), 5)

        bad = self.d / "bad-mask.png"
        Image.fromarray(np.zeros((8, 8), np.uint8)).save(bad)
        rs_bad = json.loads(json.dumps(rs))
        rs_bad["moduleContract"]["modules"][0]["visibleMasks"] = [str(bad)]
        with self.assertRaises(ValueError):
            render_rig_mask_overlay(path, rs_bad, self.d / "bad-overlay.png")

    def test_build_rig_does_not_swallow_overlay_size_error(self):
        path, rs, made = self._compiled()
        mask_path = Path(rs["moduleContract"]["modules"][0]["visibleMasks"][0])
        Image.fromarray(np.zeros((8, 8), np.uint8)).save(mask_path)
        with self.assertRaises(ValueError):
            build_rig(self.d / "parts", rs, self.d / "rig2")


    def test_empty_visible_masks_raise(self):
        path, rs, made = self._compiled()
        rs_bad = json.loads(json.dumps(rs))
        rs_bad["moduleContract"]["modules"][0]["visibleMasks"] = []
        with self.assertRaises(ValueError):
            render_rig_mask_overlay(path, rs_bad, self.d / "empty-masks-overlay.png")
        rs_bad["moduleContract"]["modules"][0].pop("visibleMasks", None)
        with self.assertRaises(ValueError):
            render_rig_mask_overlay(path, rs_bad, self.d / "missing-masks-overlay.png")

    def test_build_rig_missing_source_raises_and_relative_source_resolves(self):
        path, rs, made = self._compiled()
        parts_dir = self.d / "parts"
        pj_path = parts_dir / "parts.json"
        pj = json.loads(pj_path.read_text(encoding="utf-8"))
        pj["source"] = str(self.d / "no-such-source.png")
        pj_path.write_text(json.dumps(pj), encoding="utf-8")
        with self.assertRaises(FileNotFoundError):
            build_rig(parts_dir, rs, self.d / "rig-missing-src")

        rel_name = "portrait-src.png"
        dest = parts_dir / rel_name
        dest.write_bytes(path.read_bytes())
        pj["source"] = rel_name
        pj_path.write_text(json.dumps(pj), encoding="utf-8")
        elsewhere = self.d / "elsewhere"
        elsewhere.mkdir()
        old = os.getcwd()
        try:
            os.chdir(elsewhere)
            made2 = build_rig(parts_dir.resolve(), rs, self.d / "rig-rel-src")
        finally:
            os.chdir(old)
        overlay = self.d / "rig-rel-src" / made2["report"]["rigMaskOverlay"]
        self.assertTrue(overlay.is_file())
        self.assertGreater(overlay.stat().st_size, 32)

    def test_foot_anchor_point_on_foot_not_upper_passes(self):
        h, w = 80, 60
        upper = np.zeros((h, w), dtype=bool)
        upper[8:28, 18:40] = True
        lower = np.zeros((h, w), dtype=bool)
        lower[26:56, 20:42] = True
        foot = np.zeros((h, w), dtype=bool)
        foot[54:72, 18:44] = True
        modules = [
            {"id": "upper", "pivot": [29 / w, 18 / h], "motion_type": "rigid"},
            {"id": "lower", "pivot": [31 / w, 40 / h], "motion_type": "rigid"},
            {"id": "foot", "pivot": [31 / w, 55 / h], "motion_type": "static"},
        ]
        conns = [{
            "id": "anchor", "type": "foot_anchor", "a": "upper", "b": "foot",
            "chain": ["upper", "lower"], "point": [31 / w, 55 / h],
        }]
        _validate_rig_mask_placement(
            modules, conns, {"upper": upper, "lower": lower, "foot": foot}, w, h, 70.0
        )

    def test_foot_anchor_off_lower_or_foot_raises(self):
        h, w = 80, 60
        upper = np.zeros((h, w), dtype=bool)
        upper[8:28, 18:40] = True
        lower = np.zeros((h, w), dtype=bool)
        lower[26:56, 20:42] = True
        foot = np.zeros((h, w), dtype=bool)
        foot[54:72, 18:44] = True
        modules = [
            {"id": "upper", "pivot": [29 / w, 18 / h], "motion_type": "rigid"},
            {"id": "lower", "pivot": [31 / w, 40 / h], "motion_type": "rigid"},
            {"id": "foot", "pivot": [31 / w, 55 / h], "motion_type": "static"},
        ]
        masks = {"upper": upper, "lower": lower, "foot": foot}
        far = {"id": "anchor", "type": "foot_anchor", "a": "upper", "b": "foot",
               "chain": ["upper", "lower"], "point": [4 / w, 4 / h]}
        with self.assertRaises(VisionReviewRequired) as ctx:
            _validate_rig_mask_placement(modules, [far], masks, w, h, 70.0)
        self.assertNotIn("inf", str(ctx.exception).lower())
        # on foot but below lower
        only_foot = dict(far)
        only_foot["point"] = [31 / w, 68 / h]
        with self.assertRaises(VisionReviewRequired):
            _validate_rig_mask_placement(modules, [only_foot], masks, w, h, 70.0)


if __name__ == "__main__":
    unittest.main()
