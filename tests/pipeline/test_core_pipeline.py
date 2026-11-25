"""Comprehensive unit tests for Portrait Spine Studio core pipeline, Mivo adapter, gates, and export."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

from src.export.exporter import export_default_package, export_transparent_web
from src.gates.review_binding import ExportApprovalError
from src.gates.audit import GateFailureError, run_audit, stress_test, count_enclosed_holes
from src.mivo.adapter import (
    DeterministicFallbackMock,
    FileBackedMivoAdapter,
    MivoCanvasError,
    StageRunner,
)
from src.pipeline.build_rig import build_rig
from src.pipeline.preprocess import preprocess_portrait
from src.pipeline.presets import list_presets
from src.pipeline.split_parts import split_parts
from src.render.renderer import SpineRenderer, find_ffmpeg
from src.render.spine_runtime import Skeleton


def _make_synthetic_portrait(path: Path, width: int = 160, height: int = 240) -> Tuple[Path, List[Dict[str, Any]], Dict[str, Any]]:
    """Create a clean multi-part RGBA synthetic character portrait + matching part_specs & rig_spec."""
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    dr = ImageDraw.Draw(img)

    # Back hair (purple-dark) behind head & torso
    dr.polygon([(40, 24), (126, 24), (138, 128), (112, 148), (44, 136), (32, 74)], fill=(70, 48, 96, 255))
    # Legs (blue-grey)
    dr.rectangle([56, 150, 76, 224], fill=(60, 72, 92, 255))
    dr.rectangle([84, 150, 104, 224], fill=(60, 72, 92, 255))
    # Torso / waist (teal cloth + warm skin neck)
    dr.rectangle([54, 78, 106, 158], fill=(42, 138, 142, 255))
    dr.rectangle([70, 62, 90, 84], fill=(226, 186, 154, 255))
    # Left arm (upper + forearm overlapping torso edge)
    dr.polygon([(54, 84), (36, 118), (52, 126), (64, 96)], fill=(224, 182, 150, 255))
    dr.polygon([(36, 118), (56, 146), (66, 138), (50, 114)], fill=(195, 195, 208, 255))
    # Head (warm skin oval)
    dr.ellipse([58, 26, 102, 72], fill=(230, 190, 158, 255))

    img.save(path, format="PNG")

    part_specs: List[Dict[str, Any]] = [
        {
            "name": "back_hair",
            "kind": "soft",
            "parent": "torso",
            "pivot": [80, 50],
            "own": [[(32, 24), (138, 24), (138, 148), (32, 148)]],
            "band": 3,
            "depth": 22,
            "guides": [(None, None, "notskin", "patch")],
            "centerline": [[92, 54], [110, 88], [122, 126]],
        },
        {
            "name": "legs",
            "kind": "static",
            "parent": None,
            "pivot": [80, 155],
            "own": [[(54, 150), (106, 150), (106, 226), (54, 226)]],
            "band": 2,
            "depth": 18,
            "guides": [(None, None, None, "smooth")],
        },
        {
            "name": "torso",
            "kind": "soft",
            "parent": "legs",
            "pivot": [80, 118],
            "own": [[(52, 62), (108, 62), (108, 158), (52, 158)]],
            "band": 3,
            "depth": 22,
            "guides": [
                ([[(64, 56), (96, 56), (96, 90), (64, 90)]], None, "skin", "smooth"),
                (None, None, None, "smooth"),
            ],
        },
        {
            "name": "arm_L",
            "kind": "soft",
            "parent": "torso",
            "pivot": [56, 88],
            "own": [[(34, 82), (66, 82), (68, 148), (34, 148)]],
            "band": 2,
            "depth": 16,
            "guides": [(None, None, None, "smooth")],
        },
        {
            "name": "head",
            "kind": "rigid",
            "parent": "torso",
            "pivot": [80, 68],
            "own": [[(56, 24), (104, 24), (104, 74), (56, 74)]],
            "band": 2,
            "depth": 12,
            "guides": [(None, None, "skin", "smooth")],
        },
    ]

    rig_spec: Dict[str, Any] = {
        "character": "synth",
        "duration": 8.0,
        "fps": 30,
        "key_step": 0.25,
        "bones": [
            {"name": "root", "parent": None, "pos": [80, 240]},
            {"name": "hips", "parent": "root", "pos": [80, 152], "aim": [80, 134]},
            {"name": "spine", "parent": "hips", "pos": [80, 134], "aim": [80, 108]},
            {"name": "chest", "parent": "spine", "pos": [80, 108], "aim": [80, 82]},
            {"name": "ribcage", "parent": "chest", "pos": [80, 108]},
            {"name": "neck", "parent": "chest", "pos": [80, 82], "aim": [80, 68]},
            {"name": "head", "parent": "neck", "pos": [80, 68], "aim": [80, 32]},
            {"name": "ear_R", "parent": "head", "pos": [96, 36], "aim": [104, 24]},
            {"name": "shoulder_L", "parent": "chest", "pos": [56, 88], "aim": [42, 118]},
            {"name": "elbow_L", "parent": "shoulder_L", "pos": [42, 118], "aim": [60, 142]},
            {"name": "wrist_target_L", "parent": "hips", "pos": [60, 142]},
            {"name": "foot_L", "parent": "root", "pos": [66, 210], "aim": [66, 224]},
            {"name": "toe_target_L", "parent": "root", "pos": [66, 224]},
        ],
        "chains": {
            "back_hair": {"parent": "neck", "part": "back_hair", "count": 2},
        },
        "ik": [
            # 2-bone IK with bendPositive omitted -> auto-computed
            {"name": "left_arm_ik", "bones": ["shoulder_L", "elbow_L"], "target": "wrist_target_L"},
            # 1-bone aim IK
            {"name": "left_foot_ik", "bones": ["foot_L"], "target": "toe_target_L"},
        ],
        "skin": {
            "back_hair": {
                "type": "mesh",
                "recipe": "chain",
                "chain": "back_hair",
                "cell": 16,
                "contact_overrides": [
                    {"part": "arm_L", "bone": "shoulder_L", "d_in": 4, "d_out": 16},
                ],
            },
            "legs": {"type": "region", "bone": "foot_L"},
            "torso": {
                "type": "mesh",
                "recipe": "linear_y",
                "bones": ["chest", "spine", "hips"],
                "ranges": [[108, 134], [134, 152]],
                "cell": 16,
                "contact_overrides": [
                    {"part": "arm_L", "bone": "shoulder_L", "d_in": 2, "d_out": 12},
                ],
            },
            "arm_L": {
                "type": "mesh",
                "recipe": "limb",
                "start_bone": "chest",
                "mid_bone": "shoulder_L",
                "end_bone": "elbow_L",
                "cell": 12,
            },
            "head": {"type": "region", "bone": "head"},
        },
        "motion": {
            # Gentle hips translation so synthetic 160x240 arm IK remains comfortably reachable
            "hips": {
                "tx": [(2.0, 1, 0.2)],
                "ty": [(1.0, 2, 0.1)],
            },
        },
    }
    return path, part_specs, rig_spec


class TestCorePipeline(unittest.TestCase):
    def test_old_transparency_cannot_exempt_new_tear(self) -> None:
        rest = np.full((128, 128), 255, dtype=np.uint8)
        rest[:8] = rest[-8:] = 0
        rest[:, :8] = rest[:, -8:] = 0
        rest[60:68, 8:42] = 0
        frame = np.zeros((128, 128, 4), dtype=np.uint8)
        frame[..., 3] = 255
        frame[:8, :, 3] = frame[-8:, :, 3] = 0
        frame[:, :8, 3] = frame[:, -8:, 3] = 0
        frame[50:80, 40:100, 3] = 0
        self.assertEqual(count_enclosed_holes(frame, rest), 1784)
        # Small tears and tears adjoining an old enclosed hole remain visible to the gate.
        rest[24:28, 24:28] = 0
        frame[..., 3] = rest
        frame[27:29, 27:29, 3] = 0
        self.assertEqual(count_enclosed_holes(frame, rest), 3)

    def test_empty_ik_and_full_body_skin_fail_overall(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source, parts, spec = _make_synthetic_portrait(root / "input.png")
            split_parts(source, parts, root / "parts")
            build_rig(root / "parts", spec, root / "rig", preset="agile_full")
            baseline = run_audit(root / "parts", root / "rig", sample_count=2, stress_samples=2)
            self.assertTrue(baseline["structuralChecks"]["ik_binding"]["passed"])
            self.assertTrue(baseline["structuralChecks"]["required_parts"]["passed"])
            spine_path = root / "rig" / "synth.spine.json"
            spine = json.loads(spine_path.read_text())
            for slot in spine["slots"]:
                if slot["name"] == "legs":
                    slot["bone"] = "root"
            spine_path.write_text(json.dumps(spine))
            failed = run_audit(root / "parts", root / "rig", sample_count=2, stress_samples=2)
            self.assertEqual(failed["status"], "fail")
            self.assertIn("left_foot_ik", failed["structuralChecks"]["ik_binding"]["hollowIkConstraints"])
            self.assertIn("lower_body", failed["structuralChecks"]["required_parts"]["missingOrUnweighted"])
            self.assertFalse(failed["structuralChecks"]["weapon_rigidity"]["applicable"])
            with self.assertRaises(GateFailureError) as error:
                run_audit(root / "parts", root / "rig", sample_count=2, stress_samples=2, raise_on_fail=True)
            self.assertIn("ik_binding", str(error.exception))
            # Half-body presets do not require lower limbs, but still detect missing core skin.
            report_path = root / "rig" / "rig-build-report.json"
            report = json.loads(report_path.read_text())
            report["preset"] = "bust_half"
            report_path.write_text(json.dumps(report))
            spine["ik"] = [ik for ik in spine["ik"] if ik["name"] != "left_foot_ik"]
            for slot in spine["slots"]:
                if slot["name"] == "head":
                    slot["bone"] = "root"
            spine_path.write_text(json.dumps(spine))
            half = run_audit(root / "parts", root / "rig", sample_count=2, stress_samples=2)
            missing = half["structuralChecks"]["required_parts"]["missingOrUnweighted"]
            self.assertNotIn("lower_body", missing)
            self.assertIn("head", missing)

    def test_preprocess_downscales_over_2400px_with_notice(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            big_src = tdp / "big.png"
            Image.new("RGBA", (3000, 1500), (120, 80, 200, 255)).save(big_src)
            out_png = tdp / "out_big.png"

            res = preprocess_portrait(big_src, out_png)
            self.assertTrue(res["resized"])
            self.assertEqual(res["original_size"], [3000, 1500])
            self.assertEqual(res["new_size"], [2400, 1200])
            self.assertIsNotNone(res["notice"])
            self.assertIn("2400", res["notice"])
            with Image.open(out_png) as im:
                self.assertEqual(im.size, (2400, 1200))
                self.assertEqual(im.mode, "RGBA")

            small_src = tdp / "small.png"
            Image.new("RGB", (800, 1200), (20, 40, 60)).save(small_src)
            out_small = tdp / "out_small.png"
            res_small = preprocess_portrait(small_src, out_small)
            self.assertFalse(res_small["resized"])
            self.assertEqual(res_small["new_size"], [800, 1200])
            self.assertIsNone(res_small["notice"])

    def test_split_parts_with_guides_cache_and_rest_diff(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            src_png, part_specs, _ = _make_synthetic_portrait(tdp / "portrait.png")
            parts_dir = tdp / "parts"

            r1 = split_parts(src_png, part_specs, parts_dir)
            self.assertTrue(r1["rest_diff"]["passed"])
            self.assertLessEqual(r1["rest_diff"]["maxRGBA"], 2)
            self.assertTrue((parts_dir / "parts.json").is_file())
            self.assertTrue((parts_dir / "rest_composite.png").is_file())
            self.assertTrue((parts_dir / "rest_diff.json").is_file())
            self.assertTrue((parts_dir / "review" / "part_map.png").is_file())
            self.assertEqual(r1["cache_hits"], 0)

            # Second run should hit per-part hash cache in work/
            r2 = split_parts(src_png, part_specs, parts_dir)
            self.assertGreater(r2["cache_hits"], 0)
            self.assertLessEqual(r2["rest_diff"]["maxRGBA"], 2)

    def test_build_rig_contact_dist_ik_presets_and_gain(self) -> None:
        self.assertEqual(set(list_presets()), {"agile_full", "heavy_weapon_full", "bust_half"})
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            src_png, part_specs, rig_spec = _make_synthetic_portrait(tdp / "portrait.png")
            parts_dir = tdp / "parts"
            split_parts(src_png, part_specs, parts_dir)

            # Test standalone contact_dist recipe as well
            rig_spec_cd = json.loads(json.dumps(rig_spec))
            rig_spec_cd["skin"]["torso"] = {
                "type": "mesh",
                "recipe": "contact_dist",
                "base_bone": "ribcage",
                "contacts": [{"part": "arm_L", "bone": "shoulder_L", "d_in": 2, "d_out": 14}],
                "cell": 16,
            }

            for preset_name in ("agile_full", "heavy_weapon_full", "bust_half"):
                out_dir = tdp / f"rig_{preset_name}"
                res = build_rig(parts_dir, rig_spec_cd, out_dir, preset=preset_name, gain=1.0)
                spine_path = Path(res["spine_path"])
                self.assertTrue(spine_path.is_file())
                self.assertTrue((out_dir / "rig-build-report.json").is_file())

                skel = json.loads(spine_path.read_text(encoding="utf-8"))
                self.assertEqual(skel["skeleton"]["spine"], "4.2.43")
                self.assertEqual(len(skel["ik"]), 2)
                self.assertIn("bendPositive", skel["ik"][0])
                self.assertIsInstance(skel["ik"][0]["bendPositive"], bool)

                # Verify 4px-padded images exist
                for slot in skel["slots"]:
                    self.assertTrue((out_dir / "images" / f"{slot['attachment']}.png").is_file())

            # Verify gain scales animation amplitude
            out_g1 = tdp / "rig_g1"
            out_g2 = tdp / "rig_g2"
            build_rig(parts_dir, rig_spec, out_g1, preset="agile_full", gain=1.0)
            build_rig(parts_dir, rig_spec, out_g2, preset="agile_full", gain=2.0)
            sk1 = Skeleton(out_g1 / "synth.spine.json")
            sk2 = Skeleton(out_g2 / "synth.spine.json")
            sk1.apply("idle", 2.0)
            sk2.apply("idle", 2.0)
            base_rot = float(sk1.by_name["spine"].data.get("rotation", 0.0))
            d1 = abs(sk1.by_name["spine"].rotation - base_rot)
            d2 = abs(sk2.by_name["spine"].rotation - base_rot)
            self.assertAlmostEqual(d2, 2.0 * d1, places=3)

    def test_audit_all_four_hard_gates_pass_and_soft_hints(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            src_png, part_specs, rig_spec = _make_synthetic_portrait(tdp / "portrait.png")
            parts_dir = tdp / "parts"
            rig_dir = tdp / "rig"
            split_parts(src_png, part_specs, parts_dir)
            build_rig(parts_dir, rig_spec, rig_dir, preset="agile_full", gain=1.0)

            st = stress_test(parts_dir, rig_dir, stress_gain=1.5, sample_count=4)
            self.assertTrue(st["passed"])
            self.assertEqual(st["enclosedHolesMax"], 0)

            hr = run_audit(parts_dir, rig_dir, sample_count=8, stress_gain=1.5, stress_samples=4)
            self.assertEqual(hr["status"], "pass")
            for gate_name in ("rest_diff", "setup_identity", "no_holes", "anchor_stability"):
                self.assertTrue(hr["hardGates"][gate_name]["passed"], f"Gate {gate_name} failed: {hr['hardGates'][gate_name]}")
            self.assertTrue((rig_dir / "health-report.json").is_file())
            for lm in ("head", "chest", "hips", "hair_tip"):
                self.assertIn(lm, hr["softHints"]["landmarks"])
                self.assertIn(hr["softHints"]["landmarks"][lm]["status"], {"ok", "warn_low", "warn_high"})

    def test_audit_intentional_hard_gate_failures(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            src_png, part_specs, rig_spec = _make_synthetic_portrait(tdp / "portrait.png")
            parts_dir = tdp / "parts"
            rig_dir = tdp / "rig"
            split_parts(src_png, part_specs, parts_dir)
            build_rig(parts_dir, rig_spec, rig_dir, preset="agile_full", gain=1.0)

            # 1. Fail rest_diff
            rd_path = parts_dir / "rest_diff.json"
            orig_rd = rd_path.read_text(encoding="utf-8")
            bad_rd = json.loads(orig_rd)
            bad_rd["maxRGBA"] = 18
            rd_path.write_text(json.dumps(bad_rd), encoding="utf-8")
            hr1 = run_audit(parts_dir, rig_dir, sample_count=4, stress_samples=2)
            self.assertEqual(hr1["status"], "fail")
            self.assertFalse(hr1["hardGates"]["rest_diff"]["passed"])
            rd_path.write_text(orig_rd, encoding="utf-8")

            # 2. Fail setup_identity (perturb frame 0 rotation)
            spine_path = rig_dir / "synth.spine.json"
            orig_spine = spine_path.read_text(encoding="utf-8")
            bad_skel = json.loads(orig_spine)
            bad_skel["animations"]["idle"]["bones"]["head"]["rotate"][0]["value"] = 12.0
            spine_path.write_text(json.dumps(bad_skel), encoding="utf-8")
            hr2 = run_audit(parts_dir, rig_dir, sample_count=4, stress_samples=2)
            self.assertFalse(hr2["hardGates"]["setup_identity"]["passed"])
            self.assertGreater(hr2["hardGates"]["setup_identity"]["measured"]["frame0VsSetup"], 0)
            spine_path.write_text(orig_spine, encoding="utf-8")

            # 3. Fail anchor_stability (push IK target far beyond arm reach)
            bad_ik = json.loads(orig_spine)
            bad_ik["animations"]["idle"]["bones"]["wrist_target_L"] = {
                "translate": [
                    {"x": 0.0, "y": 0.0},
                    {"time": 4.0, "x": 140.0, "y": 0.0},
                    {"time": 8.0, "x": 0.0, "y": 0.0},
                ]
            }
            spine_path.write_text(json.dumps(bad_ik), encoding="utf-8")
            hr3 = run_audit(parts_dir, rig_dir, sample_count=8, stress_samples=2)
            self.assertFalse(hr3["hardGates"]["anchor_stability"]["passed"])
            self.assertGreater(hr3["hardGates"]["anchor_stability"]["measured"]["maxDriftPx"], 1.5)
            with self.assertRaises(GateFailureError):
                run_audit(parts_dir, rig_dir, sample_count=8, stress_samples=2, raise_on_fail=True)
            spine_path.write_text(orig_spine, encoding="utf-8")

            # 4. Fail no_holes (remove hidden fill under head & shift head sharply inside torso)
            no_fill_specs = [dict(p, depth=0) for p in part_specs]
            parts_nofill = tdp / "parts_nofill"
            rig_nofill = tdp / "rig_nofill"
            split_parts(src_png, no_fill_specs, parts_nofill)
            rig_spec_tear = json.loads(json.dumps(rig_spec))
            rig_spec_tear["motion"]["head"] = {"tx": [(18.0, 1, 0.0)], "ty": [(18.0, 1, 0.0)]}
            build_rig(parts_nofill, rig_spec_tear, rig_nofill, preset="bust_half", gain=1.0)
            hr4 = run_audit(parts_nofill, rig_nofill, sample_count=8, stress_samples=4)
            # Shifting head without neck fill exposes a gap or enclosed hole when surrounded;
            # punch an explicit enclosed hole scenario by shifting torso slightly with zero fill behind arm
            if hr4["hardGates"]["no_holes"]["passed"]:
                # Create a ring-shaped occluder with a center patch that shifts out
                ring_src = tdp / "ring.png"
                im = Image.new("RGBA", (120, 120), (0, 0, 0, 0))
                d = ImageDraw.Draw(im)
                d.rectangle([15, 15, 105, 105], fill=(80, 120, 160, 255))
                d.rectangle([45, 45, 75, 75], fill=(220, 160, 100, 255))
                im.save(ring_src)
                ps_ring = [
                    {"name": "base", "kind": "static", "parent": None, "pivot": [60, 60], "own": [[(15, 15), (105, 15), (105, 105), (15, 105)]], "band": 0, "depth": 0},
                    {"name": "patch", "kind": "rigid", "parent": "base", "pivot": [60, 60], "own": [[(45, 45), (75, 45), (75, 75), (45, 75)]], "band": 0, "depth": 0},
                ]
                rs_ring = {
                    "character": "ring",
                    "bones": [
                        {"name": "root", "parent": None, "pos": [60, 120]},
                        {"name": "patch", "parent": "root", "pos": [60, 60]},
                    ],
                    "skin": {"base": {"type": "region", "bone": "root"}, "patch": {"type": "region", "bone": "patch"}},
                    "motion": {"patch": {"tx": [(14.0, 1, 0.0)]}},
                }
                split_parts(ring_src, ps_ring, tdp / "parts_ring")
                build_rig(tdp / "parts_ring", rs_ring, tdp / "rig_ring", preset="bust_half", gain=1.0)
                hr4 = run_audit(tdp / "parts_ring", tdp / "rig_ring", sample_count=8, stress_samples=2)
            self.assertTrue(hr4["hardGates"]["no_holes"]["passed"])
            self.assertTrue(hr4["hardGates"]["no_holes"].get("advisory"))
            self.assertGreater(hr4["hardGates"]["no_holes"]["measured"]["enclosedHolesMax"], 0)

    def test_mivo_adapter_and_fail_stop_preserves_intermediates(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            src_png = tdp / "src.png"
            mask_png = tdp / "mask.png"
            Image.new("RGBA", (64, 64), (100, 140, 180, 255)).save(src_png)
            Image.new("L", (64, 64), 0).save(mask_png)

            # 1. Mock fails at inpaint_occluded -> StageRunner fail-stops, writes stage-error.json, preserves matted.png
            mock = DeterministicFallbackMock(fail_on="inpaint_occluded", error_code="MIVO_TIMEOUT")
            work_dir = tdp / "work"
            runner = StageRunner(work_dir)
            matted_path = work_dir / "matted.png"
            inpainted_path = work_dir / "inpainted.png"

            with self.assertRaises(MivoCanvasError):
                runner.run_stages(
                    [
                        ("preprocess", lambda: preprocess_portrait(src_png, work_dir / "preprocessed.png")),
                        ("auto_matting", lambda: mock.auto_matting(work_dir / "preprocessed.png", matted_path)),
                        ("inpaint_occluded", lambda: mock.inpaint_occluded(matted_path, mask_png, "fill skin", inpainted_path)),
                    ],
                    raise_on_error=True,
                )

            err_json_path = work_dir / "stage-error.json"
            self.assertTrue(err_json_path.is_file())
            err_data = json.loads(err_json_path.read_text(encoding="utf-8"))
            self.assertEqual(err_data["status"], "failed")
            self.assertEqual(err_data["failedStage"], "inpaint_occluded")
            self.assertEqual(err_data["errorCode"], "MIVO_TIMEOUT")
            self.assertIn("retry", err_data["recoveryActions"])
            self.assertIn("discard", err_data["recoveryActions"])
            # Prior intermediate artifacts must still exist on disk
            self.assertTrue((work_dir / "preprocessed.png").is_file())
            self.assertTrue(matted_path.is_file())
            self.assertIn("auto_matting", err_data["preservedIntermediates"])

            # 2. FileBackedMivoAdapter succeeds when handoff file exists, fails cleanly when absent
            handoff = tdp / "handoff"
            handoff.mkdir()
            input_dir = handoff / hashlib.sha256(src_png.read_bytes()).hexdigest()
            input_dir.mkdir()
            cutout = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
            ImageDraw.Draw(cutout).rectangle([16, 8, 48, 56], fill=(100, 140, 180, 255))
            cutout.save(input_dir / "matted.png")
            fb = FileBackedMivoAdapter(handoff)
            res_ok = fb.auto_matting(src_png, tdp / "matted_out.png")
            self.assertEqual((res_ok.width, res_ok.height), (64, 64))
            self.assertEqual(res_ok.metadata["source"], "file_handoff")
            # A different image with the same name cannot reuse the previous cutout.
            Image.new("RGBA", (64, 64), (110, 140, 180, 255)).save(src_png)
            with self.assertRaises(MivoCanvasError) as mismatch:
                fb.auto_matting(src_png, tdp / "matted_out.png")
            self.assertEqual(mismatch.exception.code, "MIVO_HANDOFF_MISSING")
            bad_dir = handoff / hashlib.sha256(src_png.read_bytes()).hexdigest()
            bad_dir.mkdir()
            cutout.resize((32, 32)).save(bad_dir / "matted.png")
            with self.assertRaises(MivoCanvasError) as size_error:
                fb.auto_matting(src_png, tdp / "matted_out.png")
            self.assertEqual(size_error.exception.code, "MIVO_HANDOFF_SIZE_MISMATCH")
            Image.new("RGB", (64, 64), (100, 140, 180)).save(bad_dir / "matted.png")
            with self.assertRaises(MivoCanvasError) as opaque_error:
                fb.auto_matting(src_png, tdp / "matted_out.png")
            self.assertEqual(opaque_error.exception.code, "MIVO_MATTING_REQUIRED")
            with self.assertRaises(MivoCanvasError):
                fb.inpaint_occluded(src_png, mask_png, "fill", tdp / "missing_inpaint.png")

    def test_legacy_machine_gates_cannot_approve_production_exports(self) -> None:
        # Positive ZIP and Web codec checks now use a real V2 fixture plus explicit
        # structured review in tests.studio.test_complete_module_flow. A legacy
        # four-gate pass must no longer be treated as complete-module acceptance.
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            src_png, part_specs, rig_spec = _make_synthetic_portrait(tdp / "portrait.png")
            parts_dir, rig_dir = tdp / "parts", tdp / "rig"
            split_parts(src_png, part_specs, parts_dir)
            build_rig(parts_dir, rig_spec, rig_dir, preset="agile_full", gain=1.0)
            run_audit(parts_dir, rig_dir, sample_count=6, stress_samples=2)
            with self.assertRaises(ExportApprovalError):
                export_default_package(rig_dir, tdp / "export")
            with self.assertRaises(ExportApprovalError):
                export_transparent_web(rig_dir, tdp / "web")
            self.assertFalse((tdp / "export").exists())
            self.assertFalse((tdp / "web").exists())


if __name__ == "__main__":
    unittest.main()
