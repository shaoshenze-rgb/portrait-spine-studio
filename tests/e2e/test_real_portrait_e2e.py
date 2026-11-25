"""End-to-end acceptance tests on a real 2D character portrait (portrait-erika.webp)."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_E2E_SCRIPT = REPO_ROOT / "scripts" / "run_e2e.py"
REAL_ERIKA_PATH = REPO_ROOT.parent / "torchlight-newweb" / "assets" / "source" / "select" / "portrait-erika.webp"


def _build_erika_specs(width: int = 525, height: int = 1192) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Build parts_spec and rig_spec tailored to portrait-erika.webp (525x1192 RGBA full-body)."""
    sx = width / 525.0
    sy = height / 1192.0

    def pt(x: float, y: float) -> List[float]:
        return [round(x * sx, 2), round(y * sy, 2)]

    def poly(coords: List[Tuple[float, float]]) -> List[List[float]]:
        return [pt(x, y) for x, y in coords]

    part_specs: List[Dict[str, Any]] = [
        {
            "name": "tail",
            "kind": "soft",
            "parent": "torso",
            "pivot": pt(375, 760),
            "own": [poly([(365, 675), (525, 675), (525, 1192), (365, 1192)])],
            "band": 4,
            "depth": 40,
            "guides": [(None, None, "dark", "patch")],
            "centerline": [pt(385, 770), pt(445, 860), pt(495, 975), pt(505, 1060)],
        },
        {
            "name": "back_hair",
            "kind": "soft",
            "parent": "torso",
            "pivot": pt(245, 220),
            "own": [poly([(225, 0), (525, 0), (525, 515), (275, 515), (225, 310)])],
            "band": 5,
            "depth": 56,
            "guides": [(None, None, "notskin", "patch")],
            "centerline": [pt(280, 195), pt(355, 265), pt(435, 345), pt(495, 435)],
        },
        {
            "name": "legs",
            "kind": "static",
            "parent": None,
            "pivot": pt(255, 650),
            "own": [poly([(0, 610), (435, 610), (435, 1192), (0, 1192)])],
            "band": 4,
            "depth": 44,
            "guides": [(None, None, None, "smooth")],
        },
        {
            "name": "torso",
            "kind": "soft",
            "parent": "legs",
            "pivot": pt(250, 445),
            "own": [poly([(0, 250), (525, 250), (525, 675), (0, 675)])],
            "band": 4,
            "depth": 56,
            "guides": [
                ([poly([(175, 235), (275, 235), (275, 325), (175, 325)])], None, "skin", "smooth"),
                (None, None, None, "smooth"),
            ],
        },
        {
            "name": "arm_R",
            "kind": "soft",
            "parent": "torso",
            "pivot": pt(305, 320),
            "own": [poly([(315, 290), (525, 290), (525, 675), (365, 675)])],
            "band": 4,
            "depth": 44,
            "guides": [(None, None, None, "smooth")],
        },
        {
            "name": "arm_L",
            "kind": "soft",
            "parent": "torso",
            "pivot": pt(155, 315),
            "own": [poly([(0, 285), (175, 285), (210, 610), (0, 610)])],
            "band": 4,
            "depth": 44,
            "guides": [(None, None, None, "smooth")],
        },
        {
            "name": "head",
            "kind": "rigid",
            "parent": "torso",
            "pivot": pt(215, 270),
            "own": [poly([(0, 0), (295, 0), (295, 290), (0, 290)])],
            "band": 4,
            "depth": 32,
            "guides": [(None, None, "skin", "smooth")],
        },
    ]

    rig_spec: Dict[str, Any] = {
        "character": "erika",
        "duration": 8.0,
        "fps": 30,
        "key_step": 0.25,
        "bones": [
            {"name": "root", "parent": None, "pos": pt(262, 1192)},
            {"name": "hips", "parent": "root", "pos": pt(250, 580), "aim": pt(246, 510)},
            {"name": "spine", "parent": "hips", "pos": pt(246, 510), "aim": pt(238, 425)},
            {"name": "chest", "parent": "spine", "pos": pt(238, 425), "aim": pt(225, 330)},
            {"name": "ribcage", "parent": "chest", "pos": pt(238, 425)},
            {"name": "neck", "parent": "chest", "pos": pt(222, 315), "aim": pt(215, 265)},
            {"name": "head", "parent": "neck", "pos": pt(215, 265), "aim": pt(195, 130)},
            {"name": "ear_L", "parent": "head", "pos": pt(130, 145), "aim": pt(75, 75)},
            {"name": "ear_R", "parent": "head", "pos": pt(225, 115), "aim": pt(225, 20)},
            {"name": "shoulder_L", "parent": "chest", "pos": pt(155, 315), "aim": pt(55, 430)},
            {"name": "elbow_L", "parent": "shoulder_L", "pos": pt(55, 430), "aim": pt(185, 530)},
            {"name": "wrist_target_L", "parent": "hips", "pos": pt(185, 530)},
            {"name": "shoulder_R", "parent": "chest", "pos": pt(305, 320), "aim": pt(405, 445)},
            {"name": "elbow_R", "parent": "shoulder_R", "pos": pt(405, 445), "aim": pt(495, 540)},
            {"name": "foot_L", "parent": "root", "pos": pt(170, 1140), "aim": pt(170, 1185)},
            {"name": "toe_target_L", "parent": "root", "pos": pt(170, 1185)},
        ],
        "chains": {
            "back_hair": {"parent": "neck", "part": "back_hair", "count": 3},
            "tail": {"parent": "hips", "part": "tail", "count": 3},
        },
        "ik": [
            {"name": "left_arm_ik", "bones": ["shoulder_L", "elbow_L"], "target": "wrist_target_L"},
            {"name": "left_foot_ik", "bones": ["foot_L"], "target": "toe_target_L"},
        ],
        "skin": {
            "tail": {
                "type": "mesh",
                "recipe": "chain",
                "chain": "tail",
                "cell": 22,
            },
            "back_hair": {
                "type": "mesh",
                "recipe": "chain",
                "chain": "back_hair",
                "cell": 24,
            },
            "legs": {
                "type": "region",
                "bone": "foot_L",
            },
            "torso": {
                "type": "mesh",
                "recipe": "linear_y",
                "bones": ["chest", "spine", "hips"],
                "ranges": [[round(425 * sy, 2), round(510 * sy, 2)], [round(510 * sy, 2), round(580 * sy, 2)]],
                "cell": 24,
                "contact_overrides": [
                    {"part": "arm_L", "bone": "shoulder_L", "d_in": 4, "d_out": 20},
                ],
            },
            "arm_R": {
                "type": "mesh",
                "recipe": "limb",
                "start_bone": "chest",
                "mid_bone": "shoulder_R",
                "end_bone": "elbow_R",
                "cell": 20,
            },
            "arm_L": {
                "type": "mesh",
                "recipe": "limb",
                "start_bone": "chest",
                "mid_bone": "shoulder_L",
                "end_bone": "elbow_L",
                "cell": 20,
            },
            "head": {
                "type": "region",
                "bone": "head",
            },
        },
        "motion": {},
    }
    return part_specs, rig_spec


def _make_fallback_portrait(path: Path) -> Tuple[Path, int, int]:
    """Fallback synthetic portrait used only if portrait-erika.webp is absent."""
    w, h = 525, 1192
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    dr = ImageDraw.Draw(im)
    dr.polygon([(240, 120), (500, 180), (510, 480), (260, 460)], fill=(55, 42, 72, 255))
    dr.polygon([(380, 740), (505, 860), (510, 1080), (470, 1090), (370, 790)], fill=(45, 38, 52, 255))
    dr.rectangle([140, 620, 380, 1185], fill=(210, 212, 220, 255))
    dr.rectangle([155, 270, 355, 650], fill=(44, 138, 138, 255))
    dr.polygon([(310, 310), (495, 535), (460, 565), (295, 355)], fill=(215, 175, 145, 255))
    dr.polygon([(155, 315), (45, 430), (185, 545), (205, 515), (90, 430), (175, 345)], fill=(215, 175, 145, 255))
    dr.ellipse([125, 75, 285, 275], fill=(226, 184, 152, 255))
    im.save(path, format="PNG")
    return path, w, h


class TestRealPortraitE2E(unittest.TestCase):
    def test_real_portrait_full_e2e_partial_rerun_and_js_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            if REAL_ERIKA_PATH.is_file():
                portrait_path = REAL_ERIKA_PATH
                with Image.open(portrait_path) as im:
                    w, h = im.size
                using_real = True
            else:
                portrait_path, w, h = _make_fallback_portrait(tdp / "fallback-erika.png")
                using_real = False

            part_specs, rig_spec = _build_erika_specs(w, h)
            parts_spec_path = tdp / "parts_spec.json"
            rig_spec_path = tdp / "rig_spec.json"
            out_dir = tdp / "e2e_out"

            parts_spec_path.write_text(json.dumps(part_specs, indent=2), encoding="utf-8")
            rig_spec_path.write_text(json.dumps(rig_spec, indent=2), encoding="utf-8")

            # 1. Full E2E run via CLI
            proc = subprocess.run(
                [
                    sys.executable,
                    str(RUN_E2E_SCRIPT),
                    "--image", str(portrait_path),
                    "--parts-spec", str(parts_spec_path),
                    "--rig-spec", str(rig_spec_path),
                    "--out-dir", str(out_dir),
                    "--preset", "agile_full",
                    "--gain", "1.0",
                    "--sample-count", "12",
                    "--stress-gain", "1.6",
                    "--stress-samples", "6",
                    "--skip-previews",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(
                proc.returncode,
                0,
                f"Full E2E failed (using_real={using_real}):\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}",
            )

            # Verify core artifacts
            health_path = out_dir / "rig" / "health-report.json"
            build_report_path = out_dir / "rig" / "rig-build-report.json"
            spine_json_path = out_dir / "rig" / "erika.spine.json"
            zip_path = out_dir / "export" / "erika-spine42.zip"
            preview_html_path = out_dir / "preview.html"
            summary_path = out_dir / "pipeline-summary.json"

            for p in (health_path, build_report_path, spine_json_path, zip_path, preview_html_path, summary_path):
                self.assertTrue(p.is_file(), f"Missing expected artifact: {p}")

            # Verify all 4 hard gates PASS
            hr = json.loads(health_path.read_text(encoding="utf-8"))
            self.assertEqual(hr["status"], "pass", f"Health report failed: {json.dumps(hr['hardGates'], indent=2)}")
            gates = hr["hardGates"]
            self.assertTrue(gates["rest_diff"]["passed"])
            self.assertLessEqual(gates["rest_diff"]["measured"]["maxRGBA"], 2)
            self.assertTrue(gates["setup_identity"]["passed"])
            self.assertEqual(gates["setup_identity"]["measured"]["frame0VsSetup"], 0)
            self.assertEqual(gates["setup_identity"]["measured"]["frameLoopEndVsSetup"], 0)
            self.assertLessEqual(gates["setup_identity"]["measured"]["setupVsPartsMaxRGBA"], 2)
            self.assertTrue(gates["no_holes"]["passed"])
            self.assertEqual(gates["no_holes"]["measured"]["enclosedHolesMax"], 0)
            self.assertTrue(gates["anchor_stability"]["passed"])
            self.assertLessEqual(gates["anchor_stability"]["measured"]["maxDriftPx"], 1.5)

            # Verify soft displacement hints computed cleanly
            soft_lms = hr["softHints"]["landmarks"]
            for lm in ("head", "chest", "hips", "hair_tip"):
                self.assertIn(lm, soft_lms)
                self.assertIsNotNone(soft_lms[lm]["ptpPx"])
                self.assertIn(soft_lms[lm]["status"], {"ok", "warn_low", "warn_high"})

            # Verify default export zip contents
            with zipfile.ZipFile(zip_path, "r") as zf:
                names = set(zf.namelist())
                self.assertIn("erika.spine.json", names)
                self.assertIn("rig-build-report.json", names)
                self.assertIn("health-report.json", names)
                self.assertIn("images/head.png", names)
                self.assertIn("images/back_hair.png", names)
                self.assertIn("images/tail.png", names)

            # 2. Minimal rerun (--partial-rerun motion with gain=1.2) reuses cached parts/
            parts_json_path = out_dir / "parts" / "parts.json"
            rest_comp_path = out_dir / "parts" / "rest_composite.png"
            mtime_parts_before = parts_json_path.stat().st_mtime_ns
            mtime_comp_before = rest_comp_path.stat().st_mtime_ns

            proc_rerun = subprocess.run(
                [
                    sys.executable,
                    str(RUN_E2E_SCRIPT),
                    "--rig-spec", str(rig_spec_path),
                    "--out-dir", str(out_dir),
                    "--preset", "agile_full",
                    "--gain", "1.2",
                    "--partial-rerun", "motion",
                    "--sample-count", "12",
                    "--stress-gain", "1.6",
                    "--stress-samples", "6",
                    "--skip-previews",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(
                proc_rerun.returncode,
                0,
                f"Partial rerun failed:\nSTDOUT:\n{proc_rerun.stdout}\nSTDERR:\n{proc_rerun.stderr}",
            )

            rerun_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(rerun_summary["mode"], "partial:motion")
            self.assertIn("preprocess", rerun_summary["skippedStages"])
            self.assertIn("split_parts", rerun_summary["skippedStages"])
            self.assertEqual(parts_json_path.stat().st_mtime_ns, mtime_parts_before)
            self.assertEqual(rest_comp_path.stat().st_mtime_ns, mtime_comp_before)

            br_rerun = json.loads(build_report_path.read_text(encoding="utf-8"))
            self.assertAlmostEqual(float(br_rerun["gain"]), 1.2, places=4)

            hr_rerun = json.loads(health_path.read_text(encoding="utf-8"))
            self.assertEqual(hr_rerun["status"], "pass")
            for gname in ("rest_diff", "setup_identity", "no_holes", "anchor_stability"):
                self.assertTrue(hr_rerun["hardGates"][gname]["passed"], f"Rerun gate {gname} failed")

            # 3. Cross-check T3's src/player/spine42-runtime.js in Node on the real-portrait .spine.json
            runtime_js = (REPO_ROOT / "src" / "player" / "spine42-runtime.js").resolve().as_uri()
            node_script = f"""
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {{ Spine42Skeleton }} from {json.dumps(runtime_js)};

const skJson = JSON.parse(fs.readFileSync({json.dumps(str(spine_json_path))}, 'utf8'));
const sk = new Spine42Skeleton(skJson);
assert.equal(sk.slots.length, 7);
const dur = sk.getAnimationDuration('idle');
assert.equal(dur, 8);

const snap = () => sk.bones.map(b => [b.name, b.worldX, b.worldY, b.a, b.b, b.c, b.d]);
sk.setupPose();
sk.updateWorld();
const setupSnap = snap();

for (const t of [0, 8]) {{
  sk.apply('idle', t);
  const cur = snap();
  for (let i = 0; i < setupSnap.length; i++) {{
    for (let j = 1; j < 7; j++) {{
      const diff = Math.abs(cur[i][j] - setupSnap[i][j]);
      assert.ok(diff <= 1e-6, `t=${{t}} bone=${{setupSnap[i][0]}} field=${{j}} diff=${{diff}}`);
    }}
  }}
}}

sk.apply('idle', 2.0);
const midSnap = snap();
let maxMove = 0;
for (let i = 0; i < setupSnap.length; i++) {{
  const dx = midSnap[i][1] - setupSnap[i][1];
  const dy = midSnap[i][2] - setupSnap[i][2];
  maxMove = Math.max(maxMove, Math.hypot(dx, dy));
}}
assert.ok(maxMove > 1.0, `Expected non-zero motion at t=2.0, got maxMove=${{maxMove}}`);
console.log(JSON.stringify({{ ok: true, bones: sk.bones.length, slots: sk.slots.length, duration: dur, maxMoveAt2s: Number(maxMove.toFixed(3)) }}));
"""
            node_proc = subprocess.run(
                ["node", "--input-type=module", "-e", node_script],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(
                node_proc.returncode,
                0,
                f"Node Spine42Skeleton check failed:\nSTDOUT:\n{node_proc.stdout}\nSTDERR:\n{node_proc.stderr}",
            )
            node_out = json.loads(node_proc.stdout.strip().splitlines()[-1])
            self.assertTrue(node_out["ok"])

    def test_fail_stop_writes_stage_error_and_preserves_intermediates(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            src_png, w, h = _make_fallback_portrait(tdp / "fail_portrait.png")
            part_specs, rig_spec = _build_erika_specs(w, h)
            # Push IK target far beyond arm reach so anchor_stability hard gate fails
            rig_spec["motion"]["wrist_target_L"] = {
                "tx": [(260.0, 1, 0.0)],
            }
            parts_spec_path = tdp / "parts_spec.json"
            rig_spec_path = tdp / "rig_spec.json"
            out_dir = tdp / "fail_out"
            parts_spec_path.write_text(json.dumps(part_specs, indent=2), encoding="utf-8")
            rig_spec_path.write_text(json.dumps(rig_spec, indent=2), encoding="utf-8")

            proc = subprocess.run(
                [
                    sys.executable,
                    str(RUN_E2E_SCRIPT),
                    "--image", str(src_png),
                    "--parts-spec", str(parts_spec_path),
                    "--rig-spec", str(rig_spec_path),
                    "--out-dir", str(out_dir),
                    "--sample-count", "8",
                    "--stress-samples", "2",
                    "--skip-previews",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(proc.returncode, 0)
            err_path = out_dir / "stage-error.json"
            self.assertTrue(err_path.is_file())
            err = json.loads(err_path.read_text(encoding="utf-8"))
            self.assertEqual(err["status"], "failed")
            self.assertEqual(err["failedStage"], "audit")
            self.assertEqual(err["errorCode"], "HARD_GATE_FAILED")
            # Prior intermediates preserved
            self.assertTrue((out_dir / "preprocessed.png").is_file())
            self.assertTrue((out_dir / "parts" / "parts.json").is_file())
            self.assertTrue((out_dir / "rig" / "erika.spine.json").is_file())


if __name__ == "__main__":
    unittest.main()
