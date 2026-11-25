"""Integration tests for Portrait Spine Studio local HTTP server & NL adjuster."""
from __future__ import annotations

import base64
import io
import json
import re
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from PIL import Image, ImageDraw

from src.studio.nl_adjust import classify_and_polish_adjustment
from src.studio.server import StudioJobManager, create_studio_server
from src.studio.vision_provider import GeometryVisionProvider


def _make_portrait_png_b64(width: int = 180, height: int = 260) -> str:
    """Create a clean RGBA character portrait PNG encoded as Base64."""
    im = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    dr = ImageDraw.Draw(im)

    cx = width // 2
    # Back hair behind head & torso
    dr.polygon(
        [
            (int(cx - width * 0.28), int(height * 0.10)),
            (int(cx + width * 0.28), int(height * 0.10)),
            (int(cx + width * 0.34), int(height * 0.56)),
            (int(cx - width * 0.30), int(height * 0.54)),
        ],
        fill=(68, 52, 98, 255),
    )
    # Legs
    dr.rectangle(
        [int(cx - width * 0.16), int(height * 0.62), int(cx - width * 0.03), int(height * 0.94)],
        fill=(56, 70, 92, 255),
    )
    dr.rectangle(
        [int(cx + width * 0.03), int(height * 0.62), int(cx + width * 0.16), int(height * 0.94)],
        fill=(56, 70, 92, 255),
    )
    # Torso
    dr.rectangle(
        [int(cx - width * 0.18), int(height * 0.30), int(cx + width * 0.18), int(height * 0.66)],
        fill=(44, 136, 144, 255),
    )
    # Neck
    dr.rectangle(
        [int(cx - width * 0.06), int(height * 0.24), int(cx + width * 0.06), int(height * 0.33)],
        fill=(226, 188, 156, 255),
    )
    # Head
    dr.ellipse(
        [int(cx - width * 0.14), int(height * 0.10), int(cx + width * 0.14), int(height * 0.28)],
        fill=(232, 194, 162, 255),
    )

    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class TestNLAdjustClassifier(unittest.TestCase):
    def test_layer_classification(self) -> None:
        a = classify_and_polish_adjustment("右边长发摆动加大", current_preset="agile_full", current_gain=1.0)
        self.assertEqual(a["rerun_layer"], "motion")
        self.assertIn("split_parts", a["skipped_stages"])
        self.assertIn("mivo", a["skipped_stages"])
        self.assertTrue(a["spec_patch"]["rig_motion_overrides"])

        b = classify_and_polish_adjustment("左手固定别动", current_preset="agile_full", current_gain=1.0)
        self.assertEqual(b["rerun_layer"], "rig")
        self.assertIn("split_parts", b["skipped_stages"])

        c = classify_and_polish_adjustment("重新拆分披风并扩大遮挡补全", current_preset="agile_full", current_gain=1.0)
        self.assertEqual(c["rerun_layer"], "split")
        self.assertNotIn("split_parts", c["skipped_stages"])
        self.assertIn("mivo", c["skipped_stages"])


class TestStudioServerIntegration(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.work_dir = Path(self._tmpdir.name) / "studio-work"
        mgr = StudioJobManager(self.work_dir, audit_samples=8, stress_samples=4,
                               vision_provider=GeometryVisionProvider())
        self.server = create_studio_server(host="127.0.0.1", port=0, work_dir=self.work_dir, manager=mgr)
        self.port = self.server.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}"
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self._thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self._thread.join(timeout=3)
        self._tmpdir.cleanup()

    def _request(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Tuple[int, bytes, Dict[str, str]]:
        url = self.base_url + path
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"Content-Type": "application/json"} if data is not None else {}
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.status, resp.read(), dict(resp.headers.items())
        except urllib.error.HTTPError as err:
            return err.code, err.read(), dict(err.headers.items())

    def _json(self, method: str, path: str, payload: Optional[Dict[str, Any]] = None) -> Tuple[int, Dict[str, Any]]:
        status, raw, _ = self._request(method, path, payload)
        return status, json.loads(raw.decode("utf-8"))

    def test_health_and_static_assets_zero_external_urls(self) -> None:
        status, health = self._json("GET", "/api/health")
        self.assertEqual(status, 200)
        self.assertTrue(health["ok"])
        self.assertFalse(health["busy"])
        self.assertIsNone(health["active_job_id"])

        status_html, html_bytes, _ = self._request("GET", "/")
        self.assertEqual(status_html, 200)
        html_text = html_bytes.decode("utf-8")
        self.assertIn("Portrait Spine Studio", html_text)
        self.assertIn("/player/player-widget.js", html_text)
        self.assertIsNone(re.search(r"https?://", html_text), "index.html must contain zero external URLs")

        status_rt, rt_bytes, headers = self._request("GET", "/player/spine42-runtime.js")
        self.assertEqual(status_rt, 200)
        self.assertIn("Spine42Skeleton", rt_bytes.decode("utf-8"))
        self.assertIn("javascript", headers.get("Content-Type", "").lower())

    def test_full_diagnostic_stays_pending_and_over_2400px_resize_notice(self) -> None:
        # 360x2560 tall portrait exceeds 2400px long edge -> triggers LANCZOS auto-downscale notice
        b64_large = _make_portrait_png_b64(width=360, height=2560)
        status, job = self._json(
            "POST",
            "/api/jobs",
            {
                "filename": "hero_oversize.png",
                "pause_before_animation": False,
                "image_base64": b64_large,
                "preset": "agile_full",
                "gain": 1.0,
                "pause_after_split": False,
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(job["status"], "needs_review")
        self.assertIsNotNone(job["resize_notice"])
        self.assertIn("2400", job["resize_notice"])
        self.assertTrue(job["preprocess"]["resized"])
        self.assertLessEqual(max(job["preprocess"]["new_size"]), 2400)

        hr = job["health_report"]
        self.assertEqual(hr["status"], "pass")
        for gate_name in ("rest_diff", "setup_identity", "no_holes", "anchor_stability"):
            self.assertTrue(hr["hardGates"][gate_name]["passed"], f"Gate {gate_name} should PASS")

        # Verify GET /api/jobs/<id> and served rig JSON
        status_get, job2 = self._json("GET", f"/api/jobs/{job['job_id']}")
        self.assertEqual(status_get, 200)
        self.assertEqual(job2["status"], "needs_review")
        status_rig, rig_bytes, _ = self._request("GET", job["rig_url"])
        self.assertEqual(status_rig, 200)
        rig_json = json.loads(rig_bytes.decode("utf-8"))
        self.assertEqual(rig_json["skeleton"]["spine"], "4.2.43")

    def test_pause_after_split_and_continue(self) -> None:
        b64 = _make_portrait_png_b64()
        status, job = self._json(
            "POST",
            "/api/jobs",
            {
                "filename": "paused_char.png",
                "pause_before_animation": False,
                "image_base64": b64,
                "preset": "bust_half",
                "pause_after_split": True,
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(job["status"], "awaiting_split_confirm")
        self.assertTrue(job["part_map_url"])
        self.assertIsNone(job["rig_url"])

        # part_map.png should be served
        pm_status, pm_bytes, _ = self._request("GET", job["part_map_url"])
        self.assertEqual(pm_status, 200)
        self.assertGreater(len(pm_bytes), 100)

        # Resume with /continue
        c_status, resumed = self._json("POST", f"/api/jobs/{job['job_id']}/continue", {})
        self.assertEqual(c_status, 200)
        self.assertEqual(resumed["status"], "needs_review")
        self.assertEqual(resumed["health_report"]["status"], "pass")
        self.assertEqual(resumed["split_run_count"], 1)

    def test_fail_stop_on_mivo_error_retry_and_discard(self) -> None:
        b64 = _make_portrait_png_b64()
        status, failed = self._json(
            "POST",
            "/api/jobs",
            {
                "filename": "fail_char.png",
                "pause_before_animation": False,
                "image_base64": b64,
                "preset": "agile_full",
                "simulate_mivo_error": True,
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["stage_error"]["failedStage"], "mivo")
        self.assertEqual(failed["stage_error"]["errorCode"], "MIVO_SERVICE_UNAVAILABLE")

        job_dir = self.work_dir / failed["job_id"]
        stage_err_path = job_dir / "stage-error.json"
        self.assertTrue(stage_err_path.is_file())
        self.assertTrue((job_dir / "preprocessed.png").is_file())
        self.assertIn("preprocess.output_path", failed["preserved_intermediates"])

        # Retry succeeds reusing preserved preprocessed.png
        r_status, retried = self._json("POST", f"/api/jobs/{failed['job_id']}/retry", {})
        self.assertEqual(r_status, 200)
        self.assertEqual(retried["status"], "needs_review")
        self.assertEqual(retried["health_report"]["status"], "pass")
        self.assertFalse(stage_err_path.exists())

        # Second failed job -> verify /discard cleans error state
        _, failed2 = self._json(
            "POST",
            "/api/jobs",
            {
                "filename": "discard_char.png",
                "pause_before_animation": False,
                "image_base64": b64,
                "simulate_mivo_error": True,
            },
        )
        self.assertEqual(failed2["status"], "failed")
        d_status, discarded = self._json("POST", f"/api/jobs/{failed2['job_id']}/discard", {})
        self.assertEqual(d_status, 200)
        self.assertEqual(discarded["status"], "discarded")
        self.assertIsNone(discarded["stage_error"])
        self.assertFalse((self.work_dir / failed2["job_id"] / "stage-error.json").exists())

    def test_single_task_serial_lock_returns_409(self) -> None:
        b64 = _make_portrait_png_b64()
        results: Dict[str, Any] = {}

        def run_first() -> None:
            st, body = self._json(
                "POST",
                "/api/jobs",
                {
                    "filename": "slow_char.png",
                    "pause_before_animation": False,
                    "image_base64": b64,
                    "simulate_delay_ms": 800,
                },
            )
            results["first"] = (st, body)

        t1 = threading.Thread(target=run_first)
        t1.start()

        # Wait until server reports busy=True
        busy_seen = False
        deadline = time.time() + 5.0
        while time.time() < deadline:
            _, h = self._json("GET", "/api/health")
            if h.get("busy"):
                busy_seen = True
                break
            time.sleep(0.03)
        self.assertTrue(busy_seen, "Server should report busy=True while first job is executing")

        # Concurrent POST /api/jobs must receive 409 Conflict
        st2, body2 = self._json(
            "POST",
            "/api/jobs",
            {"filename": "second_char.png", "image_base64": b64},
        )
        self.assertEqual(st2, 409)
        self.assertEqual(body2["errorCode"], "STUDIO_BUSY")

        t1.join(timeout=30)
        self.assertEqual(results["first"][0], 200)
        self.assertEqual(results["first"][1]["status"], "needs_review")

    def test_adjust_minimal_rerun_remains_legacy_diagnostic(self) -> None:
        b64 = _make_portrait_png_b64()
        _, job = self._json(
            "POST",
            "/api/jobs",
            {
                "filename": "adjust_char.png",
                "pause_before_animation": False,
                "image_base64": b64,
                "preset": "agile_full",
                "gain": 1.0,
            },
        )
        self.assertEqual(job["status"], "needs_review")
        self.assertEqual(job["split_run_count"], 1)
        self.assertEqual(job["rig_build_count"], 1)

        # Layer A motion adjustment -> skips split_parts, updates gain/motion, keeps 4 hard gates PASS
        st_adj, adj_job = self._json(
            "POST",
            f"/api/jobs/{job['job_id']}/adjust",
            {"instruction": "整体呼吸更柔和"},
        )
        self.assertEqual(st_adj, 200)
        self.assertEqual(adj_job["status"], "needs_review")
        self.assertEqual(adj_job["split_run_count"], 1, "motion rerun must skip split_parts")
        self.assertEqual(adj_job["rig_build_count"], 2)
        self.assertLess(adj_job["gain"], 1.0)
        self.assertEqual(adj_job["health_report"]["status"], "pass")
        self.assertEqual(len(adj_job["adjustment_log"]), 1)
        self.assertEqual(adj_job["adjustment_log"][0]["rerun_layer"], "motion")
        self.assertIn("split_parts", adj_job["adjustment_log"][0]["skipped_stages"])

    def test_pause_before_animation_rig_edit_rebind_and_confirm(self) -> None:
        b64 = _make_portrait_png_b64()
        st, job = self._json(
            "POST",
            "/api/jobs",
            {
                "filename": "rig_edit_char.png",
                "image_base64": b64,
                "preset": "agile_full",
            },
        )
        self.assertEqual(st, 200)
        self.assertEqual(job["status"], "awaiting_rig_confirm")
        self.assertEqual(job["semantic_source"], "geometry")
        self.assertEqual(job["matting_source"], "transparent_input")
        self.assertFalse(job["matting_metadata"]["performed_matting"])
        self.assertGreaterEqual(job["bone_count"], 7)

        # Fetch rig spec for interactive preview/editing
        st_rs, rs = self._json("GET", f"/api/jobs/{job['job_id']}/rig-spec")
        self.assertEqual(st_rs, 200)
        self.assertTrue(rs["ok"])
        orig_bones = rs["bones"]
        orig_count = len(orig_bones)

        # Add a custom weapon bone under chest/hand and move head slightly, then rebind
        edited_bones = list(orig_bones) + [
            {
                "name": "custom_blade",
                "parent": "chest",
                "pos": [160.0, 120.0],
                "aim": [190.0, 95.0],
                "role": "weapon",
                "label": "自定义武器",
            }
        ]
        st_rb, rb_job = self._json(
            "POST",
            f"/api/jobs/{job['job_id']}/rebind",
            {"bones": edited_bones, "pose_type": "airborne"},
        )
        self.assertEqual(st_rb, 200)
        self.assertEqual(rb_job["status"], "awaiting_rig_confirm")
        _, after_rebind = self._json("GET", f"/api/jobs/{job['job_id']}/rig-spec")
        expected_names = {b["name"] for b in edited_bones} - {"toe_target_L", "toe_target_R"}
        self.assertEqual({b["name"] for b in after_rebind["bones"]}, expected_names)
        self.assertEqual(rb_job["bone_count"], len(expected_names))
        self.assertEqual(rb_job["pose_type"], "airborne")

        # Delete custom bone and rebind again
        trimmed_bones = [b for b in after_rebind["bones"] if b["name"] != "custom_blade"]
        st_rb2, rb_job2 = self._json(
            "POST",
            f"/api/jobs/{job['job_id']}/rebind",
            {"bones": trimmed_bones, "pose_type": "airborne"},
        )
        self.assertEqual(st_rb2, 200)
        self.assertEqual(rb_job2["bone_count"], len(expected_names) - 1)

        # Confirm binding -> generates animation & passes all gates
        st_cf, ready_job = self._json("POST", f"/api/jobs/{job['job_id']}/confirm-rig", {})
        self.assertEqual(st_cf, 200)
        self.assertEqual(ready_job["status"], "needs_review")
        self.assertEqual(ready_job["health_report"]["status"], "pass")
        self.assertTrue(ready_job["health_report"]["structuralChecks"]["weapon_rigidity"]["passed"])


        # Legacy diagnostic gates cannot approve a complete-module production package.
        st_exp, exp = self._json("POST", f"/api/jobs/{job['job_id']}/export", {"kind": "spine_package"})
        self.assertEqual(st_exp, 400)
        self.assertIn("完整模块", exp["message"])

    def test_semantic_hints_dual_limb_and_opaque_bg_warning(self) -> None:
        # 1. Vision keypoints (`semantic_hints`) with dual legs & custom accessory
        b64 = _make_portrait_png_b64()
        partial_status, partial = self._json("POST", "/api/jobs", {
            "image_base64": b64, "semantic_hints": {"pose_type": "airborne", "joints": {"head": [0.5, 0.2]}},
        })
        self.assertEqual(partial_status, 200)
        self.assertEqual(partial["status"], "failed")
        self.assertEqual(partial["semantic_source"], "provided_hints")
        self.assertEqual(partial["stage_error"]["errorCode"], "VISION_PROVIDED_HINTS_INVALID")
        hints = {
            "coord_mode": "normalized",
            "pose_type": "grounded",
            "joints": {
                "head_top": [0.50, 0.10],
                "neck": [0.50, 0.25],
                "chest": [0.50, 0.36],
                "hips": [0.50, 0.56],
                "shoulder_R": [0.38, 0.34],
                "elbow_R": [0.32, 0.46],
                "wrist_R": [0.30, 0.56],
                "shoulder_L": [0.62, 0.34],
                "elbow_L": [0.68, 0.46],
                "wrist_L": [0.70, 0.56],
                "hip_R": [0.44, 0.58],
                "knee_R": [0.43, 0.74],
                "ankle_R": [0.43, 0.90],
                "hip_L": [0.56, 0.58],
                "knee_L": [0.57, 0.74],
                "ankle_L": [0.57, 0.90],
            },
        }
        st, job = self._json(
            "POST",
            "/api/jobs",
            {
                "filename": "vision_hinted.png",
                "pause_before_animation": False,
                "image_base64": b64,
                "preset": "agile_full",
                "semantic_hints": hints,
            },
        )
        self.assertEqual(st, 200)
        self.assertEqual(job["status"], "needs_review")
        self.assertEqual(job["health_report"]["status"], "pass")
        self.assertEqual(job["health_report"]["hardGates"]["anchor_stability"]["measured"]["maxDriftPx"], 0.0)
        self.assertEqual(job["warnings"], [])
        self.assertEqual(job["semantic_source"], "provided_hints")

        # 2. Opaque-background image triggers warning (and can be rejected when reject_opaque_bg=True)
        opaque = Image.new("RGB", (240, 320), (40, 50, 70))
        buf = io.BytesIO()
        opaque.save(buf, format="PNG")
        opaque_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        st_op, op_job = self._json(
            "POST",
            "/api/jobs",
            {
                "filename": "opaque_bg.png",
                "pause_before_animation": False,
                "image_base64": opaque_b64,
                "reject_opaque_bg": False,
            },
        )
        self.assertEqual(st_op, 200)
        self.assertEqual(op_job["status"], "failed")
        self.assertEqual(op_job["matting_source"], "unavailable")
        self.assertEqual(op_job["stage_error"]["errorCode"], "MIVO_MATTING_REQUIRED")
        self.assertEqual(op_job["split_run_count"], 0)
        self.assertTrue(len(op_job["warnings"]) >= 1)
        self.assertIn("不透明背景", op_job["warnings"][0])



    def test_refine_mask_wrong_state_returns_409(self) -> None:
        b64 = _make_portrait_png_b64()
        status, job = self._json(
            "POST",
            "/api/jobs",
            {
                "filename": "hero.png",
                "pause_before_animation": False,
                "image_base64": b64,
                "preset": "agile_full",
            },
        )
        self.assertEqual(status, 200)
        self.assertNotIn(job["status"], ("awaiting_analysis", "awaiting_rig_confirm"))
        st, body = self._json("POST", f"/api/jobs/{job['job_id']}/refine-mask", {"target_id": "body"})
        self.assertEqual(st, 409)
        self.assertFalse(body.get("ok", True))


class TestRefineMaskApi(unittest.TestCase):
    def setUp(self) -> None:
        import copy
        from tests.pipeline.test_complete_modules import fixture

        self._tmpdir = tempfile.TemporaryDirectory()
        self.work_dir = Path(self._tmpdir.name) / "studio-work"
        self.image = Path(self._tmpdir.name) / "source.png"
        self.plan = fixture(self.image)
        self.plan_holder = {"plan": self.plan, "calls": []}

        def runner(ctx):
            self.plan_holder["calls"].append(dict(ctx))
            out = Path(ctx["out_dir"])
            out.mkdir(parents=True, exist_ok=True)
            if self.plan_holder.get("hold"):
                started = self.plan_holder.get("started")
                go = self.plan_holder.get("go")
                if started is not None:
                    started.set()
                if go is not None:
                    go.wait(timeout=5)
                self.plan_holder["seen_override"] = json.loads(Path(ctx["override_path"]).read_text(encoding="utf-8"))
            if self.plan_holder.get("boom"):
                raise RuntimeError("runner boom")
            if self.plan_holder.get("fail"):
                return {
                    "ok": False,
                    "errors": [{"id": ctx["target_id"], "code": "neg_prompt_failed", "message": "empty mask"}],
                    "message": "neg prompt failed",
                }
            analysis = copy.deepcopy(self.plan_holder["plan"])
            (out / "analysis.json").write_text(json.dumps(analysis, ensure_ascii=False), encoding="utf-8")
            (out / "segment-report.json").write_text(json.dumps({"ok": True, "errors": []}), encoding="utf-8")
            Image.new("RGB", (48, 48), (110, 110, 110)).save(out / "ownership-color-overlay.jpg", quality=90)
            Image.new("RGB", (48, 48), (90, 90, 90)).save(out / "module-crops-sheet.jpg", quality=90)
            return {"ok": True, "modules": [{"id": m["id"]} for m in analysis.get("modules", [])]}

        mgr = StudioJobManager(self.work_dir, audit_samples=6, stress_samples=4, refine_runner=runner)
        self.server = create_studio_server(host="127.0.0.1", port=0, work_dir=self.work_dir, manager=mgr)
        self.port = self.server.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}"
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self._thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self._thread.join(timeout=3)
        self._tmpdir.cleanup()

    def _json(self, method: str, path: str, payload=None):
        url = self.base_url + path
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"Content-Type": "application/json"} if data is not None else {}
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as err:
            return err.code, json.loads(err.read().decode("utf-8"))

    def _prepare(self):
        b64 = base64.b64encode(self.image.read_bytes()).decode("ascii")
        status, job = self._json("POST", "/api/jobs", {
            "filename": "source.png",
            "preset": "bust_half",
            "image_base64": b64,
        })
        self.assertEqual(status, 200, job)
        self.assertEqual(job["status"], "awaiting_analysis")
        req = job["analysis_request"]
        self.plan["job_id"] = job["job_id"]
        self.plan["input_sha256"] = req["input_sha256"]
        self.plan["image_size"] = req["image_size"]
        self.plan_holder["plan"] = self.plan
        return job

    def test_missing_target_and_unknown_field_and_missing_spec_400(self) -> None:
        job = self._prepare()
        st, body = self._json("POST", f"/api/jobs/{job['job_id']}/refine-mask", {"pos": [[1, 1]]})
        self.assertEqual(st, 400)
        st, body = self._json("POST", f"/api/jobs/{job['job_id']}/refine-mask", {
            "target_id": "body", "foo": 1,
        })
        self.assertEqual(st, 400)
        self.assertIn("未知字段", body.get("message", ""))
        st, body = self._json("POST", f"/api/jobs/{job['job_id']}/refine-mask", {"target_id": "body"})
        self.assertEqual(st, 400)
        self.assertIn("spec", body.get("message", ""))

    def test_refine_then_from_refined_apply_analysis(self) -> None:
        job = self._prepare()
        st, rec = self._json("POST", f"/api/jobs/{job['job_id']}/refine-mask", {
            "target_id": "body",
            "spec": self.plan,
            "pos": [[20, 40]],
            "neg": [[10, 10]],
        })
        self.assertEqual(st, 200, rec)
        self.assertTrue(rec["ok"])
        self.assertTrue(rec["analysis_ready"])
        self.assertTrue(rec["overlay_url"].endswith("refine_work/ownership-color-overlay.jpg"))
        st, job2 = self._json("GET", f"/api/jobs/{job['job_id']}")
        self.assertEqual(job2["status"], "awaiting_analysis")
        st, bound = self._json("POST", f"/api/jobs/{job['job_id']}/apply-analysis", {"from_refined": True})
        self.assertEqual(st, 200, bound)
        self.assertEqual(bound["status"], "awaiting_rig_confirm", bound.get("stage_error"))
        self.assertTrue(bound.get("rig_mask_overlay_url"))
        self.assertIn("rig/review/rig_mask_overlay.png", bound["rig_mask_overlay_url"])

    def test_neg_prompt_failed_does_not_change_status(self) -> None:
        job = self._prepare()
        self.plan_holder["fail"] = True
        st, rec = self._json("POST", f"/api/jobs/{job['job_id']}/refine-mask", {
            "target_id": "body",
            "spec": self.plan,
            "neg": [[8, 8]],
        })
        self.assertEqual(st, 422, rec)
        self.assertFalse(rec["ok"])
        codes = {e.get("code") for e in rec.get("errors") or []}
        self.assertIn("neg_prompt_failed", codes)
        st, job2 = self._json("GET", f"/api/jobs/{job['job_id']}")
        self.assertEqual(job2["status"], "awaiting_analysis")

    def test_from_refined_without_analysis_rejected(self) -> None:
        job = self._prepare()
        st, body = self._json("POST", f"/api/jobs/{job['job_id']}/apply-analysis", {"from_refined": True})
        self.assertEqual(st, 409)
        self.assertIn("refine_work", body.get("message", ""))
        st, job2 = self._json("GET", f"/api/jobs/{job['job_id']}")
        self.assertEqual(job2["status"], "awaiting_analysis")


    def test_from_refined_rejects_failed_marker_or_report(self) -> None:
        job = self._prepare()
        st, rec = self._json("POST", f"/api/jobs/{job['job_id']}/refine-mask", {
            "target_id": "body",
            "spec": self.plan,
        })
        self.assertEqual(st, 200, rec)
        refine_dir = self.work_dir / job["job_id"] / "refine_work"
        self.assertTrue((refine_dir / "analysis.json").is_file())
        (refine_dir / "SEGMENT_FAILED").write_text("failed", encoding="utf-8")
        st, body = self._json("POST", f"/api/jobs/{job['job_id']}/apply-analysis", {"from_refined": True})
        self.assertEqual(st, 409, body)
        (refine_dir / "SEGMENT_FAILED").unlink()
        (refine_dir / "segment-report.json").write_text(json.dumps({"ok": False, "errors": [{"code": "stale"}]}), encoding="utf-8")
        st, body = self._json("POST", f"/api/jobs/{job['job_id']}/apply-analysis", {"from_refined": True})
        self.assertEqual(st, 409, body)
        st, job2 = self._json("GET", f"/api/jobs/{job['job_id']}")
        self.assertEqual(job2["status"], "awaiting_analysis")

    def test_invalid_refine_params_do_not_call_runner(self) -> None:
        job = self._prepare()
        self.plan_holder["calls"] = []
        cases = [
            {"target_id": "../etc/passwd", "spec": self.plan},
            {"target_id": "body", "spec": self.plan, "pos": [["bad", None]]},
            {"target_id": "body", "spec": self.plan, "box": [10, 10, 10, 20]},
        ]
        for payload in cases:
            st, body = self._json("POST", f"/api/jobs/{job['job_id']}/refine-mask", payload)
            self.assertEqual(st, 400, body)
        self.assertEqual(self.plan_holder["calls"], [])


    def test_concurrent_refine_does_not_overwrite_override(self) -> None:
        job = self._prepare()
        self.plan_holder["hold"] = True
        self.plan_holder["started"] = threading.Event()
        self.plan_holder["go"] = threading.Event()
        first_pos = [[12.0, 18.0]]
        second_pos = [[30.0, 40.0]]
        results = {}

        def run_first():
            results["first"] = self._json("POST", f"/api/jobs/{job['job_id']}/refine-mask", {
                "target_id": "body",
                "spec": self.plan,
                "pos": first_pos,
            })

        t1 = threading.Thread(target=run_first)
        t1.start()
        self.assertTrue(self.plan_holder["started"].wait(timeout=5))
        st2, body2 = self._json("POST", f"/api/jobs/{job['job_id']}/refine-mask", {
            "target_id": "body",
            "spec": self.plan,
            "pos": second_pos,
        })
        self.assertEqual(st2, 409, body2)
        self.assertEqual(body2.get("errorCode"), "STUDIO_BUSY")
        self.plan_holder["go"].set()
        t1.join(timeout=30)
        self.assertEqual(results["first"][0], 200, results["first"])
        seen = self.plan_holder.get("seen_override") or {}
        self.assertEqual(seen.get("pos"), first_pos)

    def test_runner_exception_returns_422_and_blocks_stale_analysis(self) -> None:
        job = self._prepare()
        refine_dir = self.work_dir / job["job_id"] / "refine_work"
        refine_dir.mkdir(parents=True, exist_ok=True)
        (refine_dir / "analysis.json").write_text(json.dumps({"ok": True, "marker": "stale"}), encoding="utf-8")
        self.plan_holder["boom"] = True
        st, rec = self._json("POST", f"/api/jobs/{job['job_id']}/refine-mask", {
            "target_id": "body",
            "spec": self.plan,
        })
        self.assertEqual(st, 422, rec)
        self.assertFalse(rec.get("ok"))
        self.assertFalse(rec.get("analysis_ready"))
        codes = {e.get("code") for e in rec.get("errors") or []}
        self.assertIn("runner_exception", codes)
        self.assertTrue((refine_dir / "SEGMENT_FAILED").is_file())
        st, body = self._json("POST", f"/api/jobs/{job['job_id']}/apply-analysis", {"from_refined": True})
        self.assertEqual(st, 409, body)
        (refine_dir / "SEGMENT_FAILED").unlink()
        if (refine_dir / "segment-report.json").exists():
            (refine_dir / "segment-report.json").unlink()
        (refine_dir / "analysis.json").write_text(json.dumps({"ok": True, "marker": "orphan"}), encoding="utf-8")
        st, body = self._json("POST", f"/api/jobs/{job['job_id']}/apply-analysis", {"from_refined": True})
        self.assertEqual(st, 409, body)
        self.assertIn("segment-report", body.get("message", ""))

    def test_index_html_refine_contract(self) -> None:
        html = (Path(__file__).resolve().parents[2] / "src" / "studio" / "web" / "index.html").read_text(encoding="utf-8")
        self.assertIn("refine-mask", html)
        self.assertIn("from_refined", html)
        self.assertIn("rig_mask_overlay_url", html)
        self.assertIn("refineMaskCanvas", html)
        self.assertIn("btnRefineRun", html)
        self.assertIn("btnRefineApply", html)


if __name__ == "__main__":
    unittest.main()
