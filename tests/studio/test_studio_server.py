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
        mgr = StudioJobManager(self.work_dir, audit_samples=8, stress_samples=4)
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

    def test_full_generate_ready_and_over_2400px_resize_notice(self) -> None:
        # 360x2560 tall portrait exceeds 2400px long edge -> triggers LANCZOS auto-downscale notice
        b64_large = _make_portrait_png_b64(width=360, height=2560)
        status, job = self._json(
            "POST",
            "/api/jobs",
            {
                "filename": "hero_oversize.png",
                "image_base64": b64_large,
                "preset": "agile_full",
                "gain": 1.0,
                "pause_after_split": False,
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(job["status"], "ready")
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
        self.assertEqual(job2["status"], "ready")
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
        self.assertEqual(resumed["status"], "ready")
        self.assertEqual(resumed["health_report"]["status"], "pass")
        self.assertEqual(resumed["split_run_count"], 1)

    def test_fail_stop_on_mivo_error_retry_and_discard(self) -> None:
        b64 = _make_portrait_png_b64()
        status, failed = self._json(
            "POST",
            "/api/jobs",
            {
                "filename": "fail_char.png",
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
        self.assertEqual(retried["status"], "ready")
        self.assertEqual(retried["health_report"]["status"], "pass")
        self.assertFalse(stage_err_path.exists())

        # Second failed job -> verify /discard cleans error state
        _, failed2 = self._json(
            "POST",
            "/api/jobs",
            {
                "filename": "discard_char.png",
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
        self.assertEqual(results["first"][1]["status"], "ready")

    def test_adjust_minimal_rerun_and_export_spine_package(self) -> None:
        b64 = _make_portrait_png_b64()
        _, job = self._json(
            "POST",
            "/api/jobs",
            {
                "filename": "adjust_char.png",
                "image_base64": b64,
                "preset": "agile_full",
                "gain": 1.0,
            },
        )
        self.assertEqual(job["status"], "ready")
        self.assertEqual(job["split_run_count"], 1)
        self.assertEqual(job["rig_build_count"], 1)

        # Layer A motion adjustment -> skips split_parts, updates gain/motion, keeps 4 hard gates PASS
        st_adj, adj_job = self._json(
            "POST",
            f"/api/jobs/{job['job_id']}/adjust",
            {"instruction": "整体呼吸更柔和"},
        )
        self.assertEqual(st_adj, 200)
        self.assertEqual(adj_job["status"], "ready")
        self.assertEqual(adj_job["split_run_count"], 1, "motion rerun must skip split_parts")
        self.assertEqual(adj_job["rig_build_count"], 2)
        self.assertLess(adj_job["gain"], 1.0)
        self.assertEqual(adj_job["health_report"]["status"], "pass")
        self.assertEqual(len(adj_job["adjustment_log"]), 1)
        self.assertEqual(adj_job["adjustment_log"][0]["rerun_layer"], "motion")
        self.assertIn("split_parts", adj_job["adjustment_log"][0]["skipped_stages"])

        # Export default spine_package -> downloadable .zip URL
        st_exp, exp = self._json(
            "POST",
            f"/api/jobs/{job['job_id']}/export",
            {"kind": "spine_package"},
        )
        self.assertEqual(st_exp, 200)
        self.assertEqual(exp["kind"], "spine_package")
        self.assertEqual(len(exp["files"]), 1)
        zip_url = exp["files"][0]["url"]
        self.assertTrue(zip_url.endswith(".zip"))

        z_status, z_bytes, _ = self._request("GET", zip_url)
        self.assertEqual(z_status, 200)
        with zipfile.ZipFile(io.BytesIO(z_bytes), "r") as zf:
            names = zf.namelist()
            self.assertTrue(any(n.endswith(".spine.json") for n in names))
            self.assertIn("rig-build-report.json", names)
            self.assertIn("health-report.json", names)
            self.assertTrue(any(n.startswith("images/") and n.endswith(".png") for n in names))


if __name__ == "__main__":
    unittest.main()
