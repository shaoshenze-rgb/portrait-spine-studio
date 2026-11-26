import base64
import hashlib
import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

from src.pipeline.canvas_layer_parts import materialize_canvas_layer_parts
from src.pipeline.local_body_split import BodyLayer, split_local_body
from src.render.renderer import find_ffmpeg
from src.studio.batch_manager import BatchItemLockedError, BatchManager
from src.studio.server import create_studio_server


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_portrait_png(path: Path, color=(200, 140, 80, 255), size=(120, 160)) -> None:
    w, h = size
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[16:64, 34:86] = (230, 190, 150, 255)    # head
    rgba[58:146, 28:92] = color                   # torso
    Image.fromarray(rgba).save(path)


def _build_local_parts_for_job(job_dir: Path, out_dir: Path):
    matted = job_dir / "matted.png"
    src = np.asarray(Image.open(matted).convert("RGBA"))
    h, w = src.shape[:2]
    m_head = np.zeros((h, w), dtype=bool)
    m_head[:62, :] = src[:62, :, 3] > 10
    m_torso = np.zeros((h, w), dtype=bool)
    m_torso[56:, :] = src[56:, :, 3] > 10
    split = split_local_body(
        src,
        [BodyLayer("torso", m_torso, 0, False), BodyLayer("head", m_head, 1, False)],
        padding_px=14,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    p_torso = out_dir / "torso.png"
    p_head = out_dir / "head.png"
    Image.fromarray(split.rgba["torso"]).save(p_torso)
    Image.fromarray(split.rgba["head"]).save(p_head)
    src_sha = _sha(matted)

    def prov(p: Path):
        return {
            "operation": "local-direct-split",
            "source_asset_sha256": src_sha,
            "result_asset_sha256": _sha(p),
            "transform": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            "local_derivation": {"method": "edge_refined_seed_mask", "parent_aligned_sha256": src_sha},
        }

    manifest = {
        "version": 1,
        "material_mode": "canvas_layers",
        "source_image": str(matted),
        "source_sha256": src_sha,
        "image_size": [w, h],
        "pieces": [
            {
                "id": "torso", "module_id": "torso", "parent": None, "kind": "rigid",
                "pivot": [w // 2, int(h * 0.72)], "draw_order": 0, "label": "躯干",
                "image": str(p_torso), "sha256": _sha(p_torso), "provenance": prov(p_torso),
            },
            {
                "id": "head", "module_id": "head", "parent": "torso", "kind": "rigid",
                "pivot": [w // 2, 58], "draw_order": 1, "label": "头部",
                "image": str(p_head), "sha256": _sha(p_head), "provenance": prov(p_head),
            },
        ],
    }
    parts_dir = out_dir / "parts"
    materialize_canvas_layer_parts(manifest, parts_dir, evidence_root=job_dir.parent)
    plan = {
        "pose_type": "airborne",
        "anatomy": "humanoid_bust",
        "uncertainties": [],
        "modules": [
            {
                "id": "torso", "role": "torso", "motion_type": "rigid", "parent": None,
                "pivot": [0.5, int(h * 0.72) / h], "motion": "breath", "confidence": 0.95,
            },
            {
                "id": "head", "role": "head", "motion_type": "rigid", "parent": "torso",
                "pivot": [0.5, 58.0 / h], "motion": "sway", "confidence": 0.95,
            },
        ],
        "connections": [
            {"id": "neck", "type": "joint", "a": "torso", "b": "head", "point": [0.5, 58.0 / h]},
        ],
    }
    plan_path = out_dir / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    return parts_dir, plan_path


class TestBatchWorkflow(unittest.TestCase):
    def _http(self, base_url: str, path: str, payload=None):
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            base_url + path,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode("utf-8"))

    def test_folder_rescan_preserves_locked_items_and_appends_new_ones(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src_dir = root / "portraits"
            src_dir.mkdir()
            _make_portrait_png(src_dir / "01_alice.png", (190, 90, 90, 255))
            _make_portrait_png(src_dir / "02_bob.png", (80, 150, 210, 255))

            bm = BatchManager(root / "work")
            b1 = bm.create_or_rescan_batch(src_dir, name="test-batch")
            bid = b1["batch_id"]
            self.assertEqual(len(b1["items"]), 2)
            self.assertEqual(b1["export_profile"]["size_preset"], "4k_portrait")
            self.assertEqual(b1["export_profile"]["formats"], ["transparent_web"])

            item0 = b1["items"][0]
            bm.lock_item(bid, item0["item_id"], revision=1, rig_version=1)

            with self.assertRaises(BatchItemLockedError):
                bm.update_item(bid, item0["item_id"], {"status": "pending"})
            with self.assertRaises(BatchItemLockedError):
                bm.reject_item(bid, item0["item_id"], "should fail")

            _make_portrait_png(src_dir / "01_alice.png", (10, 220, 10, 255))
            _make_portrait_png(src_dir / "03_carol.png", (200, 180, 60, 255))
            b2 = bm.create_or_rescan_batch(src_dir, batch_id=bid)
            self.assertEqual(b2["rescan_summary"]["added"], 1)
            self.assertEqual(b2["rescan_summary"]["skipped_locked"], 1)
            self.assertEqual(len(b2["items"]), 3)
            self.assertTrue(b2["items"][0]["locked"])
            self.assertEqual(b2["items"][0]["status"], "approved_locked")
            self.assertEqual(b2["items"][0]["source_sha256"], item0["source_sha256"])

    def test_full_http_batch_lock_reject_restore_and_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src_dir = root / "folder"
            src_dir.mkdir()
            _make_portrait_png(src_dir / "alpha.png", (190, 95, 70, 255))
            _make_portrait_png(src_dir / "beta.png", (70, 130, 210, 255))
            work_dir = root / "studio-work"

            srv = create_studio_server(host="127.0.0.1", port=0, work_dir=work_dir)
            port = srv.server_address[1]
            base = f"http://127.0.0.1:{port}"
            th = threading.Thread(target=srv.serve_forever, daemon=True)
            th.start()
            try:
                from scripts import batch_portrait
                rc = batch_portrait.main([
                    "--work-dir", str(work_dir),
                    "--base-url", base,
                    "init",
                    "--input-dir", str(src_dir),
                    "--name", "heroes-batch",
                    "--workers", "2",
                ])
                self.assertEqual(rc, 0)

                st, batches_resp = self._http(base, "/api/batches")
                self.assertEqual(st, 200)
                self.assertEqual(len(batches_resp["batches"]), 1)
                bid = batches_resp["batches"][0]["batch_id"]

                st, bv = self._http(base, f"/api/batches/{bid}")
                self.assertEqual(st, 200)
                self.assertEqual(bv["counts"]["total"], 2)
                self.assertTrue(bv["matte_overview_url"])
                item1, item2 = bv["items"][0], bv["items"][1]
                self.assertTrue(item1["job_id"] and item2["job_id"])

                for it in (item1, item2):
                    jid = it["job_id"]
                    parts_dir, plan_path = _build_local_parts_for_job(work_dir / jid, work_dir / f"split_{jid}")
                    st_a, applied = self._http(
                        base,
                        f"/api/jobs/{jid}/apply-canvas-layers",
                        {"parts_dir": str(parts_dir), "plan": str(plan_path), "cleanup": True, "wait_if_busy": True},
                    )
                    self.assertEqual(st_a, 200, applied)
                    self.assertEqual(applied["status"], "canvas_preview")

                st_l, bv_after_lock = self._http(base, f"/api/batches/{bid}/items/{item1['item_id']}/lock", {"reviewer": "tester"})
                self.assertEqual(st_l, 200, bv_after_lock)
                self.assertEqual(bv_after_lock["counts"]["locked"], 1)

                st_tw, tw_err = self._http(base, f"/api/jobs/{item1['job_id']}/layer-tweaks", {"global_motion_scale": 1.2})
                self.assertEqual(st_tw, 409)
                self.assertEqual(tw_err["errorCode"], "JOB_LOCKED")

                st_ds, ds_err = self._http(base, f"/api/jobs/{item1['job_id']}/discard", {})
                self.assertEqual(st_ds, 409)
                self.assertEqual(ds_err["errorCode"], "JOB_LOCKED")

                st_rj_locked, rj_err = self._http(
                    base, f"/api/batches/{bid}/items/{item1['item_id']}/reject", {"fix_note": "try reject locked"}
                )
                self.assertEqual(st_rj_locked, 409)
                self.assertEqual(rj_err["errorCode"], "JOB_LOCKED")

                st_rj, _ = self._http(
                    base, f"/api/batches/{bid}/items/{item2['item_id']}/reject", {"fix_note": "右脚底有灰边"}
                )
                self.assertEqual(st_rj, 200)

                st_fl, fix_list = self._http(base, f"/api/batches/{bid}/fix-list")
                self.assertEqual(st_fl, 200)
                self.assertEqual(fix_list["count"], 1)
                self.assertEqual(fix_list["items"][0]["item_id"], item2["item_id"])
                self.assertEqual(fix_list["items"][0]["fix_note"], "右脚底有灰边")

                out_export = root / "final_exports"
                formats = ["spine_package"]
                if find_ffmpeg():
                    formats.insert(0, "transparent_web")
                st_ex, ex_res = self._http(
                    base,
                    f"/api/batches/{bid}/export",
                    {
                        "async": False,
                        "max_frames": 2,
                        "output_dir": str(out_export),
                        "export_profile": {"size_preset": "1080p_portrait", "formats": formats, "fps": 30},
                    },
                )
                self.assertEqual(st_ex, 200, ex_res)
                self.assertEqual(ex_res["exported_count"], 1)
                self.assertEqual(ex_res["error_count"], 0)
                self.assertTrue((out_export / "batch-export-manifest.json").is_file())
                self.assertTrue((out_export / item1["character"] / f"{item1['character']}-spine42.zip").is_file())
                self.assertFalse((out_export / item2["character"]).exists())
                if find_ffmpeg():
                    self.assertTrue((out_export / item1["character"] / f"{item1['character']}-alpha-vp9.webm").is_file())
                    self.assertTrue((out_export / item1["character"] / f"{item1['character']}-poster.webp").is_file())

                st_un, _ = self._http(base, f"/api/batches/{bid}/items/{item1['item_id']}/unlock", {})
                self.assertEqual(st_un, 200)
                st_tw2, tw_ok = self._http(base, f"/api/jobs/{item1['job_id']}/layer-tweaks", {"global_motion_scale": 1.1})
                self.assertEqual(st_tw2, 200, tw_ok)

                self._http(base, f"/api/batches/{bid}/items/{item1['item_id']}/lock", {})
            finally:
                srv.shutdown()
                srv.server_close()

            srv2 = create_studio_server(host="127.0.0.1", port=0, work_dir=work_dir)
            base2 = f"http://127.0.0.1:{srv2.server_address[1]}"
            th2 = threading.Thread(target=srv2.serve_forever, daemon=True)
            th2.start()
            try:
                st_j, restored_job = self._http(base2, f"/api/jobs/{item1['job_id']}")
                self.assertEqual(st_j, 200)
                self.assertTrue(restored_job["locked"])
                st_tw3, _ = self._http(base2, f"/api/jobs/{item1['job_id']}/layer-tweaks", {"global_motion_scale": 0.9})
                self.assertEqual(st_tw3, 409)
            finally:
                srv2.shutdown()
                srv2.server_close()

    def test_concurrent_prepare_with_wait_if_busy_queues_cleanly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p1 = root / "a.png"
            _make_portrait_png(p1)
            b64 = base64.b64encode(p1.read_bytes()).decode("ascii")

            srv = create_studio_server(host="127.0.0.1", port=0, work_dir=root / "work")
            base = f"http://127.0.0.1:{srv.server_address[1]}"
            th = threading.Thread(target=srv.serve_forever, daemon=True)
            th.start()
            results = []
            try:
                def call_prepare(char):
                    st, body = self._http(
                        base,
                        "/api/jobs",
                        {
                            "filename": f"{char}.png",
                            "image_base64": b64,
                            "character": char,
                            "max_long_edge": 4096,
                            "wait_if_busy": True,
                            "simulate_delay_ms": 150,
                        },
                    )
                    results.append((st, body.get("max_long_edge")))

                t1 = threading.Thread(target=call_prepare, args=("c1",))
                t2 = threading.Thread(target=call_prepare, args=("c2",))
                t1.start(); t2.start()
                t1.join(); t2.join()
            finally:
                srv.shutdown()
                srv.server_close()

            self.assertEqual(len(results), 2)
            self.assertEqual([r[0] for r in results], [200, 200])
            self.assertEqual([r[1] for r in results], [4096, 4096])


if __name__ == "__main__":
    unittest.main()
