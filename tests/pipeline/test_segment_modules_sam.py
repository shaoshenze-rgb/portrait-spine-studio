"""Mock-SAM end-to-end gates for scripts/segment_modules_sam.py."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]

try:
    import cv2  # noqa: F401
    import scipy  # noqa: F401
    import skimage  # noqa: F401
    HAS_SEG_DEPS = True
except ImportError:
    HAS_SEG_DEPS = False

from scripts.segment_modules_sam import (
    _bind_cache_dirs,
    isolate_tile,
    masked_rgba,
    quarantine_success_analysis,
)


def _write_spec(path: Path, modules):
    path.write_text(json.dumps({"version": 2, "modules": modules}, ensure_ascii=False), encoding="utf-8")
    return path


def _rgba(h, w, paint):
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    for y1, x1, y2, x2, rgb in paint:
        arr[y1:y2, x1:x2, :3] = rgb
        arr[y1:y2, x1:x2, 3] = 255
    return arr


class TestSegmentHelpersNoCv2(unittest.TestCase):
    def setUp(self):
        work = ROOT / ".studio-work"
        work.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=work, prefix="seg-helper-")
        self.addCleanup(self.tmp.cleanup)
        self.d = Path(self.tmp.name)

    def test_isolated_tile_hides_neighbor_pixels(self):
        src = _rgba(20, 40, [
            (0, 0, 20, 20, (220, 10, 10)),
            (0, 20, 20, 40, (10, 220, 10)),
        ])
        mask_a = np.zeros((20, 40), dtype=bool)
        mask_a[:, :20] = True
        isolated = masked_rgba(src, mask_a)
        leak = (isolated[..., 1] > 180) & (isolated[..., 3] > 0)
        self.assertFalse(bool(leak.any()))
        self.assertGreater(int((isolated[:, :20, 0] > 180).sum()), 100)
        _source, iso_tile = isolate_tile(src, mask_a)
        iso_arr = np.asarray(iso_tile.convert("RGBA"))
        # Gray backdrop is 145; neighbor green must not appear as opaque green.
        green = (iso_arr[..., 1] > 180) & (iso_arr[..., 0] < 80) & (iso_arr[..., 2] < 80)
        self.assertFalse(bool(green.any()))

    def test_bind_sets_yolo_config_dir_inside_project(self):
        old = os.environ.get("YOLO_CONFIG_DIR")
        os.environ.pop("YOLO_CONFIG_DIR", None)
        try:
            _bind_cache_dirs()
            value = Path(os.environ["YOLO_CONFIG_DIR"]).resolve()
            models = (ROOT / ".studio-work" / "models").resolve()
            self.assertEqual(value, models)
        finally:
            if old is None:
                os.environ.pop("YOLO_CONFIG_DIR", None)
            else:
                os.environ["YOLO_CONFIG_DIR"] = old

    def test_quarantine_keeps_bytes_and_removes_consumable_name(self):
        analysis = self.d / "analysis.json"
        payload = b'{"ok": true, "marker": "old-success"}'
        analysis.write_bytes(payload)
        moved = quarantine_success_analysis(self.d)
        self.assertIsNotNone(moved)
        self.assertFalse(analysis.exists())
        self.assertTrue(moved.is_file())
        self.assertEqual(moved.read_bytes(), payload)
        self.assertIn("prev-success", moved.name)


@unittest.skipUnless(HAS_SEG_DEPS, "cv2/scipy/skimage required; system python does not install them")
class TestSegmentMockSam(unittest.TestCase):
    def setUp(self):
        work = ROOT / ".studio-work"
        work.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=work, prefix="seg-sam-")
        self.addCleanup(self.tmp.cleanup)
        self.d = Path(self.tmp.name)
        from scripts.segment_modules_sam import DEFAULT_WEIGHT, segment
        self.segment = segment
        self.weight = DEFAULT_WEIGHT

    def _run(self, arr, modules, raw):
        img = self.d / "src.png"
        Image.fromarray(arr).save(img)
        spec = _write_spec(self.d / "spec.json", modules)
        out = self.d / "out"
        out.mkdir(exist_ok=True)
        return self.segment(img, spec, out, self.weight, raw_masks=raw), out

    def test_valid_multi_piece_writes_analysis(self):
        arr = _rgba(32, 64, [
            (4, 4, 28, 28, (200, 30, 30)),
            (4, 36, 28, 60, (30, 30, 200)),
        ])
        raw_a = np.zeros((32, 64), dtype=bool); raw_a[4:28, 4:28] = True
        raw_b = np.zeros((32, 64), dtype=bool); raw_b[4:28, 36:60] = True
        modules = [{
            "id": "staff", "draw_order": 1, "role": "weapon",
            "pieces": [
                {"id": "staff_upper", "draw_order": 1, "box": [4, 4, 28, 28], "pos": [[16, 16]], "max_cc": 4},
                {"id": "staff_lower", "draw_order": 2, "box": [36, 4, 60, 28], "pos": [[48, 16]], "max_cc": 4},
            ],
        }]
        summary, out = self._run(arr, modules, {"staff_upper": raw_a, "staff_lower": raw_b})
        self.assertTrue(summary["ok"], summary)
        self.assertEqual(summary["unassigned"], 0)
        self.assertEqual(summary["errors"], [])
        self.assertTrue((out / "analysis.json").is_file())
        analysis = json.loads((out / "analysis.json").read_text())
        ids = [p["id"] for p in analysis["modules"][0]["pieces"]]
        self.assertEqual(ids, ["staff_upper", "staff_lower"])
        self.assertNotEqual(
            analysis["modules"][0]["pieces"][0]["mask_sha256"],
            analysis["modules"][0]["pieces"][1]["mask_sha256"],
        )
        sheet = np.asarray(Image.open(out / "module-crops-sheet.jpg"))
        self.assertGreater(sheet.shape[0], sheet.shape[1] // 3)

    def test_empty_piece_fails_and_does_not_publish_analysis(self):
        arr = _rgba(32, 32, [(4, 4, 20, 20, (200, 30, 30))])
        raw_a = np.zeros((32, 32), dtype=bool); raw_a[4:20, 4:20] = True
        raw_b = np.zeros((32, 32), dtype=bool)
        modules = [{
            "id": "staff", "draw_order": 1,
            "pieces": [
                {"id": "staff_upper", "draw_order": 1, "box": [4, 4, 20, 20], "pos": [[10, 10]]},
                {"id": "staff_lower", "draw_order": 2, "box": [20, 20, 30, 30], "pos": [[24, 24]]},
            ],
        }]
        summary, out = self._run(arr, modules, {"staff_upper": raw_a, "staff_lower": raw_b})
        self.assertFalse(summary["ok"])
        codes = {(e["id"], e["code"]) for e in summary["errors"]}
        self.assertIn(("staff_lower", "empty_piece"), codes)
        self.assertFalse((out / "analysis.json").exists())
        self.assertTrue((out / "SEGMENT_FAILED").is_file())
        self.assertTrue((out / "segment-report.json").is_file())
        self.assertFalse(json.loads((out / "segment-report.json").read_text())["ok"])

    def test_cc_exceeded_after_nearest_fails(self):
        h, w = 20, 120
        paint = []
        raw_a = np.zeros((h, w), dtype=bool)
        for i in range(5):
            x = 4 + i * 22
            paint.append((4, x, 16, x + 12, (200, 20, 20)))
            raw_a[4:16, x:x + 12] = True
        arr = _rgba(h, w, paint)
        modules = [{
            "id": "blob", "draw_order": 1, "box": [0, 0, 120, 20], "pos": [[10, 10]], "max_cc": 4,
            "pieces": [{"id": "blob_p0", "draw_order": 1, "box": [0, 0, 120, 20], "pos": [[10, 10]], "max_cc": 4}],
        }]
        summary, out = self._run(arr, modules, {"blob_p0": raw_a})
        self.assertFalse(summary["ok"], summary)
        cc_err = [e for e in summary["errors"] if e.get("code") == "cc_exceeded"]
        self.assertTrue(cc_err, summary["errors"])
        self.assertGreater(cc_err[0]["components"], cc_err[0]["max_cc"])
        self.assertEqual(cc_err[0]["max_cc"], 4)
        self.assertFalse((out / "analysis.json").exists())

    def test_failure_cannot_consume_old_analysis(self):
        arr = _rgba(32, 32, [(4, 4, 20, 20, (200, 30, 30))])
        raw_a = np.zeros((32, 32), dtype=bool); raw_a[4:20, 4:20] = True
        modules = [{
            "id": "staff", "draw_order": 1,
            "pieces": [
                {"id": "staff_upper", "draw_order": 1, "box": [4, 4, 20, 20], "pos": [[10, 10]]},
                {"id": "staff_lower", "draw_order": 2, "box": [20, 20, 30, 30], "pos": [[24, 24]]},
            ],
        }]
        img = self.d / "src.png"
        Image.fromarray(arr).save(img)
        spec = _write_spec(self.d / "spec.json", modules)
        out = self.d / "out"
        out.mkdir()
        stale = {"ok": True, "marker": "do-not-consume"}
        (out / "analysis.json").write_text(json.dumps(stale), encoding="utf-8")
        summary = self.segment(img, spec, out, self.weight, raw_masks={"staff_upper": raw_a, "staff_lower": np.zeros((32, 32), dtype=bool)})
        self.assertFalse(summary["ok"])
        self.assertFalse((out / "analysis.json").exists())
        prev = list(out.glob("analysis.json.prev-success-*.json"))
        self.assertEqual(len(prev), 1)
        self.assertEqual(json.loads(prev[0].read_text())["marker"], "do-not-consume")
        self.assertIsNone(summary["analysis"])

    def test_exclude_guard_not_taken_back_by_nearest(self):
        arr = _rgba(24, 48, [
            (2, 2, 22, 22, (200, 30, 30)),
            (2, 26, 22, 46, (30, 200, 30)),
        ])
        raw_a = np.zeros((24, 48), dtype=bool); raw_a[2:22, 2:46] = True
        raw_b = np.zeros((24, 48), dtype=bool); raw_b[2:22, 26:46] = True
        modules = [
            {"id": "left", "draw_order": 1, "box": [2, 2, 22, 22], "pos": [[10, 10]],
             "exclude_boxes": [[26, 2, 46, 22]],
             "pieces": [{"id": "left_p0", "draw_order": 1, "box": [2, 2, 22, 22], "pos": [[10, 10]],
                          "exclude_boxes": [[26, 2, 46, 22]]}]},
            {"id": "right", "draw_order": 2, "box": [26, 2, 46, 22], "pos": [[36, 10]],
             "pieces": [{"id": "right_p0", "draw_order": 2, "box": [26, 2, 46, 22], "pos": [[36, 10]]}]},
        ]
        summary, out = self._run(arr, modules, {"left_p0": raw_a, "right_p0": raw_b})
        self.assertTrue(summary["ok"], summary)
        from PIL import Image as _Image
        left = np.asarray(_Image.open(out / "masks" / "left_p0.png")) > 127
        self.assertFalse(bool(left[2:22, 26:46].any()))
        right = np.asarray(_Image.open(out / "masks" / "right_p0.png")) > 127
        self.assertGreater(int(right[2:22, 26:46].sum()), 50)

    def test_publish_exception_does_not_leave_half_analysis(self):
        arr = _rgba(32, 64, [
            (4, 4, 28, 28, (200, 30, 30)),
            (4, 36, 28, 60, (30, 30, 200)),
        ])
        raw_a = np.zeros((32, 64), dtype=bool); raw_a[4:28, 4:28] = True
        raw_b = np.zeros((32, 64), dtype=bool); raw_b[4:28, 36:60] = True
        modules = [{
            "id": "staff", "draw_order": 1,
            "pieces": [
                {"id": "staff_upper", "draw_order": 1, "box": [4, 4, 28, 28], "pos": [[16, 16]]},
                {"id": "staff_lower", "draw_order": 2, "box": [36, 4, 60, 28], "pos": [[48, 16]]},
            ],
        }]
        img = self.d / "src.png"
        Image.fromarray(arr).save(img)
        spec = _write_spec(self.d / "spec.json", modules)
        out = self.d / "pub"
        out.mkdir()
        (out / "analysis.json").write_text(json.dumps({"ok": True, "marker": "pre-publish"}), encoding="utf-8")
        import scripts.segment_modules_sam as mod
        orig = mod.publish_success_artifacts
        def boom(out_dir, analysis, fragments, summary):
            Path(out_dir).joinpath("analysis.json").write_text('{"half": true}', encoding="utf-8")
            raise RuntimeError("publish-boom")
        mod.publish_success_artifacts = boom
        try:
            summary = self.segment(img, spec, out, self.weight, raw_masks={"staff_upper": raw_a, "staff_lower": raw_b})
        finally:
            mod.publish_success_artifacts = orig
        self.assertFalse(summary["ok"], summary)
        self.assertEqual(summary.get("error"), "RuntimeError")
        self.assertFalse((out / "analysis.json").exists())
        self.assertTrue((out / "SEGMENT_FAILED").is_file())
        report = json.loads((out / "segment-report.json").read_text())
        self.assertFalse(report["ok"])
        self.assertIsNone(report.get("analysis"))
        prev = list(out.glob("analysis.json.prev-*.json"))
        self.assertGreaterEqual(len(prev), 1)
        bodies = [p.read_text(encoding="utf-8") for p in prev]
        self.assertTrue(any("pre-publish" in b for b in bodies))
        self.assertTrue(any("half" in b for b in bodies))

    def test_success_after_failure_clears_failed_marker(self):
        arr = _rgba(32, 64, [
            (4, 4, 28, 28, (200, 30, 30)),
            (4, 36, 28, 60, (30, 30, 200)),
        ])
        raw_a = np.zeros((32, 64), dtype=bool); raw_a[4:28, 4:28] = True
        raw_b = np.zeros((32, 64), dtype=bool); raw_b[4:28, 36:60] = True
        good = [{
            "id": "staff", "draw_order": 1,
            "pieces": [
                {"id": "staff_upper", "draw_order": 1, "box": [4, 4, 28, 28], "pos": [[16, 16]]},
                {"id": "staff_lower", "draw_order": 2, "box": [36, 4, 60, 28], "pos": [[48, 16]]},
            ],
        }]
        empty_raw = {"staff_upper": raw_a, "staff_lower": np.zeros((32, 64), dtype=bool)}
        fail_summary, out = self._run(arr, good, empty_raw)
        self.assertFalse(fail_summary["ok"])
        self.assertTrue((out / "SEGMENT_FAILED").is_file())
        img = self.d / "src.png"
        spec = _write_spec(self.d / "spec.json", good)
        ok_summary = self.segment(img, spec, out, self.weight, raw_masks={"staff_upper": raw_a, "staff_lower": raw_b})
        self.assertTrue(ok_summary["ok"], ok_summary)
        self.assertTrue((out / "analysis.json").is_file())
        self.assertFalse((out / "SEGMENT_FAILED").exists())
        report = json.loads((out / "segment-report.json").read_text())
        self.assertTrue(report["ok"])


    def test_analysis_replace_failure_isolates_fragments(self):
        arr = _rgba(32, 64, [
            (4, 4, 28, 28, (200, 30, 30)),
            (4, 36, 28, 60, (30, 30, 200)),
        ])
        raw_a = np.zeros((32, 64), dtype=bool); raw_a[4:28, 4:28] = True
        raw_b = np.zeros((32, 64), dtype=bool); raw_b[4:28, 36:60] = True
        modules = [{
            "id": "staff", "draw_order": 1,
            "pieces": [
                {"id": "staff_upper", "draw_order": 1, "box": [4, 4, 28, 28], "pos": [[16, 16]]},
                {"id": "staff_lower", "draw_order": 2, "box": [36, 4, 60, 28], "pos": [[48, 16]]},
            ],
        }]
        img = self.d / "src.png"
        Image.fromarray(arr).save(img)
        spec = _write_spec(self.d / "spec.json", modules)
        out = self.d / "rep-analysis"
        out.mkdir()
        import scripts.segment_modules_sam as mod
        def boom(src, dest):
            if dest.name == "analysis.json":
                raise OSError("injected replace failure: analysis.json")
        mod._TEST_REPLACE_HOOK = boom
        try:
            summary = self.segment(img, spec, out, self.weight, raw_masks={"staff_upper": raw_a, "staff_lower": raw_b})
        finally:
            mod._TEST_REPLACE_HOOK = None
        self.assertFalse(summary["ok"], summary)
        self.assertFalse((out / "analysis.json").exists())
        self.assertFalse((out / "mask-fragments.json").exists())
        kept = list(out.glob("mask-fragments.json.prev-*.json"))
        self.assertTrue(kept, list(out.iterdir()))
        self.assertTrue(json.loads(kept[0].read_text()))
        self.assertTrue((out / "SEGMENT_FAILED").is_file())

    def test_report_replace_failure_isolates_fragments_and_analysis(self):
        arr = _rgba(32, 64, [
            (4, 4, 28, 28, (200, 30, 30)),
            (4, 36, 28, 60, (30, 30, 200)),
        ])
        raw_a = np.zeros((32, 64), dtype=bool); raw_a[4:28, 4:28] = True
        raw_b = np.zeros((32, 64), dtype=bool); raw_b[4:28, 36:60] = True
        modules = [{
            "id": "staff", "draw_order": 1,
            "pieces": [
                {"id": "staff_upper", "draw_order": 1, "box": [4, 4, 28, 28], "pos": [[16, 16]]},
                {"id": "staff_lower", "draw_order": 2, "box": [36, 4, 60, 28], "pos": [[48, 16]]},
            ],
        }]
        img = self.d / "src.png"
        Image.fromarray(arr).save(img)
        spec = _write_spec(self.d / "spec.json", modules)
        out = self.d / "rep-report"
        out.mkdir()
        import scripts.segment_modules_sam as mod
        def boom(src, dest):
            if dest.name == "segment-report.json":
                raise OSError("injected replace failure: segment-report.json")
        mod._TEST_REPLACE_HOOK = boom
        try:
            summary = self.segment(img, spec, out, self.weight, raw_masks={"staff_upper": raw_a, "staff_lower": raw_b})
        finally:
            mod._TEST_REPLACE_HOOK = None
        self.assertFalse(summary["ok"], summary)
        self.assertFalse((out / "mask-fragments.json").exists())
        self.assertFalse((out / "analysis.json").exists() and json.loads((out / "analysis.json").read_text()).get("ok") is True)
        self.assertTrue(list(out.glob("mask-fragments.json.prev-*.json")))
        self.assertTrue(list(out.glob("analysis.json.prev-*.json")))



class TestStaleSuccessBeforeValidation(unittest.TestCase):
    """These probes do not need cv2: they fail before watershed."""

    def setUp(self):
        work = ROOT / ".studio-work"
        work.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=work, prefix="seg-stale-")
        self.addCleanup(self.tmp.cleanup)
        self.d = Path(self.tmp.name)
        from scripts.segment_modules_sam import DEFAULT_WEIGHT, segment
        self.segment = segment
        self.weight = DEFAULT_WEIGHT

    def _plant(self, out):
        out.mkdir(parents=True, exist_ok=True)
        (out / "analysis.json").write_text(json.dumps({"ok": True, "marker": "old-success"}), encoding="utf-8")
        (out / "segment-report.json").write_text(json.dumps({"ok": True, "marker": "old-report"}), encoding="utf-8")
        (out / "mask-fragments.json").write_text(json.dumps({"old": True}), encoding="utf-8")
        return b'{"ok": true, "marker": "old-success"}'

    def _assert_isolated(self, out, summary):
        self.assertFalse(summary.get("ok"))
        self.assertIsNone(summary.get("analysis"))
        self.assertFalse((out / "analysis.json").exists())
        prev = list(out.glob("analysis.json.prev-success-*.json"))
        self.assertEqual(len(prev), 1)
        self.assertEqual(json.loads(prev[0].read_text())["marker"], "old-success")
        self.assertEqual(prev[0].read_bytes(), b'{"ok": true, "marker": "old-success"}')
        self.assertTrue((out / "SEGMENT_FAILED").is_file())
        report = json.loads((out / "segment-report.json").read_text())
        self.assertFalse(report["ok"])
        self.assertIsNone(report.get("analysis"))
        prev_r = list(out.glob("segment-report.json.prev-success-*.json"))
        self.assertEqual(len(prev_r), 1)
        self.assertEqual(json.loads(prev_r[0].read_text())["marker"], "old-report")

    def _tiny_png(self):
        img = self.d / "src.png"
        Image.fromarray(_rgba(16, 16, [(2, 2, 14, 14, (10, 20, 30))])).save(img)
        return img

    def _tiny_spec(self, path=None):
        spec = path or (self.d / "spec.json")
        _write_spec(spec, [{
            "id": "staff", "draw_order": 1,
            "pieces": [
                {"id": "staff_upper", "draw_order": 1, "box": [2, 2, 8, 8], "pos": [[4, 4]]},
                {"id": "staff_lower", "draw_order": 2, "box": [8, 8, 14, 14], "pos": [[10, 10]]},
            ],
        }])
        return spec

    def test_bad_json_spec_isolates_old_success(self):
        out = self.d / "bad-spec"
        self._plant(out)
        img = self._tiny_png()
        spec = self.d / "broken.json"
        spec.write_text("{not json", encoding="utf-8")
        summary = self.segment(img, spec, out, self.weight)
        self.assertEqual(summary.get("error"), "JSONDecodeError")
        self.assertTrue(summary.get("persisted"))
        self._assert_isolated(out, summary)

    def test_bad_image_isolates_old_success(self):
        out = self.d / "bad-img"
        self._plant(out)
        img = self.d / "missing.png"
        spec = self._tiny_spec()
        summary = self.segment(img, spec, out, self.weight)
        self.assertIn(summary.get("error"), {"FileNotFoundError", "UnidentifiedImageError", "OSError"})
        self._assert_isolated(out, summary)

    def test_raw_shape_mismatch_isolates_old_success(self):
        out = self.d / "shape"
        self._plant(out)
        img = self._tiny_png()
        spec = self._tiny_spec()
        bad = np.zeros((3, 3), dtype=bool)
        summary = self.segment(img, spec, out, self.weight, raw_masks={"staff_upper": bad, "staff_lower": bad})
        self.assertEqual(summary.get("error"), "ValueError")
        self.assertIn("shape mismatch", summary.get("message", ""))
        self._assert_isolated(out, summary)

    def test_sam_exception_isolates_old_success(self):
        from unittest.mock import patch
        out = self.d / "sam-boom"
        self._plant(out)
        img = self._tiny_png()
        spec = self._tiny_spec()
        with patch("scripts.segment_modules_sam._run_sam", side_effect=RuntimeError("sam-down")):
            summary = self.segment(img, spec, out, self.weight, raw_masks=None)
        self.assertEqual(summary.get("error"), "RuntimeError")
        self.assertIn("sam-down", summary.get("message", ""))
        self._assert_isolated(out, summary)

    def test_unwritable_out_dir_does_not_claim_isolation(self):
        blocker = self.d / "not-a-dir"
        blocker.write_text("keep-me", encoding="utf-8")
        img = self._tiny_png()
        spec = self._tiny_spec()
        summary = self.segment(img, spec, blocker, self.weight)
        self.assertFalse(summary["ok"])
        self.assertEqual(summary.get("error"), "OutputDirUnwritable")
        self.assertFalse(summary.get("persisted"))
        self.assertEqual(blocker.read_text(encoding="utf-8"), "keep-me")
        self.assertFalse(blocker.is_dir())

    def test_cli_bad_spec_nonzero_and_isolates(self):
        import subprocess
        import sys
        out = self.d / "cli"
        self._plant(out)
        img = self._tiny_png()
        spec = self.d / "cli-broken.json"
        spec.write_text("{nope", encoding="utf-8")
        run = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "segment_modules_sam.py"),
             "segment", "--image", str(img), "--spec", str(spec), "--out-dir", str(out)],
            capture_output=True, text=True, cwd=str(ROOT),
        )
        self.assertNotEqual(run.returncode, 0, run.stdout + run.stderr)
        payload = json.loads(run.stdout.strip().splitlines()[-1])
        self.assertFalse(payload.get("ok"))
        self.assertFalse((out / "analysis.json").exists())
        prev = list(out.glob("analysis.json.prev-success-*.json"))
        self.assertEqual(len(prev), 1)
        self.assertEqual(json.loads(prev[0].read_text())["marker"], "old-success")
        self.assertTrue((out / "SEGMENT_FAILED").is_file())

class TestArchiveFailureContainment(unittest.TestCase):
    def setUp(self):
        work = ROOT / ".studio-work"
        work.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=work, prefix="seg-arch-")
        self.addCleanup(self.tmp.cleanup)
        self.d = Path(self.tmp.name)
        from scripts.segment_modules_sam import DEFAULT_WEIGHT, segment
        self.segment = segment
        self.weight = DEFAULT_WEIGHT
        import scripts.segment_modules_sam as mod
        self.mod = mod
        self.mod._TEST_RENAME_HOOK = None
        self.mod._TEST_REPLACE_HOOK = None
        self.addCleanup(self._clear_hooks)

    def _clear_hooks(self):
        self.mod._TEST_RENAME_HOOK = None
        self.mod._TEST_REPLACE_HOOK = None

    def _plant(self, out):
        out.mkdir(parents=True, exist_ok=True)
        (out / "analysis.json").write_text(json.dumps({"ok": True, "marker": "old-a"}), encoding="utf-8")
        (out / "segment-report.json").write_text(json.dumps({"ok": True, "marker": "old-r"}), encoding="utf-8")
        (out / "mask-fragments.json").write_text(json.dumps({"old": True}), encoding="utf-8")

    def _tiny(self):
        img = self.d / "src.png"
        Image.fromarray(_rgba(16, 16, [(2, 2, 14, 14, (10, 20, 30))])).save(img)
        spec = _write_spec(self.d / "spec.json", [{
            "id": "staff", "draw_order": 1,
            "pieces": [
                {"id": "staff_upper", "draw_order": 1, "box": [2, 2, 8, 8], "pos": [[4, 4]]},
                {"id": "staff_lower", "draw_order": 2, "box": [8, 8, 14, 14], "pos": [[10, 10]]},
            ],
        }])
        return img, spec

    def test_first_archive_failure_stops_without_sam_and_keeps_bytes(self):
        out = self.d / "first"
        self._plant(out)
        img, spec = self._tiny()
        def hook(src, dest):
            if src.name == "analysis.json":
                raise OSError("injected first archive failure")
        self.mod._TEST_RENAME_HOOK = hook
        summary = self.segment(img, spec, out, self.weight)
        self.assertFalse(summary["ok"])
        self.assertIsNone(summary.get("analysis"))
        self.assertEqual(summary.get("error"), "ArchiveIncomplete")
        self.assertFalse(summary.get("persisted"))
        self.assertTrue((out / "analysis.json").exists())
        self.assertEqual(json.loads((out / "analysis.json").read_text())["marker"], "old-a")
        incomplete = summary.get("isolation_incomplete") or []
        self.assertTrue(any(i.get("file") == "analysis.json" for i in incomplete), incomplete)
        self.assertFalse((out / "masks").exists())

    def test_second_archive_failure_keeps_first_record(self):
        out = self.d / "second"
        self._plant(out)
        img, spec = self._tiny()
        def hook(src, dest):
            if src.name == "segment-report.json":
                raise OSError("injected second archive failure")
        self.mod._TEST_RENAME_HOOK = hook
        summary = self.segment(img, spec, out, self.weight)
        self.assertFalse(summary["ok"])
        self.assertFalse(summary.get("persisted"))
        self.assertFalse((out / "analysis.json").exists())
        prev = list(out.glob("analysis.json.prev-success-*.json"))
        self.assertEqual(len(prev), 1)
        self.assertEqual(json.loads(prev[0].read_text())["marker"], "old-a")
        archived_from = [i.get("from") for i in (summary.get("success_archive") or [])]
        self.assertIn("analysis.json", archived_from)
        self.assertTrue((out / "segment-report.json").exists())
        self.assertEqual(json.loads((out / "segment-report.json").read_text())["marker"], "old-r")
        incomplete = summary.get("isolation_incomplete") or []
        self.assertTrue(any(i.get("file") == "segment-report.json" for i in incomplete), incomplete)
        self.assertFalse((out / "masks").exists())

    def test_persist_failure_does_not_raise_when_archive_fails(self):
        out = self.d / "persist"
        self._plant(out)
        def hook(src, dest):
            raise OSError("injected persist archive failure")
        self.mod._TEST_RENAME_HOOK = hook
        from scripts.segment_modules_sam import persist_failure
        summary = persist_failure(out, {"error": "Boom", "message": "x"}, {"archived": [], "incomplete": [], "analysis": None})
        self.assertFalse(summary["ok"])
        self.assertIsNone(summary.get("analysis"))
        self.assertFalse(summary.get("persisted"))
        self.assertTrue((out / "analysis.json").exists())

    def test_state_write_failure_persisted_false(self):
        out = self.d / "state-write"
        out.mkdir()
        (out / "SEGMENT_FAILED").mkdir()
        img, spec = self._tiny()
        spec.write_text("{nope", encoding="utf-8")
        summary = self.segment(img, spec, out, self.weight)
        self.assertFalse(summary["ok"])
        self.assertFalse(summary.get("persisted"))
        self.assertTrue(summary.get("persist_error"))

    def test_cli_first_archive_failure(self):
        out = self.d / "cli-arch"
        self._plant(out)
        img, spec = self._tiny()
        def hook(src, dest):
            if src.name == "analysis.json":
                raise OSError("archive-cli-boom")
        self.mod._TEST_RENAME_HOOK = hook
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = self.mod.main(["segment", "--image", str(img), "--spec", str(spec), "--out-dir", str(out)])
        self.assertEqual(rc, 1)
        self.assertIn('"ok": false', buf.getvalue())
        self.assertTrue((out / "analysis.json").exists())
        self.assertEqual(json.loads((out / "analysis.json").read_text())["marker"], "old-a")


class TestChainBandAndExclusiveFill(unittest.TestCase):
    def test_parse_chain_band_rejects_illegal(self):
        from scripts.segment_modules_sam import parse_chain_band, expand_segment_targets
        with self.assertRaises(ValueError):
            parse_chain_band("nope", "x")
        with self.assertRaises(ValueError):
            parse_chain_band({"pts": [[0, 0], [10, 0]]}, "x")
        with self.assertRaises(ValueError):
            parse_chain_band({"pts": [[0, 0]], "radius_px": 4}, "x")
        with self.assertRaises(ValueError):
            parse_chain_band({"pts": [[1, 1], [1, 1]], "radius_px": 4}, "x")
        with self.assertRaises(ValueError):
            parse_chain_band({"pts": [[0, 0], [8, 0]], "radius_px": 0}, "x")
        with self.assertRaises(ValueError):
            parse_chain_band({"pts": [[0, 0], [8, 0]], "radius_px": float("inf")}, "x")
        with self.assertRaises(ValueError):
            expand_segment_targets([{"id": "cape", "box": [1, 2, 3, 4], "chain_band": {"pts": [[0, 0]], "radius_px": 3}}])

    def test_piece_overrides_and_inherits_module_band(self):
        from scripts.segment_modules_sam import expand_segment_targets
        mod_band = {"pts": [[0, 0], [20, 0]], "radius_px": 4}
        piece_band = {"pts": [[0, 10], [20, 10]], "radius_px": 6}
        targets = expand_segment_targets([{
            "id": "ribbon", "draw_order": 1, "chain_band": mod_band,
            "pieces": [
                {"id": "ribbon_a", "draw_order": 1, "box": [0, 0, 10, 10], "pos": [[1, 1]]},
                {"id": "ribbon_b", "draw_order": 2, "box": [10, 0, 20, 10], "pos": [[12, 1]], "chain_band": piece_band},
            ],
        }])
        by = {t["id"]: t for t in targets}
        self.assertEqual(by["ribbon_a"]["chain_band"]["radius_px"], 4.0)
        self.assertEqual(by["ribbon_b"]["chain_band"]["radius_px"], 6.0)
        none = expand_segment_targets([{
            "id": "ribbon", "box": [0, 0, 10, 10], "chain_band": mod_band,
            "pieces": [{"id": "ribbon_p0", "draw_order": 1, "box": [0, 0, 10, 10], "chain_band": None}],
        }])
        self.assertIsNone(none[0]["chain_band"])

    def test_band_does_not_cover_adjacent_thigh_and_scales(self):
        from scripts.segment_modules_sam import polyline_band_mask, allowed_region
        h, w = 40, 80
        pts = [[10, 10], [70, 10]]
        band4 = polyline_band_mask(h, w, pts, 4)
        band8 = polyline_band_mask(h, w, pts, 8)
        self.assertTrue(bool(band4[10, 40]))
        self.assertFalse(bool(band4[16, 40]))  # 6px off axis, outside r=4
        self.assertTrue(bool(band8[16, 40]))
        self.assertFalse(bool(band8[30, 40]))  # thigh-like 20px away
        item = {"exclude_boxes": [[0, 0, 20, 40]], "chain_band": {"pts": pts, "radius_px": 8}}
        allow = allowed_region(item, h, w)
        self.assertFalse(bool(allow[10, 5]))   # excluded x<20
        self.assertTrue(bool(allow[10, 40]))
        self.assertFalse(bool(allow[30, 40]))

    def test_missing_chain_band_keeps_full_allowed_minus_excludes(self):
        from scripts.segment_modules_sam import allowed_region
        allow = allowed_region({"exclude_boxes": [[0, 0, 5, 5]]}, 10, 10)
        self.assertFalse(bool(allow[1, 1]))
        self.assertTrue(bool(allow[8, 8]))
        self.assertEqual(int(allow.sum()), 100 - 25)


@unittest.skipUnless(HAS_SEG_DEPS, "cv2/scipy/skimage required; system python does not install them")
class TestChainBandSegment(unittest.TestCase):
    def setUp(self):
        work = ROOT / ".studio-work"
        work.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=work, prefix="seg-band-")
        self.addCleanup(self.tmp.cleanup)
        self.d = Path(self.tmp.name)
        from scripts.segment_modules_sam import DEFAULT_WEIGHT, segment
        self.segment = segment
        self.weight = DEFAULT_WEIGHT

    def test_front_blob_not_swallowed_by_back_hole_fill(self):
        h, w = 48, 48
        arr = np.zeros((h, w, 4), dtype=np.uint8)
        arr[:, :, 3] = 255
        arr[:, :, :3] = 40
        front = np.zeros((h, w), dtype=bool)
        front[18:30, 18:30] = True  # 12x12=144 > PRE_WS_MIN_PX, inside the hole
        back = np.zeros((h, w), dtype=bool)
        back[4:44, 4:12] = True
        back[4:44, 36:44] = True
        back[4:12, 4:44] = True
        back[36:44, 4:44] = True  # frame, hole 12:36
        Image.fromarray(arr).save(self.d / "src.png")
        modules = [
            {"id": "back", "draw_order": 1, "box": [0, 0, 48, 48], "pos": [[8, 8]],
             "pieces": [{"id": "back_p0", "draw_order": 1, "box": [0, 0, 48, 48], "pos": [[8, 8]]}]},
            {"id": "front", "draw_order": 2, "box": [18, 18, 30, 30], "pos": [[24, 24]],
             "pieces": [{"id": "front_p0", "draw_order": 2, "box": [18, 18, 30, 30], "pos": [[24, 24]]}]},
        ]
        spec = _write_spec(self.d / "spec.json", modules)
        out = self.d / "out"
        out.mkdir()
        summary = self.segment(self.d / "src.png", spec, out, self.weight, raw_masks={"back_p0": back, "front_p0": front})
        from PIL import Image as _Image
        fm = np.asarray(_Image.open(out / "masks" / "front_p0.png")) > 127
        bm = np.asarray(_Image.open(out / "masks" / "back_p0.png")) > 127
        self.assertEqual(int((fm & bm).sum()), 0)
        self.assertEqual(int((fm & front).sum()), int(front.sum()))
        if not summary["ok"]:
            codes = {e.get("code") for e in summary.get("errors") or []}
            self.assertNotIn("overlap", codes, summary)

    def test_band_keeps_thigh_out_and_empty_band_fails(self):
        h, w = 32, 64
        arr = _rgba(h, w, [
            (8, 4, 20, 40, (200, 30, 30)),   # ribbon along top
            (22, 4, 30, 40, (30, 200, 30)),  # thigh below
        ])
        raw_r = np.zeros((h, w), dtype=bool); raw_r[8:20, 4:40] = True
        raw_t = np.zeros((h, w), dtype=bool); raw_t[22:30, 4:40] = True
        Image.fromarray(arr).save(self.d / "src.png")
        modules = [
            {"id": "ribbon", "draw_order": 1, "box": [0, 0, 64, 32], "pos": [[20, 12]],
             "chain_band": {"pts": [[4, 14], [40, 14]], "radius_px": 5},
             "pieces": [{"id": "ribbon_p0", "draw_order": 1, "box": [0, 0, 64, 32], "pos": [[20, 12]]}]},
            {"id": "leg", "draw_order": 2, "box": [0, 18, 64, 32], "pos": [[20, 26]],
             "pieces": [{"id": "leg_p0", "draw_order": 2, "box": [0, 18, 64, 32], "pos": [[20, 26]]}]},
        ]
        spec = _write_spec(self.d / "spec.json", modules)
        out = self.d / "band"
        out.mkdir()
        summary = self.segment(self.d / "src.png", spec, out, self.weight, raw_masks={"ribbon_p0": raw_r, "leg_p0": raw_t})
        from PIL import Image as _Image
        rm = np.asarray(_Image.open(out / "masks" / "ribbon_p0.png")) > 127
        self.assertEqual(int((rm[22:30, 4:40]).sum()), 0)
        far = {
            "id": "ribbon", "draw_order": 1, "box": [0, 0, 64, 32], "pos": [[20, 12]],
            "chain_band": {"pts": [[2, 2], [10, 2]], "radius_px": 1},
            "pieces": [{"id": "ribbon_p0", "draw_order": 1, "box": [0, 0, 64, 32], "pos": [[20, 12]]}],
        }
        spec2 = _write_spec(self.d / "spec2.json", [far])
        out2 = self.d / "emptyband"
        out2.mkdir()
        summary2 = self.segment(self.d / "src.png", spec2, out2, self.weight, raw_masks={"ribbon_p0": raw_r})
        self.assertFalse(summary2["ok"])
        codes = {e.get("code") for e in summary2.get("errors") or []}
        self.assertTrue("empty_piece" in codes or "unassigned" in codes, summary2)

class _FakeMasks:
    def __init__(self, arr):
        self.data = arr


class _FakeResult:
    def __init__(self, arr):
        self.masks = _FakeMasks(arr)


def _bbox_predict_factory(calls, *, empty_if_neg=False, empty_always=False):
    def fake(model, source, device, bboxes, points, labels):
        from PIL import Image as _Im
        calls.append({"source": str(source), "bboxes": bboxes, "points": points, "labels": list(labels or [])})
        with _Im.open(source) as im:
            w, h = im.size
        arr = np.zeros((1, h, w), dtype=np.float32)
        if empty_always or (empty_if_neg and labels is not None and 0 in list(labels)):
            return [_FakeResult(arr)]
        if bboxes:
            x1, y1, x2, y2 = [int(round(float(v))) for v in bboxes[0]]
            xa, xb = sorted((x1, x2))
            ya, yb = sorted((y1, y2))
            xa, xb = max(0, xa), min(w, xb)
            ya, yb = max(0, ya), min(h, yb)
            if xb > xa and yb > ya:
                arr[0, ya:yb, xa:xb] = 1.0
        elif points:
            for (x, y), lab in zip(points, labels or []):
                if lab != 1:
                    continue
                xi, yi = int(round(float(x))), int(round(float(y)))
                arr[0, max(0, yi - 8):min(h, yi + 9), max(0, xi - 8):min(w, xi + 9)] = 1.0
        else:
            arr[0] = 1.0
        return [_FakeResult(arr)]
    return fake


class TestIssue03HelpersNoCv2(unittest.TestCase):
    def setUp(self):
        work = ROOT / ".studio-work"
        work.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=work, prefix="seg-i3-")
        self.addCleanup(self.tmp.cleanup)
        self.d = Path(self.tmp.name)

    def test_validate_prompt_points_rejects_oob_and_coincident(self):
        from scripts.segment_modules_sam import validate_prompt_points
        validate_prompt_points([[1, 1]], [[8, 8]], 16, 16)
        with self.assertRaises(ValueError):
            validate_prompt_points([[1, 1]], [[16, 8]], 16, 16)
        with self.assertRaises(ValueError):
            validate_prompt_points([[1, 1]], [[-1, 8]], 16, 16)
        with self.assertRaises(ValueError):
            validate_prompt_points([[4.0, 5.0]], [[4.0, 5.0]], 16, 16)
        with self.assertRaises(ValueError):
            validate_prompt_points([[4.2, 5.2]], [[4.4, 5.4]], 16, 16)

    def test_sam_attempts_never_drops_neg(self):
        from scripts.segment_modules_sam import sam_attempts
        tags = [t[0] for t in sam_attempts([1, 2, 10, 12], [[3, 4]], [[8, 9]])]
        self.assertEqual(tags, ["box+all"])
        tags = [t[0] for t in sam_attempts(None, [[3, 4]], [[8, 9]])]
        self.assertEqual(tags, ["pos+neg"])
        tags = [t[0] for t in sam_attempts([1, 2, 10, 12], [[3, 4]], [])]
        self.assertEqual(tags, ["box+pos", "box-only"])

    def test_compute_roi_box_rules(self):
        from scripts.segment_modules_sam import compute_roi_box
        self.assertEqual(compute_roi_box([10, 10, 20, 20], [], [], 100, 80, roi_mode="full"), (0, 0, 100, 80))
        box = compute_roi_box([100, 100, 110, 110], [], [], 400, 400)
        self.assertEqual(box, (41, 41, 169, 169))
        self.assertGreaterEqual(box[2] - box[0], 128)
        self.assertGreaterEqual(box[3] - box[1], 128)
        self.assertEqual(compute_roi_box([0, 0, 900, 900], [], [], 1000, 1000), (0, 0, 1000, 1000))
        mid = compute_roi_box([100, 100, 200, 200], [], [], 1000, 1000)
        self.assertEqual(mid, (52, 52, 248, 248))
        pts = compute_roi_box(None, [[10, 10], [20, 30]], [[40, 50]], 200, 200)
        self.assertTrue(0 <= pts[0] < pts[2] <= 200)
        self.assertTrue(0 <= pts[1] < pts[3] <= 200)

    def test_map_roi_mask_zero_offset(self):
        from scripts.segment_modules_sam import map_roi_mask_to_full
        roi = np.zeros((6, 5), dtype=bool)
        roi[1, 2] = True
        full = map_roi_mask_to_full(roi, (5, 2, 10, 8), (20, 30))
        self.assertEqual(full.shape, (20, 30))
        self.assertTrue(bool(full[3, 7]))
        self.assertEqual(int(full.sum()), 1)
        self.assertFalse(bool(full[1, 2]))

    def test_prompt_cache_key_changes_with_neg_and_prior(self):
        from scripts.segment_modules_sam import prompt_cache_key
        item = {"exclude_boxes": None, "exclude_halfplanes": None}
        sp = {"box": [1, 2, 10, 12], "pos": [[3, 4]], "neg": [[8, 9]], "roi_mode": "auto", "roi_box": [0, 0, 64, 64]}
        a = prompt_cache_key("imgsha", "sam2.1_b.pt", item, sp, None)
        sp2 = dict(sp, neg=[[8, 10]])
        b = prompt_cache_key("imgsha", "sam2.1_b.pt", item, sp2, None)
        c = prompt_cache_key("imgsha", "sam2.1_b.pt", item, sp, "priorhash")
        self.assertEqual(len(a), 12)
        self.assertNotEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertEqual(a, prompt_cache_key("imgsha", "sam2.1_b.pt", item, sp, None))

    def test_prior_path_traversal_and_size_mismatch(self):
        from scripts.segment_modules_sam import load_prior_mask
        prior = self.d / "priors"
        prior.mkdir()
        Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(prior / "blob.png")
        with self.assertRaises(ValueError) as ctx:
            load_prior_mask(prior, {"id": "blob", "prior_mask": "../../escape.png"}, 8, 8)
        self.assertIn("路径穿越", str(ctx.exception))
        Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(prior / "tiny.png")
        with self.assertRaises(ValueError) as ctx:
            load_prior_mask(prior, {"id": "tiny"}, 8, 8)
        self.assertIn("尺寸", str(ctx.exception))

    def test_parse_override_json_file_and_inline(self):
        from scripts.segment_modules_sam import parse_override_json, apply_target_override
        self.assertEqual(parse_override_json(None), {})
        inline = parse_override_json('{"pos": [[1, 2]], "neg": [[3, 4]]}')
        self.assertEqual(inline["pos"], [[1, 2]])
        path = self.d / "ov.json"
        path.write_text('{"box": [0, 0, 8, 8], "roi_mode": "full"}', encoding="utf-8")
        from_file = parse_override_json(str(path))
        self.assertEqual(from_file["roi_mode"], "full")
        spec = {"version": 2, "modules": [{
            "id": "staff",
            "pieces": [
                {"id": "staff_upper", "pos": [[1, 1]], "box": [0, 0, 4, 4]},
                {"id": "staff_lower", "pos": [[6, 6]], "box": [4, 4, 8, 8]},
            ],
        }]}
        out = apply_target_override(spec, "staff_upper", {"neg": [[2, 2]]})
        self.assertEqual(out["modules"][0]["pieces"][0]["neg"], [[2, 2]])
        self.assertEqual(out["modules"][0]["pieces"][0]["id"], "staff_upper")
        self.assertEqual(out["modules"][0]["pieces"][1]["pos"], [[6, 6]])
        with self.assertRaises(ValueError):
            apply_target_override(spec, "missing", {"pos": [[1, 1]]})
        with self.assertRaises(ValueError):
            apply_target_override(spec, "staff_upper", {"neg": [[2, 2]], "id": "hacked"})
        with self.assertRaises(ValueError):
            apply_target_override(spec, "staff_upper", {"sub_prompts": []})



    def test_connection_point_sync_rules(self):
        from scripts.segment_modules_sam import sync_connection_points
        from src.pipeline.mask_geometry import derive_rigid_pivot
        h, w = 40, 40
        torso = np.zeros((h, w), dtype=bool); torso[4:18, 10:30] = True
        arm = np.zeros((h, w), dtype=bool); arm[8:18, 28:38] = True
        lower = np.zeros((h, w), dtype=bool); lower[16:28, 10:30] = True
        foot = np.zeros((h, w), dtype=bool); foot[26:38, 10:30] = True
        owned = {"torso_p0": torso, "arm_p0": arm, "lower_p0": lower, "foot_p0": foot}
        analysis = {
            "modules": [
                {"id": "torso", "parent": None, "pivot": [0.5, 0.25], "pieces": [{"id": "torso_p0"}]},
                {"id": "arm", "parent": "torso", "pivot": [0.8, 0.3], "pieces": [{"id": "arm_p0"}]},
                {"id": "lower", "parent": "torso", "pivot": [0.5, 0.55], "pieces": [{"id": "lower_p0"}]},
                {"id": "foot", "parent": "lower", "pivot": [0.5, 0.85], "pieces": [{"id": "foot_p0"}]},
            ],
            "connections": [
                {"id": "shoulder", "type": "joint", "a": "torso", "b": "arm", "point": [0.11, 0.11]},
                {"id": "locked", "type": "joint", "a": "torso", "b": "arm", "point": [0.12, 0.13], "manual_point": True},
                {"id": "cross", "type": "attachment", "a": "arm", "b": "foot", "point": [0.22, 0.33]},
                {"id": "ankle", "type": "foot_anchor", "a": "torso", "b": "foot", "chain": ["lower"], "point": [0.01, 0.01]},
            ],
        }
        sync_connection_points(analysis, owned, w, h)
        by = {c["id"]: c for c in analysis["connections"]}
        self.assertEqual(by["shoulder"]["point"], [0.8, 0.3])
        self.assertEqual(by["locked"]["point"], [0.12, 0.13])
        self.assertEqual(by["cross"]["point"], [0.22, 0.33])
        px, py = derive_rigid_pivot(foot, lower)
        self.assertAlmostEqual(by["ankle"]["point"][0], px / w, places=6)
        self.assertAlmostEqual(by["ankle"]["point"][1], py / h, places=6)
        yi = int(round(by["ankle"]["point"][1] * h))
        xi = int(round(by["ankle"]["point"][0] * w))
        self.assertTrue(bool(foot[yi, xi]))
        self.assertLess(abs(yi - 26), 5)


class TestIssue03RunSamNoCv2(unittest.TestCase):
    def setUp(self):
        work = ROOT / ".studio-work"
        work.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=work, prefix="seg-i3run-")
        self.addCleanup(self.tmp.cleanup)
        self.d = Path(self.tmp.name)
        from scripts.segment_modules_sam import DEFAULT_WEIGHT
        self.weight = DEFAULT_WEIGHT
        self.rgb = np.zeros((32, 64, 3), dtype=np.uint8)
        self.fg = np.ones((32, 64), dtype=bool)
        self.cache = self.d / "cache"
        self.cache.mkdir()

    def _patch(self, calls, **kw):
        from unittest.mock import patch
        fake = _bbox_predict_factory(calls, **kw)
        p1 = patch("scripts.segment_modules_sam._predict_sam_mask", side_effect=fake)
        p2 = patch("scripts.segment_modules_sam._load_sam", return_value=object())
        self.addCleanup(p1.stop)
        self.addCleanup(p2.stop)
        p1.start()
        p2.start()

    def test_neg_empty_mask_does_not_fallback(self):
        from scripts.segment_modules_sam import SegmentPromptError, _run_sam
        calls = []
        self._patch(calls, empty_if_neg=True)
        items = [{
            "id": "blade",
            "box": [2, 2, 30, 30],
            "pos": [[8, 8]],
            "neg": [[20, 20]],
        }]
        with self.assertRaises(SegmentPromptError) as ctx:
            _run_sam(self.rgb, self.fg, items, self.cache, self.weight, image_sha256="abc")
        self.assertEqual(ctx.exception.errors[0]["code"], "neg_prompt_failed")
        self.assertEqual(len(calls), 1)
        self.assertIn(0, calls[0]["labels"])

    def test_prompt_hash_cache_and_no_cache(self):
        from scripts.segment_modules_sam import _run_sam
        calls = []
        self._patch(calls)
        items = [{"id": "blade", "box": [2, 2, 30, 30], "pos": [[8, 8]], "neg": [[20, 20]]}]
        _run_sam(self.rgb, self.fg, items, self.cache, self.weight, image_sha256="abc")
        self.assertEqual(len(calls), 1)
        cached = list(self.cache.glob("blade.*.png"))
        self.assertTrue(cached)
        _run_sam(self.rgb, self.fg, items, self.cache, self.weight, image_sha256="abc")
        self.assertEqual(len(calls), 1)
        items2 = [{"id": "blade", "box": [2, 2, 30, 30], "pos": [[8, 8]], "neg": [[21, 20]]}]
        _run_sam(self.rgb, self.fg, items2, self.cache, self.weight, image_sha256="abc")
        self.assertEqual(len(calls), 2)
        _run_sam(self.rgb, self.fg, items2, self.cache, self.weight, image_sha256="abc", no_cache=True)
        self.assertEqual(len(calls), 3)

    def test_prior_dir_default_box_and_fingerprint(self):
        from scripts.segment_modules_sam import _run_sam, prompt_cache_key, compute_roi_box
        prior = self.d / "prior"
        prior.mkdir()
        mask = np.zeros((32, 64), dtype=np.uint8)
        mask[4:20, 8:24] = 255
        Image.fromarray(mask).save(prior / "blade.png")
        calls = []
        self._patch(calls)
        items = [{"id": "blade", "pos": [[12, 10]]}]
        _run_sam(self.rgb, self.fg, items, self.cache, self.weight, image_sha256="abc", prior_dir=prior)
        self.assertEqual(len(calls), 1)
        self.assertIsNotNone(calls[0]["bboxes"])
        digest = hashlib.sha256((prior / "blade.png").read_bytes()).hexdigest()
        roi = compute_roi_box([8, 4, 24, 20], [[12, 10]], [], 64, 32)
        key = prompt_cache_key("abc", Path(self.weight).name, items[0], {
            "box": [8, 4, 24, 20], "pos": [[12, 10]], "neg": [], "roi_mode": "auto", "roi_box": list(roi),
        }, digest)
        self.assertTrue((self.cache / ("blade.%s.png" % key)).is_file())

    def test_segment_neg_failure_isolates_old_analysis(self):
        from unittest.mock import patch
        from scripts.segment_modules_sam import DEFAULT_WEIGHT, segment
        out = self.d / "iso"
        out.mkdir()
        (out / "analysis.json").write_text(json.dumps({"ok": True, "marker": "old-success"}), encoding="utf-8")
        img = self.d / "src.png"
        Image.fromarray(_rgba(32, 64, [(0, 0, 32, 64, (10, 20, 30))])).save(img)
        spec = _write_spec(self.d / "spec.json", [{
            "id": "blade",
            "pieces": [{"id": "blade_p0", "box": [2, 2, 30, 30], "pos": [[8, 8]], "neg": [[20, 20]]}],
        }])
        calls = []
        fake = _bbox_predict_factory(calls, empty_if_neg=True)
        with patch("scripts.segment_modules_sam._predict_sam_mask", side_effect=fake), patch(
            "scripts.segment_modules_sam._load_sam", return_value=object()
        ):
            summary = segment(img, spec, out, DEFAULT_WEIGHT)
        self.assertFalse(summary["ok"])
        codes = {e.get("code") for e in summary.get("errors") or []}
        self.assertIn("neg_prompt_failed", codes)
        self.assertFalse((out / "analysis.json").exists())
        prev = list(out.glob("analysis.json.prev-success-*.json"))
        self.assertEqual(len(prev), 1)
        self.assertEqual(json.loads(prev[0].read_text())["marker"], "old-success")
        self.assertEqual(len(calls), 1)
        self.assertIn(0, calls[0]["labels"])

    def test_cli_refine_one_missing_target_isolates(self):
        from scripts.segment_modules_sam import main
        out = self.d / "cli-refine"
        out.mkdir()
        (out / "analysis.json").write_text(json.dumps({"ok": True, "marker": "old-success"}), encoding="utf-8")
        img = self.d / "src.png"
        Image.fromarray(_rgba(16, 16, [(2, 2, 14, 14, (10, 20, 30))])).save(img)
        spec = _write_spec(self.d / "spec.json", [{
            "id": "blade",
            "pieces": [{"id": "blade_p0", "box": [2, 2, 8, 8], "pos": [[4, 4]]}],
        }])
        from io import StringIO
        from unittest.mock import patch
        with patch("sys.stdout", new=StringIO()):
            rc = main([
                "refine-one", "--image", str(img), "--spec", str(spec), "--out-dir", str(out),
                "--target-id", "nope",
            ])
        self.assertEqual(rc, 1)
        self.assertFalse((out / "analysis.json").exists())
        prev = list(out.glob("analysis.json.prev-success-*.json"))
        self.assertEqual(len(prev), 1)



    def test_override_top_level_pos_neg_replaces_old_sub_prompts(self):
        from scripts.segment_modules_sam import apply_target_override, expand_segment_targets, _run_sam, _subprompts
        spec = {"version": 2, "modules": [{
            "id": "blade",
            "pieces": [{
                "id": "blade_p0",
                "box": [2, 2, 30, 30],
                "sub_prompts": [{"box": [2, 2, 30, 30], "pos": [[5, 5]], "neg": [[25, 5]]}],
            }],
        }]}
        out = apply_target_override(spec, "blade_p0", {"pos": [[10, 12]], "neg": [[22, 18]]})
        piece = out["modules"][0]["pieces"][0]
        self.assertEqual(piece["pos"], [[10, 12]])
        self.assertEqual(piece["neg"], [[22, 18]])
        self.assertFalse(bool(piece.get("sub_prompts")))
        sps = _subprompts(expand_segment_targets(out["modules"])[0])
        self.assertEqual(len(sps), 1)
        self.assertEqual(sps[0]["pos"], [[10, 12]])
        self.assertEqual(sps[0]["neg"], [[22, 18]])
        calls = []
        self._patch(calls)
        items = expand_segment_targets(out["modules"])
        _run_sam(self.rgb, self.fg, items, self.cache, self.weight, image_sha256="abc")
        self.assertEqual(len(calls), 1)
        pts = calls[0]["points"]
        labels = calls[0]["labels"]
        pos_pts = [p for p, lab in zip(pts, labels) if lab == 1]
        neg_pts = [p for p, lab in zip(pts, labels) if lab == 0]
        self.assertTrue(any(abs(p[0] - 10) < 1 and abs(p[1] - 12) < 1 for p in pos_pts), pos_pts)
        self.assertTrue(any(abs(p[0] - 22) < 1 and abs(p[1] - 18) < 1 for p in neg_pts), neg_pts)
        self.assertFalse(any(abs(p[0] - 5) < 1 and abs(p[1] - 5) < 1 for p in pos_pts))

    def test_override_unknown_or_empty_subprompts_isolates(self):
        from scripts.segment_modules_sam import refine_one, DEFAULT_WEIGHT
        out = self.d / "ov-bad"
        out.mkdir()
        (out / "analysis.json").write_text(json.dumps({"ok": True, "marker": "old-success"}), encoding="utf-8")
        img = self.d / "src.png"
        Image.fromarray(_rgba(16, 16, [(2, 2, 14, 14, (10, 20, 30))])).save(img)
        spec = _write_spec(self.d / "spec.json", [{
            "id": "blade",
            "pieces": [{"id": "blade_p0", "box": [2, 2, 8, 8], "pos": [[4, 4]],
                        "sub_prompts": [{"box": [2, 2, 8, 8], "pos": [[4, 4]]}]}],
        }])
        summary = refine_one(img, spec, out, DEFAULT_WEIGHT, "blade_p0", {"pos": [[5, 5]], "foo": 1})
        self.assertFalse(summary["ok"])
        self.assertIn("未知字段", summary.get("message", ""))
        self.assertFalse((out / "analysis.json").exists())
        prev = list(out.glob("analysis.json.prev-success-*.json"))
        self.assertEqual(len(prev), 1)
        (out / "analysis.json").write_text(json.dumps({"ok": True, "marker": "old-success-2"}), encoding="utf-8")
        summary2 = refine_one(img, spec, out, DEFAULT_WEIGHT, "blade_p0", {"sub_prompts": []})
        self.assertFalse(summary2["ok"])
        self.assertIn("sub_prompts", summary2.get("message", ""))
        self.assertFalse((out / "analysis.json").exists())


@unittest.skipUnless(HAS_SEG_DEPS, "cv2/scipy/skimage required; system python does not install them")
class TestIssue03RefineOne(unittest.TestCase):
    def setUp(self):
        work = ROOT / ".studio-work"
        work.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=work, prefix="seg-i3ref-")
        self.addCleanup(self.tmp.cleanup)
        self.d = Path(self.tmp.name)
        from scripts.segment_modules_sam import DEFAULT_WEIGHT, segment, refine_one
        self.segment = segment
        self.refine_one = refine_one
        self.weight = DEFAULT_WEIGHT

    def _two_piece_scene(self):
        arr = np.zeros((32, 64, 4), dtype=np.uint8)
        arr[:, :32, :3] = (200, 40, 40)
        arr[:, 32:, :3] = (40, 40, 200)
        arr[..., 3] = 255
        img = self.d / "src.png"
        Image.fromarray(arr).save(img)
        modules = [
            {
                "id": "torso", "draw_order": 1, "role": "torso", "parent": None,
                "pieces": [{"id": "torso_p0", "draw_order": 1, "box": [0, 0, 32, 32], "pos": [[8, 16]]}],
            },
            {
                "id": "ribbon", "draw_order": 2, "role": "cloth", "parent": "torso", "motion_type": "soft",
                "pieces": [{"id": "ribbon_p0", "draw_order": 2, "box": [32, 0, 64, 32], "pos": [[40, 16]]}],
            },
        ]
        spec = _write_spec(self.d / "spec.json", modules)
        return img, spec, arr

    def _patch(self, calls, **kw):
        from unittest.mock import patch
        fake = _bbox_predict_factory(calls, **kw)
        p1 = patch("scripts.segment_modules_sam._predict_sam_mask", side_effect=fake)
        p2 = patch("scripts.segment_modules_sam._load_sam", return_value=object())
        self.addCleanup(p1.stop)
        self.addCleanup(p2.stop)
        p1.start()
        p2.start()

    def test_refine_one_reruns_target_only_and_refreshes_geometry(self):
        img, spec, _arr = self._two_piece_scene()
        out = self.d / "out"
        out.mkdir()
        calls = []
        self._patch(calls)
        summary = self.segment(img, spec, out, self.weight)
        self.assertTrue(summary["ok"], summary)
        first_calls = len(calls)
        self.assertEqual(first_calls, 2)
        analysis = json.loads((out / "analysis.json").read_text())
        ribbon = [m for m in analysis["modules"] if m["id"] == "ribbon"][0]
        self.assertGreaterEqual(len(ribbon.get("centerline") or []), 3)
        self.assertEqual(len(ribbon["pivot"]), 2)
        summary2 = self.refine_one(
            img, spec, out, self.weight, "ribbon_p0",
            {"pos": [[44, 16]], "neg": [[50, 8]]},
        )
        self.assertTrue(summary2["ok"], summary2)
        self.assertEqual(len(calls), first_calls + 1)
        self.assertIn("ribbon", Path(calls[-1]["source"]).name)
        analysis2 = json.loads((out / "analysis.json").read_text())
        ribbon2 = [m for m in analysis2["modules"] if m["id"] == "ribbon"][0]
        self.assertGreaterEqual(len(ribbon2.get("centerline") or []), 3)

    def test_refine_one_failure_keeps_old_analysis_bytes(self):
        img, spec, _arr = self._two_piece_scene()
        out = self.d / "keep"
        out.mkdir()
        calls = []
        self._patch(calls)
        summary = self.segment(img, spec, out, self.weight)
        self.assertTrue(summary["ok"], summary)
        old = (out / "analysis.json").read_bytes()
        summary2 = self.refine_one(
            img, spec, out, self.weight, "ribbon_p0",
            {"pos": [[40, 16]], "neg": [[40, 16]]},
        )
        self.assertFalse(summary2["ok"])
        self.assertFalse((out / "analysis.json").exists())
        prev = list(out.glob("analysis.json.prev-success-*.json"))
        self.assertEqual(len(prev), 1)
        self.assertEqual(prev[0].read_bytes(), old)
        self.assertTrue((out / "SEGMENT_FAILED").is_file())
        report = json.loads((out / "segment-report.json").read_text())
        self.assertFalse(report["ok"])
