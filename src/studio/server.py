"""Local offline HTTP server for Portrait Spine Studio (Python 3.9 stdlib only).

Serves:
- `/`                -> `src/studio/web/`
- `/player/<path>`   -> `src/player/<path>`
- `/jobs/<id>/<path>`-> `<work_dir>/<id>/<path>`

API endpoints:
- `GET  /api/health`
- `POST /api/jobs`
- `GET  /api/jobs/<job_id>`
- `POST /api/jobs/<job_id>/continue`
- `POST /api/jobs/<job_id>/retry`
- `POST /api/jobs/<job_id>/discard`
- `POST /api/jobs/<job_id>/adjust`
- `POST /api/jobs/<job_id>/export`
"""
from __future__ import annotations

import argparse
import base64
import copy
import datetime
import json
import mimetypes
import threading
import time
import urllib.parse
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from src.export.exporter import export_default_package, export_transparent_web
from src.gates.audit import GateFailureError, run_audit
from src.mivo.adapter import (
    DeterministicFallbackMock,
    FileBackedMivoAdapter,
    MivoCanvasAdapter,
    MivoCanvasError,
    StageRunner,
)
from src.pipeline.build_rig import build_rig
from src.pipeline.preprocess import MAX_LONG_EDGE, preprocess_portrait
from src.pipeline.presets import list_presets
from src.pipeline.split_parts import split_parts

from .auto_spec import build_default_specs, sanitize_character_name
from .nl_adjust import apply_spec_patch, classify_and_polish_adjustment

REPO_ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = Path(__file__).resolve().parent / "web"
PLAYER_DIR = REPO_ROOT / "src" / "player"

mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("application/javascript", ".mjs")
mimetypes.add_type("image/webp", ".webp")
mimetypes.add_type("video/mp4", ".mp4")
mimetypes.add_type("video/webm", ".webm")


class StudioBusyError(RuntimeError):
    """Raised when a mutating job request arrives while another task holds the serial lock."""

    def __init__(self, active_job_id: Optional[str]):
        super().__init__(
            "当前已有任务正在执行中（单任务串行模式），请等待当前任务完成或放弃后再提交新任务。"
        )
        self.active_job_id = active_job_id


class JobNotFoundError(KeyError):
    """Raised when a requested job_id does not exist."""


class InvalidJobStateError(ValueError):
    """Raised when an action is invalid for the job's current state."""


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()


def _normalize_parts_spec(raw: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if isinstance(raw, list):
        return [dict(p) for p in raw], {}
    if isinstance(raw, dict):
        parts = raw.get("parts")
        if not isinstance(parts, list) or not parts:
            raise ValueError("parts_spec 必须包含非空的 parts 列表")
        opts: Dict[str, Any] = {}
        if isinstance(raw.get("seams"), dict):
            seams_out = {}
            for k, v in raw["seams"].items():
                pair = tuple(k.split("|", 1)) if isinstance(k, str) and "|" in k else tuple(k)
                seams_out[pair] = v
            opts["seams"] = seams_out
        if isinstance(raw.get("pair_bands"), dict):
            bands_out = {}
            for k, v in raw["pair_bands"].items():
                pair = tuple(k.split("|", 1)) if isinstance(k, str) and "|" in k else tuple(k)
                bands_out[pair] = int(v)
            opts["pair_bands"] = bands_out
        for k in ("opaque_threshold", "excl_px", "feather_px", "min_island_px"):
            if k in raw:
                opts[k] = int(raw[k])
        return [dict(p) for p in parts], opts
    raise ValueError("parts_spec 必须是列表或包含 parts 的对象")


def _decode_image_base64(raw_b64: str) -> bytes:
    if not raw_b64 or not isinstance(raw_b64, str):
        raise ValueError("缺少有效的 image_base64 图片数据")
    payload = raw_b64.strip()
    if "," in payload and payload.startswith("data:"):
        payload = payload.split(",", 1)[1]
    try:
        data = base64.b64decode(payload, validate=False)
    except Exception as exc:
        raise ValueError(f"图片 Base64 解码失败：{exc}") from exc
    if not data:
        raise ValueError("上传图片内容为空")
    return data


def _initial_stages() -> List[Dict[str, str]]:
    return [
        {"id": "preprocess", "label": "输入预处理（<=2400px 缩放）", "status": "pending"},
        {"id": "mivo", "label": "Mivo Canvas 智能抠图", "status": "pending"},
        {"id": "split_parts", "label": "分层分件与遮挡补全", "status": "pending"},
        {"id": "build_rig", "label": "骨骼 / 权重 / Spine 4.2 动画构建", "status": "pending"},
        {"id": "audit", "label": "4 项硬门禁与软提示验收", "status": "pending"},
    ]


class StudioJobManager:
    """Thread-safe single-task serial job manager backed by `<work_dir>/<job_id>/`."""

    def __init__(
        self,
        work_dir: Union[str, Path],
        *,
        mivo_handoff_dir: Optional[Union[str, Path]] = None,
        audit_samples: int = 12,
        stress_samples: int = 6,
    ):
        self.work_dir = Path(work_dir).resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.mivo_handoff_dir = Path(mivo_handoff_dir).resolve() if mivo_handoff_dir else None
        self.audit_samples = audit_samples
        self.stress_samples = stress_samples

        self._exec_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._active_job_id: Optional[str] = None
        self._jobs: Dict[str, Dict[str, Any]] = {}

    def health(self) -> Dict[str, Any]:
        with self._state_lock:
            return {
                "ok": True,
                "busy": self._exec_lock.locked(),
                "active_job_id": self._active_job_id,
                "work_dir": str(self.work_dir),
                "presets": list_presets(),
            }

    def get_job(self, job_id: str) -> Dict[str, Any]:
        with self._state_lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise JobNotFoundError(job_id)
            return self._public_view(job)

    def _public_view(self, job: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "job_id": job["job_id"],
            "character": job["character"],
            "filename": job["filename"],
            "status": job["status"],
            "preset": job["preset"],
            "gain": job["gain"],
            "pause_after_split": job["pause_after_split"],
            "created_at": job["created_at"],
            "updated_at": job["updated_at"],
            "stages": copy.deepcopy(job["stages"]),
            "resize_notice": job["resize_notice"],
            "preprocess": copy.deepcopy(job["preprocess"]),
            "health_report": copy.deepcopy(job["health_report"]),
            "rig_url": job["rig_url"],
            "health_report_url": job["health_report_url"],
            "part_map_url": job["part_map_url"],
            "rig_version": job["rig_build_count"],
            "split_run_count": job["split_run_count"],
            "rig_build_count": job["rig_build_count"],
            "stage_error": copy.deepcopy(job["stage_error"]),
            "preserved_intermediates": copy.deepcopy(job["preserved_intermediates"]),
            "adjustment_log": copy.deepcopy(job["adjustment_log"]),
            "exports": copy.deepcopy(job["exports"]),
        }

    def _set_stage(self, job: Dict[str, Any], stage_id: str, status: str) -> None:
        with self._state_lock:
            for st in job["stages"]:
                if st["id"] == stage_id:
                    st["status"] = status
                    break
            job["updated_at"] = _now_iso()

    def _job_url(self, job_id: str, rel_posix: str) -> str:
        clean = rel_posix.lstrip("/")
        return f"/jobs/{job_id}/{clean}"

    def _make_mivo_adapter(self, simulate_error: bool) -> MivoCanvasAdapter:
        if simulate_error:
            return DeterministicFallbackMock(fail_on="auto_matting", error_code="MIVO_SERVICE_UNAVAILABLE")
        if self.mivo_handoff_dir is not None:
            return FileBackedMivoAdapter(self.mivo_handoff_dir)
        return DeterministicFallbackMock()

    def _acquire_serial(self, job_id: str) -> None:
        if not self._exec_lock.acquire(blocking=False):
            with self._state_lock:
                active = self._active_job_id
            raise StudioBusyError(active)
        with self._state_lock:
            self._active_job_id = job_id

    def _release_serial(self) -> None:
        with self._state_lock:
            self._active_job_id = None
        if self._exec_lock.locked():
            self._exec_lock.release()

    def _persist_job_meta(self, job: Dict[str, Any]) -> None:
        job_dir = Path(job["job_dir"])
        (job_dir / "job-status.json").write_text(
            json.dumps(self._public_view(job), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def _run_pipeline_stages(
        self,
        job: Dict[str, Any],
        *,
        start_from: str = "preprocess",
        stop_after_split: bool = False,
        simulate_mivo_error: bool = False,
        simulate_delay_ms: int = 0,
    ) -> Dict[str, Any]:
        job_id = job["job_id"]
        job_dir = Path(job["job_dir"])
        input_path = Path(job["input_path"])
        preprocessed_path = job_dir / "preprocessed.png"
        matted_path = job_dir / "matted.png"
        parts_dir = job_dir / "parts_out"
        rig_dir = job_dir / "rig"

        runner = StageRunner(job_dir)
        # Seed runner with any previously preserved intermediates
        for k, v in job["preserved_intermediates"].items():
            runner.intermediates[k] = v

        order = ["preprocess", "mivo", "split_parts", "build_rig", "audit"]
        start_idx = order.index(start_from)

        with self._state_lock:
            job["status"] = "running"
            job["stage_error"] = None
            for idx, st in enumerate(job["stages"]):
                if idx >= start_idx:
                    st["status"] = "pending"
            job["updated_at"] = _now_iso()

        if simulate_delay_ms > 0:
            time.sleep(min(5.0, float(simulate_delay_ms) / 1000.0))

        def stage_preprocess() -> Dict[str, Any]:
            self._set_stage(job, "preprocess", "running")
            res = preprocess_portrait(input_path, preprocessed_path, max_long_edge=MAX_LONG_EDGE)
            with self._state_lock:
                job["preprocess"] = res
                job["resize_notice"] = res.get("notice")
            self._set_stage(job, "preprocess", "done")
            return res

        def stage_mivo() -> Any:
            self._set_stage(job, "mivo", "running")
            adapter = self._make_mivo_adapter(simulate_error=simulate_mivo_error)
            res = adapter.auto_matting(preprocessed_path, matted_path)
            self._set_stage(job, "mivo", "done")
            return res

        def stage_split() -> Dict[str, Any]:
            self._set_stage(job, "split_parts", "running")
            if job["parts_spec"] is None or job["rig_spec"] is None:
                auto_parts, auto_rig = build_default_specs(
                    matted_path,
                    character=job["character"],
                    preset=job["preset"],
                )
                if job["parts_spec"] is None:
                    job["parts_spec"] = auto_parts
                if job["rig_spec"] is None:
                    job["rig_spec"] = auto_rig
            part_list, part_opts = _normalize_parts_spec(job["parts_spec"])
            res = split_parts(matted_path, part_list, parts_dir, **part_opts)
            with self._state_lock:
                job["split_run_count"] += 1
                job["part_map_url"] = self._job_url(job_id, "parts_out/review/part_map.png")
            self._set_stage(job, "split_parts", "done")
            return res

        def stage_build_rig() -> Dict[str, Any]:
            self._set_stage(job, "build_rig", "running")
            res = build_rig(
                parts_dir,
                job["rig_spec"],
                rig_dir,
                preset=job["preset"],
                gain=float(job["gain"]),
            )
            spine_file = res["report"]["spineFile"]
            with self._state_lock:
                job["rig_build_count"] += 1
                job["rig_url"] = self._job_url(job_id, f"rig/{spine_file}")
            self._set_stage(job, "build_rig", "done")
            return res

        def stage_audit() -> Dict[str, Any]:
            self._set_stage(job, "audit", "running")
            hr = run_audit(
                parts_dir,
                rig_dir,
                sample_count=self.audit_samples,
                stress_samples=self.stress_samples,
                raise_on_fail=True,
            )
            with self._state_lock:
                job["health_report"] = hr
                job["health_report_url"] = self._job_url(job_id, "rig/health-report.json")
            self._set_stage(job, "audit", "done")
            return hr

        stage_fns: Dict[str, Callable[[], Any]] = {
            "preprocess": stage_preprocess,
            "mivo": stage_mivo,
            "split_parts": stage_split,
            "build_rig": stage_build_rig,
            "audit": stage_audit,
        }

        end_idx = order.index("split_parts") + 1 if stop_after_split else len(order)
        plan = [(name, stage_fns[name]) for name in order[start_idx:end_idx]]

        outcome = runner.run_stages(plan, raise_on_error=False)
        with self._state_lock:
            job["preserved_intermediates"] = dict(runner.intermediates)
            if outcome.get("status") == "failed":
                failed_stage = outcome["failedStage"]
                for st in job["stages"]:
                    if st["id"] == failed_stage:
                        st["status"] = "failed"
                job["status"] = "failed"
                job["stage_error"] = outcome
                if failed_stage == "audit" and isinstance(outcome.get("details"), dict):
                    job["health_report"] = outcome["details"]
            elif stop_after_split:
                job["status"] = "awaiting_split_confirm"
            else:
                job["status"] = "ready"
            job["updated_at"] = _now_iso()

        self._persist_job_meta(job)
        return self.get_job(job_id)

    def create_job(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        filename = str(payload.get("filename") or "portrait.png")
        image_bytes = _decode_image_base64(payload.get("image_base64", ""))
        preset = str(payload.get("preset") or "agile_full")
        if preset not in list_presets():
            raise ValueError(f"未知预设 {preset!r}，可选：{list_presets()}")
        gain = float(payload.get("gain", 1.0))
        pause_after_split = bool(payload.get("pause_after_split", False))
        simulate_mivo_error = bool(payload.get("simulate_mivo_error", False))
        simulate_delay_ms = int(payload.get("simulate_delay_ms", 0))

        job_id = uuid.uuid4().hex[:12]
        self._acquire_serial(job_id)
        try:
            job_dir = self.work_dir / job_id
            input_dir = job_dir / "input"
            input_dir.mkdir(parents=True, exist_ok=True)
            raw_name = Path(filename.replace("\\", "/")).name
            safe_name = raw_name if raw_name and raw_name not in (".", "..") and "\x00" not in raw_name else "portrait.png"
            input_path = input_dir / safe_name
            input_path.write_bytes(image_bytes)

            character = sanitize_character_name(str(payload.get("character") or safe_name))
            now = _now_iso()
            job: Dict[str, Any] = {
                "job_id": job_id,
                "job_dir": str(job_dir),
                "input_path": str(input_path),
                "character": character,
                "filename": safe_name,
                "status": "running",
                "preset": preset,
                "gain": gain,
                "pause_after_split": pause_after_split,
                "created_at": now,
                "updated_at": now,
                "stages": _initial_stages(),
                "resize_notice": None,
                "preprocess": None,
                "health_report": None,
                "rig_url": None,
                "health_report_url": None,
                "part_map_url": None,
                "stage_error": None,
                "preserved_intermediates": {"input": str(input_path)},
                "adjustment_log": [],
                "exports": {},
                "parts_spec": copy.deepcopy(payload.get("parts_spec")),
                "rig_spec": copy.deepcopy(payload.get("rig_spec")),
                "split_run_count": 0,
                "rig_build_count": 0,
            }
            with self._state_lock:
                self._jobs[job_id] = job

            return self._run_pipeline_stages(
                job,
                start_from="preprocess",
                stop_after_split=pause_after_split,
                simulate_mivo_error=simulate_mivo_error,
                simulate_delay_ms=simulate_delay_ms,
            )
        finally:
            self._release_serial()

    def continue_job(self, job_id: str) -> Dict[str, Any]:
        with self._state_lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise JobNotFoundError(job_id)
            if job["status"] != "awaiting_split_confirm":
                raise InvalidJobStateError(f"任务当前状态为 {job['status']}，仅 awaiting_split_confirm 状态可继续生成。")

        self._acquire_serial(job_id)
        try:
            return self._run_pipeline_stages(job, start_from="build_rig", stop_after_split=False)
        finally:
            self._release_serial()

    def retry_job(self, job_id: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        opts = payload or {}
        with self._state_lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise JobNotFoundError(job_id)
            if job["status"] != "failed" or not job["stage_error"]:
                raise InvalidJobStateError(f"任务当前状态为 {job['status']}，仅失败任务可重试当前阶段。")
            failed_stage = str(job["stage_error"].get("failedStage") or "preprocess")
            if opts.get("preset") in list_presets():
                job["preset"] = str(opts["preset"])
            if opts.get("gain") is not None:
                job["gain"] = float(opts["gain"])

        self._acquire_serial(job_id)
        try:
            stop_after_split = bool(job["pause_after_split"]) and failed_stage in ("preprocess", "mivo", "split_parts")
            return self._run_pipeline_stages(
                job,
                start_from=failed_stage,
                stop_after_split=stop_after_split,
                simulate_mivo_error=bool(opts.get("simulate_mivo_error", False)),
            )
        finally:
            self._release_serial()

    def discard_job(self, job_id: str) -> Dict[str, Any]:
        with self._state_lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise JobNotFoundError(job_id)
            job_dir = Path(job["job_dir"])
            err_file = job_dir / "stage-error.json"
            if err_file.exists():
                err_file.unlink()
            job["status"] = "discarded"
            job["stage_error"] = None
            job["updated_at"] = _now_iso()
        self._persist_job_meta(job)
        return self.get_job(job_id)

    def adjust_job(self, job_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self._state_lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise JobNotFoundError(job_id)
            if job["status"] not in ("ready", "failed"):
                raise InvalidJobStateError(f"任务当前状态为 {job['status']}，无法执行微调。")

        instruction = str(payload.get("instruction") or "")
        req_preset = payload.get("preset")
        if req_preset is not None and req_preset not in list_presets():
            raise ValueError(f"未知预设 {req_preset!r}，可选：{list_presets()}")
        req_gain = float(payload["gain"]) if payload.get("gain") is not None else None

        self._acquire_serial(job_id)
        try:
            adj = classify_and_polish_adjustment(
                instruction,
                current_preset=job["preset"],
                current_gain=float(job["gain"]),
                requested_preset=req_preset,
                requested_gain=req_gain,
                rig_spec=job["rig_spec"],
            )
            new_parts, new_rig, new_preset, new_gain = apply_spec_patch(
                job["parts_spec"],
                job["rig_spec"],
                adj,
            )
            with self._state_lock:
                job["parts_spec"] = new_parts
                job["rig_spec"] = new_rig
                job["preset"] = new_preset
                job["gain"] = new_gain

            start_stage = "split_parts" if adj["rerun_layer"] == "split" else "build_rig"
            updated = self._run_pipeline_stages(job, start_from=start_stage, stop_after_split=False)

            log_entry = {
                "timestamp": _now_iso(),
                "raw_instruction": adj["raw_instruction"],
                "polished_instruction": adj["polished_instruction"],
                "rerun_layer": adj["rerun_layer"],
                "skipped_stages": adj["skipped_stages"],
                "executed_stages": adj["executed_stages"],
                "preset": new_preset,
                "gain": new_gain,
                "audit_status": (updated.get("health_report") or {}).get("status", updated["status"]),
            }
            with self._state_lock:
                job["adjustment_log"].append(log_entry)
                job["updated_at"] = _now_iso()
            self._persist_job_meta(job)
            return self.get_job(job_id)
        finally:
            self._release_serial()

    def export_job(self, job_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        kind = str(payload.get("kind") or "spine_package")
        if kind not in ("spine_package", "preview_video", "transparent_web"):
            raise ValueError(f"不支持的导出类型 {kind!r}，可选：spine_package / preview_video / transparent_web")

        with self._state_lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise JobNotFoundError(job_id)
            if job["status"] != "ready":
                raise InvalidJobStateError(f"仅硬门禁全部通过（ready）的任务可导出，当前状态为 {job['status']}。")
            job_dir = Path(job["job_dir"])

        rig_dir = job_dir / "rig"
        exports_dir = job_dir / "exports"

        self._acquire_serial(job_id)
        try:
            if kind == "spine_package":
                res = export_default_package(rig_dir, exports_dir, render_previews=False)
                zip_name = Path(res["zip_path"]).name
                record = {
                    "kind": kind,
                    "created_at": _now_iso(),
                    "files": [
                        {
                            "label": "Spine 4.2 工程包 (.zip)",
                            "filename": zip_name,
                            "url": self._job_url(job_id, f"exports/{zip_name}"),
                        }
                    ],
                    "manifest": res["zip_entries"],
                    "notice": None,
                }
            elif kind == "preview_video":
                res = export_default_package(
                    rig_dir,
                    exports_dir,
                    render_previews=True,
                    preview_fps=int(payload.get("fps", 15)),
                    preview_max_frames=payload.get("max_frames"),
                )
                files: List[Dict[str, str]] = []
                zip_name = Path(res["zip_path"]).name
                files.append(
                    {
                        "label": "Spine 4.2 工程包 (.zip)",
                        "filename": zip_name,
                        "url": self._job_url(job_id, f"exports/{zip_name}"),
                    }
                )
                if res["previews"].get("dark"):
                    dn = Path(res["previews"]["dark"]).name
                    files.append(
                        {
                            "label": "深底预览视频 (.mp4)",
                            "filename": dn,
                            "url": self._job_url(job_id, f"exports/{dn}"),
                        }
                    )
                if res["previews"].get("light"):
                    ln = Path(res["previews"]["light"]).name
                    files.append(
                        {
                            "label": "浅底预览视频 (.mp4)",
                            "filename": ln,
                            "url": self._job_url(job_id, f"exports/{ln}"),
                        }
                    )
                record = {
                    "kind": kind,
                    "created_at": _now_iso(),
                    "files": files,
                    "manifest": [f["filename"] for f in files],
                    "notice": res.get("preview_notice"),
                }
            else:
                web_dir = exports_dir / "web"
                res = export_transparent_web(
                    rig_dir,
                    web_dir,
                    fps=int(payload.get("fps", 30)),
                    max_frames=payload.get("max_frames"),
                )
                files = []
                mapping = [
                    ("packed_h264_mp4", "透明打包 H.264 (.mp4)"),
                    ("packed_av1_mp4", "透明打包 AV1 (.mp4)"),
                    ("alpha_vp9_webm", "原生透明 VP9 (.webm)"),
                    ("poster_webp", "首帧静态图 (.webp)"),
                ]
                for key, label in mapping:
                    fp = res.get(key)
                    if fp:
                        fn = Path(fp).name
                        files.append(
                            {
                                "label": label,
                                "filename": fn,
                                "url": self._job_url(job_id, f"exports/web/{fn}"),
                            }
                        )
                record = {
                    "kind": kind,
                    "created_at": _now_iso(),
                    "files": files,
                    "manifest": [f["filename"] for f in files],
                    "notice": None,
                }

            with self._state_lock:
                job["exports"][kind] = record
                job["updated_at"] = _now_iso()
            self._persist_job_meta(job)
            return record
        finally:
            self._release_serial()


def _safe_resolve_under(base_dir: Path, rel_path: str) -> Optional[Path]:
    clean = urllib.parse.unquote(rel_path)
    if "\x00" in clean:
        return None
    clean = clean.replace("\\", "/").lstrip("/")
    try:
        base_resolved = base_dir.resolve()
        target = (base_resolved / clean).resolve()
        target.relative_to(base_resolved)
    except (ValueError, OSError):
        return None
    return target


def create_studio_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    work_dir: Optional[Union[str, Path]] = None,
    *,
    manager: Optional[StudioJobManager] = None,
) -> ThreadingHTTPServer:
    """Create a configured `ThreadingHTTPServer` for Portrait Spine Studio."""
    wdir = Path(work_dir).resolve() if work_dir else (REPO_ROOT / ".studio-work")
    job_manager = manager or StudioJobManager(wdir)

    class StudioRequestHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            # Keep server output concise unless debugging
            pass

        def _send_json(self, status: int, data: Dict[str, Any]) -> None:
            raw = (json.dumps(data, ensure_ascii=False) + "\n").encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)

        def _send_file(self, path: Path) -> None:
            if not path.is_file():
                self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "errorCode": "NOT_FOUND", "message": "文件不存在"})
                return
            mime, _ = mimetypes.guess_type(str(path))
            if path.suffix.lower() in (".js", ".mjs"):
                mime = "application/javascript; charset=utf-8"
            elif path.suffix.lower() == ".html":
                mime = "text/html; charset=utf-8"
            elif path.suffix.lower() == ".json":
                mime = "application/json; charset=utf-8"
            elif not mime:
                mime = "application/octet-stream"

            data = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            if path.suffix.lower() == ".zip":
                self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
            self.end_headers()
            self.wfile.write(data)

        def _read_json_body(self) -> Dict[str, Any]:
            length = int(self.headers.get("Content-Length") or "0")
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            try:
                parsed = json.loads(raw.decode("utf-8"))
            except Exception as exc:
                raise ValueError(f"请求体 JSON 格式错误：{exc}") from exc
            if not isinstance(parsed, dict):
                raise ValueError("请求体必须是 JSON 对象")
            return parsed

        def do_GET(self) -> None:  # noqa: N802
            parsed = urllib.parse.urlsplit(self.path)
            path = parsed.path

            if path == "/api/health":
                self._send_json(HTTPStatus.OK, job_manager.health())
                return

            if path.startswith("/api/jobs/"):
                parts = [p for p in path.split("/") if p]
                if len(parts) == 3 and parts[0] == "api" and parts[1] == "jobs":
                    try:
                        self._send_json(HTTPStatus.OK, job_manager.get_job(parts[2]))
                    except JobNotFoundError:
                        self._send_json(
                            HTTPStatus.NOT_FOUND,
                            {"ok": False, "errorCode": "JOB_NOT_FOUND", "message": f"未找到任务 {parts[2]}"},
                        )
                    return

            if path == "/" or path == "/index.html":
                self._send_file(WEB_DIR / "index.html")
                return

            if path.startswith("/player/"):
                rel = path[len("/player/") :]
                target = _safe_resolve_under(PLAYER_DIR, rel)
                if target is None:
                    self._send_json(HTTPStatus.FORBIDDEN, {"ok": False, "errorCode": "FORBIDDEN", "message": "非法路径"})
                    return
                self._send_file(target)
                return

            if path.startswith("/jobs/"):
                rel = path[len("/jobs/") :]
                target = _safe_resolve_under(job_manager.work_dir, rel)
                if target is None:
                    self._send_json(HTTPStatus.FORBIDDEN, {"ok": False, "errorCode": "FORBIDDEN", "message": "非法路径"})
                    return
                self._send_file(target)
                return

            target = _safe_resolve_under(WEB_DIR, path.lstrip("/"))
            if target is not None and target.is_file():
                self._send_file(target)
                return

            self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "errorCode": "NOT_FOUND", "message": f"未知路径：{path}"})

        def do_POST(self) -> None:  # noqa: N802
            parsed = urllib.parse.urlsplit(self.path)
            path = parsed.path
            try:
                body = self._read_json_body()
                if path == "/api/jobs":
                    res = job_manager.create_job(body)
                    self._send_json(HTTPStatus.OK, res)
                    return

                parts = [p for p in path.split("/") if p]
                if len(parts) == 4 and parts[0] == "api" and parts[1] == "jobs":
                    job_id, action = parts[2], parts[3]
                    if action == "continue":
                        self._send_json(HTTPStatus.OK, job_manager.continue_job(job_id))
                        return
                    if action == "retry":
                        self._send_json(HTTPStatus.OK, job_manager.retry_job(job_id, body))
                        return
                    if action == "discard":
                        self._send_json(HTTPStatus.OK, job_manager.discard_job(job_id))
                        return
                    if action == "adjust":
                        self._send_json(HTTPStatus.OK, job_manager.adjust_job(job_id, body))
                        return
                    if action == "export":
                        self._send_json(HTTPStatus.OK, job_manager.export_job(job_id, body))
                        return

                self._send_json(
                    HTTPStatus.NOT_FOUND,
                    {"ok": False, "errorCode": "NOT_FOUND", "message": f"未知接口：{path}"},
                )
            except StudioBusyError as exc:
                self._send_json(
                    HTTPStatus.CONFLICT,
                    {
                        "ok": False,
                        "errorCode": "STUDIO_BUSY",
                        "message": str(exc),
                        "active_job_id": exc.active_job_id,
                    },
                )
            except JobNotFoundError as exc:
                self._send_json(
                    HTTPStatus.NOT_FOUND,
                    {"ok": False, "errorCode": "JOB_NOT_FOUND", "message": f"未找到任务：{exc.args[0]}"},
                )
            except (InvalidJobStateError, ValueError) as exc:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"ok": False, "errorCode": "INVALID_REQUEST", "message": str(exc)},
                )
            except Exception as exc:
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"ok": False, "errorCode": "INTERNAL_ERROR", "message": str(exc)},
                )

    server = ThreadingHTTPServer((host, port), StudioRequestHandler)
    server.daemon_threads = True
    server.job_manager = job_manager  # type: ignore[attr-defined]
    return server


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Portrait Spine Studio 本地离线工具服务")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址（默认 127.0.0.1）")
    parser.add_argument("--port", type=int, default=8765, help="监听端口（默认 8765）")
    parser.add_argument("--work-dir", default=str(REPO_ROOT / ".studio-work"), help="任务工作目录")
    args = parser.parse_args(argv)

    server = create_studio_server(host=args.host, port=args.port, work_dir=args.work_dir)
    actual_port = server.server_address[1]
    print(f"Portrait Spine Studio 已启动：http://{args.host}:{actual_port}/", flush=True)
    print(f"任务工作目录：{Path(args.work_dir).resolve()}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n正在关闭 Portrait Spine Studio...", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
