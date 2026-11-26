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
- `POST /api/jobs/<job_id>/refine-mask`
"""
from __future__ import annotations

import argparse
import base64
import copy
import datetime
import hashlib
import json
import math
import mimetypes
import os
import re
import subprocess
import sys
import threading
import time
import urllib.parse
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from src.export.export_profile import export_job_with_profile, normalize_export_profile
from src.export.exporter import export_default_package, export_transparent_web, export_standalone_html
from src.studio.batch_manager import (
    BatchItemLockedError,
    BatchItemNotFoundError,
    BatchManager,
    BatchNotFoundError,
    export_catalog_metadata,
)
from src.gates.audit import GateFailureError, run_audit
from src.mivo.adapter import (
    FileBackedMivoAdapter,
    MivoCanvasAdapter,
    MivoCanvasError,
    StageRunner,
    TransparentInputAdapter,
)
from src.pipeline.build_rig import build_rig
from src.pipeline.preprocess import MAX_LONG_EDGE, MAX_LONG_EDGE_4K, preprocess_portrait
from src.pipeline.presets import list_presets
from src.pipeline.split_parts import split_parts

from .auto_spec import BONE_ROLE_COLORS, apply_rig_edits, build_default_specs, sanitize_character_name
from .nl_adjust import apply_spec_patch, classify_and_polish_adjustment
from .vision_provider import CindyVisionProvider, ConversationHandoffProvider, VisionProviderError

REPO_ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = Path(__file__).resolve().parent / "web"

_CANVAS_ROLES = {
    'head': '头部', 'face': '面部', 'torso': '躯干', 'body': '身体', 'pelvis': '腰胯',
    'weapon': '武器', 'prop': '悬浮道具', 'arm': '手臂', 'upper_arm': '上臂', 'forearm': '前臂',
    'hand': '手部', 'upper_leg': '大腿', 'lower_leg': '小腿', 'leg': '腿部', 'foot': '脚部',
    'hair': '头发', 'cloth': '布料', 'cape': '披风', 'soft_accessory': '软饰件', 'tail': '尾部',
    'armor': '护甲', 'effect_group': '特效', 'fin': '鳍饰', 'fish_tail': '鱼尾',
}


def _canvas_label(part, module):
    for label in (part.get('label'), module.get('label')):
        if isinstance(label, str) and label.strip():
            return label.strip()[:60]
    tokens = set(re.split(r'[_-]+', module['id'].lower()))
    side = {'L': '左', 'R': '右'}.get(module.get('side'), '')
    if not side:
        side = '左' if tokens & {'l', 'left'} else '右' if tokens & {'r', 'right'} else ''
    role = module.get('role')
    if role in ('cloth', 'cape', 'soft_accessory'):
        if 'scarf' in tokens:
            return side + ('围巾飘尾' if 'tail' in tokens else '领口围巾' if 'neck' in tokens else '围巾')
        if tokens & {'flap', 'skirt', 'coat'}:
            return side + ('侧后衣摆' if side and 'back' in tokens else '后衣摆' if 'back' in tokens else '前衣摆' if 'front' in tokens else '衣摆')
    return side + _CANVAS_ROLES.get(role, '饰件')


def _canvas_workbench_files(destination, parts, spec, spine_file):
    """Expose an editable rig without adding disabled slots to the exported rig."""
    rig_dir = destination / 'rig'
    spine = json.loads((rig_dir / spine_file).read_text())
    built = json.loads((rig_dir / 'rig-build-report.json').read_text())
    by_slot = {slot['name']: slot for slot in spine['slots']}
    attachments = {entry['part']: entry['bone'] for entry in built['attachments']}
    modules = {module['id']: module for module in spec['moduleContract']['modules']}
    ratios = spec.get('canvas_motion_ratios', {})
    requests = spec.get('canvas_motion_requests', {})
    animated = spine.get('animations', {}).get('idle', {}).get('bones', {})
    rows = []
    spine['slots'] = []
    for part in sorted(parts, key=lambda piece: piece['drawOrder']):
        name, module_id = part['name'], part['moduleId']
        spine['slots'].append(by_slot.get(name) or {'name': name, 'bone': attachments[name], 'attachment': name})
        module = modules[module_id]
        driver = module
        while driver.get('motionProfile') == 'follow' and driver.get('parent') in modules:
            driver = modules[driver['parent']]
        has_motion = any(bone in animated for bone in [driver['bone'], *driver.get('softBones', [])])
        rows.append({'name': name, 'module': module_id, 'drawOrder': part['drawOrder'],
                     'enabled': part.get('enabled', True), 'role': module['role'], 'side': module.get('side'),
                     'motion_type': module['motionType'], 'label': _canvas_label(part, module),
                     'motion_scale': module.get('motion_scale', 1.), 'driver_module': driver['id'],
                     'motion_adjustable': has_motion and driver.get('motion_scale', 1.) > 0,
                     'motion_ratio': requests.get(driver['id'], 1.), 'baked_motion_ratio':ratios.get(driver['id'], 1.),
                     'baked_motion_scale': driver.get('motion_scale', 1.) * ratios.get(driver['id'], 1.)})
    (rig_dir / 'live.spine.json').write_text(json.dumps(spine, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    return rows
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

    def __init__(self, message: str, http_status: int = 400, code: Optional[str] = None):
        super().__init__(message)
        self.http_status = http_status
        self.code = code


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _module_machine_clear(report: Dict[str, Any]) -> bool:
    quality = report.get("quality") or {}
    return quality.get("contractApplicable") is True and all(
        (quality.get(key) or {}).get("status") in ("pass", "not_applicable")
        for key in ("inputOwnership", "bindingMotion", "machineQC"))


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
        if raw.get("module_contract") is not None:
            opts["module_contract"] = copy.deepcopy(raw["module_contract"])
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



def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("%s 必须是有限数字" % label)
    if not math.isfinite(value):
        raise ValueError("%s 必须是有限数字" % label)
    return float(value)


def _as_point_list(raw: Any, label: str) -> list:
    if not isinstance(raw, list):
        raise ValueError("%s 必须是点列表" % label)
    out = []
    for i, pt in enumerate(raw):
        if not isinstance(pt, (list, tuple)) or len(pt) != 2:
            raise ValueError("%s[%d] 必须是长度为 2 的 [x, y]" % (label, i))
        x = _finite_number(pt[0], "%s[%d][0]" % (label, i))
        y = _finite_number(pt[1], "%s[%d][1]" % (label, i))
        out.append([x, y])
    return out


def _as_box(raw: Any) -> list:
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        raise ValueError("box 必须是 4 个数字")
    box = [_finite_number(v, "box[%d]" % i) for i, v in enumerate(raw)]
    x1, y1, x2, y2 = box
    if not (x2 > x1 and y2 > y1):
        raise ValueError("box 必须满足 x2 > x1 且 y2 > y1")
    return box

def _initial_stages() -> List[Dict[str, str]]:
    return [
        {"id": "preprocess", "label": "输入预处理（<=2400px 缩放）", "status": "pending"},
        {"id": "mivo", "label": "透明输入 / 抠图结果校验", "status": "pending"},
        {"id": "vision", "label": "当前 Cindy 对话规划完整模块", "status": "pending"},
        {"id": "split_parts", "label": "完整模块分件与遮挡校验", "status": "pending"},
        {"id": "bind_rig", "label": "骨骼绑定效果预览与编辑", "status": "pending"},
        {"id": "build_rig", "label": "保形轻摆与软模块轻弯", "status": "pending"},
        {"id": "audit", "label": "模块、接触与运动质量检查", "status": "pending"},
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
        vision_provider: Optional[Any] = None,
        vision_connection_path: Optional[Union[str, Path]] = None,
        refine_runner: Optional[Callable[..., Any]] = None,
    ):
        self.work_dir = Path(work_dir).resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.mivo_handoff_dir = Path(mivo_handoff_dir).resolve() if mivo_handoff_dir else None
        self.audit_samples = audit_samples
        self.stress_samples = stress_samples
        self.vision_provider = vision_provider if vision_provider is not None else ConversationHandoffProvider()
        self.refine_runner = refine_runner

        self._exec_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._active_job_id: Optional[str] = None
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self.batch_manager = BatchManager(self.work_dir)
        self._batch_export_lock = threading.Lock()
        self._batch_export_states: Dict[str, Dict[str, Any]] = {}
        self._restore_persisted_jobs()

    def health(self) -> Dict[str, Any]:
        from src.gates.module_quality import RULES_VERSION
        with self._state_lock:
            return {
                "ok": True,
                "busy": self._exec_lock.locked(),
                "active_job_id": self._active_job_id,
                "work_dir": str(self.work_dir),
                "presets": list_presets(),
                "vision_provider": self.vision_provider.source,
                "analysis_contract_version": getattr(self.vision_provider, "required_contract_version", 2),
                "module_qc_rules_version": RULES_VERSION,
                "legacy_diagnostic_mode": bool(getattr(self.vision_provider, "diagnostic_mode", False)),
            }

    def get_job(self, job_id: str) -> Dict[str, Any]:
        with self._state_lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise JobNotFoundError(job_id)
            return self._public_view(job)

    def _public_view(self, job: Dict[str, Any]) -> Dict[str, Any]:
        contract = job.get("module_contract") or {}
        review_images = {}
        semantic_binding = None
        manifest_path = Path(job["job_dir"]) / "review-modules/module-review.json"
        if contract.get("version") == 2 and manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                from .module_contract import contract_fingerprint
                current_source_sha = hashlib.sha256((Path(job["job_dir"]) / "matted.png").read_bytes()).hexdigest()
                current_analysis_sha = hashlib.sha256((Path(job["job_dir"]) / "vision/vision.json").read_bytes()).hexdigest()
                if (manifest.get("contract_sha256") == contract_fingerprint(contract)
                        and manifest.get("source_sha256") == contract.get("input_sha256") == current_source_sha
                        and manifest.get("analysis_sha256") == current_analysis_sha):
                    review_images = {item["id"]: self._job_url(job["job_id"], "review-modules/" + item["id"] + "-comparison.jpg")
                                     for item in manifest.get("modules", [])
                                     if item.get("id") in {m["id"] for m in contract["modules"]}
                                     and (manifest_path.parent / (item["id"] + "-comparison.jpg")).is_file()}
            except (OSError, ValueError, KeyError, TypeError):
                pass  # Missing/stale review images never count as completed visual review.
            try:
                from src.gates.review_binding import module_review_binding
                semantic_binding = module_review_binding(job["job_dir"], contract)
            except (OSError, ValueError, KeyError, TypeError):
                pass
        modules = [{
            "id": module.get("id"), "role": module.get("role"),
            "motion_type": module.get("motion_type"), "parent": module.get("parent"),
            "motion": module.get("motion"), "pivot": module.get("pivot"),
            "piece_count": len(module.get("pieces") or []),
            "static_reason": module.get("static_reason"),
            "comparison_url": review_images.get(module.get("id")),
        } for module in contract.get("modules", [])]
        return {
            "canvas_mode": job.get("canvas_mode", False),
            "canvas_layers": copy.deepcopy(job.get("canvas_layers", [])),
            "canvas_checks": copy.deepcopy(job.get("canvas_checks", {})),
            "canvas_diagnostics": copy.deepcopy(job.get("canvas_diagnostics", {})),
            "canvas_preview_html": job.get("canvas_preview_html"),
            "canvas_seam_preview": job.get("canvas_seam_preview"),
            "canvas_live_rig_url": job.get("canvas_live_rig_url"),
            "canvas_global_motion_scale": job.get("canvas_global_motion_scale", 1.),
            "canvas_diagnostics_url": job.get("canvas_diagnostics_url"),
            "canvas_all_layers_url": job.get("canvas_all_layers_url"),
            "locked": bool(job.get("locked", False)),
            "locked_at": job.get("locked_at"),
            "locked_revision": job.get("locked_revision"),
            "locked_rig_version": job.get("locked_rig_version"),
            "max_long_edge": int(job.get("max_long_edge") or MAX_LONG_EDGE),
            "batch_id": job.get("batch_id"),
            "job_id": job["job_id"],
            "character": job["character"],
            "filename": job["filename"],
            "status": job["status"],
            "preset": job["preset"],
            "gain": job["gain"],
            "pause_after_split": job["pause_after_split"],
            "pause_before_animation": job.get("pause_before_animation", False),
            "pose_type": (job.get("rig_spec") or {}).get("pose_type", "grounded"),
            "matted_url": job.get("matted_url"),
            "bone_count": len((job.get("rig_spec") or {}).get("bones", [])),
            "part_names": [p["name"] for p in (job.get("parts_spec") or {}).get("parts", [])] if isinstance(job.get("parts_spec"), dict) else [],
            "created_at": job["created_at"],
            "updated_at": job["updated_at"],
            "stages": copy.deepcopy(job["stages"]),
            "resize_notice": job["resize_notice"],
            "warnings": copy.deepcopy(job.get("warnings") or []),
            "semantic_source": job.get("semantic_source", "pending"),
            "vision_metadata": copy.deepcopy(job.get("vision_metadata") or {}),
            "analysis_request": copy.deepcopy(job.get("analysis_request")),
            "module_contract_version": contract.get("version"),
            "module_review_binding": semantic_binding,
            "module_review": copy.deepcopy(job.get("module_review")),
            "modules": modules,
            "connections": copy.deepcopy(contract.get("connections", [])),
            "occlusions": [{key: item.get(key) for key in ("front", "back", "completion")}
                           for item in contract.get("occlusions", [])],
            "analysis_issues": copy.deepcopy(job.get("analysis_issues") or []),
            "visual_review": copy.deepcopy(job.get("visual_review")),
            "review_binding": copy.deepcopy(job.get("review_binding")),
            "quality": copy.deepcopy((job.get("health_report") or {}).get("quality")),
            "matting_source": job.get("matting_source", "pending"),
            "matting_metadata": copy.deepcopy(job.get("matting_metadata") or {}),
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
            "rig_mask_overlay_url": self._rig_mask_overlay_url(job),
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
            raise MivoCanvasError("Injected Mivo auto_matting failure", code="MIVO_SERVICE_UNAVAILABLE")
        if self.mivo_handoff_dir is not None:
            return FileBackedMivoAdapter(self.mivo_handoff_dir)
        return TransparentInputAdapter()

    def _assert_unlocked(self, job: Dict[str, Any]) -> None:
        if job.get("locked"):
            raise InvalidJobStateError(
                "该角色已确认锁定，禁止修改；如需返修请先在界面点击「解锁」。",
                409,
                code="JOB_LOCKED",
            )

    def _restore_persisted_jobs(self) -> None:
        if not self.work_dir.is_dir():
            return
        for child in sorted(self.work_dir.iterdir()):
            if not child.is_dir() or child.name in ("batches", "models"):
                continue
            state_file = child / "job-state.json"
            status_file = child / "job-status.json"
            try:
                if state_file.is_file():
                    data = json.loads(state_file.read_text(encoding="utf-8"))
                    jid = str(data.get("job_id") or child.name)
                    data["job_id"] = jid
                    data["job_dir"] = str(child)
                    data.setdefault("locked", False)
                    data.setdefault("rig_build_count", 0)
                    data.setdefault("split_run_count", 0)
                    data.setdefault("adjustment_log", [])
                    data.setdefault("exports", {})
                    data.setdefault("warnings", [])
                    data.setdefault("stages", _initial_stages())
                    data.setdefault("preserved_intermediates", {})
                    self._jobs[jid] = data
                elif status_file.is_file():
                    pub = json.loads(status_file.read_text(encoding="utf-8"))
                    jid = str(pub.get("job_id") or child.name)
                    pub["job_id"] = jid
                    pub["job_dir"] = str(child)
                    pub.setdefault("input_path", str(child / "input" / str(pub.get("filename") or "portrait.png")))
                    pub.setdefault("locked", False)
                    pub.setdefault("rig_build_count", 0)
                    pub.setdefault("split_run_count", 0)
                    pub.setdefault("adjustment_log", [])
                    pub.setdefault("exports", {})
                    pub.setdefault("warnings", [])
                    pub.setdefault("stages", _initial_stages())
                    pub.setdefault("preserved_intermediates", {})
                    if pub.get("canvas_mode") and not pub.get("canvas_output"):
                        revs = sorted(child.glob("canvas-r*"), key=lambda p: int(p.name.split("-r")[-1]) if p.name.split("-r")[-1].isdigit() else 0)
                        if revs:
                            pub["canvas_output"] = str(revs[-1])
                            pub.setdefault("canvas_revision", len(revs))
                    self._jobs[jid] = pub
            except Exception:
                continue

    def _acquire_serial(self, job_id: str, *, wait: bool = False, timeout: float = 180.0) -> None:
        acquired = self._exec_lock.acquire(blocking=True, timeout=timeout) if wait else self._exec_lock.acquire(blocking=False)
        if not acquired:
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
        pub = self._public_view(job)
        (job_dir / "job-status.json").write_text(
            json.dumps(pub, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        state = {
            k: v for k, v in job.items()
            if not k.startswith("_")
        }
        tmp_state = job_dir / "job-state.json.tmp"
        tmp_state.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(tmp_state, job_dir / "job-state.json")

    def _review_binding(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """Bind a visual decision to the exact accepted plan and generated image/rig files."""
        from src.gates.review_binding import actual_review_binding
        job_dir = Path(job["job_dir"])
        rig_dir = job_dir / "rig"
        plan_path = job_dir / "vision/vision.json"
        plan_bytes = plan_path.read_bytes()
        if _canonical_digest(json.loads(plan_bytes)) != _canonical_digest(job.get("module_contract")):
            raise InvalidJobStateError("磁盘模块规划与本次已接受分析不一致，请重新提交分析。")
        return actual_review_binding(rig_dir, source_image=job_dir / "matted.png",
                                     analysis_file=plan_path, job_id=job["job_id"])

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

        order = ["preprocess", "mivo", "vision", "split_parts", "bind_rig", "build_rig", "audit"]
        start_idx = order.index(start_from)

        # Invalidate acceptance before a rebuild can fail. Preserve the prior report
        # as history, while both the HTTP view and low-level exporters see it revoked.
        previous_report = rig_dir / "health-report.json"
        if previous_report.is_file():
            from src.gates.module_quality import recompute_quality
            raw = previous_report.read_bytes()
            history = job_dir / "history"
            history.mkdir(parents=True, exist_ok=True)
            archived = history / ("health-report-" + uuid.uuid4().hex + ".json")
            archived.write_bytes(raw)
            runner.record_intermediate("previous_health_report", archived)
            try:
                invalidated = json.loads(raw)
            except (ValueError, UnicodeError):
                invalidated = {"version": "2.0", "quality": {"contractApplicable": False}}
            invalidated["status"] = "needs_review"
            quality = invalidated.setdefault("quality", {})
            quality["visual"] = {"status": "pending", "required": True,
                                 "reasons": ["A new build or audit was requested; prior acceptance is invalidated"]}
            recompute_quality(invalidated)
            invalidated["invalidatedAt"] = _now_iso()
            previous_report.write_text(json.dumps(invalidated, ensure_ascii=False, indent=2), encoding="utf-8")

        with self._state_lock:
            job["status"] = "running"
            job["stage_error"] = None
            job["health_report"] = None
            job["health_report_url"] = None
            job["visual_review"] = None
            job["review_binding"] = None
            job["exports"] = {}
            if start_idx <= order.index("build_rig"):
                job["rig_url"] = None
            for idx, st in enumerate(job["stages"]):
                if idx >= start_idx:
                    st["status"] = "pending"
            job["updated_at"] = _now_iso()

        if simulate_delay_ms > 0:
            time.sleep(min(5.0, float(simulate_delay_ms) / 1000.0))

        def stage_preprocess() -> Dict[str, Any]:
            self._set_stage(job, "preprocess", "running")
            res = preprocess_portrait(input_path, preprocessed_path, max_long_edge=int(job.get("max_long_edge") or MAX_LONG_EDGE))
            with self._state_lock:
                job["preprocess"] = res
                job["resize_notice"] = res.get("notice")
            self._set_stage(job, "preprocess", "done")
            return res

        def stage_mivo() -> Any:
            self._set_stage(job, "mivo", "running")
            try:
                adapter = self._make_mivo_adapter(simulate_error=simulate_mivo_error)
                res = adapter.auto_matting(preprocessed_path, matted_path)
            except MivoCanvasError as exc:
                with self._state_lock:
                    job["matting_source"] = "unavailable"
                    if str(exc) not in job["warnings"]:
                        job["warnings"].append(str(exc))
                raise
            with self._state_lock:
                job["matting_source"] = res.metadata.get("source", "unknown")
                job["matting_metadata"] = copy.deepcopy(res.metadata)
                for stage in job["stages"]:
                    if stage["id"] == "mivo":
                        stage["label"] = ("透明输入校验（无需去底）" if job["matting_source"] == "transparent_input" else "抠图交接结果校验")
                job["matted_url"] = self._job_url(job_id, "matted.png")
            self._set_stage(job, "mivo", "done")
            return res

        def stage_vision() -> Dict[str, Any]:
            from .vision_schema import VisionValidationError, vision_result_to_semantic_hints
            self._set_stage(job, "vision", "running")
            vision_dir = job_dir / "vision"
            vision_dir.mkdir(parents=True, exist_ok=True)
            runner.record_intermediate("vision", vision_dir)
            if job.get("_provided_semantics"):
                if not getattr(self.vision_provider, "diagnostic_mode", False):
                    raise VisionProviderError(
                        "生产流程只接收完整模块 V2。请为当前任务提交模块规划；旧关键点、parts_spec 或 rig_spec 仅可用于显式历史诊断。",
                        code="MODULE_CONTRACT_REQUIRED", retryable=False,
                    )
                # Caller-supplied hints/specs never acquire live-provider provenance.
                joints = (job.get("semantic_hints") or {}).get("joints")
                if not (isinstance(joints, dict) and len(joints) >= 3) and not (job.get("parts_spec") is not None and job.get("rig_spec") is not None):
                    raise VisionProviderError("外部关键点或配置不完整，已停止；不会用几何推断补成自动识别结果。",
                                              code="VISION_PROVIDED_HINTS_INVALID", retryable=False)
                metadata = {
                    "automatically_recognized": False,
                    "input_sha256": hashlib.sha256(matted_path.read_bytes()).hexdigest(),
                    "source": "provided_hints",
                }
                (vision_dir / "provided-input.json").write_text(json.dumps({
                    "metadata": metadata, "semantic_hints": job.get("semantic_hints"),
                    "provided_parts_spec": job.get("parts_spec") is not None,
                    "provided_rig_spec": job.get("rig_spec") is not None,
                }, ensure_ascii=False, indent=2), encoding="utf-8")
                with self._state_lock:
                    job["semantic_source"] = "provided_hints"
                    job["vision_metadata"] = metadata
            elif isinstance(self.vision_provider, ConversationHandoffProvider):
                if job.get("_conversation_analysis") is None:
                    request = self.vision_provider.prepare(job_id=job_id, image_path=matted_path,
                                                           out_dir=vision_dir, preset=job["preset"])
                    with self._state_lock:
                        job["analysis_request"] = request
                        job["semantic_source"] = "conversation_handoff"
                        job["vision_metadata"] = {"application_invoked_model": False, "analysis_applied": False}
                    self._set_stage(job, "vision", "awaiting")
                    return {"out_dir": str(vision_dir), "awaiting_analysis": True}
                analysis = self.vision_provider.apply(
                    job_id=job_id, image_path=matted_path, out_dir=vision_dir,
                    request=job.get("analysis_request") or {}, payload=job["_conversation_analysis"],
                )
                hints = vision_result_to_semantic_hints(
                    analysis.result, image_path=matted_path, mask_dir=job_dir / "vision_masks")
                from scripts.inspect_module_plan import inspect_plan
                inspect_plan(matted_path, vision_dir / "vision.json", job_dir / "review-modules")
                with self._state_lock:
                    job["semantic_hints"] = hints
                    job["semantic_source"] = "conversation_handoff"
                    job["vision_metadata"] = {**analysis.metadata, "analysis_applied": True}
                    job["module_contract"] = copy.deepcopy((analysis.result or {}) if (analysis.result or {}).get("version") == 2 else None)
                    job["analysis_issues"] = []
                    job["parts_spec"] = None
                    job["rig_spec"] = None
            else:
                def update_vision_status(label: str) -> None:
                    with self._state_lock:
                        for stage in job["stages"]:
                            if stage["id"] == "vision":
                                stage["label"] = label
                        job["updated_at"] = _now_iso()
                analysis = self.vision_provider.analyze(
                    job_id=job_id, image_path=matted_path, out_dir=vision_dir,
                    preset=job["preset"], on_status=update_vision_status,
                )
                if analysis.source not in ("actual_cindy_agent", "geometry"):
                    raise VisionProviderError("视觉提供方来源无效。", code="VISION_SOURCE_INVALID", retryable=False)
                hints = None
                if analysis.source == "actual_cindy_agent":
                    if not isinstance(analysis.result, dict) or analysis.result.get("version") != 2:
                        raise VisionProviderError("当前视觉结果不是完整模块 V2。", code="MODULE_CONTRACT_REQUIRED", retryable=False)
                    try:
                        hints = vision_result_to_semantic_hints(
                            analysis.result, image_path=matted_path, mask_dir=job_dir / "vision_masks",
                        )
                    except VisionValidationError as exc:
                        exc.details = {key: analysis.metadata.get(key) for key in
                                       ("task_id", "run_id", "output_message_id", "input_sha256", "result_path")}
                        raise
                with self._state_lock:
                    job["semantic_hints"] = hints
                    job["semantic_source"] = analysis.source
                    job["vision_metadata"] = analysis.metadata
                    job["module_contract"] = copy.deepcopy(analysis.result if (analysis.result or {}).get("version") == 2 else None)
                    # A fresh live result owns both downstream specs; stale/manual specs cannot override it.
                    job["parts_spec"] = None
                    job["rig_spec"] = None
            self._set_stage(job, "vision", "done")
            return {"out_dir": str(vision_dir), "source": job["semantic_source"]}

        def stage_split() -> Dict[str, Any]:
            self._set_stage(job, "split_parts", "running")
            if job["parts_spec"] is None or job["rig_spec"] is None:
                auto_parts, auto_rig = build_default_specs(
                    matted_path,
                    character=job["character"],
                    preset=job["preset"],
                    mask_dir=job_dir / "parts_masks",
                    semantic_hints=job.get("semantic_hints"),
                    legacy_diagnostic=bool(getattr(self.vision_provider, "diagnostic_mode", False)),
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

        def stage_bind_rig() -> Dict[str, Any]:
            self._set_stage(job, "bind_rig", "running")
            if not (stop_for_binding and not job.get("_bypass_pause")):
                self._set_stage(job, "bind_rig", "done")
                return {"skipped_preview_bind": True}
            res = build_rig(
                parts_dir,
                job["rig_spec"],
                rig_dir,
                preset=job["preset"],
                gain=float(job["gain"]),
                include_animation=False,
            )
            spine_file = res["report"]["spineFile"]
            with self._state_lock:
                job["rig_url"] = self._job_url(job_id, f"rig/{spine_file}")
            self._set_stage(job, "bind_rig", "done")
            return res

        def stage_build_rig() -> Dict[str, Any]:
            self._set_stage(job, "build_rig", "running")
            if (job.get("module_contract") or {}).get("version") == 2:
                from src.gates.review_binding import module_review_binding, validate_module_review
                validate_module_review(job.get("module_review") or {},
                                       module_review_binding(job_dir, job["module_contract"]), job["module_contract"])
            res = build_rig(
                parts_dir,
                job["rig_spec"],
                rig_dir,
                preset=job["preset"],
                gain=float(job["gain"]),
                include_animation=True,
            )
            spine_file = res["report"]["spineFile"]
            with self._state_lock:
                job["rig_build_count"] += 1
                job["rig_url"] = self._job_url(job_id, f"rig/{spine_file}")
                job["visual_review"] = None
                job["exports"] = {}
            if (job.get("module_contract") or {}).get("version") == 2:
                job["review_binding"] = self._review_binding(job)
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
            "vision": stage_vision,
            "split_parts": stage_split,
            "bind_rig": stage_bind_rig,
            "build_rig": stage_build_rig,
            "audit": stage_audit,
        }

        stop_for_binding = bool(job.get("pause_before_animation")) and start_from in (
            "preprocess",
            "mivo",
            "vision",
            "split_parts",
            "bind_rig",
        )
        stop_only_split = bool(stop_after_split) and not stop_for_binding and start_from in (
            "preprocess",
            "mivo",
            "vision",
            "split_parts",
        )
        stop_for_analysis = (isinstance(self.vision_provider, ConversationHandoffProvider)
                             and not job.get("_provided_semantics")
                             and job.get("_conversation_analysis") is None
                             and start_idx <= order.index("vision"))
        if stop_for_analysis:
            end_idx = order.index("vision") + 1
        elif stop_only_split and not job.get("_bypass_pause"):
            end_idx = order.index("split_parts") + 1
        elif stop_for_binding and not job.get("_bypass_pause"):
            end_idx = order.index("bind_rig") + 1
        else:
            end_idx = len(order)
        plan = [(name, stage_fns[name]) for name in order[start_idx:end_idx]]

        outcome = runner.run_stages(plan, raise_on_error=False)
        with self._state_lock:
            job["preserved_intermediates"] = dict(runner.intermediates)
            if outcome.get("status") == "failed":
                failed_stage = outcome["failedStage"]
                pending_quality = failed_stage == "audit" and (outcome.get("details") or {}).get("status") == "needs_review"
                for st in job["stages"]:
                    if st["id"] == failed_stage:
                        st["status"] = "awaiting" if pending_quality else "failed"
                job["status"] = "needs_review" if pending_quality else "failed"
                job["stage_error"] = outcome
                if failed_stage == "audit" and isinstance(outcome.get("details"), dict):
                    job["health_report"] = outcome["details"]
                    job["health_report_url"] = self._job_url(job_id, "rig/health-report.json")
                if failed_stage == "vision":
                    submitted = (job.get("_conversation_analysis") or {}).get("analysis") or {}
                    issues = submitted.get("uncertainties") if isinstance(submitted, dict) else []
                    job["analysis_issues"] = (issues if isinstance(issues, list) else []) + [outcome.get("message", "模块规划需要复核。")]
            elif stop_for_analysis:
                job["status"] = "awaiting_analysis"
            elif stop_only_split and not job.get("_bypass_pause"):
                job["status"] = "awaiting_split_confirm"
            elif stop_for_binding and not job.get("_bypass_pause"):
                job["status"] = "awaiting_rig_confirm"
            else:
                quality = (job.get("health_report") or {}).get("quality") or {}
                if (job.get("module_contract") or {}).get("version") == 2:
                    job["status"] = ("ready" if quality.get("overall", {}).get("exportAllowed") is True else
                                     "awaiting_visual_review" if _module_machine_clear(job.get("health_report") or {}) else "needs_review")
                else:
                    job["status"] = "needs_review"  # Legacy diagnostics never imply complete-module production approval.
            job["updated_at"] = _now_iso()

        self._persist_job_meta(job)
        return self.get_job(job_id)

    def create_job(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        filename = str(payload.get("filename") or "portrait.png")
        image_bytes = _decode_image_base64(payload.get("image_base64", ""))
        preset = str(payload.get("preset") or "agile_full")
        if preset not in list_presets():
            raise ValueError(f"未知预设 {preset!r}，可选：{list_presets()}")
        semantic_hints = payload.get("semantic_hints")
        if semantic_hints is not None and not isinstance(semantic_hints, dict):
            raise ValueError("关键点输入无效：semantic_hints 必须是对象。")
        provided_semantics = (semantic_hints is not None or payload.get("parts_spec") is not None
                              or payload.get("rig_spec") is not None)
        semantic_source = "provided_hints" if provided_semantics else ("conversation_handoff" if isinstance(self.vision_provider, ConversationHandoffProvider) else "pending")
        gain = float(payload.get("gain", 1.0))
        pause_after_split = bool(payload.get("pause_after_split", False))
        pause_before_animation = bool(payload.get("pause_before_animation", True))
        simulate_mivo_error = bool(payload.get("simulate_mivo_error", False))
        simulate_delay_ms = int(payload.get("simulate_delay_ms", 0))
        raw_edge = payload.get("max_long_edge")
        max_long_edge = MAX_LONG_EDGE if raw_edge is None else int(raw_edge)
        if not (64 <= max_long_edge <= 8192):
            raise ValueError("max_long_edge 必须在 64..8192 像素之间。")
        wait_if_busy = bool(payload.get("wait_if_busy", False))
        batch_id = str(payload.get("batch_id")).strip() if payload.get("batch_id") else None

        job_id = uuid.uuid4().hex[:12]
        self._acquire_serial(job_id, wait=wait_if_busy)
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
                "max_long_edge": max_long_edge,
                "batch_id": batch_id,
                "locked": False,
                "locked_at": None,
                "locked_revision": None,
                "locked_rig_version": None,
                "pause_after_split": pause_after_split,
                "pause_before_animation": pause_before_animation,
                "created_at": now,
                "updated_at": now,
                "stages": _initial_stages(),
                "resize_notice": None,
                "warnings": [],
                "preprocess": None,
                "health_report": None,
                "rig_url": None,
                "health_report_url": None,
                "part_map_url": None,
                "matted_url": None,
                "stage_error": None,
                "preserved_intermediates": {"input": str(input_path)},
                "adjustment_log": [],
                "exports": {},
                "parts_spec": copy.deepcopy(payload.get("parts_spec")),
                "rig_spec": copy.deepcopy(payload.get("rig_spec")),
                "semantic_hints": copy.deepcopy(semantic_hints),
                "semantic_source": semantic_source,
                "vision_metadata": {},
                "analysis_request": None,
                "module_contract": None,
                "analysis_issues": [],
                "visual_review": None,
                "review_binding": None,
                "analysis_sha256": None,
                "_conversation_analysis": None,
                "_provided_semantics": provided_semantics,
                "matting_source": "pending",
                "matting_metadata": {},
                "split_run_count": 0,
                "rig_build_count": 0,
                "_bypass_pause": False,
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

    def get_rig_spec(self, job_id: str) -> Dict[str, Any]:
        with self._state_lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise JobNotFoundError(job_id)
            rs = copy.deepcopy(job.get("rig_spec") or {})
            ps = copy.deepcopy(job.get("parts_spec") or {})
            job_dir = Path(job["job_dir"])
            parts_json_path = job_dir / "parts_out" / "parts.json"
            canvas = [0, 0]
            parts_meta = []
            if parts_json_path.is_file():
                pj = json.loads(parts_json_path.read_text(encoding="utf-8"))
                canvas = pj.get("canvas", [0, 0])
                parts_meta = pj.get("parts", [])
            return {
                "ok": True,
                "job_id": job_id,
                "status": job["status"],
                "canvas": canvas,
                "pose_type": rs.get("pose_type", "grounded"),
                "bones": rs.get("bones", []),
                "chains": rs.get("chains", {}),
                "ik": rs.get("ik", []),
                "skin": rs.get("skin", {}),
                "module_contract_version": (job.get("module_contract") or {}).get("version"),
                "modules": self._public_view(job)["modules"],
                "connections": copy.deepcopy((job.get("module_contract") or {}).get("connections", [])),
                "occlusions": self._public_view(job)["occlusions"],
                "parts": [
                    {
                        "name": p["name"],
                        "kind": p.get("kind", "soft"),
                        "bbox": p.get("bbox"),
                        "pivot": p.get("pivot"),
                    }
                    for p in parts_meta
                ],
                "role_colors": dict(BONE_ROLE_COLORS),
                "matted_url": job.get("matted_url"),
                "part_map_url": job.get("part_map_url"),
                "rig_url": job.get("rig_url"),
                "rig_mask_overlay_url": self._rig_mask_overlay_url(job),
            }

    def rebind_job(self, job_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Update bones/skin/pose_type and re-run bind_rig to refresh binding preview before animation."""
        with self._state_lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise JobNotFoundError(job_id)
            self._assert_unlocked(job)
            if job["status"] not in ("awaiting_rig_confirm", "awaiting_split_confirm", "ready", "failed", "needs_review", "awaiting_visual_review"):
                raise InvalidJobStateError(f"当前状态 {job['status']} 不支持重新绑定。")
            if job.get("rig_spec") is None:
                raise InvalidJobStateError("任务尚未生成骨骼配置，无法重新绑定。")

        self._acquire_serial(job_id)
        try:
            with self._state_lock:
                job["rig_spec"] = apply_rig_edits(job["rig_spec"], payload)
                job["pause_before_animation"] = True
                job["_bypass_pause"] = False
            return self._run_pipeline_stages(job, start_from="bind_rig", stop_after_split=False)
        finally:
            self._release_serial()

    def _rig_mask_overlay_url(self, job: Dict[str, Any]) -> Optional[str]:
        overlay = Path(job["job_dir"]) / "rig" / "review" / "rig_mask_overlay.png"
        if overlay.is_file():
            return self._job_url(job["job_id"], "rig/review/rig_mask_overlay.png")
        return None

    def _initial_segment_spec(self, job: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        conv = job.get("_conversation_analysis")
        if isinstance(conv, dict):
            if isinstance(conv.get("analysis"), dict):
                return copy.deepcopy(conv["analysis"])
            if conv.get("version") == 2:
                return copy.deepcopy(conv)
        path = Path(job["job_dir"]) / "analysis.json"
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                data = None
            if isinstance(data, dict):
                if isinstance(data.get("analysis"), dict):
                    return copy.deepcopy(data["analysis"])
                return data
        return None

    def _default_refine_runner(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        py = REPO_ROOT / ".venv" / "bin" / "python3"
        if not py.is_file():
            py = Path(sys.executable)
        cmd = [
            str(py),
            str(REPO_ROOT / "scripts" / "segment_modules_sam.py"),
            "refine-one",
            "--image", str(ctx["image"]),
            "--spec", str(ctx["spec"]),
            "--out-dir", str(ctx["out_dir"]),
            "--target-id", str(ctx["target_id"]),
            "--override-json", str(ctx["override_path"]),
        ]
        if ctx.get("no_cache"):
            cmd.append("--no-cache")
        env = os.environ.copy()
        env.setdefault("YOLO_CONFIG_DIR", str(REPO_ROOT / ".studio-work" / "models"))
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120, cwd=str(REPO_ROOT), env=env,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "message": "refine-one 超时", "errors": [{"code": "timeout"}]}
        payload = None
        for line in reversed((proc.stdout or "").splitlines()):
            line = line.strip()
            if line.startswith("{"):
                try:
                    payload = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue
        if not isinstance(payload, dict):
            return {
                "ok": False,
                "message": (proc.stderr or "").strip() or "refine-one 无 JSON 输出",
                "errors": [{"code": "runner_error", "returncode": proc.returncode}],
            }
        return payload

    def refine_mask_job(self, job_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("refine-mask 请求必须是 JSON 对象")
        allowed = {"target_id", "pos", "neg", "box", "roi_mode", "spec", "no_cache"}
        unknown = sorted(k for k in payload.keys() if k not in allowed)
        if unknown:
            raise ValueError("refine-mask 含未知字段: %s" % ", ".join(unknown))
        target_id = payload.get("target_id")
        if not isinstance(target_id, str) or not target_id.strip():
            raise ValueError("target_id 必填且必须是非空字符串")
        target_id = target_id.strip()
        if (".." in target_id or "/" in target_id or "\\" in target_id
                or not re.fullmatch(r"[A-Za-z0-9_.-]+", target_id)):
            raise ValueError("target_id 含非法路径或字符")
        override: Dict[str, Any] = {}
        if "pos" in payload:
            override["pos"] = _as_point_list(payload.get("pos"), "pos")
        if "neg" in payload:
            override["neg"] = _as_point_list(payload.get("neg"), "neg")
        if "box" in payload:
            override["box"] = _as_box(payload.get("box"))
        if "roi_mode" in payload:
            roi_mode = payload.get("roi_mode")
            if roi_mode not in ("auto", "full"):
                raise ValueError("roi_mode 只允许 auto 或 full")
            override["roi_mode"] = roi_mode
        if "no_cache" in payload and not isinstance(payload.get("no_cache"), bool):
            raise ValueError("no_cache 必须是布尔值")
        if "spec" in payload and not isinstance(payload.get("spec"), dict):
            raise ValueError("spec 必须是对象")
        with self._state_lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise JobNotFoundError(job_id)
            self._assert_unlocked(job)
            if job["status"] not in ("awaiting_analysis", "awaiting_rig_confirm"):
                raise InvalidJobStateError(
                    "当前状态 %s 不支持分件点选修正。" % job["status"], http_status=409)
        self._acquire_serial(job_id)
        try:
            with self._state_lock:
                job = self._jobs.get(job_id)
                if job is None:
                    raise JobNotFoundError(job_id)
                if job["status"] not in ("awaiting_analysis", "awaiting_rig_confirm"):
                    raise InvalidJobStateError(
                        "当前状态 %s 不支持分件点选修正。" % job["status"], http_status=409)
                job_dir = Path(job["job_dir"])
            refine_dir = job_dir / "refine_work"
            refine_dir.mkdir(parents=True, exist_ok=True)
            spec_path = refine_dir / "segment_spec.json"
            if "spec" in payload:
                spec_path.write_text(json.dumps(payload["spec"], ensure_ascii=False, indent=2), encoding="utf-8")
            elif spec_path.is_file():
                pass
            else:
                spec = self._initial_segment_spec(job)
                if spec is None:
                    raise ValueError("缺少分件 spec，请在请求中提供 spec")
                spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
            override_path = refine_dir / "override.json"
            override_path.write_text(json.dumps(override, ensure_ascii=False, indent=2), encoding="utf-8")
            matted = job_dir / "matted.png"
            if not matted.is_file():
                raise ValueError("缺少去底图，无法修正分件")
            ctx = {
                "image": matted,
                "spec": spec_path,
                "out_dir": refine_dir,
                "target_id": target_id,
                "override_path": override_path,
                "no_cache": bool(payload.get("no_cache")),
                "job_id": job_id,
            }
            runner = self.refine_runner or self._default_refine_runner
            try:
                summary = runner(ctx)
            except Exception as exc:
                summary = {
                    "ok": False,
                    "analysis_ready": False,
                    "errors": [{
                        "code": "runner_exception",
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }],
                    "message": str(exc),
                }
            if not isinstance(summary, dict):
                summary = {"ok": False, "message": "refine runner 返回非法结果", "errors": [{"code": "runner_error"}]}
            analysis_path = refine_dir / "analysis.json"
            if summary.get("ok") and analysis_path.is_file():
                overlay = refine_dir / "ownership-color-overlay.jpg"
                crops = refine_dir / "module-crops-sheet.jpg"
                return {
                    "ok": True,
                    "analysis_ready": True,
                    "overlay_url": self._job_url(job_id, "refine_work/ownership-color-overlay.jpg") if overlay.is_file() else None,
                    "crops_url": self._job_url(job_id, "refine_work/module-crops-sheet.jpg") if crops.is_file() else None,
                    "modules": summary.get("modules") or [],
                    "target_id": target_id,
                }
            failed_marker = refine_dir / "SEGMENT_FAILED"
            failed_marker.write_text(json.dumps({
                "ok": False,
                "message": summary.get("message") or "分件修正失败",
                "errors": summary.get("errors") or [],
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            return {
                "ok": False,
                "analysis_ready": False,
                "errors": summary.get("errors") or [],
                "message": summary.get("message") or "分件修正失败",
                "overlay_url": None,
                "crops_url": None,
            }
        finally:
            self._release_serial()

    def apply_analysis_job(self, job_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the current conversation's image-bound JSON, then stop for binding review."""
        if not isinstance(payload, dict):
            raise ValueError("模块规划请求必须是 JSON 对象。")
        if not isinstance(self.vision_provider, ConversationHandoffProvider):
            raise InvalidJobStateError("当前服务未使用对话交接模式。")
        with self._state_lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise JobNotFoundError(job_id)
            self._assert_unlocked(job)
            if job["status"] not in ("awaiting_analysis", "awaiting_rig_confirm", "failed", "needs_review", "awaiting_visual_review", "ready"):
                raise InvalidJobStateError(f"当前状态 {job['status']} 不支持提交分析。")
            if not job.get("analysis_request"):
                raise InvalidJobStateError("尚未准备待分析图片，请先完成预处理与去底校验。")
        self._acquire_serial(job_id)
        try:
            if payload.get("from_refined") is True:
                refine_dir = Path(job["job_dir"]) / "refine_work"
                if (refine_dir / "SEGMENT_FAILED").exists():
                    raise InvalidJobStateError(
                        "refine_work 含失败标记，不能应用过期或失败产物", http_status=409)
                analysis_path = refine_dir / "analysis.json"
                if not analysis_path.is_file():
                    raise InvalidJobStateError(
                        "尚未完成分件修正，缺少 refine_work/analysis.json", http_status=409)
                report_path = refine_dir / "segment-report.json"
                if not report_path.is_file():
                    raise InvalidJobStateError(
                        "缺少 refine_work/segment-report.json，不能应用孤立 analysis.json", http_status=409)
                try:
                    report = json.loads(report_path.read_text(encoding="utf-8"))
                except (OSError, ValueError) as exc:
                    raise InvalidJobStateError(
                        "refine_work/segment-report.json 无法读取", http_status=409) from exc
                if not isinstance(report, dict) or report.get("ok") is not True:
                    raise InvalidJobStateError(
                        "refine_work/segment-report.json 未成功，不能应用失败产物", http_status=409)
                try:
                    loaded = json.loads(analysis_path.read_text(encoding="utf-8"))
                except (OSError, ValueError) as exc:
                    raise ValueError("refine_work/analysis.json 无法读取: %s" % exc) from exc
                if not isinstance(loaded, dict):
                    raise ValueError("refine_work/analysis.json 必须是对象")
                if isinstance(loaded.get("analysis"), dict):
                    analysis = copy.deepcopy(loaded["analysis"])
                else:
                    analysis = copy.deepcopy(loaded)
                analysis["job_id"] = job_id
                payload = {"job_id": job_id, "analysis": analysis}
            with self._state_lock:
                incoming_digest = _canonical_digest(payload)
                if (job.get("analysis_sha256") == incoming_digest
                        and job["status"] == "awaiting_rig_confirm"):
                    return self._public_view(job)
                job["_conversation_analysis"] = copy.deepcopy(payload)
                job["_provided_semantics"] = False
                job["pause_before_animation"] = True
                job["_bypass_pause"] = False
                job["vision_metadata"] = {"application_invoked_model": False, "analysis_applied": False}
                job["analysis_sha256"] = incoming_digest
                job["module_contract"] = None
                job["module_review"] = None
                job["parts_spec"] = None
                job["rig_spec"] = None
                job["rig_url"] = None
                job["health_report"] = None
                job["health_report_url"] = None
                job["review_binding"] = None
                job["visual_review"] = None
                job["exports"] = {}
            return self._run_pipeline_stages(job, start_from="vision")
        finally:
            self._release_serial()

    def continue_job(self, job_id: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        with self._state_lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise JobNotFoundError(job_id)
            if job["status"] not in ("awaiting_split_confirm", "awaiting_rig_confirm"):
                raise InvalidJobStateError(
                    f"任务当前状态为 {job['status']}，仅 awaiting_rig_confirm / awaiting_split_confirm 状态可继续生成动画。"
                )

        self._acquire_serial(job_id)
        try:
            if (job.get("module_contract") or {}).get("version") == 2:
                from src.gates.review_binding import module_review_binding, validate_module_review
                binding = module_review_binding(job["job_dir"], job["module_contract"])
                evidence = (payload or {}).get("module_review") or {}
                validate_module_review(evidence, binding, job["module_contract"])
                with self._state_lock:
                    job["module_review"] = copy.deepcopy(evidence)
                (Path(job["job_dir"]) / "module-review-acceptance.json").write_text(
                    json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
            with self._state_lock:
                job["_bypass_pause"] = True
            return self._run_pipeline_stages(job, start_from="build_rig", stop_after_split=False)
        finally:
            self._release_serial()

    def retry_job(self, job_id: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        opts = payload or {}
        with self._state_lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise JobNotFoundError(job_id)
            self._assert_unlocked(job)
            if job["status"] != "failed" or not job["stage_error"]:
                raise InvalidJobStateError(f"任务当前状态为 {job['status']}，仅失败任务可重试当前阶段。")
            failed_stage = str(job["stage_error"].get("failedStage") or "preprocess")
            if opts.get("preset") in list_presets():
                job["preset"] = str(opts["preset"])
            if opts.get("gain") is not None:
                job["gain"] = float(opts["gain"])

        self._acquire_serial(job_id)
        try:
            stop_after_split = bool(job["pause_after_split"]) and failed_stage in ("preprocess", "mivo", "vision", "split_parts")
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
            self._assert_unlocked(job)
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
            self._assert_unlocked(job)
            if job["status"] not in ("ready", "failed", "needs_review", "awaiting_visual_review"):
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

    def review_job(self, job_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Record an explicit visual decision without inventing model or Host attestations."""
        from src.gates.audit import finalize_visual_review
        from src.gates.review_binding import validate_visual_evidence
        decision = payload.get("decision")
        source = payload.get("source")
        notes = str(payload.get("notes") or "").strip()
        reviewer = str(payload.get("reviewer") or "").strip()
        if decision not in ("pass", "fail", "pending") or source not in ("current_cindy_conversation", "user_ui"):
            raise ValueError("视觉复核需要明确的 pass/fail/pending 决定及来源。")
        if not notes or not reviewer:
            raise ValueError("请记录复核者和实际观察结果，不能仅点击空白通过。")
        if decision == "pass" and payload.get("full_loop_observed") is not True:
            raise ValueError("视觉通过需要先实际检查原速完整循环。")
        with self._state_lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise JobNotFoundError(job_id)
            if job["status"] not in ("failed", "needs_review", "awaiting_visual_review", "ready"):
                raise InvalidJobStateError("尚未完成当前动画检查，不能记录成品视觉结论。")
            if (job.get("module_contract") or {}).get("version") != 2 or not job.get("health_report"):
                raise InvalidJobStateError("仅完整模块 V2 的当前生成结果可以提交最终视觉复核。")
        self._acquire_serial(job_id)
        try:
            current_binding = self._review_binding(job)
            if payload.get("review_binding") != job.get("review_binding") or current_binding != job.get("review_binding"):
                raise InvalidJobStateError("复核对象已变化或版本不匹配，请重新读取当前任务，不能复用旧通过。")
            if decision == "pass" and not _module_machine_clear(job["health_report"]):
                raise InvalidJobStateError("机器检查仍有失败或待审项，不能以视觉确认直接放行。")
            evidence = {
                "source": source, "reviewer": reviewer, "notes": notes,
                "full_loop_observed": payload.get("full_loop_observed") is True,
                "original_and_modules_viewed": payload.get("original_and_modules_viewed") is True,
                "observations": copy.deepcopy(payload.get("observations")),
                "review_binding": current_binding, "observed_at": _now_iso(),
                "paths": {"source_image": str(Path(job["job_dir"]) / "matted.png"),
                          "analysis_file": str(Path(job["job_dir"]) / "vision/vision.json")},
                "attestation": "explicit_reviewer_declaration_not_host_or_model_certification",
            }
            validate_visual_evidence(evidence, job["module_contract"], decision=decision)
            report = finalize_visual_review(copy.deepcopy(job["health_report"]), decision=decision, evidence=evidence)
            with self._state_lock:
                job["health_report"] = report
                job["visual_review"] = {"decision": decision, "evidence": evidence}
                allowed = (report.get("quality") or {}).get("overall", {}).get("exportAllowed") is True
                job["status"] = ("ready" if allowed else "failed" if decision == "fail" or report.get("status") == "fail"
                                 else "awaiting_visual_review" if _module_machine_clear(report) else "needs_review")
                job["exports"] = {}
                job["updated_at"] = _now_iso()
            (Path(job["job_dir"]) / "rig/health-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            self._persist_job_meta(job)
            return self.get_job(job_id)
        finally:
            self._release_serial()

    def apply_canvas_layers_job(self, job_id, payload):
        """Local aligned layers go directly to a preview; no model/service calls."""
        from scripts.canvas_to_spine import run_preview
        from PIL import Image
        with self._state_lock:
            job = self._jobs.get(job_id)
            if job is None: raise JobNotFoundError(job_id)
            self._assert_unlocked(job)
        self._acquire_serial(job_id, wait=bool(payload.get('wait_if_busy', False)))
        previous = copy.deepcopy(job)
        try:
            config = copy.deepcopy(payload)
            expected = config.pop('expected_rig_version', None)
            if expected is not None and expected != job['rig_build_count']:
                raise InvalidJobStateError('预览版本已经变化，请刷新后重新调整。', 409)
            parts = Path(config['parts_dir']).resolve()
            plan_path = Path(config['plan']).resolve()
            material = json.loads((parts/'parts.json').read_text())
            if not (Path(job['job_dir'])/'matted.png').is_file():
                raise ValueError('请先完成当前原图预处理。')
            with Image.open(material['originalSource']) as source, Image.open(Path(job['job_dir'])/'matted.png') as current:
                import numpy as np
                if not np.array_equal(np.asarray(source.convert('RGBA')), np.asarray(current.convert('RGBA'))):
                    raise ValueError('Canvas原图与当前上传角色不同。')
            revision = job.get('canvas_revision', 0) + 1
            destination = Path(job['job_dir'])/('canvas-r'+str(revision))
            while destination.exists():
                revision += 1
                destination = Path(job['job_dir'])/('canvas-r'+str(revision))
            root = Path(os.path.commonpath([str(parts), str(plan_path.parent), job['job_dir']]))
            job.update(status='running', exports={}, visual_review=None, canvas_signoff=None, canvas_revision=revision)
            diagnostics = run_preview(parts, plan_path, destination, evidence_root=root,
                character=job['character'], preset=job['preset'], gain=job['gain'], fps=30,
                cleanup=config.get('cleanup', True), tweaks=config.get('tweaks'))
            prepared = destination/'prepared'/'parts'/'parts.json'
            actual = json.loads(prepared.read_text()) if prepared.exists() else material
            spec = json.loads((destination/'rig-spec/rig-spec.json').read_text())
            checks = {'files_complete': diagnostics['files_complete'],
                'loop_seam': diagnostics['loop_seam_max_diff'] <= 2,
                'original_coverage': not diagnostics.get('cleanup',{}).get('missing_original_regions', []),
                'rigid_no_stretch': diagnostics['dynamic_checks']['rigid_max_singular_deviation'] < 1e-5}
            health = {'hardGates': {'rest_diff': {'passed':diagnostics['rest_composite_diff']['maxRGBA'] <= 2}},
                'quality': {'overall': {'status':'pending','exportAllowed':False},
                            'bindingMotion': {'checks': {'rigidity': {'status':'pass' if checks['rigid_no_stretch'] else 'fail'}}}}}
            (destination/'rig/health-report.json').write_text(json.dumps(health))
            seam_url = self._job_url(job_id, destination.name+'/seam-worst-crops.jpg')
            rows = _canvas_workbench_files(destination, actual['parts'], spec, job['character']+'.spine.json')
            for row in rows:
                row['thumb_url'] = self._job_url(job_id, destination.name+'/rig/images/'+row['name']+'.png')
                row['seam_crops_url'] = seam_url
            job.update(canvas_mode=True, canvas_config=config, canvas_output=str(destination),
                canvas_checks=checks, canvas_diagnostics=diagnostics,
                canvas_layers=rows, canvas_global_motion_scale=(config.get('tweaks') or {}).get('global_motion_scale', 1.),
                status='canvas_preview', rig_spec=spec, health_report=health, module_contract=None,
                rig_url=self._job_url(job_id, destination.name+'/rig/'+job['character']+'.spine.json'),
                health_report_url=self._job_url(job_id,destination.name+'/rig/health-report.json'),
                canvas_preview_html=self._job_url(job_id,destination.name+'/'+job['character']+'-standalone-preview.html'),
                canvas_seam_preview=seam_url,
                canvas_live_rig_url=self._job_url(job_id,destination.name+'/rig/live.spine.json'),
                canvas_diagnostics_url=self._job_url(job_id,destination.name+'/preview-diagnostics.json'),
                canvas_all_layers_url=self._job_url(job_id,destination.name+'/all-layers-sheet.jpg'),
                rig_build_count=job['rig_build_count']+1, updated_at=_now_iso())
            self._persist_job_meta(job)
            linked = self.batch_manager.find_item_by_job_id(job_id)
            if linked is not None:
                bid, bit = linked
                if not bit.get('locked'):
                    self.batch_manager.update_item(bid, bit['item_id'], {'status': 'preview_ready', 'error': None})
            return self._public_view(job)
        except Exception:
            if job.get('status')=='running':
                if previous.get('canvas_mode'):
                    job.update(previous)
                else:
                    job.update(status='failed',canvas_signoff=None,exports={})
                self._persist_job_meta(job)
            raise
        finally: self._release_serial()

    def layer_tweaks_job(self, job_id, payload):
        with self._state_lock:
            job=self._jobs.get(job_id)
            if job is None: raise JobNotFoundError(job_id)
            self._assert_unlocked(job)
            if not job.get('canvas_mode'): raise InvalidJobStateError('先接入Canvas图层。')
            config=copy.deepcopy(job['canvas_config'])
        tweaks=config.setdefault('tweaks',{})
        if 'expected_rig_version' in payload:
            config['expected_rig_version'] = payload['expected_rig_version']
        if 'global_motion_scale' in payload:
            value = payload['global_motion_scale']
            if type(value) not in (float, int) or not 0 <= value <= 1.8:
                raise ValueError('整体动感必须在 0%–180% 之间。')
            tweaks['global_motion_scale'] = value
        for group in ('layers','modules'):
            for name,changes in payload.get(group,{}).items():
                changes = copy.deepcopy(changes)
                if group == 'layers' and changes.get('enabled') is True and 'effect_mode' not in changes:
                    changes['effect_mode'] = 'keep'
                tweaks.setdefault(group,{}).setdefault(name,{}).update(changes)
        return self.apply_canvas_layers_job(job_id,config)

    def canvas_signoff_job(self, job_id, payload):
        with self._state_lock:
            job=self._jobs.get(job_id)
            if job is None: raise JobNotFoundError(job_id)
            if not job.get('canvas_mode') or job['status'] not in ('canvas_preview','ready'):
                raise InvalidJobStateError('当前没有可签收的Canvas预览。')
            if payload.get('decision') != 'pass' or not all(job['canvas_checks'].values()):
                raise InvalidJobStateError('四项基础检查通过且实际验收后才能签收。')
            if (job['canvas_diagnostics'].get('exposure_stress') or {}).get('status') != 'pass':
                raise InvalidJobStateError('先通过满滑块露底压力体检，再人工签收。')
        self._acquire_serial(job_id)
        try:
            expected = payload.get('expected_rig_version')
            if expected is not None and expected != job['rig_build_count']:
                raise InvalidJobStateError('预览版本已经变化，请重新观看并验收。', 409)
            out=Path(job['canvas_output']);signoff=out/'user-signoff.json'
            signoff.write_text(json.dumps({'decision':'pass','reviewer':str(payload.get('reviewer') or '工具页使用者'),
                'user_quote':'人工验收通过','revision':job['canvas_revision'],'date':_now_iso()},ensure_ascii=False))
            result=export_default_package(out/'rig',out/'exports',render_previews=False,user_signoff_path=signoff)
            files=[Path(result['zip_path']),Path(result['standalone_html'])]
            files += sorted(out.glob('*-preview-*.mp4'))
            job['exports']={'spine_package':{'kind':'spine_package','files':[{
                'label':p.name,'filename':p.name,'url':self._job_url(job_id,str(p.relative_to(Path(job['job_dir']))))} for p in files]}}
            job.update(status='ready',canvas_signoff=str(signoff),updated_at=_now_iso())
            self._persist_job_meta(job)
            return self._public_view(job)
        finally: self._release_serial()

    def export_job(self, job_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self._state_lock:
            current = self._jobs.get(job_id)
            if current and current.get('canvas_mode'):
                if not current.get('canvas_signoff'): raise InvalidJobStateError('请先人工验收签收。')
                return copy.deepcopy(current['exports']['spine_package'])
        kind = str(payload.get("kind") or "spine_package")
        if kind not in ("spine_package", "preview_video", "transparent_web", "standalone_html"):
            raise ValueError(f"不支持的导出类型 {kind!r}，可选：spine_package / preview_video / transparent_web / standalone_html")

        with self._state_lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise JobNotFoundError(job_id)
            quality = (job.get("health_report") or {}).get("quality") or {}
            if (job["status"] != "ready" or (job.get("module_contract") or {}).get("version") != 2
                    or quality.get("overall", {}).get("exportAllowed") is not True):
                raise InvalidJobStateError(f"完整模块、机器检查及当前版本的视觉验收全部通过后才能导出，当前状态为 {job['status']}。")
            job_dir = Path(job["job_dir"])

        rig_dir = job_dir / "rig"
        exports_dir = job_dir / "exports"

        self._acquire_serial(job_id)
        try:
            if job.get("review_binding") != self._review_binding(job):
                raise InvalidJobStateError("验收后的输入、模块或动画文件已变化，请重新检查，旧通过结果不能用于导出。")
            if kind == "spine_package":
                res = export_default_package(rig_dir, exports_dir, render_previews=False)
                zip_name = Path(res["zip_path"]).name
                files = [
                    {
                        "label": "Spine 4.2 工程包 (.zip)",
                        "filename": zip_name,
                        "url": self._job_url(job_id, f"exports/{zip_name}"),
                    }
                ]
                html_item = _standalone_html_download(job_id, res.get("standalone_html"), self._job_url)
                if html_item:
                    files.append(html_item)
                record = {
                    "kind": kind,
                    "created_at": _now_iso(),
                    "files": files,
                    "manifest": res["zip_entries"] + ([html_item["filename"]] if html_item else []),
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
                html_item = _standalone_html_download(job_id, res.get("standalone_html"), self._job_url)
                if html_item:
                    files.append(html_item)
                record = {
                    "kind": kind,
                    "created_at": _now_iso(),
                    "files": files,
                    "manifest": [f["filename"] for f in files],
                    "notice": res.get("preview_notice"),
                }
            elif kind == "standalone_html":
                html_path = exports_dir / "standalone-preview.html"
                html_res = export_standalone_html(rig_dir, html_path)
                html_item = _standalone_html_download(job_id, html_res["html_path"], self._job_url)
                record = {
                    "kind": kind,
                    "created_at": _now_iso(),
                    "files": [html_item] if html_item else [],
                    "manifest": [html_item["filename"]] if html_item else [],
                    "notice": None,
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

    def lock_job(self, job_id: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = payload or {}
        with self._state_lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise JobNotFoundError(job_id)
            is_canvas = bool(job.get("canvas_mode"))
            if is_canvas:
                if job["status"] not in ("canvas_preview", "ready"):
                    raise InvalidJobStateError("当前没有可确认锁定的动画预览。")
                checks = job.get("canvas_checks") or {}
                if not checks or not all(checks.values()):
                    raise InvalidJobStateError("四项基础检查全部通过后才能确认锁定。")
                stress = (job.get("canvas_diagnostics") or {}).get("exposure_stress") or {}
                if stress.get("status") != "pass":
                    raise InvalidJobStateError("请先通过 100%/150% 满滑块露底压力体检再确认锁定。")
            else:
                if job["status"] != "ready":
                    raise InvalidJobStateError("任务未就绪，无法确认锁定。")

        self._acquire_serial(job_id, wait=True)
        try:
            with self._state_lock:
                job = self._jobs[job_id]
                expected = payload.get("expected_rig_version")
                if expected is not None and expected != job.get("rig_build_count"):
                    raise InvalidJobStateError("预览版本已经变化，请刷新后再锁定。", 409)
                now = _now_iso()
                if job.get("canvas_mode"):
                    out = Path(job["canvas_output"])
                    signoff = out / "user-signoff.json"
                    signoff.write_text(
                        json.dumps(
                            {
                                "decision": "pass",
                                "reviewer": str(payload.get("reviewer") or "批量验收使用者"),
                                "user_quote": "人工过目确认锁定",
                                "revision": job.get("canvas_revision"),
                                "date": now,
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    job["canvas_signoff"] = str(signoff)
                    job["status"] = "ready"
                job["locked"] = True
                job["locked_at"] = now
                job["locked_revision"] = job.get("canvas_revision")
                job["locked_rig_version"] = job.get("rig_build_count")
                job["updated_at"] = now
                self._persist_job_meta(job)
                return self._public_view(job)
        finally:
            self._release_serial()

    def unlock_job(self, job_id: str) -> Dict[str, Any]:
        with self._state_lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise JobNotFoundError(job_id)
            job["locked"] = False
            job["locked_at"] = None
            job["updated_at"] = _now_iso()
            self._persist_job_meta(job)
            return self._public_view(job)

    def _compute_item_traffic_light(self, item: Dict[str, Any], job: Optional[Dict[str, Any]]) -> Tuple[str, str]:
        if item.get("status") == "failed" or (job and job.get("status") == "failed"):
            return "red", item.get("error") or "处理失败"
        if item.get("status") == "needs_fix":
            return "yellow", f"待修改：{item.get('fix_note') or ''}"
        if job is None or job.get("status") in ("running", "awaiting_analysis", "awaiting_rig_confirmation"):
            return "pending", "等待分层预览"
        if job.get("canvas_mode"):
            checks = job.get("canvas_checks") or {}
            diag = job.get("canvas_diagnostics") or {}
            stress = diag.get("exposure_stress") or {}
            warnings = diag.get("cleanup_warnings") or []
            severe = [w for w in warnings if w.get("code") in ("local_layer_pixel_bleed", "separated_soft_islands")]
            if not all(checks.values()) or stress.get("status") == "fail":
                return "red", "基础或压力体检未通过"
            if severe or stress.get("status") != "pass":
                return "yellow", "存在串层/软层或边缘告警"
            return "green", "体检全绿（0露底）"
        if job.get("status") == "ready":
            return "green", "就绪"
        return "yellow", str(job.get("status"))

    def list_batches(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "batches": self.batch_manager.list_batches(),
            "export_catalog": export_catalog_metadata(),
        }

    def create_or_rescan_batch(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        source_dir = payload.get("source_dir")
        if not source_dir or not str(source_dir).strip():
            raise ValueError("请提供本地文件夹路径 source_dir。")
        batch = self.batch_manager.create_or_rescan_batch(
            str(source_dir),
            name=payload.get("name"),
            batch_id=payload.get("batch_id"),
            export_profile=payload.get("export_profile"),
        )
        return self.get_batch_view(batch["batch_id"])

    def get_batch_view(self, batch_id: str) -> Dict[str, Any]:
        batch = self.batch_manager.reload_batch(batch_id)
        enriched_items: List[Dict[str, Any]] = []
        counts = {
            "total": 0, "pending": 0, "matted": 0, "prepared": 0,
            "preview_ready": 0, "needs_fix": 0, "approved_locked": 0,
            "exported": 0, "failed": 0, "locked": 0,
            "green": 0, "yellow": 0, "red": 0,
        }
        for item in batch.get("items") or []:
            jid = item.get("job_id")
            with self._state_lock:
                job = self._jobs.get(jid) if jid else None
            if job is None and jid and (self.work_dir / jid / "job-status.json").is_file():
                self._restore_persisted_jobs()
                with self._state_lock:
                    job = self._jobs.get(jid)
            pub_job = self._public_view(job) if job else None
            light, summary_text = self._compute_item_traffic_light(item, job)

            thumb_url = None
            if job and job.get("matted_url"):
                thumb_url = job["matted_url"]
            elif item.get("matted_path"):
                mp = Path(item["matted_path"]).resolve()
                try:
                    rel = mp.relative_to(self.work_dir / "batches" / batch_id)
                    thumb_url = f"/batches/{batch_id}/{rel.as_posix()}"
                except ValueError:
                    pass

            enriched = copy.deepcopy(item)
            enriched["traffic_light"] = light
            enriched["traffic_summary"] = summary_text
            enriched["thumb_url"] = thumb_url
            enriched["job_status"] = pub_job["status"] if pub_job else None
            enriched["canvas_checks"] = pub_job.get("canvas_checks") if pub_job else None
            enriched["exposure_stress"] = ((pub_job.get("canvas_diagnostics") or {}).get("exposure_stress") if pub_job else None)
            enriched["canvas_revision"] = pub_job.get("canvas_revision") if pub_job else None
            enriched["rig_build_count"] = pub_job.get("rig_build_count") if pub_job else None
            enriched["source_size"] = ((pub_job.get("preprocess") or {}).get("new_size") if pub_job else None)
            enriched_items.append(enriched)

            counts["total"] += 1
            st = str(item.get("status") or "pending")
            if st in counts:
                counts[st] += 1
            if item.get("locked"):
                counts["locked"] += 1
            if light in ("green", "yellow", "red"):
                counts[light] += 1

        overview_url = None
        if batch.get("matte_overview_path") and Path(batch["matte_overview_path"]).is_file():
            overview_url = f"/batches/{batch_id}/matte-overview.jpg"

        with self._batch_export_lock:
            exp_progress = copy.deepcopy(self._batch_export_states.get(batch_id))

        return {
            "ok": True,
            "batch_id": batch["batch_id"],
            "name": batch.get("name"),
            "source_dir": batch.get("source_dir"),
            "created_at": batch.get("created_at"),
            "updated_at": batch.get("updated_at"),
            "export_profile": batch.get("export_profile"),
            "export_catalog": export_catalog_metadata(),
            "matte_overview_url": overview_url,
            "last_batch_export": batch.get("last_batch_export"),
            "export_progress": exp_progress,
            "counts": counts,
            "items": enriched_items,
            "batch_url": f"/?batch={batch['batch_id']}",
        }

    def update_batch_profile(self, batch_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        self.batch_manager.update_profile(batch_id, payload)
        return self.get_batch_view(batch_id)

    def lock_batch_item(self, batch_id: str, item_id: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        item = self.batch_manager.get_item(batch_id, item_id)
        jid = item.get("job_id")
        if not jid:
            raise InvalidJobStateError("该角色尚未生成预览，无法确认锁定。")
        job_view = self.lock_job(jid, payload)
        self.batch_manager.lock_item(
            batch_id,
            item["item_id"],
            revision=job_view.get("canvas_revision"),
            rig_version=job_view.get("rig_build_count"),
        )
        return self.get_batch_view(batch_id)

    def unlock_batch_item(self, batch_id: str, item_id: str) -> Dict[str, Any]:
        item = self.batch_manager.get_item(batch_id, item_id)
        jid = item.get("job_id")
        if jid:
            with self._state_lock:
                has_job = jid in self._jobs
            if has_job:
                self.unlock_job(jid)
        self.batch_manager.unlock_item(batch_id, item["item_id"])
        return self.get_batch_view(batch_id)

    def reject_batch_item(self, batch_id: str, item_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        note = str((payload or {}).get("fix_note") or (payload or {}).get("note") or "").strip()
        self.batch_manager.reject_item(batch_id, item_id, note)
        return self.get_batch_view(batch_id)

    def get_batch_fix_list(self, batch_id: str) -> Dict[str, Any]:
        items = self.batch_manager.get_fix_list(batch_id)
        return {"ok": True, "batch_id": batch_id, "count": len(items), "items": items}

    def _run_batch_export_worker(
        self,
        batch_id: str,
        targets: List[Dict[str, Any]],
        profile: Dict[str, Any],
        out_root: Path,
        max_frames: Optional[int],
    ) -> Dict[str, Any]:
        out_root.mkdir(parents=True, exist_ok=True)
        results: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []
        total = len(targets)

        for idx, item in enumerate(targets):
            char_name = str(item.get("character") or item["item_id"])
            with self._batch_export_lock:
                self._batch_export_states[batch_id] = {
                    "running": True,
                    "completed": idx,
                    "total": total,
                    "current_character": char_name,
                    "errors": list(errors),
                    "started_at": self._batch_export_states.get(batch_id, {}).get("started_at") or _now_iso(),
                }
            jid = item.get("job_id")
            try:
                with self._state_lock:
                    job = self._jobs.get(jid) if jid else None
                if job is None:
                    raise JobNotFoundError(str(jid))
                if job.get("canvas_mode"):
                    rig_dir = Path(job["canvas_output"]) / "rig"
                    signoff_path = Path(job["canvas_signoff"]) if job.get("canvas_signoff") else (Path(job["canvas_output"]) / "user-signoff.json")
                else:
                    rig_dir = Path(job["job_dir"]) / "rig"
                    signoff_path = None
                char_out = out_root / char_name
                res = export_job_with_profile(
                    rig_dir,
                    char_out,
                    profile,
                    character=char_name,
                    label=str(item.get("display_name") or char_name),
                    user_signoff_path=signoff_path if signoff_path and signoff_path.is_file() else None,
                    max_frames=max_frames,
                    require_approved=True,
                )
                results.append({"item_id": item["item_id"], "job_id": jid, **res})
                self.batch_manager.update_item(
                    batch_id,
                    item["item_id"],
                    {"status": "exported", "last_export": res},
                    allow_locked=True,
                )
            except Exception as exc:
                errors.append({"item_id": item["item_id"], "character": char_name, "error": str(exc)})

        manifest = {
            "version": 1,
            "batch_id": batch_id,
            "exported_at": _now_iso(),
            "output_dir": str(out_root),
            "profile": profile,
            "exported_count": len(results),
            "error_count": len(errors),
            "items": results,
            "errors": errors,
        }
        manifest_path = out_root / "batch-export-manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        summary = {
            "exported_at": manifest["exported_at"],
            "output_dir": str(out_root),
            "manifest_path": str(manifest_path),
            "profile": profile,
            "exported_count": len(results),
            "error_count": len(errors),
            "errors": errors,
        }
        self.batch_manager.record_batch_export(batch_id, summary)
        with self._batch_export_lock:
            self._batch_export_states[batch_id] = {
                "running": False,
                "completed": total,
                "total": total,
                "current_character": None,
                "errors": errors,
                "finished_at": manifest["exported_at"],
                "output_dir": str(out_root),
                "manifest_path": str(manifest_path),
            }
        return manifest

    def export_batch(self, batch_id: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = payload or {}
        if payload.get("export_profile"):
            self.batch_manager.update_profile(batch_id, payload["export_profile"])
        batch = self.batch_manager.get_batch(batch_id)
        profile = normalize_export_profile(batch.get("export_profile"))
        out_dir_raw = payload.get("output_dir") or profile.get("output_dir")
        out_root = Path(out_dir_raw).expanduser().resolve() if out_dir_raw else (self.work_dir / "batches" / batch_id / "exports")

        wanted_ids = set(payload.get("item_ids") or [])
        targets = [
            it for it in batch.get("items") or []
            if it.get("locked") and (not wanted_ids or it["item_id"] in wanted_ids)
        ]
        if not targets:
            raise InvalidJobStateError("当前没有已确认锁定的角色可导出；请先过目并锁定至少 1 个角色。")

        with self._batch_export_lock:
            cur = self._batch_export_states.get(batch_id)
            if cur and cur.get("running"):
                raise StudioBusyError(batch_id)
            self._batch_export_states[batch_id] = {
                "running": True,
                "completed": 0,
                "total": len(targets),
                "current_character": targets[0].get("character"),
                "errors": [],
                "started_at": _now_iso(),
            }

        max_frames = int(payload["max_frames"]) if payload.get("max_frames") is not None else None
        async_mode = bool(payload.get("async", False))
        if async_mode:
            t = threading.Thread(
                target=self._run_batch_export_worker,
                args=(batch_id, targets, profile, out_root, max_frames),
                daemon=True,
            )
            t.start()
            return {
                "ok": True,
                "batch_id": batch_id,
                "started": True,
                "total": len(targets),
                "output_dir": str(out_root),
                "profile": profile,
            }

        manifest = self._run_batch_export_worker(batch_id, targets, profile, out_root, max_frames)
        return {"ok": True, "started": False, **manifest}


def _standalone_html_download(job_id: str, html_path: Optional[str], job_url) -> Optional[Dict[str, str]]:
    if not html_path:
        return None
    path = Path(html_path)
    if not path.is_file():
        return None
    return {
        "label": "自包含单文件 HTML 预览",
        "filename": path.name,
        "url": job_url(job_id, "exports/" + path.name),
    }


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
    vision_connection_path: Optional[Union[str, Path]] = None,
) -> ThreadingHTTPServer:
    """Create a configured `ThreadingHTTPServer` for Portrait Spine Studio."""
    wdir = Path(work_dir).resolve() if work_dir else (REPO_ROOT / ".studio-work")
    job_manager = manager or StudioJobManager(wdir, vision_connection_path=vision_connection_path)

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
            connection = getattr(job_manager.vision_provider, "connection_path", None)
            if (connection is not None and path.resolve() == Path(connection).resolve()) or "vision-bridge" in path.parts:
                self._send_json(HTTPStatus.FORBIDDEN, {"ok": False, "errorCode": "FORBIDDEN", "message": "内部连接配置不通过网页提供。"})
                return
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

            if path == "/api/batches":
                self._send_json(HTTPStatus.OK, job_manager.list_batches())
                return

            if path.startswith("/api/batches/"):
                parts = [p for p in path.split("/") if p]
                try:
                    if len(parts) == 3:
                        self._send_json(HTTPStatus.OK, job_manager.get_batch_view(parts[2]))
                        return
                    if len(parts) == 4 and parts[3] == "fix-list":
                        self._send_json(HTTPStatus.OK, job_manager.get_batch_fix_list(parts[2]))
                        return
                except BatchNotFoundError as exc:
                    self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "errorCode": "BATCH_NOT_FOUND", "message": str(exc)})
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
                if len(parts) == 4 and parts[0] == "api" and parts[1] == "jobs" and parts[3] == "rig-spec":
                    try:
                        self._send_json(HTTPStatus.OK, job_manager.get_rig_spec(parts[2]))
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

            if path.startswith("/batches/"):
                rel = path[len("/batches/") :]
                target = _safe_resolve_under(job_manager.work_dir / "batches", rel)
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

                if path == "/api/batches":
                    self._send_json(HTTPStatus.OK, job_manager.create_or_rescan_batch(body))
                    return

                parts = [p for p in path.split("/") if p]
                if len(parts) == 4 and parts[0] == "api" and parts[1] == "jobs":
                    job_id, action = parts[2], parts[3]
                    if action == "apply-canvas-layers":
                        self._send_json(HTTPStatus.OK, job_manager.apply_canvas_layers_job(job_id, body)); return
                    if action == "layer-tweaks":
                        self._send_json(HTTPStatus.OK, job_manager.layer_tweaks_job(job_id, body)); return
                    if action == "user-signoff":
                        self._send_json(HTTPStatus.OK, job_manager.canvas_signoff_job(job_id, body)); return
                    if action == "apply-analysis":
                        self._send_json(HTTPStatus.OK, job_manager.apply_analysis_job(job_id, body))
                        return
                    if action == "refine-mask":
                        res = job_manager.refine_mask_job(job_id, body)
                        status = HTTPStatus.OK if res.get("ok") else HTTPStatus.UNPROCESSABLE_ENTITY
                        self._send_json(status, res)
                        return
                    if action in ("continue", "confirm-rig"):
                        self._send_json(HTTPStatus.OK, job_manager.continue_job(job_id, body))
                        return
                    if action == "rebind":
                        self._send_json(HTTPStatus.OK, job_manager.rebind_job(job_id, body))
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
                    if action == "review":
                        self._send_json(HTTPStatus.OK, job_manager.review_job(job_id, body))
                        return
                    if action == "export":
                        self._send_json(HTTPStatus.OK, job_manager.export_job(job_id, body))
                        return
                    if action == "lock":
                        self._send_json(HTTPStatus.OK, job_manager.lock_job(job_id, body))
                        return
                    if action == "unlock":
                        self._send_json(HTTPStatus.OK, job_manager.unlock_job(job_id))
                        return

                if len(parts) == 4 and parts[0] == "api" and parts[1] == "batches":
                    batch_id, action = parts[2], parts[3]
                    if action == "profile":
                        self._send_json(HTTPStatus.OK, job_manager.update_batch_profile(batch_id, body)); return
                    if action == "export":
                        self._send_json(HTTPStatus.OK, job_manager.export_batch(batch_id, body)); return

                if len(parts) == 6 and parts[0] == "api" and parts[1] == "batches" and parts[3] == "items":
                    batch_id, item_id, action = parts[2], parts[4], parts[5]
                    if action == "lock":
                        self._send_json(HTTPStatus.OK, job_manager.lock_batch_item(batch_id, item_id, body)); return
                    if action == "unlock":
                        self._send_json(HTTPStatus.OK, job_manager.unlock_batch_item(batch_id, item_id)); return
                    if action == "reject":
                        self._send_json(HTTPStatus.OK, job_manager.reject_batch_item(batch_id, item_id, body)); return

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
            except (JobNotFoundError, BatchNotFoundError, BatchItemNotFoundError) as exc:
                self._send_json(
                    HTTPStatus.NOT_FOUND,
                    {"ok": False, "errorCode": "NOT_FOUND" if isinstance(exc, (BatchNotFoundError, BatchItemNotFoundError)) else "JOB_NOT_FOUND", "message": str(exc)},
                )
            except BatchItemLockedError as exc:
                self._send_json(
                    HTTPStatus.CONFLICT,
                    {"ok": False, "errorCode": "JOB_LOCKED", "message": str(exc)},
                )
            except InvalidJobStateError as exc:
                status = HTTPStatus.CONFLICT if getattr(exc, "http_status", 400) == 409 else HTTPStatus.BAD_REQUEST
                err_code = getattr(exc, "code", None) or ("INVALID_STATE" if status == 409 else "INVALID_REQUEST")
                self._send_json(
                    status,
                    {"ok": False, "errorCode": err_code, "message": str(exc)},
                )
            except ValueError as exc:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"ok": False, "errorCode": getattr(exc, "code", None) or "INVALID_REQUEST", "message": str(exc)},
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
