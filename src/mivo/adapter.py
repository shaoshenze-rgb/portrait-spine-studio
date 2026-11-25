"""Mivo Canvas adapter protocol, file-backed & deterministic mock adapters, and fail-stop stage runner."""
from __future__ import annotations

import hashlib
import json
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image


class MivoCanvasError(RuntimeError):
    """Structured error raised when a Mivo Canvas operation fails."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "MIVO_OPERATION_FAILED",
        stage: str = "mivo",
        retryable: bool = True,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.code = code
        self.stage = stage
        self.retryable = retryable
        self.details = details or {}


@dataclass
class MivoResult:
    operation: str
    output_path: str
    width: int
    height: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class MivoCanvasAdapter(ABC):
    """Adapter boundary for Mivo Canvas image operations."""

    @abstractmethod
    def auto_matting(self, image_path: Union[str, Path], out_path: Union[str, Path]) -> MivoResult:
        """Remove background and output an RGBA matted portrait PNG."""

    @abstractmethod
    def inpaint_occluded(
        self,
        image_path: Union[str, Path],
        mask_path: Union[str, Path],
        prompt: str,
        out_path: Union[str, Path],
    ) -> MivoResult:
        """Inpaint occluded regions indicated by non-zero pixels in `mask_path`."""


def validate_matted_portrait(path: Union[str, Path], *, expected_size: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
    """Reject opaque, empty or resized handoffs before they enter the rig pipeline."""
    with Image.open(path) as im:
        size = im.size
        alpha = np.asarray(im.convert("RGBA"), dtype=np.uint8)[..., 3]
    if expected_size is not None and size != expected_size:
        raise MivoCanvasError("抠图交接结果尺寸与输入不一致，无法保持关键点坐标。", code="MIVO_HANDOFF_SIZE_MISMATCH")
    if float((alpha < 20).mean()) < 0.03:
        raise MivoCanvasError("检测到不透明背景，尚未取得真实去底结果；已保留输入，请先完成 Mivo Canvas 去底。", code="MIVO_MATTING_REQUIRED")
    if not (alpha >= 128).any():
        raise MivoCanvasError("抠图结果没有可绑定的角色前景。", code="MIVO_EMPTY_FOREGROUND")
    return size


class TransparentInputAdapter(MivoCanvasAdapter):
    """Pass through already-transparent input; never pretend to remove a background."""

    def auto_matting(self, image_path: Union[str, Path], out_path: Union[str, Path]) -> MivoResult:
        src, dst = Path(image_path).resolve(), Path(out_path).resolve()
        w, h = validate_matted_portrait(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src) as im:
            im.convert("RGBA").save(dst, format="PNG")
        return MivoResult("auto_matting", str(dst), w, h, {"source": "transparent_input", "performed_matting": False})

    def inpaint_occluded(self, image_path, mask_path, prompt, out_path) -> MivoResult:
        raise MivoCanvasError("尚未接入真实遮挡补全服务。", code="MIVO_INPAINT_REQUIRED")


class FileBackedMivoAdapter(MivoCanvasAdapter):
    """Reads pre-produced Mivo Canvas outputs from a handoff directory (or calls optional hooks)."""

    def __init__(
        self,
        handoff_dir: Union[str, Path],
        *,
        matting_hook: Optional[Callable[[Path, Path], None]] = None,
        inpaint_hook: Optional[Callable[[Path, Path, str, Path], None]] = None,
    ):
        self.handoff_dir = Path(handoff_dir).resolve()
        self._matting_hook = matting_hook
        self._inpaint_hook = inpaint_hook

    def auto_matting(self, image_path: Union[str, Path], out_path: Union[str, Path]) -> MivoResult:
        src = Path(image_path).resolve()
        dst = Path(out_path).resolve()
        dst.parent.mkdir(parents=True, exist_ok=True)
        if self._matting_hook is not None:
            try:
                self._matting_hook(src, dst)
            except Exception as exc:
                raise MivoCanvasError(
                    f"Mivo auto_matting hook failed: {exc}",
                    code="MIVO_MATTING_FAILED",
                    stage="auto_matting",
                    retryable=True,
                ) from exc
        else:
            # Bind each result to the exact preprocessed input, never a shared filename.
            input_sha = hashlib.sha256(src.read_bytes()).hexdigest()
            cand = self.handoff_dir / input_sha / "matted.png"
            if not cand.is_file():
                raise MivoCanvasError(
                    f"Expected Mivo matting output missing at {cand}",
                    code="MIVO_HANDOFF_MISSING",
                    stage="auto_matting",
                    retryable=True,
                    details={"expected_path": str(cand)},
                )
            shutil.copy2(cand, dst)
        with Image.open(src) as im:
            source_size = im.size
        w, h = validate_matted_portrait(dst, expected_size=source_size)
        return MivoResult(operation="auto_matting", output_path=str(dst), width=w, height=h, metadata={
            "source": "hook" if self._matting_hook is not None else "file_handoff",
            "input_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
            "output_sha256": hashlib.sha256(dst.read_bytes()).hexdigest(),
        })

    def inpaint_occluded(
        self,
        image_path: Union[str, Path],
        mask_path: Union[str, Path],
        prompt: str,
        out_path: Union[str, Path],
    ) -> MivoResult:
        src = Path(image_path).resolve()
        mask = Path(mask_path).resolve()
        dst = Path(out_path).resolve()
        dst.parent.mkdir(parents=True, exist_ok=True)
        if self._inpaint_hook is not None:
            try:
                self._inpaint_hook(src, mask, prompt, dst)
            except Exception as exc:
                raise MivoCanvasError(
                    f"Mivo inpaint_occluded hook failed: {exc}",
                    code="MIVO_INPAINT_FAILED",
                    stage="inpaint_occluded",
                    retryable=True,
                    details={"prompt": prompt},
                ) from exc
        else:
            cand = self.handoff_dir / dst.name
            if not cand.is_file():
                raise MivoCanvasError(
                    f"Expected Mivo inpaint output missing at {cand}",
                    code="MIVO_HANDOFF_MISSING",
                    stage="inpaint_occluded",
                    retryable=True,
                    details={"expected_path": str(cand), "prompt": prompt},
                )
            shutil.copy2(cand, dst)
        with Image.open(dst) as im:
            w, h = im.size
        return MivoResult(
            operation="inpaint_occluded",
            output_path=str(dst),
            width=w,
            height=h,
            metadata={"prompt": prompt},
        )


class DeterministicFallbackMock(MivoCanvasAdapter):
    """Deterministic mock Mivo adapter for unit tests (supports failure injection via `fail_on`)."""

    def __init__(self, *, fail_on: Optional[str] = None, error_code: str = "MIVO_SERVICE_UNAVAILABLE"):
        self.fail_on = fail_on
        self.error_code = error_code
        self.calls: List[Dict[str, Any]] = []

    def auto_matting(self, image_path: Union[str, Path], out_path: Union[str, Path]) -> MivoResult:
        self.calls.append({"op": "auto_matting", "image_path": str(image_path), "out_path": str(out_path)})
        if self.fail_on == "auto_matting":
            raise MivoCanvasError(
                "Injected Mivo auto_matting failure",
                code=self.error_code,
                stage="auto_matting",
                retryable=True,
            )
        src = Path(image_path).resolve()
        dst = Path(out_path).resolve()
        dst.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src) as im:
            rgba = im.convert("RGBA")
            w, h = rgba.size
            rgba.save(dst, format="PNG")
        return MivoResult(operation="auto_matting", output_path=str(dst), width=w, height=h, metadata={"mock": True})

    def inpaint_occluded(
        self,
        image_path: Union[str, Path],
        mask_path: Union[str, Path],
        prompt: str,
        out_path: Union[str, Path],
    ) -> MivoResult:
        self.calls.append(
            {
                "op": "inpaint_occluded",
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "prompt": prompt,
                "out_path": str(out_path),
            }
        )
        if self.fail_on == "inpaint_occluded":
            raise MivoCanvasError(
                "Injected Mivo inpaint_occluded failure",
                code=self.error_code,
                stage="inpaint_occluded",
                retryable=True,
                details={"prompt": prompt},
            )
        src = Path(image_path).resolve()
        msk = Path(mask_path).resolve()
        dst = Path(out_path).resolve()
        dst.parent.mkdir(parents=True, exist_ok=True)

        img = np.asarray(Image.open(src).convert("RGBA"), dtype=np.uint8).copy()
        mask = np.asarray(Image.open(msk).convert("L"), dtype=np.uint8) > 0
        known = (~mask) & (img[..., 3] > 0)
        if mask.any() and known.any():
            mean_rgb = np.rint(img[known, :3].mean(axis=0)).astype(np.uint8)
            img[mask, :3] = mean_rgb
            img[mask, 3] = 255
        Image.fromarray(img).save(dst, format="PNG")
        h, w = img.shape[:2]
        return MivoResult(
            operation="inpaint_occluded",
            output_path=str(dst),
            width=w,
            height=h,
            metadata={"mock": True, "prompt": prompt},
        )


class StageRunner:
    """Executes pipeline stages with strict fail-stop and intermediate artifact preservation.

    On any MivoCanvasError, GateFailureError, or unexpected exception:
    - writes `<work_dir>/stage-error.json` with actionable diagnostics and all preserved intermediate paths;
    - never deletes intermediate artifacts so callers can retry or discard cleanly;
    - never silently passes.
    """

    def __init__(self, work_dir: Union[str, Path]):
        self.work_dir = Path(work_dir).resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.intermediates: Dict[str, str] = {}
        self.results: Dict[str, Any] = {}

    def record_intermediate(self, key: str, path: Union[str, Path]) -> None:
        self.intermediates[key] = str(Path(path).resolve())

    def run_stages(
        self,
        stages: Sequence[Tuple[str, Callable[[], Any]]],
        *,
        raise_on_error: bool = True,
    ) -> Dict[str, Any]:
        error_file = self.work_dir / "stage-error.json"
        if error_file.exists():
            error_file.unlink()

        for stage_name, fn in stages:
            try:
                res = fn()
                self.results[stage_name] = res
                if isinstance(res, dict):
                    for k in ("output_path", "spine_path", "out_dir", "report_path"):
                        if k in res and isinstance(res[k], str):
                            self.intermediates[f"{stage_name}.{k}"] = res[k]
                elif isinstance(res, MivoResult):
                    self.intermediates[stage_name] = res.output_path
            except Exception as exc:
                code = getattr(exc, "code", exc.__class__.__name__)
                retryable = bool(getattr(exc, "retryable", True))
                details = getattr(exc, "details", {})
                payload = {
                    "status": "failed",
                    "failedStage": stage_name,
                    "errorCode": code,
                    "message": str(exc),
                    "retryable": retryable,
                    "details": details,
                    "preservedIntermediates": dict(self.intermediates),
                    "recoveryActions": ["retry", "discard"],
                }
                error_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
                if raise_on_error:
                    raise
                return payload

        return {
            "status": "ok",
            "completedStages": [name for name, _ in stages],
            "intermediates": dict(self.intermediates),
            "results": self.results,
        }
