"""Mivo Canvas adapter protocol, file-backed & deterministic mock adapters, and fail-stop stage runner."""
from __future__ import annotations

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
            cand = self.handoff_dir / dst.name
            if not cand.is_file():
                raise MivoCanvasError(
                    f"Expected Mivo matting output missing at {cand}",
                    code="MIVO_HANDOFF_MISSING",
                    stage="auto_matting",
                    retryable=True,
                    details={"expected_path": str(cand)},
                )
            shutil.copy2(cand, dst)
        with Image.open(dst) as im:
            w, h = im.size
        return MivoResult(operation="auto_matting", output_path=str(dst), width=w, height=h)

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
