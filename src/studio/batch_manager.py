"""Folder-based batch manager with hard-lock protection, fix-queue, and 4K export profile."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw

from src.export.export_profile import (
    DEFAULT_EXPORT_PROFILE,
    FORMAT_OPTIONS,
    SIZE_PRESETS,
    normalize_export_profile,
)

SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


class BatchNotFoundError(KeyError):
    def __init__(self, batch_id: str):
        super().__init__(f"未找到批次：{batch_id}")
        self.batch_id = batch_id


class BatchItemNotFoundError(KeyError):
    def __init__(self, batch_id: str, item_id: str):
        super().__init__(f"批次 {batch_id} 中未找到角色项：{item_id}")
        self.batch_id = batch_id
        self.item_id = item_id


class BatchItemLockedError(RuntimeError):
    def __init__(self, item_id: str, message: Optional[str] = None):
        super().__init__(
            message or f"角色 {item_id} 已确认锁定，禁止覆盖或修改；如需返修请先在界面点击「解锁」。"
        )
        self.item_id = item_id
        self.http_status = 409


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _natural_key(name: str) -> List[Any]:
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", name)]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sanitize_slug(raw: str, fallback: str = "character") -> str:
    stem = Path(raw).stem
    ascii_part = re.sub(r"[^a-zA-Z0-9_-]+", "_", stem).strip("_-").lower()
    if ascii_part:
        return ascii_part[:42]
    digest = hashlib.sha1(stem.encode("utf-8")).hexdigest()[:8]
    return f"{fallback}_{digest}"


def scan_portrait_folder(source_dir: Union[str, Path]) -> List[Path]:
    """Return naturally sorted portrait images directly under `source_dir`."""
    folder = Path(source_dir).expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"输入文件夹不存在：{folder}")
    found: List[Path] = []
    for entry in folder.iterdir():
        if not entry.is_file():
            continue
        if entry.name.startswith("."):
            continue
        if entry.suffix.lower() not in SUPPORTED_IMAGE_EXTS:
            continue
        found.append(entry)
    found.sort(key=lambda p: _natural_key(p.name))
    return found


def _checkerboard(w: int, h: int, tile: int = 16) -> Image.Image:
    ys, xs = np.indices((h, w))
    grid = ((xs // tile) + (ys // tile)) % 2
    arr = np.where(grid[..., None] == 0, 56, 76).astype(np.uint8)
    rgb = np.repeat(arr, 3, axis=2)
    return Image.fromarray(rgb)


def render_matte_overview(
    items: Sequence[Dict[str, Any]],
    out_path: Union[str, Path],
    *,
    cell_w: int = 300,
    cell_h: int = 380,
    cols: int = 5,
) -> Optional[Path]:
    """Generate a labeled checkerboard contact sheet of matted batch portraits."""
    matted_items = [
        it for it in items if it.get("matted_path") and Path(str(it["matted_path"])).is_file()
    ]
    if not matted_items:
        return None
    cols = max(1, min(cols, len(matted_items)))
    rows = (len(matted_items) + cols - 1) // cols
    header_h = 34
    pad = 12
    sheet_w = cols * (cell_w + pad) + pad
    sheet_h = rows * (cell_h + header_h + pad) + pad
    sheet = Image.new("RGB", (sheet_w, sheet_h), (24, 27, 34))
    draw = ImageDraw.Draw(sheet)
    lanczos = getattr(Image, "Resampling", Image).LANCZOS

    for idx, it in enumerate(matted_items):
        r, c = divmod(idx, cols)
        x0 = pad + c * (cell_w + pad)
        y0 = pad + r * (cell_h + header_h + pad)
        bg = _checkerboard(cell_w, cell_h, tile=14).convert("RGBA")
        with Image.open(it["matted_path"]) as im:
            rgba = im.convert("RGBA")
            iw, ih = rgba.size
            scale = min((cell_w - 16) / max(1, iw), (cell_h - 16) / max(1, ih))
            nw = max(1, int(round(iw * scale)))
            nh = max(1, int(round(ih * scale)))
            thumb = rgba.resize((nw, nh), resample=lanczos)
            ox = (cell_w - nw) // 2
            oy = cell_h - nh - 8
            bg.alpha_composite(thumb, (ox, oy))
        sheet.paste(bg.convert("RGB"), (x0, y0))
        label = f"#{idx + 1:02d} {it.get('character') or it.get('filename')} ({iw}x{ih})"
        if len(label) > 36:
            label = label[:33] + "..."
        draw.rectangle([x0, y0 + cell_h, x0 + cell_w, y0 + cell_h + header_h], fill=(34, 39, 49))
        draw.text((x0 + 8, y0 + cell_h + 10), label, fill=(232, 236, 245))

    dst = Path(out_path).resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(dst, format="JPEG", quality=90)
    return dst


class BatchManager:
    """Thread-safe persisted manager for folder-based 20-30 portrait batches."""

    def __init__(self, work_dir: Union[str, Path]):
        self.work_dir = Path(work_dir).resolve()
        self.batches_root = self.work_dir / "batches"
        self.batches_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._batches: Dict[str, Dict[str, Any]] = {}
        self._load_existing()

    def _batch_dir(self, batch_id: str) -> Path:
        return self.batches_root / batch_id

    def _batch_file(self, batch_id: str) -> Path:
        return self._batch_dir(batch_id) / "batch.json"

    def _load_existing(self) -> None:
        if not self.batches_root.is_dir():
            return
        for child in sorted(self.batches_root.iterdir()):
            meta = child / "batch.json"
            if not meta.is_file():
                continue
            try:
                data = json.loads(meta.read_text(encoding="utf-8"))
                bid = str(data.get("batch_id") or child.name)
                data["batch_id"] = bid
                data["export_profile"] = normalize_export_profile(data.get("export_profile"))
                if not data["export_profile"].get("output_dir"):
                    data["export_profile"]["output_dir"] = str(child / "exports")
                self._batches[bid] = data
            except Exception:
                continue

    def _persist(self, batch: Dict[str, Any]) -> None:
        bdir = self._batch_dir(batch["batch_id"])
        bdir.mkdir(parents=True, exist_ok=True)
        target = self._batch_file(batch["batch_id"])
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(batch, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, target)

    def _require_batch(self, batch_id: str) -> Dict[str, Any]:
        batch = self._batches.get(batch_id)
        if batch is None:
            # Reload from disk in case created by CLI process while server is running
            meta = self._batch_file(batch_id)
            if meta.is_file():
                data = json.loads(meta.read_text(encoding="utf-8"))
                data["export_profile"] = normalize_export_profile(data.get("export_profile"))
                if not data["export_profile"].get("output_dir"):
                    data["export_profile"]["output_dir"] = str(self._batch_dir(batch_id) / "exports")
                self._batches[batch_id] = data
                batch = data
        if batch is None:
            raise BatchNotFoundError(batch_id)
        return batch

    def reload_batch(self, batch_id: str) -> Dict[str, Any]:
        with self._lock:
            meta = self._batch_file(batch_id)
            if not meta.is_file():
                raise BatchNotFoundError(batch_id)
            data = json.loads(meta.read_text(encoding="utf-8"))
            data["export_profile"] = normalize_export_profile(data.get("export_profile"))
            if not data["export_profile"].get("output_dir"):
                data["export_profile"]["output_dir"] = str(self._batch_dir(batch_id) / "exports")
            self._batches[batch_id] = data
            return copy.deepcopy(data)

    def reload_all(self) -> None:
        with self._lock:
            self._load_existing()

    def list_batches(self) -> List[Dict[str, Any]]:
        with self._lock:
            self._load_existing()
            out = []
            for batch in sorted(self._batches.values(), key=lambda b: b.get("updated_at", ""), reverse=True):
                out.append({
                    "batch_id": batch["batch_id"],
                    "name": batch.get("name"),
                    "source_dir": batch.get("source_dir"),
                    "created_at": batch.get("created_at"),
                    "updated_at": batch.get("updated_at"),
                    "counts": self.compute_counts(batch),
                    "export_profile": copy.deepcopy(batch.get("export_profile")),
                })
            return out

    @staticmethod
    def compute_counts(batch: Dict[str, Any]) -> Dict[str, int]:
        counts = {
            "total": 0,
            "pending": 0,
            "matted": 0,
            "prepared": 0,
            "preview_ready": 0,
            "needs_fix": 0,
            "approved_locked": 0,
            "exported": 0,
            "failed": 0,
            "locked": 0,
        }
        for item in batch.get("items") or []:
            counts["total"] += 1
            st = str(item.get("status") or "pending")
            if st in counts:
                counts[st] += 1
            if item.get("locked"):
                counts["locked"] += 1
        return counts

    def create_or_rescan_batch(
        self,
        source_dir: Union[str, Path],
        *,
        name: Optional[str] = None,
        batch_id: Optional[str] = None,
        export_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a new batch or idempotently rescan an existing folder without touching locked items."""
        folder = Path(source_dir).expanduser().resolve()
        images = scan_portrait_folder(folder)
        if not images:
            raise ValueError(f"文件夹中未找到支持的立绘图片（png/jpg/jpeg/webp）：{folder}")

        now = _now_iso()
        with self._lock:
            if batch_id:
                batch = self._require_batch(batch_id)
                if export_profile is not None:
                    prof = normalize_export_profile({**batch["export_profile"], **export_profile})
                    if not prof.get("output_dir"):
                        prof["output_dir"] = str(self._batch_dir(batch_id) / "exports")
                    batch["export_profile"] = prof
            else:
                bid = f"b_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
                bdir = self._batch_dir(bid)
                bdir.mkdir(parents=True, exist_ok=True)
                prof = normalize_export_profile(export_profile)
                if not prof.get("output_dir"):
                    prof["output_dir"] = str(bdir / "exports")
                batch = {
                    "version": 1,
                    "batch_id": bid,
                    "name": str(name or folder.name),
                    "source_dir": str(folder),
                    "created_at": now,
                    "updated_at": now,
                    "export_profile": prof,
                    "matte_overview_path": None,
                    "last_batch_export": None,
                    "items": [],
                }
                self._batches[bid] = batch

            existing_by_name = {it["filename"]: it for it in batch["items"]}
            existing_by_sha = {it["source_sha256"]: it for it in batch["items"] if it.get("source_sha256")}
            used_chars = {it["character"] for it in batch["items"]}
            added = 0
            skipped_locked = 0

            for img in images:
                sha = _sha256_file(img)
                existing = existing_by_name.get(img.name) or existing_by_sha.get(sha)
                if existing is not None:
                    if existing.get("locked"):
                        skipped_locked += 1
                        continue
                    # Unlocked existing item: preserve progress if source unchanged
                    if existing.get("source_sha256") != sha:
                        existing["source_path"] = str(img)
                        existing["source_sha256"] = sha
                        existing["status"] = "pending"
                        existing["matted_path"] = None
                        existing["matte_check_path"] = None
                        existing["job_id"] = None
                        existing["error"] = None
                        existing["updated_at"] = now
                    continue

                base_char = sanitize_slug(img.name)
                char_name = base_char
                suffix = 2
                while char_name in used_chars:
                    char_name = f"{base_char}_{suffix}"
                    suffix += 1
                used_chars.add(char_name)
                idx = len(batch["items"]) + 1
                item_id = f"{idx:03d}_{char_name}"
                batch["items"].append({
                    "item_id": item_id,
                    "character": char_name,
                    "display_name": img.stem,
                    "filename": img.name,
                    "source_path": str(img),
                    "source_sha256": sha,
                    "matted_path": None,
                    "matte_check_path": None,
                    "job_id": None,
                    "status": "pending",
                    "locked": False,
                    "locked_revision": None,
                    "locked_rig_version": None,
                    "locked_at": None,
                    "fix_note": "",
                    "fix_requested_at": None,
                    "error": None,
                    "last_export": None,
                    "created_at": now,
                    "updated_at": now,
                })
                added += 1

            batch["updated_at"] = now
            self._persist(batch)
            res = copy.deepcopy(batch)
            res["rescan_summary"] = {"added": added, "skipped_locked": skipped_locked, "total": len(batch["items"])}
            return res

    def get_batch(self, batch_id: str) -> Dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._require_batch(batch_id))

    def get_item(self, batch_id: str, item_id: str) -> Dict[str, Any]:
        with self._lock:
            batch = self._require_batch(batch_id)
            for item in batch["items"]:
                if item["item_id"] == item_id or item.get("job_id") == item_id:
                    return copy.deepcopy(item)
            raise BatchItemNotFoundError(batch_id, item_id)

    def find_item_by_job_id(self, job_id: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        with self._lock:
            for bid, batch in self._batches.items():
                for item in batch.get("items") or []:
                    if item.get("job_id") == job_id:
                        return bid, copy.deepcopy(item)
            return None

    def update_profile(self, batch_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            batch = self._require_batch(batch_id)
            merged = {**batch["export_profile"], **(payload or {})}
            prof = normalize_export_profile(merged)
            if not prof.get("output_dir"):
                prof["output_dir"] = str(self._batch_dir(batch_id) / "exports")
            batch["export_profile"] = prof
            batch["updated_at"] = _now_iso()
            self._persist(batch)
            return copy.deepcopy(batch)

    def update_item(
        self,
        batch_id: str,
        item_id: str,
        patch: Dict[str, Any],
        *,
        allow_locked: bool = False,
    ) -> Dict[str, Any]:
        with self._lock:
            batch = self._require_batch(batch_id)
            target = None
            for item in batch["items"]:
                if item["item_id"] == item_id or item.get("job_id") == item_id:
                    target = item
                    break
            if target is None:
                raise BatchItemNotFoundError(batch_id, item_id)
            if target.get("locked") and not allow_locked:
                raise BatchItemLockedError(target["item_id"])
            for k, v in patch.items():
                if k in ("item_id", "created_at"):
                    continue
                target[k] = v
            target["updated_at"] = _now_iso()
            batch["updated_at"] = target["updated_at"]
            self._persist(batch)
            return copy.deepcopy(target)

    def lock_item(
        self,
        batch_id: str,
        item_id: str,
        *,
        revision: Optional[int] = None,
        rig_version: Optional[int] = None,
    ) -> Dict[str, Any]:
        now = _now_iso()
        return self.update_item(
            batch_id,
            item_id,
            {
                "locked": True,
                "status": "approved_locked",
                "locked_revision": revision,
                "locked_rig_version": rig_version,
                "locked_at": now,
                "fix_note": "",
                "error": None,
            },
            allow_locked=True,
        )

    def unlock_item(self, batch_id: str, item_id: str) -> Dict[str, Any]:
        with self._lock:
            item = self.get_item(batch_id, item_id)
            next_status = "preview_ready" if item.get("job_id") else "pending"
            return self.update_item(
                batch_id,
                item_id,
                {
                    "locked": False,
                    "status": next_status,
                    "locked_at": None,
                },
                allow_locked=True,
            )

    def reject_item(self, batch_id: str, item_id: str, fix_note: str) -> Dict[str, Any]:
        note = str(fix_note or "").strip()
        if not note:
            raise ValueError("打回待修请填写一句修改说明（例如：右脚有灰边 / 后摆露底）。")
        with self._lock:
            item = self.get_item(batch_id, item_id)
            if item.get("locked"):
                raise BatchItemLockedError(
                    item["item_id"],
                    "该角色已确认锁定，请先点击「解锁」后再打回待修。",
                )
            return self.update_item(
                batch_id,
                item_id,
                {
                    "status": "needs_fix",
                    "fix_note": note,
                    "fix_requested_at": _now_iso(),
                },
                allow_locked=False,
            )

    def get_fix_list(self, batch_id: str) -> List[Dict[str, Any]]:
        """Return ONLY unlocked items explicitly marked `needs_fix`."""
        with self._lock:
            batch = self._require_batch(batch_id)
            return [
                copy.deepcopy(it)
                for it in batch["items"]
                if it.get("status") == "needs_fix" and not it.get("locked")
            ]

    def record_batch_export(self, batch_id: str, summary: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            batch = self._require_batch(batch_id)
            batch["last_batch_export"] = copy.deepcopy(summary)
            batch["updated_at"] = _now_iso()
            self._persist(batch)
            return copy.deepcopy(batch)

    def build_matte_overview(self, batch_id: str) -> Optional[Path]:
        with self._lock:
            batch = self._require_batch(batch_id)
            out = self._batch_dir(batch_id) / "matte-overview.jpg"
            rendered = render_matte_overview(batch["items"], out)
            if rendered is not None:
                batch["matte_overview_path"] = str(rendered)
                batch["updated_at"] = _now_iso()
                self._persist(batch)
            return rendered


def export_catalog_metadata() -> Dict[str, Any]:
    return {
        "default_profile": copy.deepcopy(DEFAULT_EXPORT_PROFILE),
        "size_presets": copy.deepcopy(SIZE_PRESETS),
        "format_options": copy.deepcopy(FORMAT_OPTIONS),
    }
