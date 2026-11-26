"""RGBA portrait loader and automatic <= 2400px long-edge downscaler."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from PIL import Image
import numpy as np

MAX_LONG_EDGE = 2400
MAX_LONG_EDGE_4K = 4096


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def preprocess_portrait(
    image_path: Union[str, Path],
    out_path: Union[str, Path],
    max_long_edge: int = MAX_LONG_EDGE,
) -> Dict[str, Any]:
    """Load an RGBA portrait; downscale proportionally with LANCZOS if long edge > max_long_edge.

    Returns a structured dict containing `resized`, `original_size`, `new_size`, `scale`,
    `source_sha256`, `output_sha256`, `output_path`, and `notice`.
    """
    src = Path(image_path).resolve()
    dst = Path(out_path).resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Portrait image not found: {src}")
    if max_long_edge <= 0:
        raise ValueError(f"max_long_edge must be positive, got {max_long_edge}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    source_sha = _sha256_file(src)

    with Image.open(src) as im:
        rgba = im.convert("RGBA")
        orig_w, orig_h = rgba.size
        long_edge = max(orig_w, orig_h)
        resized = long_edge > max_long_edge
        if resized:
            scale = max_long_edge / float(long_edge)
            new_w = max(1, int(round(orig_w * scale)))
            new_h = max(1, int(round(orig_h * scale)))
            if max(new_w, new_h) > max_long_edge:
                if new_w >= new_h:
                    new_w = max_long_edge
                else:
                    new_h = max_long_edge
            rgba = rgba.resize((new_w, new_h), Image.Resampling.LANCZOS)
            notice: Optional[str] = (
                f"Input portrait {orig_w}x{orig_h} exceeded {max_long_edge}px long edge; "
                f"automatically downscaled proportionally via LANCZOS to {new_w}x{new_h}."
            )
        else:
            scale = 1.0
            new_w, new_h = orig_w, orig_h
            notice = None

        arr = np.array(rgba, dtype=np.uint8, copy=True)
        arr[arr[..., 3] == 0, :3] = 0
        rgba = Image.fromarray(arr)
        rgba.save(dst, format="PNG")

    return {
        "source_path": str(src),
        "output_path": str(dst),
        "resized": resized,
        "original_size": [orig_w, orig_h],
        "new_size": [new_w, new_h],
        "scale": round(scale, 6),
        "source_sha256": source_sha,
        "output_sha256": _sha256_file(dst),
        "notice": notice,
    }
