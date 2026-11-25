#!/usr/bin/env python3
"""SAM 2 original-resolution module segmentation CLI. Heavy deps stay out of src/."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
MODELS = Path(os.environ.get("MODELS_DIR", REPO / ".studio-work/models"))
DEFAULT_WEIGHT = MODELS / "sam2.1_b.pt"

COLORS = [
    (220, 70, 70), (70, 180, 90), (70, 120, 220), (230, 180, 50),
    (180, 80, 200), (50, 200, 200), (230, 110, 40), (90, 90, 220),
    (220, 80, 140), (140, 200, 60), (80, 160, 230), (200, 160, 90),
]

# Final connected-component count uses CC_COUNT_MIN_PX. Do not raise max_cc or lower this to pass.
CC_COUNT_MIN_PX = 80
HOLE_MAX_PX = 300
PRE_WS_MIN_PX = 120
DEFAULT_MAX_CC = 4
GRAY = (145, 145, 145)
SHEET_MAX_ROW_PX = 1000


def _bind_cache_dirs() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(MODELS / "torch"))
    os.environ.setdefault("HF_HOME", str(MODELS / "hf"))
    # Ultralytics reads YOLO_CONFIG_DIR and appends /Ultralytics. Do not delete any existing user config.
    os.environ.setdefault("YOLO_CONFIG_DIR", str(MODELS))
    os.environ.setdefault("ULTRALYTICS_CONFIG_DIR", str(MODELS / "ultralytics"))
    os.environ.setdefault("PIP_CACHE_DIR", str(MODELS / "pip"))


def emit_grid(image_path: Path, out_path: Path, step: int = 50) -> Path:
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    with Image.open(image_path) as im:
        rgba = im.convert("RGBA")
    w, h = rgba.size
    gray = Image.new("RGBA", (w, h), (125, 125, 125, 255))
    gray.alpha_composite(rgba)
    canvas = gray.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        font_s = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
    except Exception:
        font = ImageFont.load_default()
        font_s = font
    for x in range(0, w + 1, step):
        major = x % (step * 2) == 0
        draw.line([(x, 0), (x, h)], fill=(40, 40, 40) if major else (90, 90, 90), width=2 if major else 1)
        draw.text((x + 2, 2), str(x), fill=(20, 20, 20), font=font if major else font_s)
    for y in range(0, h + 1, step):
        major = y % (step * 2) == 0
        draw.line([(0, y), (w, y)], fill=(40, 40, 40) if major else (90, 90, 90), width=2 if major else 1)
        if y:
            draw.text((2, y + 2), str(y), fill=(20, 20, 20), font=font if major else font_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=95)
    return out_path


def _erode(mask, r):
    import cv2
    import numpy as np
    if r <= 0:
        return mask.copy()
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.erode(mask.astype(np.uint8), k).astype(bool)


def _dilate(mask, r):
    import cv2
    import numpy as np
    if r <= 0:
        return mask.copy()
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(mask.astype(np.uint8), k).astype(bool)


def _fill_small_holes(mask, max_hole=300):
    import cv2
    import numpy as np
    h, w = mask.shape
    inv = (~mask).astype(np.uint8)
    num, cc = cv2.connectedComponents(inv, connectivity=8)
    out = mask.copy()
    for k in range(1, num):
        comp = cc == k
        ys, xs = np.where(comp)
        if ys.size == 0:
            continue
        if ys.min() == 0 or xs.min() == 0 or ys.max() == h - 1 or xs.max() == w - 1:
            continue
        if int(comp.sum()) < max_hole:
            out[comp] = True
    return out


def _filter_small_cc(mask, min_px=120, max_keep=4):
    import cv2
    import numpy as np
    num, cc = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    parts = []
    for k in range(1, num):
        s = int((cc == k).sum())
        if s >= min_px:
            parts.append((s, k))
    parts.sort(reverse=True)
    out = np.zeros_like(mask)
    for _, k in parts[:max_keep]:
        out[cc == k] = True
    return out


def _count_cc(mask, min_px=80):
    import cv2
    import numpy as np
    num, cc = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    n = 0
    for k in range(1, num):
        if int((cc == k).sum()) >= min_px:
            n += 1
    return n



def parse_chain_band(value, label: str):
    """Return normalized {pts, radius_px} or None. Invalid values raise; never drop silently."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("%s chain_band 必须是对象 {pts, radius_px}" % label)
    if "pts" not in value or "radius_px" not in value:
        raise ValueError("%s chain_band 必须包含 pts 与 radius_px" % label)
    pts_raw = value.get("pts")
    radius = value.get("radius_px")
    if not isinstance(pts_raw, list) or len(pts_raw) < 2:
        raise ValueError("%s chain_band.pts 至少需要两个点" % label)
    pts = []
    for i, pt in enumerate(pts_raw):
        if not isinstance(pt, (list, tuple)) or len(pt) != 2:
            raise ValueError("%s chain_band.pts[%d] 必须是 [x,y]" % (label, i))
        try:
            x = float(pt[0]); y = float(pt[1])
        except (TypeError, ValueError) as exc:
            raise ValueError("%s chain_band.pts[%d] 不是有限坐标" % (label, i)) from exc
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("%s chain_band.pts[%d] 不是有限坐标" % (label, i))
        pts.append([x, y])
    distinct = {(p[0], p[1]) for p in pts}
    if len(distinct) < 2:
        raise ValueError("%s chain_band.pts 至少需要两个不同点" % label)
    try:
        radius_f = float(radius)
    except (TypeError, ValueError) as exc:
        raise ValueError("%s chain_band.radius_px 必须是有限正数" % label) from exc
    if not math.isfinite(radius_f) or radius_f <= 0:
        raise ValueError("%s chain_band.radius_px 必须是有限正数" % label)
    return {"pts": pts, "radius_px": radius_f}


def polyline_band_mask(h, w, pts, radius):
    import numpy as np
    yy, xx = np.ogrid[:h, :w]
    dist = np.full((h, w), np.inf, dtype=np.float64)
    arr = np.asarray(pts, dtype=np.float64)
    for i in range(len(arr) - 1):
        x0, y0 = arr[i]
        x1, y1 = arr[i + 1]
        dx = x1 - x0
        dy = y1 - y0
        seg2 = dx * dx + dy * dy
        if seg2 == 0:
            d = np.hypot(xx - x0, yy - y0)
        else:
            t = np.clip(((xx - x0) * dx + (yy - y0) * dy) / seg2, 0.0, 1.0)
            d = np.hypot(xx - (x0 + t * dx), yy - (y0 + t * dy))
        dist = np.minimum(dist, d)
    return dist <= float(radius)


def resolve_target_chain_band(module, piece, target_id):
    piece = piece if isinstance(piece, dict) else {}
    if "chain_band" in piece:
        return parse_chain_band(piece.get("chain_band"), "piece %s" % target_id)
    if "chain_band" in module:
        return parse_chain_band(module.get("chain_band"), "module %s" % module.get("id"))
    return None


def _apply_excludes(mask, item, h, w):
    import numpy as np
    out = mask.copy()
    yy, xx = np.ogrid[:h, :w]
    for box in item.get("exclude_boxes") or []:
        x1, y1, x2, y2 = [int(v) for v in box]
        out[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = False
    for plane in item.get("exclude_halfplanes") or []:
        axis = plane.get("axis")
        op = plane.get("op")
        value = float(plane.get("value"))
        coord = xx if axis == "x" else yy
        if op == ">=":
            out &= ~(coord >= value)
        elif op == "<=":
            out &= ~(coord <= value)
        elif op == ">":
            out &= ~(coord > value)
        elif op == "<":
            out &= ~(coord < value)
    return out


def _subprompts(item):
    if item.get("sub_prompts"):
        return item["sub_prompts"]
    return [{"box": item.get("box"), "pos": item.get("pos", []), "neg": item.get("neg", [])}]


class SegmentPromptError(ValueError):
    def __init__(self, errors: List[Dict[str, Any]]):
        self.errors = errors
        msg = errors[0]["message"] if errors else "prompt failed"
        super().__init__(msg)


def _finite_xy_pair(pt, label: str):
    if not isinstance(pt, (list, tuple)) or len(pt) < 2:
        raise ValueError("%s 坐标非法" % label)
    x, y = float(pt[0]), float(pt[1])
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("%s 坐标必须有限" % label)
    return x, y


def validate_prompt_points(pos, neg, w: int, h: int, *, label: str = "prompt") -> None:
    pos = list(pos or [])
    neg = list(neg or [])
    for i, pt in enumerate(pos):
        x, y = _finite_xy_pair(pt, "%s.pos[%d]" % (label, i))
        if not (0 <= x < w and 0 <= y < h):
            raise ValueError("%s.pos[%d] 越界: (%s, %s) 不在 [0,%s)x[0,%s)" % (label, i, x, y, w, h))
    for i, pt in enumerate(neg):
        x, y = _finite_xy_pair(pt, "%s.neg[%d]" % (label, i))
        if not (0 <= x < w and 0 <= y < h):
            raise ValueError("%s.neg[%d] 越界: (%s, %s) 不在 [0,%s)x[0,%s)" % (label, i, x, y, w, h))
        for j, ppt in enumerate(pos):
            px, py = _finite_xy_pair(ppt, "%s.pos[%d]" % (label, j))
            if abs(px - x) < 1e-6 and abs(py - y) < 1e-6:
                raise ValueError("%s.neg[%d] 与 pos[%d] 重合" % (label, i, j))
            if int(round(px)) == int(round(x)) and int(round(py)) == int(round(y)):
                raise ValueError("%s.neg[%d] 与 pos[%d] 整数像素重合" % (label, i, j))


def sam_attempts(box, pos, neg):
    pos = list(pos or [])
    neg = list(neg or [])
    if neg:
        labels = [1] * len(pos) + [0] * len(neg)
        pts = pos + neg
        if box:
            return [("box+all", box, pts, labels)]
        return [("pos+neg", None, pts, labels)]
    attempts = []
    if box and pos:
        attempts.append(("box+pos", box, pos[:5], [1] * len(pos[:5])))
    if box:
        attempts.append(("box-only", box, None, None))
    elif pos:
        attempts.append(("pos-only", None, pos[:5], [1] * len(pos[:5])))
    return attempts


def compute_roi_box(box, pos, neg, w, h, *, roi_mode="auto"):
    w, h = int(w), int(h)
    if roi_mode == "full" or w <= 0 or h <= 0:
        return (0, 0, w, h)
    if box:
        x1, y1, x2, y2 = [float(v) for v in box]
        bx0, bx1 = min(x1, x2), max(x1, x2)
        by0, by1 = min(y1, y2), max(y1, y2)
    else:
        pts = list(pos or []) + list(neg or [])
        if not pts:
            return (0, 0, w, h)
        xs = [float(p[0]) for p in pts]
        ys = [float(p[1]) for p in pts]
        bx0, bx1 = min(xs), max(xs)
        by0, by1 = min(ys), max(ys)
    bw = max(1.0, bx1 - bx0)
    bh = max(1.0, by1 - by0)
    pad = max(48, int(round(0.25 * max(bw, bh))))
    x0 = max(0, int(math.floor(bx0)) - pad)
    y0 = max(0, int(math.floor(by0)) - pad)
    x1 = min(w, int(math.ceil(bx1)) + pad)
    y1 = min(h, int(math.ceil(by1)) + pad)

    def _grow(a0, a1, min_size, limit):
        size = a1 - a0
        if limit <= min_size:
            return 0, limit
        if size >= min_size:
            return a0, a1
        a0 -= (min_size - size) // 2
        a1 = a0 + min_size
        if a0 < 0:
            a1 -= a0
            a0 = 0
        if a1 > limit:
            a0 = max(0, limit - min_size)
            a1 = limit
        return a0, a1

    x0, x1 = _grow(x0, x1, 128, w)
    y0, y1 = _grow(y0, y1, 128, h)
    if (x1 - x0) * (y1 - y0) >= 0.70 * w * h:
        return (0, 0, w, h)
    return (int(x0), int(y0), int(x1), int(y1))


def map_roi_mask_to_full(roi_mask, roi_box, full_shape):
    import numpy as np
    H, W = int(full_shape[0]), int(full_shape[1])
    x0, y0, x1, y1 = [int(v) for v in roi_box]
    out = np.zeros((H, W), dtype=bool)
    rw, rh = x1 - x0, y1 - y0
    if rw <= 0 or rh <= 0:
        return out
    mask = np.asarray(roi_mask).astype(bool)
    if mask.shape != (rh, rw):
        try:
            import cv2
            mask = cv2.resize(mask.astype("uint8"), (rw, rh), interpolation=cv2.INTER_NEAREST).astype(bool)
        except Exception:
            from PIL import Image as _Im
            im = _Im.fromarray((mask.astype("uint8") * 255))
            mask = np.asarray(im.resize((rw, rh), _Im.NEAREST)) > 127
    out[y0:y1, x0:x1] = mask[:rh, :rw]
    return out


def prompt_cache_key(image_sha256: str, weight_name: str, item: Dict[str, Any], sp: Dict[str, Any], prior_sha256: Optional[str]) -> str:
    payload = {
        "image_sha256": image_sha256,
        "weight_name": weight_name,
        "roi_mode": sp.get("roi_mode", item.get("roi_mode", "auto")),
        "roi_box": sp.get("roi_box"),
        "sub_prompts": {
            "box": sp.get("box", item.get("box")),
            "pos": sp.get("pos", item.get("pos", [])),
            "neg": sp.get("neg", item.get("neg", [])),
        },
        "excludes": {
            "exclude_boxes": item.get("exclude_boxes"),
            "exclude_halfplanes": item.get("exclude_halfplanes"),
        },
        "prior_sha256": prior_sha256,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


def safe_cache_png(base: Path, ident: str, digest12: str) -> Path:
    require_ident(ident, "cache id")
    if not re.fullmatch(r"[0-9a-f]{12}", digest12):
        raise ValueError("非法缓存指纹: %s" % digest12)
    root = base.resolve()
    path = (root / ("%s.%s.png" % (ident, digest12))).resolve()
    path.relative_to(root)
    return path


def _prior_file(prior_dir: Path, item: Dict[str, Any]) -> Optional[Path]:
    fname = item.get("prior_mask")
    if fname is None or fname == "":
        ident = item["id"]
    else:
        raw = str(fname)
        if "/" in raw or "\\" in raw or raw != Path(raw).name:
            raise ValueError("先验掩码路径穿越被拒绝: %s" % raw)
        ident = raw[:-4] if raw.lower().endswith(".png") else raw
        require_ident(ident, "prior_mask")
    path = safe_png_path(Path(prior_dir), ident)
    return path if path.is_file() else None


def load_prior_mask(prior_dir: Optional[Path], item: Dict[str, Any], w: int, h: int):
    import numpy as np
    from PIL import Image as _Im
    if prior_dir is None:
        return None, None, None
    path = _prior_file(Path(prior_dir), item)
    if path is None:
        return None, None, None
    with _Im.open(path) as im:
        if im.size != (w, h):
            raise ValueError("先验掩码尺寸 %s 与原图 %dx%d 不一致: %s" % (im.size, w, h, path))
        mask = np.asarray(im.convert("L")) > 127
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    ys, xs = np.nonzero(mask)
    box = None
    if ys.size:
        box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    return mask, digest, box


def _load_sam(weight: Path):
    from ultralytics import SAM
    return SAM(str(weight))


def _predict_sam_mask(model, source, device, bboxes, points, labels):
    kw = dict(source=str(source), verbose=False, device=device)
    if bboxes is not None:
        kw["bboxes"] = bboxes
    if points:
        kw["points"] = [points]
        kw["labels"] = [labels]
    return model.predict(**kw)


def _run_sam(rgb, fg, items, cache_dir: Path, weight: Path, *, no_cache: bool = False,
             image_sha256: str = "", prior_dir: Optional[Path] = None):
    from PIL import Image
    import numpy as np
    h, w = fg.shape
    cache_dir.mkdir(parents=True, exist_ok=True)
    model = None

    def get_model():
        nonlocal model
        if model is None:
            model = _load_sam(weight)
        return model

    def extract(results, target_hw):
        r0 = results[0]
        if getattr(r0, "masks", None) is None or r0.masks is None:
            return None
        data = r0.masks.data
        arr = data.detach().cpu().numpy() if hasattr(data, "detach") else np.asarray(data)
        if arr.size == 0:
            return None
        mask = arr[0] > 0.5 if arr.ndim == 3 else arr > 0.5
        th, tw = target_hw
        if mask.shape != (th, tw):
            mask = map_roi_mask_to_full(mask, (0, 0, tw, th), (th, tw))
        return mask

    def predict_one(name, box, pos, neg, roi_mode):
        pos = list(pos or [])
        neg = list(neg or [])
        validate_prompt_points(pos, neg, w, h, label=name)
        roi = compute_roi_box(box, pos, neg, w, h, roi_mode=roi_mode or "auto")
        x0, y0, x1, y1 = roi
        roi_rgb = rgb[y0:y1, x0:x1]
        tmp = cache_dir / ("sam_roi_%s.png" % name)
        Image.fromarray(roi_rgb).save(tmp)

        def shift_pt(pt):
            return [float(pt[0]) - x0, float(pt[1]) - y0]

        def shift_box(b):
            if not b:
                return None
            bx1, by1, bx2, by2 = [float(v) for v in b]
            return [bx1 - x0, by1 - y0, bx2 - x0, by2 - y0]

        sbox = shift_box(box)
        spos = [shift_pt(p) for p in pos]
        sneg = [shift_pt(p) for p in neg]
        for px, py in spos + sneg:
            if not (0 <= px < (x1 - x0) and 0 <= py < (y1 - y0)):
                raise ValueError("%s 提示点未落在 ROI 内" % name)
        attempts = sam_attempts(sbox, spos, sneg)
        if neg and not attempts:
            raise SegmentPromptError([{
                "id": name,
                "code": "neg_prompt_failed",
                "message": "neg prompt has no legal SAM attempt",
            }])
        last_exc = None
        for tag, abox, apts, alabels in attempts:
            bboxes = [abox] if abox is not None else None
            try:
                results = _predict_sam_mask(get_model(), tmp, "mps", bboxes, apts, alabels)
            except Exception as exc:
                last_exc = exc
                if neg:
                    raise SegmentPromptError([{
                        "id": name,
                        "code": "neg_prompt_failed",
                        "message": "%s: %s" % (tag, exc),
                    }])
                continue
            m = extract(results, (y1 - y0, x1 - x0))
            if m is not None and int(m.sum()) > 40:
                return map_roi_mask_to_full(m, roi, fg.shape) & fg
            if neg:
                raise SegmentPromptError([{
                    "id": name,
                    "code": "neg_prompt_failed",
                    "message": "%s 返回空或有效像素<=40" % tag,
                }])
        if neg:
            raise SegmentPromptError([{
                "id": name,
                "code": "neg_prompt_failed",
                "message": str(last_exc) if last_exc else "neg prompt produced no mask",
            }])
        return np.zeros(fg.shape, dtype=bool)

    raw = {}
    for item in items:
        name = item["id"]
        prior_mask, prior_sha, prior_box = load_prior_mask(prior_dir, item, w, h)
        union = np.zeros(fg.shape, dtype=bool)
        for sp in _subprompts(item):
            box = sp.get("box") or item.get("box") or prior_box
            pos = sp.get("pos", item.get("pos", []))
            neg = sp.get("neg", item.get("neg", []))
            roi_mode = sp.get("roi_mode", item.get("roi_mode", "auto"))
            validate_prompt_points(pos, neg, w, h, label=name)
            roi = compute_roi_box(box, pos, neg, w, h, roi_mode=roi_mode or "auto")
            digest = prompt_cache_key(
                image_sha256,
                Path(weight).name,
                item,
                {"box": box, "pos": pos, "neg": neg, "roi_mode": roi_mode, "roi_box": list(roi)},
                prior_sha,
            )
            cached = safe_cache_png(cache_dir, name, digest)
            if cached.is_file() and not no_cache:
                from PIL import Image as _Image
                union |= (np.asarray(_Image.open(cached).convert("L")) > 127) & fg
                continue
            m = predict_one(name, box, pos, neg, roi_mode)
            if box:
                x1, y1, x2, y2 = [int(v) for v in box]
                clip = np.zeros_like(union)
                pad = 12
                clip[max(0, y1 - pad):min(fg.shape[0], y2 + pad), max(0, x1 - pad):min(fg.shape[1], x2 + pad)] = True
                m &= clip
            union |= m
            from PIL import Image as _Image
            _Image.fromarray((m.astype(np.uint8) * 255)).save(cached)
        union = _apply_excludes(union, item, *fg.shape)
        raw[name] = union & fg
    return raw


def _watershed_refine(rgb, fg, seeds):
    import cv2
    import numpy as np
    from scipy import ndimage as ndi
    from skimage.segmentation import watershed
    from skimage.filters import sobel
    h, w = fg.shape
    marker = np.zeros((h, w), np.int32)
    for i, mask in enumerate(seeds, start=1):
        core = _erode(mask, 2)
        if int(core.sum()) < 16:
            core = _erode(mask, 1)
        if int(core.sum()) < 16:
            core = mask
        marker[core] = i
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    grad = sobel(lab[..., 0]) + 0.5 * sobel(lab[..., 1]) + 0.5 * sobel(lab[..., 2])
    labels = watershed(grad, markers=marker, mask=fg)
    n = len(seeds)
    for i in range(1, n + 1):
        m = labels == i
        num, cc = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)
        for k in range(1, num):
            comp = cc == k
            if int(comp.sum()) >= 150:
                continue
            border = _dilate(comp, 1) & ~comp & fg
            neigh = labels[border]
            neigh = neigh[neigh > 0]
            if neigh.size:
                vals, cnts = np.unique(neigh, return_counts=True)
                labels[comp] = int(vals[cnts.argmax()])
    leftover = fg & (labels <= 0)
    if leftover.any() and (labels > 0).any():
        _, (iy, ix) = ndi.distance_transform_edt(labels <= 0, return_indices=True)
        labels[leftover] = labels[iy[leftover], ix[leftover]]
        labels[~fg] = 0
    return labels


def _encode_mask(mask) -> Dict[str, str]:
    from PIL import Image
    import numpy as np
    import io
    arr = np.where(np.asarray(mask), 255, 0).astype(np.uint8)
    im = Image.fromarray(arr)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    raw = buf.getvalue()
    return {
        "mask_png_base64": base64.b64encode(raw).decode("ascii"),
        "mask_sha256": hashlib.sha256(raw).hexdigest(),
    }


IDENT_RE = re.compile(r"[a-z][a-z0-9_]{0,47}")


def require_ident(value: Any, label: str) -> str:
    if not isinstance(value, str) or not IDENT_RE.fullmatch(value):
        raise ValueError("%s 不是合法标识: %r" % (label, value))
    return value


def safe_png_path(base: Path, name: str) -> Path:
    require_ident(name, "path")
    root = base.resolve()
    path = (root / (name + ".png")).resolve()
    path.relative_to(root)
    return path


def _has_piece_prompts(obj: Dict[str, Any]) -> bool:
    return any(obj.get(k) for k in ("box", "pos", "neg", "sub_prompts", "exclude_boxes", "exclude_halfplanes"))


def expand_segment_targets(modules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One SAM target per piece. Multi-piece modules require per-piece prompts."""
    targets: List[Dict[str, Any]] = []
    seen = set()
    def take_id(value, label):
        ident = require_ident(value, label)
        if ident in seen:
            raise ValueError("重复标识: " + ident)
        seen.add(ident)
        return ident
    for module in modules:
        mid = take_id(module.get("id"), "module.id")
        pieces = module.get("pieces") or []
        if len(pieces) > 1:
            for piece in pieces:
                if not isinstance(piece, dict) or not _has_piece_prompts(piece):
                    raise ValueError("module %s 有多个piece但未提供分piece提示: %s" % (
                        mid, (piece or {}).get("id")))
                pid = take_id(piece.get("id"), "piece.id")
                target = dict(module)
                target.update(piece)
                target["id"] = pid
                target["module_id"] = mid
                target["draw_order"] = int(piece.get("draw_order", module.get("draw_order", 0)))
                target["chain_band"] = resolve_target_chain_band(module, piece, pid)
                targets.append(target)
        elif len(pieces) == 1:
            piece = pieces[0]
            src = piece if _has_piece_prompts(piece) else module
            if not _has_piece_prompts(src):
                raise ValueError("module %s 缺少分割提示" % mid)
            pid = take_id(piece.get("id"), "piece.id") if piece.get("id") else mid
            if piece.get("id") is None:
                # module id already recorded
                pass
            target = dict(module)
            target.update(src)
            target["id"] = pid
            target["module_id"] = mid
            target["draw_order"] = int(piece.get("draw_order", module.get("draw_order", 0)))
            target["chain_band"] = resolve_target_chain_band(module, piece, pid)
            targets.append(target)
        else:
            if not _has_piece_prompts(module):
                raise ValueError("module %s 缺少分割提示" % mid)
            target = dict(module)
            target["id"] = mid
            target["module_id"] = mid
            target["chain_band"] = resolve_target_chain_band(module, None, mid)
            targets.append(target)
    return sorted(targets, key=lambda m: int(m.get("draw_order", 0)))


def _alpha_gray(src, mask, color):
    from PIL import Image
    import numpy as np
    h, w = mask.shape
    base = np.full((h, w, 3), 145, np.uint8)
    overlay = src.copy()
    tint = np.zeros_like(overlay)
    tint[..., 0], tint[..., 1], tint[..., 2], tint[..., 3] = (*color, 90)
    vis = mask & (src[..., 3] > 0)
    out = Image.fromarray(base)
    layer = Image.fromarray(overlay)
    out.paste(Image.fromarray(src), (0, 0), layer)
    return out


# --- inserted helpers ---

def masked_rgba(src, mask):
    import numpy as np
    out = np.asarray(src).copy()
    out[~np.asarray(mask, dtype=bool)] = 0
    return out


def gray_composite(rgba):
    from PIL import Image
    import numpy as np
    arr = np.asarray(rgba)
    h, w = arr.shape[:2]
    base = Image.new("RGBA", (w, h), (*GRAY, 255))
    base.alpha_composite(Image.fromarray(arr))
    return base.convert("RGB")


def _bbox(mask, pad=8, limit=None):
    import numpy as np
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    h, w = mask.shape
    return [
        max(0, int(xs.min()) - pad),
        max(0, int(ys.min()) - pad),
        min(w, int(xs.max()) + pad + 1),
        min(h, int(ys.max()) + pad + 1),
    ]


def isolate_tile(src, mask):
    """Return (source_labeled_rgb, isolated_gray_rgb) cropped to mask bbox."""
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    h, w = mask.shape
    box = _bbox(mask) or [0, 0, min(w, 120), min(h, 48)]
    x1, y1, x2, y2 = box
    src_im = Image.fromarray(src).crop((x1, y1, x2, y2))
    source = Image.new("RGBA", src_im.size, (*GRAY, 255))
    source.alpha_composite(src_im)
    iso_full = masked_rgba(src, mask)
    iso_im = Image.fromarray(iso_full).crop((x1, y1, x2, y2))
    isolated = Image.new("RGBA", iso_im.size, (*GRAY, 255))
    isolated.alpha_composite(iso_im)
    return source.convert("RGB"), isolated.convert("RGB")


def _label_bar(width, text):
    from PIL import Image, ImageDraw, ImageFont
    bar = Image.new("RGB", (width, 28), (30, 30, 30))
    draw = ImageDraw.Draw(bar)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except Exception:
        font = ImageFont.load_default()
    draw.text((8, 6), text, fill=(255, 255, 255), font=font)
    return bar


def _scale_to_width(im, max_w):
    from PIL import Image
    if im.size[0] <= max_w:
        return im
    nh = max(1, int(im.size[1] * (max_w / im.size[0])))
    return im.resize((max_w, nh), getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC))


def build_module_sheet(src, owned_items):
    """One row per piece: SOURCE | ISOLATED. Never a single 12-column strip."""
    from PIL import Image
    rows = []
    for name, mask in owned_items:
        if int(mask.sum()) == 0:
            empty = Image.new("RGB", (240, 48), GRAY)
            source, isolated = empty.copy(), empty.copy()
        else:
            source, isolated = isolate_tile(src, mask)
        col_w = max(source.size[0], isolated.size[0], 160)
        source = _pad_canvas(source, col_w)
        isolated = _pad_canvas(isolated, col_w)
        pair_w = col_w * 2
        if pair_w > SHEET_MAX_ROW_PX:
            scale = SHEET_MAX_ROW_PX / pair_w
            nw = max(1, int(col_w * scale))
            source = source.resize((nw, max(1, int(source.size[1] * scale))), getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC))
            isolated = isolated.resize((nw, max(1, int(isolated.size[1] * scale))), getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC))
            col_w = nw
        src_col = Image.new("RGB", (col_w, 28 + source.size[1]), GRAY)
        iso_col = Image.new("RGB", (col_w, 28 + isolated.size[1]), GRAY)
        src_col.paste(_label_bar(col_w, "SOURCE %s" % name), (0, 0))
        iso_col.paste(_label_bar(col_w, "ISOLATED %s" % name), (0, 0))
        src_col.paste(source, (0, 28))
        iso_col.paste(isolated, (0, 28))
        h = max(src_col.size[1], iso_col.size[1])
        row = Image.new("RGB", (col_w * 2 + 8, h), (90, 90, 90))
        row.paste(src_col, (0, 0))
        row.paste(iso_col, (col_w + 8, 0))
        rows.append(row)
    if not rows:
        return Image.new("RGB", (320, 48), GRAY)
    width = max(r.size[0] for r in rows)
    height = sum(r.size[1] for r in rows) + 8 * (len(rows) - 1)
    sheet = Image.new("RGB", (width, height), (90, 90, 90))
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.size[1] + 8
    return sheet


def _pad_canvas(im, width):
    from PIL import Image
    if im.size[0] == width:
        return im
    canvas = Image.new("RGB", (width, im.size[1]), GRAY)
    canvas.paste(im, (0, 0))
    return canvas


def allowed_region(item, h, w):
    import numpy as np
    allowed = np.ones((h, w), dtype=bool)
    allowed = _apply_excludes(allowed, item, h, w)
    band = item.get("chain_band")
    if band:
        parsed = band if isinstance(band, dict) and "pts" in band and "radius_px" in band else parse_chain_band(band, item.get("id") or "item")
        if parsed:
            allowed &= polyline_band_mask(h, w, parsed["pts"], parsed["radius_px"])
    return allowed


class OutputDirUnwritable(Exception):
    def __init__(self, path, cause=None):
        self.path = Path(path)
        self.cause = cause
        detail = "%s" % (cause,) if cause is not None else "unwritable"
        super().__init__("输出目录不可写，无法落盘隔离或失败报告: %s (%s)" % (self.path, detail))



def _archive_named(path: Path, kind: str = "prev-success") -> Dict[str, Any]:
    rec = {"from": path.name, "ok": False, "to": None, "sha256": None, "error": None, "kind": kind, "still_present": path.is_file()}
    if not path.is_file():
        rec.update({"ok": True, "skipped": True, "still_present": False})
        return rec
    try:
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        dest = path.parent / ("%s.%s-%s.json" % (path.name, kind, digest[:12]))
        n = 1
        while dest.exists():
            dest = path.parent / ("%s.%s-%s-%d.json" % (path.name, kind, digest[:12], n))
            n += 1
        hook = globals().get("_TEST_RENAME_HOOK")
        if callable(hook):
            hook(path, dest)
        path.rename(dest)
        rec.update({"ok": True, "to": str(dest), "sha256": digest, "still_present": False})
        return rec
    except OSError as exc:
        rec["error"] = str(exc)
        rec["still_present"] = path.is_file()
        return rec


def quarantine_success_analysis(out_dir: Path):
    rec = _archive_named(Path(out_dir) / "analysis.json", "prev-success")
    if rec.get("ok") and rec.get("to"):
        return Path(rec["to"])
    return None


def _report_is_success(path: Path) -> bool:
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("ok") is True
    except Exception:
        return False


CONSUMABLE_SUCCESS_NAMES = ("analysis.json", "segment-report.json", "mask-fragments.json")


def archive_consumable_files(out_dir: Path, kind: str, *, only_success_report: bool = True) -> Dict[str, Any]:
    """Archive each consumable independently. Never deletes bytes. Does not raise OSError."""
    out_dir = Path(out_dir)
    archived = []
    incomplete = []
    for name in CONSUMABLE_SUCCESS_NAMES:
        path = out_dir / name
        if not path.is_file():
            continue
        if name == "segment-report.json" and only_success_report and not _report_is_success(path):
            continue
        rec = _archive_named(path, kind)
        if rec.get("ok") and rec.get("to"):
            archived.append({"from": rec["from"], "to": rec["to"], "sha256": rec["sha256"], "kind": kind})
        elif not rec.get("ok"):
            incomplete.append({"file": name, "reason": rec.get("error"), "still_present": rec.get("still_present", True)})
    return {"archived": archived, "incomplete": incomplete, "ok": not incomplete}


def begin_segment_run(out_dir: Path) -> Dict[str, Any]:
    """Prepare out_dir and isolate previous consumable success artifacts. Never deletes bytes."""
    out_dir = Path(out_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        probe = out_dir / ".segment-write-probe"
        probe.write_bytes(b"ok")
        probe.unlink()
    except OSError as exc:
        raise OutputDirUnwritable(out_dir, exc) from exc
    isolation = archive_consumable_files(out_dir, "prev-success", only_success_report=True)
    analysis_to = None
    for item in isolation["archived"]:
        if item.get("from") == "analysis.json":
            analysis_to = item.get("to")
            break
    record = {
        "ok": isolation["ok"],
        "archived": isolation["archived"],
        "incomplete": isolation["incomplete"],
        "analysis": analysis_to,
        "out_dir": str(out_dir),
    }
    try:
        if isolation["archived"] or isolation["incomplete"]:
            with (out_dir / "success-archive.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        record["incomplete"] = list(record["incomplete"]) + [{"file": "success-archive.jsonl", "reason": str(exc), "still_present": True}]
        record["ok"] = False
    return record


def persist_failure(out_dir: Path, summary: Dict[str, Any], archive: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    try:
        return _persist_failure_inner(out_dir, summary, archive)
    except Exception as exc:
        return {
            "ok": False,
            "analysis": None,
            "persisted": False,
            "error": type(exc).__name__,
            "message": str(exc),
            "persist_error": str(exc),
            "out_dir": str(out_dir),
        }


def _persist_failure_inner(out_dir: Path, summary: Dict[str, Any], archive: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    summary = dict(summary or {})
    summary["ok"] = False
    summary["analysis"] = None
    summary.setdefault("out_dir", str(out_dir))
    try:
        leftover = archive_consumable_files(out_dir, "prev-failed", only_success_report=True)
    except Exception as exc:
        leftover = {"archived": [], "incomplete": [{"file": "*", "reason": str(exc), "still_present": True}], "ok": False}
    archived = list((archive or {}).get("archived") or []) + list(leftover.get("archived") or [])
    incomplete = []
    seen = set()
    for item in list((archive or {}).get("incomplete") or []) + list(leftover.get("incomplete") or []):
        key = item.get("file")
        if key in seen:
            continue
        seen.add(key)
        incomplete.append(item)
    summary["success_archive"] = archived
    summary["isolation_incomplete"] = incomplete
    prev = (archive or {}).get("analysis")
    for item in leftover.get("archived") or []:
        if item.get("from") == "analysis.json":
            prev = item.get("to")
    if prev:
        summary["prev_success_analysis"] = prev
    report_path = out_dir / "segment-report.json"
    report_blocked = report_path.is_file() and _report_is_success(report_path)
    if report_blocked:
        summary["persisted"] = False
        summary["persist_error"] = "refusing to overwrite unarchived success segment-report.json"
        try:
            (out_dir / "SEGMENT_FAILED").write_text(
                json.dumps({"ok": False, "error": summary.get("error"), "message": summary.get("message"), "isolation_incomplete": incomplete}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            summary["persist_error"] += "; marker: " + str(exc)
        return summary
    try:
        (out_dir / "SEGMENT_FAILED").write_text(
            json.dumps({"ok": False, "error": summary.get("error"), "message": summary.get("message"), "errors": summary.get("errors"), "isolation_incomplete": incomplete}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (out_dir / "segment-report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["persisted"] = not incomplete
        if incomplete:
            summary["persist_error"] = "isolation incomplete: " + ", ".join(i.get("file", "?") for i in incomplete)
    except OSError as exc:
        summary["persisted"] = False
        summary["persist_error"] = str(exc)
    return summary


def _replace_file(src: Path, dest: Path) -> None:
    hook = globals().get("_TEST_REPLACE_HOOK")
    if callable(hook):
        hook(src, dest)
    src.replace(dest)


def publish_success_artifacts(out_dir: Path, analysis: Optional[Dict[str, Any]], fragments: Dict[str, Any], summary: Dict[str, Any]) -> Path:
    """Stage fragments, analysis, then report. Report is the last published consumable. Staging files are diagnostic residue only."""
    out_dir = Path(out_dir)
    staging_report = out_dir / "segment-report.json.staging"
    staging_frags = out_dir / "mask-fragments.json.staging"
    staging_analysis = out_dir / "analysis.json.staging"
    staging_frags.write_text(json.dumps(fragments, ensure_ascii=False, indent=2), encoding="utf-8")
    analysis_path = None
    if analysis is not None:
        staging_analysis.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
        analysis_path = out_dir / "analysis.json"
        summary = dict(summary)
        summary["analysis"] = str(analysis_path)
    staging_report.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _replace_file(staging_frags, out_dir / "mask-fragments.json")
    if analysis is not None:
        _replace_file(staging_analysis, analysis_path)
    _replace_file(staging_report, out_dir / "segment-report.json")
    marker = out_dir / "SEGMENT_FAILED"
    if marker.exists():
        marker.unlink()
    return analysis_path


def _failure_summary(exc: BaseException, image_path, out_dir) -> Dict[str, Any]:
    errors = getattr(exc, "errors", None)
    if not isinstance(errors, list) or not errors:
        errors = [{"code": "exception", "type": type(exc).__name__, "message": str(exc)}]
    return {
        "ok": False,
        "error": type(exc).__name__,
        "message": str(exc),
        "errors": errors,
        "analysis": None,
        "image": str(image_path),
        "out_dir": str(out_dir),
    }


def validate_final_owned(owned, fg, items):
    import numpy as np
    errors = []
    names = [item["id"] for item, _ in items] if items and isinstance(items[0], tuple) else [it["id"] for it in items]
    ordered_items = [it[0] if isinstance(it, tuple) else it for it in items]
    masks = []
    for item in ordered_items:
        name = item["id"]
        mask = np.asarray(owned.get(name, np.zeros(fg.shape, dtype=bool)), dtype=bool) & fg
        pixels = int(mask.sum())
        cc = _count_cc(mask, CC_COUNT_MIN_PX)
        max_cc = int(item.get("max_cc", DEFAULT_MAX_CC))
        if pixels <= 0:
            errors.append({"id": name, "code": "empty_piece", "pixels": pixels, "components": cc, "max_cc": max_cc})
        elif cc > max_cc:
            errors.append({"id": name, "code": "cc_exceeded", "pixels": pixels, "components": cc, "max_cc": max_cc})
        masks.append(mask)
    if masks:
        stacked = np.stack(masks).astype(np.uint8)
        overlap = int((stacked.sum(axis=0) > 1).sum())
        covered = stacked.sum(axis=0) > 0
        unassigned = int((fg & ~covered).sum())
    else:
        overlap = 0
        unassigned = int(fg.sum())
    if overlap:
        errors.append({"id": "*", "code": "overlap", "pixels": overlap})
    if unassigned:
        errors.append({"id": "*", "code": "unassigned", "pixels": unassigned})
    return errors, unassigned, overlap



def _union_owned_for_module(module: Dict[str, Any], owned: Dict[str, Any]):
    import numpy as np
    pieces = module.get("pieces") or []
    masks = []
    for piece in pieces:
        pid = piece.get("id") if isinstance(piece, dict) else None
        if pid in owned:
            masks.append(np.asarray(owned[pid], dtype=bool))
    mid = module.get("id")
    if not masks and mid in owned:
        masks.append(np.asarray(owned[mid], dtype=bool))
    if not masks:
        return None
    out = masks[0].copy()
    for extra in masks[1:]:
        out |= extra
    return out


def sync_connection_points(analysis: Dict[str, Any], owned: Dict[str, Any], w: int, h: int) -> None:
    """In-place connections[].point sync after auto geometry."""
    import numpy as np
    from src.pipeline.mask_geometry import derive_rigid_pivot
    modules = analysis.get("modules") or []
    by_id = {m["id"]: m for m in modules}
    mod_masks = {m["id"]: _union_owned_for_module(m, owned) for m in modules}
    width, height = float(w), float(h)
    for conn in analysis.get("connections") or []:
        if not isinstance(conn, dict):
            continue
        if conn.get("manual_point") is True:
            continue
        a = conn.get("a")
        b = conn.get("b")
        if conn.get("type") == "foot_anchor":
            chain = conn.get("chain") or []
            if not chain:
                continue
            foot_mask = mod_masks.get(b)
            lower_mask = mod_masks.get(chain[-1])
            if foot_mask is None or lower_mask is None or (not np.any(foot_mask)) or (not np.any(lower_mask)):
                continue
            px, py = derive_rigid_pivot(foot_mask, lower_mask)
            conn["point"] = [float(px) / width, float(py) / height]
            continue
        child = by_id.get(b)
        if child and child.get("parent") == a and child.get("pivot"):
            conn["point"] = list(child["pivot"])


OVERRIDE_KEYS = {"pos", "neg", "box", "roi_mode", "sub_prompts"}


def parse_override_json(value: Optional[str]) -> Dict[str, Any]:
    if value is None or str(value).strip() == "":
        return {}
    raw = str(value).strip()
    path = Path(raw)
    if raw[:1] in "{[":
        data = json.loads(raw)
    elif path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("override-json 不是合法 JSON 且文件不存在: %s" % value) from exc
    if not isinstance(data, dict):
        raise ValueError("override-json 必须是对象")
    return data


def apply_target_override(spec: Dict[str, Any], target_id: str, override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    spec = json.loads(json.dumps(spec))
    override = override or {}
    if not isinstance(override, dict):
        raise ValueError("override-json 必须是对象")
    unknown = sorted(k for k in override.keys() if k not in OVERRIDE_KEYS)
    if unknown:
        raise ValueError("override-json 含未知字段: %s" % ", ".join(unknown))
    if "sub_prompts" in override:
        sp = override.get("sub_prompts")
        if not isinstance(sp, list) or len(sp) == 0:
            raise ValueError("override-json sub_prompts 必须是非空列表")
    patch = {k: override[k] for k in OVERRIDE_KEYS if k in override}
    drop_old_sub = ("sub_prompts" not in patch) and any(k in patch for k in ("pos", "neg", "box"))

    def apply_patch(obj: Dict[str, Any]) -> None:
        obj.update(patch)
        if drop_old_sub:
            obj["sub_prompts"] = None

    found = False
    modules = spec.get("modules") or spec.get("prompts") or []
    if isinstance(modules, dict):
        modules = [{"id": k, **v} for k, v in modules.items()]
        spec["modules"] = modules
        spec.pop("prompts", None)
    for module in modules:
        if module.get("id") == target_id:
            apply_patch(module)
            found = True
        for piece in module.get("pieces") or []:
            if isinstance(piece, dict) and piece.get("id") == target_id:
                apply_patch(piece)
                found = True
    if not found:
        raise ValueError("target-id 不存在: %s" % target_id)
    return spec


def _isolate_pre_execute_failure(out_dir: Path, image_path, exc: BaseException) -> Dict[str, Any]:
    archive = None
    try:
        archive = begin_segment_run(Path(out_dir))
    except OutputDirUnwritable as err:
        return {
            "ok": False,
            "persisted": False,
            "error": "OutputDirUnwritable",
            "message": str(err),
            "analysis": None,
            "out_dir": str(out_dir),
        }
    except Exception as err:
        return persist_failure(Path(out_dir), _failure_summary(err, image_path, out_dir), archive)
    if not archive.get("ok"):
        return persist_failure(Path(out_dir), {
            "ok": False,
            "error": "ArchiveIncomplete",
            "message": "未能完整隔离旧成功产物，已停止切分",
            "errors": [{"code": "archive_incomplete", "items": archive.get("incomplete")}],
            "analysis": None,
            "image": str(image_path),
            "out_dir": str(out_dir),
        }, archive)
    return persist_failure(Path(out_dir), _failure_summary(exc, image_path, out_dir), archive)


def refine_one(image_path: Path, spec_path: Path, out_dir: Path, weight: Path, target_id: str,
               override: Optional[Dict[str, Any]] = None, *, no_cache: bool = False,
               prior_dir: Optional[Path] = None, auto_geometry: bool = True) -> Dict[str, Any]:
    _bind_cache_dirs()
    image_path = Path(image_path)
    spec_path = Path(spec_path)
    out_dir = Path(out_dir)
    try:
        require_ident(target_id, "target-id")
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        raw_modules = spec.get("modules") or spec.get("prompts") or []
        if isinstance(raw_modules, dict):
            raw_list = [{"id": k, **v} for k, v in raw_modules.items()]
        else:
            raw_list = list(raw_modules)
        targets = expand_segment_targets(raw_list)
        if target_id not in {t["id"] for t in targets}:
            raise ValueError("target-id 不存在: %s" % target_id)
        spec2 = apply_target_override(spec, target_id, override or {})
        out_dir.mkdir(parents=True, exist_ok=True)
        work_spec = out_dir / "refine-one-spec.json"
        work_spec.write_text(json.dumps(spec2, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        return _isolate_pre_execute_failure(out_dir, image_path, exc)
    return segment(
        image_path, work_spec, out_dir, weight, raw_masks=None,
        no_cache=no_cache, prior_dir=prior_dir, auto_geometry=auto_geometry,
    )


def segment(image_path: Path, spec_path: Path, out_dir: Path, weight: Path, raw_masks=None,
            *, no_cache: bool = False, prior_dir: Optional[Path] = None, auto_geometry: bool = True) -> Dict[str, Any]:
    _bind_cache_dirs()
    out_dir = Path(out_dir)
    archive = None
    try:
        archive = begin_segment_run(out_dir)
    except OutputDirUnwritable as exc:
        return {
            "ok": False,
            "persisted": False,
            "error": "OutputDirUnwritable",
            "message": str(exc),
            "analysis": None,
            "out_dir": str(out_dir),
        }
    except Exception as exc:
        return persist_failure(out_dir, _failure_summary(exc, image_path, out_dir), archive)
    if not archive.get("ok"):
        return persist_failure(out_dir, {
            "ok": False,
            "error": "ArchiveIncomplete",
            "message": "未能完整隔离旧成功产物，已停止切分",
            "errors": [{"code": "archive_incomplete", "items": archive.get("incomplete")}],
            "analysis": None,
            "image": str(image_path),
            "out_dir": str(out_dir),
        }, archive)
    try:
        return _segment_execute(image_path, spec_path, out_dir, weight, raw_masks, archive,
                               no_cache=no_cache, prior_dir=prior_dir, auto_geometry=auto_geometry)
    except (KeyboardInterrupt, SystemExit):
        persist_failure(out_dir, _failure_summary(KeyboardInterrupt("run interrupted"), image_path, out_dir), archive)
        raise
    except Exception as exc:
        return persist_failure(out_dir, _failure_summary(exc, image_path, out_dir), archive)


def _segment_execute(image_path: Path, spec_path: Path, out_dir: Path, weight: Path, raw_masks, archive,
                     *, no_cache: bool = False, prior_dir: Optional[Path] = None, auto_geometry: bool = True) -> Dict[str, Any]:
    from PIL import Image, ImageDraw
    import numpy as np
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    with Image.open(image_path) as im:
        src = np.asarray(im.convert("RGBA"), dtype=np.uint8)
    h, w = src.shape[:2]
    fg = src[..., 3] > 0
    rgb = src[..., :3].copy()
    items = spec.get("modules") or spec.get("prompts") or []
    if isinstance(items, dict):
        items = [{"id": k, **v} for k, v in items.items()]
    items = expand_segment_targets(items)
    cache = out_dir / "sam-raw"
    image_sha = hashlib.sha256(Path(image_path).read_bytes()).hexdigest()
    if raw_masks is None:
        raw = _run_sam(rgb, fg, items, cache, weight, no_cache=no_cache,
                       image_sha256=image_sha, prior_dir=Path(prior_dir) if prior_dir else None)
    else:
        raw = {}
        for item in items:
            name = item["id"]
            mask = np.asarray(raw_masks.get(name, np.zeros(fg.shape, dtype=bool)), dtype=bool)
            if mask.shape != fg.shape:
                raise ValueError("raw mask shape mismatch: %s" % name)
            raw[name] = mask & fg
    ordered = []
    for item in items:
        name = item["id"]
        mask = raw.get(name, np.zeros(fg.shape, dtype=bool))
        allow = allowed_region(item, h, w)
        mask = _fill_small_holes(mask, HOLE_MAX_PX) & allow & fg
        mask = _filter_small_cc(mask, PRE_WS_MIN_PX, int(item.get("max_cc", DEFAULT_MAX_CC)))
        mask &= allow & fg
        ordered.append((item, mask))
    seeds = [m for _, m in ordered]
    labels = _watershed_refine(rgb, fg, seeds)
    remaining = fg.copy()
    owned = {}
    for item, _seed in reversed(ordered):
        idx = [i for i, (it, _) in enumerate(ordered, start=1) if it["id"] == item["id"]][0]
        allow = allowed_region(item, h, w)
        mask = (labels == idx) & remaining & allow
        mask = _fill_small_holes(mask, HOLE_MAX_PX)
        mask &= remaining & allow & fg
        mask = _filter_small_cc(mask, CC_COUNT_MIN_PX, int(item.get("max_cc", DEFAULT_MAX_CC)))
        mask &= remaining & allow & fg
        owned[item["id"]] = mask
        remaining &= ~mask
    empty_before_nearest = [item["id"] for item, _ in ordered if int(owned[item["id"]].sum()) <= 0]
    leftover = remaining
    if leftover.any() and not empty_before_nearest:
        from scipy import ndimage as ndi
        canvas = np.full(fg.shape, -1, np.int32)
        allow = {}
        for i, (item, _) in enumerate(ordered):
            canvas[owned[item["id"]]] = i
            allow[i] = allowed_region(item, h, w)
        if (canvas >= 0).any():
            _, (iy, ix) = ndi.distance_transform_edt(canvas < 0, return_indices=True)
            nearest = canvas[iy, ix]
            for i, (item, _) in enumerate(ordered):
                take = leftover & (nearest == i) & allow[i]
                owned[item["id"]] |= take
            leftover = fg & ~np.logical_or.reduce([owned[k] for k in owned])
    for item, _ in ordered:
        owned[item["id"]] &= allowed_region(item, h, w) & fg
    errors, unassigned, overlap = validate_final_owned(owned, fg, ordered)
    if empty_before_nearest:
        for err in errors:
            if err.get("code") == "empty_piece" and err.get("id") in empty_before_nearest:
                err["note"] = "empty before nearest; nearest stuffing is not used to fill empty pieces"
        present = {err.get("id") for err in errors if err.get("code") == "empty_piece"}
        by_item = {it["id"]: it for it, _ in ordered}
        for name in empty_before_nearest:
            if name in present:
                continue
            item = by_item[name]
            errors.append({
                "id": name,
                "code": "empty_piece",
                "pixels": int(owned[name].sum()),
                "components": _count_cc(owned[name], CC_COUNT_MIN_PX),
                "max_cc": int(item.get("max_cc", DEFAULT_MAX_CC)),
                "note": "empty before nearest; nearest stuffing is not used to fill empty pieces",
            })
    mask_dir = out_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    fragments = {}
    report = []
    overlay = Image.fromarray(np.full((h, w, 3), GRAY[0], np.uint8))
    overlay.paste(Image.fromarray(src), (0, 0), Image.fromarray(src))
    overlay = overlay.convert("RGB")
    sheet_items = []
    for i, (item, _) in enumerate(ordered):
        name = item["id"]
        mask = owned[name] & fg
        cc = _count_cc(mask, CC_COUNT_MIN_PX)
        Image.fromarray(np.where(mask, 255, 0).astype(np.uint8)).save(safe_png_path(mask_dir, name))
        fragments[name] = _encode_mask(mask)
        color = COLORS[i % len(COLORS)]
        tint = np.zeros((h, w, 4), np.uint8)
        tint[mask] = (*color, 110)
        overlay = Image.alpha_composite(overlay.convert("RGBA"), Image.fromarray(tint)).convert("RGB")
        report.append({
            "id": name,
            "pixels": int(mask.sum()),
            "components": cc,
            "max_cc": int(item.get("max_cc", DEFAULT_MAX_CC)),
            "cc_min_px": CC_COUNT_MIN_PX,
        })
        sheet_items.append((name, mask))
    overlay.save(out_dir / "ownership-color-overlay.jpg", quality=95)
    build_module_sheet(src, sheet_items).save(out_dir / "module-crops-sheet.jpg", quality=95)
    if errors:
        summary = {
            "ok": False,
            "image": str(image_path),
            "out_dir": str(out_dir),
            "unassigned": unassigned,
            "overlap": overlap,
            "errors": errors,
            "modules": report,
            "analysis": None,
            "cc_count_min_px": CC_COUNT_MIN_PX,
        }
        (out_dir / "mask-fragments.failed.json").write_text(json.dumps(fragments, ensure_ascii=False, indent=2), encoding="utf-8")
        return persist_failure(out_dir, summary, archive)
    analysis = None
    if spec.get("version") == 2:
        analysis = json.loads(json.dumps(spec))
        for module in analysis.get("modules", []):
            pieces = module.get("pieces") or []
            if not pieces:
                pid = module.get("id")
                if pid in fragments:
                    module.setdefault("pieces", [{"id": pid + "_p0", "draw_order": int(module.get("draw_order", 0))}])
                    pieces = module["pieces"]
            if len(pieces) > 1:
                for piece in pieces:
                    frag = fragments.get(piece.get("id"))
                    if frag is None:
                        raise ValueError("缺少piece掩码: %s" % piece.get("id"))
                    piece["mask_png_base64"] = frag["mask_png_base64"]
                    piece["mask_sha256"] = frag["mask_sha256"]
            elif pieces:
                piece = pieces[0]
                frag = fragments.get(piece.get("id")) or fragments.get(module.get("id"))
                if frag is None:
                    raise ValueError("缺少piece掩码: %s" % piece.get("id"))
                piece["mask_png_base64"] = frag["mask_png_base64"]
                piece["mask_sha256"] = frag["mask_sha256"]
        if "image_size" not in analysis:
            analysis["image_size"] = [w, h]
        if "input_sha256" not in analysis:
            analysis["input_sha256"] = hashlib.sha256(Path(image_path).read_bytes()).hexdigest()
        if auto_geometry:
            from src.pipeline.mask_geometry import derive_contract_geometry
            masks_by_piece = {name: owned[name] for name in owned}
            analysis["modules"] = derive_contract_geometry(
                [w, h], analysis.get("modules") or [], masks_by_piece, preserve_manual=True)
            sync_connection_points(analysis, owned, w, h)
    summary = {
        "ok": True,
        "image": str(image_path),
        "out_dir": str(out_dir),
        "unassigned": unassigned,
        "overlap": overlap,
        "errors": [],
        "modules": report,
        "analysis": str(out_dir / "analysis.json") if analysis is not None else None,
        "cc_count_min_px": CC_COUNT_MIN_PX,
        "prev_success_analysis": archive.get("analysis") if archive else None,
    }
    publish_success_artifacts(out_dir, analysis, fragments, summary)
    return summary


def _add_segment_flags(parser):
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHT)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--prior-dir", type=Path, default=None)
    parser.add_argument("--no-auto-geometry", action="store_true")


def main(argv=None) -> int:
    _bind_cache_dirs()
    parser = argparse.ArgumentParser(description="SAM 2 精确掩码切分（独立脚本，不进入 src/ 服务）")
    sub = parser.add_subparsers(dest="command", required=True)
    g = sub.add_parser("emit-grid", help="生成带坐标刻度的灰底参考图")
    g.add_argument("--image", required=True, type=Path)
    g.add_argument("--out", required=True, type=Path)
    g.add_argument("--step", type=int, default=50)
    s = sub.add_parser("segment", help="按提示框切分并输出掩码/分析JSON")
    _add_segment_flags(s)
    r = sub.add_parser("refine-one", help="只重算单个目标件并刷新 analysis")
    _add_segment_flags(r)
    r.add_argument("--target-id", required=True)
    r.add_argument("--override-json", default=None)
    args = parser.parse_args(argv)
    if args.command == "emit-grid":
        try:
            path = emit_grid(args.image, args.out, args.step)
            print(json.dumps({"ok": True, "out": str(path)}, ensure_ascii=False))
            return 0
        except Exception as exc:
            print(json.dumps({"ok": False, "error": type(exc).__name__, "message": str(exc)}, ensure_ascii=False))
            return 1
    try:
        if args.command == "refine-one":
            override = parse_override_json(args.override_json)
            summary = refine_one(
                args.image, args.spec, args.out_dir, args.weights, args.target_id, override,
                no_cache=args.no_cache,
                prior_dir=args.prior_dir,
                auto_geometry=not args.no_auto_geometry,
            )
        else:
            summary = segment(
                args.image, args.spec, args.out_dir, args.weights,
                no_cache=args.no_cache,
                prior_dir=args.prior_dir,
                auto_geometry=not args.no_auto_geometry,
            )
    except Exception as exc:
        try:
            summary = persist_failure(Path(args.out_dir), _failure_summary(exc, args.image, args.out_dir))
        except Exception:
            summary = {"ok": False, "error": type(exc).__name__, "message": str(exc), "analysis": None, "persisted": False, "out_dir": str(args.out_dir)}
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
