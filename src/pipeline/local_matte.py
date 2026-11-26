"""Local portrait foreground matting for opaque-background inputs.

Works in both lightweight system Python (numpy + Pillow) and full .venv
(OpenCV GrabCut + optional local SAM 2.1 weights at .studio-work/models/sam2.1_b.pt).
Already-transparent RGBA portraits pass through losslessly without alteration.
"""
from __future__ import annotations

import hashlib
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from src.mivo.adapter import MivoCanvasError, validate_matted_portrait


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAM_WEIGHT = REPO_ROOT / ".studio-work/models/sam2.1_b.pt"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rgb_to_perceptual(rgb: np.ndarray) -> np.ndarray:
    """Lightweight perceptual color space (weighted RGB approximation) using numpy only."""
    arr = rgb.astype(np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = (b - y) * 0.564 + 128.0
    cr = (r - y) * 0.713 + 128.0
    return np.stack([y * 1.15, cb, cr], axis=-1)


def _border_pixels(rgb: np.ndarray, width: int = 4) -> np.ndarray:
    h, w = rgb.shape[:2]
    bw = max(1, min(width, h // 8, w // 8))
    parts = [
        rgb[:bw, :, :].reshape(-1, 3),
        rgb[h - bw :, :, :].reshape(-1, 3),
        rgb[bw : h - bw, :bw, :].reshape(-1, 3),
        rgb[bw : h - bw, w - bw :, :].reshape(-1, 3),
    ]
    return np.concatenate(parts, axis=0)


def _border_stats(rgb: np.ndarray) -> Tuple[np.ndarray, float]:
    border = _border_pixels(rgb, width=4)
    median = np.median(border, axis=0).astype(np.float32)
    dist = np.linalg.norm(border.astype(np.float32) - median[None, :], axis=1)
    p90 = float(np.percentile(dist, 90))
    return median, p90


def _border_flood_bg(rgb: np.ndarray, *, tol: float = 28.0, step_tol: float = 16.0) -> np.ndarray:
    """Return boolean background mask reachable from image outer border without punching interior holes."""
    h, w = rgb.shape[:2]
    perc = _rgb_to_perceptual(rgb)
    border_rgb = _border_pixels(rgb, width=3)
    border_perc = _rgb_to_perceptual(border_rgb.reshape(-1, 1, 3)).reshape(-1, 3)
    bg_ref = np.median(border_perc, axis=0)
    dist_ref = np.linalg.norm(perc - bg_ref[None, None, :], axis=2)

    bg = np.zeros((h, w), dtype=bool)
    q: deque[Tuple[int, int]] = deque()
    seed_tol = max(tol, float(np.percentile(np.linalg.norm(border_perc - bg_ref[None, :], axis=1), 85)) + 10.0)

    def try_seed(y: int, x: int) -> None:
        if not bg[y, x] and dist_ref[y, x] <= seed_tol:
            bg[y, x] = True
            q.append((y, x))

    for x in range(w):
        try_seed(0, x)
        try_seed(h - 1, x)
    for y in range(1, h - 1):
        try_seed(y, 0)
        try_seed(y, w - 1)

    if not q:
        for y, x in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)):
            bg[y, x] = True
            q.append((y, x))

    global_limit = max(seed_tol * 1.35, tol + 12.0)
    while q:
        y, x = q.popleft()
        cur = perc[y, x]
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < h and 0 <= nx < w and not bg[ny, nx]:
                if dist_ref[ny, nx] <= global_limit:
                    d_step = float(np.linalg.norm(perc[ny, nx] - cur))
                    if d_step <= step_tol or dist_ref[ny, nx] <= seed_tol * 0.78:
                        bg[ny, nx] = True
                        q.append((ny, nx))
    return bg


def _cleanup_fg_mask(fg: np.ndarray) -> np.ndarray:
    """Remove tiny detached speckles and fill tiny internal pinholes."""
    h, w = fg.shape
    min_island = max(24, int(round(h * w * 0.0004)))
    max_hole = max(24, int(round(h * w * 0.0005)))
    try:
        import cv2

        num, labels, stats, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), connectivity=8)
        cleaned = np.zeros_like(fg)
        for idx in range(1, num):
            if int(stats[idx, cv2.CC_STAT_AREA]) >= min_island:
                cleaned |= labels == idx
        inv = (~cleaned).astype(np.uint8)
        num_h, lab_h, st_h, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
        for idx in range(1, num_h):
            area = int(st_h[idx, cv2.CC_STAT_AREA])
            if area > max_hole:
                continue
            x, y, bw, bh = int(st_h[idx, 0]), int(st_h[idx, 1]), int(st_h[idx, 2]), int(st_h[idx, 3])
            if x > 0 and y > 0 and x + bw < w and y + bh < h:
                cleaned |= lab_h == idx
        return cleaned
    except Exception:
        im = Image.fromarray(fg.astype(np.uint8) * 255)
        closed = np.asarray(im.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.MinFilter(3))) > 127
        return closed


def _refine_grabcut_mask(rgb: np.ndarray, seed_fg: np.ndarray, *, band_px: int = 8) -> np.ndarray:
    try:
        import cv2
    except Exception:
        return seed_fg
    h, w = seed_fg.shape
    k = max(3, band_px * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    fg_u8 = seed_fg.astype(np.uint8)
    inner = cv2.erode(fg_u8, kernel).astype(bool)
    outer = cv2.dilate(fg_u8, kernel).astype(bool)
    if not inner.any() or (~outer).sum() < 16:
        return seed_fg
    gc = np.full((h, w), cv2.GC_BGD, dtype=np.uint8)
    gc[outer] = cv2.GC_PR_BGD
    gc[seed_fg] = cv2.GC_PR_FGD
    gc[inner] = cv2.GC_FGD
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(np.ascontiguousarray(rgb), gc, None, bgd, fgd, 3, cv2.GC_INIT_WITH_MASK)
    refined = (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)
    return refined if refined.any() else seed_fg


def _sam2_foreground(rgb: np.ndarray, fg_box: Optional[Tuple[int, int, int, int]], weight_path: Path) -> Optional[np.ndarray]:
    if not weight_path.is_file():
        return None
    try:
        from ultralytics import SAM
    except Exception:
        return None
    h, w = rgb.shape[:2]
    if fg_box is None:
        box = [int(w * 0.08), int(h * 0.03), int(w * 0.92), int(h * 0.98)]
    else:
        box = [int(v) for v in fg_box]
    corners = [[4, 4], [max(4, w - 5), 4], [4, max(4, h - 5)], [max(4, w - 5), max(4, h - 5)]]
    model = SAM(str(weight_path))
    results = model.predict(
        np.ascontiguousarray(rgb),
        bboxes=[box],
        points=corners,
        labels=[0, 0, 0, 0],
        verbose=False,
    )
    if not results or results[0].masks is None or results[0].masks.data is None:
        return None
    masks = results[0].masks.data.cpu().numpy()
    best = None
    best_score = -1.0
    total = float(h * w)
    for m in masks:
        cand = m > 0.5
        if cand.shape != (h, w):
            cand = np.asarray(Image.fromarray(cand.astype(np.uint8) * 255).resize((w, h), Image.Resampling.NEAREST)) > 127
        ratio = float(cand.sum()) / total
        if 0.04 <= ratio <= 0.93:
            border_touch = float(cand[0, :].mean() + cand[-1, :].mean() + cand[:, 0].mean() + cand[:, -1].mean()) / 4.0
            score = ratio - 1.5 * border_touch
            if score > best_score:
                best_score = score
                best = cand
    return best


def _defringe_edges(rgb: np.ndarray, fg: np.ndarray, bg_median: np.ndarray) -> np.ndarray:
    """Build clean antialiased RGBA and remove single-pixel backdrop color fringe."""
    h, w = fg.shape
    out = np.zeros((h, w, 4), dtype=np.uint8)
    out[..., :3] = rgb

    try:
        import cv2

        eroded = cv2.erode(fg.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        rim = fg & ~eroded
        alpha = np.where(fg, 255, 0).astype(np.uint8)
        if rim.any() and eroded.any():
            # Replace rim pixels closely tinted by the background with clean interior neighbor color.
            bg_dist = np.linalg.norm(rgb.astype(np.float32) - bg_median[None, None, :], axis=2)
            tinted = rim & (bg_dist < 42.0)
            if tinted.any():
                _, nearest = cv2.distanceTransformWithLabels(
                    (~eroded).astype(np.uint8), cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL
                )
                ys, xs = np.nonzero(eroded)
                # Fallback: simple 3x3 interior dilation color pull for tinted rim pixels.
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    dst = np.s_[max(0, dy) : h + min(0, dy), max(0, dx) : w + min(0, dx)]
                    src = np.s_[max(0, -dy) : h - max(0, dy), max(0, -dx) : w - max(0, dx)]
                    take = tinted[dst] & eroded[src]
                    out[dst][take, :3] = rgb[src][take]
                    tinted[dst] &= ~take
            alpha[rim] = 235
        out[..., 3] = alpha
    except Exception:
        out[..., 3] = np.where(fg, 255, 0).astype(np.uint8)

    out[out[..., 3] == 0] = 0
    return out


def parse_fg_box(raw: Optional[str], size: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
    if not raw:
        return None
    parts = [float(x.strip()) for x in raw.split(",")]
    if len(parts) != 4:
        raise ValueError("fg_box must contain 4 comma-separated numbers: x1,y1,x2,y2")
    w, h = size
    if max(parts) <= 1.0 and min(parts) >= 0.0:
        x1, y1, x2, y2 = parts[0] * w, parts[1] * h, parts[2] * w, parts[3] * h
    else:
        x1, y1, x2, y2 = parts
    x1_i = max(0, min(w - 2, int(round(x1))))
    y1_i = max(0, min(h - 2, int(round(y1))))
    x2_i = max(x1_i + 1, min(w - 1, int(round(x2))))
    y2_i = max(y1_i + 1, min(h - 1, int(round(y2))))
    return (x1_i, y1_i, x2_i, y2_i)


def remove_background_local(
    rgba: np.ndarray,
    *,
    fg_box: Optional[Tuple[int, int, int, int]] = None,
    sam_weight: Optional[Path] = None,
    prefer_sam: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Remove opaque portrait background locally; pass through already-transparent RGBA unchanged."""
    arr = np.asarray(rgba, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 4 or not arr.size:
        raise ValueError("Expected non-empty uint8 RGBA image")
    h, w = arr.shape[:2]
    trans_ratio = float((arr[..., 3] < 20).mean())
    if trans_ratio >= 0.03 and bool((arr[..., 3] >= 128).any()):
        return arr.copy(), {
            "method": "transparent_passthrough",
            "already_transparent": True,
            "size": [w, h],
            "transparent_ratio": round(trans_ratio, 4),
            "foreground_ratio": round(float((arr[..., 3] >= 128).mean()), 4),
        }

    rgb = arr[..., :3]
    bg_median, border_p90 = _border_stats(rgb)
    method = "border_flood"
    fg: Optional[np.ndarray] = None

    if border_p90 <= 32.0 and fg_box is None:
        bg = _border_flood_bg(rgb, tol=28.0, step_tol=16.0)
        seed_fg = ~bg
        seed_trans = float(bg.mean())
        if seed_trans >= 0.03 and seed_fg.any():
            fg = _refine_grabcut_mask(rgb, seed_fg, band_px=6)
            method = "border_flood_grabcut"

    if fg is None and prefer_sam:
        weight = Path(sam_weight) if sam_weight is not None else DEFAULT_SAM_WEIGHT
        sam_fg = _sam2_foreground(rgb, fg_box, weight)
        if sam_fg is not None:
            fg = _refine_grabcut_mask(rgb, sam_fg, band_px=6)
            method = "sam2_grabcut"

    if fg is None:
        try:
            import cv2

            rect = fg_box or (int(w * 0.08), int(h * 0.04), int(w * 0.92), int(h * 0.97))
            x1, y1, x2, y2 = rect
            gc = np.zeros((h, w), dtype=np.uint8)
            bgd = np.zeros((1, 65), np.float64)
            fgd = np.zeros((1, 65), np.float64)
            cv2.grabCut(
                np.ascontiguousarray(rgb),
                gc,
                (x1, y1, max(2, x2 - x1), max(2, y2 - y1)),
                bgd,
                fgd,
                4,
                cv2.GC_INIT_WITH_RECT,
            )
            cand = (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)
            if cand.any() and float((~cand).mean()) >= 0.03:
                fg = cand
                method = "grabcut_rect"
        except Exception:
            fg = None

    if fg is None:
        bg = _border_flood_bg(rgb, tol=max(32.0, border_p90 + 8.0), step_tol=20.0)
        fg = ~bg
        method = "border_flood_fallback"

    fg = _cleanup_fg_mask(fg)
    matted = _defringe_edges(rgb, fg, bg_median)
    out_trans = float((matted[..., 3] < 20).mean())
    out_fg = float((matted[..., 3] >= 128).mean())
    if out_trans < 0.03 or out_fg <= 0.0:
        raise MivoCanvasError(
            "本机去底未能分离有效角色前景与背景，可传入 --fg-box 框选角色范围或按需使用 Canvas 去底。",
            code="LOCAL_MATTING_FAILED",
        )
    return matted, {
        "method": method,
        "already_transparent": False,
        "border_p90_spread": round(border_p90, 2),
        "size": [w, h],
        "transparent_ratio": round(out_trans, 4),
        "foreground_ratio": round(out_fg, 4),
    }


def save_matte_check(original_rgba: np.ndarray, matted_rgba: np.ndarray, out_path: Path) -> Path:
    """Save a 3-panel visual check sheet (Original / Light Gray / Dark Gray)."""
    orig = Image.fromarray(np.asarray(original_rgba, dtype=np.uint8), mode="RGBA")
    mat = Image.fromarray(np.asarray(matted_rgba, dtype=np.uint8), mode="RGBA")
    w, h = orig.size
    scale = min(1.0, 960.0 / float(max(h, 1)))
    tw, th = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    orig_r = orig.resize((tw, th), Image.Resampling.LANCZOS)
    mat_r = mat.resize((tw, th), Image.Resampling.LANCZOS)

    panels = []
    for title, bg_col, src_im in (
        ("Original", (120, 120, 120, 255), orig_r),
        ("Matte · Light", (226, 229, 234, 255), mat_r),
        ("Matte · Dark", (32, 36, 44, 255), mat_r),
    ):
        canvas = Image.new("RGBA", (tw, th), bg_col)
        canvas.alpha_composite(src_im)
        panels.append((title, canvas.convert("RGB")))

    header = 28
    sheet = Image.new("RGB", (tw * 3, th + header), (26, 29, 36))
    draw = ImageDraw.Draw(sheet)
    for idx, (title, panel) in enumerate(panels):
        sheet.paste(panel, (idx * tw, header))
        draw.text((idx * tw + 10, 7), title, fill=(240, 244, 250))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=92)
    return out


def matte_portrait_file(
    input_path: Path,
    out_path: Path,
    *,
    check_out: Optional[Path] = None,
    fg_box: Optional[str] = None,
    sam_weight: Optional[Path] = None,
    prefer_sam: bool = True,
) -> Dict[str, Any]:
    src = Path(input_path).resolve()
    dst = Path(out_path).resolve()
    if not src.is_file():
        raise FileNotFoundError(src)
    with Image.open(src) as im:
        orig_rgba = np.asarray(im.convert("RGBA"), dtype=np.uint8)
        size = im.size
    box = parse_fg_box(fg_box, size)
    matted, report = remove_background_local(
        orig_rgba,
        fg_box=box,
        sam_weight=sam_weight,
        prefer_sam=prefer_sam,
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(matted, mode="RGBA").save(dst)
    validate_matted_portrait(dst, expected_size=size)
    report["input"] = str(src)
    report["output"] = str(dst)
    report["source_sha256"] = _sha256(src)
    report["output_sha256"] = _sha256(dst)
    if check_out is not None:
        saved = save_matte_check(orig_rgba, matted, Path(check_out).resolve())
        report["check_image"] = str(saved)
    return report
