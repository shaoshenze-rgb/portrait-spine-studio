"""Derive rigid pivots and soft centerlines from exclusive piece masks (numpy only)."""
from __future__ import annotations

import copy
import heapq
import math
from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

SQRT2 = math.sqrt(2.0)
_NEIGH8 = (
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, SQRT2),
    (-1, 1, SQRT2),
    (1, -1, SQRT2),
    (1, 1, SQRT2),
)
_SHORT_PATH_PX = 24.0
_MAX_GEODESIC_EDGE = 512


def _as_bool_mask(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError("mask 必须是 (H, W) 二维数组")
    return arr.astype(bool, copy=False)


def _require_nonempty(mask: np.ndarray, label: str = "mask") -> np.ndarray:
    m = _as_bool_mask(mask)
    if not np.any(m):
        raise ValueError("%s 为空" % label)
    return m


def _bbox(mask: np.ndarray, pad: int = 0) -> Optional[Tuple[int, int, int, int]]:
    rows = np.any(mask, axis=1)
    if not np.any(rows):
        return None
    cols = np.any(mask, axis=0)
    ys = np.flatnonzero(rows)
    xs = np.flatnonzero(cols)
    h, w = mask.shape
    r = int(max(0, pad))
    y0 = max(0, int(ys[0]) - r)
    x0 = max(0, int(xs[0]) - r)
    y1 = min(h, int(ys[-1]) + 1 + r)
    x1 = min(w, int(xs[-1]) + 1 + r)
    return y0, x0, y1, x1


def _dilate_bool(mask: np.ndarray, radius: int) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    r = int(radius)
    if r <= 0:
        return m.copy()
    box = _bbox(m, pad=0)
    if box is None:
        return np.zeros_like(m, dtype=bool)
    y0, x0, y1, x1 = box
    h, w = m.shape
    y0e = max(0, y0 - r)
    x0e = max(0, x0 - r)
    y1e = min(h, y1 + r)
    x1e = min(w, x1 + r)
    crop = m[y0e:y1e, x0e:x1e]
    k = 2 * r + 1
    padded = np.pad(crop, r, mode="constant", constant_values=False)
    windows = np.lib.stride_tricks.sliding_window_view(padded, (k, k))
    out = np.zeros_like(m, dtype=bool)
    out[y0e:y1e, x0e:x1e] = np.any(windows, axis=(-1, -2))
    return out


def _erode_bool(mask: np.ndarray, radius: int) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    r = int(radius)
    if r <= 0:
        return m.copy()
    box = _bbox(m, pad=0)
    if box is None:
        return np.zeros_like(m, dtype=bool)
    y0, x0, y1, x1 = box
    crop = m[y0:y1, x0:x1]
    k = 2 * r + 1
    padded = np.pad(crop, r, mode="constant", constant_values=False)
    windows = np.lib.stride_tricks.sliding_window_view(padded, (k, k))
    out = np.zeros_like(m, dtype=bool)
    out[y0:y1, x0:x1] = np.all(windows, axis=(-1, -2))
    return out


def _close_bool(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    return _erode_bool(_dilate_bool(mask, radius), radius)


def _flood_largest(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    h, w = m.shape
    fg = int(np.count_nonzero(m))
    if fg == 0:
        return np.zeros((h, w), dtype=bool)
    visited = np.zeros((h, w), dtype=bool)
    ys, xs = np.nonzero(m)
    best_cells: List[Tuple[int, int]] = []
    best_n = 0
    remaining = fg
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if visited[y0, x0]:
            continue
        stack = [(y0, x0)]
        visited[y0, x0] = True
        cells = [(y0, x0)]
        while stack:
            y, x = stack.pop()
            for dy, dx, _cost in _NEIGH8:
                ny = y + dy
                nx = x + dx
                if ny < 0 or ny >= h or nx < 0 or nx >= w:
                    continue
                if visited[ny, nx] or not m[ny, nx]:
                    continue
                visited[ny, nx] = True
                stack.append((ny, nx))
                cells.append((ny, nx))
        n = len(cells)
        remaining -= n
        if n > best_n:
            best_n = n
            best_cells = cells
            if best_n >= remaining + best_n:
                break
            if best_n * 2 >= fg:
                break
    out = np.zeros((h, w), dtype=bool)
    if best_cells:
        yy, xx = zip(*best_cells)
        out[np.asarray(yy), np.asarray(xx)] = True
    return out


def _pad_to_blocks(crop: np.ndarray, scale: int) -> Tuple[np.ndarray, int, int]:
    m = np.asarray(crop, dtype=bool)
    ch, cw = m.shape
    nh = (ch + scale - 1) // scale
    nw = (cw + scale - 1) // scale
    padded = np.zeros((nh * scale, nw * scale), dtype=bool)
    padded[:ch, :cw] = m
    return padded, nh, nw


def _block_maxpool(crop: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return np.asarray(crop, dtype=bool).copy()
    padded, nh, nw = _pad_to_blocks(crop, scale)
    return padded.reshape(nh, scale, nw, scale).any(axis=(1, 3))


def _compute_seam_edges(
    padded: np.ndarray, scale: int, nh: int, nw: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """4-tuple of bool edges on the original-pixel 8-neighborhood at block seams."""
    s = int(scale)
    edge_right = np.zeros((nh, max(nw - 1, 0)), dtype=bool)
    edge_down = np.zeros((max(nh - 1, 0), nw), dtype=bool)
    edge_dr = np.zeros((max(nh - 1, 0), max(nw - 1, 0)), dtype=bool)
    edge_dl = np.zeros((max(nh - 1, 0), max(nw - 1, 0)), dtype=bool)
    if nh == 0 or nw == 0:
        return edge_right, edge_down, edge_dr, edge_dl
    if nw >= 2:
        col_l = padded[:, s - 1 :: s][:, : nw - 1].reshape(nh, s, nw - 1)
        col_r = padded[:, s :: s].reshape(nh, s, nw - 1)
        same = col_l & col_r
        diag_a = col_l[:, 1:, :] & col_r[:, :-1, :]
        diag_b = col_l[:, :-1, :] & col_r[:, 1:, :]
        edge_right = same.any(axis=1) | diag_a.any(axis=1) | diag_b.any(axis=1)
    if nh >= 2:
        row_t = padded[s - 1 :: s, :][: nh - 1, :].reshape(nh - 1, nw, s)
        row_b = padded[s :: s, :].reshape(nh - 1, nw, s)
        same = row_t & row_b
        diag_a = row_t[:, :, 1:] & row_b[:, :, :-1]
        diag_b = row_t[:, :, :-1] & row_b[:, :, 1:]
        edge_down = same.any(axis=2) | diag_a.any(axis=2) | diag_b.any(axis=2)
    if nh >= 2 and nw >= 2:
        br = padded[s - 1 :: s, s - 1 :: s][: nh - 1, : nw - 1]
        tl = padded[s :: s, s :: s]
        edge_dr = br & tl
        bl = padded[s - 1 :: s, :: s][: nh - 1, 1:]
        tr = padded[s :: s, s - 1 :: s][:, : nw - 1]
        edge_dl = bl & tr
    return edge_right, edge_down, edge_dr, edge_dl


def _build_block_graph(crop: np.ndarray, scale: int):
    padded, nh, nw = _pad_to_blocks(crop, scale)
    blocks = padded.reshape(nh, scale, nw, scale)
    small = blocks.any(axis=(1, 3))
    counts = blocks.sum(axis=(1, 3)).astype(np.int32)
    edges = _compute_seam_edges(padded, scale, nh, nw)
    return small, counts, edges


def _seam_neighbors(
    y: int, x: int, nh: int, nw: int, edges: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
) -> List[Tuple[int, int, float]]:
    er, ed, edr, edl = edges
    out: List[Tuple[int, int, float]] = []
    if x + 1 < nw and er[y, x]:
        out.append((y, x + 1, 1.0))
    if x > 0 and er[y, x - 1]:
        out.append((y, x - 1, 1.0))
    if y + 1 < nh and ed[y, x]:
        out.append((y + 1, x, 1.0))
    if y > 0 and ed[y - 1, x]:
        out.append((y - 1, x, 1.0))
    if y + 1 < nh and x + 1 < nw and edr[y, x]:
        out.append((y + 1, x + 1, SQRT2))
    if y > 0 and x > 0 and edr[y - 1, x - 1]:
        out.append((y - 1, x - 1, SQRT2))
    if y + 1 < nh and x > 0 and edl[y, x - 1]:
        out.append((y + 1, x - 1, SQRT2))
    if y > 0 and x + 1 < nw and edl[y - 1, x]:
        out.append((y - 1, x + 1, SQRT2))
    return out


def _flood_largest_seams(
    small: np.ndarray,
    counts: np.ndarray,
    edges: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    nh, nw = small.shape
    visited = np.zeros((nh, nw), dtype=bool)
    ys, xs = np.nonzero(small)
    best_cells: List[Tuple[int, int]] = []
    best_n = 0
    remaining = int(counts.sum())
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if visited[y0, x0]:
            continue
        stack = [(y0, x0)]
        visited[y0, x0] = True
        cells = [(y0, x0)]
        total = int(counts[y0, x0])
        while stack:
            y, x = stack.pop()
            for ny, nx, _cost in _seam_neighbors(y, x, nh, nw, edges):
                if visited[ny, nx] or not small[ny, nx]:
                    continue
                visited[ny, nx] = True
                stack.append((ny, nx))
                cells.append((ny, nx))
                total += int(counts[ny, nx])
        remaining -= total
        if total > best_n:
            best_n = total
            best_cells = cells
            if best_n >= remaining + best_n:
                break
    out = np.zeros((nh, nw), dtype=bool)
    if best_cells:
        yy, xx = zip(*best_cells)
        out[np.asarray(yy), np.asarray(xx)] = True
    return out


def _upsample_blocks(sel_small: np.ndarray, crop: np.ndarray, scale: int) -> np.ndarray:
    ch, cw = crop.shape
    yy = np.minimum(np.arange(ch) // scale, sel_small.shape[0] - 1)
    xx = np.minimum(np.arange(cw) // scale, sel_small.shape[1] - 1)
    return sel_small[yy[:, None], xx[None, :]] & crop


def _largest_component(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    box = _bbox(m)
    if box is None:
        return np.zeros_like(m, dtype=bool)
    y0, x0, y1, x1 = box
    crop = m[y0:y1, x0:x1]
    long_edge = max(crop.shape)
    scale = 1 if long_edge <= _MAX_GEODESIC_EDGE else int(math.ceil(long_edge / float(_MAX_GEODESIC_EDGE)))
    if scale == 1:
        selected = _flood_largest(crop)
    else:
        small, counts, edges = _build_block_graph(crop, scale)
        if not np.any(small):
            selected = crop
        else:
            sel_small = _flood_largest_seams(small, counts, edges)
            selected = _upsample_blocks(sel_small, crop, scale)
            if not np.any(selected):
                selected = crop
    out = np.zeros_like(m, dtype=bool)
    out[y0:y1, x0:x1] = selected
    return out


def _centroid_xy(mask: np.ndarray) -> Tuple[float, float]:
    box = _bbox(mask)
    if box is None:
        raise ValueError("mask 为空")
    y0, x0, y1, x1 = box
    ys, xs = np.nonzero(mask[y0:y1, x0:x1])
    return float(xs.mean() + x0), float(ys.mean() + y0)


def _nearest_mask_xy(mask: np.ndarray, x: float, y: float) -> Tuple[float, float]:
    box = _bbox(mask)
    if box is None:
        raise ValueError("mask 为空")
    y0, x0, y1, x1 = box
    ys, xs = np.nonzero(mask[y0:y1, x0:x1])
    d2 = (xs.astype(np.float64) + x0 - float(x)) ** 2 + (ys.astype(np.float64) + y0 - float(y)) ** 2
    i = int(np.argmin(d2))
    return float(xs[i] + x0), float(ys[i] + y0)


def _in_mask(mask: np.ndarray, x: float, y: float) -> bool:
    h, w = mask.shape
    xi = int(round(x))
    yi = int(round(y))
    return 0 <= yi < h and 0 <= xi < w and bool(mask[yi, xi])


def _snap_xy(mask: np.ndarray, x: float, y: float) -> List[float]:
    if _in_mask(mask, x, y):
        return [float(x), float(y)]
    sx, sy = _nearest_mask_xy(mask, x, y)
    return [float(sx), float(sy)]


def _shift_pivot(parent_pivot: Optional[Sequence[float]], x0: float, y0: float):
    if parent_pivot is None or len(parent_pivot) < 2:
        return None
    return (float(parent_pivot[0]) - x0, float(parent_pivot[1]) - y0)


def _crop_for_geometry(
    mask: np.ndarray,
    parent_mask: Optional[np.ndarray],
    parent_pivot: Optional[Sequence[float]],
    pad: int = 4,
):
    original = _require_nonempty(mask)
    box = _bbox(original, pad=pad)
    if box is None:
        raise ValueError("mask 为空")
    y0, x0, y1, x1 = box
    child = original[y0:y1, x0:x1]
    parent_c = None
    pivot_c = _shift_pivot(parent_pivot, x0, y0)
    if parent_mask is not None:
        parent = np.asarray(parent_mask, dtype=bool)
        if parent.shape != original.shape:
            raise ValueError("parent_mask 尺寸必须与 mask 一致")
        parent_c = parent[y0:y1, x0:x1]
        if not np.any(parent_c):
            if pivot_c is None and np.any(parent):
                pcx, pcy = _centroid_xy(parent)
                pivot_c = (pcx - x0, pcy - y0)
            parent_c = None
    return original, child, parent_c, pivot_c, x0, y0


def derive_rigid_pivot(
    mask: np.ndarray,
    parent_mask: Optional[np.ndarray] = None,
    parent_pivot: Optional[Sequence[float]] = None,
) -> List[float]:
    _original, child, parent_c, pivot_c, x0, y0 = _crop_for_geometry(mask, parent_mask, parent_pivot, pad=4)
    local = _derive_rigid_pivot_local(child, parent_c, pivot_c)
    return [local[0] + x0, local[1] + y0]


def _derive_rigid_pivot_local(
    child: np.ndarray,
    parent: Optional[np.ndarray],
    parent_pivot: Optional[Sequence[float]],
) -> List[float]:
    if parent is not None:
        parent = np.asarray(parent, dtype=bool)
        if parent.shape != child.shape:
            raise ValueError("parent_mask 尺寸必须与 mask 一致")
    if parent is not None and np.any(parent):
        contact = _dilate_bool(child, 3) & parent
        if np.any(contact):
            cx, cy = _centroid_xy(contact)
            band = _dilate_bool(contact, 2) & child
            target = band if np.any(band) else child
            px, py = _nearest_mask_xy(target, cx, cy)
            return [float(px), float(py)]
        if parent_pivot is not None and len(parent_pivot) >= 2:
            px, py = _nearest_mask_xy(child, float(parent_pivot[0]), float(parent_pivot[1]))
        else:
            pcx, pcy = _centroid_xy(parent)
            px, py = _nearest_mask_xy(child, pcx, pcy)
        return [float(px), float(py)]
    if parent_pivot is not None and len(parent_pivot) >= 2:
        px, py = _nearest_mask_xy(child, float(parent_pivot[0]), float(parent_pivot[1]))
        return [float(px), float(py)]
    cx, cy = _centroid_xy(child)
    if _in_mask(child, cx, cy):
        return [float(cx), float(cy)]
    px, py = _nearest_mask_xy(child, cx, cy)
    return [float(px), float(py)]


def _root_point(
    mask: np.ndarray,
    parent_mask: Optional[np.ndarray],
    parent_pivot: Optional[Sequence[float]],
) -> Tuple[float, float]:
    parent = None if parent_mask is None else np.asarray(parent_mask, dtype=bool)
    if parent is not None and parent.shape != mask.shape:
        raise ValueError("parent_mask 尺寸必须与 mask 一致")
    if parent is not None and np.any(parent):
        contact = _dilate_bool(mask, 3) & parent
        if np.any(contact):
            child_edge = _dilate_bool(contact, 1) & mask
            use = child_edge if np.any(child_edge) else (contact & mask)
            if not np.any(use):
                use = mask
            cx, cy = _centroid_xy(use)
            return _nearest_mask_xy(mask, cx, cy)
        if parent_pivot is not None and len(parent_pivot) >= 2:
            return _nearest_mask_xy(mask, float(parent_pivot[0]), float(parent_pivot[1]))
        pcx, pcy = _centroid_xy(parent)
        return _nearest_mask_xy(mask, pcx, pcy)
    if parent_pivot is not None and len(parent_pivot) >= 2:
        return _nearest_mask_xy(mask, float(parent_pivot[0]), float(parent_pivot[1]))
    ys, xs = np.nonzero(mask)
    top = int(ys.min())
    on_top = ys == top
    cx = float(xs[on_top].mean())
    return _nearest_mask_xy(mask, cx, float(top))


def _prepare_work_mask(mask: np.ndarray) -> np.ndarray:
    closed = _close_bool(mask, 1)
    work = _largest_component(closed)
    if not np.any(work):
        work = _largest_component(mask)
    return work


def _downsample_mask(mask: np.ndarray):
    box = _bbox(mask)
    if box is None:
        return mask, 1, 0, 0, mask, None
    y0, x0, y1, x1 = box
    crop = mask[y0:y1, x0:x1]
    long_edge = max(crop.shape)
    scale = 1 if long_edge <= _MAX_GEODESIC_EDGE else int(math.ceil(long_edge / float(_MAX_GEODESIC_EDGE)))
    if scale == 1:
        return crop, scale, y0, x0, crop, None
    small, _counts, edges = _build_block_graph(crop, scale)
    if not np.any(small):
        return crop, 1, y0, x0, crop, None
    return small, scale, y0, x0, crop, edges


def _map_to_small(x: float, y: float, scale: int, y0: int, x0: int, small: np.ndarray) -> Tuple[float, float]:
    if scale <= 1:
        sx = float(x) - x0
        sy = float(y) - y0
    else:
        sx = math.floor((float(x) - x0) / float(scale))
        sy = math.floor((float(y) - y0) / float(scale))
    return _nearest_mask_xy(small, sx, sy)


def _map_to_full(
    x: float, y: float, scale: int, y0: int, x0: int, crop: np.ndarray
) -> Tuple[float, float]:
    if scale <= 1:
        return float(x) + x0, float(y) + y0
    ix = int(round(x))
    iy = int(round(y))
    fx = (float(ix) + 0.5) * scale + x0
    fy = (float(iy) + 0.5) * scale + y0
    # snap later on the original crop; keep this O(1) on the hot path
    ch, cw = crop.shape
    cx = fx - x0
    cy = fy - y0
    if 0 <= int(round(cy)) < ch and 0 <= int(round(cx)) < cw and crop[int(round(cy)), int(round(cx))]:
        return fx, fy
    by0 = max(0, iy * scale)
    bx0 = max(0, ix * scale)
    by1 = min(ch, by0 + scale)
    bx1 = min(cw, bx0 + scale)
    if by0 < by1 and bx0 < bx1:
        block = crop[by0:by1, bx0:bx1]
        ys, xs = np.nonzero(block)
        if ys.size:
            return float(xs[ys.size // 2] + bx0 + x0), float(ys[ys.size // 2] + by0 + y0)
    nx, ny = _nearest_mask_xy(crop, cx, cy)
    return float(nx + x0), float(ny + y0)


def _geodesic(mask: np.ndarray, root_xy: Tuple[float, float], edges=None):
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    n = int(ys.size)
    index_of = np.full((h, w), -1, dtype=np.int32)
    index_of[ys, xs] = np.arange(n, dtype=np.int32)
    rx, ry = _nearest_mask_xy(mask, root_xy[0], root_xy[1])
    start = int(index_of[int(round(ry)), int(round(rx))])
    if start < 0:
        start = 0
        rx, ry = float(xs[0]), float(ys[0])
    dist = np.full(n, np.inf, dtype=np.float64)
    prev = np.full(n, -1, dtype=np.int32)
    dist[start] = 0.0
    heap = [(0.0, start)]
    while heap:
        d, i = heapq.heappop(heap)
        if d != dist[i]:
            continue
        y = int(ys[i])
        x = int(xs[i])
        if edges is None:
            for dy, dx, cost in _NEIGH8:
                ny = y + dy
                nx = x + dx
                if ny < 0 or ny >= h or nx < 0 or nx >= w:
                    continue
                j = int(index_of[ny, nx])
                if j < 0:
                    continue
                nd = d + cost
                if nd < dist[j]:
                    dist[j] = nd
                    prev[j] = i
                    heapq.heappush(heap, (nd, j))
        else:
            for ny, nx, cost in _seam_neighbors(y, x, h, w, edges):
                j = int(index_of[ny, nx])
                if j < 0:
                    continue
                nd = d + cost
                if nd < dist[j]:
                    dist[j] = nd
                    prev[j] = i
                    heapq.heappush(heap, (nd, j))
    return ys, xs, dist, prev, start, rx, ry


def _backtrack(ys, xs, prev, start, tip):
    path = []
    i = int(tip)
    seen = set()
    while i >= 0 and i not in seen:
        seen.add(i)
        path.append((float(xs[i]), float(ys[i])))
        if i == start:
            break
        i = int(prev[i])
    path.reverse()
    if not path or path[0] != (float(xs[start]), float(ys[start])):
        path.insert(0, (float(xs[start]), float(ys[start])))
    return path


def _arclength(path: Sequence[Sequence[float]]) -> np.ndarray:
    pts = np.asarray(path, dtype=np.float64)
    if len(pts) <= 1:
        return np.array([0.0], dtype=np.float64)
    seg = np.sqrt(((pts[1:] - pts[:-1]) ** 2).sum(axis=1))
    return np.concatenate([[0.0], np.cumsum(seg)])


def _resample_path(path: Sequence[Sequence[float]], num_points: int) -> List[List[float]]:
    pts = np.asarray(path, dtype=np.float64)
    s = _arclength(pts)
    total = float(s[-1])
    if total <= 1e-6:
        p = [float(pts[0, 0]), float(pts[0, 1])]
        return [p[:] for _ in range(num_points)]
    targets = np.linspace(0.0, total, num_points)
    out: List[List[float]] = []
    for t in targets:
        k = int(np.searchsorted(s, t, side="right") - 1)
        k = max(0, min(k, len(pts) - 2))
        span = s[k + 1] - s[k]
        u = 0.0 if span <= 1e-9 else (t - s[k]) / span
        xy = pts[k] * (1.0 - u) + pts[k + 1] * u
        out.append([float(xy[0]), float(xy[1])])
    return out


def _center_along_normal(
    mask: np.ndarray, x: float, y: float, tx: float, ty: float, max_r: int = 48
) -> Tuple[float, float]:
    nlen = math.hypot(tx, ty) or 1.0
    nx, ny = -ty / nlen, tx / nlen
    h, w = mask.shape

    def inside(px: float, py: float) -> bool:
        xi = int(round(px))
        yi = int(round(py))
        return 0 <= yi < h and 0 <= xi < w and bool(mask[yi, xi])

    lo = 0
    hi = 0
    for s in range(1, max_r + 1):
        if inside(x + s * nx, y + s * ny):
            hi = s
        else:
            break
    for s in range(1, max_r + 1):
        if inside(x - s * nx, y - s * ny):
            lo = s
        else:
            break
    mid = 0.5 * (hi - lo)
    return x + mid * nx, y + mid * ny



def _segment_empty_count(mask: np.ndarray, p0: Sequence[float], p1: Sequence[float]) -> int:
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    dist = math.hypot(x1 - x0, y1 - y0)
    n = max(1, int(math.ceil(dist)))
    empty = 0
    for i in range(n + 1):
        t = i / float(n)
        x = x0 + t * (x1 - x0)
        y = y0 + t * (y1 - y0)
        if not _in_mask(mask, x, y):
            empty += 1
    return empty


def _segment_in_mask(mask: np.ndarray, p0: Sequence[float], p1: Sequence[float]) -> bool:
    return _segment_empty_count(mask, p0, p1) == 0


def _bfs_in_mask(mask: np.ndarray, p0: Sequence[float], p1: Sequence[float]) -> List[List[float]]:
    h, w = mask.shape
    x0, y0 = int(round(p0[0])), int(round(p0[1]))
    x1, y1 = int(round(p1[0])), int(round(p1[1]))
    if not (0 <= y0 < h and 0 <= x0 < w and mask[y0, x0]):
        x0, y0 = (int(v) for v in _nearest_mask_xy(mask, p0[0], p0[1]))
    if not (0 <= y1 < h and 0 <= x1 < w and mask[y1, x1]):
        x1, y1 = (int(v) for v in _nearest_mask_xy(mask, p1[0], p1[1]))
    if (x0, y0) == (x1, y1):
        return [[float(x1), float(y1)]]
    came = {(x0, y0): None}
    q = deque([(x0, y0)])
    found = False
    while q and len(came) < 20000:
        x, y = q.popleft()
        if (x, y) == (x1, y1):
            found = True
            break
        for dy, dx, _cost in _NEIGH8:
            nx = x + dx
            ny = y + dy
            if ny < 0 or ny >= h or nx < 0 or nx >= w:
                continue
            if not mask[ny, nx] or (nx, ny) in came:
                continue
            came[(nx, ny)] = (x, y)
            q.append((nx, ny))
    if (x1, y1) not in came:
        return []
    chain = []
    cur = (x1, y1)
    while cur is not None:
        chain.append([float(cur[0]), float(cur[1])])
        cur = came[cur]
    chain.reverse()
    return chain[1:]


def _densify_path(mask: np.ndarray, path: Sequence[Sequence[float]]) -> List[List[float]]:
    if not path:
        return []
    out = [_snap_xy(mask, path[0][0], path[0][1])]
    for raw in path[1:]:
        nxt = _snap_xy(mask, raw[0], raw[1])
        prev = out[-1]
        if nxt == prev:
            continue
        if _segment_in_mask(mask, prev, nxt):
            out.append(nxt)
            continue
        walked = _bfs_in_mask(mask, prev, nxt)
        if walked:
            for q in walked:
                if q != out[-1]:
                    out.append(q)
        else:
            out.append(nxt)
    return out


def _center_dense_path(mask: np.ndarray, path: Sequence[Sequence[float]]) -> List[List[float]]:
    pts = [p[:] for p in path]
    if len(pts) < 3:
        return pts
    for i in range(1, len(pts) - 1):
        tx = pts[i + 1][0] - pts[i - 1][0]
        ty = pts[i + 1][1] - pts[i - 1][1]
        cx, cy = _center_along_normal(mask, pts[i][0], pts[i][1], tx, ty)
        pts[i] = _snap_xy(mask, cx, cy)
    return pts


def _strict_indices(idxs: Sequence[int], n: int) -> List[int]:
    out = [int(i) for i in idxs]
    npts = len(out)
    if npts == 0 or n <= 1:
        return [0] * max(1, npts)
    out[0] = 0
    out[-1] = n - 1
    for i in range(1, npts):
        if out[i] <= out[i - 1]:
            out[i] = out[i - 1] + 1
    for i in range(npts - 2, -1, -1):
        if out[i] >= out[i + 1]:
            out[i] = out[i + 1] - 1
    out[0] = 0
    out[-1] = n - 1
    for i in range(1, npts):
        if out[i] <= out[i - 1]:
            out[i] = min(n - 1 - (npts - 1 - i), out[i - 1] + 1)
    out[-1] = n - 1
    return out


def _perp_abs(a: Sequence[float], b: Sequence[float], p: Sequence[float]) -> float:
    vx, vy = b[0] - a[0], b[1] - a[1]
    L = math.hypot(vx, vy)
    if L <= 1e-9:
        return math.hypot(p[0] - a[0], p[1] - a[1])
    return abs((p[0] - a[0]) * (-vy) + (p[1] - a[1]) * vx) / L


def _max_perp_index(dense: Sequence[Sequence[float]], i0: int, i1: int) -> Optional[int]:
    if i1 - i0 <= 1:
        return None
    a, b = dense[i0], dense[i1]
    best_i = None
    best_d = -1.0
    for i in range(i0 + 1, i1):
        d = _perp_abs(a, b, dense[i])
        if d > best_d:
            best_d = d
            best_i = i
    return best_i


def _chord_empty_total(mask: np.ndarray, dense: Sequence[Sequence[float]], idxs: Sequence[int]) -> int:
    total = 0
    for k in range(len(idxs) - 1):
        total += _segment_empty_count(mask, dense[idxs[k]], dense[idxs[k + 1]])
    return total


def _select_chord_points(
    mask: np.ndarray, dense: Sequence[Sequence[float]], num_points: int
) -> List[List[float]]:
    n = len(dense)
    npts = int(num_points)
    if n <= npts:
        pts = [p[:] for p in dense]
        while len(pts) < npts:
            pts.append(pts[-1][:])
        return pts[:npts]
    arc = _arclength(dense)
    total = float(arc[-1])
    targets = np.linspace(0.0, total, npts)
    idxs = []
    for t in targets:
        k = int(np.searchsorted(arc, t, side="left"))
        k = max(0, min(n - 1, k))
        idxs.append(k)
    idxs = _strict_indices(idxs, n)
    for _ in range(2 * npts):
        worst_k = -1
        worst_empty = 0
        for k in range(len(idxs) - 1):
            empty = _segment_empty_count(mask, dense[idxs[k]], dense[idxs[k + 1]])
            if empty > worst_empty:
                worst_empty = empty
                worst_k = k
        if worst_k < 0 or worst_empty == 0:
            break
        corner = _max_perp_index(dense, idxs[worst_k], idxs[worst_k + 1])
        if corner is None or corner in idxs:
            break
        best_j = None
        best_cost = None
        for j in range(1, npts - 1):
            a = idxs[j - 1]
            b = idxs[j + 1]
            if b - a <= 1:
                continue
            if not _segment_in_mask(mask, dense[a], dense[b]):
                continue
            cost = _perp_abs(dense[a], dense[b], dense[idxs[j]])
            cost += 0.01 * abs((arc[idxs[j]] - arc[a]) - (arc[b] - arc[idxs[j]]))
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_j = j
        if best_j is None:
            break
        old_empty = _chord_empty_total(mask, dense, idxs)
        removed = idxs[best_j]
        trial = [i for i in idxs if i != removed]
        if corner in trial:
            break
        trial.append(corner)
        trial.sort()
        if trial[0] != 0 or trial[-1] != n - 1 or len(trial) != npts:
            break
        new_empty = _chord_empty_total(mask, dense, trial)
        if new_empty >= old_empty:
            break
        idxs = trial
    return [dense[i][:] for i in idxs]


def _center_samples_safe(mask: np.ndarray, sampled: Sequence[Sequence[float]]) -> List[List[float]]:
    out = [p[:] for p in sampled]
    if len(out) < 3:
        return [_snap_xy(mask, p[0], p[1]) for p in out]
    out[0] = _snap_xy(mask, out[0][0], out[0][1])
    out[-1] = _snap_xy(mask, out[-1][0], out[-1][1])
    for i in range(1, len(out) - 1):
        tx = out[i + 1][0] - out[i - 1][0]
        ty = out[i + 1][1] - out[i - 1][1]
        cx, cy = _center_along_normal(mask, out[i][0], out[i][1], tx, ty)
        cand = _snap_xy(mask, cx, cy)
        if _segment_in_mask(mask, out[i - 1], cand) and _segment_in_mask(mask, cand, out[i + 1]):
            out[i] = cand
    return out


def _pca_line_points(mask: np.ndarray, root: Sequence[float], num_points: int) -> List[List[float]]:
    ys, xs = np.nonzero(mask)
    coords = np.column_stack((xs.astype(np.float64), ys.astype(np.float64)))
    mean = coords.mean(axis=0)
    centered = coords - mean
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0]
    t = centered @ axis
    tr = (np.asarray(root, dtype=np.float64) - mean) @ axis
    tmin, tmax = float(t.min()), float(t.max())
    if (tmax - tr) >= (tr - tmin):
        t0, t1 = tr, tmax
    else:
        t0, t1 = tr, tmin
    if abs(t1 - t0) < 1e-6:
        t0, t1 = tmin, tmax
    n = max(3, int(num_points))
    samples = np.linspace(t0, t1, n)
    points = []
    for ti in samples:
        cand = mean + ti * axis
        points.append(_snap_xy(mask, float(cand[0]), float(cand[1])))
    return points


def derive_soft_centerline(
    mask: np.ndarray,
    parent_mask: Optional[np.ndarray] = None,
    parent_pivot: Optional[Sequence[float]] = None,
    num_points: int = 5,
) -> List[List[float]]:
    if int(num_points) < 3:
        raise ValueError("num_points 必须 >= 3")
    original, child, parent_c, pivot_c, x0, y0 = _crop_for_geometry(mask, parent_mask, parent_pivot, pad=4)
    work = _prepare_work_mask(child)
    if not np.any(work):
        raise ValueError("mask 为空")
    root_x, root_y = _root_point(work, parent_c, pivot_c)
    root_x, root_y = _nearest_mask_xy(child, root_x, root_y)
    original = child

    small, scale, sy0, sx0, crop, edges = _downsample_mask(work)
    sroot = _map_to_small(root_x, root_y, scale, sy0, sx0, small)
    ys, xs, dist, prev, start, _srx, _sry = _geodesic(small, sroot, edges=edges)
    finite = np.isfinite(dist)
    if not np.any(finite):
        points = _pca_line_points(original, (root_x, root_y), int(num_points))
        return [[p[0] + x0, p[1] + y0] for p in points]
    tip = int(np.nanargmax(np.where(finite, dist, -np.inf)))
    path_small = _backtrack(ys, xs, prev, start, tip)
    path_full: List[List[float]] = []
    for px, py in path_small:
        fx, fy = _map_to_full(px, py, scale, sy0, sx0, crop)
        path_full.append(_snap_xy(original, fx, fy))
    if math.hypot(path_full[0][0] - root_x, path_full[0][1] - root_y) > 1.5:
        path_full.insert(0, [float(root_x), float(root_y)])
    else:
        path_full[0] = [float(root_x), float(root_y)]

    arc = _arclength(path_full)
    total = float(arc[-1])
    if len(path_full) < 2 or total < _SHORT_PATH_PX:
        points = _pca_line_points(original, (root_x, root_y), int(num_points))
        points[0] = _snap_xy(original, root_x, root_y)
        out = [_snap_xy(original, p[0], p[1]) for p in points]
        return [[p[0] + x0, p[1] + y0] for p in out]

    dense = _densify_path(original, path_full)
    sampled = _select_chord_points(original, dense, int(num_points))
    sampled[0] = _snap_xy(original, root_x, root_y)
    sampled = _center_samples_safe(original, sampled)
    for p in sampled:
        if not _in_mask(original, p[0], p[1]):
            q = _snap_xy(original, p[0], p[1])
            p[0], p[1] = q[0], q[1]
    return [[p[0] + x0, p[1] + y0] for p in sampled]


def _union_module_mask(
    module: Dict[str, Any],
    masks_by_piece: Dict[str, np.ndarray],
    shape: Optional[Tuple[int, int]] = None,
) -> Optional[np.ndarray]:
    pieces = module.get("pieces") or []
    acc = None
    for piece in pieces:
        pid = piece.get("id") if isinstance(piece, dict) else None
        if pid is None or pid not in masks_by_piece:
            continue
        m = _as_bool_mask(masks_by_piece[pid])
        if shape is None:
            shape = m.shape
        if m.shape != shape:
            raise ValueError("piece 掩码尺寸不一致: %s" % pid)
        acc = m if acc is None else (acc | m)
    if acc is None and module.get("id") in masks_by_piece:
        acc = _as_bool_mask(masks_by_piece[module["id"]])
    return acc


def _keep_manual_pivot(module: Dict[str, Any]) -> bool:
    if module.get("manual_pivot") is True:
        return True
    override = module.get("geometry_override")
    if override is True:
        return True
    if isinstance(override, dict) and override.get("pivot"):
        return True
    return False


def _keep_manual_centerline(module: Dict[str, Any]) -> bool:
    if module.get("manual_centerline") is True:
        return True
    override = module.get("geometry_override")
    if override is True:
        return True
    if isinstance(override, dict) and override.get("centerline"):
        return True
    return False


def _topo_order(modules: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ids = []
    for m in modules:
        mid = m.get("id")
        if mid is None or mid == "":
            raise ValueError("module 缺少 id")
        ids.append(mid)
    if len(ids) != len(set(ids)):
        raise ValueError("重复的 module id")
    by_id = {m["id"]: m for m in modules}
    for m in modules:
        parent = m.get("parent")
        if parent is None or parent == "":
            continue
        if parent not in by_id:
            raise ValueError("缺失父级: %s" % parent)
        if parent == m["id"]:
            raise ValueError("存在父子环")
    remaining = dict(by_id)
    ordered: List[Dict[str, Any]] = []
    while remaining:
        ready = [
            m
            for m in remaining.values()
            if not m.get("parent") or m.get("parent") not in remaining
        ]
        if not ready:
            raise ValueError("存在父子环")
        for m in ready:
            ordered.append(m)
            del remaining[m["id"]]
    return ordered


def _require_piece_masks(
    modules: Sequence[Dict[str, Any]],
    masks_by_piece: Dict[str, np.ndarray],
) -> None:
    for module in modules:
        pieces = module.get("pieces") or []
        if not pieces:
            raise ValueError("module pieces 为空: %s" % module.get("id"))
        for piece in pieces:
            pid = piece.get("id") if isinstance(piece, dict) else None
            if pid is None or pid not in masks_by_piece:
                raise ValueError("缺失 piece 掩码: %s" % pid)
            mask = _as_bool_mask(masks_by_piece[pid])
            if not np.any(mask):
                raise ValueError("piece 掩码为空: %s" % pid)


def derive_contract_geometry(
    image_size: Sequence[int],
    modules: Sequence[Dict[str, Any]],
    masks_by_piece: Dict[str, np.ndarray],
    *,
    preserve_manual: bool = True,
) -> List[Dict[str, Any]]:
    if len(image_size) < 2:
        raise ValueError("image_size 必须是 (width, height)")
    width, height = float(image_size[0]), float(image_size[1])
    if width <= 0 or height <= 0:
        raise ValueError("image_size 非法")
    updated = copy.deepcopy(list(modules))
    _topo_order(updated)
    _require_piece_masks(updated, masks_by_piece)
    by_id = {m["id"]: m for m in updated}
    mask_cache: Dict[str, np.ndarray] = {}
    for module in _topo_order(updated):
        mod_mask = _union_module_mask(module, masks_by_piece)
        if mod_mask is not None:
            mask_cache[module["id"]] = mod_mask
        parent_id = module.get("parent")
        parent_mask = mask_cache.get(parent_id) if parent_id else None
        parent_mod = by_id.get(parent_id) if parent_id else None
        parent_pivot_px = None
        if parent_mod and parent_mod.get("pivot") and len(parent_mod["pivot"]) >= 2:
            parent_pivot_px = [
                float(parent_mod["pivot"][0]) * width,
                float(parent_mod["pivot"][1]) * height,
            ]
        if mod_mask is None or not np.any(mod_mask):
            continue
        if not (preserve_manual and _keep_manual_pivot(module)):
            px, py = derive_rigid_pivot(mod_mask, parent_mask, parent_pivot_px)
            module["pivot"] = [px / width, py / height]
        if module.get("motion_type") == "soft" and not (
            preserve_manual and _keep_manual_centerline(module)
        ):
            existing = module.get("centerline") or []
            npts = len(existing) if len(existing) >= 3 else 5
            pts = derive_soft_centerline(mod_mask, parent_mask, parent_pivot_px, num_points=npts)
            module["centerline"] = [[p[0] / width, p[1] / height] for p in pts]
            if not (preserve_manual and _keep_manual_pivot(module)):
                # One auto-generated soft root, not two independent estimators.
                # Explicit/manual pivots are never moved to satisfy the contract.
                module["pivot"] = list(module["centerline"][0])
    return updated
