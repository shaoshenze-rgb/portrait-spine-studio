"""Generalized deterministic part splitter and amodal hidden-fill inpainter (Erika v2 engine).

Preserves:
- drawOrder ownership with watershed boundary refinement (`band`) and island cleanup;
- local `guides` (target zone polygons, source zone polygons, optional colour predicate,
  method `"smooth"` multigrid Laplace membrane or `"patch"` owner-excluded multi-scale PatchMatch);
- per-part SHA-256 hash caching in `<out_dir>/work/`;
- outputs `parts.json`, `parts/<name>.png`, `rest_composite.png`, `rest_diff.json`, `review/part_map.png`.
"""
from __future__ import annotations

import hashlib
import heapq
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw

ALGO_VERSION = "pss-split-v1"
OPAQUE_DEFAULT = 250
EXCL_DEFAULT = 4
FEATHER_DEFAULT = 4
LAM_DEFAULT = 220.0

N8 = [
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, 1.41421356),
    (-1, 1, 1.41421356),
    (1, -1, 1.41421356),
    (1, 1, 1.41421356),
]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _shift(arr: np.ndarray, dy: int, dx: int, fill: Any) -> np.ndarray:
    out = np.full_like(arr, fill)
    h, w = arr.shape[:2]
    sy0, sy1 = max(0, -dy), min(h, h - dy)
    sx0, sx1 = max(0, -dx), min(w, w - dx)
    dy0, dy1 = max(0, dy), min(h, h + dy)
    dx0, dx1 = max(0, dx), min(w, w + dx)
    if sy1 > sy0 and sx1 > sx0:
        out[dy0:dy1, dx0:dx1] = arr[sy0:sy1, sx0:sx1]
    return out


def _poly_mask(
    width: int,
    height: int,
    polys: Optional[Sequence[Sequence[Sequence[float]]]] = None,
    lines: Optional[Sequence[Tuple[Sequence[Sequence[float]], int]]] = None,
) -> np.ndarray:
    im = Image.new("L", (width, height), 0)
    dr = ImageDraw.Draw(im)
    if polys:
        for poly in polys:
            if len(poly) >= 3:
                dr.polygon([tuple(map(float, pt)) for pt in poly], fill=255)
    if lines:
        for pts, lw in lines:
            if len(pts) >= 2:
                dr.line([tuple(map(float, pt)) for pt in pts], fill=255, width=int(lw))
                r = float(lw) * 0.5
                for px, py in pts:
                    dr.ellipse((px - r, py - r, px + r, py + r), fill=255)
    return np.asarray(im, dtype=np.uint8) > 0


def erode_bool(mask: np.ndarray, r: int) -> np.ndarray:
    """Binary erosion via dilation of the complement. r<=0 is a copy."""
    m = np.asarray(mask, dtype=bool)
    if r <= 0:
        return m.copy()
    return ~_dilate(~m, r)


def _dilate(mask: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return mask.copy()
    a = mask.copy()
    s = 1
    while s <= r:
        step = min(s, r - s + 1) if s * 2 > r else s
        b = a.copy()
        b[:, step:] |= a[:, :-step]
        b[:, :-step] |= a[:, step:]
        c = b.copy()
        c[step:, :] |= b[:-step, :]
        c[:-step, :] |= b[step:, :]
        a = c
        s += step
    return a


def _bbox(mask: np.ndarray, pad: int = 0) -> Tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    return (
        max(0, int(ys.min()) - pad),
        min(mask.shape[0], int(ys.max()) + 1 + pad),
        max(0, int(xs.min()) - pad),
        min(mask.shape[1], int(xs.max()) + 1 + pad),
    )


def _dist_small(seed: np.ndarray, r: int) -> np.ndarray:
    INF = np.float32(1e9)
    d = np.where(seed, np.float32(0.0), INF).astype(np.float32)
    for _ in range(int(math.ceil(r)) + 1):
        best = d
        for dy, dx, c in N8:
            best = np.minimum(best, _shift(d, dy, dx, INF) + np.float32(c))
        d = best
    return d


def _chamfer(seed: np.ndarray, allowed: np.ndarray, maxd: float) -> np.ndarray:
    INF = np.float32(1e9)
    ok = seed | allowed
    d = np.where(seed, np.float32(0.0), INF).astype(np.float32)
    for _ in range(int(math.ceil(maxd)) + 2):
        best = d
        for dy, dx, c in N8:
            best = np.minimum(best, _shift(d, dy, dx, INF) + np.float32(c))
        best = np.where(ok, best, INF)
        best = np.minimum(best, d)
        if np.array_equal(best, d):
            break
        d = best
    return d


def _gradient(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    f = rgb.astype(np.float32)
    k = np.pad(f, ((1, 1), (1, 1), (0, 0)), mode="edge")
    sm = sum(k[dy : dy + h, dx : dx + w] for dy in range(3) for dx in range(3)) / 9.0
    p = np.pad(sm, ((1, 1), (1, 1), (0, 0)), mode="edge")
    gx = p[1:-1, 2:] - p[1:-1, :-2]
    gy = p[2:, 1:-1] - p[:-2, 1:-1]
    return np.sqrt((gx ** 2 + gy ** 2).sum(axis=-1))


def _components(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    idx = np.arange(h * w, dtype=np.int64).reshape(h, w)
    parent = idx.ravel().copy()
    ea = np.concatenate([idx[:, :-1][mask[:, :-1] & mask[:, 1:]], idx[:-1, :][mask[:-1, :] & mask[1:, :]]])
    eb = np.concatenate([idx[:, 1:][mask[:, :-1] & mask[:, 1:]], idx[1:, :][mask[:-1, :] & mask[1:, :]]])
    while len(ea) > 0:
        pa, pb = parent[ea], parent[eb]
        if np.array_equal(pa, pb):
            break
        lo = np.minimum(pa, pb)
        np.minimum.at(parent, pa, lo)
        np.minimum.at(parent, pb, lo)
        while True:
            nxt = parent[parent]
            if np.array_equal(nxt, parent):
                break
            parent = nxt
    lab = np.where(mask.ravel(), parent + 1, 0).reshape(h, w)
    _u, inv = np.unique(lab, return_inverse=True)
    return inv.reshape(h, w)


def _clean_islands(lab: np.ndarray, num_parts: int, min_px: int = 12,
                   locked: Optional[np.ndarray] = None,
                   allowed_masks: Optional[List[np.ndarray]] = None) -> np.ndarray:
    out = lab.copy()
    for i in range(num_parts):
        m = out == i
        if not m.any():
            continue
        comp = _components(m)
        sizes = np.bincount(comp.ravel())
        for c in np.nonzero((sizes < min_px) & (np.arange(len(sizes)) > 0))[0]:
            cm = comp == c
            if locked is not None and (locked[cm] >= 0).any():
                continue
            ring = _dilate(cm, 1) & ~cm & (out >= 0)
            if ring.any():
                vals = out[ring]
                counts = np.bincount(vals, minlength=num_parts)
                candidates = [idx for idx in np.argsort(-counts) if counts[idx] > 0
                              and (allowed_masks is None or allowed_masks[idx][cm].all())]
                if candidates:
                    out[cm] = int(candidates[0])
    return out


def _colour_pred(rgb: np.ndarray, pred: Optional[str]) -> np.ndarray:
    if pred is None:
        return np.ones(rgb.shape[:2], dtype=bool)
    r = rgb[..., 0].astype(np.int32)
    g = rgb[..., 1].astype(np.int32)
    b = rgb[..., 2].astype(np.int32)
    if pred == "skin":
        return (r > 140) & (r - b > 35) & (g > 80) & (r >= g)
    if pred == "notskin":
        return ~((r > 140) & (r - b > 35) & (g > 80) & (r >= g))
    if pred == "teal":
        return (g > r * 1.05) & (b > r * 0.9) & (g > 45)
    if pred == "dark":
        return (0.3 * r + 0.59 * g + 0.11 * b) < 95
    if pred == "bright":
        return (0.3 * r + 0.59 * g + 0.11 * b) >= 160
    raise ValueError(f"Unknown colour predicate: {pred!r}")


def _diffuse_init(img: np.ndarray, known: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    img = img.copy()
    have = known.copy()
    todo = target & ~have
    for _ in range(400):
        if not todo.any():
            break
        acc = np.zeros_like(img)
        cnt = np.zeros(known.shape, dtype=np.float32)
        for dy, dx, _c in N8:
            hv = _shift(have, dy, dx, False)
            acc += _shift(img, dy, dx, 0.0) * hv[..., None]
            cnt += hv
        front = todo & (cnt > 0)
        if not front.any():
            break
        img[front] = acc[front] / cnt[front][:, None]
        have |= front
        todo &= ~front
    return img, todo


def _down_known(img: np.ndarray, known: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = known.shape
    h2, w2 = (h + 1) // 2, (w + 1) // 2
    pad = ((0, h2 * 2 - h), (0, w2 * 2 - w))
    k = np.pad(known, pad).reshape(h2, 2, w2, 2)
    t = np.pad(target, pad).reshape(h2, 2, w2, 2)
    im = np.pad(img, pad + ((0, 0),)).reshape(h2, 2, w2, 2, 3)
    kc = k.sum(axis=(1, 3))
    col = (im * k[..., None]).sum(axis=(1, 3)) / np.maximum(kc, 1)[..., None]
    k2 = kc > 0
    t2 = (t.sum(axis=(1, 3)) > 0) & ~k2
    return col.astype(np.float32), k2, t2


def smooth_membrane(img: np.ndarray, known: np.ndarray, target: np.ndarray, n_fine: int = 100) -> np.ndarray:
    """Multigrid coarse-to-fine Jacobi Laplace membrane completion."""
    levels = [(img.astype(np.float32), known.copy(), target & ~known)]
    while min(levels[-1][1].shape) >= 12 and len(levels) < 7:
        nxt = _down_known(*levels[-1])
        if not nxt[1].any():
            break
        levels.append(nxt)
    cur = None
    for li in range(len(levels) - 1, -1, -1):
        im, k, t = levels[li]
        t = t & ~k
        if cur is None:
            base, left = _diffuse_init(im, k, t)
            if left.any() and k.any():
                base[left] = im[k].mean(axis=0)
        else:
            up = np.repeat(np.repeat(cur, 2, axis=0), 2, axis=1)[: im.shape[0], : im.shape[1]]
            base = im.copy()
            base[t] = up[t]
        reg = (k | t).astype(np.float32)
        cnt = sum(_shift(reg, dy, dx, 0.0) for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)))
        upd = t & (cnt > 0)
        n_it = n_fine if li == 0 else 50
        for _ in range(n_it):
            v = base * reg[..., None]
            acc = _shift(v, 0, 1, 0.0) + _shift(v, 0, -1, 0.0) + _shift(v, 1, 0, 0.0) + _shift(v, -1, 0, 0.0)
            new = acc / np.maximum(cnt, 1.0)[..., None]
            base[upd] = new[upd]
        cur = base
    assert cur is not None
    return cur


def _down_pm(img: np.ndarray, known: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = known.shape
    h2, w2 = (h + 1) // 2, (w + 1) // 2
    pad = ((0, h2 * 2 - h), (0, w2 * 2 - w))
    k = np.pad(known, pad).reshape(h2, 2, w2, 2)
    t = np.pad(target, pad).reshape(h2, 2, w2, 2)
    im = np.pad(img, pad + ((0, 0),)).reshape(h2, 2, w2, 2, 3)
    kc = k.sum(axis=(1, 3))
    tc = t.sum(axis=(1, 3))
    col = (im * k[..., None]).sum(axis=(1, 3)) / np.maximum(kc, 1)[..., None]
    t2 = tc > 0
    k2 = (kc > 0) & ~t2
    return col.astype(np.float32), k2, t2


class _PatchMatch:
    def __init__(
        self,
        img: np.ndarray,
        known: np.ndarray,
        target: np.ndarray,
        r: int,
        rng: np.random.Generator,
        lam: float,
        scale_ref: float,
        source_ok: Optional[np.ndarray] = None,
    ):
        self.r = r
        pad = r + 1
        self.pad = pad
        self.img = np.pad(img, ((pad, pad), (pad, pad), (0, 0))).astype(np.float32)
        self.known = np.pad(known, pad)
        self.target = np.pad(target, pad)
        self.h, self.w = self.known.shape
        oy, ox = np.mgrid[-r : r + 1, -r : r + 1]
        self.oy, self.ox = oy.ravel(), ox.ravel()
        self.off = (self.oy * self.w + self.ox).astype(np.int64)
        win = np.ones(self.known.shape, dtype=bool)
        for y, x in zip(self.oy, self.ox):
            win &= _shift(self.known, -int(y), -int(x), False)
        if source_ok is not None:
            win &= np.pad(source_ok, pad)
        self.valid = win
        self.sy, self.sx = np.nonzero(win)
        self.ty, self.tx = np.nonzero(self.target)
        self.rng = rng
        self.lam = lam
        self.sref = float(scale_ref)
        wt = np.where(self.known, 1.0, np.where(self.target, 0.5, 0.0)).astype(np.float32)
        self.wt = wt.ravel()
        self.ny = np.zeros(0, dtype=np.int64)
        self.nx = np.zeros(0, dtype=np.int64)
        self.d = np.zeros(0, dtype=np.float32)

    def ok(self) -> bool:
        return len(self.sy) >= 8 and len(self.ty) > 0

    def dist(self, sy: np.ndarray, sx: np.ndarray) -> np.ndarray:
        t = (self.ty * self.w + self.tx)[:, None] + self.off[None]
        s = (sy * self.w + sx)[:, None] + self.off[None]
        f = self.img.reshape(-1, 3)
        a = f[t]
        b = f[s]
        w = self.wt[t]
        d = (((a - b) ** 2).sum(axis=-1) * w).sum(axis=1) / (w.sum(axis=1) + 1e-6)
        dd = ((self.ty - sy) ** 2 + (self.tx - sx) ** 2) / (self.sref ** 2)
        return d + self.lam * dd

    def random_src(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        k = self.rng.integers(0, len(self.sy), n)
        return self.sy[k].copy(), self.sx[k].copy()

    def init_nnf(self, prev: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> None:
        n = len(self.ty)
        if prev is None:
            ny, nx = self.random_src(n)
            self.ny, self.nx = ny, nx
            self.d = self.dist(ny, nx)
            for _ in range(5):
                cy, cx = self.random_src(n)
                self._try(cy, cx)
        else:
            pmap_y, pmap_x = prev
            ny = np.clip(pmap_y, self.r + 1, self.h - self.r - 2)
            nx = np.clip(pmap_x, self.r + 1, self.w - self.r - 2)
            bad = ~self.valid[ny, nx]
            ry, rx = self.random_src(n)
            ny = np.where(bad, ry, ny)
            nx = np.where(bad, rx, nx)
            self.ny, self.nx = ny, nx
            self.d = self.dist(ny, nx)

    def _try(self, cy: np.ndarray, cx: np.ndarray) -> None:
        cy = np.clip(cy, 0, self.h - 1)
        cx = np.clip(cx, 0, self.w - 1)
        good = self.valid[cy, cx]
        if not good.any():
            return
        cd = np.full(len(cy), np.inf, dtype=np.float32)
        idx = np.nonzero(good)[0]
        save = self.ty, self.tx
        self.ty, self.tx = save[0][idx], save[1][idx]
        cd[idx] = self.dist(cy[idx], cx[idx])
        self.ty, self.tx = save
        better = cd < self.d
        self.ny = np.where(better, cy, self.ny)
        self.nx = np.where(better, cx, self.nx)
        self.d = np.where(better, cd, self.d)

    def iterate(self, n_iter: int) -> None:
        h, w = self.h, self.w
        tmap = np.full((h, w), -1, dtype=np.int64)
        tmap[self.ty, self.tx] = np.arange(len(self.ty))
        for _ in range(n_iter):
            for step in (1, 2, 4, 1):
                for dy, dx in ((0, step), (0, -step), (step, 0), (-step, 0)):
                    yy, xx = self.ty + dy, self.tx + dx
                    nb = tmap[np.clip(yy, 0, h - 1), np.clip(xx, 0, w - 1)]
                    has = nb >= 0
                    cy = np.where(has, self.ny[np.maximum(nb, 0)] - dy, 0)
                    cx = np.where(has, self.nx[np.maximum(nb, 0)] - dx, 0)
                    self._try(cy, cx)
            R = max(h, w) // 2
            while R >= 1:
                cy = self.ny + self.rng.integers(-R, R + 1, len(self.ny))
                cx = self.nx + self.rng.integers(-R, R + 1, len(self.nx))
                self._try(cy, cx)
                R //= 2
            cy, cx = self.random_src(len(self.ny))
            self._try(cy, cx)

    def vote(self) -> None:
        h, w = self.h, self.w
        f = self.img.reshape(-1, 3)
        sig2 = max(float(np.percentile(self.d, 60)), 1.0)
        wq = np.exp(-self.d / (2.0 * sig2)).astype(np.float64) + 1e-8
        acc = np.zeros((h * w, 3), dtype=np.float64)
        wacc = np.zeros(h * w, dtype=np.float64)
        tflat = self.target.ravel()
        for oy, ox in zip(self.oy, self.ox):
            p = (self.ty + oy) * w + (self.tx + ox)
            s = (self.ny + oy) * w + (self.nx + ox)
            m = tflat[p]
            if not m.any():
                continue
            pm = p[m]
            ww = wq[m]
            wacc += np.bincount(pm, weights=ww, minlength=h * w)
            val = f[s[m]]
            for c in range(3):
                acc[:, c] += np.bincount(pm, weights=ww * val[:, c], minlength=h * w)
        sel = tflat & (wacc > 0)
        f[sel] = (acc[sel] / wacc[sel][:, None]).astype(np.float32)

    def result(self) -> np.ndarray:
        p = self.pad
        return self.img[p:-p, p:-p]

    def nnf_maps(self) -> Tuple[np.ndarray, np.ndarray]:
        p = self.pad
        my = np.full((self.h, self.w), -1, dtype=np.int64)
        mx = np.full((self.h, self.w), -1, dtype=np.int64)
        my[self.ty, self.tx] = self.ny - p
        mx[self.ty, self.tx] = self.nx - p
        return my[p:-p, p:-p], mx[p:-p, p:-p]


def patch_complete(
    img: np.ndarray,
    known: np.ndarray,
    target: np.ndarray,
    seed: int,
    depth: int,
    lam: float = LAM_DEFAULT,
    source_ok: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Multi-scale PatchMatch exemplar completion with optional owner-excluded `source_ok`."""
    rng = np.random.default_rng(seed)
    levels = [(img.astype(np.float32), known.copy(), target & ~known)]
    oks: List[Optional[np.ndarray]] = [source_ok]
    nlev = max(0, min(4, int(math.ceil(math.log2(max(depth, 8) / 6.0)))))
    for _ in range(nlev):
        im, k, t = levels[-1]
        if min(k.shape) < 24:
            break
        levels.append(_down_pm(im, k, t))
        so = oks[-1]
        if so is not None:
            h2, w2 = levels[-1][1].shape
            so = np.pad(so, ((0, h2 * 2 - so.shape[0]), (0, w2 * 2 - so.shape[1]))).reshape(h2, 2, w2, 2).any(axis=(1, 3))
        oks.append(so)
    im, k, t = levels[-1]
    cur, _ = _diffuse_init(im, k, t)
    prev_maps: Optional[Tuple[np.ndarray, np.ndarray]] = None
    for li in range(len(levels) - 1, -1, -1):
        im, k, t = levels[li]
        t = t & ~k
        if li < len(levels) - 1:
            up = np.repeat(np.repeat(cur, 2, axis=0), 2, axis=1)[: im.shape[0], : im.shape[1]]
            base = im.copy()
            base[t] = up[t]
        else:
            base = cur
        r = 3 if li == 0 else 2
        sref = max(depth, 8) * 2.5 / (2 ** li)
        pm = _PatchMatch(base, k, t, r, rng, lam, scale_ref=sref, source_ok=oks[li])
        if not pm.ok():
            pm = _PatchMatch(base, k, t, 1, rng, lam, scale_ref=sref, source_ok=oks[li])
        if not pm.ok():
            cur = base
            prev_maps = None
            continue
        if prev_maps is None:
            pm.init_nnf()
        else:
            my, mx = prev_maps
            yy, xx = pm.ty - pm.pad, pm.tx - pm.pad
            cy = my[np.minimum(yy // 2, my.shape[0] - 1), np.minimum(xx // 2, my.shape[1] - 1)]
            cx = mx[np.minimum(yy // 2, mx.shape[0] - 1), np.minimum(xx // 2, mx.shape[1] - 1)]
            have = cy >= 0
            cy = np.where(have, cy * 2 + (yy % 2), 0) + pm.pad
            cx = np.where(have, cx * 2 + (xx % 2), 0) + pm.pad
            pm.init_nnf((cy, cx))
        n_em = 3 if li > 0 else 2
        for e in range(n_em):
            pm.iterate(2 if e == 0 else 1)
            pm.vote()
            pm.d = pm.dist(pm.ny, pm.nx)
        cur = pm.result()
        prev_maps = pm.nnf_maps()
    return cur


def _straight_over(top: np.ndarray, bot: np.ndarray) -> np.ndarray:
    ta, ba = top[..., 3:4], bot[..., 3:4]
    oa = ta + ba * (1.0 - ta)
    oc = np.where(oa > 1e-6, (top[..., :3] * ta + bot[..., :3] * ba * (1.0 - ta)) / np.maximum(oa, 1e-6), 0.0)
    return np.concatenate([oc, oa], axis=-1)


def _part_seed(name: str) -> int:
    return int(hashlib.sha256(name.encode("utf-8")).hexdigest()[:8], 16)


def _normalize_guide(g: Any) -> Tuple[Any, Any, Optional[str], str]:
    if isinstance(g, dict):
        return (
            g.get("target_zone"),
            g.get("source_zone"),
            g.get("color_pred", g.get("colour_pred")),
            g.get("method", "patch"),
        )
    gt = tuple(g)
    if len(gt) == 3:
        return gt[0], gt[1], gt[2], "patch"
    if len(gt) == 4:
        return gt[0], gt[1], gt[2], gt[3]
    raise ValueError(f"Invalid guide spec: {g!r}")


def _load_part_mask(mask_spec: Any, width: int, height: int) -> Optional[np.ndarray]:
    if mask_spec is None:
        return None
    if isinstance(mask_spec, np.ndarray):
        arr = mask_spec
        if arr.ndim == 3:
            arr = arr[..., -1]
        if arr.shape != (height, width):
            im = Image.fromarray((arr > 0).astype(np.uint8) * 255).resize((width, height), Image.Resampling.NEAREST)
            return np.asarray(im, dtype=np.uint8) > 127
        return arr > 0
    if isinstance(mask_spec, (str, Path)):
        p = Path(mask_spec)
        if not p.is_file():
            return None
        im = Image.open(p).convert("L")
        if im.size != (width, height):
            im = im.resize((width, height), Image.Resampling.NEAREST)
        return np.asarray(im, dtype=np.uint8) > 127
    return None


def _load_exact_ownership_mask(spec: Any, width: int, height: int) -> np.ndarray:
    """Contract masks must not silently disappear or change size."""
    if isinstance(spec, np.ndarray):
        arr = spec
        if arr.ndim == 3:
            arr = arr[..., -1]
        if arr.shape != (height, width):
            raise ValueError("Locked ownership mask size mismatch")
        return arr > 0
    if not isinstance(spec, (str, Path)) or not Path(spec).is_file():
        raise ValueError("Locked ownership mask is missing")
    with Image.open(spec) as image:
        if image.size != (width, height):
            raise ValueError("Locked ownership mask size mismatch")
        return np.asarray(image.convert("L"), dtype=np.uint8) > 127


def _compute_ownership(
    src: np.ndarray,
    parts: List[Dict[str, Any]],
    seams: Dict[frozenset, Any],
    pair_bands: Dict[frozenset, int],
    min_island_px: int,
) -> np.ndarray:
    h, w = src.shape[:2]
    alpha = src[..., 3]
    lab = np.full((h, w), -1, dtype=np.int16)
    constrained = any(p.get("ownership_locked") or p.get("allowed_mask") is not None for p in parts)
    allowed_masks = []
    locked = np.full((h,w), -1, dtype=np.int16)
    for i, p in enumerate(parts):
        allowed = (_load_exact_ownership_mask(p["allowed_mask"], w, h)
                   if p.get("allowed_mask") is not None else np.ones((h,w), dtype=bool))
        allowed_masks.append(allowed)
        m = (_load_exact_ownership_mask(p.get("mask"), w, h) if p.get("ownership_locked")
             else _load_part_mask(p.get("mask"), w, h))
        if m is None:
            m = _poly_mask(w, h, p.get("own"), p.get("line"))
        if p.get("ownership_locked") and (m & ~allowed & (alpha > 0)).any():
            raise ValueError("Locked ownership mask exceeds its allowed domain")
        m = m & allowed & (alpha > 0)
        if p.get("ownership_locked"):
            if ((locked >= 0) & m).any():
                raise ValueError("Overlapping locked ownership requires review")
            locked[m] = i
        lab[m] = i
    lab[locked >= 0] = locked[locked >= 0]
    lab[alpha == 0] = -1

    # Fill any stray alpha>0 pixels from nearest labeled neighbor until convergence
    max_iters = max(60, (h + w) // 4)
    for _ in range(max_iters):
        miss = (lab < 0) & (alpha > 0)
        if not miss.any():
            break
        changed = False
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nb = _shift(lab, dy, dx, -1)
            take = miss & (nb >= 0) & (lab < 0)
            if constrained:
                for idx in np.unique(nb[take]):
                    take[(nb == idx) & ~allowed_masks[int(idx)]] = False
            if take.any():
                lab[take] = nb[take]
                changed = True
        if not changed:
            # Disconnected island with no labeled neighbor: assign to closest labeled part by bbox center
            miss_y, miss_x = np.nonzero((lab < 0) & (alpha > 0))
            if miss_y.size > 0:
                labeled_parts = []
                for idx in range(len(parts)):
                    py, px = np.nonzero(lab == idx)
                    if py.size > 0:
                        labeled_parts.append((idx, float(py.mean()), float(px.mean())))
                if labeled_parts:
                    if constrained:
                        best = np.full(miss_y.shape, -1, dtype=np.int16)
                        distances = np.full(miss_y.shape, np.inf)
                        for idx, cy, cx in labeled_parts:
                            distance = (miss_y-cy)**2 + (miss_x-cx)**2
                            use = allowed_masks[idx][miss_y,miss_x] & (distance < distances)
                            best[use], distances[use] = idx, distance[use]
                        lab[miss_y, miss_x] = best
                    else:
                        cy_m, cx_m = float(miss_y.mean()), float(miss_x.mean())
                        best_idx = min(labeled_parts, key=lambda t: (t[1] - cy_m) ** 2 + (t[2] - cx_m) ** 2)[0]
                        lab[miss_y, miss_x] = best_idx
            break

    zones = {}
    for pair, zone in seams.items():
        if zone == "none":
            zones[pair] = None
        elif zone is None:
            zones[pair] = np.ones((h, w), dtype=bool)
        else:
            zones[pair] = _poly_mask(w, h, zone)

    unknown = np.zeros((h, w), dtype=bool)
    allow = np.zeros((h, w), dtype=np.int64)
    masks = [lab == i for i in range(len(parts))]
    names = [p["name"] for p in parts]

    for i in range(len(parts)):
        if not masks[i].any():
            continue
        near1 = _dilate(masks[i], 1)
        for j in range(i + 1, len(parts)):
            if not (near1 & masks[j]).any():
                continue
            pair = frozenset((names[i], names[j]))
            r = int(pair_bands.get(pair, min(int(parts[i].get("band", 8)), int(parts[j].get("band", 8)))))
            if r <= 0:
                continue
            band = (_dilate(masks[i], r) & masks[j]) | (_dilate(masks[j], r) & masks[i])
            z = zones.get(pair)
            if z is not None:
                band &= ~z
            unknown |= band
            if i < 62 and j < 62:
                allow[band] |= (1 << i) | (1 << j)

    unknown &= (alpha > 0) & (locked < 0)
    if unknown.any():
        grad = _gradient(src[..., :3])
        lab2 = lab.copy()
        lab2[unknown] = -1
        heap: List[Tuple[float, int, int, int, int]] = []
        cnt = 0
        ys, xs = np.nonzero(unknown)
        for y, x in zip(ys.tolist(), xs.tolist()):
            for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w and lab2[yy, xx] >= 0:
                    lbl = int(lab2[yy, xx])
                    if (lbl >= 62 or ((allow[y, x] >> lbl) & 1)) and allowed_masks[lbl][y,x]:
                        heap.append((float(grad[y, x]), cnt, y, x, lbl))
                        cnt += 1
        heapq.heapify(heap)
        while heap:
            _g, _, y, x, l = heapq.heappop(heap)
            if lab2[y, x] >= 0:
                continue
            lab2[y, x] = l
            for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w and unknown[yy, xx] and lab2[yy, xx] < 0:
                    if (l >= 62 or ((allow[yy, xx] >> l) & 1)) and allowed_masks[l][yy,xx]:
                        cnt += 1
                        heapq.heappush(heap, (float(grad[yy, xx]), cnt, yy, xx, l))
        lab2[(lab2 < 0) & unknown] = lab[(lab2 < 0) & unknown]
        lab = lab2

    if min_island_px > 1:
        lab = _clean_islands(lab, len(parts), min_island_px,
                             locked=locked if constrained else None,
                             allowed_masks=allowed_masks if constrained else None)
    if constrained:
        if ((alpha > 0) & (lab < 0)).any():
            raise ValueError("Unassigned foreground requires visual ownership review")
        if not np.array_equal(lab[locked >= 0], locked[locked >= 0]):
            raise ValueError("Locked ownership changed during splitting")
        if any(((lab == i) & ~allowed_masks[i]).any() for i in range(len(parts))):
            raise ValueError("Part escaped its visual ownership domain")
    return lab


def _render_part_map(lab: np.ndarray, num_parts: int, out_path: Path) -> None:
    h, w = lab.shape
    out = np.full((h, w, 3), 24, dtype=np.uint8)
    for i in range(num_parts):
        m = lab == i
        if not m.any():
            continue
        hue = (i * 137.508) % 360.0
        rad = math.radians(hue)
        col = np.array(
            [
                int(128 + 100 * math.cos(rad)),
                int(128 + 100 * math.cos(rad - 2.094)),
                int(128 + 100 * math.cos(rad + 2.094)),
            ],
            dtype=np.uint8,
        )
        out[m] = col
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out).save(out_path)


def split_parts(
    source_image: Union[str, Path],
    part_specs: Sequence[Dict[str, Any]],
    out_dir: Union[str, Path],
    *,
    seams: Optional[Dict[Tuple[str, str], Any]] = None,
    pair_bands: Optional[Dict[Tuple[str, str], int]] = None,
    opaque_threshold: int = OPAQUE_DEFAULT,
    excl_px: int = EXCL_DEFAULT,
    feather_px: int = FEATHER_DEFAULT,
    min_island_px: int = 12,
    module_contract: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Split an RGBA portrait into layered parts with deterministic amodal hidden fills.

    Writes:
    - `<out_dir>/parts/<name>.png`
    - `<out_dir>/parts.json`
    - `<out_dir>/rest_composite.png`
    - `<out_dir>/rest_diff.json`
    - `<out_dir>/review/part_map.png`
    - `<out_dir>/work/labels.npy` and per-part fill caches `<out_dir>/work/fill_<name>_<hash>.npy`
    """
    src_path = Path(source_image).resolve()
    out = Path(out_dir).resolve()
    parts_img_dir = out / "parts"
    work_dir = out / "work"
    review_dir = out / "review"
    parts_img_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)

    src = np.asarray(Image.open(src_path).convert("RGBA"), dtype=np.uint8)
    h, w = src.shape[:2]
    alpha = src[..., 3]
    opaque = alpha >= opaque_threshold

    parts = [dict(p) for p in part_specs]
    if module_contract:
        if module_contract.get("sourceSHA256") != _sha256_file(src_path) or module_contract.get("canvas") != [w,h]:
            raise ValueError("Module contract is not bound to this source")
        if any(not p.get("ownership_locked") or not p.get("module_id") for p in parts):
            raise ValueError("Complete modules require explicit locked visible ownership")
    seams_map = {frozenset(k): v for k, v in (seams or {}).items()}
    bands_map = {frozenset(k): int(v) for k, v in (pair_bands or {}).items()}

    lab = _compute_ownership(src, parts, seams_map, bands_map, min_island_px=min_island_px)
    ownership_domains = [(_load_exact_ownership_mask(p["allowed_mask"], w, h)
                          if p.get("allowed_mask") is not None else None) for p in parts]
    np.save(work_dir / "labels.npy", lab)
    _render_part_map(lab, len(parts), review_dir / "part_map.png")

    # Precompute visible masks, seam feathers, and hidden-fill targets
    n = len(parts)
    V = [lab == i for i in range(n)]
    extra = [np.zeros((h, w), dtype=bool) for _ in range(n)]
    ramps = [np.ones((h, w), dtype=np.float32) for _ in range(n)]

    for i in range(n):
        if not V[i].any():
            continue
        for j in range(i):
            pair = frozenset((parts[i]["name"], parts[j]["name"]))
            if pair not in seams_map or seams_map[pair] == "none":
                continue
            # Entity ownership constrains visible pixels, not hidden amodal fill.
            # Do not feather by copying a different entity's visible RGB into this part.
            if parts[i].get("ownership_locked") or parts[j].get("ownership_locked"):
                continue
            if (ownership_domains[i] is not None and ownership_domains[j] is not None
                    and not np.array_equal(ownership_domains[i], ownership_domains[j])):
                continue
            zone = np.ones((h, w), dtype=bool) if seams_map[pair] is None else _poly_mask(w, h, seams_map[pair])
            y0, y1, x0, x1 = _bbox(V[i], pad=feather_px + 6)
            sl = (slice(y0, y1), slice(x0, x1))
            dj = _dist_small(V[j][sl] & zone[sl], feather_px + 1)
            r = np.clip((dj - 0.5) / max(1, feather_px), 0.0, 1.0)
            rij = np.where(V[i][sl] & _dilate(zone, feather_px + 1)[sl], r, 1.0)
            extra[j][sl] |= (rij < 1.0) & opaque[sl]
            ramps[i][sl] = np.minimum(ramps[i][sl], rij)

    parts_meta: List[Dict[str, Any]] = []
    comp_f = np.zeros((h, w, 4), dtype=np.float32)
    cache_hits = 0

    for i, p in enumerate(parts):
        name = p["name"]
        depth = int(p.get("depth", 30))
        Vi = V[i]
        known = (Vi & opaque) | extra[i]
        target = np.zeros((h, w), dtype=bool)
        fdist = np.zeros((h, w), dtype=np.float32)
        hidden_domain = (_load_exact_ownership_mask(p["hidden_fill_mask"], w, h)
                         if p.get("hidden_fill_mask") is not None else None)
        provided=None;provided_mask=np.zeros((h,w),dtype=bool)
        if p.get('provided_image'):
            from src.gates.review_binding import read_evidence_rgba
            _,_,provided=read_evidence_rgba(p['provided_image'],p.get('provided_image_sha256'),(w,h),root=out.resolve().parent)
            provided_mask=provided[...,3]>0
            if hidden_domain is None or (provided_mask&~hidden_domain).any():
                raise ValueError('Canvas material escapes the declared hidden domain')
        if p.get("module_rule_version") and depth > 0 and hidden_domain is None:
            raise ValueError("Module hidden fill requires an explicit occlusion region")
        if hidden_domain is not None and ((hidden_domain & (alpha != 255)).any() or (hidden_domain & Vi).any()):
            raise ValueError("Module hidden fill requires opaque occlusion and must not modify visible pixels")

        if (Vi | extra[i]).any() and depth > 0:
            y0, y1, x0, x1 = _bbox(Vi | extra[i], pad=depth + 6)
            sl = (slice(y0, y1), slice(x0, x1))
            allowed = (lab[sl] > i) & opaque[sl]
            if hidden_domain is not None:
                allowed &= hidden_domain[sl]
            # A lower layer may complete its own texture underneath higher layers.
            # This never changes lab/visible ownership; depth and the connected
            # contact frontier still limit the fill as in the original pipeline.
            if p.get("clip") or p.get("clip_line"):
                allowed &= _poly_mask(w, h, p.get("clip"), p.get("clip_line"))[sl]
            if p.get("clip_minus"):
                allowed &= ~_poly_mask(w, h, p.get("clip_minus"))[sl]
            allowed &= ~known[sl]
            seed = known[sl] | Vi[sl]
            dist = _chamfer(seed, allowed, maxd=depth + 2)
            target[sl] = allowed & (dist <= depth) & ~provided_mask[sl]
            fdist[sl] = np.where(target[sl], np.minimum(dist, float(depth)), 0.0)
        if hidden_domain is not None and p.get("strict_hidden_fill") and (hidden_domain & ~target & ~provided_mask).any():
            raise ValueError("Declared hidden region exceeds bounded local completion; review rather than extrapolate")

        L = np.zeros((h, w, 4), dtype=np.uint8)
        L[Vi] = src[Vi]
        L[extra[i]] = src[extra[i]]
        if not p.get("module_rule_version"):
            L[..., 3] = np.where((Vi | extra[i]) & opaque, 255, L[..., 3])

        if target.any():
            region = known | target
            fy0, fy1, fx0, fx1 = _bbox(region, pad=8)
            sub = (slice(fy0, fy1), slice(fx0, fx1))
            guides = [_normalize_guide(g) for g in p.get("guides", ())]

            key = hashlib.sha256()
            key.update(ALGO_VERSION.encode("utf-8"))
            key.update(str(p.get("module_rule_version", "legacy")).encode("utf-8"))
            if p.get("module_rule_version"):
                key.update(b"clean-inner-v2-own-gamut")
            key.update(src[sub][..., :3].tobytes())
            key.update(json.dumps({"name": name, "depth": depth, "guides": guides}, sort_keys=True).encode("utf-8"))
            key.update(np.packbits(known[sub]).tobytes())
            key.update(np.packbits(target[sub]).tobytes())
            key.update(np.packbits(Vi[sub]).tobytes())
            key.update(f"{fy0},{fy1},{fx0},{fx1},{excl_px}".encode("utf-8"))
            hk = key.hexdigest()[:16]
            cf = work_dir / f"fill_{name}_{hk}.npy"

            if cf.exists():
                filled_rgb = np.load(cf)
                cache_hits += 1
            else:
                cur = src[sub][..., :3].astype(np.float32)
                k_sub = known[sub]
                t_sub = target[sub]
                own_sub = Vi[sub]
                if p.get("module_rule_version"):
                    # Clean inner source: erode visible by 2px so antialiased bright
                    # rims never seed the hidden fill. Never sample a neighbour.
                    cur[~k_sub] = 0.0
                    clean_known = erode_bool(own_sub, 2)
                    if int(clean_known.sum()) < 16:
                        clean_known = erode_bool(own_sub, 1)
                    if int(clean_known.sum()) < 16:
                        clean_known = own_sub.copy()
                    fill_region = (own_sub & ~clean_known) | t_sub
                    smooth = smooth_membrane(cur, clean_known, fill_region)
                    patched = patch_complete(
                        cur.copy(), clean_known, fill_region, _part_seed(name), depth,
                        source_ok=clean_known,
                    )
                    cur = 0.75 * smooth + 0.25 * patched
                    if clean_known.any():
                        seed_rgb = cur[clean_known]
                        lo = seed_rgb.min(axis=0)
                        hi = seed_rgb.max(axis=0)
                        invent = fill_region & ~clean_known
                        if invent.any():
                            cur[invent] = np.minimum(np.maximum(cur[invent], lo), hi)
                    filled_rgb = np.clip(np.rint(cur), 0, 255).astype(np.uint8)
                else:
                    deep = _dist_small(~own_sub, excl_px + 1) >= excl_px
                    kc = k_sub.copy()
                    todo = t_sub.copy()

                    for gi, (tz, sz, pred, method) in enumerate(guides + [(None, None, None, "patch")]):
                        tk = todo & (_poly_mask(w, h, tz)[sub] if tz else True)
                        if not tk.any():
                            continue
                        so = k_sub & (_poly_mask(w, h, sz)[sub] if sz else True) & _colour_pred(src[sub], pred)
                        if method == "smooth":
                            kb = so | (kc & ~k_sub)
                            if not kb.any():
                                continue
                            _, unreached = _diffuse_init(cur, kb, tk)
                            tk_eff = tk & ~unreached
                            if not tk_eff.any():
                                continue
                            cur = smooth_membrane(cur, kb, tk_eff)
                            kc |= tk_eff
                            todo &= ~tk_eff
                        elif method == "patch":
                            so_d = so & deep
                            if int(so_d.sum()) < 200:
                                so_d = so & (_dist_small(~own_sub, 3) >= 2)
                            if int(so_d.sum()) < 50:
                                so_d = so
                            cur = patch_complete(cur, kc, tk, _part_seed(name) + gi, depth, source_ok=so_d)
                            kc |= tk
                            todo &= ~tk
                        else:
                            raise ValueError(f"Unsupported fill guide method: {method!r}")

                    filled_rgb = np.clip(np.rint(cur), 0, 255).astype(np.uint8)
                for old in work_dir.glob(f"fill_{name}_*.npy"):
                    old.unlink()
                np.save(cf, filled_rgb)

            fade = max(4.0, depth / 4.0)
            fade_start = max(0.0, float(depth) - fade)
            t_sub = target[sub]
            fd_sub = fdist[sub]
            alpha_fill = np.clip((float(depth) - fd_sub) / max(1.0, fade), 0.0, 1.0)
            alpha_fill = np.where(fd_sub <= fade_start, 1.0, alpha_fill)
            if p.get("module_rule_version") and p.get("strict_hidden_fill"):
                # Keep the vis-target seam fully opaque (no 2px groove) and fade
                # only the deepest 4px of the hidden band.
                fade = 4.0
                opaque_until = max(1.0, float(depth) - fade)
                alpha_fill = np.clip((float(depth) - fd_sub) / max(1.0, fade), 0.0, 1.0)
                alpha_fill = np.where(fd_sub <= opaque_until, 1.0, alpha_fill)
            L_sub = L[sub]
            L_sub[t_sub, :3] = filled_rgb[t_sub]
            L_sub[t_sub, 3] = np.clip(np.rint(alpha_fill[t_sub] * 255.0), 0, 255).astype(np.uint8)
            L[sub] = L_sub

        # The external material has already been registered to the source canvas
        # and clipped to an opaque occlusion. Do not replace it with local colour
        # diffusion or a depth-limited approximation; visible pixels remain frozen.
        if provided is not None:
            L[provided_mask]=provided[provided_mask]
            target|=provided_mask

        # Apply seam ramp feathering on visible pixels where applicable
        if (ramps[i] < 1.0).any():
            L[..., 3] = np.clip(np.rint(L[..., 3].astype(np.float32) * ramps[i]), 0, 255).astype(np.uint8)

        # Visible source pixels are bitwise frozen after any hidden fill.
        if p.get("module_rule_version"):
            L[Vi] = src[Vi]

        # Zero RGB on completely transparent pixels
        L[L[..., 3] == 0] = 0

        # Composite into rest_composite
        comp_f = _straight_over(L.astype(np.float32) / 255.0, comp_f)

        # Save tight cropped part PNG
        nz = L[..., 3] > 0
        if nz.any():
            py0, py1, px0, px1 = _bbox(nz, pad=0)
            crop = L[py0:py1, px0:px1]
            part_bbox = [int(px0), int(py0), int(px1 - px0), int(py1 - py0)]
        else:
            crop = np.zeros((1, 1, 4), dtype=np.uint8)
            part_bbox = [0, 0, 1, 1]

        rel_img = f"parts/{name}.png"
        Image.fromarray(crop).save(out / rel_img)

        piv = p.get("pivot")
        if piv is None:
            piv = [part_bbox[0] + part_bbox[2] * 0.5, part_bbox[1] + part_bbox[3] * 0.5]

        entry: Dict[str, Any] = {
            "name": name,
            "drawOrder": i,
            "kind": p.get("kind", "rigid"),
            "parent": p.get("parent"),
            "pivot": [float(piv[0]), float(piv[1])],
            "bbox": part_bbox,
            "image": rel_img,
            "joints": p.get("joints") or {},
            "notes": p.get("notes", ""),
        }
        if p.get("module_id"):
            entry.update(moduleId=p["module_id"], pieceId=p["piece_id"],
                         moduleRuleVersion=p.get("module_rule_version"),
                         visibleMask=p.get("mask"), completionMode=p.get("completion_mode","none"),
                         hiddenMask=p.get("hidden_fill_mask"), hiddenPixels=int(target.sum()),
                         completionVisualReview="pending" if target.any() else "not_applicable")
            if p.get('provided_image'):
                entry.update(providedImage=p['provided_image'],providedImageSHA256=p['provided_image_sha256'])
        if p.get("centerline"):
            entry["centerline"] = [[float(q[0]), float(q[1])] for q in p["centerline"]]
        parts_meta.append(entry)

    rest_comp = np.clip(np.rint(comp_f * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(rest_comp).save(out / "rest_composite.png")

    # Evaluate rest_diff against source (normalizing interior >= opaque_threshold alpha to 255)
    src_norm = src.copy()
    complete_modules = bool(module_contract) or any(p.get("module_rule_version") for p in parts)
    if not complete_modules:
        src_norm[..., 3] = np.where(src_norm[..., 3] >= opaque_threshold, 255, src_norm[..., 3])
    src_norm[src_norm[..., 3] == 0] = 0
    rest_norm = rest_comp.copy()
    rest_norm[rest_norm[..., 3] == 0] = 0
    diff = np.abs(rest_norm.astype(np.int16) - src_norm.astype(np.int16))
    max_rgba = int(diff.max()) if diff.size else 0
    pixels_over_2 = int((diff.max(axis=-1) > 2).sum()) if diff.size else 0
    mean_abs = round(float(diff.mean()), 6) if diff.size else 0.0

    rest_diff = {
        "source": str(src_path),
        "sourceSHA256": _sha256_file(src_path),
        "maxRGBA": max_rgba,
        "pixelsOver2": pixels_over_2,
        "meanAbsDiff": mean_abs,
        "passed": max_rgba <= 2,
    }
    (out / "rest_diff.json").write_text(json.dumps(rest_diff, indent=2) + "\n", encoding="utf-8")

    parts_json = {
        "version": "1.0",
        "source": str(src_path),
        "sourceSHA256": rest_diff["sourceSHA256"],
        "canvas": [w, h],
        "opaqueThreshold": 256 if complete_modules else opaque_threshold,
        "preserveSourceRGBA": complete_modules,
        "cacheHits": cache_hits,
        "parts": parts_meta,
    }
    if module_contract:
        parts_json["moduleContract"] = module_contract
    (out / "parts.json").write_text(json.dumps(parts_json, indent=2) + "\n", encoding="utf-8")
    return {
        "parts_json": parts_json,
        "rest_diff": rest_diff,
        "out_dir": str(out),
        "cache_hits": cache_hits,
    }
