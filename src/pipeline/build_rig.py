"""Generalized Spine 4.2.43 rig and harmonic idle animation builder.

Supports:
- `region` attachments and weighted `mesh` attachments (`grid_mesh`);
- weight recipes: `chain`, `linear_y`, `linear_x`, `limb`, `radial`, and `contact_dist`
  (distance-based contact weights matching Erika v2's CONTACT_DIST so overlapping arm/hair/torso seams never slide);
- 1-bone and 2-bone IK constraints (auto-computing `bendPositive` when omitted);
- zero-at-start cubic-bezier 8s periodic loop + optional `twitch` + preset application + `gain`;
- outputs `<name>.spine.json`, 4px-padded `images/*.png`, and `rig-build-report.json`.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image, ImageFilter

from src.render.spine_runtime import Skeleton

from .presets import DEFAULT_DURATION, DEFAULT_FPS, DEFAULT_KEY_STEP, get_preset, resolve_preset_motion

SPINE_VERSION = "4.2.43"
REVISION = "portrait-spine-studio-v1"
PAD = 4


def smoothstep(a: float, b: float, v: np.ndarray) -> np.ndarray:
    if abs(b - a) < 1e-12:
        return (v >= b).astype(np.float64)
    u = np.clip((v - a) / (b - a), 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def pad_image(src_png: Path, dst_png: Path, pad: int = PAD) -> np.ndarray:
    im = np.asarray(Image.open(src_png).convert("RGBA"), dtype=np.uint8)
    out = np.zeros((im.shape[0] + 2 * pad, im.shape[1] + 2 * pad, 4), dtype=np.uint8)
    out[pad:-pad, pad:-pad] = im
    out[out[..., 3] == 0] = 0
    dst_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out).save(dst_png, optimize=True)
    return out


def resample_polyline(pts: Sequence[Sequence[float]], count: int) -> np.ndarray:
    arr = np.asarray(pts, dtype=np.float64)
    if len(arr) < 2:
        raise ValueError("Chain polyline must have at least 2 points")
    if count == len(arr) - 1:
        return arr
    seg = np.linalg.norm(np.diff(arr, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-9:
        return np.repeat(arr[:1], count + 1, axis=0)
    q = np.linspace(0.0, s[-1], count + 1)
    return np.stack([np.interp(q, s, arr[:, 0]), np.interp(q, s, arr[:, 1])], axis=1)


def grid_mesh(img: np.ndarray, bbox_pad: Tuple[float, float], cell: int = 18) -> Tuple[np.ndarray, np.ndarray, List[int], int]:
    """Regular grid mesh over padded RGBA image: perimeter hull first, then interior, triangles where alpha > 0."""
    h, w = img.shape[:2]
    nx = max(1, int(round(w / float(cell))))
    ny = max(1, int(round(h / float(cell))))
    xs = np.linspace(0.0, float(w), nx + 1)
    ys = np.linspace(0.0, float(h), ny + 1)
    perim = (
        [(i, 0) for i in range(nx)]
        + [(nx, j) for j in range(ny)]
        + [(i, ny) for i in range(nx, 0, -1)]
        + [(0, j) for j in range(ny, 0, -1)]
    )
    interior = [(i, j) for j in range(1, ny) for i in range(1, nx)]
    order = perim + interior
    index = {ij: k for k, ij in enumerate(order)}
    a = img[..., 3] > 0
    tris: List[int] = []
    for j in range(ny):
        for i in range(nx):
            x0, x1 = int(math.floor(xs[i])), int(math.ceil(xs[i + 1]))
            y0, y1 = int(math.floor(ys[j])), int(math.ceil(ys[j + 1]))
            if not a[max(0, y0 - 1) : y1 + 1, max(0, x0 - 1) : x1 + 1].any():
                continue
            v00, v10, v11, v01 = index[(i, j)], index[(i + 1, j)], index[(i + 1, j + 1)], index[(i, j + 1)]
            if (i + j) % 2 == 0:
                tris.extend([v00, v10, v11, v00, v11, v01])
            else:
                tris.extend([v00, v10, v01, v10, v11, v01])
    if not tris:
        # Fallback: include full quad so slot always has valid geometry
        v00, v10, v11, v01 = index[(0, 0)], index[(nx, 0)], index[(nx, ny)], index[(0, ny)]
        tris = [v00, v10, v11, v00, v11, v01]
    uv = np.array([[xs[i] / w, ys[j] / h] for i, j in order], dtype=np.float64)
    canvas = np.array([[bbox_pad[0] + xs[i], bbox_pad[1] + ys[j]] for i, j in order], dtype=np.float64)
    return canvas, uv, tris, len(perim)


def build_contact_distance_field(parts_dir: Path, part_meta: Dict[str, Any], canvas_w: int, canvas_h: int, max_dist: int = 64) -> Callable[[np.ndarray], np.ndarray]:
    """Compute Chebyshev distance transform (0..max_dist) from a part's opaque pixels (alpha >= 128)."""
    a = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    x, y, w, h = part_meta["bbox"]
    im_a = np.asarray(Image.open(parts_dir / part_meta["image"]).convert("RGBA"), dtype=np.uint8)[..., 3]
    a[y : y + h, x : x + w] = im_a
    mask = a >= 128
    d = np.full(mask.shape, float(max_dist), dtype=np.float32)
    d[mask] = 0.0
    im = Image.fromarray(mask.astype(np.uint8) * 255)
    for k in range(1, max_dist + 1):
        im = im.filter(ImageFilter.MaxFilter(3))
        take = (np.asarray(im) > 0) & (d == float(max_dist))
        if not take.any():
            break
        d[take] = float(k)

    def query(P: np.ndarray) -> np.ndarray:
        yy = np.clip(np.rint(P[:, 1]).astype(np.int64), 0, canvas_h - 1)
        xx = np.clip(np.rint(P[:, 0]).astype(np.int64), 0, canvas_w - 1)
        return d[yy, xx].astype(np.float64)

    return query


def chain_weights(P: np.ndarray, joints: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-vertex weights along a bone chain (column 0 = parent anchor, columns 1..N = chain links)."""
    J = np.asarray(joints, dtype=np.float64)
    n = len(J) - 1
    seg = np.diff(J, axis=0)
    L = np.maximum(np.linalg.norm(seg, axis=1), 1e-9)
    cum = np.concatenate([[0.0], np.cumsum(L)])
    best_d = np.full(len(P), np.inf, dtype=np.float64)
    s_par = np.zeros(len(P), dtype=np.float64)
    for i in range(n):
        d = seg[i] / L[i]
        rel = P - J[i]
        t = rel @ d
        lo = -np.inf if i == 0 else 0.0
        hi = np.inf if i == n - 1 else L[i]
        tc = np.clip(t, lo, hi)
        q = J[i] + tc[:, None] * d
        dist = np.linalg.norm(P - q, axis=1)
        take = dist < best_d
        best_d[take] = dist[take]
        s_par[take] = cum[i] + tc[take]
    r = 0.35 * float(L.min())
    Wt = np.zeros((len(P), n + 1), dtype=np.float64)
    edges = np.concatenate([[0.0], cum[1:-1]])
    for i in range(n + 1):
        if i == 0:
            Wt[:, i] = 1.0
        else:
            start = float(edges[i - 1])
            Wt[:, i] = smoothstep(start - r, start + r, s_par)
    out = np.zeros_like(Wt)
    for i in range(n + 1):
        out[:, i] = Wt[:, i] - (Wt[:, i + 1] if i + 1 <= n else 0.0)
    return np.clip(out, 0.0, 1.0), best_d


def _sparse_weights(cols: List[np.ndarray], names: List[str], max_influences: int = 4) -> List[List[Tuple[str, float]]]:
    """Normalize and keep top-K weights per vertex with exact partition of unity."""
    mat = np.stack(cols, axis=1)
    mat = np.clip(mat, 0.0, 1.0)
    out: List[List[Tuple[str, float]]] = []
    for k in range(len(mat)):
        # Merge duplicate bone names if any
        merged: Dict[str, float] = {}
        for j, nm in enumerate(names):
            v = float(mat[k, j])
            if v > 1e-6:
                merged[nm] = merged.get(nm, 0.0) + v
        if not merged:
            merged[names[0]] = 1.0
        top = sorted(merged.items(), key=lambda kv: -kv[1])[:max_influences]
        total = sum(v for _, v in top)
        if total <= 1e-12:
            out.append([(top[0][0], 1.0)])
            continue
        ws = [round(v / total, 5) for _, v in top]
        ws[0] = round(1.0 - sum(ws[1:]), 5)
        out.append([(nm, float(w)) for (nm, _), w in zip(top, ws) if w > 0.0])
    return out


def _compute_mesh_weights(
    recipe: Dict[str, Any],
    P: np.ndarray,
    bone_pos_canvas: Dict[str, np.ndarray],
    chains_info: Dict[str, Dict[str, Any]],
    contact_fields: Dict[str, Callable[[np.ndarray], np.ndarray]],
) -> Tuple[List[List[Tuple[str, float]]], str]:
    """Compute per-vertex bone weights for a mesh recipe + optional `contact_overrides`."""
    rtype = recipe["recipe"]
    x, y = P[:, 0], P[:, 1]
    cols: List[np.ndarray]
    names: List[str]
    slot_bone: str

    if rtype == "rigid":
        names = [recipe["bone"]]
        cols = [np.ones(len(P), dtype=np.float64)]
        slot_bone = recipe["bone"]
    elif rtype == "chain":
        cname = recipe["chain"]
        cinfo = chains_info[cname]
        w, _ = chain_weights(P, cinfo["joints"])
        names = [cinfo["parent"]] + list(cinfo["bones"])
        cols = [w[:, i] for i in range(w.shape[1])]
        slot_bone = recipe.get("slot_bone", cinfo["parent"])
    elif rtype in ("linear_y", "linear_x"):
        coord = y if rtype == "linear_y" else x
        bones = list(recipe["bones"])
        ranges = list(recipe["ranges"])  # len == len(bones) - 1, each [a, b]
        if len(bones) < 2 or len(ranges) != len(bones) - 1:
            raise ValueError(f"{rtype} requires N>=2 bones and N-1 ranges")
        cum_on = [np.ones(len(P), dtype=np.float64)]
        for a, b in ranges:
            cum_on.append(smoothstep(float(a), float(b), coord))
        cols = [np.clip(cum_on[i] - (cum_on[i + 1] if i + 1 < len(cum_on) else 0.0), 0.0, 1.0) for i in range(len(bones))]
        names = bones
        slot_bone = recipe.get("slot_bone", bones[0])
    elif rtype == "limb":
        start_bone = recipe["start_bone"]
        mid_bone = recipe["mid_bone"]
        end_bone = recipe.get("end_bone")
        p0 = bone_pos_canvas[mid_bone]
        p1 = bone_pos_canvas[end_bone] if end_bone else (p0 + np.array([0.0, 100.0]))
        axis = p1 - p0
        denom = max(float(np.sum(axis ** 2)), 1e-9)
        f = np.clip(((P - p0) @ axis) / denom, 0.0, 1.0)
        u0, u1 = recipe.get("root_blend", [0.0, 0.30])
        v0, v1 = recipe.get("tip_blend", [0.55, 0.95])
        u = smoothstep(float(u0), float(u1), f)
        if end_bone:
            v = smoothstep(float(v0), float(v1), f)
            cols = [1.0 - u, u * (1.0 - v), u * v]
            names = [start_bone, mid_bone, end_bone]
        else:
            cols = [1.0 - u, u]
            names = [start_bone, mid_bone]
        slot_bone = recipe.get("slot_bone", mid_bone)
    elif rtype == "dual_limb":
        # Two-branch limb weighting (e.g. left & right legs sharing one `legs` slot)
        root_bone = recipe["root_bone"]
        branches = recipe["branches"]  # list of {mid_bone, end_bone}
        names = [root_bone]
        cols = [np.zeros(len(P), dtype=np.float64)]
        dists = []
        per_branch_t = []
        for br in branches:
            mb = br["mid_bone"]
            eb = br.get("end_bone")
            p0 = bone_pos_canvas[mb]
            p1 = bone_pos_canvas[eb] if eb else (p0 + np.array([0.0, 100.0]))
            axis = p1 - p0
            denom = max(float(np.sum(axis ** 2)), 1e-9)
            t = np.clip(((P - p0) @ axis) / denom, 0.0, 1.0)
            proj = p0 + t[:, None] * axis
            dists.append(np.linalg.norm(P - proj, axis=1))
            per_branch_t.append((mb, eb, t))
        # Soft nearest-branch assignment (sharp enough to prevent cross-leg bleed, smooth at crotch)
        d_stack = np.stack(dists, axis=1)
        scale_d = max(12.0, float(np.percentile(d_stack, 25)) * 0.35)
        inv_w = np.exp(-d_stack / scale_d)
        inv_w /= np.maximum(inv_w.sum(axis=1, keepdims=True), 1e-9)
        u0, u1 = recipe.get("root_blend", [0.0, 0.25])
        v0, v1 = recipe.get("tip_blend", [0.45, 0.85])
        root_acc = np.zeros(len(P), dtype=np.float64)
        for bi, (mb, eb, t) in enumerate(per_branch_t):
            bw_b = inv_w[:, bi]
            u = smoothstep(float(u0), float(u1), t)
            root_acc += bw_b * (1.0 - u)
            if eb:
                v = smoothstep(float(v0), float(v1), t)
                cols.append(bw_b * u * (1.0 - v))
                names.append(mb)
                cols.append(bw_b * u * v)
                names.append(eb)
            else:
                cols.append(bw_b * u)
                names.append(mb)
        cols[0] = root_acc
        slot_bone = recipe.get("slot_bone", root_bone)
    elif rtype == "radial":
        base_bone = recipe["base_bone"]
        centers = recipe["centers"]  # list of {bone, pos?, r_in, r_out}
        ws_list: List[np.ndarray] = []
        cnames: List[str] = []
        for c in centers:
            bn = c["bone"]
            pos = np.asarray(c["pos"], dtype=np.float64) if "pos" in c else bone_pos_canvas[bn]
            dist = np.linalg.norm(P - pos, axis=1)
            ws_list.append(1.0 - smoothstep(float(c["r_in"]), float(c["r_out"]), dist))
            cnames.append(bn)
        s = sum(ws_list)
        scale = np.where(s > 1.0, 1.0 / np.maximum(s, 1e-9), 1.0)
        ws_list = [w * scale for w in ws_list]
        base_w = np.clip(1.0 - sum(ws_list), 0.0, 1.0)
        cols = [base_w] + ws_list
        names = [base_bone] + cnames
        slot_bone = recipe.get("slot_bone", base_bone)
    elif rtype == "contact_dist":
        base_bone = recipe["base_bone"]
        contacts = recipe["contacts"]  # list of {part, bone, d_in, d_out}
        cols = [np.ones(len(P), dtype=np.float64)]
        names = [base_bone]
        for ct in contacts:
            q = 1.0 - smoothstep(float(ct["d_in"]), float(ct["d_out"]), contact_fields[ct["part"]](P))
            cols = [c * (1.0 - q) for c in cols] + [q]
            names.append(ct["bone"])
        slot_bone = recipe.get("slot_bone", base_bone)
    else:
        raise ValueError(f"Unsupported mesh weight recipe: {rtype!r}")

    # Optional contact_dist overrides composable on top of ANY mesh recipe
    for ct in recipe.get("contact_overrides", ()):
        q = 1.0 - smoothstep(float(ct["d_in"]), float(ct["d_out"]), contact_fields[ct["part"]](P))
        if "y_min" in ct and "y_max" in ct:
            q = q * smoothstep(float(ct["y_min"]), float(ct["y_max"]), y)
        cols = [c * (1.0 - q) for c in cols] + [q]
        names.append(ct["bone"])

    return _sparse_weights(cols, names), slot_bone


def harmonic(spec: Sequence[Tuple[float, int, float]], t: float, duration: float, gain: float = 1.0, deriv: bool = False) -> float:
    w = 2.0 * math.pi / duration
    if deriv:
        return sum(a * gain * k * w * math.cos(k * w * t - ph) for a, k, ph in spec)
    return sum(a * gain * (math.sin(k * w * t - ph) - math.sin(-ph)) for a, k, ph in spec)


def twitch_val(t: float, start: float, amp: float, gain: float = 1.0) -> float:
    up, down = 0.25, 0.80
    a = amp * gain
    if start <= t < start + up:
        u = (t - start) / up
        return a * u * u * (3.0 - 2.0 * u)
    if start + up <= t < start + up + down:
        u = (t - start - up) / down
        return a * (1.0 - u * u * (3.0 - 2.0 * u))
    return 0.0


def _fn_from(
    spec: Optional[Sequence[Tuple[float, int, float]]],
    duration: float,
    gain: float,
    offset: float = 0.0,
    twitch_spec: Optional[Tuple[float, float]] = None,
) -> Callable[[float, bool], float]:
    def f(t: float, deriv: bool = False) -> float:
        v = harmonic(spec, t, duration, gain, deriv=deriv) if spec else 0.0
        if twitch_spec is not None:
            st, amp = float(twitch_spec[0]), float(twitch_spec[1])
            if deriv:
                h = 1e-4
                v += (twitch_val(t + h, st, amp, gain) - twitch_val(t - h, st, amp, gain)) / (2.0 * h)
            else:
                v += twitch_val(t, st, amp, gain)
        return v + (0.0 if deriv else offset)

    return f


def _keyframes(fn: Callable[[float, bool], float], duration: float, key_step: float, extra_times: Sequence[float] = ()) -> Tuple[List[float], List[float], List[float]]:
    n_steps = int(round(duration / key_step))
    base_times = [round(i * key_step, 4) for i in range(n_steps + 1)]
    if base_times[-1] != round(duration, 4):
        base_times.append(round(duration, 4))
    times = sorted(set(base_times) | {round(t, 4) for t in extra_times if 0.0 <= t <= duration})
    vals = [fn(t, False) for t in times]
    ders = [fn(t, True) for t in times]
    vals[-1] = vals[0]
    ders[-1] = ders[0]
    return times, vals, ders


def bezier_keys(channels: Sequence[Tuple[List[float], List[float]]], value_names: Sequence[str], times: Sequence[float]) -> List[Dict[str, Any]]:
    keys: List[Dict[str, Any]] = []
    for i, t in enumerate(times):
        k: Dict[str, Any] = {}
        if t:
            k["time"] = round(float(t), 4)
        for (vals, _ders), nm in zip(channels, value_names):
            k[nm] = round(float(vals[i]), 5)
        if i + 1 < len(times):
            dt = float(times[i + 1] - t)
            curve: List[float] = []
            for vals, ders in channels:
                curve.extend(
                    [
                        round(float(t + dt / 3.0), 5),
                        round(float(vals[i] + ders[i] * dt / 3.0), 5),
                        round(float(times[i + 1] - dt / 3.0), 5),
                        round(float(vals[i + 1] - ders[i + 1] * dt / 3.0), 5),
                    ]
                )
            k["curve"] = curve
        keys.append(k)
    return keys


def check_zero_at_start(anim: Dict[str, Any]) -> bool:
    ident = {"rotate": {"value": 0.0}, "translate": {"x": 0.0, "y": 0.0}, "scale": {"x": 1.0, "y": 1.0}}
    for bname, tls in anim.get("bones", {}).items():
        for kind, keys in tls.items():
            for k in (keys[0], keys[-1]):
                for field, v0 in ident[kind].items():
                    if abs(float(k.get(field, v0)) - v0) > 1e-6:
                        raise AssertionError(
                            f"Non-zero boundary key on {bname}.{kind}.{field} at t={k.get('time', 0.0)}: {k.get(field)}"
                        )
    return True


def build_rig(
    parts_dir: Union[str, Path],
    rig_spec: Dict[str, Any],
    out_dir: Union[str, Path],
    *,
    preset: str = "agile_full",
    gain: float = 1.0,
    include_animation: bool = True,
) -> Dict[str, Any]:
    """Build a Spine 4.2.43 rig from `parts_dir` and `rig_spec` into `out_dir`."""
    module_contract = rig_spec.get("moduleContract")
    if module_contract:
        from .module_rig import validate_module_rig_spec, MAX_MODULE_GAIN
        from src.studio.vision_schema import VisionReviewRequired
        validate_module_rig_spec(rig_spec)
        if not math.isfinite(gain) or not 0 < gain <= MAX_MODULE_GAIN:
            raise VisionReviewRequired("模块动作gain必须非零且在统一上限内，不以冻结或过幅换取通过。")
    pdir = Path(parts_dir).resolve()
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    pj_path = pdir / "parts.json"
    pj = json.loads(pj_path.read_text(encoding="utf-8"))
    parts = sorted(pj["parts"], key=lambda x: x["drawOrder"])
    by_part = {p["name"]: p for p in parts}

    W, H = int(pj["canvas"][0]), int(pj["canvas"][1])
    origin = rig_spec.get("origin")
    OX, OY = (float(origin[0]), float(origin[1])) if origin else (W / 2.0, float(H))

    preset_obj = get_preset(preset)
    duration = float(rig_spec.get("duration", preset_obj["duration"]))
    fps = int(rig_spec.get("fps", preset_obj["fps"]))
    key_step = float(rig_spec.get("key_step", preset_obj["key_step"]))
    character = str(rig_spec.get("character", "character"))
    bone_roles = {b["name"]: b["role"] for b in rig_spec["bones"] if b.get("role")}

    def to_spine(pt: Sequence[float]) -> np.ndarray:
        return np.array([float(pt[0]) - OX, OY - float(pt[1])], dtype=np.float64)

    # 1. Assemble bone rows (base bones + chain bones)
    rows: List[Dict[str, Any]] = []
    for b in rig_spec["bones"]:
        rows.append(
            {
                "name": b["name"],
                "parent": b.get("parent"),
                "pos": [float(b["pos"][0]), float(b["pos"][1])],
                "aim": [float(b["aim"][0]), float(b["aim"][1])] if b.get("aim") is not None else None,
            }
        )

    chains_info: Dict[str, Dict[str, Any]] = {}
    chains_bones_map: Dict[str, List[str]] = {}
    for cname, cspec in rig_spec.get("chains", {}).items():
        pts = cspec.get("pts")
        if pts is None and cspec.get("part") in by_part and by_part[cspec["part"]].get("centerline"):
            pts = by_part[cspec["part"]]["centerline"]
        if pts is None:
            raise ValueError(f"Chain {cname!r} has neither pts nor part centerline")
        count = int(cspec.get("count", len(pts) - 1))
        joints = resample_polyline(pts, count)
        parent = cspec["parent"]
        link_names: List[str] = []
        for i in range(count):
            bn = f"{cname}{i + 1}"
            rows.append(
                {
                    "name": bn,
                    "parent": parent if i == 0 else f"{cname}{i}",
                    "pos": [float(joints[i, 0]), float(joints[i, 1])],
                    "aim": [float(joints[i + 1, 0]), float(joints[i + 1, 1])],
                }
            )
            link_names.append(bn)
        chains_info[cname] = {"parent": parent, "joints": joints, "bones": link_names}
        chains_bones_map[cname] = link_names

    # 2. Convert bone rows to Spine local coordinates
    world_map: Dict[str, Tuple[np.ndarray, float]] = {}
    bone_pos_canvas: Dict[str, np.ndarray] = {}
    bones_json: List[Dict[str, Any]] = []
    for r in rows:
        bone_pos_canvas[r["name"]] = np.asarray(r["pos"], dtype=np.float64)
        p = to_spine(r["pos"])
        if r["aim"] is not None:
            dv = to_spine(r["aim"]) - p
            ang = math.degrees(math.atan2(dv[1], dv[0]))
            length = float(np.linalg.norm(dv))
        else:
            ang = 0.0
            length = 0.0
        world_map[r["name"]] = (p, ang)
        bd: Dict[str, Any] = {"name": r["name"]}
        if r["parent"]:
            pp, pa = world_map[r["parent"]]
            rad = math.radians(-pa)
            c, s = math.cos(rad), math.sin(rad)
            loc = np.array([[c, -s], [s, c]], dtype=np.float64) @ (p - pp)
            bd["parent"] = r["parent"]
            bd["x"] = round(float(loc[0]), 4)
            bd["y"] = round(float(loc[1]), 4)
            bd["rotation"] = round(float(ang - pa), 4)
        else:
            bd["x"] = round(float(p[0]), 4)
            bd["y"] = round(float(p[1]), 4)
            if abs(ang) > 1e-9:
                bd["rotation"] = round(float(ang), 4)
        if length > 0.0:
            bd["length"] = round(length, 4)
        bones_json.append(bd)

    # 3. Build IK constraints (auto-computing bendPositive for 2-bone IK when omitted)
    ik_json: List[Dict[str, Any]] = []
    for order_idx, c in enumerate(rig_spec.get("ik", [])):
        cbones = list(c["bones"])
        entry: Dict[str, Any] = {
            "name": c["name"],
            "order": int(c.get("order", order_idx)),
            "bones": cbones,
            "target": c["target"],
            "mix": 1.0,
        }
        if c.get("stretch"):
            if len(cbones) != 2 or c.get("uniform") is not True:
                raise ValueError("Only explicit two-bone uniform IK stretch is supported")
            entry.update(stretch=True, uniform=True)
        if len(cbones) == 2:
            if "bendPositive" in c and c["bendPositive"] is not None:
                entry["bendPositive"] = bool(c["bendPositive"])
            else:
                p0 = world_map[cbones[0]][0]
                p1 = world_map[cbones[1]][0]
                pt = world_map[c["target"]][0]
                u = p1 - p0
                v = pt - p0
                cross = float(u[0] * v[1] - u[1] * v[0])
                entry["bendPositive"] = cross >= 0.0
        ik_json.append(entry)

    skel: Dict[str, Any] = {
        "skeleton": {
            "hash": "",
            "spine": SPINE_VERSION,
            "x": -OX,
            "y": float(OY - H),
            "width": W,
            "height": H,
            "fps": fps,
            "images": "./images/",
        },
        "bones": bones_json,
        "ik": ik_json,
        "slots": [],
        "skins": [{"name": "default", "attachments": {}}],
        "animations": {},
    }

    # Evaluate setup world matrices
    sk = Skeleton(skel)
    sk.setup_pose()
    sk.update_world()
    if module_contract:
        from .module_rig import validate_expanded_soft_chains
        parent_of = {b.name: (None if b.parent is None else b.parent.name) for b in sk.bones}
        validate_expanded_soft_chains(module_contract, parent_of)
    mats = {b.name: b.matrix() for b in sk.bones}
    bidx = sk.bone_index

    # Precompute contact distance fields only for referenced parts
    skin_specs = rig_spec.get("skin", {})
    needed_contact_parts = set()
    for _pname, sspec in skin_specs.items():
        if sspec.get("recipe") == "contact_dist":
            for ct in sspec.get("contacts", ()):
                needed_contact_parts.add(ct["part"])
        for ct in sspec.get("contact_overrides", ()):
            needed_contact_parts.add(ct["part"])
    contact_fields = {
        pn: build_contact_distance_field(pdir, by_part[pn], W, H)
        for pn in sorted(needed_contact_parts)
        if pn in by_part
    }

    # 4. Build padded images, slots, and region/mesh attachments
    att_report: List[Dict[str, Any]] = []
    for p in parts:
        name = p["name"]
        im = pad_image(pdir / p["image"], img_dir / f"{name}.png", pad=PAD)
        h_pad, w_pad = im.shape[:2]
        bx, by_ = float(p["bbox"][0] - PAD), float(p["bbox"][1] - PAD)

        sspec = skin_specs.get(name)
        if sspec is None:
            # Sensible default: rigid attachment on matching bone name or root
            default_bone = name if name in mats else "root"
            sspec = {"type": "region", "bone": default_bone}

        att_type = sspec.get("type", "region")
        if att_type == "region":
            slot_bone = sspec["bone"]
            M = mats[slot_bone]
            center_sp = to_spine((bx + w_pad / 2.0, by_ + h_pad / 2.0))
            loc = np.linalg.solve(M[:, :2], center_sp - M[:, 2])
            rot = math.degrees(math.atan2(M[1, 0], M[0, 0]))
            att: Dict[str, Any] = {
                "x": round(float(loc[0]), 4),
                "y": round(float(loc[1]), 4),
                "rotation": round(float(-rot), 4),
                "width": w_pad,
                "height": h_pad,
            }
            recipe_label = "region"
        elif att_type == "mesh":
            cell = int(sspec.get("cell", 18))
            P_canvas, uv, tris, hull = grid_mesh(im, (bx, by_), cell=cell)
            ws, slot_bone = _compute_mesh_weights(sspec, P_canvas, bone_pos_canvas, chains_info, contact_fields)
            verts: List[Union[int, float]] = []
            for xy, ww in zip(P_canvas, ws):
                verts.append(len(ww))
                sp = to_spine(xy)
                for bn, wt in ww:
                    M = mats[bn]
                    loc = np.linalg.solve(M[:, :2], sp - M[:, 2])
                    verts.extend([bidx[bn], round(float(loc[0]), 4), round(float(loc[1]), 4), round(float(wt), 5)])
            att = {
                "type": "mesh",
                "uvs": [round(float(v), 6) for v in uv.reshape(-1)],
                "triangles": tris,
                "vertices": verts,
                "hull": hull,
                "width": w_pad,
                "height": h_pad,
            }
            recipe_label = str(sspec.get("recipe", "mesh"))
        else:
            raise ValueError(f"Unsupported attachment type {att_type!r} for part {name!r}")

        skel["slots"].append({"name": name, "bone": slot_bone, "attachment": name})
        skel["skins"][0]["attachments"][name] = {name: att}
        att_report.append({
            "part": name, "attachment": att_type, "recipe": recipe_label, "bone": slot_bone,
            "semanticRole": sspec.get("semantic_role") or sspec.get("role") or bone_roles.get(sspec.get("bone")),
            **({"moduleId":sspec.get("module_id"), "pieceId":sspec.get("piece_id")} if module_contract else {}),
        })

    # 5. Build zero-at-start cubic-bezier idle animation (skipped when generating binding-only preview)
    # Estimate body height for proportional translation normalization (1080px reference height)
    if rig_spec.get("body_height_px"):
        body_h_px = float(rig_spec["body_height_px"])
    else:
        ys_min = min((float(p["bbox"][1]) for p in parts), default=0.0)
        ys_max = max((float(p["bbox"][1] + p["bbox"][3]) for p in parts), default=float(H))
        body_h_px = max(64.0, ys_max - ys_min)
    height_scale = max(0.35, min(1.35, body_h_px / 1080.0))

    bone_roles = {b["name"]: str(b.get("role")) for b in rig_spec.get("bones", []) if b.get("role")}
    ik_driven = {bn for c in ik_json for bn in c.get("bones", [])}
    ik_targets = {c.get("target") for c in ik_json if c.get("target")}

    if include_animation:
        parent_world_rot = {
            b.name: (math.degrees(math.atan2(b.parent.c, b.parent.a)) if b.parent is not None else 0.0)
            for b in sk.bones
        }
        if module_contract:
            from .module_rig import resolve_module_motion
            motion = resolve_module_motion(rig_spec, preset, duration=duration)
        else:
            motion = resolve_preset_motion(
                preset, bone_names=[b.name for b in sk.bones], chains=chains_bones_map,
                overrides=rig_spec.get("motion"), duration=duration, bone_roles=bone_roles,
                ik_driven_bones=ik_driven | ik_targets,
            )

        anim_bones: Dict[str, Dict[str, Any]] = {}
        for bname, ch in motion.items():
            tl: Dict[str, Any] = {}
            extra: List[float] = []
            if "twitch" in ch and ch["twitch"] is not None:
                s0 = float(ch["twitch"][0])
                extra = [round(s0 + d, 4) for d in (0.0, 0.25, 0.50, 0.75, 1.05)]
            if "rotate" in ch or "twitch" in ch:
                times, v, d = _keyframes(
                    _fn_from(ch.get("rotate"), duration, gain, 0.0, ch.get("twitch")),
                    duration,
                    key_step,
                    extra,
                )
                tl["rotate"] = bezier_keys([(v, d)], ["value"], times)
            if "tx" in ch or "ty" in ch:
                tx_gain = gain * height_scale
                times, vx, dx = _keyframes(_fn_from(ch.get("tx"), duration, tx_gain), duration, key_step)
                _, vy, dy = _keyframes(_fn_from(ch.get("ty"), duration, tx_gain), duration, key_step)
                pa = math.radians(parent_world_rot[bname])
                c_, s_ = math.cos(-pa), math.sin(-pa)
                lx = [c_ * a - s_ * b for a, b in zip(vx, vy)]
                ly = [s_ * a + c_ * b for a, b in zip(vx, vy)]
                ldx = [c_ * a - s_ * b for a, b in zip(dx, dy)]
                ldy = [s_ * a + c_ * b for a, b in zip(dx, dy)]
                tl["translate"] = bezier_keys([(lx, ldx), (ly, ldy)], ["x", "y"], times)
            if "sx" in ch or "sy" in ch:
                times, vx, dx = _keyframes(_fn_from(ch.get("sx"), duration, gain, 1.0), duration, key_step)
                _, vy, dy = _keyframes(_fn_from(ch.get("sy"), duration, gain, 1.0), duration, key_step)
                tl["scale"] = bezier_keys([(vx, dx), (vy, dy)], ["x", "y"], times)
            if tl:
                anim_bones[bname] = tl

        idle_anim = {"bones": anim_bones}
        check_zero_at_start(idle_anim)
        skel["animations"]["idle"] = idle_anim

    module_preflight = None
    if module_contract and include_animation:
        from .module_rig import preflight_module_motion
        module_preflight = preflight_module_motion(skel, rig_spec, duration=duration)
    skel["skeleton"]["hash"] = hashlib.sha1(json.dumps(skel, sort_keys=True).encode("utf-8")).hexdigest()[:22]
    spine_filename = f"{character}.spine.json"
    spine_path = out / spine_filename
    spine_path.write_text(json.dumps(skel, separators=(",", ":")) + "\n", encoding="utf-8")

    # Scale soft displacement reference ranges proportionally to actual body height
    ref_scale = max(0.5, body_h_px / 1080.0)
    scaled_soft_ranges = {
        k: [round(float(lo) * ref_scale, 2), round(float(hi) * ref_scale, 2)]
        for k, (lo, hi) in preset_obj["soft_displacement_ranges"].items()
    }

    report = {
        "revision": REVISION,
        "generationMode": ("canvas_layers_binding_candidate"
                           if module_contract and module_contract.get('materialMode') == 'canvas_layers'
                           else "complete_modules_v2" if module_contract else "legacy_diagnostic"),
        "spineVersion": SPINE_VERSION,
        "character": character,
        "spineFile": spine_filename,
        "preset": preset,
        "gain": float(gain),
        "includeAnimation": bool(include_animation),
        "bodyHeightPx": round(body_h_px, 1),
        "heightScale": round(height_scale, 4),
        "poseType": rig_spec.get("pose_type", "grounded"),
        "duration": duration,
        "fps": fps,
        "canvas": [W, H],
        "origin": [OX, OY],
        "padPx": PAD,
        "partsDir": str(pdir),
        "partsJsonSHA256": _sha256_file(pj_path),
        "boneCount": len(bones_json),
        "boneRoles": bone_roles,
        "slotCount": len(skel["slots"]),
        "ikCount": len(ik_json),
        "ik": ik_json,
        "attachments": att_report,
        "frame0": "setup; all channel offsets zero at 0 and duration",
        "softReferenceRanges": scaled_soft_ranges,
    }
    if module_contract:
        report["moduleContract"] = module_contract
        report["moduleMotionPreflight"] = module_preflight
        source = pj.get("source")
        if isinstance(source, str) and source.strip():
            src = Path(source)
            if src.is_absolute():
                candidates = [src]
            else:
                candidates = [(pdir / src).resolve(), src.resolve()]
            src_path = next((c for c in candidates if c.is_file()), None)
            if src_path is None:
                raise FileNotFoundError("V2 叠加图原图缺失: %s" % source)
            from .module_rig import render_rig_mask_overlay
            overlay_rel = "review/rig_mask_overlay.png"
            render_rig_mask_overlay(src_path, rig_spec, out / overlay_rel)
            report["rigMaskOverlay"] = overlay_rel
    report_path = out / "rig-build-report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return {
        "spine_path": str(spine_path),
        "report_path": str(report_path),
        "report": report,
    }
