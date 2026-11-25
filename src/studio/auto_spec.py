"""Topology-, gradient-watershed-, and vision-keypoint-driven semantic splitter & Spine 4.2 rig generator.

Capabilities:
1. Vision-keypoint (`semantic_hints`) driven segmentation + rigging:
   - Accepts anatomical joints (`head_top`, `neck`, `chest`, `hips`, `shoulder_L/R`, `elbow_L/R`,
     `wrist_L/R`, `weapon_L/R_tip`, `hip_L/R`, `knee_L/R`, `ankle_L/R`), optional `accessories`
     (e.g., mechanical pets, giant wings, floating orbs), and `pose_type` (`airborne` / `grounded`).
   - Seeds slender capsules along true skeleton segments and grows part boundaries via
     color-gradient-weighted geodesic watershed so seams snap naturally to clothing/armor edges.
2. Upgraded offline topology fallback (when no `semantic_hints` provided):
   - Replaces hard rectangular torso boxes with spine-axis core seeds + color-gradient watershed.
   - Detects grounded vs airborne poses, separates dual legs with `dual_limb` weighting,
     and guarantees 2-bone IK extension margin (0.0px drift).
3. Automatic full-seam `CONTACT_DIST` (`contact_overrides`) across adjacent torso/arm/leg/cloth
   boundaries so 1.6x stress deformation never tears interior holes.
4. Interactive rig editing (`apply_rig_edits`) for add/delete bones, joint drag, and rebinding.
"""
from __future__ import annotations

import copy
import hashlib
import heapq
import math
import re
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw


BONE_ROLE_COLORS: Dict[str, str] = {
    "root": "#94a3b8",
    "body": "#22c55e",
    "head": "#38bdf8",
    "arm": "#f59e0b",
    "hand": "#fb923c",
    "weapon": "#ef4444",
    "leg": "#a855f7",
    "soft": "#14b8a6",
}


def sanitize_character_name(filename: str) -> str:
    stem = Path(filename).stem
    clean = re.sub(r"[^a-zA-Z0-9_-]+", "_", stem).strip("_").lower()
    return clean[:32] or "portrait"


def infer_bone_role(bone_name: str, explicit_role: Optional[str] = None) -> str:
    if explicit_role in BONE_ROLE_COLORS:
        return str(explicit_role)
    low = bone_name.lower()
    if low == "root":
        return "root"
    if any(k in low for k in ("weapon", "blade", "sword", "dagger", "gun", "staff", "spear", "bow")):
        return "weapon"
    if any(k in low for k in ("head", "neck", "bang", "ear")):
        return "head"
    if any(k in low for k in ("hand", "wrist")):
        return "hand"
    if any(k in low for k in ("shoulder", "elbow", "arm")):
        return "arm"
    if any(k in low for k in ("thigh", "knee", "shin", "calf", "ankle", "foot", "toe", "leg")):
        return "leg"
    if any(k in low for k in ("hair", "cloth", "ribbon", "cape", "sash", "skirt", "tail", "tassel", "wing", "pet", "acc")):
        return "soft"
    return "body"


def _compute_thickness(fg: np.ndarray) -> np.ndarray:
    cur = fg.copy()
    thick = np.zeros(fg.shape, dtype=np.int32)
    step = 0
    while cur.any():
        step += 1
        thick[cur] = step
        nxt = cur.copy()
        nxt[1:, :] &= cur[:-1, :]
        nxt[:-1, :] &= cur[1:, :]
        nxt[:, 1:] &= cur[:, :-1]
        nxt[:, :-1] &= cur[:, 1:]
        nxt[0, :] = False
        nxt[-1, :] = False
        nxt[:, 0] = False
        nxt[:, -1] = False
        cur = nxt
    return thick


def _dilate(mask: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return mask.copy()
    a = mask.copy()
    for _ in range(r):
        b = a.copy()
        b[1:, :] |= a[:-1, :]
        b[:-1, :] |= a[1:, :]
        b[:, 1:] |= a[:, :-1]
        b[:, :-1] |= a[:, 1:]
        a = b
    return a


def _cc_components(mask: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
    h, w = mask.shape
    lab = np.zeros((h, w), dtype=np.int32)
    comps: List[np.ndarray] = []
    ys, xs = np.nonzero(mask)
    cid = 0
    for y, x in zip(ys.tolist(), xs.tolist()):
        if lab[y, x] != 0:
            continue
        cid += 1
        q = deque([(y, x)])
        lab[y, x] = cid
        pts: List[Tuple[int, int]] = []
        while q:
            cy, cx = q.popleft()
            pts.append((cy, cx))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and lab[ny, nx] == 0:
                    lab[ny, nx] = cid
                    q.append((ny, nx))
        comps.append(np.asarray(pts, dtype=np.int32))
    return lab, comps


def _color_gradient(small_rgba: np.ndarray) -> np.ndarray:
    """Normalized Sobel-like RGB gradient magnitude in [0, 1]."""
    rgb = small_rgba[..., :3].astype(np.float32) / 255.0
    p = np.pad(rgb, ((1, 1), (1, 1), (0, 0)), mode="edge")
    gx = p[1:-1, 2:] - p[1:-1, :-2]
    gy = p[2:, 1:-1] - p[:-2, 1:-1]
    g = np.sqrt((gx * gx + gy * gy).sum(axis=-1))
    gmax = float(np.percentile(g, 98)) if g.size else 1.0
    return np.clip(g / max(gmax, 1e-4), 0.0, 1.5).astype(np.float32)


def _gradient_watershed(small_rgba: np.ndarray, fg: np.ndarray, seed_labels: np.ndarray, grad_weight: float = 3.8,
                        allowed_masks: Optional[Dict[int, np.ndarray]] = None) -> np.ndarray:
    """Grow labeled seeds across `fg` via color-gradient-weighted Dijkstra so seams snap to real edges."""
    h, w = fg.shape
    grad = _color_gradient(small_rgba)
    out = np.full((h, w), -1, dtype=np.int16)
    dist = np.full((h, w), np.inf, dtype=np.float32)

    heap: List[Tuple[float, int, int, int]] = []
    ys, xs = np.nonzero((seed_labels >= 0) & fg)
    for y, x in zip(ys.tolist(), xs.tolist()):
        lbl = int(seed_labels[y, x])
        out[y, x] = lbl
        dist[y, x] = 0.0
        # Only push seed boundary pixels into priority queue
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and fg[ny, nx] and seed_labels[ny, nx] < 0:
                heap.append((0.0, y, x, lbl))
                break

    heapq.heapify(heap)
    while heap:
        cd, cy, cx, lbl = heapq.heappop(heap)
        if cd > dist[cy, cx] + 1e-6:
            continue
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w and fg[ny, nx]:
                if allowed_masks is not None and not allowed_masks[lbl][ny, nx]:
                    continue
                step_cost = 1.0 + grad_weight * float(grad[ny, nx])
                nd = cd + step_cost
                if nd < dist[ny, nx]:
                    dist[ny, nx] = nd
                    out[ny, nx] = lbl
                    heapq.heappush(heap, (nd, ny, nx, lbl))

    # Any disconnected foreground islands get Euclidean-nearest label
    miss_y, miss_x = np.nonzero(fg & (out < 0))
    if miss_y.size > 0:
        valid_labels = sorted(set(int(v) for v in np.unique(out) if v >= 0))
        if valid_labels:
            centers = []
            for lbl in valid_labels:
                ly, lx = np.nonzero(out == lbl)
                centers.append((lbl, float(ly.mean()), float(lx.mean())))
            for my, mx in zip(miss_y.tolist(), miss_x.tolist()):
                eligible = [c for c in centers if allowed_masks is None or allowed_masks[c[0]][my, mx]]
                if not eligible:
                    continue
                best = min(eligible, key=lambda c: (c[1] - my) ** 2 + (c[2] - mx) ** 2)[0]
                out[my, mx] = best
    return out


def _draw_capsule_seed(h: int, w: int, pts: Sequence[Tuple[float, float]], radius: int) -> np.ndarray:
    im = Image.new("L", (w, h), 0)
    dr = ImageDraw.Draw(im)
    r = max(1, int(radius))
    if len(pts) == 1:
        x, y = pts[0]
        dr.ellipse((x - r, y - r, x + r, y + r), fill=255)
    elif len(pts) >= 2:
        dr.line([tuple(map(float, p)) for p in pts], fill=255, width=max(2, r * 2))
        for x, y in pts:
            dr.ellipse((x - r, y - r, x + r, y + r), fill=255)
    return np.asarray(im, dtype=np.uint8) > 0


def _detect_airborne_pose(legs_mask: np.ndarray, x0: int, x1: int, y0: int, y1: int, torso_cx: float, preset: str) -> bool:
    if preset == "bust_half" or not legs_mask.any():
        return False
    bw = max(1.0, float(x1 - x0 + 1))
    bh = max(1.0, float(y1 - y0 + 1))
    lys, lxs = np.nonzero(legs_mask)
    ly1 = int(lys.max())
    bot_y0 = max(y0, int(round(ly1 - 0.12 * bh)))
    bot_slice = legs_mask[bot_y0 : ly1 + 1, :]
    bys, bxs = np.nonzero(bot_slice)
    if bxs.size == 0:
        return False
    span = float(bxs.max() - bxs.min() + 1)
    _, bot_comps = _cc_components(legs_mask[max(y0, int(round(ly1 - 0.16 * bh))) : ly1 + 1, :])
    major_feet = [c for c in bot_comps if len(c) >= max(8, int(0.01 * legs_mask.sum()))]
    bot_cx = float(bxs.mean())
    off_center = abs(bot_cx - torso_cx) / bw
    if len(major_feet) <= 1 and (span / bw < 0.28 or off_center > 0.12):
        return True
    if span / bw < 0.18:
        return True
    return False


def _save_mask_png(mask_small: np.ndarray, out_path: Path, target_size: Tuple[int, int]) -> str:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.fromarray((mask_small.astype(np.uint8)) * 255)
    if im.size != target_size:
        im = im.resize(target_size, Image.Resampling.NEAREST)
    im.save(out_path)
    return str(out_path.resolve())


def _mask_to_bbox_poly(mask_small: np.ndarray, inv_scale_x: float, inv_scale_y: float) -> List[List[float]]:
    ys, xs = np.nonzero(mask_small)
    if ys.size == 0:
        return [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    x0, x1 = float(xs.min()) * inv_scale_x, float(xs.max() + 1) * inv_scale_x
    y0, y1 = float(ys.min()) * inv_scale_y, float(ys.max() + 1) * inv_scale_y
    return [[round(x0, 1), round(y0, 1)], [round(x1, 1), round(y0, 1)], [round(x1, 1), round(y1, 1)], [round(x0, 1), round(y1, 1)]]


def _mask_to_polygons(mask_small: np.ndarray, inv_sx: float, inv_sy: float) -> List[List[List[float]]]:
    h, w = mask_small.shape
    polys: List[List[List[float]]] = []
    prev_spans: Dict[Tuple[int, int], int] = {}
    for y in range(h):
        row = mask_small[y]
        xs = np.nonzero(row)[0]
        cur_spans: List[Tuple[int, int]] = []
        if xs.size > 0:
            s = int(xs[0])
            p = s
            for v in xs[1:].tolist():
                if v == p + 1:
                    p = v
                else:
                    cur_spans.append((s, p + 1))
                    s = v
                    p = v
            cur_spans.append((s, p + 1))
        cur_set = set(cur_spans)
        for span, y_start in list(prev_spans.items()):
            if span not in cur_set:
                x0, x1 = span
                polys.append(
                    [
                        [round(x0 * inv_sx, 1), round(y_start * inv_sy, 1)],
                        [round(x1 * inv_sx, 1), round(y_start * inv_sy, 1)],
                        [round(x1 * inv_sx, 1), round(y * inv_sy, 1)],
                        [round(x0 * inv_sx, 1), round(y * inv_sy, 1)],
                    ]
                )
                del prev_spans[span]
        for span in cur_spans:
            if span not in prev_spans:
                prev_spans[span] = y
    for (x0, x1), y_start in prev_spans.items():
        polys.append(
            [
                [round(x0 * inv_sx, 1), round(y_start * inv_sy, 1)],
                [round(x1 * inv_sx, 1), round(y_start * inv_sy, 1)],
                [round(x1 * inv_sx, 1), round(h * inv_sy, 1)],
                [round(x0 * inv_sx, 1), round(h * inv_sy, 1)],
            ]
        )
    return polys or [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]]


def _find_adjacent_parts(part_masks: Dict[str, np.ndarray]) -> Dict[str, set]:
    adj: Dict[str, set] = {nm: set() for nm in part_masks}
    dilated = {nm: _dilate(m, 2) for nm, m in part_masks.items()}
    names = list(part_masks.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ni, nj = names[i], names[j]
            if (dilated[ni] & part_masks[nj]).any():
                adj[ni].add(nj)
                adj[nj].add(ni)
    return adj


def _ensure_ik_margin(
    th_p: List[float],
    sh_p: List[float],
    ft_p: List[float],
    side: str,
    body_h_px: float,
) -> List[float]:
    """Ensure 2-bone leg IK has enough knee-bend extension margin so max hip rise never causes >0px drift."""
    # Max hips translation at 1.6x stress is ~35px scaled by body_h / 1080
    height_scale = max(0.35, min(1.35, body_h_px / 1080.0))
    req_margin = max(20.0, 44.0 * height_scale)
    dx = ft_p[0] - th_p[0]
    dy = ft_p[1] - th_p[1]
    chord = max(1.0, math.hypot(dx, dy))
    l1 = math.hypot(sh_p[0] - th_p[0], sh_p[1] - th_p[1])
    l2 = math.hypot(ft_p[0] - sh_p[0], ft_p[1] - sh_p[1])
    if (l1 + l2 - chord) >= req_margin:
        return sh_p
    # Push knee slightly perpendicular to hip->ankle chord so L1 + L2 - chord >= req_margin
    ux, uy = dx / chord, dy / chord
    # Normal vector pointing outward/forward
    nx, ny = (-uy, ux) if side == "L" else (uy, -ux)
    mid_x = (th_p[0] + ft_p[0]) * 0.5
    mid_y = (th_p[1] + ft_p[1]) * 0.5
    # Solve 2 * sqrt((chord/2)^2 + h^2) = chord + req_margin  =>  h = sqrt(((chord + m)/2)^2 - (chord/2)^2)
    half_ext = (chord + req_margin) * 0.5
    half_c = chord * 0.5
    h_off = math.sqrt(max(1.0, half_ext * half_ext - half_c * half_c))
    # Keep bend visually subtle (capped at 22% of chord)
    h_off = min(h_off, chord * 0.22)
    return [round(mid_x + nx * h_off, 2), round(mid_y + ny * h_off, 2)]


def _segment_from_semantic_hints(
    small: np.ndarray,
    fg: np.ndarray,
    W: int,
    H: int,
    sw: int,
    sh: int,
    hints: Dict[str, Any],
    *,
    preset: str,
    mask_dir: Optional[Path],
) -> Dict[str, Any]:
    """Skeleton-guided color-gradient watershed segmentation driven by explicit vision keypoints."""
    full_owned_masks: Dict[str, np.ndarray] = {}
    body_fg = fg
    subject_path = hints.get("subject_mask") if hints.get("vision_contract") else None
    full_subject = None
    def read_exact_mask(path):
        with Image.open(path) as image:
            if image.size != (W, H):
                raise ValueError("视觉归属掩码尺寸不匹配，不能自动缩放。")
            return np.asarray(image.convert("L"), dtype=np.uint8) > 127
    def small_mask(mask):
        return np.asarray(Image.fromarray(mask.astype(np.uint8)*255).resize((sw,sh), Image.Resampling.NEAREST)) > 127
    if subject_path:
        full_subject = read_exact_mask(subject_path)
        body_fg = small_mask(full_subject) & fg
        full_owned_masks = {name: read_exact_mask(path) for name,path in hints.get("ownership_masks", {}).items()}
    inv_sx = float(W) / float(sw)
    inv_sy = float(H) / float(sh)
    sx = float(sw) / float(W)
    sy = float(sh) / float(H)

    joints_norm = hints.get("joints") or {}
    coord_mode = str(hints.get("coord_mode") or "normalized")  # "normalized" (0..1) or "canvas" (px)

    def to_small(pt: Sequence[float]) -> Tuple[float, float]:
        if coord_mode == "canvas":
            return float(pt[0]) * sx, float(pt[1]) * sy
        return float(pt[0]) * float(sw), float(pt[1]) * float(sh)

    def to_canvas(pt_s: Tuple[float, float]) -> List[float]:
        return [round(float(pt_s[0]) * inv_sx, 2), round(float(pt_s[1]) * inv_sy, 2)]

    j: Dict[str, Tuple[float, float]] = {k: to_small(v) for k, v in joints_norm.items() if isinstance(v, (list, tuple)) and len(v) == 2}

    nz_y, nz_x = np.nonzero(body_fg)
    if not nz_y.size:
        raise ValueError("人物主体在分析尺寸下没有可识别前景，需要复核视觉掩码。")
    x0, x1 = int(nz_x.min()), int(nz_x.max())
    y0, y1 = int(nz_y.min()), int(nz_y.max())
    bw = max(8.0, float(x1 - x0 + 1))
    bh = max(8.0, float(y1 - y0 + 1))
    seed_r = max(3, int(round(min(bw, bh) * 0.032)))

    # Build ordered part seeds (drawOrder: accessories/cloth -> legs -> torso -> arms -> weapons -> head)
    ordered_specs: List[Tuple[str, np.ndarray, str, bool]] = []
    acc_metas: List[Dict[str, Any]] = []

    for acc in hints.get("accessories") or []:
        aname = str(acc.get("name") or "").strip()
        pts_raw = acc.get("pts") or []
        if not aname or not pts_raw:
            continue
        pts_s = [to_small(p) for p in pts_raw]
        m = (small_mask(full_owned_masks[aname]) if aname in full_owned_masks else _draw_capsule_seed(sh, sw, pts_s, max(2, int(seed_r * 0.9)))) & fg
        if m.any() or aname in full_owned_masks:
            role = str(acc.get("role") or "soft")
            is_rigid = bool(acc.get("rigid", False))
            ordered_specs.append((aname, m, role, is_rigid))
            acc_metas.append(
                {
                    "name": aname,
                    "parent_bone": str(acc.get("parent_bone") or "hips"),
                    "pts_s": pts_s,
                    "rigid": is_rigid,
                    "static": bool(acc.get("static", False)),
                }
            )

    # Legs seed
    leg_chains: Dict[str, Dict[str, Tuple[float, float]]] = {}
    legs_seed = np.zeros_like(fg)
    for side in ("L", "R"):
        hk, kk, ak = f"hip_{side}", f"knee_{side}", f"ankle_{side}"
        if kk in j and ak in j:
            hp = j.get(hk, j.get("hips", (sw * 0.5, sh * 0.6)))
            kp = j[kk]
            ap = j[ak]
            tp = j.get(f"toe_{side}", (ap[0], min(float(sh - 1), ap[1] + bh * 0.04)))
            leg_chains[side] = {"thigh": hp, "shin": kp, "foot": ap, "toe": tp}
            legs_seed |= _draw_capsule_seed(sh, sw, [hp, kp, ap, tp], max(3, int(seed_r * 1.15))) & body_fg
    if legs_seed.any() and preset != "bust_half":
        ordered_specs.append(("legs", legs_seed, "leg", False))

    # Torso seed
    neck_s = j.get("neck", (sw * 0.5, y0 + bh * 0.25))
    chest_s = j.get("chest", (neck_s[0], neck_s[1] + bh * 0.12))
    hips_s = j.get("hips", (chest_s[0], chest_s[1] + bh * 0.22))
    spine_s = ((chest_s[0] + hips_s[0]) * 0.5, (chest_s[1] + hips_s[1]) * 0.5)
    torso_pts = [neck_s, chest_s, spine_s, hips_s]
    torso_seed = _draw_capsule_seed(sh, sw, torso_pts, max(4, int(seed_r * 1.5))) & body_fg
    ordered_specs.append(("torso", torso_seed, "body", False))

    # Arms & rigid weapons
    arm_metas: Dict[str, Dict[str, Any]] = {}
    weapon_metas: Dict[str, Dict[str, Any]] = {}
    for side in ("R", "L"):
        sk_k, ek_k, wk_k, wtp_k = f"shoulder_{side}", f"elbow_{side}", f"wrist_{side}", f"weapon_{side}_tip"
        if sk_k in j and wk_k in j:
            sp_pt = j[sk_k]
            wp_pt = j[wk_k]
            ep_pt = j.get(ek_k, ((sp_pt[0] + wp_pt[0]) * 0.5, (sp_pt[1] + wp_pt[1]) * 0.5))
            am = _draw_capsule_seed(sh, sw, [sp_pt, ep_pt, wp_pt], max(3, int(seed_r * 0.95))) & body_fg
            if am.any():
                ordered_specs.append((f"arm_{side}", am, "arm", False))
                arm_metas[side] = {"side": side, "shoulder": sp_pt, "elbow": ep_pt, "hand": wp_pt}
            if wtp_k in j:
                wt_pt = j[wtp_k]
                wm = _draw_capsule_seed(sh, sw, [wp_pt, wt_pt], max(2, int(seed_r * 0.85))) & body_fg
                if wm.any():
                    ordered_specs.append((f"weapon_{side}", wm, "weapon", True))
                    weapon_metas[side] = {"side": side, "pivot": wp_pt, "tip": wt_pt}

    # Head seed
    head_top_s = j.get("head_top", (neck_s[0], max(float(y0), neck_s[1] - bh * 0.16)))
    head_mid = ((head_top_s[0] + neck_s[0]) * 0.5, (head_top_s[1] + neck_s[1]) * 0.5)
    head_seed = _draw_capsule_seed(sh, sw, [head_top_s, head_mid], max(4, int(seed_r * 1.25))) & body_fg
    ordered_specs.append(("head", head_seed, "head", False))

    seed_lab = np.full((sh, sw), -1, dtype=np.int16)
    for idx, (_nm, m, _role, _rig) in enumerate(ordered_specs):
        seed_lab[m] = idx

    allowed_masks = None
    if full_subject is not None:
        allowed_masks = {idx: small_mask(full_owned_masks[nm]) if nm in full_owned_masks else body_fg
                         for idx,(nm,_m,_role,_rig) in enumerate(ordered_specs)}
    final_lab = _gradient_watershed(small, fg, seed_lab, grad_weight=3.8, allowed_masks=allowed_masks)
    part_masks: Dict[str, np.ndarray] = {}
    for idx, (nm, _m, _role, _rig) in enumerate(ordered_specs):
        pm = final_lab == idx
        if pm.any() or nm in full_owned_masks:
            part_masks[nm] = pm

    pose_type = str(hints.get("pose_type") or "")
    if pose_type not in ("airborne", "grounded"):
        airborne = _detect_airborne_pose(part_masks.get("legs", np.zeros_like(fg)), x0, x1, y0, y1, hips_s[0], preset)
        pose_type = "airborne" if airborne else "grounded"
    airborne = pose_type == "airborne"

    body_h_px = round(bh * inv_sy, 1)
    fill_depth = max(96, min(200, int(round(max(bw * inv_sx, bh * inv_sy) * 0.14))))
    mesh_cell = max(18, min(44, int(round(min(W, H) / 14.0))))

    parts_out: List[Dict[str, Any]] = []
    chains: Dict[str, Any] = {}
    for nm, _m, role, is_rigid in ordered_specs:
        if nm not in part_masks:
            continue
        pm = part_masks[nm]
        if nm == "head":
            pivot, parent_part, kind = to_canvas(neck_s), "torso", "rigid"
        elif nm == "torso":
            pivot, parent_part, kind = to_canvas(chest_s), ("legs" if "legs" in part_masks else None), "soft"
        elif nm == "legs":
            pivot, parent_part, kind = to_canvas(hips_s), None, "soft"
        elif nm.startswith("arm_"):
            s = nm[-1]
            pivot, parent_part, kind = to_canvas(arm_metas[s]["shoulder"]), "torso", "soft"
        elif nm.startswith("weapon_"):
            s = nm[-1]
            pivot, parent_part, kind = to_canvas(weapon_metas[s]["pivot"]), f"arm_{s}", "rigid"
        else:
            am_info = next((a for a in acc_metas if a["name"] == nm), None)
            pts_c = [to_canvas(p) for p in am_info["pts_s"]] if am_info else [to_canvas(hips_s)]
            pivot = pts_c[0]
            parent_part = "torso"
            kind = "static" if (am_info and am_info["static"]) else ("rigid" if is_rigid else "soft")
            if kind == "soft" and len(pts_c) >= 2:
                if len(pts_c) == 2:
                    mid_c = [round((pts_c[0][0] + pts_c[1][0]) * 0.5, 2), round((pts_c[0][1] + pts_c[1][1]) * 0.5, 2)]
                    pts_c = [pts_c[0], mid_c, pts_c[1]]
                chains[nm] = {"parent": am_info["parent_bone"], "part": nm, "count": 2}
        p_entry: Dict[str, Any] = {
            "name": nm,
            "kind": kind,
            "role": role,
            "rigid": bool(is_rigid),
            "parent": parent_part,
            "pivot": pivot,
            "own": [_mask_to_bbox_poly(full_owned_masks[nm], 1.0, 1.0) if nm in full_owned_masks else _mask_to_bbox_poly(pm, inv_sx, inv_sy)],
            "band": 8,
            "depth": max(36, fill_depth // 2) if is_rigid else fill_depth,
            "guides": [(None, None, None, "smooth")],
        }
        if nm in chains:
            am_info = next(a for a in acc_metas if a["name"] == nm)
            pts_c = [to_canvas(p) for p in am_info["pts_s"]]
            if len(pts_c) == 2:
                pts_c = [pts_c[0], [round((pts_c[0][0] + pts_c[1][0]) * 0.5, 2), round((pts_c[0][1] + pts_c[1][1]) * 0.5, 2)], pts_c[1]]
            p_entry["centerline"] = pts_c
        if nm in full_owned_masks:
            p_entry["mask"] = hints["ownership_masks"][nm]
            p_entry["allowed_mask"] = hints["ownership_masks"][nm]
            p_entry["ownership_locked"] = True
        elif mask_dir is not None:
            up = np.asarray(Image.fromarray(pm.astype(np.uint8)*255).resize((W,H), Image.Resampling.NEAREST)) > 127
            if full_subject is not None:
                up &= full_subject
            p_entry["mask"] = _save_mask_png(up, mask_dir / f"{nm}_mask.png", (W, H))
        else:
            p_entry["own"] = _mask_to_polygons(pm, inv_sx, inv_sy)
        if full_subject is not None and nm not in full_owned_masks:
            p_entry["allowed_mask"] = subject_path
        parts_out.append(p_entry)

    # Assemble bones
    root_pos = [round(hips_s[0] * inv_sx, 2), float(H)]
    hips_pos = to_canvas(hips_s)
    spine_pos = to_canvas(spine_s)
    chest_pos = to_canvas(chest_s)
    neck_pos = to_canvas(neck_s)
    head_top_pos = to_canvas(head_top_s)

    bones: List[Dict[str, Any]] = [
        {"name": "root", "parent": None, "pos": root_pos, "role": "root", "label": "根节点"},
        {"name": "hips", "parent": "root", "pos": hips_pos, "aim": spine_pos, "role": "body", "label": "骨盆/重心"},
        {"name": "spine", "parent": "hips", "pos": spine_pos, "aim": chest_pos, "role": "body", "label": "脊柱"},
        {"name": "chest", "parent": "spine", "pos": chest_pos, "aim": neck_pos, "role": "body", "label": "胸腔"},
        {"name": "ribcage", "parent": "chest", "pos": chest_pos, "role": "body", "label": "呼吸胸廓"},
        {"name": "neck", "parent": "chest", "pos": neck_pos, "aim": head_top_pos, "role": "head", "label": "颈部"},
        {"name": "head", "parent": "neck", "pos": neck_pos, "aim": head_top_pos, "role": "head", "label": "头部"},
    ]

    for side in ("R", "L"):
        if side not in arm_metas:
            continue
        am = arm_metas[side]
        sh_p, el_p, hd_p = to_canvas(am["shoulder"]), to_canvas(am["elbow"]), to_canvas(am["hand"])
        lbl = "右" if side == "R" else "左"
        bones.extend(
            [
                {"name": f"shoulder_{side}", "parent": "chest", "pos": sh_p, "aim": el_p, "role": "arm", "label": f"{lbl}肩/大臂"},
                {"name": f"elbow_{side}", "parent": f"shoulder_{side}", "pos": el_p, "aim": hd_p, "role": "arm", "label": f"{lbl}肘/小臂"},
                {
                    "name": f"hand_{side}",
                    "parent": f"elbow_{side}",
                    "pos": hd_p,
                    "aim": to_canvas(weapon_metas[side]["tip"]) if side in weapon_metas else hd_p,
                    "role": "hand",
                    "label": f"{lbl}手",
                },
            ]
        )
        if side in weapon_metas:
            wm = weapon_metas[side]
            bones.append(
                {
                    "name": f"weapon_{side}",
                    "parent": f"hand_{side}",
                    "pos": to_canvas(wm["pivot"]),
                    "aim": to_canvas(wm["tip"]),
                    "role": "weapon",
                    "label": f"{lbl}手武器（刚性）",
                }
            )

    ik_list: List[Dict[str, Any]] = []
    for side in ("L", "R"):
        if side not in leg_chains:
            continue
        lc = leg_chains[side]
        th_p, sh_p, ft_p, to_p = to_canvas(lc["thigh"]), to_canvas(lc["shin"]), to_canvas(lc["foot"]), to_canvas(lc["toe"])
        if not airborne and preset != "bust_half" and not hints.get("vision_contract"):
            sh_p = _ensure_ik_margin(th_p, sh_p, ft_p, side, body_h_px)
        lbl = "左" if side == "L" else "右"
        bones.extend(
            [
                {"name": f"thigh_{side}", "parent": "hips", "pos": th_p, "aim": sh_p, "role": "leg", "label": f"{lbl}大腿"},
                {"name": f"shin_{side}", "parent": f"thigh_{side}", "pos": sh_p, "aim": ft_p, "role": "leg", "label": f"{lbl}小腿"},
                {"name": f"foot_{side}", "parent": f"shin_{side}", "pos": ft_p, "aim": to_p, "role": "leg", "label": f"{lbl}脚"},
            ]
        )
        if not airborne and preset != "bust_half":
            tgt_name = f"toe_target_{side}"
            bones.append({"name": tgt_name, "parent": "root", "pos": ft_p, "role": "leg", "label": f"{lbl}脚底锚点"})
            constraint = {"name": f"foot_ik_{side}", "bones": [f"thigh_{side}", f"shin_{side}"], "target": tgt_name}
            if hints.get("vision_contract"):
                constraint.update(stretch=True, uniform=True)
            ik_list.append(constraint)

    # Skin bindings with automatic full-seam CONTACT_DIST overrides
    adj = _find_adjacent_parts(part_masks)
    c_out = float(max(44, int(round(mesh_cell * 1.8))))
    skin: Dict[str, Any] = {}
    for p in parts_out:
        nm = p["name"]
        if nm == "head":
            skin["head"] = {"type": "region", "bone": "head"}
        elif nm == "torso":
            skin["torso"] = {
                "type": "mesh",
                "recipe": "linear_y",
                "bones": ["chest", "spine", "hips"],
                "ranges": [
                    [round(chest_pos[1], 1), round(spine_pos[1], 1)],
                    [round(spine_pos[1], 1), round(hips_pos[1], 1)],
                ],
                "cell": mesh_cell,
            }
        elif nm.startswith("arm_"):
            s = nm[-1]
            skin[nm] = {
                "type": "mesh",
                "recipe": "limb",
                "start_bone": "chest",
                "mid_bone": f"shoulder_{s}",
                "end_bone": f"elbow_{s}",
                "root_blend": [0.0, 0.38],
                "tip_blend": [0.48, 0.88],
                "cell": mesh_cell,
            }
        elif nm.startswith("weapon_"):
            skin[nm] = {"type": "region", "bone": nm}
        elif nm == "legs":
            cts = [{"part": "torso", "bone": "hips", "d_in": 0.0, "d_out": c_out}] if "torso" in adj.get("legs", set()) else []
            if "L" in leg_chains and "R" in leg_chains:
                skin["legs"] = {
                    "type": "mesh",
                    "recipe": "dual_limb",
                    "root_bone": "hips",
                    "branches": [
                        {"mid_bone": "thigh_L", "end_bone": "shin_L"},
                        {"mid_bone": "thigh_R", "end_bone": "shin_R"},
                    ],
                    "root_blend": [0.0, 0.24],
                    "tip_blend": [0.45, 0.85],
                    "contact_overrides": cts,
                    "cell": mesh_cell,
                }
            elif "L" in leg_chains or "R" in leg_chains:
                s = "L" if "L" in leg_chains else "R"
                skin["legs"] = {
                    "type": "mesh",
                    "recipe": "limb",
                    "start_bone": "hips",
                    "mid_bone": f"thigh_{s}",
                    "end_bone": f"shin_{s}",
                    "root_blend": [0.0, 0.22],
                    "tip_blend": [0.45, 0.85],
                    "contact_overrides": cts,
                    "cell": mesh_cell,
                }
            else:
                skin["legs"] = {"type": "region", "bone": "hips"}
        elif nm in chains:
            par_b = chains[nm]["parent"]
            cts = []
            for nb in sorted(adj.get(nm, set())):
                if nb in ("torso", "legs", "head"):
                    cts.append({"part": nb, "bone": par_b if nb != "head" else "head", "d_in": 0.0, "d_out": c_out})
            skin[nm] = {
                "type": "mesh",
                "recipe": "chain",
                "chain": nm,
                "contact_overrides": cts,
                "cell": mesh_cell,
            }
        else:
            am_info = next((a for a in acc_metas if a["name"] == nm), None)
            skin[nm] = {"type": "region", "bone": "root" if (am_info and am_info["static"]) else (am_info["parent_bone"] if am_info else "hips")}

    if full_subject is not None:
        for part in parts_out:
            skin[part["name"]]["semantic_role"] = part["role"]
    return {
        "parts": parts_out,
        "bones": bones,
        "chains": chains,
        "ik": ik_list,
        "skin": skin,
        "pose_type": pose_type,
        "body_height_px": body_h_px,
        "origin": [round(W * 0.5, 2), float(H)],
    }


def _segment_portrait_semantics(
    rgba: np.ndarray,
    *,
    preset: str = "agile_full",
    mask_dir: Optional[Path] = None,
    semantic_hints: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Segment RGBA portrait into semantic anatomical regions and locate joints."""
    H, W = rgba.shape[:2]
    work_max = 260.0
    scale = min(1.0, work_max / float(max(W, H)))
    sw = max(32, int(round(W * scale)))
    sh = max(32, int(round(H * scale)))
    inv_sx = float(W) / float(sw)
    inv_sy = float(H) / float(sh)

    small_im = Image.fromarray(rgba).resize((sw, sh), Image.Resampling.BILINEAR)
    small = np.asarray(small_im, dtype=np.uint8)
    fg = small[..., 3] > 20
    nz_y, nz_x = np.nonzero(fg)
    if nz_y.size == 0:
        fg[:, :] = True
        nz_y, nz_x = np.nonzero(fg)

    # Fast path: explicit vision keypoints (`semantic_hints.joints`)
    if semantic_hints and isinstance(semantic_hints.get("joints"), dict) and len(semantic_hints["joints"]) >= 3:
        return _segment_from_semantic_hints(
            small, fg, W, H, sw, sh, semantic_hints, preset=preset, mask_dir=mask_dir
        )

    x0, x1 = int(nz_x.min()), int(nz_x.max())
    y0, y1 = int(nz_y.min()), int(nz_y.max())
    bw = max(8.0, float(x1 - x0 + 1))
    bh = max(8.0, float(y1 - y0 + 1))
    fg_area = int(fg.sum())

    thick = _compute_thickness(fg)
    d_max = max(2, int(thick.max()))

    # 1. Core torso detection in middle vertical band
    mid_y0 = int(round(y0 + (0.24 if preset == "bust_half" else 0.18) * bh))
    mid_y1 = int(round(y0 + (0.86 if preset == "bust_half" else 0.66) * bh))
    mid_band = np.zeros_like(fg)
    mid_band[max(0, mid_y0) : min(sh, mid_y1 + 1), :] = True

    core_thr = max(2, int(round(0.55 * d_max)))
    core_cand = fg & mid_band & (thick >= core_thr)
    if not core_cand.any():
        core_cand = fg & mid_band
    if not core_cand.any():
        core_cand = fg
    _, core_comps = _cc_components(core_cand)
    core_comps.sort(key=lambda pts: -len(pts))
    core_pts = core_comps[0]
    core_seed = np.zeros_like(fg)
    core_seed[core_pts[:, 0], core_pts[:, 1]] = True

    cys, cxs = np.nonzero(core_seed)
    tcx = float(cxs.mean())
    tcy = float(cys.mean())
    core_y0, core_y1 = int(cys.min()), int(cys.max())
    core_x0, core_x1 = int(cxs.min()), int(cxs.max())
    core_w = max(6.0, float(core_x1 - core_x0 + 1))

    upper_core_y = int(round(core_y0 + 0.25 * max(4, core_y1 - core_y0)))
    lower_core_y = int(round(core_y0 + 0.78 * max(4, core_y1 - core_y0)))
    upper_xs = cxs[cys <= upper_core_y]
    lower_xs = cxs[cys >= lower_core_y]
    chest_cx = float(upper_xs.mean()) if upper_xs.size else tcx
    hips_cx = float(lower_xs.mean()) if lower_xs.size else tcx

    yy_grid, xx_grid = np.mgrid[0:sh, 0:sw]
    if preset == "bust_half":
        y_neck_est = y0 + bh * 0.38
        y_chest_est = max(y_neck_est + 4.0, tcy - bh * 0.02)
        y_spine_est = min(y1 - 4.0, tcy + bh * 0.16)
        y_hips_est = min(y1 - 2.0, y0 + bh * 0.86)
        y_pelvis_cut = min(y1 - 1.0, y0 + bh * 0.90)
    else:
        center_col = fg & (np.abs(xx_grid - chest_cx) <= max(4.0, 0.12 * bw)) & (yy_grid <= core_y0 + 0.12 * bh)
        head_top_true = float(y0)
        if center_col.any():
            _, cc_top = _cc_components(center_col)
            best_c = min(cc_top, key=lambda pts: max(0.0, float(core_y0) - float(pts[:, 0].max())))
            head_top_true = float(best_c[:, 0].min())
        y_neck_est = min(float(core_y0) + 0.14 * bh, head_top_true + 0.21 * bh)
        y_chest_est = max(y_neck_est + 4.0, y_neck_est + 0.10 * bh)
        y_spine_est = (y_chest_est + float(core_y1)) * 0.50
        y_hips_est = min(y1 - bh * 0.22, float(core_y1) - 0.06 * bh)
        y_pelvis_cut = min(y1 - bh * 0.18, max(y_hips_est + 0.02 * bh, y0 + bh * 0.54))

    # 2. Head core seed (centered above neck)
    head_half_w = min(bw * 0.17, max(core_w * 0.38, d_max * 0.68, bw * 0.10))
    head_band = (
        fg
        & (yy_grid >= int(max(0, y_neck_est - 0.26 * bh)))
        & (yy_grid <= int(min(sh - 1, y_neck_est + 3)))
        & (np.abs(xx_grid - chest_cx) <= head_half_w * 1.15)
    )
    if head_band.any():
        _, h_comps = _cc_components(head_band)
        best_h = max(
            h_comps,
            key=lambda pts: float(len(pts)) / (1.0 + abs(float(pts[:, 1].mean()) - chest_cx) / max(1.0, head_half_w)),
        )
        head_cx = float(best_h[:, 1].mean())
        head_y0_actual = float(best_h[:, 0].min())
        y_neck = max(head_y0_actual + 0.16 * bh, y_neck_est)
        head_rx = min(head_half_w, max(6.0, float(best_h[:, 1].max() - best_h[:, 1].min()) * 0.50))
        head_ry = max(8.0, (y_neck - head_y0_actual) * 0.52)
        head_cy = (head_y0_actual + y_neck) * 0.48
        head_mask = (
            fg
            & (((xx_grid - head_cx) / max(4.0, head_rx)) ** 2 + ((yy_grid - head_cy) / max(4.0, head_ry)) ** 2 <= 1.05)
            & (yy_grid <= y_neck + 1)
        )
    else:
        head_cx = chest_cx
        y_neck = y_neck_est
        head_y0_actual = max(float(y0), y_neck - 0.20 * bh)
        head_mask = fg & (yy_grid >= head_y0_actual) & (yy_grid <= y_neck) & (np.abs(xx_grid - head_cx) <= head_half_w)

    if head_mask.any():
        _, hc = _cc_components(head_mask)
        hc.sort(key=lambda pts: -len(pts))
        head_mask = np.zeros_like(fg)
        head_mask[hc[0][:, 0], hc[0][:, 1]] = True
        hys, hxs = np.nonzero(head_mask)
        head_cx = float(hxs.mean())
        head_y0_actual = float(hys.min())

    # 3. Torso soft oval/capsule seed (replaces hard rectangular cut; gradient watershed will snap to clothing edges)
    if (bw / max(1.0, bh)) < 0.25:
        torso_w_half = bw
    else:
        torso_w_half = min(bw * 0.22, max(core_w * 0.45, d_max * 0.75, bw * 0.13))
    t_param = np.clip((yy_grid - y_neck) / max(1.0, y_pelvis_cut - y_neck), 0.0, 1.0)
    spine_x_at_y = chest_cx * (1.0 - t_param) + hips_cx * t_param
    taper = 0.86 + 0.18 * np.sin(t_param * math.pi)
    torso_mask = (
        fg
        & ~head_mask
        & (yy_grid >= y_neck + 1)
        & (yy_grid <= y_pelvis_cut - 1)
        & (np.abs(xx_grid - spine_x_at_y) <= torso_w_half * taper)
    )

    # 4. Separate external limbs / weapons / legs / ribbons outside (head | torso)
    outer = fg & ~(head_mask | torso_mask)
    _, ext_comps = _cc_components(outer)

    core_border = _dilate(torso_mask | head_mask, 2)
    arm_R_mask = np.zeros_like(fg)
    arm_L_mask = np.zeros_like(fg)
    weapon_R_mask = np.zeros_like(fg)
    weapon_L_mask = np.zeros_like(fg)
    legs_mask = np.zeros_like(fg)
    cloth_R_mask = np.zeros_like(fg)
    cloth_L_mask = np.zeros_like(fg)

    arm_R_meta: Optional[Dict[str, Any]] = None
    arm_L_meta: Optional[Dict[str, Any]] = None
    weapon_R_meta: Optional[Dict[str, Any]] = None
    weapon_L_meta: Optional[Dict[str, Any]] = None
    soft_metas: List[Dict[str, Any]] = []

    def _classify_arm_and_weapon(
        pts: np.ndarray,
        attach_pt: Tuple[float, float],
        tip_pt: Tuple[float, float],
        side: str,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], Optional[Dict[str, Any]]]:
        ax, ay = attach_pt
        tx, ty = tip_pt
        vx, vy = tx - ax, ty - ay
        reach = max(1.0, math.hypot(vx, vy))
        ux, uy = vx / reach, vy / reach

        proj = (pts[:, 1] - ax) * ux + (pts[:, 0] - ay) * uy
        s_norm = np.clip(proj / reach, 0.0, 1.0)

        outer_pts = pts[s_norm >= 0.55]
        outer_sat = 0.0
        if len(outer_pts) > 10:
            rgb_o = small[outer_pts[:, 0], outer_pts[:, 1], :3].astype(np.float32)
            mx_c = rgb_o.max(axis=1)
            mn_c = rgb_o.min(axis=1)
            outer_sat = float(((mx_c - mn_c) / np.maximum(mx_c, 1.0)).mean())

        has_weapon = ((bw / max(1.0, bh)) >= 0.30) and ((reach >= 0.28 * bh) or (reach >= 0.22 * bh and outer_sat > 0.38))
        a_mask = np.zeros_like(fg)
        w_mask = np.zeros_like(fg)

        if has_weapon:
            best_s = 0.52
            best_cnt = 1e9
            for k in range(8):
                sv = 0.40 + k * (0.24 / 7.0)
                band = np.abs(s_norm - sv) <= 0.04
                cnt = int(band.sum())
                if cnt > 0 and cnt < best_cnt:
                    best_cnt = cnt
                    best_s = sv
            arm_sel = s_norm <= best_s
            wpn_sel = s_norm > best_s
            if arm_sel.sum() < 12 or wpn_sel.sum() < 12:
                arm_sel = s_norm <= 0.52
                wpn_sel = s_norm > 0.52
            a_pts = pts[arm_sel]
            w_pts = pts[wpn_sel]
            a_mask[a_pts[:, 0], a_pts[:, 1]] = True
            w_mask[w_pts[:, 0], w_pts[:, 1]] = True

            def _slice_center(p_arr: np.ndarray, s_arr: np.ndarray, target_s: float, fallback: Tuple[float, float]) -> Tuple[float, float]:
                sel = np.abs(s_arr - target_s) <= 0.08
                if sel.any():
                    return float(p_arr[sel, 1].mean()), float(p_arr[sel, 0].mean())
                return fallback

            sh_pt = (ax, ay)
            el_pt = _slice_center(pts, s_norm, best_s * 0.50, (ax + vx * best_s * 0.5, ay + vy * best_s * 0.5))
            hd_pt = _slice_center(pts, s_norm, best_s, (ax + vx * best_s, ay + vy * best_s))
            wt_pt = (tx, ty)
            return a_mask, w_mask, {"side": side, "shoulder": sh_pt, "elbow": el_pt, "hand": hd_pt}, {"side": side, "pivot": hd_pt, "tip": wt_pt}

        a_mask[pts[:, 0], pts[:, 1]] = True
        return (
            a_mask,
            w_mask,
            {"side": side, "shoulder": (ax, ay), "elbow": (ax + vx * 0.48, ay + vy * 0.48), "hand": (tx, ty)},
            None,
        )

    ext_comps.sort(key=lambda pts: -len(pts))
    for pts in ext_comps:
        area = len(pts)
        if area < max(10, int(0.0015 * fg_area)):
            continue
        my, mx = float(pts[:, 0].mean()), float(pts[:, 1].mean())
        md = int(thick[pts[:, 0], pts[:, 1]].max())
        on_b = core_border[pts[:, 0], pts[:, 1]]
        if on_b.any():
            ap = pts[on_b]
            ay, ax = float(ap[:, 0].mean()), float(ap[:, 1].mean())
        else:
            d0 = (pts[:, 0] - tcy) ** 2 + (pts[:, 1] - tcx) ** 2
            p_near = pts[int(np.argmin(d0))]
            ay, ax = float(p_near[0]), float(p_near[1])

        tip_idx = int(np.argmax((pts[:, 0] - ay) ** 2 + (pts[:, 1] - ax) ** 2))
        ty, tx = float(pts[tip_idx, 0]), float(pts[tip_idx, 1])
        reach = math.hypot(tx - ax, ty - ay)
        elong = reach / max(1.0, 2.0 * float(md))

        is_below_pelvis = (ay >= y_pelvis_cut - 0.06 * bh) and (my > y_pelvis_cut - 0.02 * bh)
        is_slender_ribbon = (md <= max(3, int(0.20 * d_max))) and (elong >= 3.2) and (area < 0.065 * fg_area)

        if is_below_pelvis and preset != "bust_half":
            if is_slender_ribbon and legs_mask.any():
                if mx < hips_cx:
                    cloth_R_mask[pts[:, 0], pts[:, 1]] = True
                else:
                    cloth_L_mask[pts[:, 0], pts[:, 1]] = True
            else:
                legs_mask[pts[:, 0], pts[:, 1]] = True
        elif is_slender_ribbon:
            if mx < tcx:
                cloth_R_mask[pts[:, 0], pts[:, 1]] = True
            else:
                cloth_L_mask[pts[:, 0], pts[:, 1]] = True
        else:
            if mx < tcx:
                if arm_R_meta is None:
                    am, wm, ainfo, winfo = _classify_arm_and_weapon(pts, (ax, ay), (tx, ty), "R")
                    arm_R_mask |= am
                    weapon_R_mask |= wm
                    arm_R_meta = ainfo
                    weapon_R_meta = winfo
                else:
                    if md <= max(4, int(0.28 * d_max)):
                        cloth_R_mask[pts[:, 0], pts[:, 1]] = True
                    else:
                        arm_R_mask[pts[:, 0], pts[:, 1]] = True
            else:
                if arm_L_meta is None:
                    am, wm, ainfo, winfo = _classify_arm_and_weapon(pts, (ax, ay), (tx, ty), "L")
                    arm_L_mask |= am
                    weapon_L_mask |= wm
                    arm_L_meta = ainfo
                    weapon_L_meta = winfo
                else:
                    if md <= max(4, int(0.28 * d_max)):
                        cloth_L_mask[pts[:, 0], pts[:, 1]] = True
                    else:
                        arm_L_mask[pts[:, 0], pts[:, 1]] = True

    # Prevent arms/weapons from swallowing lower legs below pelvis cut
    if preset != "bust_half":
        leg_floor = yy_grid >= int(round(y_pelvis_cut + 0.08 * bh))
        for m_arm in (arm_R_mask, arm_L_mask, weapon_R_mask, weapon_L_mask):
            leak = m_arm & leg_floor
            if leak.any():
                legs_mask |= leak
                m_arm &= ~leak

    if preset != "bust_half" and not legs_mask.any():
        legs_mask = fg & (yy_grid >= y_pelvis_cut)
        torso_mask &= ~legs_mask

    # Fallback arms for normal-aspect standing silhouettes where arms hug the torso
    if (bw / max(1.0, bh)) >= 0.30:
        if arm_R_meta is None and torso_mask.any():
            arm_band = torso_mask & (yy_grid >= y_neck) & (yy_grid <= y_hips_est)
            left_wing = arm_band & (xx_grid < spine_x_at_y - torso_w_half * 0.58)
            if int(left_wing.sum()) >= max(16, int(0.015 * fg_area)):
                arm_R_mask |= left_wing
                torso_mask &= ~left_wing
                ays, axs = np.nonzero(arm_R_mask)
                arm_R_meta = {
                    "side": "R",
                    "shoulder": (float(axs.max()), float(ays.min())),
                    "elbow": (float(axs.mean()), float(ays.mean())),
                    "hand": (float(axs.min()), float(ays.max())),
                }
        if arm_L_meta is None and torso_mask.any():
            arm_band = torso_mask & (yy_grid >= y_neck) & (yy_grid <= y_hips_est)
            right_wing = arm_band & (xx_grid > spine_x_at_y + torso_w_half * 0.58)
            if int(right_wing.sum()) >= max(16, int(0.015 * fg_area)):
                arm_L_mask |= right_wing
                torso_mask &= ~right_wing
                ays, axs = np.nonzero(arm_L_mask)
                arm_L_meta = {
                    "side": "L",
                    "shoulder": (float(axs.min()), float(ays.min())),
                    "elbow": (float(axs.mean()), float(ays.mean())),
                    "hand": (float(axs.max()), float(ays.max())),
                }

    # 5. Color-gradient-weighted watershed consolidation (snaps seams to real clothing/armor contours!)
    ordered_candidates: List[Tuple[str, np.ndarray, str, bool]] = []
    if cloth_R_mask.any():
        ordered_candidates.append(("cloth_R", cloth_R_mask, "soft", False))
    if cloth_L_mask.any():
        ordered_candidates.append(("cloth_L", cloth_L_mask, "soft", False))
    if legs_mask.any():
        ordered_candidates.append(("legs", legs_mask, "leg", False))
    ordered_candidates.append(("torso", torso_mask, "body", False))
    if arm_R_mask.any():
        ordered_candidates.append(("arm_R", arm_R_mask, "arm", False))
    if weapon_R_mask.any():
        ordered_candidates.append(("weapon_R", weapon_R_mask, "weapon", True))
    if arm_L_mask.any():
        ordered_candidates.append(("arm_L", arm_L_mask, "arm", False))
    if weapon_L_mask.any():
        ordered_candidates.append(("weapon_L", weapon_L_mask, "weapon", True))
    if head_mask.any():
        ordered_candidates.append(("head", head_mask, "head", False))

    # Erode seam boundaries by 2px before gradient watershed so boundaries snap naturally onto real color edges
    seed_lab = np.full((sh, sw), -1, dtype=np.int16)
    for idx, (_nm, m, _role, _rigid) in enumerate(ordered_candidates):
        seed_lab[m] = idx
    # Clear 2px seam band between torso and neighbors so gradient watershed snaps to natural edges
    torso_idx = next((i for i, c in enumerate(ordered_candidates) if c[0] == "torso"), -1)
    if torso_idx >= 0:
        t_m = seed_lab == torso_idx
        seam_band = (_dilate(t_m, 2) & (seed_lab >= 0) & ~t_m) | (_dilate((seed_lab >= 0) & ~t_m, 2) & t_m)
        seed_lab[seam_band] = -1

    final_lab = _gradient_watershed(small, fg, seed_lab, grad_weight=3.8)
    torso_id = next((i for i, c in enumerate(ordered_candidates) if c[0] == "torso"), -1)
    part_masks: Dict[str, np.ndarray] = {}
    for idx, (name, _orig_m, _role, _is_rigid) in enumerate(ordered_candidates):
        pm = final_lab == idx
        if not pm.any():
            continue
        # Keep only primary connected component for limbs/head/weapons so watershed never leaves
        # tiny detached islands inside torso that tear open during animation.
        if name not in ("torso", "legs") and torso_id >= 0:
            _, pcs = _cc_components(pm)
            if len(pcs) > 1:
                pcs.sort(key=lambda c: -len(c))
                keep = np.zeros_like(fg)
                keep[pcs[0][:, 0], pcs[0][:, 1]] = True
                stray = pm & ~keep
                final_lab[stray] = torso_id
                pm = keep
        part_masks[name] = pm
    if torso_id >= 0:
        part_masks["torso"] = final_lab == torso_id

    # 6. Leg skeleton geometry & pose type (filter out thin ribbons and detached far-side pets)
    leg_chains: Dict[str, Dict[str, Tuple[float, float]]] = {}
    thick_legs_mask = np.zeros_like(fg)
    if "legs" in part_masks:
        lm = part_masks["legs"]
        leg_thick = _compute_thickness(lm)
        l_dmax = max(2, int(leg_thick.max()))
        lys, lxs = np.nonzero(lm)
        ly0, ly1 = float(lys.min()), float(lys.max())
        lh = max(8.0, ly1 - ly0)
        thick_legs_mask = lm & (leg_thick >= max(2, int(round(0.28 * l_dmax))))
        if not thick_legs_mask.any():
            thick_legs_mask = lm
        lower_half = thick_legs_mask & (yy_grid >= int(round(ly0 + 0.35 * lh)))
        _, leg_comps = _cc_components(lower_half)
        # Exclude components far outside reasonable hip stance width (e.g. side companion pets)
        leg_comps = [
            c
            for c in leg_comps
            if len(c) >= max(16, int(0.015 * fg_area))
            and int(leg_thick[c[:, 0], c[:, 1]].max()) >= max(3, int(round(0.35 * l_dmax)))
            and abs(float(c[:, 1].mean()) - hips_cx) <= max(torso_w_half * 2.2, 0.32 * bw)
        ]
        leg_comps.sort(key=lambda c: float(c[:, 1].mean()))

        def _leg_pts_from_region(pts_y: np.ndarray, pts_x: np.ndarray, hip_x_hint: float) -> Dict[str, Tuple[float, float]]:
            y_top = float(pts_y.min())
            y_bot = float(pts_y.max())
            span_y = max(6.0, y_bot - y_top)

            def _row_cx(frac: float, fb: float) -> float:
                yt = y_top + frac * span_y
                sel = np.abs(pts_y - yt) <= max(2.0, 0.08 * span_y)
                return float(pts_x[sel].mean()) if sel.any() else fb

            hip_pt = (hip_x_hint, y_hips_est)
            knee_y = y_top + 0.42 * span_y
            knee_pt = (_row_cx(0.42, hip_x_hint), knee_y)
            ankle_y = y_top + 0.84 * span_y
            ankle_pt = (_row_cx(0.84, knee_pt[0]), ankle_y)
            foot_pt = (_row_cx(0.96, ankle_pt[0]), y_bot)
            return {"thigh": hip_pt, "shin": knee_pt, "foot": ankle_pt, "toe": foot_pt}

        if len(leg_comps) >= 2:
            cR, cL = leg_comps[0], leg_comps[-1]
            leg_chains["R"] = _leg_pts_from_region(cR[:, 0], cR[:, 1], hips_cx - torso_w_half * 0.35)
            leg_chains["L"] = _leg_pts_from_region(cL[:, 0], cL[:, 1], hips_cx + torso_w_half * 0.35)
        else:
            tlys, tlxs = np.nonzero(thick_legs_mask)
            leg_chains["L"] = _leg_pts_from_region(tlys, tlxs, hips_cx)

    airborne = _detect_airborne_pose(thick_legs_mask if thick_legs_mask.any() else part_masks.get("legs", legs_mask), x0, x1, y0, y1, hips_cx, preset)
    if semantic_hints and semantic_hints.get("pose_type") in ("airborne", "grounded"):
        airborne = semantic_hints["pose_type"] == "airborne"

    # 7. Scale coordinates back to full canvas
    def to_canvas(pt: Tuple[float, float]) -> List[float]:
        return [round(float(pt[0]) * inv_sx, 2), round(float(pt[1]) * inv_sy, 2)]

    body_h_px = round(bh * inv_sy, 1)
    fill_depth = max(96, min(200, int(round(max(bw * inv_sx, bh * inv_sy) * 0.14))))
    mesh_cell = max(18, min(44, int(round(min(W, H) / 14.0))))

    parts_out: List[Dict[str, Any]] = []
    for _idx, (name, _orig_m, role, is_rigid) in enumerate(ordered_candidates):
        if name not in part_masks:
            continue
        pm = part_masks[name]
        pys, pxs = np.nonzero(pm)

        if name == "head":
            pivot, parent_part, kind = to_canvas((head_cx, y_neck)), "torso", "rigid"
        elif name == "torso":
            pivot, parent_part, kind = to_canvas((chest_cx, y_chest_est)), ("legs" if "legs" in part_masks else None), "soft"
        elif name == "legs":
            pivot, parent_part, kind = to_canvas((hips_cx, y_hips_est)), None, "soft"
        elif name == "arm_R" and arm_R_meta:
            pivot, parent_part, kind = to_canvas(arm_R_meta["shoulder"]), "torso", "soft"
        elif name == "arm_L" and arm_L_meta:
            pivot, parent_part, kind = to_canvas(arm_L_meta["shoulder"]), "torso", "soft"
        elif name == "weapon_R" and weapon_R_meta:
            pivot, parent_part, kind = to_canvas(weapon_R_meta["pivot"]), "arm_R", "rigid"
        elif name == "weapon_L" and weapon_L_meta:
            pivot, parent_part, kind = to_canvas(weapon_L_meta["pivot"]), "arm_L", "rigid"
        else:
            d_core = (pys - tcy) ** 2 + (pxs - tcx) ** 2
            root_i = int(np.argmin(d_core))
            rx, ry = float(pxs[root_i]), float(pys[root_i])
            tip_i = int(np.argmax((pys - ry) ** 2 + (pxs - rx) ** 2))
            tx, ty = float(pxs[tip_i]), float(pys[tip_i])
            mx_c, my_c = (rx + tx) * 0.5, (ry + ty) * 0.5
            pivot, parent_part, kind = to_canvas((rx, ry)), "torso", "soft"
            soft_metas.append(
                {
                    "name": name,
                    "parent_bone": "hips" if ry >= y_spine_est else "chest",
                    "centerline": [to_canvas((rx, ry)), to_canvas((mx_c, my_c)), to_canvas((tx, ty))],
                }
            )

        p_entry: Dict[str, Any] = {
            "name": name,
            "kind": kind,
            "role": role,
            "rigid": bool(is_rigid),
            "parent": parent_part,
            "pivot": pivot,
            "own": [_mask_to_bbox_poly(pm, inv_sx, inv_sy)],
            "band": 8,
            "depth": max(36, fill_depth // 2) if is_rigid else fill_depth,
            "guides": [(None, None, None, "smooth")],
        }
        if mask_dir is not None:
            p_entry["mask"] = _save_mask_png(pm, mask_dir / f"{name}_mask.png", (W, H))
        else:
            p_entry["own"] = _mask_to_polygons(pm, inv_sx, inv_sy)
        for sm in soft_metas:
            if sm["name"] == name:
                p_entry["centerline"] = sm["centerline"]
        parts_out.append(p_entry)

    # 8. Build five-region bone hierarchy
    root_pos = [round(hips_cx * inv_sx, 2), float(H)]
    hips_pos = to_canvas((hips_cx, y_hips_est))
    spine_pos = to_canvas(((hips_cx + chest_cx) * 0.5, y_spine_est))
    chest_pos = to_canvas((chest_cx, y_chest_est))
    neck_pos = to_canvas((head_cx, y_neck))
    head_top_pos = to_canvas((head_cx, max(float(y0), head_y0_actual)))

    bones: List[Dict[str, Any]] = [
        {"name": "root", "parent": None, "pos": root_pos, "role": "root", "label": "根节点"},
        {"name": "hips", "parent": "root", "pos": hips_pos, "aim": spine_pos, "role": "body", "label": "骨盆/重心"},
        {"name": "spine", "parent": "hips", "pos": spine_pos, "aim": chest_pos, "role": "body", "label": "脊柱"},
        {"name": "chest", "parent": "spine", "pos": chest_pos, "aim": neck_pos, "role": "body", "label": "胸腔"},
        {"name": "ribcage", "parent": "chest", "pos": chest_pos, "role": "body", "label": "呼吸胸廓"},
        {"name": "neck", "parent": "chest", "pos": neck_pos, "aim": head_top_pos, "role": "head", "label": "颈部"},
        {"name": "head", "parent": "neck", "pos": neck_pos, "aim": head_top_pos, "role": "head", "label": "头部"},
    ]

    if arm_R_meta is not None:
        sh_p, el_p, hd_p = to_canvas(arm_R_meta["shoulder"]), to_canvas(arm_R_meta["elbow"]), to_canvas(arm_R_meta["hand"])
        bones.extend(
            [
                {"name": "shoulder_R", "parent": "chest", "pos": sh_p, "aim": el_p, "role": "arm", "label": "右肩/大臂"},
                {"name": "elbow_R", "parent": "shoulder_R", "pos": el_p, "aim": hd_p, "role": "arm", "label": "右肘/小臂"},
                {"name": "hand_R", "parent": "elbow_R", "pos": hd_p, "aim": to_canvas(weapon_R_meta["tip"]) if weapon_R_meta else hd_p, "role": "hand", "label": "右手"},
            ]
        )
        if weapon_R_meta is not None:
            bones.append(
                {"name": "weapon_R", "parent": "hand_R", "pos": to_canvas(weapon_R_meta["pivot"]), "aim": to_canvas(weapon_R_meta["tip"]), "role": "weapon", "label": "右手武器（刚性）"}
            )

    if arm_L_meta is not None:
        sh_p, el_p, hd_p = to_canvas(arm_L_meta["shoulder"]), to_canvas(arm_L_meta["elbow"]), to_canvas(arm_L_meta["hand"])
        bones.extend(
            [
                {"name": "shoulder_L", "parent": "chest", "pos": sh_p, "aim": el_p, "role": "arm", "label": "左肩/大臂"},
                {"name": "elbow_L", "parent": "shoulder_L", "pos": el_p, "aim": hd_p, "role": "arm", "label": "左肘/小臂"},
                {"name": "hand_L", "parent": "elbow_L", "pos": hd_p, "aim": to_canvas(weapon_L_meta["tip"]) if weapon_L_meta else hd_p, "role": "hand", "label": "左手"},
            ]
        )
        if weapon_L_meta is not None:
            bones.append(
                {"name": "weapon_L", "parent": "hand_L", "pos": to_canvas(weapon_L_meta["pivot"]), "aim": to_canvas(weapon_L_meta["tip"]), "role": "weapon", "label": "左手武器（刚性）"}
            )

    ik_list: List[Dict[str, Any]] = []
    for side in ("L", "R"):
        if side not in leg_chains:
            continue
        lc = leg_chains[side]
        th_p, sh_p, ft_p, to_p = to_canvas(lc["thigh"]), to_canvas(lc["shin"]), to_canvas(lc["foot"]), to_canvas(lc["toe"])
        if not airborne and preset != "bust_half":
            sh_p = _ensure_ik_margin(th_p, sh_p, ft_p, side, body_h_px)
        lbl_side = "左" if side == "L" else "右"
        bones.extend(
            [
                {"name": f"thigh_{side}", "parent": "hips", "pos": th_p, "aim": sh_p, "role": "leg", "label": f"{lbl_side}大腿"},
                {"name": f"shin_{side}", "parent": f"thigh_{side}", "pos": sh_p, "aim": ft_p, "role": "leg", "label": f"{lbl_side}小腿"},
                {"name": f"foot_{side}", "parent": f"shin_{side}", "pos": ft_p, "aim": to_p, "role": "leg", "label": f"{lbl_side}脚"},
            ]
        )
        if not airborne and preset != "bust_half":
            tgt_name = f"toe_target_{side}"
            bones.append({"name": tgt_name, "parent": "root", "pos": ft_p, "role": "leg", "label": f"{lbl_side}脚底锚点"})
            ik_list.append({"name": f"foot_ik_{side}", "bones": [f"thigh_{side}", f"shin_{side}"], "target": tgt_name})

    chains: Dict[str, Any] = {}
    for sm in soft_metas:
        chains[sm["name"]] = {"parent": sm["parent_bone"], "part": sm["name"], "count": 2}

    # 9. Skin bindings with automatic full-seam CONTACT_DIST overrides
    adj = _find_adjacent_parts(part_masks)
    c_out = float(max(44, int(round(mesh_cell * 1.8))))
    skin: Dict[str, Any] = {}
    for p in parts_out:
        pname = p["name"]
        if pname == "head":
            skin["head"] = {"type": "region", "bone": "head"}
        elif pname == "torso":
            skin["torso"] = {
                "type": "mesh",
                "recipe": "linear_y",
                "bones": ["chest", "spine", "hips"],
                "ranges": [
                    [round(y_chest_est * inv_sy, 1), round(y_spine_est * inv_sy, 1)],
                    [round(y_spine_est * inv_sy, 1), round(y_hips_est * inv_sy, 1)],
                ],
                "cell": mesh_cell,
            }
        elif pname in ("arm_R", "arm_L"):
            s = pname[-1]
            # Do NOT force-lock arm edges to chest along the entire torso (arms hanging near waist/hips
            # would tear against hips-weighted lower torso). Rely on deep amodal fill + limb root_blend.
            skin[pname] = {
                "type": "mesh",
                "recipe": "limb",
                "start_bone": "chest",
                "mid_bone": f"shoulder_{s}",
                "end_bone": f"elbow_{s}",
                "root_blend": [0.0, 0.38],
                "tip_blend": [0.48, 0.88],
                "cell": mesh_cell,
            }
        elif pname in ("weapon_R", "weapon_L"):
            skin[pname] = {"type": "region", "bone": pname}
        elif pname == "legs":
            cts = []
            if "torso" in adj.get("legs", set()):
                cts.append({"part": "torso", "bone": "hips", "d_in": 0.0, "d_out": c_out})
            if "L" in leg_chains and "R" in leg_chains:
                skin["legs"] = {
                    "type": "mesh",
                    "recipe": "dual_limb",
                    "root_bone": "hips",
                    "branches": [
                        {"mid_bone": "thigh_L", "end_bone": "shin_L"},
                        {"mid_bone": "thigh_R", "end_bone": "shin_R"},
                    ],
                    "root_blend": [0.0, 0.24],
                    "tip_blend": [0.45, 0.85],
                    "contact_overrides": cts,
                    "cell": mesh_cell,
                }
            elif "L" in leg_chains:
                skin["legs"] = {
                    "type": "mesh",
                    "recipe": "limb",
                    "start_bone": "hips",
                    "mid_bone": "thigh_L",
                    "end_bone": "shin_L",
                    "root_blend": [0.0, 0.22],
                    "tip_blend": [0.45, 0.85],
                    "contact_overrides": cts,
                    "cell": mesh_cell,
                }
            else:
                skin["legs"] = {"type": "region", "bone": "hips"}
        elif pname in chains:
            par_b = chains[pname]["parent"]
            cts = []
            for nb in sorted(adj.get(pname, set())):
                if nb in ("torso", "legs", "head"):
                    b_anchor = "head" if nb == "head" else ("hips" if nb == "legs" else par_b)
                    cts.append({"part": nb, "bone": b_anchor, "d_in": 0.0, "d_out": c_out})
            skin[pname] = {
                "type": "mesh",
                "recipe": "chain",
                "chain": pname,
                "contact_overrides": cts,
                "cell": mesh_cell,
            }

    return {
        "parts": parts_out,
        "bones": bones,
        "chains": chains,
        "ik": ik_list,
        "skin": skin,
        "pose_type": "airborne" if airborne else "grounded",
        "body_height_px": body_h_px,
        "origin": [round(W * 0.5, 2), float(H)],
    }


def build_default_specs(
    image_path: Union[str, Path],
    *,
    character: str = "portrait",
    preset: str = "agile_full",
    mask_dir: Optional[Union[str, Path]] = None,
    semantic_hints: Optional[Dict[str, Any]] = None,
    legacy_diagnostic: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Analyze the preprocessed RGBA portrait and generate semantic `(parts_spec, rig_spec)`."""
    if semantic_hints and semantic_hints.get("module_contract"):
        from src.pipeline.module_rig import build_module_specs
        return build_module_specs(image_path, character=sanitize_character_name(character),
                                  preset=preset, semantic_hints=semantic_hints)
    if not legacy_diagnostic:
        from .vision_schema import VisionReviewRequired
        error = VisionReviewRequired("生产生成需要完整V2模块规划；旧V1/几何路径仅限显式诊断。")
        error.code = "MODULE_PLAN_REQUIRED"
        raise error
    rgba = np.asarray(Image.open(image_path).convert("RGBA"), dtype=np.uint8)
    if semantic_hints and semantic_hints.get("vision_contract"):
        contract = semantic_hints["vision_contract"]
        actual_sha = hashlib.sha256(Path(image_path).read_bytes()).hexdigest()
        if contract.get("input_sha256") != actual_sha or contract.get("image_size") != [rgba.shape[1], rgba.shape[0]]:
            raise ValueError("视觉归属结果不属于当前图像，不能复用旧分析。")
    mdir = Path(mask_dir).resolve() if mask_dir is not None else None
    seg = _segment_portrait_semantics(rgba, preset=preset, mask_dir=mdir, semantic_hints=semantic_hints)

    parts_spec: Dict[str, Any] = {
        "parts": seg["parts"],
        "generation_mode": "legacy_diagnostic",
        "min_island_px": 12,
        "pose_type": seg["pose_type"],
    }

    rig_spec: Dict[str, Any] = {
        "character": sanitize_character_name(character),
        "generation_mode": "legacy_diagnostic",
        "duration": 8.0,
        "fps": 30,
        "key_step": 0.25,
        "origin": seg["origin"],
        "pose_type": seg["pose_type"],
        "body_height_px": seg["body_height_px"],
        "bones": seg["bones"],
        "chains": seg["chains"],
        "ik": seg["ik"],
        "skin": seg["skin"],
        "preserve_visual_joints": bool(semantic_hints and semantic_hints.get("vision_contract")),
    }
    return parts_spec, rig_spec


def apply_rig_edits(
    rig_spec: Dict[str, Any],
    edits: Dict[str, Any],
) -> Dict[str, Any]:
    """Apply user bone additions, removals, joint drags, and skin rebindings safely."""
    if rig_spec.get("moduleContract"):
        from .vision_schema import VisionReviewRequired
        error = VisionReviewRequired("完整模块不能经旧骨骼编辑改写；请在当前对话更新V2模块规划并重新提交。")
        error.code = "MODULE_REPLAN_REQUIRED"
        raise error
    updated = copy.deepcopy(rig_spec)

    if isinstance(edits.get("bones"), list) and edits["bones"]:
        cleaned_bones: List[Dict[str, Any]] = []
        seen_names = set()
        for b in edits["bones"]:
            name = str(b.get("name") or "").strip()
            if not name or name in seen_names:
                continue
            seen_names.add(name)
            pos = b.get("pos") or [0.0, 0.0]
            entry: Dict[str, Any] = {
                "name": name,
                "parent": None if name == "root" else (str(b["parent"]) if b.get("parent") else "root"),
                "pos": [round(float(pos[0]), 2), round(float(pos[1]), 2)],
                "role": infer_bone_role(name, b.get("role")),
                "label": str(b.get("label") or name),
            }
            if b.get("aim") is not None and len(b["aim"]) == 2:
                entry["aim"] = [round(float(b["aim"][0]), 2), round(float(b["aim"][1]), 2)]
            cleaned_bones.append(entry)

        if "root" not in seen_names:
            orig_root = next((b for b in updated.get("bones", []) if b["name"] == "root"), None)
            cleaned_bones.insert(
                0,
                copy.deepcopy(orig_root)
                if orig_root
                else {"name": "root", "parent": None, "pos": [0.0, 0.0], "role": "root", "label": "根节点"},
            )
            seen_names.add("root")

        for b in cleaned_bones:
            if b["name"] == "root":
                b["parent"] = None
            elif b["parent"] not in seen_names or b["parent"] == b["name"]:
                b["parent"] = "root"

        by_name = {b["name"]: b for b in cleaned_bones}
        ordered: List[Dict[str, Any]] = []
        placed = set()

        def _place(bn: str, stack: set) -> None:
            if bn in placed or bn not in by_name:
                return
            if bn in stack:
                by_name[bn]["parent"] = "root"
            else:
                par = by_name[bn].get("parent")
                if par and par not in placed:
                    _place(par, stack | {bn})
            placed.add(bn)
            ordered.append(by_name[bn])

        for b in cleaned_bones:
            _place(b["name"], set())

        updated["bones"] = ordered
        valid_bones = set(by_name.keys())
        updated["chains"] = {
            cname: cspec
            for cname, cspec in (updated.get("chains") or {}).items()
            if cspec.get("parent") in valid_bones
        }
        updated["ik"] = [
            c
            for c in (updated.get("ik") or [])
            if c.get("target") in valid_bones and all(bn in valid_bones for bn in c.get("bones", []))
        ]

    if isinstance(edits.get("skin"), dict):
        valid_bones = {b["name"] for b in updated.get("bones", [])}
        for cname, cspec in (updated.get("chains") or {}).items():
            for i in range(int(cspec.get("count", 2))):
                valid_bones.add(f"{cname}{i + 1}")
        skin_out = copy.deepcopy(updated.get("skin") or {})
        for pname, sspec in edits["skin"].items():
            if not isinstance(sspec, dict):
                continue
            sc = copy.deepcopy(sspec)
            if sc.get("type") == "region":
                if sc.get("bone") not in valid_bones:
                    sc["bone"] = "hips" if "hips" in valid_bones else "root"
            elif sc.get("type") == "mesh":
                rtype = sc.get("recipe")
                if rtype == "limb":
                    for k in ("start_bone", "mid_bone", "end_bone"):
                        if sc.get(k) and sc[k] not in valid_bones:
                            sc[k] = "hips" if "hips" in valid_bones else "root"
                elif rtype in ("linear_y", "linear_x"):
                    sc["bones"] = [bn if bn in valid_bones else "root" for bn in sc.get("bones", ["root", "root"])]
            skin_out[pname] = sc
        updated["skin"] = skin_out

    valid_bones = {b["name"] for b in updated.get("bones", [])}
    for cname, cspec in (updated.get("chains") or {}).items():
        for i in range(int(cspec.get("count", 2))):
            valid_bones.add(f"{cname}{i + 1}")
    for pname, sspec in list((updated.get("skin") or {}).items()):
        if sspec.get("type") == "region":
            if sspec.get("bone") not in valid_bones:
                sspec["bone"] = "hips" if "hips" in valid_bones else "root"
        elif sspec.get("type") == "mesh":
            rtype = sspec.get("recipe")
            if rtype == "chain" and sspec.get("chain") not in (updated.get("chains") or {}):
                updated["skin"][pname] = {"type": "region", "bone": "hips" if "hips" in valid_bones else "root"}
            elif rtype == "limb":
                for k in ("start_bone", "mid_bone", "end_bone"):
                    if sspec.get(k) and sspec[k] not in valid_bones:
                        sspec[k] = "hips" if "hips" in valid_bones else "root"
            elif rtype == "dual_limb":
                if sspec.get("root_bone") not in valid_bones:
                    sspec["root_bone"] = "hips" if "hips" in valid_bones else "root"
                for br in sspec.get("branches", []):
                    for k in ("mid_bone", "end_bone"):
                        if br.get(k) and br[k] not in valid_bones:
                            br[k] = sspec["root_bone"]
            elif rtype in ("linear_y", "linear_x"):
                sspec["bones"] = [bn if bn in valid_bones else "root" for bn in sspec.get("bones", [])]

    requested_pose = edits.get("pose_type")
    if requested_pose in ("airborne", "grounded") and requested_pose != rig_spec.get("pose_type"):
        by_name = {b["name"]: b for b in updated.get("bones", [])}
        leg_names = {f"{part}_{side}" for side in ("L", "R") for part in ("thigh", "shin", "foot")}
        # Release only leg constraints; hand and accessory IK remain intact.
        updated["ik"] = [
            c for c in updated.get("ik", [])
            if not leg_names.intersection(c.get("bones", []))
            and c.get("name") not in ("foot_ik_L", "foot_ik_R")
        ]
        if requested_pose == "airborne":
            # Remove generated targets only when no remaining user binding uses them.
            def referenced(value: Any, name: str) -> bool:
                if isinstance(value, dict):
                    return name in value or any(referenced(v, name) for v in value.values())
                if isinstance(value, (list, tuple)):
                    return any(referenced(v, name) for v in value)
                return isinstance(value, str) and value == name

            for side in ("L", "R"):
                name = f"toe_target_{side}"
                target = by_name.get(name)
                if target is None or target.get("parent") != "root":
                    continue
                other_bones = [b for b in updated["bones"] if b["name"] != name]
                references = [other_bones, updated.get("ik", []), updated.get("skin", {}),
                              updated.get("chains", {}), updated.get("motion", {})]
                if not referenced(references, name):
                    updated["bones"] = other_bones
        else:
            if "root" not in by_name or "hips" not in by_name:
                raise ValueError("无法切换站立姿态：缺少 root 或 hips 骨骼。")
            leg_count = 0
            for side in ("L", "R"):
                names = [f"{part}_{side}" for part in ("thigh", "shin", "foot")]
                if not any(name in by_name for name in names):
                    continue
                if not all(name in by_name for name in names):
                    raise ValueError(f"无法切换站立姿态：{side} 侧腿链不完整，需要大腿、小腿和脚骨骼。")
                thigh, shin, foot = (by_name[name] for name in names)
                if (thigh.get("parent") != "hips" or shin.get("parent") != names[0]
                        or foot.get("parent") != names[1]):
                    raise ValueError(f"无法切换站立姿态：{side} 侧腿链父子关系不正确。")
                points = [np.asarray(b["pos"], dtype=float) for b in (thigh, shin, foot)]
                if any(p.shape != (2,) or not np.isfinite(p).all() for p in points):
                    raise ValueError(f"无法切换站立姿态：{side} 侧关节点坐标无效。")
                p0, p1, p2 = points
                l1, l2 = float(np.linalg.norm(p1 - p0)), float(np.linalg.norm(p2 - p1))
                slack = l1 + l2 - float(np.linalg.norm(p2 - p0))
                if min(l1, l2) < 1.0 or (slack < 1.5 and not updated.get("preserve_visual_joints")):
                    raise ValueError(f"无法切换站立姿态：{side} 侧腿链过直或过短，请先调整膝关节留出弯曲余量。")
                target_name = f"toe_target_{side}"
                target = by_name.get(target_name)
                if target is not None and (target.get("parent") != "root"
                        or not np.allclose(target["pos"], foot["pos"], atol=0.01, rtol=0)):
                    raise ValueError(f"无法切换站立姿态：{target_name} 已被修改，请先将锚点对齐脚关节。")
                if target is None:
                    target = {"name": target_name, "parent": "root", "pos": list(foot["pos"]),
                              "role": "leg", "label": f"{side} 脚底锚点"}
                    updated["bones"].append(target)
                # Reconnect bone lengths to edited joints without moving any joint.
                thigh["aim"] = list(shin["pos"])
                shin["aim"] = list(foot["pos"])
                constraint = {"name": f"foot_ik_{side}", "bones": names[:2], "target": target_name}
                if updated.get("preserve_visual_joints"):
                    constraint.update(stretch=True, uniform=True)
                updated["ik"].append(constraint)
                leg_count += 1
            if not leg_count:
                raise ValueError("无法切换站立姿态：没有可绑定的完整腿链，请先添加或修正腿部骨骼。")
        updated["pose_type"] = requested_pose
    elif requested_pose in ("airborne", "grounded"):
        updated["pose_type"] = requested_pose

    return updated
