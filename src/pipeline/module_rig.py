"""Compile verified V2 modules without anatomical guessing or cross-module skinning."""
from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw

from src.studio.module_contract import (MODULE_RULE_VERSION, LOCAL_FILL_PX_AT_1080,
    AUTO_SEAM_UNDERLAP_PX_AT_1080, BODY_ROLES, validate_module_contract, contract_fingerprint,
    validate_hidden_joint_region)
from src.studio.vision_schema import VisionValidationError, VisionReviewRequired

# Shared engineering profiles. They are not per-character correction tables.
ROLE_SWAY_DEG = {
    "head": 2.8, "face": 2.1, "torso": 2.4, "body": 2.4, "pelvis": 1.3,
    "upper_arm": 2.0, "forearm": 2.0, "arm": 2.0, "hand": 1.6,
    "upper_leg": 2.3, "lower_leg": 2.1, "leg": 2.3, "foot": 0.6,
    "lower_body": 1.9, "weapon": 0.0, "armor": 1.8, "hair": 2.4,
    "cloth": 2.2, "cape": 2.4, "tail": 2.4, "fish_tail": 2.0, "fin": 2.0,
    "effect_group": 2.4, "pet": 2.4, "prop": 2.8, "soft_accessory": 2.2,
}
# Tiny synthetic canvases keep the previous engineering floor so 1.5x stress
# hole checks stay compatible; 1080px portraits use the full table above.
ROLE_SWAY_FLOOR_DEG = {
    "head": .55, "face": .3, "torso": .30, "body": .30, "pelvis": .24,
    "upper_arm": .55, "forearm": .50, "arm": .50, "hand": .30,
    "upper_leg": .42, "lower_leg": .38, "leg": .42, "foot": .25,
    "lower_body": .30, "weapon": .0, "armor": .0, "hair": .35,
    "cloth": .30, "cape": .30, "tail": .35, "fish_tail": .30, "fin": .30,
    "effect_group": .20, "pet": .20, "prop": .20, "soft_accessory": .30,
}
ROLE_LAG_S = {"pelvis": .0, "body": .10, "torso": .15, "head": .45,
              "upper_arm": .35, "arm": .35, "forearm": .55, "hand": .65,
              "upper_leg": .20, "lower_leg": .40, "leg": .30, "lower_body": .25}
LOCKED_ROOT_DEG = 0.25
MAX_MODULE_GAIN = 1.5
AIRBORNE_BREATH_TX = (9.5, 4.5)
AIRBORNE_BREATH_TY = (17.0, 4.2)


def soft_bend_deg_for_link(index: int, *, locked_root: bool = False, scale: float = 1.0) -> float:
    """Progressive chain bend; locked_root keeps the palmar/blade root at 0.25deg."""
    if locked_root and index == 0:
        return LOCKED_ROOT_DEG
    target = min(4.6, 2.0 + 0.65 * index)
    scaled = target * min(1.0, max(float(scale), 0.0))
    return max(scaled, 0.50)



SOFT_CENTERLINE_MAX_TURN_DEG = 45.0


def _centerline_max_turn_deg(pts):
    if len(pts) < 3:
        return 0.0
    worst = 0.0
    for i in range(1, len(pts) - 1):
        ax, ay = float(pts[i][0]) - float(pts[i - 1][0]), float(pts[i][1]) - float(pts[i - 1][1])
        bx, by = float(pts[i + 1][0]) - float(pts[i][0]), float(pts[i + 1][1]) - float(pts[i][1])
        la = math.hypot(ax, ay)
        lb = math.hypot(bx, by)
        if la < 1e-6 or lb < 1e-6:
            continue
        cos_t = max(-1.0, min(1.0, (ax * bx + ay * by) / (la * lb)))
        deg = math.degrees(math.acos(cos_t))
        if deg > worst:
            worst = deg
    return worst


def smooth_sharp_centerline(pts, *, max_turn_deg=SOFT_CENTERLINE_MAX_TURN_DEG, target_turn_deg=42.0):
    """Auto-relax sharp hairpin kinks (>45 deg) in soft-cloth centerlines.

    Acute kinks in a soft_bend bone chain cause LBS mesh self-folding / striped
    tears inside solid cloth. Smooth curves (<= max_turn_deg) are returned
    unchanged; sharp kinks relax interior joints while preserving exact start
    pivot and end tip.
    """
    orig_deg = _centerline_max_turn_deg(pts)
    if len(pts) < 3 or orig_deg <= max_turn_deg:
        return pts, None
    cur = [[float(p[0]), float(p[1])] for p in pts]
    for _ in range(32):
        if _centerline_max_turn_deg(cur) <= target_turn_deg:
            break
        nxt = [cur[0]]
        for i in range(1, len(cur) - 1):
            nxt.append([
                0.5 * cur[i][0] + 0.25 * cur[i - 1][0] + 0.25 * cur[i + 1][0],
                0.5 * cur[i][1] + 0.25 * cur[i - 1][1] + 0.25 * cur[i + 1][1],
            ])
        nxt.append(cur[-1])
        cur = nxt
    final_deg = _centerline_max_turn_deg(cur)
    return cur, {
        "code": "soft_centerline_sharp_kink_smoothed",
        "original_max_turn_deg": round(orig_deg, 1),
        "smoothed_max_turn_deg": round(final_deg, 1),
    }


def _bone(mid: str) -> str:
    return "module_" + mid


def _topological(modules: list) -> list:
    by_id = {m["id"]: m for m in modules}
    result, visited = [], set()
    def visit(mid):
        if mid in visited:
            return
        parent = by_id[mid]["parent"]
        if parent is not None:
            visit(parent)
        visited.add(mid)
        result.append(by_id[mid])
    for mid in by_id:
        visit(mid)
    return result



def _placement_tol(body_h: float) -> int:
    return max(2, min(6, int(round(float(body_h) / 300.0))))


def _finite_xy(pt: Any, label: str) -> Tuple[float, float]:
    if not isinstance(pt, (list, tuple)) or len(pt) < 2:
        raise VisionReviewRequired("%s 坐标非法。" % label)
    x, y = float(pt[0]), float(pt[1])
    if not math.isfinite(x) or not math.isfinite(y):
        raise VisionReviewRequired("%s 坐标非法。" % label)
    return x, y


def _true_min_mask_distance(mask: np.ndarray, x: float, y: float) -> float:
    h, w = mask.shape
    if not math.isfinite(x) or not math.isfinite(y):
        return float(math.hypot(w, h) + abs(x) + abs(y) + 1.0)
    xi, yi = int(round(x)), int(round(y))
    if 0 <= yi < h and 0 <= xi < w and bool(mask[yi, xi]):
        return 0.0
    if not np.any(mask):
        return float(math.hypot(w, h) + abs(x) + abs(y) + 1.0)
    ys, xs = np.nonzero(mask)
    d2 = (xs.astype(np.float64) - x) ** 2 + (ys.astype(np.float64) - y) ** 2
    dist = float(np.sqrt(np.min(d2)))
    if not math.isfinite(dist):
        return float(math.hypot(w, h) + abs(x) + abs(y) + 1.0)
    return dist


def _mask_distance(mask: np.ndarray, x: float, y: float, *, window: int = 32) -> float:
    h, w = mask.shape
    if not math.isfinite(x) or not math.isfinite(y):
        return _true_min_mask_distance(mask, x, y)
    xi, yi = int(round(x)), int(round(y))
    if 0 <= yi < h and 0 <= xi < w and bool(mask[yi, xi]):
        return 0.0
    r = max(1, int(window))
    y0, y1 = max(0, yi - r), min(h, yi + r + 1)
    x0, x1 = max(0, xi - r), min(w, xi + r + 1)
    sub = mask[y0:y1, x0:x1]
    if np.any(sub):
        ys, xs = np.nonzero(sub)
        d2 = (xs.astype(np.float64) + x0 - x) ** 2 + (ys.astype(np.float64) + y0 - y) ** 2
        dist = float(np.sqrt(np.min(d2)))
        if math.isfinite(dist):
            return dist
    return _true_min_mask_distance(mask, x, y)


def _validate_rig_mask_placement(
    modules: Sequence[dict],
    connections: Sequence[dict],
    mod_masks: Dict[str, np.ndarray],
    width: int,
    height: int,
    body_h: float,
    hidden_joint_support: Optional[dict] = None,
    hidden_attachment_support: Optional[dict] = None,
) -> None:
    tol = _placement_tol(body_h)
    soft_tol = 2
    w, h = float(width), float(height)

    def px_point(pt: Any, label: str) -> Tuple[float, float]:
        x, y = _finite_xy(pt, label)
        return x * w, y * h

    for module in modules:
        mid = module["id"]
        mask = mod_masks[mid]
        x, y = px_point(module.get("pivot"), "模块 %s 的 pivot" % mid)
        dist = _mask_distance(mask, x, y)
        if dist > tol:
            true_d = _true_min_mask_distance(mask, x, y)
            raise VisionReviewRequired(
                "模块 %s 的 pivot 落在可见掩码外 %.4fpx（像素坐标 [%.2f, %.2f]），容差 %dpx。"
                % (mid, true_d, x, y, tol)
            )
        if module.get("motion_type") == "soft":
            for idx, pt in enumerate(module.get("centerline") or []):
                cx, cy = px_point(pt, "模块 %s 的 centerline[%s]" % (mid, idx))
                dist = _mask_distance(mask, cx, cy)
                if dist > soft_tol:
                    true_d = _true_min_mask_distance(mask, cx, cy)
                    raise VisionReviewRequired(
                        "模块 %s 的 centerline 点序号 %d 落在可见掩码外 %.4fpx（像素坐标 [%.2f, %.2f]），容差 %dpx。"
                        % (mid, idx, true_d, cx, cy, soft_tol)
                    )
    for conn in connections:
        cid = conn.get("id", "connection")
        a, b = conn.get("a"), conn.get("b")
        x, y = px_point(conn.get("point"), "连接 %s 的 point" % cid)
        chain = conn.get("chain") or []
        if conn.get("type") == "foot_anchor" and chain:
            targets = [chain[-1], b]
        else:
            targets = [a, b]
        for mid in targets:
            if mid not in mod_masks:
                raise VisionReviewRequired("连接 %s 引用了缺失模块 %s。" % (cid, mid))
            dist = _mask_distance(mod_masks[mid], x, y)
            if dist > tol:
                if (mid == a and conn.get('type') == 'attachment' and conn.get('hidden_by')
                        and cid in (hidden_attachment_support or {})):
                    support = hidden_attachment_support[cid]
                    xi, yi = int(round(x)), int(round(y))
                    if 0 <= yi < support.shape[0] and 0 <= xi < support.shape[1] and support[yi, xi]:
                        continue
                if (mid==a and conn.get('type')=='joint' and conn.get('hidden_by')
                        and cid in (hidden_joint_support or {})):
                    support=hidden_joint_support[cid]
                    xi,yi=int(round(x)),int(round(y))
                    if 0<=yi<support.shape[0] and 0<=xi<support.shape[1] and support[yi,xi]:
                        continue
                true_d = _true_min_mask_distance(mod_masks[mid], x, y)
                raise VisionReviewRequired(
                    "连接 %s 的 point 距模块 %s 可见掩码 %.4fpx（像素坐标 [%.2f, %.2f]），容差 %dpx；涉事模块 %s/%s。"
                    % (cid, mid, true_d, x, y, tol, a, b)
                )


def _mask_outline(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    if not np.any(m):
        return m
    er = m.copy()
    er[1:, :] &= m[:-1, :]
    er[:-1, :] &= m[1:, :]
    er[:, 1:] &= m[:, :-1]
    er[:, :-1] &= m[:, 1:]
    return m & ~er


_OVERLAY_COLORS = [
    (220, 70, 70), (70, 140, 220), (80, 180, 90), (230, 180, 50),
    (160, 90, 200), (50, 180, 180), (230, 120, 60), (90, 90, 200),
    (200, 90, 140), (100, 160, 70),
]


def render_rig_mask_overlay(image_path: Union[str, Path], rig_spec: dict, out_path: Union[str, Path]) -> Path:
    """Composite source + module tints + pivot/centerline/contact overlay."""
    src_path = Path(image_path)
    dest = Path(out_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    contract = (rig_spec or {}).get("moduleContract") or {}
    modules = contract.get("modules") or []
    with Image.open(src_path) as im:
        src_rgba = np.asarray(im.convert("RGBA"), dtype=np.float32)
    h, w = src_rgba.shape[0], src_rgba.shape[1]
    canvas = contract.get("canvas") or [w, h]
    if list(canvas)[:2] != [w, h] and canvas != [w, h]:
        # overlay still uses the source pixels; mask files must match source size
        pass
    base = np.full((h, w, 3), 125.0, dtype=np.float32)
    alpha = src_rgba[..., 3:4] / 255.0
    base = base * (1.0 - alpha) + src_rgba[..., :3] * alpha
    for i, module in enumerate(modules):
        mask_list = module.get("visibleMasks")
        if not mask_list:
            raise ValueError("模块 %s 的 visibleMasks 为空" % module.get("id"))
        acc = np.zeros((h, w), dtype=bool)
        for mask_path in mask_list:
            mp = Path(mask_path)
            if not mp.is_file():
                raise ValueError("visibleMasks 不存在: %s" % mask_path)
            with Image.open(mp) as mm:
                if mm.size != (w, h):
                    raise ValueError(
                        "visibleMasks 尺寸 %s 与原图 %dx%d 不一致: %s"
                        % (mm.size, w, h, mask_path)
                    )
                acc |= np.asarray(mm.convert("L")) > 127
        if not np.any(acc):
            continue
        color = np.array(_OVERLAY_COLORS[i % len(_OVERLAY_COLORS)], dtype=np.float32)
        tint = 0.32
        base[acc] = base[acc] * (1.0 - tint) + color * tint
        edge = _mask_outline(acc)
        base[edge] = color
    long_edge = max(w, h)
    scale = 1600.0 / long_edge if long_edge > 1600 else 1.0
    ow, oh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    if scale != 1.0:
        img = img.resize((ow, oh), Image.Resampling.BILINEAR)
    draw = ImageDraw.Draw(img)

    def xy(pt):
        return float(pt[0]) * scale, float(pt[1]) * scale

    by_id = {m.get("id"): m for m in modules}
    for module in modules:
        parent_id = module.get("parent")
        if parent_id and parent_id in by_id and module.get("pivot") and by_id[parent_id].get("pivot"):
            x0, y0 = xy(by_id[parent_id]["pivot"])
            x1, y1 = xy(module["pivot"])
            draw.line([(x0, y0), (x1, y1)], fill=(40, 40, 40), width=max(1, int(round(2 * scale))))
    r = max(3, int(round(4 * scale)))
    for i, module in enumerate(modules):
        color = _OVERLAY_COLORS[i % len(_OVERLAY_COLORS)]
        if module.get("centerline") and len(module["centerline"]) >= 2:
            pts = [xy(p) for p in module["centerline"]]
            draw.line(pts, fill=color, width=max(1, int(round(2 * scale))))
            for idx, (px, py) in enumerate(pts):
                draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=(255, 255, 255), outline=color)
                draw.text((px + 3, py + 3), str(idx), fill=(20, 20, 20))
        if module.get("pivot") and len(module["pivot"]) >= 2:
            px, py = xy(module["pivot"])
            draw.ellipse((px - r, py - r, px + r, py + r), fill=color, outline=(255, 255, 255))
            label = str(module.get("id") or "")[:12]
            draw.text((px + r + 1, py - r - 1), label, fill=(20, 20, 20))
    for contact in contract.get("contacts") or []:
        pt = None
        if isinstance(contact.get("a"), dict):
            pt = contact["a"].get("point")
        if pt is None and isinstance(contact.get("b"), dict):
            pt = contact["b"].get("point")
        if pt is None:
            pt = contact.get("world") or contact.get("point")
        if not pt or len(pt) < 2:
            continue
        px, py = xy(pt)
        s = max(3, int(round(3 * scale)))
        draw.polygon([(px, py - s), (px + s, py), (px, py + s), (px - s, py)], outline=(20, 20, 20), fill=(255, 255, 255))
    img.save(dest)
    return dest


def build_module_specs(image_path: Union[str, Path], *, character: str,
                       preset: str, semantic_hints: Dict[str, Any]) -> Tuple[dict, dict]:
    source = Path(image_path).resolve()
    with Image.open(source) as image:
        w, h = image.size
        source_alpha = np.asarray(image.convert("RGBA"), dtype=np.uint8)[..., 3]
    contract = validate_module_contract(semantic_hints.get("module_contract"),
        input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), image_size=(w, h))
    masks = semantic_hints.get("module_masks") or {}
    from src.gates.review_binding import resolve_evidence_path, read_evidence_mask
    # Materialization emits all per-input masks in one explicit directory.
    # A secondary evidence reference cannot jump outside that directory.
    if not masks:raise VisionReviewRequired('完整模块没有物化掩码。')
    evidence_root=Path(next(iter(masks.values()))).absolute().parent.resolve()
    hidden_masks = semantic_hints.get("module_hidden_masks") or {}
    provided_images = semantic_hints.get('module_provided_images') or {}
    owner_projection = copy.deepcopy(semantic_hints.get("module_projection") or {})
    if (owner_projection.get("sourceSHA256") != contract["input_sha256"]
            or owner_projection.get("canvas") != [w, h]
            or owner_projection.get("contractSHA256") != contract_fingerprint(contract)):
        raise VisionValidationError("完整模块缺少本图可见归属校验。")
    modules = _topological(contract["modules"])
    by_id = {m["id"]: m for m in modules}
    def point(p):
        return [round(p[0]*w, 5), round(p[1]*h, 5)]
    arrays, coverage = {}, np.zeros((h,w), dtype=bool)
    for m in modules:
        for p in m["pieces"]:
            path = masks.get(p["id"])
            if not path or not Path(path).is_file():
                raise VisionValidationError("模块可见掩码不存在。")
            try:path=resolve_evidence_path(path,root=evidence_root)
            except ValueError as exc:raise VisionReviewRequired('模块掩码越过物化证据目录。') from exc
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != owner_projection.get("visibleMaskSHA256",{}).get(p["id"]):
                raise VisionReviewRequired("可见掩码已脱离本次模块规划，不能复用被改写的归属。")
            with Image.open(path) as im:
                if im.size != (w,h):
                    raise VisionValidationError("模块掩码不可缩放复用。")
                mask = np.asarray(im.convert("L")) > 127
            if not mask.any() or (coverage & mask).any() or (mask & (source_alpha==0)).any():
                raise VisionReviewRequired("模块掩码为空、重叠或包含非前景。")
            coverage |= mask
            arrays[p["id"]] = mask
    if not np.array_equal(coverage, source_alpha>0):
        raise VisionReviewRequired("完整模块仍有未归属前景。")
    body_mask = np.zeros_like(coverage)
    for m in modules:
        if m["role"] in BODY_ROLES or m["role"] == "fish_tail":
            for p in m["pieces"]:
                body_mask |= arrays[p["id"]]
    ys, _ = np.nonzero(body_mask)
    body_h = max(1.0, float(ys.max()-ys.min()+1)) if ys.size else float(h)
    fill_depth = max(1, int(round(LOCAL_FILL_PX_AT_1080*body_h/1080.0)))
    seam_depth = max(4, int(round(AUTO_SEAM_UNDERLAP_PX_AT_1080*body_h/1080.0)))
    fill_depth = max(fill_depth, seam_depth)
    mod_masks = {}
    for m in modules:
        acc = np.zeros((h, w), dtype=bool)
        for p in m["pieces"]:
            acc |= arrays[p["id"]]
        mod_masks[m["id"]] = acc
    hidden_joint_support={}
    projections={(o['front'],o['back']):o for o in owner_projection.get('occlusions',[]) if not o.get('autoSeamUnderlap')}
    for oc in projections.values():
        for key in ('sourceHiddenTargetMask','remainingHiddenMask'):
            try:
                path,_,_=read_evidence_mask(oc.get(key,''),oc.get(key+'SHA256'),(w,h),root=evidence_root)
                oc[key]=str(path)
            except ValueError as exc:
                raise VisionReviewRequired('遮挡证据路径/摘要无效：'+str(exc)) from exc
    pieces_by_id={p['id']:(m,p) for m in modules for p in m['pieces']}
    for conn in contract['connections']:
        if not conn.get('hidden_by'):continue
        targets=[]
        for ref in conn['hidden_by']:
            key=(ref['front'],ref['back']);o=projections.get(key)
            if not o or not o.get('sourceHiddenTargetMask') or not o.get('sourceHiddenTargetMaskSHA256'):
                raise VisionReviewRequired('遮挡joint缺少本次登记遮挡域：'+conn['id'])
            try:
                target_path,_,target=read_evidence_mask(o['sourceHiddenTargetMask'],o['sourceHiddenTargetMaskSHA256'],
                                             (w,h),root=evidence_root)
                remaining_path,_,_=read_evidence_mask(o.get('remainingHiddenMask',''),o.get('remainingHiddenMaskSHA256'),
                                   (w,h),root=evidence_root)
                o['sourceHiddenTargetMask']=str(target_path)
                o['remainingHiddenMask']=str(remaining_path)
            except ValueError as exc:
                raise VisionReviewRequired('遮挡joint证据路径/摘要无效：'+conn['id']+'；'+str(exc)) from exc
            # Recompute expectation from the validated source contract. A forged
            # projection/hash cannot silently authorize a larger corridor.
            oc=next(v for v in contract['occlusions'] if (v['front'],v['back'])==key)
            from src.studio.vision_schema import rasterize_polygons
            zone=rasterize_polygons(oc['polygons'],w,h)
            expected=zone & arrays[ref['front']] & ~arrays[ref['back']]
            back_piece=pieces_by_id[ref['back']][1]
            if not back_piece.get('mask_png_base64'):
                expected &= rasterize_polygons(back_piece['polygons'],w,h)
            if not np.array_equal(target,expected):raise VisionReviewRequired('遮挡joint登记范围与原合同不一致。')
            targets.append(target)
        hidden_joint_support[conn['id']]=validate_hidden_joint_region(
            mod_masks[conn['a']],targets,point(conn['point']),source_alpha)
    _validate_rig_mask_placement(modules, contract.get("connections") or [], mod_masks, w, h, body_h,
                                 hidden_joint_support=hidden_joint_support)
    bones = [{"name":"root", "parent":None, "pos":[w/2.0,float(h)], "role":"root"}]
    chains, skin, parts, projection_modules, ik, centerline_warnings = {}, {}, [], [], [], []
    for m in modules:
        mid, driver = m["id"], _bone(m["id"])
        pos = point(m["pivot"])
        bones.append({"name":driver, "parent":_bone(m["parent"]) if m["parent"] else "root",
                      "pos":pos, "role":m["role"], "label":mid})
        soft_bones = []
        if m["motion_type"] == "soft":
            cname = "bend_" + mid
            pts = [point(q) for q in m["centerline"]]
            pts, kink_warn = smooth_sharp_centerline(pts)
            if kink_warn is not None:
                centerline_warnings.append({"module": mid, **kink_warn})
            count = len(pts)-1
            chains[cname] = {"parent":driver, "pts":pts, "count":count}
            soft_bones = [cname + str(i+1) for i in range(count)]
        module_pieces, module_masks = [], []
        for p in m["pieces"]:
            pid, part_name = p["id"], "piece_" + p["id"]
            module_pieces.append(part_name)
            module_masks.append(masks[pid])
            msk = arrays[pid]
            py, px = np.nonzero(msk)
            part = {"name":part_name, "module_id":mid, "piece_id":pid,
                "module_motion_type":m["motion_type"], "role":m["role"],
                "kind":"soft" if m["motion_type"]=="soft" else "rigid",
                "pivot":pos, "parent":m["parent"], "mask":masks[pid],
                "allowed_mask":masks[pid], "ownership_locked":True,
                "draw_order":p["draw_order"], "band":0, "depth":0,
                "own":[[[int(px.min()),int(py.min())],[int(px.max()+1),int(py.min())],
                        [int(px.max()+1),int(py.max()+1)],[int(px.min()),int(py.max()+1)]]],
                "module_rule_version":MODULE_RULE_VERSION, "completion_mode":"none"}
            if pid in hidden_masks:
                try:resolve_evidence_path(hidden_masks[pid],root=evidence_root)
                except ValueError as exc:raise VisionReviewRequired('隐藏补全掩码越过物化证据目录。') from exc
                if hashlib.sha256(Path(hidden_masks[pid]).read_bytes()).hexdigest() != owner_projection.get("hiddenMaskSHA256",{}).get(pid):
                    raise VisionReviewRequired("隐藏补全域已脱离本次模块规划。")
                # Materialization already honours the declared local depth.
                # Do not clip that authorized mask back to the default during
                # splitting. The validated contract (not projection metadata)
                # supplies the bound; the mask still limits every filled pixel.
                declared_depths = [max(1, int(round(float(oc.get("band_depth_px_at_1080", LOCAL_FILL_PX_AT_1080)) * body_h / 1080.0)))
                                   for oc in contract["occlusions"]
                                   if oc["back"] == pid and oc["completion"] == "local"]
                part.update(hidden_fill_mask=hidden_masks[pid], depth=max([fill_depth] + declared_depths),
                    completion_mode="local", completion_review="pending",
                    guides=[(None,None,None,"patch")], strict_hidden_fill=True)
            if pid in provided_images:
                supplied_occlusions=[oc for oc in contract['occlusions'] if oc['back']==pid and oc['completion']=='provided']
                if not supplied_occlusions:raise VisionReviewRequired('Canvas材质必须在原合同显式声明provided，不能只改投影。')
                provided_path=resolve_evidence_path(provided_images[pid],root=evidence_root)
                refs=[o for o in owner_projection.get('occlusions',[]) if o['back']==pid and o.get('providedImage')]
                digest=hashlib.sha256(provided_path.read_bytes()).hexdigest()
                if not refs or any(o.get('providedImageSHA256')!=digest for o in refs):
                    raise VisionReviewRequired('Canvas回接材质已脱离当前规划。')
                with Image.open(provided_path) as image:
                    if image.mode!='RGBA' or image.size!=(w,h):raise VisionReviewRequired('Canvas材质尺寸/格式不一致。')
                    actual=np.asarray(image).copy()
                from src.mivo.canvas_assets import decode_canvas_completion
                from src.studio.vision_schema import rasterize_polygons
                expected=np.zeros_like(actual)
                for oc in supplied_occlusions:
                    candidate=decode_canvas_completion(oc['provided'],image_size=(w,h),input_sha256=contract['input_sha256'])
                    target=rasterize_polygons(oc.get('completion_polygons') or oc['polygons'],w,h)&arrays[oc['front']]&~arrays[pid]
                    if not p.get('mask_png_base64'):target &= rasterize_polygons(p['polygons'],w,h)
                    target &= (source_alpha==255)&(candidate[...,3]>0)
                    expected[target]=candidate[target]
                if not np.array_equal(actual,expected):raise VisionReviewRequired('Canvas物化材质与原合同内容不一致。')
                part.update(provided_image=str(provided_path),provided_image_sha256=digest,completion_mode='provided')
            if m["motion_type"] == "soft":
                part["centerline"] = [point(q) for q in m["centerline"]]
                ss = {"type":"mesh", "recipe":"chain", "chain":"bend_"+mid,
                      "cell":max(8,min(24,int(round(body_h/80))))}
            elif m["representation"] == "mesh":
                ss = {"type":"mesh", "recipe":"rigid", "bone":driver,
                      "cell":max(8,min(24,int(round(body_h/80))))}
            else:
                ss = {"type":"region", "bone":driver}
            ss.update(module_id=mid, piece_id=pid, semantic_role=m["role"])
            skin[part_name] = ss
            parts.append(part)
        pm = {"id":mid,"shape":"soft" if m["motion_type"]=="soft" else "rigid",
              "role":m["role"], "motion":"static" if m["motion_type"]=="static" else "required",
              "motionProfile":m["motion"], "motionType":m["motion_type"],
              "pieces":module_pieces,"visibleMasks":module_masks,"bone":driver,
              "softBones":soft_bones,"parent":m["parent"],"pivot":pos,
              "side":m.get("side"),"confidence":m["confidence"],"motion_scale":m.get("motion_scale",1.)}
        if m.get("static_reason"):
            pm["staticReason"] = m["static_reason"]
        if m.get("centerline"):
            pm["centerline"] = [point(q) for q in m["centerline"]]
        if m.get("locked_root"):
            pm["lockedRoot"] = True
        projection_modules.append(pm)
    parts.sort(key=lambda p:p["draw_order"])
    by_bone = {b["name"]:b for b in bones}
    contacts = []
    for c in contract["connections"]:
        pt = point(c["point"])
        a, b = c["a"], c["b"]
        pa = next(m for m in projection_modules if m["id"]==a)["pieces"][0]
        pb = next(m for m in projection_modules if m["id"]==b)["pieces"][0]
        contact = {"id":c["id"],"type":"foot" if c["type"]=="foot_anchor" else c["type"],
                   "a":{"module":a,"piece":pa,"point":pt},
                   "b":{"module":b,"piece":pb,"point":pt}}
        if c.get('hidden_by'):
            contact['hiddenBy']=[{'front':'piece_'+v['front'],'back':'piece_'+v['back']} for v in c['hidden_by']]
        if c["type"] == "foot_anchor":
            upper, lower = c["chain"]
            up, low, target = by_bone[_bone(upper)], by_bone[_bone(lower)], by_bone[_bone(b)]
            up["aim"] = low["pos"]
            low["aim"] = target["pos"]
            l1, l2 = math.dist(up["pos"],low["pos"]), math.dist(low["pos"],target["pos"])
            if min(l1,l2)<1.0:
                raise VisionReviewRequired("刚体足锚肢段退化，不能拉伸或移点补救。")
            ik.append({"name":"contact_"+c["id"],"bones":[_bone(upper),_bone(lower)],
                       "target":_bone(b)})
            contact["world"] = pt
            contact["chain"] = [_bone(upper),_bone(lower)]
            contact["a"] = {"module":lower,"piece":next(m for m in projection_modules if m["id"]==lower)["pieces"][0],"point":pt}
        contacts.append(contact)
    piece_slots = {p["piece_id"]:p["name"] for p in parts}
    piece_depths = {p["piece_id"]:p["depth"] for p in parts}
    occlusions = []
    for o in owner_projection.get("occlusions", []):
        item = copy.deepcopy(o)
        item["front"] = piece_slots[o["front"]]
        item["back"] = piece_slots[o["back"]]
        item["maxLocalDepthPx"] = piece_depths[o["back"]] if o["completionMode"]=="local" else 0
        occlusions.append(item)
    module_projection = {"version":1,"sourceVersion":2,"ruleVersion":MODULE_RULE_VERSION,
        "sourceSHA256":contract["input_sha256"],"canvas":[w,h],"anatomy":contract["anatomy"],
        "contractSHA256":owner_projection["contractSHA256"],
        "visibleMaskSHA256":{piece_slots[pid]:digest for pid,digest in owner_projection["visibleMaskSHA256"].items()},
        "hiddenMaskSHA256":{piece_slots[pid]:digest for pid,digest in owner_projection.get("hiddenMaskSHA256",{}).items()},
        "requiredModules":[m["id"] for m in modules],"modules":projection_modules,
        "contacts":contacts,"occlusions":occlusions,
        "boundaryRefinements":copy.deepcopy(owner_projection.get("boundaryRefinements", [])),
        "ownership":{"unassignedPixels":0,"visibleOverlapPixels":0,
                     "semanticFidelity":"requires_visual_review"},
        "motionPolicy":"typed-rigid-and-bounded-soft-v1", "numericalCalibration":"engineering_initial"}
    rig = {"character":character,"duration":8.0,"fps":30,"key_step":.125,
           "origin":[w/2.0,float(h)],"body_height_px":body_h,"pose_type":contract["pose_type"],
           "bones":bones,"chains":chains,"ik":ik,"skin":skin,
           "centerlineWarnings":centerline_warnings,"moduleContract":module_projection,"preserve_visual_joints":True}
    return {"parts":parts,"min_island_px":0,"feather_px":0,"pose_type":contract["pose_type"],
            "module_contract":module_projection}, rig


def _soft_chain_expected(module: dict, rig: dict) -> tuple:
    chain_name = None
    for piece in module.get("pieces") or []:
        spec = (rig.get("skin") or {}).get(piece) or {}
        if spec.get("chain"):
            chain_name = spec.get("chain")
            break
    chain = (rig.get("chains") or {}).get(chain_name) or {}
    count = chain.get("count")
    if not isinstance(count, int) or count < 1 or not chain_name:
        raise VisionReviewRequired("软模块缺少完整链定义。")
    expected = [chain_name + str(i + 1) for i in range(count)]
    return chain_name, chain, expected


def _parent_map(bones) -> dict:
    mapping = {}
    if isinstance(bones, dict):
        items = bones.items()
        for name, data in items:
            if isinstance(data, dict):
                mapping[name] = data.get("parent")
            else:
                parent = getattr(data, "parent", None)
                mapping[name] = getattr(parent, "name", parent)
    else:
        for data in bones or []:
            if isinstance(data, dict):
                mapping[data["name"]] = data.get("parent")
            else:
                parent = getattr(data, "parent", None)
                mapping[data.name] = None if parent is None else getattr(parent, "name", parent)
    return mapping


def _validate_soft_bone_graph(module: dict, parent_of: dict, *, require_present: bool) -> None:
    driver = module.get("bone")
    soft = list(module.get("softBones") or [])
    present = [name for name in soft if name in parent_of]
    if not require_present and not present:
        return
    if not soft or len(present) != len(soft):
        raise VisionReviewRequired("软模块骨链缺失。")
    for index, name in enumerate(soft):
        expected_parent = driver if index == 0 else soft[index - 1]
        if parent_of.get(name) != expected_parent:
            raise VisionReviewRequired("软模块骨链父级错接。")


def _validate_soft_bones(module: dict, rig: dict, bones: dict) -> None:
    chain_name, chain, expected = _soft_chain_expected(module, rig)
    soft = module.get("softBones")
    if not isinstance(soft, list) or not soft or list(soft) != expected:
        raise VisionReviewRequired("软模块softBones与链定义不一致。")
    if chain.get("parent") != module.get("bone"):
        raise VisionReviewRequired("软链父级必须是本模块驱动骨。")
    parent_of = _parent_map(bones)
    present = [name for name in expected if name in parent_of]
    _validate_soft_bone_graph(module, parent_of, require_present=bool(present))


def validate_expanded_soft_chains(contract: dict, parent_of: dict) -> None:
    """Post-expansion check: every soft chain bone exists and is wired to its driver."""
    for module in contract.get("modules") or []:
        if module.get("shape") != "soft":
            continue
        _validate_soft_bone_graph(module, parent_of, require_present=True)


def _validate_typed_motion(module: dict, by_id: dict) -> None:
    role, shape, profile = module.get("role"), module.get("shape"), module.get("motionProfile")
    motion_field = module.get("motion")
    if motion_field not in ("required", "static"):
        raise VisionReviewRequired("模块投影motion必须是required或static。")
    locked = bool(module.get("lockedRoot"))
    if locked and motion_field != "required":
        raise VisionReviewRequired("lockedRoot模块的motion必须是required。")
    if locked:
        if role != "effect_group" or shape != "soft" or profile != "soft_bend" or not module.get("parent"):
            raise VisionReviewRequired("lockedRoot仅用于挂在活动宿主下的effect_group软链。")
        cur = module.get("parent")
        active = False
        while cur is not None:
            ancestor = by_id.get(cur)
            if ancestor is None:
                break
            if ancestor.get("motionProfile") not in ("follow", "static"):
                active = True
                break
            cur = ancestor.get("parent")
        if not active:
            raise VisionReviewRequired("lockedRoot没有活动宿主。")
    if role == "effect_group" and shape == "soft" and profile == "soft_bend" and not locked:
        raise VisionReviewRequired("effect_group软链必须lockedRoot。")
    if role == "armor" and (shape != "rigid" or profile not in ("sway", "follow")):
        raise VisionReviewRequired("护甲必须是rigid且motion为sway或follow。")
    if role == "weapon" and (shape != "rigid" or profile != "follow"):
        raise VisionReviewRequired("武器必须是rigid且motion为follow。")


def validate_module_rig_spec(rig: dict) -> None:
    contract = rig.get("moduleContract")
    if not isinstance(contract,dict) or contract.get("sourceVersion")!=2:
        raise VisionValidationError("缺少V2模块投影。")
    if rig.get("motion"):
        raise VisionReviewRequired("V2只使用通用类型动作，旧逐骨曲线需要重新规划。")
    bones = {b["name"]:b for b in rig.get("bones",[])}
    for b in bones.values():
        if any(k in b for k in ("scale","scaleX","scaleY","shearX","shearY")):
            raise VisionReviewRequired("V2模块骨骼禁止缩放和剪切。")
    for c in rig.get("ik",[]):
        if c.get("stretch") or c.get("compress") or c.get("softness") or c.get("mix",1)!=1:
            raise VisionReviewRequired("硬模块IK不得伸缩或软混合。")
    for module in contract["modules"]:
        driver = module["bone"]
        if driver not in bones:
            raise VisionReviewRequired("模块驱动骨缺失。")
        expected_parent = _bone(module["parent"]) if module.get("parent") else "root"
        if bones[driver].get("parent") != expected_parent:
            raise VisionReviewRequired("模块父级被旧骨编辑改写，需要重规划。")
        for piece in module["pieces"]:
            spec = rig.get("skin",{}).get(piece)
            if not spec or spec.get("module_id")!=module["id"]:
                raise VisionReviewRequired("模块piece绑定缺失或归属不一致。")
            if spec.get("contact_overrides") or spec.get("contacts"):
                raise VisionReviewRequired("V2禁止按接近距离跨模块混权。")
            if module["shape"]=="rigid":
                if not (spec.get("bone")==driver and (spec.get("type")=="region"
                        or spec.get("type")=="mesh" and spec.get("recipe")=="rigid")):
                    raise VisionReviewRequired("硬模块必须统一单驱动保形。")
            else:
                chain = rig.get("chains",{}).get(spec.get("chain"),{})
                if spec.get("type")!="mesh" or spec.get("recipe")!="chain" or chain.get("parent")!=driver:
                    raise VisionReviewRequired("软模块只能使用本模块的受限链。")
        if module["shape"]=="soft":
            _validate_soft_bones(module, rig, bones)
        _validate_typed_motion(module, {m["id"]: m for m in contract["modules"]})


def _motion_family(role: str) -> str:
    if role in ("head", "face"):
        return "head"
    if role in ("torso", "body", "pelvis", "chest", "hips"):
        return "torso"
    if role in ("arm", "upper_arm", "forearm", "hand"):
        return "arm"
    if role in ("leg", "upper_leg", "lower_leg", "thigh", "shin", "foot"):
        return "leg"
    if role in ("cape", "cloth", "hair", "tail", "soft_accessory", "fin", "fish_tail"):
        return "soft"
    return role or "other"


def _module_axis_x(module: dict) -> float:
    pivot = module.get("pivot") or [0.0, 0.0]
    px = float(pivot[0])
    line = module.get("centerline") or []
    if module.get("shape") == "soft" and len(line) >= 2:
        tip = line[-1]
        return 0.5 * (px + float(tip[0]))
    return px


def _assign_lateral_signs(modules: Sequence[dict]) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Geometric L/R without character IDs. Smaller x is image-left (-1)."""
    roots = [m for m in modules if m.get("parent") is None]
    root_x = _module_axis_x(roots[0]) if roots else None
    groups: Dict[Any, List[dict]] = {}
    for module in modules:
        groups.setdefault((module.get("parent"), _motion_family(module.get("role") or "")), []).append(module)
    signs: Dict[str, int] = {}
    slots: Dict[str, int] = {}
    for siblings in groups.values():
        ordered = sorted(siblings, key=_module_axis_x)
        count = len(ordered)
        for index, module in enumerate(ordered):
            mid = module["id"]
            declared = module.get("side")
            if declared == "L":
                signs[mid] = -1
            elif declared == "R":
                signs[mid] = 1
            elif count == 1:
                if root_x is None:
                    signs[mid] = 1
                else:
                    signs[mid] = -1 if _module_axis_x(module) < root_x else 1
            else:
                signs[mid] = -1 if index < count / 2.0 else 1
            slots[mid] = index
    return signs, slots


def canvas_motion_policy(plan, tweaks=None):
    tweaks = tweaks or {}
    overall = tweaks.get('global_motion_scale', 1.)
    if type(overall) not in (float, int) or not 0 <= overall <= 1.8:
        raise ValueError('整体动感必须在 0%–180% 之间。')
    defaults = {m['id']: m.get('motion_scale', 1.) for m in plan['modules']}
    requested = {mid: 1. for mid in defaults}
    for module_id, change in tweaks.get('modules', {}).items():
        if module_id not in defaults:
            raise ValueError('Unknown module '+module_id)
        if 'motion_ratio' in change:
            value = change['motion_ratio']
            if type(value) not in (float, int) or not 0 <= value <= 1.5:
                raise ValueError('部件摆幅必须在 0%–150% 之间。')
        elif 'motion_scale' in change:
            value = change['motion_scale']
            if type(value) not in (float, int) or not 0 <= value <= 1.5:
                raise ValueError('motion_scale必须在0..1.5。')
            value = min(1.5, value / defaults[module_id]) if defaults[module_id] else 0.
        else:
            continue
        requested[module_id] = value
    return requested, {mid: min(1.5, ratio * overall) for mid, ratio in requested.items()}


def resolve_module_motion(rig: dict, preset: str, *, duration: float=8.0) -> dict:
    """Role/profile/parent/geometry select motion; IDs and image hashes never tune it."""
    validate_module_rig_spec(rig)
    parent_of = _parent_map(rig.get("bones") or [])
    for module in rig["moduleContract"]["modules"]:
        if module.get("shape") == "soft":
            present = any(name in parent_of for name in (module.get("softBones") or []))
            _validate_soft_bone_graph(module, parent_of, require_present=present)
    factor = .75 if preset=="heavy_weapon_full" else 1.0
    ik_bones = {b for c in rig.get("ik",[]) for b in c["bones"]}
    anchored_ancestors = set()
    parents = {b["name"]: b.get("parent") for b in rig["bones"]}
    for constraint in rig.get("ik", []):
        ancestor = parents.get(constraint["bones"][0])
        while ancestor is not None and ancestor != "root":
            anchored_ancestors.add(ancestor)
            ancestor = parents.get(ancestor)
    body_h = float(rig.get("body_height_px") or 1080.0)
    scale = body_h / 1080.0
    def phase(lag: float, harmonic: int = 1) -> float:
        return 2 * math.pi * harmonic * lag / duration
    def terms(*items):
        out = []
        for amp, harmonic, lag in items:
            out.append((amp * factor, harmonic, phase(lag, harmonic)))
        return out
    modules = list(rig["moduleContract"]["modules"])
    signs, slots = _assign_lateral_signs(modules)
    torso_mod = next((m for m in modules if m.get("parent") is None and _motion_family(m.get("role") or "") == "torso"), None)
    if torso_mod is None:
        torso_mod = next((m for m in modules if m.get("parent") is None), None)
    motion = {}
    for m in modules:
        bn, role, profile = m["bone"], m["role"], m["motionProfile"]
        if profile in ("static", "follow") or bn in ik_bones:
            continue
        family = _motion_family(role)
        sign = signs.get(m["id"], 1)
        slot_idx = slots.get(m["id"], 0)
        locked = bool(m.get("lockedRoot"))
        if m.get("motion") not in ("required", "static"):
            raise VisionReviewRequired("模块投影motion必须是required或static。")
        if locked and m.get("motion") != "required":
            raise VisionReviewRequired("lockedRoot模块的motion必须是required。")
        if locked and (role != "effect_group" or profile != "soft_bend" or not m.get("parent")):
            raise VisionReviewRequired("lockedRoot仅用于挂在活动宿主下的effect_group软链。")
        if locked:
            amp = LOCKED_ROOT_DEG
        else:
            target = ROLE_SWAY_DEG.get(role, .30)
            amp = max(target * min(1.0, scale), ROLE_SWAY_FLOOR_DEG.get(role, .30))
        if bn in anchored_ancestors:
            if m.get("parent") is None:
                ty_amp = min(1.85 * factor, 0.012 * body_h)
                motion[bn] = {"ty": [(-ty_amp, 3, math.pi / 2)]}
            continue
        if family == "torso" and profile == "breath" and m.get("parent") is None:
            motion[bn] = {
                "rotate": terms((amp * 0.95, 2, 0.15), (amp * 0.40, 3, 0.15)),
                "tx": terms((AIRBORNE_BREATH_TX[0], 2, 0.15), (AIRBORNE_BREATH_TX[1], 1, 0.15)),
                "ty": terms((AIRBORNE_BREATH_TY[0], 3, 0.0), (AIRBORNE_BREATH_TY[1], 2, 0.0)),
            }
            continue
        if m.get("parent") is None and not locked and role in ("prop", "pet") and profile != "soft_bend":
            lag = 0.22 + (0.35 if sign > 0 else 0.0) + 0.14 * slot_idx
            motion[bn] = {
                "rotate": terms((sign * amp * 1.05, 2, lag), (-sign * amp * 0.45, 3, lag + 0.18)),
                "tx": terms((sign * 6.5, 1, lag + 0.15), (sign * 3.0, 2, lag + 0.30)),
                "ty": terms((14.5, 2, lag + 0.08), (5.0, 3, lag + 0.22)),
            }
            continue
        if m.get("parent") is None and not locked and role == "effect_group" and profile != "soft_bend":
            lag = 0.18 + (0.40 if sign > 0 else 0.0) + 0.16 * slot_idx
            motion[bn] = {
                "rotate": terms((sign * amp * 1.15, 2, lag), (sign * amp * 0.50, 3, lag + 0.20)),
                "tx": terms((sign * 6.5, 1, lag + 0.12), (-sign * 3.0, 3, lag + 0.25)),
                "ty": terms((9.5, 2, lag + 0.10), (4.0, 3, lag + 0.28)),
            }
            continue
        if family == "head":
            motion[bn] = {
                "rotate": terms((-amp * 1.15, 2, 0.35), (amp * 0.55, 3, 0.25)),
            }
            continue
        if family == "arm":
            lag = 0.32 + (0.0 if sign < 0 else 0.30)
            if role in ("forearm", "hand"):
                lag += 0.18 if role == "forearm" else 0.26
                k2 = -sign
            else:
                k2 = sign
            motion[bn] = {"rotate": terms(
                (sign * amp * 1.00, 3, lag),
                (k2 * amp * 0.65, 2, lag + 0.12),
            )}
            continue
        if family == "leg":
            lag = 0.52 + (0.0 if sign < 0 else 0.30)
            motion[bn] = {"rotate": terms(
                (sign * amp * 1.0, 3, lag),
                (-sign * amp * 0.75, 2, lag + 0.14),
            )}
            continue
        if profile == "soft_bend" or family == "soft":
            if role == "cape":
                lag = 0.70 + (0.22 if sign > 0 else 0.0) + 0.08 * slot_idx
                k3, k2amp = amp * 0.91, amp * 0.286
                k2_sign = sign
                chain_k2, chain_k4 = 0.18, 0.0
                chain_delay = 0.13
            elif role in ("cloth", "soft_accessory"):
                lag = 0.32 + (0.22 if sign > 0 else 0.0) + 0.10 * slot_idx
                k3, k2amp = amp * 1.10, amp * 0.42
                k2_sign = -sign
                chain_k2, chain_k4 = 0.24, 0.10
                chain_delay = 0.26
            else:
                lag = 0.45 + (0.24 if sign > 0 else 0.0) + 0.10 * slot_idx
                k3, k2amp = amp * 0.98, amp * 0.36
                k2_sign = -sign
                chain_k2, chain_k4 = 0.24, 0.10
                chain_delay = 0.26
            links = list(m.get("softBones") or [])
            if locked:
                motion[bn] = {"rotate": terms((sign * amp, 3, lag))}
            else:
                motion[bn] = {"rotate": terms((sign * k3, 3, lag), (k2_sign * k2amp, 2, lag + 0.16))}
            for i, soft_bone in enumerate(links):
                link_amp = soft_bend_deg_for_link(i, locked_root=locked, scale=scale)
                delay = chain_delay + 0.02 * (i % 2)
                link_lag = lag + delay * (i + 1)
                if locked and i == 0:
                    motion[soft_bone] = {"rotate": terms((sign * link_amp, 3, link_lag))}
                    continue
                items = [
                    (sign * link_amp, 3, link_lag),
                    (k2_sign * link_amp * chain_k2, 2, link_lag + 0.12),
                ]
                if chain_k4 and i == len(links) - 1:
                    items.append((sign * link_amp * chain_k4, 4, link_lag + 0.08))
                motion[soft_bone] = {"rotate": terms(*items)}
            continue
        lag = ROLE_LAG_S.get(role, .65)
        motion[bn] = {"rotate": terms((sign * amp, 3, lag), (sign * amp * 0.25, 2, lag + 0.16))}
    canvas_ratios = rig.get('canvas_motion_ratios', {})
    for m in modules:
        factor = float(m.get('motion_scale', 1.0))
        if not math.isfinite(factor) or not 0 <= factor <= 1.5:
            raise VisionReviewRequired('motion_scale必须在0..1.5。')
        final = canvas_ratios.get(m['id'], 1.)
        if not math.isfinite(final) or not 0 <= final <= 1.5:
            raise VisionReviewRequired('Canvas生效摆幅必须在0..1.5倍默认动作。')
        for bone in [m['bone']] + list(m.get('softBones') or []):
            for key, values in motion.get(bone, {}).items():
                motion[bone][key] = [(a*factor*final, harmonic, phase) for a,harmonic,phase in values]
    return motion

def preflight_module_motion(skeleton: dict, rig: dict, *, duration: float) -> dict:
    """Verify the ordinary motion stays in rigid IK reach; never stretch or silently clamp."""
    from src.render.spine_runtime import Skeleton
    sk = Skeleton(skeleton)
    max_error = 0.0
    if not sk.ik:
        return {"sampleCount":0, "rigidAnchorMaxDriftPx":0.0, "stretchAllowed":False}
    times = np.linspace(0.0, duration, 129)
    for time in times:
        sk.apply("idle", float(time))
        for constraint in sk.ik:
            target = sk.by_name[constraint["target"]]
            end = sk.by_name[constraint["bones"][-1]].tip_world()
            drift = float(np.linalg.norm(end - [target.worldX,target.worldY]))
            max_error = max(max_error,drift)
            if drift > .01:
                error = VisionReviewRequired("刚体足锚超出当前动作可达范围：" + constraint["name"]
                    + "；保持原关节与形状，请重新规划合法动作，不伸长或冻结。")
                error.code = "MODULE_CONTACT_UNREACHABLE"
                raise error
    return {"sampleCount":len(times), "rigidAnchorMaxDriftPx":max_error, "stretchAllowed":False}

