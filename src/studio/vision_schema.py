"""Validate image-bound visual analysis and preserve full-resolution entity ownership.

This module consumes a provider result. It never infers entities from filenames,
loads historical hints, or claims a supplied JSON object came from a live model.
"""
from __future__ import annotations

import copy
import hashlib
import math
import re
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

MIN_ENTITY_CONFIDENCE = 0.8  # Engineering review threshold, not calibrated model accuracy.

class VisionValidationError(ValueError):
    code = "VISION_RESULT_INVALID"
    retryable = False


class VisionReviewRequired(VisionValidationError):
    code = "VISION_REVIEW_REQUIRED"


def _point(value: Any, label: str) -> list:
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not math.isfinite(v) or not 0 <= v <= 1 for v in value)):
        raise VisionValidationError(f"{label} 必须是 0..1 范围内的有限二维坐标。")
    return [float(value[0]), float(value[1])]


def _polygon(value: Any, label: str) -> list:
    if not isinstance(value, list) or not 3 <= len(value) <= 2048:
        raise VisionValidationError(f"{label} 需要 3–2048 个多边形顶点。")
    points = [_point(p, label) for p in value]
    if points[0] == points[-1]:
        points.pop()
    if len(points) < 3 or len(set(map(tuple, points))) != len(points):
        raise VisionValidationError(f"{label} 包含重复顶点或退化边界。")
    area = sum(a[0]*b[1]-b[0]*a[1] for a,b in zip(points, points[1:]+points[:1]))
    if abs(area) < 1e-10:
        raise VisionValidationError(f"{label} 面积为零。")
    # Reject crossing non-adjacent edges instead of filling an ambiguous bow-tie.
    def cross(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    def on_segment(a, b, c):
        return (abs(cross(a,b,c)) <= 1e-12
                and min(a[0],b[0])-1e-12 <= c[0] <= max(a[0],b[0])+1e-12
                and min(a[1],b[1])-1e-12 <= c[1] <= max(a[1],b[1])+1e-12)
    n = len(points)
    for i in range(n):
        a, b = points[i], points[(i+1) % n]
        for j in range(i+1, n):
            if j == i+1 or (i == 0 and j == n-1):
                continue
            c, d = points[j], points[(j+1) % n]
            if ((cross(a,b,c)*cross(a,b,d) < 0 and cross(c,d,a)*cross(c,d,b) < 0)
                    or on_segment(a,b,c) or on_segment(a,b,d) or on_segment(c,d,a) or on_segment(c,d,b)):
                raise VisionReviewRequired(f"{label} 边界自交，需要重新识别。")
    return points


def validate_vision_result(
    payload: Any, *, input_sha256: str, image_size: Sequence[int],
) -> Dict[str, Any]:
    """Return a normalized copy; mismatched image or unresolved analysis never falls back."""
    if isinstance(payload, dict) and payload.get("version") == 2:
        from .module_contract import validate_module_contract
        return validate_module_contract(payload, input_sha256=input_sha256, image_size=image_size)
    if not isinstance(payload, dict) or isinstance(payload.get("version"), bool) or payload.get("version") != 1:
        raise VisionValidationError("视觉结果必须使用 version=1。")
    result = copy.deepcopy(payload)
    digest = result.get("input_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest) or digest != input_sha256:
        raise VisionValidationError("视觉结果的输入 SHA-256 不匹配。")
    size = result.get("image_size")
    if (not isinstance(size, list) or len(size) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in size)
            or tuple(size) != tuple(image_size)):
        raise VisionValidationError("视觉结果尺寸与当前图像不匹配。")
    if result.get("coord_mode") != "normalized":
        raise VisionValidationError("视觉结果坐标必须使用 normalized。")
    if result.get("pose_type") not in ("grounded", "airborne"):
        raise VisionValidationError("视觉结果缺少有效站立/浮空姿态。")
    uncertainties = result.get("uncertainties")
    if not isinstance(uncertainties, list) or any(not isinstance(x, str) for x in uncertainties):
        raise VisionValidationError("uncertainties 必须是字符串列表。")
    if uncertainties:
        raise VisionReviewRequired("视觉结果仍有未决项：" + "；".join(uncertainties))
    joints = result.get("joints")
    if not isinstance(joints, dict) or not {"head_top", "neck", "chest", "hips"}.issubset(joints):
        raise VisionReviewRequired("视觉结果缺少头顶、颈部、胸腔或骨盆定位。")
    allowed_joints = {"head_top", "neck", "chest", "hips"}
    for side in ("L", "R"):
        allowed_joints.update(f"{part}_{side}" for part in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle", "toe"))
        allowed_joints.add(f"weapon_{side}_tip")
        for parts in (("shoulder", "elbow", "wrist"), ("hip", "knee", "ankle")):
            keys = {f"{part}_{side}" for part in parts}
            if keys.intersection(joints) and not keys.issubset(joints):
                raise VisionReviewRequired(f"{side} 侧肢体关键点不完整，不能按比例猜测缺失关节。")
    if set(joints) - allowed_joints:
        raise VisionValidationError("包含未支持的关节名称：" + ", ".join(sorted(set(joints)-allowed_joints)))
    result["joints"] = {k: _point(v, k) for k,v in joints.items()}
    entities = result.get("entities")
    if not isinstance(entities, list) or not 1 <= len(entities) <= 32:
        raise VisionValidationError("entities 需要 1–32 个可见实体。")
    names = set()
    subjects = 0
    normalized_entities = []
    for entity in entities:
        if not isinstance(entity, dict):
            raise VisionValidationError("实体必须是对象。")
        e = copy.deepcopy(entity)
        name = e.get("id")
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,47}", name) or name in names:
            raise VisionValidationError("实体 id 必须是唯一的小写字母、数字和下划线标识。")
        names.add(name)
        kind = e.get("kind")
        if kind not in ("subject", "pet", "tail", "accessory", "weapon"):
            raise VisionValidationError(f"{name} 的实体类型无效。")
        subjects += kind == "subject"
        confidence = e.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int,float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise VisionValidationError(f"{name} 需要 0..1 confidence。")
        if confidence < MIN_ENTITY_CONFIDENCE:
            raise VisionReviewRequired(f"{name} 置信度 {confidence:.2f} 低于工程待审阈值 {MIN_ENTITY_CONFIDENCE:.2f}，需复核而非自动绑定。")
        if ("polygon" in e) == ("polygons" in e):
            raise VisionValidationError(f"{name} 必须且只能提供 polygon 或 polygons。")
        polys = [e["polygon"]] if "polygon" in e else e["polygons"]
        if not isinstance(polys, list) or not 1 <= len(polys) <= 128:
            raise VisionValidationError(f"{name} 的多边形列表无效。")
        e["polygons"] = [_polygon(p, name) for p in polys]
        e.pop("polygon", None)
        centerline = e.get("centerline", [])
        if not isinstance(centerline, list) or len(centerline) > 256:
            raise VisionValidationError(f"{name} 的中心线格式无效。")
        e["centerline"] = [_point(p, name + ".centerline") for p in centerline]
        for flag in ("static", "rigid"):
            if flag in e and not isinstance(e[flag], bool):
                raise VisionValidationError(f"{name}.{flag} 必须是布尔值。")
        if kind == "pet":
            if e.get("static", True) is not True or e.get("parent_bone", "root") != "root":
                raise VisionReviewRequired("宠物必须独立静态绑定 root，不能绑定人物腿。")
            e.update(static=True, parent_bone="root", rigid=True)
        elif kind == "tail":
            if e.get("static", False) or e.get("rigid", False) or e.get("parent_bone", "hips") != "hips" or len(e["centerline"]) < 2:
                raise VisionReviewRequired("尾巴需要至少两点软链并绑定 hips，不能混入腿部。")
            e.update(static=False, parent_bone="hips", rigid=False)
        elif kind == "weapon":
            if e.get("static", False) or e.get("rigid", True) is not True or e.get("parent_bone") not in ("hand_L", "hand_R"):
                raise VisionReviewRequired("武器必须刚性绑定对应 hand_L/hand_R。")
            side = e["parent_bone"][-1]
            if f"wrist_{side}" not in joints:
                raise VisionReviewRequired("武器对应手腕未被定位。")
            e.update(static=False, rigid=True)
        elif kind == "accessory":
            if e.get("parent_bone", "hips") not in ("root", "hips", "chest", "head", "hand_L", "hand_R"):
                raise VisionValidationError("附属物父骨骼无效。")
            e.setdefault("parent_bone", "hips")
            e.setdefault("static", False)
            e.setdefault("rigid", len(e["centerline"]) < 2)
        normalized_entities.append(e)
    if subjects != 1:
        raise VisionReviewRequired("必须明确且仅有一个人物主体。")
    result["entities"] = normalized_entities
    return result


def rasterize_polygons(polygons: Sequence, width: int, height: int) -> np.ndarray:
    """Pixel-center ray casting at the original resolution; shared edges do not double-own."""
    result = np.zeros((height, width), dtype=bool)
    for polygon in polygons:
        points = np.asarray(polygon, dtype=float) * np.array([width, height])
        x0 = max(0, int(math.floor(points[:,0].min())))
        x1 = min(width, int(math.ceil(points[:,0].max())))
        y0 = max(0, int(math.floor(points[:,1].min())))
        y1 = min(height, int(math.ceil(points[:,1].max())))
        if x1 <= x0 or y1 <= y0:
            continue
        xs = np.arange(x0, x1, dtype=float)[None,:] + .5
        # Chunk rows to avoid giant temporary float arrays for high-resolution portraits.
        for ya in range(y0, y1, 128):
            yb = min(y1, ya + 128)
            ys = np.arange(ya, yb, dtype=float)[:,None] + .5
            inside = np.zeros((yb-ya, x1-x0), dtype=bool)
            for a,b in zip(points, np.roll(points,-1,axis=0)):
                if a[1] == b[1]:
                    continue
                crossing = ((a[1] > ys) != (b[1] > ys)) & (xs < (b[0]-a[0])*(ys-a[1])/(b[1]-a[1])+a[0])
                inside ^= crossing
            result[ya:yb,x0:x1] |= inside
    return result


def vision_result_to_semantic_hints(
    validated: Dict[str, Any], *, image_path: Union[str, Path], mask_dir: Union[str, Path],
) -> Dict[str, Any]:
    """Materialize image-bound masks; reject conflicts/unassigned foreground before writing."""
    if isinstance(validated, dict) and validated.get("version") == 2:
        from .module_contract import materialize_module_contract
        return materialize_module_contract(validated, image_path=image_path, mask_dir=mask_dir)
    path = Path(image_path).resolve()
    with Image.open(path) as image:
        width, height = image.size
        alpha = np.asarray(image.convert("RGBA"), dtype=np.uint8)[...,3]
    result = validate_vision_result(validated, input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), image_size=(width,height))
    occupied = np.zeros((height,width), dtype=bool)
    masks = {}
    foreground = alpha > 0
    subject_id = None
    for entity in result["entities"]:
        mask = rasterize_polygons(entity["polygons"],width,height) & foreground
        if not mask.any():
            raise VisionReviewRequired(f"实体 {entity['id']} 没有覆盖可见前景。")
        if entity["kind"] == "subject":
            subject_id = entity["id"]
            masks[subject_id] = mask
            continue
        overlap = occupied & mask
        if overlap.any():
            raise VisionReviewRequired(f"实体 {entity['id']} 与其他实体重叠 {int(overlap.sum())} 个前景像素，不能自动决定归属。")
        occupied |= mask
        masks[entity["id"]] = mask
    # The subject is an allowed region, not a claim that pets/tails inside it are human.
    masks[subject_id] &= ~occupied
    if not masks[subject_id].any():
        raise VisionReviewRequired("排除独立实体后人物主体为空。")
    occupied |= masks[subject_id]
    missing = foreground & ~occupied
    if missing.any():
        raise VisionReviewRequired(f"视觉实体未覆盖 {int(missing.sum())} 个前景像素，需要补充归属。")
    destination = Path(mask_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    mask_paths = {}
    accessories = []
    subject_path = None
    for entity in result["entities"]:
        name = "entity_" + entity["id"]
        mask = masks[entity["id"]]
        output = destination / (name + ".png")
        Image.fromarray(mask.astype(np.uint8)*255).save(output)
        if entity["kind"] == "subject":
            subject_path = str(output)
            continue
        mask_paths[name] = str(output)
        pts = entity["centerline"]
        if not pts:
            ys,xs = np.nonzero(mask)
            pts = [[float((xs.mean()+.5)/width), float((ys.mean()+.5)/height)]]
        accessories.append({
            "name": name, "role": "weapon" if entity["kind"] == "weapon" else "soft",
            "parent_bone": entity.get("parent_bone", "hips"), "pts": pts,
            "static": entity.get("static", False), "rigid": entity.get("rigid", False),
            "entity_kind": entity["kind"],
        })
    return {
        "coord_mode": "normalized", "pose_type": result["pose_type"],
        "joints": copy.deepcopy(result["joints"]), "accessories": accessories,
        "subject_mask": subject_path, "ownership_masks": mask_paths,
        "vision_contract": {
            "version":1,"input_sha256":result["input_sha256"],"image_size":[width,height],
            "entity_confidence": {e["id"]:e["confidence"] for e in result["entities"]},
            "engineering_review_threshold":MIN_ENTITY_CONFIDENCE,
            "unassigned_pixels":0,"subject_excludes_independent_entities":True,
            "semantic_acceptance":"requires_binding_review",
        },
    }
