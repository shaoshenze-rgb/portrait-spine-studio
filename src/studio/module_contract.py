"""V2 complete semantic modules: validate explicit ownership and motion intent.

This is a local data contract, not an automatic image recognizer. Unknown regions,
unsupported constraints and missing hidden material remain reviewable failures.
"""
from __future__ import annotations

import base64
import binascii
import copy
import hashlib
import io
import json
import math
import re
import struct
from pathlib import Path
from typing import Any, Dict, Sequence, Union

import numpy as np
from PIL import Image

from .vision_schema import (VisionValidationError, VisionReviewRequired, _point,
                            _polygon, rasterize_polygons)

MODULE_RULE_VERSION = "complete-modules-5"
MODULE_ROLES = frozenset(("head", "face", "torso", "body", "pelvis", "upper_arm",
    "forearm", "arm", "hand", "upper_leg", "lower_leg", "leg", "foot", "lower_body",
    "weapon", "armor", "hair", "cloth", "cape", "tail", "fish_tail", "fin",
    "effect_group", "pet", "prop", "soft_accessory", "ground"))
HARD_ROLES = frozenset(("head", "face", "torso", "body", "pelvis", "upper_arm",
    "forearm", "arm", "hand", "upper_leg", "lower_leg", "leg", "foot", "lower_body",
    "weapon", "armor"))
BODY_ROLES = HARD_ROLES - {"weapon", "armor"}
SOFT_ROLES = MODULE_ROLES - HARD_ROLES - {"ground", "pet", "prop"}
STATIC_ROLES = frozenset(("ground", "pet", "effect_group", "prop"))
MOTION_PROFILES = frozenset(("breath", "sway", "follow", "soft_bend", "static"))
MIN_MODULE_CONFIDENCE = 0.8  # Engineering review boundary, not calibrated accuracy.
LOCAL_FILL_PX_AT_1080 = 32.0
AUTO_SEAM_UNDERLAP_PX_AT_1080 = 12.0
PNG_SIGNATURE = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])
MAX_MASK_B64_CHARS = 16 * 1024 * 1024
MAX_MASK_PNG_BYTES = 12 * 1024 * 1024
MAX_MASK_PIXELS = 64_000_000


def _id(value: Any, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,47}", value):
        raise VisionValidationError(label + " 必须是小写标识（最多48字符）。")
    return value


def _keys(value: dict, allowed: set, label: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise VisionValidationError(label + " 含未支持字段：" + ", ".join(sorted(unknown)))


def _polygons(value: Any, label: str) -> list:
    if not isinstance(value, list) or not 1 <= len(value) <= 128:
        raise VisionValidationError(label + " 需要1–128个多边形。")
    return [_polygon(poly, label) for poly in value]


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise VisionValidationError(label + " 必须是有限数值。")
    return float(value)


def decode_piece_mask(piece: Dict[str, Any], image_size: Sequence[int]) -> np.ndarray:
    """Decode a SHA-checked original-resolution strictly binary PNG mask."""
    raw_b64 = piece.get("mask_png_base64")
    digest = piece.get("mask_sha256")
    if not isinstance(raw_b64, str) or not raw_b64.strip():
        raise VisionValidationError("mask_png_base64 必须是非空字符串。")
    if len(raw_b64) > MAX_MASK_B64_CHARS:
        raise VisionValidationError("mask_png_base64 超过16MB上限。")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise VisionValidationError("mask_sha256 必须是64位十六进制。")
    width, height = int(image_size[0]), int(image_size[1])
    if width * height > MAX_MASK_PIXELS:
        raise VisionValidationError("图像像素总数超过64,000,000上限。")
    try:
        raw = base64.b64decode(raw_b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise VisionValidationError("mask_png_base64 不是合法Base64。") from exc
    if len(raw) > MAX_MASK_PNG_BYTES:
        raise VisionValidationError("掩码PNG超过12MB上限。")
    actual = hashlib.sha256(raw).hexdigest()
    if actual != digest:
        raise VisionValidationError("mask_sha256 与 PNG 字节不一致。")
    if not raw.startswith(PNG_SIGNATURE) or len(raw) < 33:
        raise VisionValidationError("掩码必须是PNG。")
    chunk_len, chunk_type = struct.unpack(">I4s", raw[8:16])
    if chunk_type != b"IHDR" or chunk_len != 13:
        raise VisionValidationError("掩码缺少有效IHDR。")
    ihdr_w, ihdr_h = struct.unpack(">II", raw[16:24])
    if (ihdr_w, ihdr_h) != (width, height):
        raise VisionValidationError("IHDR尺寸必须等于 image_size。")
    try:
        with Image.open(io.BytesIO(raw)) as im:
            im.load()
            if im.format != "PNG":
                raise VisionValidationError("掩码必须是PNG。")
            if im.mode not in ("L", "1"):
                raise VisionValidationError("掩码必须是L或1位二值PNG。")
            if im.size != (width, height):
                raise VisionValidationError("掩码尺寸必须等于 image_size。")
            arr = np.asarray(im.convert("L"))
    except VisionValidationError:
        raise
    except Exception as exc:
        raise VisionValidationError("mask_png_base64 不是可读的PNG掩码。") from exc
    if not set(np.unique(arr).tolist()).issubset({0, 255}):
        raise VisionValidationError("掩码必须是严格二值 {0,255}。")
    return arr == 255


def module_has_active_host(module: Dict[str, Any], by_id: Dict[str, Dict[str, Any]]) -> bool:
    cur = module.get("parent")
    while cur is not None:
        ancestor = by_id.get(cur)
        if ancestor is None:
            return False
        if ancestor.get("motion") not in ("follow", "static"):
            return True
        cur = ancestor.get("parent")
    return False


def piece_uses_mask(piece: Dict[str, Any]) -> bool:
    return bool(piece.get("mask_png_base64"))


def contract_fingerprint(contract: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",",":"),
                                    ensure_ascii=False).encode("utf-8")).hexdigest()


def validate_hidden_joint_region(parent_mask, targets, point, source_alpha):
    """An explicit, opaque connected covering corridor, NOT generated anatomy.

    No dilation/nearest fill or extrapolation authorizes an anchor. The exact
    missing region must later remain covered throughout the actual animation.
    """
    from src.pipeline.split_parts import _components
    region=np.zeros_like(parent_mask,dtype=bool)
    for target in targets:
        if target.shape!=parent_mask.shape:raise VisionReviewRequired('遮挡joint掩码尺寸不符。')
        region|=target
    x,y=int(round(float(point[0]))),int(round(float(point[1])))
    h,w=region.shape
    if (not region.any() or not (0<=x<w and 0<=y<h) or not region[y,x]
            or (region & (source_alpha!=255)).any()):
        raise VisionReviewRequired('遮挡joint需要端点位于已登记的不透明遮挡域，不能借背景或半透明边缘。')
    support=region|parent_mask
    yy,xx=np.nonzero(support)
    x0,x1=max(0,int(xx.min())-1),min(w,int(xx.max())+2)
    y0,y1=max(0,int(yy.min())-1),min(h,int(yy.max())+2)
    labels=_components(support[y0:y1,x0:x1])
    component=labels==labels[y-y0,x-x0]
    if not (component & parent_mask[y0:y1,x0:x1]).any():
        raise VisionReviewRequired('遮挡joint域未与父模块真实可见材质连通，不能授权悬空连接。')
    return region


def validate_hidden_attachment_region(parent_mask, targets, point):
    """Validate the geometry of a declared Canvas parent-absence corridor.

    Foreground identity/alpha is checked separately by compiler and runtime.
    This is not an assertion that the corridor stays covered during motion.
    """
    from src.pipeline.split_parts import _components
    region = np.zeros_like(parent_mask, dtype=bool)
    for target in targets:
        if target.shape != parent_mask.shape:
            raise VisionReviewRequired('遮挡附件区域尺寸不符。')
        region |= target
    x, y = int(round(point[0])), int(round(point[1])); h, w = region.shape
    if not region.any() or not (0 <= x < w and 0 <= y < h) or not region[y, x]:
        raise VisionReviewRequired('隐藏附件锚点必须位于显式登记区域。')
    support = region | parent_mask; yy, xx = np.nonzero(support)
    x0, x1 = max(0, int(xx.min())-1), min(w, int(xx.max())+2)
    y0, y1 = max(0, int(yy.min())-1), min(h, int(yy.max())+2)
    labels = _components(support[y0:y1, x0:x1])
    connected = labels == labels[y-y0, x-x0]
    if not (connected & parent_mask[y0:y1, x0:x1]).any():
        raise VisionReviewRequired('隐藏附件区域没有连到父模块材质，不能授权悬空连接。')
    return region


def validate_module_contract(payload: Any, *, input_sha256: str,
                             image_size: Sequence[int], material_mode: str = 'source_sliced') -> Dict[str, Any]:
    if material_mode not in ('source_sliced', 'canvas_layers'):
        raise VisionValidationError('未知内部材质验证模式。')
    if not isinstance(payload, dict) or type(payload.get("version")) is not int or payload["version"] != 2:
        raise VisionValidationError("完整模块生产合同必须为 version=2；旧V1不能静默迁移。")
    result = copy.deepcopy(payload)
    if (not isinstance(result.get("input_sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", result["input_sha256"])
            or result["input_sha256"] != input_sha256):
        raise VisionValidationError("模块合同的图片SHA不匹配。")
    size = result.get("image_size")
    if (not isinstance(size, list) or len(size) != 2
            or any(type(v) is not int or v < 1 for v in size) or tuple(size) != tuple(image_size)):
        raise VisionValidationError("模块合同的图像尺寸不匹配。")
    if result.get("coord_mode") != "normalized":
        raise VisionValidationError("模块坐标必须为 normalized。")
    issues = result.setdefault("input_issues", [])
    if not isinstance(issues, list) or len(issues) > 32:
        raise VisionValidationError("input_issues必须是明确输入缺陷列表。")
    issue_ids = set()
    for issue in issues:
        if not isinstance(issue, dict):
            raise VisionValidationError("输入缺陷必须是对象。")
        _keys(issue, {"id","kind","polygons","notes"}, "input_issue")
        iid = _id(issue.get("id"), "input_issue.id")
        if iid in issue_ids:
            raise VisionValidationError("输入缺陷id重复。")
        issue_ids.add(iid)
        if issue.get("kind") not in ("background_residue","fringe","source_damage","precomposited_overlap"):
            raise VisionValidationError("未支持的输入缺陷类型。")
        issue["polygons"] = _polygons(issue.get("polygons"), iid)
        if not isinstance(issue.get("notes"), str) or not issue["notes"].strip():
            raise VisionValidationError("输入缺陷必须解释实际看见的问题。")
    if issues:
        if any(issue['kind']=='precomposited_overlap' for issue in issues):
            message = ("输入中需独立运动的效果／实体已预合成在同一像素；缺少原分层或可信分离材质。"
                       "保留原图待审，不用描边、补宽、删除效果或自动同动冒充完整模块。")
        else:
            message = "输入存在已明确定位的背景残留／轮廓问题；保留原图待审，不作为模块或未覆盖兜底。"
        error = VisionReviewRequired(message)
        error.code = "INPUT_REVIEW_REQUIRED"
        error.details = {"input_issues":issues, "input_sha256":input_sha256,
                         "image_size":list(image_size), "module_planning_status":"diagnostic_only"}
        raise error
    if result.get("pose_type") not in ("grounded", "airborne"):
        raise VisionValidationError("模块合同需要明确 grounded/airborne 姿态。")
    plan_mode = result.get("plan_mode", "standard")
    if plan_mode not in ("standard", "unified_upper_body"):
        raise VisionValidationError("plan_mode必须为 standard 或 unified_upper_body。")
    if plan_mode == "unified_upper_body" and material_mode != "canvas_layers":
        raise VisionReviewRequired("头肩一体基础模式仅限已校验材质层（canvas_layers）预览使用。")
    result["plan_mode"] = plan_mode
    if result.get("anatomy") not in ("humanoid_full", "humanoid_bust", "merfolk", "other"):
        raise VisionValidationError("需要明确 anatomy 分类。")
    unknown = result.get("uncertainties")
    if not isinstance(unknown, list) or any(not isinstance(v, str) for v in unknown):
        raise VisionValidationError("uncertainties 必须是字符串列表。")
    if unknown:
        raise VisionReviewRequired("模块规划仍有未决项：" + "；".join(unknown))
    if result["anatomy"] == "other":
        raise VisionReviewRequired("当前未知身体结构没有可验证的生成规则，请先完成模块规划。")
    modules = result.get("modules")
    if not isinstance(modules, list) or not 2 <= len(modules) <= 64:
        raise VisionReviewRequired("完整角色需要明确多个语义模块，不能整图单模块。")
    by_id, pieces, orders = {}, {}, set()
    allowed = {"id", "role", "motion_type", "parent", "pivot", "motion", "pieces",
               "centerline", "confidence", "static_reason", "side", "representation",
               "notes", "locked_root", "motion_scale"}
    for module in modules:
        if not isinstance(module, dict):
            raise VisionValidationError("模块必须是对象。")
        _keys(module, allowed, "module")
        mid = _id(module.get("id"), "module.id")
        if mid in by_id:
            raise VisionValidationError("模块id重复。")
        motion_scale = _number(module.get("motion_scale", 1.0), mid + ".motion_scale")
        if not 0 <= motion_scale <= 1.5: raise VisionValidationError("motion_scale必须在0..1.5。")
        module["motion_scale"] = motion_scale
        role = module.get("role")
        if role not in MODULE_ROLES:
            raise VisionValidationError(mid + " 的role不受支持。")
        kind = module.get("motion_type")
        if kind not in ("rigid", "soft", "static"):
            raise VisionValidationError(mid + " 缺少刚性/柔性/静态类型。")
        if role in HARD_ROLES and kind == "soft":
            raise VisionReviewRequired(mid + " 是硬模块，不能混权弯曲。")
        if kind == "soft" and role not in SOFT_ROLES:
            raise VisionReviewRequired(mid + " 的类型不允许软形变。")
        module["pivot"] = _point(module.get("pivot"), mid + ".pivot")
        if "parent" not in module:
            raise VisionValidationError(mid + " 必须明确parent（根模块为null）。")
        if module["parent"] is not None:
            _id(module["parent"], mid + ".parent")
        if module["parent"] == mid:
            raise VisionValidationError("模块不能以自身为父级。")
        conf = _number(module.get("confidence"), mid + ".confidence")
        if not 0 <= conf <= 1:
            raise VisionValidationError("confidence必须在0..1。")
        if conf < MIN_MODULE_CONFIDENCE:
            raise VisionReviewRequired(mid + " 语义归属未达到统一待审阈值。")
        side = module.get("side")
        if side not in (None, "L", "R"):
            raise VisionValidationError("side只能是L/R或省略。")
        default_motion = ("static" if kind == "static" else "soft_bend" if kind == "soft"
                          else "follow" if role in ("weapon", "armor", "face")
                          else "breath" if role in ("torso", "body", "pelvis") and module["parent"] is None
                          else "sway")
        module.setdefault("motion", default_motion)
        profile = module["motion"]
        if profile not in MOTION_PROFILES:
            raise VisionValidationError("motion只能引用通用类型预设，不能传自定义曲线。")
        if kind == "static":
            if profile != "static" or not isinstance(module.get("static_reason"), str) or not module["static_reason"].strip():
                raise VisionReviewRequired("静态模块需要明确语义原因。")
            grounded_lower = (result.get("pose_type") == "grounded" and role == "lower_body" and module["parent"] is None)
            if role not in STATIC_ROLES and role != "foot" and not grounded_lower:
                raise VisionReviewRequired("人体主体不能声明静态来绕过动作验收。")
        elif profile == "static":
            raise VisionReviewRequired("需要运动的模块不能将motion改成static。")
        if kind == "soft" and profile != "soft_bend":
            raise VisionReviewRequired("软模块必须使用统一受限soft_bend规则。")
        if kind != "soft" and profile == "soft_bend":
            raise VisionReviewRequired("硬模块不能选择soft_bend。")
        if role == "armor":
            if kind != "rigid" or profile not in ("sway", "follow"):
                raise VisionReviewRequired(mid + " 护甲必须是rigid且motion为sway或follow。")
        if role == "weapon":
            if kind != "rigid" or profile != "follow":
                raise VisionReviewRequired(mid + " 武器必须是rigid且motion为follow。")
        if "locked_root" in module:
            if type(module["locked_root"]) is not bool:
                raise VisionValidationError("locked_root必须是布尔值。")
        locked = bool(module.get("locked_root"))
        if locked:
            if role != "effect_group":
                raise VisionValidationError("locked_root仅用于effect_group。")
            if kind != "soft" or profile != "soft_bend":
                raise VisionReviewRequired("locked_root仅用于soft_bend软模块。")
            if module["parent"] is None:
                raise VisionReviewRequired("locked_root需要明确活动宿主父模块。")
        if role == "effect_group" and kind == "soft":
            if profile != "soft_bend" or not locked:
                raise VisionReviewRequired("effect_group软链必须声明locked_root。")
            if module["parent"] is None:
                raise VisionReviewRequired("effect_group软链需要明确活动宿主父模块。")
        representation = module.setdefault("representation", "mesh" if kind == "soft" else "region")
        if representation not in ("region", "mesh") or (kind == "soft" and representation != "mesh"):
            raise VisionValidationError("模块表示不受支持。")
        if kind == "soft":
            line = module.get("centerline")
            if not isinstance(line, list) or not 2 <= len(line) <= 5:
                raise VisionReviewRequired("软模块需要2–5个结构中心线点。")
            line = [_point(v, mid + ".centerline") for v in line]
            if any(math.dist(a, b) < 1e-5 for a, b in zip(line, line[1:])):
                raise VisionReviewRequired("软模块中心线退化。")
            if math.dist(line[0], module["pivot"]) > 1e-6:
                raise VisionReviewRequired("软模块中心线根点须与pivot一致。")
            module["centerline"] = line
        elif module.get("centerline"):
            raise VisionValidationError("硬模块不使用弯曲中心线。")
        members = module.get("pieces")
        if not isinstance(members, list) or not 1 <= len(members) <= 32:
            raise VisionValidationError("模块需要明确1–32个可见绘制piece。")
        for piece in members:
            if not isinstance(piece, dict):
                raise VisionValidationError("piece必须是对象。")
            _keys(piece, {"id", "polygons", "draw_order", "mask_png_base64", "mask_sha256", "notes"}, "piece")
            pid = _id(piece.get("id"), "piece.id")
            if pid in pieces:
                raise VisionValidationError("piece.id全局必须唯一。")
            order = piece.get("draw_order")
            if type(order) is not int or not -100000 <= order <= 100000 or order in orders:
                raise VisionValidationError("piece.draw_order须为唯一整数。")
            has_poly = "polygons" in piece
            has_mask_field = "mask_png_base64" in piece or "mask_sha256" in piece
            if has_mask_field and not ("mask_png_base64" in piece and "mask_sha256" in piece):
                raise VisionValidationError("mask_png_base64 与 mask_sha256 必须同时提供。")
            if not has_poly and not has_mask_field:
                raise VisionValidationError("piece必须提供polygons或mask_png_base64+mask_sha256。")
            if has_mask_field:
                decode_piece_mask(piece, size)
            if has_poly:
                piece["polygons"] = _polygons(piece.get("polygons"), pid)
            if "notes" in piece and (not isinstance(piece.get("notes"), str) or not piece["notes"].strip()):
                raise VisionValidationError("piece.notes必须是非空字符串。")
            orders.add(order)
            pieces[pid] = (mid, piece)
        by_id[mid] = module
    # Explicit acyclic movement hierarchy. Soft chains remain terminal modules.
    for mid, module in by_id.items():
        seen, cur = {mid}, module["parent"]
        while cur is not None:
            if cur not in by_id or cur in seen:
                raise VisionValidationError("模块父级不存在或构成循环。")
            seen.add(cur)
            if by_id[cur]["motion_type"] == "soft":
                raise VisionReviewRequired("暂不支持将其他模块挂到弯曲软链上，请明确可验证连接。")
            if module["motion_type"] == "static" and by_id[cur]["motion_type"] != "static":
                raise VisionReviewRequired("声明静态的模块不能继承活动父级。")
            cur = by_id[cur]["parent"]
        if module["motion"] == "follow":
            cur = module["parent"]
            active_parent = False
            while cur is not None:
                ancestor = by_id[cur]
                if ancestor["motion"] not in ("follow", "static"):
                    active_parent = True
                    break
                cur = ancestor["parent"]
            if not active_parent:
                raise VisionReviewRequired("follow模块没有活动父级，不能假报可见动作。")
        if module.get("locked_root") or (module["role"] == "effect_group" and module["motion_type"] == "soft"):
            if not module_has_active_host(module, by_id):
                raise VisionReviewRequired(mid + " 的locked_root没有活动宿主父级。")
    roles = {m["role"] for m in modules}
    active_hard = [m for m in modules if m["motion_type"] == "rigid"
                   and m["role"] in BODY_ROLES and m["motion"] in ("breath", "sway")]
    active_any = [m for m in modules if m["motion"] in ("breath", "sway", "soft_bend")]
    has_lower = bool({"lower_body", "upper_leg", "lower_leg", "leg"} & roles)
    if plan_mode == "unified_upper_body":
        if not ({"torso", "body"} & roles):
            raise VisionReviewRequired("头肩一体基础模式仍需明确躯干主体模块。")
        if result["anatomy"] == "humanoid_full" and result["pose_type"] == "grounded" and not has_lower:
            raise VisionReviewRequired("站姿全身人物没有真实下身固定模块，不能放行。")
        if result["anatomy"] == "merfolk" and "fish_tail" not in roles:
            raise VisionReviewRequired("人鱼需要主鱼尾模块，不能强造双腿。")
        if len(active_hard) < 1:
            raise VisionReviewRequired("头肩一体基础模式至少需要一个活动刚性躯干模块。")
        if len(active_any) < 2 and not (result["pose_type"] == "grounded" and has_lower):
            raise VisionReviewRequired("头肩一体基础模式至少需要两个活动模块（或站姿下盘+上半身呼吸），不能整图单骨。")
    else:
        if not ({"head", "face"} & roles) or not ({"torso", "body"} & roles):
            raise VisionReviewRequired("人物需要实际头部与躯干模块。")
        if result["anatomy"] == "humanoid_full" and not has_lower:
            raise VisionReviewRequired("全身人物没有真实下身模块，不能删除后放行。")
        if result["anatomy"] == "merfolk" and "fish_tail" not in roles:
            raise VisionReviewRequired("人鱼需要主鱼尾模块，不能强造双腿。")
        if len(active_hard) < 2:
            raise VisionReviewRequired("需要多个硬身体模块的相对轻摆，不能整图同骨或仅靠软附件运动。")
    connections = result.get("connections")
    if not isinstance(connections, list) or len(connections) > 128:
        raise VisionValidationError("connections必须是有界列表。")
    edge_map, foot_modules, conn_ids = {}, set(), set()
    for conn in connections:
        if not isinstance(conn, dict):
            raise VisionValidationError("connection必须是对象。")
        _keys(conn, {"id", "type", "a", "b", "point", "chain", "notes", "hidden_by"}, "connection")
        cid = _id(conn.get("id"), "connection.id")
        if cid in conn_ids:
            raise VisionValidationError("connection.id重复。")
        conn_ids.add(cid)
        kind, a, b = conn.get("type"), conn.get("a"), conn.get("b")
        if kind not in ("joint", "grip", "attachment", "foot_anchor"):
            raise VisionReviewRequired("当前连接类型不受支持（双端链/双手闭环须明确规划）。")
        if a not in by_id or b not in by_id or a == b:
            raise VisionValidationError("连接需要不同且已存在的a/b模块。")
        conn["point"] = _point(conn.get("point"), cid + ".point")
        if kind == "foot_anchor":
            chain = conn.get("chain")
            if (not isinstance(chain, list) or len(chain) != 2 or chain[0] != a
                    or any(k not in by_id for k in chain) or chain[0] == chain[1]
                    or by_id[chain[1]]["parent"] != chain[0]):
                raise VisionReviewRequired("脚锚需要明确两个刚体肢段链。")
            if any(by_id[k]["motion_type"] != "rigid" for k in chain):
                raise VisionReviewRequired("脚锚链必须保形，不能使用软骨或伸长。")
            if by_id[b]["role"] != "foot" or by_id[b]["motion_type"] != "static" or by_id[b]["parent"] is not None:
                raise VisionReviewRequired("脚锚需要独立静态foot模块。")
            if math.dist(conn["point"], by_id[b]["pivot"]) > 1e-6:
                raise VisionReviewRequired("脚锚点必须等于静态foot的真实pivot。")
            foot_modules.add(b)
            continue
        if by_id[b]["parent"] != a:
            raise VisionReviewRequired("连接a必须是b的显式运动父级，不按邻距代替。")
        if (a, b) in edge_map:
            raise VisionValidationError("同一父子模块有重复连接。")
        edge_map[a, b] = kind
        if math.dist(conn["point"], by_id[b]["pivot"]) > 1e-6:
            raise VisionReviewRequired("连接点须与子模块pivot一致。")
        if kind == "grip" and (by_id[b]["motion"] != "follow" or by_id[b]["motion_type"] != "rigid"):
            raise VisionReviewRequired("刚性握持子件必须follow，共同变换保持接触。")
    for mid, m in by_id.items():
        if m["parent"] is not None and (m["parent"], mid) not in edge_map:
            raise VisionReviewRequired(mid + " 缺少显式父子接触声明。")
        if m["role"] == "weapon" and not any(c["type"] == "grip" and c.get("b") == mid for c in connections):
            raise VisionReviewRequired("武器需要明确grip连接，不独立漂移。")
        if m["motion_type"] == "static" and m["role"] == "foot" and mid not in foot_modules:
            raise VisionReviewRequired("人体足部静止必须由明确脚锚解释。")
    grounded_lower_anchors = {
        mid for mid, m in by_id.items()
        if m["role"] == "lower_body" and m["motion_type"] == "static" and m["parent"] is None
        and any(
            child["parent"] == mid and child["role"] in ("torso", "body")
            and child["motion_type"] == "rigid" and child["motion"] in ("breath", "sway")
            for child in modules
        )
    }
    for mid, m in by_id.items():
        if m["motion_type"] == "static" and m["role"] == "lower_body" and mid not in grounded_lower_anchors:
            raise VisionReviewRequired("静态贴地 lower_body 必须挂有活动上半身 torso/body，不能整身静止。")
    if (result["pose_type"] == "grounded" and result["anatomy"] == "humanoid_full"
            and not foot_modules and not grounded_lower_anchors):
        raise VisionReviewRequired("站立全身必须明确刚体足部锚定或静态贴地下半身锚定；不能静默丢掉接地。")
    occlusions = result.get("occlusions")
    if not isinstance(occlusions, list) or len(occlusions) > 128:
        raise VisionValidationError("occlusions必须是有界列表。")
    used_pairs = set()
    for oc in occlusions:
        if not isinstance(oc, dict):
            raise VisionValidationError("occlusion必须是对象。")
        _keys(oc, {"front", "back", "polygons", "completion", "completion_polygons", "derive_band",
                   "band_depth_px_at_1080", "notes", "provided"}, "occlusion")
        front, back = oc.get("front"), oc.get("back")
        if front not in pieces or back not in pieces or front == back or (front, back) in used_pairs:
            raise VisionValidationError("遮挡piece引用不正确或重复。")
        if pieces[front][1]["draw_order"] <= pieces[back][1]["draw_order"]:
            raise VisionValidationError("遮挡front必须画在back之前景层。")
        oc["polygons"] = _polygons(oc.get("polygons"), "occlusion")
        oc.setdefault("completion", "none")
        if oc["completion"] not in ("none", "local", "provided"):
            raise VisionReviewRequired("补全方式只支持none/local/provided。")
        if oc['completion']=='provided':
            if not oc.get('provided'):
                error=VisionReviewRequired('provided补全需要真实Canvas结果；请按主链提交并回接，不以本地补色替代。')
                error.code='CANVAS_COMPLETION_REQUIRED'
                error.details={'front':front,'back':back,'input_sha256':input_sha256,
                               'canvas_operation':'edit-image','resubmit_automatically':False}
                raise error
            from src.mivo.canvas_assets import decode_canvas_completion
            decode_canvas_completion(oc.get('provided'),image_size=size,input_sha256=input_sha256)
            if oc.get('derive_band') or 'band_depth_px_at_1080' in oc:
                raise VisionValidationError('provided使用真实Canvas材质，不走local推导深度。')
        elif 'provided' in oc:
            raise VisionValidationError('只有provided补全才能携带Canvas材质。')
        if "derive_band" in oc:
            if type(oc["derive_band"]) is not bool:
                raise VisionValidationError("derive_band必须是布尔值。")
            if oc["derive_band"] and (oc["completion"] != "local" or "completion_polygons" in oc):
                raise VisionValidationError("derive_band仅用于local且不得同时指定completion_polygons。")
        if "band_depth_px_at_1080" in oc:
            depth = _number(oc.get("band_depth_px_at_1080"), "band_depth_px_at_1080")
            if not 4.0 <= depth <= 48.0:
                raise VisionValidationError("band_depth_px_at_1080必须在4.0..48.0。")
            oc["band_depth_px_at_1080"] = depth
        if "completion_polygons" in oc:
            if oc["completion"] not in ("local","provided"):
                raise VisionValidationError("仅local/provided补全可声明completion_polygons。")
            oc["completion_polygons"] = _polygons(oc["completion_polygons"], "completion_polygons")
        used_pairs.add((front, back))
    for conn in connections:
        if 'hidden_by' not in conn:
            continue
        refs=conn['hidden_by'];a,b=conn['a'],conn['b']
        canvas_attachment = (material_mode == 'canvas_layers' and conn['type'] == 'attachment'
            and by_id[b]['parent'] == a and by_id[a]['motion_type'] == 'rigid'
            and by_id[a]['role'] in BODY_ROLES-{'foot'}
            and by_id[b]['motion_type'] == 'soft'
            and by_id[b]['role'] in SOFT_ROLES-{'effect_group','pet','prop','ground'})
        rigid_joint = (conn['type'] == 'joint' and by_id[b]['parent'] == a
            and all(by_id[mid]['motion_type'] == 'rigid' and by_id[mid]['role'] in BODY_ROLES-{'foot'} for mid in (a,b)))
        if not (canvas_attachment or rigid_joint):
            raise VisionReviewRequired('hidden_by仅适用于刚体身体父子joint，不能替代抓握、脚锚或软体连接。')
        if not isinstance(refs,list) or not 1<=len(refs)<=8:
            raise VisionValidationError('hidden_by必须是1..8个既有遮挡pair引用。')
        seen_refs=set()
        for ref in refs:
            if not isinstance(ref,dict):raise VisionValidationError('hidden_by引用必须是对象。')
            _keys(ref,{'front','back'},'hidden_by')
            pair=(_id(ref.get('front'),'hidden_by.front'),_id(ref.get('back'),'hidden_by.back'))
            if pair not in used_pairs or pair in seen_refs:
                raise VisionValidationError('hidden_by必须唯一引用已有occlusion，不能凭notes或任意前景授权。')
            seen_refs.add(pair)
            front_mid=pieces[pair[0]][0];back_mid=pieces[pair[1]][0]
            if not piece_uses_mask(pieces[pair[1]][1]):
                raise VisionReviewRequired('遮挡joint目前要求父piece的精确可见掩码，避免多边形重叠声明歧义。')
            fm=by_id[front_mid]
            if (back_mid!=a or (front_mid!=b and (fm['parent']!=a or (a,front_mid) not in edge_map))
                    or fm['role'] in ('weapon','effect_group','pet','prop','foot')):
                raise VisionReviewRequired('隐藏joint只能由本子件或有真实连接的同级身体/衣物遮挡，不能借第三者前景。')
    result["rule_version"] = MODULE_RULE_VERSION
    refinements = result.get("boundary_refinements", [])
    if not isinstance(refinements, list) or len(refinements) > 128:
        raise VisionValidationError("boundary_refinements必须是有界列表。")
    refine_ids = set()
    for item in refinements:
        if not isinstance(item, dict):
            raise VisionValidationError("boundary refinement必须是对象。")
        _keys(item, {"id", "a", "b", "axis", "polygons", "notes"}, "boundary_refinement")
        rid = _id(item.get("id"), "boundary_refinement.id")
        if rid in refine_ids:
            raise VisionValidationError("boundary refinement id重复。")
        refine_ids.add(rid)
        if item.get("a") not in pieces or item.get("b") not in pieces or item["a"] == item["b"]:
            raise VisionValidationError("贴边必须明确两个已声明的不同piece。")
        if item.get("axis") not in ("x", "y"):
            raise VisionValidationError("贴边axis必须为x或y。")
        item["polygons"] = _polygons(item.get("polygons"), rid)
        if not isinstance(item.get("notes"), str) or not item["notes"].strip():
            raise VisionValidationError("贴边需要说明已确认的实际语义边界。")
    return result


# Only an analysis-selected pair and corridor can use image edges. This never
# discovers semantic modules from colors or changes ownership outside that pair.
BOUNDARY_REFINE_PX_AT_1080 = 3.0


def _refine_boundaries(contract: dict, rgba: np.ndarray, visible: dict) -> tuple:
    from src.pipeline.split_parts import _components
    h, w = rgba.shape[:2]
    fg_y = np.nonzero(rgba[..., 3])[0]
    radius = max(1, int(round(BOUNDARY_REFINE_PX_AT_1080 *
                            max(1, int(fg_y.max()-fg_y.min()+1)) / 1080)))
    evidence = []
    used_corridors = np.zeros((h, w), dtype=bool)
    def topology(mask):
        ys, xs = np.nonzero(mask)
        if not len(ys):
            return (0, 0)
        crop = np.pad(mask[ys.min():ys.max()+1, xs.min():xs.max()+1], 1)
        return (len(np.unique(_components(crop)[crop])),
                len(np.unique(_components(~crop)[~crop])))
    for item in contract.get("boundary_refinements", []):
        aid, bid = item["a"], item["b"]
        before_a, before_b = visible[aid].copy(), visible[bid].copy()
        corridor = rasterize_polygons(item["polygons"], w, h)
        if (corridor & used_corridors).any():
            raise VisionReviewRequired("贴边走廊不能交叠后逐次追逐新边界；请保留相对原归属的单次有界精化。")
        used_corridors |= corridor
        transposed = item["axis"] == "y"
        a, b = before_a.copy(), before_b.copy()
        rgb = rgba[..., :3].astype(float)
        if transposed:
            a, b, corridor, rgb = a.T, b.T, corridor.T, rgb.transpose(1, 0, 2)
        rows = np.nonzero(corridor.any(axis=1))[0]
        changes, strengths, ambiguous = [], [], 0
        for y in rows:
            contact = ((a[y, :-1] & b[y, 1:]) | (b[y, :-1] & a[y, 1:]))
            boundaries = np.flatnonzero(contact & corridor[y, :-1] & corridor[y, 1:])+1
            if len(boundaries) != 1:
                raise VisionReviewRequired("贴边走廊必须逐行只跨一个既定pair边界；交叉/多段需重新规划。")
            original = int(boundaries[0]); left_is_a = bool(a[y, original-1])
            candidates = []
            for x in range(max(1, original-radius), min(a.shape[1], original+radius+1)):
                lo, hi = min(original, x)-1, max(original, x)+1
                if not (corridor[y, lo:hi].all() and (a[y, lo:hi] | b[y, lo:hi]).all()):
                    continue
                edge = float(np.linalg.norm(rgb[y, x]-rgb[y, x-1]))
                candidates.append((edge-1.5*abs(x-original), x, edge))
            if not candidates:
                raise VisionReviewRequired("贴边走廊没有可验证的局部pair候选。")
            candidates.sort(reverse=True); _, selected, strength = candidates[0]
            # Competing strong edges are ambiguous texture/outline cues.
            if any(abs(x-selected)>1 and edge>=max(12., strength*.9)
                   for _, x, edge in candidates[1:]):
                ambiguous += 1
            strengths.append(strength)
            changes.append((int(y), original, selected, left_is_a))
        if len(changes)<3 or float(np.median(strengths))<12. or ambiguous>len(changes)*.25:
            raise VisionReviewRequired("局部语义边界缺少清晰唯一原图边缘；保持原归属待审。")
        offsets = np.array([row[2]-row[1] for row in changes])
        if len(offsets)>1 and np.max(np.abs(np.diff(offsets)))>1:
            raise VisionReviewRequired("局部贴边偏移不连续，不能追逐纹理碎边。")
        for y, original, selected, left_is_a in changes:
            lo, hi = sorted((original, selected))
            belongs_a = left_is_a if selected>original else not left_is_a
            a[y, lo:hi] = belongs_a; b[y, lo:hi] = not belongs_a
        if transposed:
            a, b = a.T, b.T
        changed = a != before_a
        if topology(a)!=topology(before_a) or topology(b)!=topology(before_b):
            raise VisionReviewRequired("贴边会改变既定模块连通/镂空拓扑，保留原归属待审。")
        if not np.array_equal(a | b, before_a | before_b) or (a & b).any():
            raise VisionReviewRequired("贴边改变前景覆盖，不能锁定。")
        visible[aid], visible[bid] = a, b
        evidence.append({"id":item["id"], "a":aid, "b":bid,
            "maxDistancePx":radius, "changedPixels":int(changed.sum()),
            "maxActualDistancePx":int(np.max(np.abs(offsets))),
            "medianEdgeStrength":float(np.median(strengths)),
            "topologyPreserved":True, "coveragePreserved":True,
            "visualReview":"pending", "_before":{aid:before_a, bid:before_b},
            "_after":{aid:a.copy(), bid:b.copy()}, "_changed":changed})
    return visible, evidence



def _modules_topologically_linked(back_m, front_m, module_parent, connected_modules):
    """True when two modules share a real attachment: direct parent/child or an
    explicit contract connection. Unconnected layer crossings must not get
    automatic seam underlap."""
    a, b = back_m["id"], front_m["id"]
    if module_parent.get(a) == b or module_parent.get(b) == a:
        return True
    return frozenset((a, b)) in connected_modules


def _auto_seam_underlaps(entries, visible, rgba, hidden, body_height, explicit_regions,
                         module_parent, connected_modules):
    """Narrow hidden underlap along contacting piece seams; never changes visible pixels.

    Only module pairs with a real topological connection (parent/child or an
    explicit contract connection) may receive auto underlap. Unrelated layer
    crossings (e.g. a weapon blade sweeping over an unconnected cloth/ribbon)
    must not get underlap painted behind them -- once the front piece swings
    away the painted block would float in empty background.
    """
    from src.pipeline.split_parts import _chamfer, _dilate
    h, w = rgba.shape[:2]
    seam_depth = max(4, int(round(AUTO_SEAM_UNDERLAP_PX_AT_1080 * max(1, body_height) / 1080.0)))
    opaque = rgba[..., 3] == 255
    meta = []
    for i, (back_m, back_p) in enumerate(entries):
        back = back_p["id"]
        vb = visible[back]
        if not vb.any():
            continue
        contact_seed = _dilate(vb, 1)
        for front_m, front_p in entries[i + 1:]:
            front = front_p["id"]
            vf = visible[front]
            if not (contact_seed & vf).any():
                continue
            pair = (front, back)
            # Explicit none is an opt-out for the pair. A local repair overrides
            # only its declared region, not unrelated seams along the pair.
            if pair in explicit_regions and explicit_regions[pair] is None:
                continue
            if back_m is not front_m and not _modules_topologically_linked(
                    back_m, front_m, module_parent, connected_modules):
                continue
            eligible = vf & opaque & ~vb
            needed = eligible & ~hidden[back]
            if pair in explicit_regions:
                needed &= ~explicit_regions[pair]
            if not needed.any():
                continue
            yy, xx = np.nonzero(eligible | vb)
            y0, y1 = max(0, int(yy.min()) - seam_depth - 1), min(h, int(yy.max()) + seam_depth + 2)
            x0, x1 = max(0, int(xx.min()) - seam_depth - 1), min(w, int(xx.max()) + seam_depth + 2)
            sl = np.s_[y0:y1, x0:x1]
            distance = _chamfer(vb[sl], eligible[sl], seam_depth)
            target = np.zeros_like(eligible)
            target[sl] = needed[sl] & (distance <= seam_depth)
            if not target.any():
                continue
            hidden[back] |= target
            meta.append({
                "front": front, "back": back, "completionMode": "local",
                "autoSeamUnderlap": True, "visualReview": "pending",
                "maxDepthPx": seam_depth, "materializedPixels": int(target.sum()),
            })
    return meta, seam_depth


def materialize_module_contract(payload: Dict[str, Any], *, image_path: Union[str, Path],
                                mask_dir: Union[str, Path]) -> Dict[str, Any]:
    source = Path(image_path).resolve()
    with Image.open(source) as image:
        rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    h, w = rgba.shape[:2]
    contract = validate_module_contract(payload,
        input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), image_size=(w, h))
    foreground = rgba[..., 3] > 0
    entries = [(m, p) for m in contract["modules"] for p in m["pieces"]]
    entries.sort(key=lambda q: q[1]["draw_order"])
    by_piece = {p["id"]: (m, p) for m, p in entries}
    claims = {}
    for _, p in entries:
        if piece_uses_mask(p):
            mask = decode_piece_mask(p, (w, h))
            if (mask & ~foreground).any():
                raise VisionReviewRequired("模块掩码包含非前景：" + p["id"])
            claims[p["id"]] = mask
        else:
            claims[p["id"]] = rasterize_polygons(p["polygons"], w, h) & foreground
    zones = {(oc["front"], oc["back"]): rasterize_polygons(oc["polygons"], w, h)
             for oc in contract["occlusions"]}
    visible = {pid: mask.copy() for pid, mask in claims.items()}
    for i, (_, back) in enumerate(entries):
        for _, front in entries[i + 1:]:
            overlap = claims[back["id"]] & claims[front["id"]]
            if not overlap.any():
                continue
            zone = zones.get((front["id"], back["id"]))
            if zone is None or (overlap & ~zone).any():
                raise VisionReviewRequired("可见模块边界重叠而无完整显式遮挡声明：" + front["id"] + "/" + back["id"])
            visible[back["id"]] &= ~overlap
    covered = np.zeros((h, w), dtype=bool)
    for mid, module in ((m["id"], m) for m in contract["modules"]):
        own = np.zeros((h, w), dtype=bool)
        for piece in module["pieces"]:
            if not visible[piece["id"]].any():
                raise VisionReviewRequired("声明可见piece没有可见像素：" + piece["id"])
            own |= visible[piece["id"]]
        if not own.any():
            raise VisionReviewRequired("模块没有可见像素：" + mid)
    for mask in visible.values():
        if (covered & mask).any():
            raise VisionReviewRequired("最终可见归属仍重叠，不能自动选择。")
        covered |= mask
    missing = foreground & ~covered
    if missing.any():
        raise VisionReviewRequired("有 " + str(int(missing.sum())) + " 个前景像素未归属，不最近邻填给人体。")
    visible, refinement_evidence = _refine_boundaries(contract, rgba, visible)
    out = Path(mask_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for evidence in refinement_evidence:
        evidence_dir = out / ("refinement-" + evidence["id"])
        evidence_dir.mkdir(exist_ok=True)
        for key in ("before", "after"):
            evidence[key + "Masks"] = {}
            for pid, mask in evidence.pop("_"+key).items():
                path = evidence_dir / (pid + "-" + key + ".png")
                Image.fromarray(mask.astype(np.uint8)*255).save(path)
                evidence[key + "Masks"][pid] = {"path":str(path),
                    "sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
        path = evidence_dir / "changed.png"
        Image.fromarray(evidence.pop("_changed").astype(np.uint8)*255).save(path)
        evidence["changedMask"] = {"path":str(path),
            "sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
    mask_paths, hidden_paths, hidden_meta = {}, {}, []
    hidden = {pid: np.zeros((h, w), dtype=bool) for pid in claims}
    provided_images={}
    body = np.zeros((h, w), dtype=bool)
    for module, piece in entries:
        if module["role"] in BODY_ROLES or module["role"] == "fish_tail":
            body |= visible[piece["id"]]
    body_ys = np.nonzero(body)[0]
    body_height = int(body_ys.max()-body_ys.min()+1) if len(body_ys) else h
    band_depth = max(1, int(round(LOCAL_FILL_PX_AT_1080*body_height/1080)))
    for oc in contract["occlusions"]:
        back, front = oc["back"], oc["front"]
        zone = zones[front, back]
        band_report = None
        # Register the whole declared back-material domain BEFORE completion.
        # A narrow completed subset (or completion=none) is not evidence that
        # the rest of a covered limb can never become visible.
        source_hidden_target = zone & visible[front] & ~visible[back]
        if not piece_uses_mask(by_piece[back][1]):
            source_hidden_target &= claims[back]
        if oc["completion"] == "local":
            # Only the explicitly occluding foreground is eligible, never arbitrary neighbours.
            fill_zone = (rasterize_polygons(oc["completion_polygons"],w,h)
                         if oc.get("completion_polygons") else zone)
            if (fill_zone & ~zone).any():
                raise VisionReviewRequired("补全窄带超出已声明遮挡区，不能扩张输入边界。")
            oc_depth = band_depth
            if oc.get("band_depth_px_at_1080") is not None:
                oc_depth = max(1, int(round(float(oc["band_depth_px_at_1080"]) * body_height / 1080)))
            mask_back = piece_uses_mask(by_piece[back][1])
            if mask_back:
                # Exclusive original-resolution masks already equal the visible back
                # claim, so claims[back] & visible[front] is empty. Derive the hidden
                # band from the occluding front without eroding a 2px groove.
                target = fill_zone & visible[front] & ~visible[back]
            else:
                target = fill_zone & claims[back] & visible[front] & ~visible[back]
            if not target.any():
                raise VisionReviewRequired("局部补全声明没有明确的后层被遮区域，不能仅凭相邻关系扩画。")
            if oc.get("derive_band"):
                from src.pipeline.split_parts import _chamfer
                translucent_pixels = int((target & (rgba[...,3] != 255)).sum())
                # Adding material behind a semitransparent source pixel changes
                # its rest alpha/color; that area needs independent source layers.
                eligible = target & (rgba[...,3] == 255)
                if not eligible.any():
                    raise VisionReviewRequired("局部补全被半透明前景遮挡；没有可保留原图RGBA的不透明补全域。")
                yy, xx = np.nonzero(eligible)
                y0,y1 = max(0,int(yy.min())-oc_depth-1), min(h,int(yy.max())+oc_depth+2)
                x0,x1 = max(0,int(xx.min())-oc_depth-1), min(w,int(xx.max())+oc_depth+2)
                sl = np.s_[y0:y1,x0:x1]
                # Paths start only on this back piece's original visible material;
                # they cannot cross another piece, transparent holes, or old fills.
                distance = _chamfer(visible[back][sl], eligible[sl], oc_depth)
                target = np.zeros_like(eligible)
                target[sl] = eligible[sl] & (distance <= oc_depth)
                band_report = {"method":"declared-back-geodesic-band-v1",
                    "maxDepthPx":oc_depth, "eligiblePixels":int(eligible.sum()),
                    "unsupportedTranslucentPixels":translucent_pixels,
                    "materializedPixels":int(target.sum()),
                    "unfilledEligiblePixels":int((eligible & ~target).sum()),
                    "completionStatus":"partial_pending_visual_review"}
                if not target.any():
                    raise VisionReviewRequired("遮挡区与本模块可见材质没有统一深度内的合法补全路径。")
            hidden[back] |= target
        elif oc['completion']=='provided':
            from src.mivo.canvas_assets import decode_canvas_completion
            supplied=decode_canvas_completion(oc['provided'],image_size=(w,h),input_sha256=contract['input_sha256'])
            fill_zone=(rasterize_polygons(oc['completion_polygons'],w,h) if oc.get('completion_polygons') else zone)
            if (fill_zone&~zone).any():raise VisionReviewRequired('Canvas补全域超出声明遮挡区。')
            # Only genuinely occluded, originally opaque pixels may receive new
            # material. Visible RGBA and transparent silhouette stay unchanged.
            target=source_hidden_target&fill_zone&(rgba[...,3]==255)&(supplied[...,3]>0)
            if not target.any():raise VisionReviewRequired('Canvas结果没有覆盖声明的缺失区域，请局部返修。')
            layer=provided_images.setdefault(back,np.zeros_like(rgba))
            conflict=target&(layer[...,3]>0)&np.any(layer!=supplied,axis=2)
            if conflict.any():raise VisionReviewRequired('同一后层的Canvas补全相互冲突，请统一回接版本。')
            layer[target]=supplied[target]
            hidden[back]|=target
        hidden_meta.append({"front": front, "back": back, "completionMode": oc["completion"],
                            "polygons": [[[x*w, y*h] for x, y in poly] for poly in oc["polygons"]],
                            "visualReview": "pending"})
        hidden_meta[-1]["_sourceHiddenTarget"] = source_hidden_target
        if oc['completion']=='provided':
            from src.mivo.canvas_assets import completion_provenance
            hidden_meta[-1]['canvasProvenance']=completion_provenance(oc['provided'])
        if band_report is not None:
            hidden_meta[-1]["bandDerivation"] = band_report
        if oc.get("completion_polygons"):
            hidden_meta[-1]["completionPolygons"] = [[[x*w,y*h] for x,y in poly] for poly in oc["completion_polygons"]]
    explicit_regions = {(oc["front"], oc["back"]):
                        zones[oc["front"], oc["back"]] if oc["completion"] in ("local","provided") else None
                        for oc in contract["occlusions"]}
    module_parent = {m["id"]: m.get("parent") for m in contract["modules"]}
    connected_modules = {
        frozenset((c["a"], c["b"])) for c in contract.get("connections", [])
    }
    auto_meta, auto_depth = _auto_seam_underlaps(
        entries, visible, rgba, hidden, body_height, explicit_regions,
        module_parent, connected_modules)
    hidden_meta.extend(auto_meta)
    provided_paths={}
    for pid,layer in provided_images.items():
        path=out/(pid+'-canvas-completion.png');Image.fromarray(layer).save(path)
        provided_paths[pid]=str(path)
    for pid, mask in visible.items():
        path = out / (pid + "-visible.png")
        Image.fromarray(mask.astype(np.uint8)*255).save(path)
        mask_paths[pid] = str(path)
        if hidden[pid].any():
            path = out / (pid + "-hidden.png")
            Image.fromarray(hidden[pid].astype(np.uint8)*255).save(path)
            hidden_paths[pid] = str(path)
    for meta in hidden_meta:
        if meta["back"] in hidden_paths:
            meta["hiddenMask"] = hidden_paths[meta["back"]]
            meta["hiddenMaskSHA256"] = hashlib.sha256(Path(hidden_paths[meta["back"]]).read_bytes()).hexdigest()
        if meta['back'] in provided_paths:
            meta['providedImage']=provided_paths[meta['back']]
            meta['providedImageSHA256']=hashlib.sha256(Path(meta['providedImage']).read_bytes()).hexdigest()
        if "_sourceHiddenTarget" in meta:
            target = meta.pop("_sourceHiddenTarget")
            remaining = target & ~hidden[meta["back"]]
            for label, mask in (("sourceHiddenTarget",target),("remainingHidden",remaining)):
                path = out / (meta["front"] + "-over-" + meta["back"] + "-" + label + ".png")
                Image.fromarray(mask.astype(np.uint8)*255).save(path)
                meta[label + "Mask"] = str(path)
                meta[label + "MaskSHA256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            meta["remainingHiddenPixels"] = int(remaining.sum())
            meta["sourceHiddenTargetRegistration"] = "original-back-claim-and-occlusion-and-visible-front"
    ownership_report = {"ruleVersion": MODULE_RULE_VERSION, "sourceVersion": 2,
        "sourceSHA256": contract["input_sha256"], "canvas": [w, h],
        "contractSHA256":contract_fingerprint(contract),
        "visibleMaskSHA256":{pid:hashlib.sha256(Path(path).read_bytes()).hexdigest() for pid,path in mask_paths.items()},
        "hiddenMaskSHA256":{pid:hashlib.sha256(Path(path).read_bytes()).hexdigest() for pid,path in hidden_paths.items()},
        "foregroundPixels": int(foreground.sum()), "coveredPixels": int(covered.sum()),
        "unassignedPixels": 0, "visibleOverlapPixels": 0,
        "boundaryRefinements":refinement_evidence,
        "semanticFidelity": "requires_visual_review", "occlusions": hidden_meta}
    (out / "ownership-report.json").write_text(json.dumps(ownership_report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"module_contract": contract, "module_masks": mask_paths,
            "module_hidden_masks": hidden_paths, "module_provided_images":provided_paths,
            "module_projection": ownership_report}

