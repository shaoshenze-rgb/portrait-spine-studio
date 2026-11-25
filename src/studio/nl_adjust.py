"""Natural-language adjustment polisher and minimal-rerun layer classifier.

Follows `.agents/skills/portrait-spine-idle/references/rerun-and-troubleshooting.md`:
- Layer A (`motion`): amplitude, breathing, hair/cloth/tail waves, gain, preset switch
  -> skips preprocess, mivo, split_parts
- Layer B (`rig`): anchor lock, IK bend flip, seam weight / pivot tweaks
  -> skips preprocess, mivo, split_parts
- Layer C (`split`): split new part, occlusion tear/hole fill depth, boundary seam refinement
  -> skips preprocess, mivo (re-runs split_parts -> build_rig -> audit)
"""
from __future__ import annotations

import copy
import math
import re
from typing import Any, Dict, List, Optional, Tuple

from src.pipeline.presets import get_preset, list_presets

LAYER_C_KEYWORDS = (
    "拆分",
    "重新拆",
    "分件",
    "披风",
    "补全",
    "修补",
    "破洞",
    "露底",
    "穿帮",
    "遮挡",
    "撕裂",
    "边界",
    "抠图",
    "inpaint",
    "split",
    "hole",
    "occlud",
    "seam tear",
    "depth",
)

LAYER_B_KEYWORDS = (
    "固定",
    "别动",
    "锁死",
    "锁定",
    "锚点",
    "不滑",
    "滑步",
    "反折",
    "膝盖",
    "手肘",
    "关节",
    "枢轴",
    "权重",
    "拉扯",
    "anchor",
    "lock",
    "pin",
    "ik",
    "bend",
    "flip",
    "pivot",
    "weight",
    "slide",
)

STRONGER_KEYWORDS = (
    "加大",
    "增大",
    "加强",
    "增强",
    "更明显",
    "大一点",
    "多一点",
    "幅度大",
    "更飘",
    "飘一点",
    "活泼",
    "stronger",
    "more",
    "increase",
    "bigger",
    "larger",
)

SOFTER_KEYWORDS = (
    "柔和",
    "减小",
    "减弱",
    "轻一点",
    "小一点",
    "弱一点",
    "克制",
    "更轻",
    "缓一点",
    "舒缓",
    "稳一点",
    "softer",
    "gentler",
    "less",
    "reduce",
    "smaller",
    "subtle",
)

FREEZE_KEYWORDS = (
    "静止",
    "不动",
    "停止",
    "关闭",
    "freeze",
    "still",
    "stop",
)

PRESET_LABELS = {
    "agile_full": "轻盈全身（agile_full）",
    "heavy_weapon_full": "重武器全身（heavy_weapon_full）",
    "bust_half": "半身立绘（bust_half）",
}


def _contains_any(text: str, keywords: Tuple[str, ...]) -> bool:
    return any(kw in text for kw in keywords)


def _scale_terms(terms: List[Tuple[float, int, float]], factor: float) -> List[Tuple[float, int, float]]:
    return [(round(float(a) * factor, 6), int(k), float(ph)) for a, k, ph in terms]


def _clamp_gain(val: float) -> float:
    return round(max(0.2, min(2.5, float(val))), 3)


def _detect_target_groups(low: str) -> List[str]:
    targets: List[str] = []
    if any(k in low for k in ("发", "长发", "刘海", "马尾", "hair", "bang", "mane")):
        targets.append("hair")
    if any(k in low for k in ("呼吸", "胸", "胸腔", "breath", "chest", "rib")):
        targets.append("breath")
    if any(k in low for k in ("腰", "胯", "重心", "摆胯", "hips", "sway", "body")):
        targets.append("hips")
    if any(k in low for k in ("头", "颈", "脖", "head", "neck")):
        targets.append("head")
    if any(k in low for k in ("左手", "左臂", "left hand", "left arm")):
        targets.append("arm_L")
    if any(k in low for k in ("右手", "右臂", "right hand", "right arm")):
        targets.append("arm_R")
    if any(k in low for k in ("裙", "衣", "披风", "布料", "飘带", "cloth", "skirt", "cape", "sash")):
        targets.append("cloth")
    if any(k in low for k in ("尾", "tail")):
        targets.append("tail")
    if any(k in low for k in ("整体", "全局", "全部", "overall", "global", "all")):
        targets.append("global")
    return targets


def _extract_explicit_gain(low: str) -> Optional[float]:
    m = re.search(r"gain\s*[=:]?\s*([0-9]+(?:\.[0-9]+)?)", low)
    if m:
        return _clamp_gain(float(m.group(1)))
    m2 = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*[x×倍]", low)
    if m2:
        return _clamp_gain(float(m2.group(1)))
    return None


def classify_and_polish_adjustment(
    instruction: str = "",
    *,
    current_preset: str = "agile_full",
    current_gain: float = 1.0,
    requested_preset: Optional[str] = None,
    requested_gain: Optional[float] = None,
    rig_spec: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Classify natural-language feedback into minimal rerun layer and build a concrete spec patch."""
    raw = (instruction or "").strip()
    low = raw.lower()

    new_preset = requested_preset if requested_preset in list_presets() else current_preset
    new_gain = _clamp_gain(requested_gain) if requested_gain is not None else _clamp_gain(current_gain)

    # Allow preset selection directly from natural language
    if requested_preset is None:
        if "heavy_weapon_full" in low or "重武器" in low:
            new_preset = "heavy_weapon_full"
        elif "bust_half" in low or "半身" in low:
            new_preset = "bust_half"
        elif "agile_full" in low or "轻盈" in low:
            new_preset = "agile_full"

    explicit_gain = _extract_explicit_gain(low)
    if requested_gain is None and explicit_gain is not None:
        new_gain = explicit_gain

    module_contract = (rig_spec or {}).get("moduleContract") or {}
    if module_contract.get("sourceVersion") == 2:
        from src.pipeline.module_rig import MAX_MODULE_GAIN
        # The old polisher edited arbitrary bone weights, enlarged hidden fills or froze
        # targets. Complete modules may only be changed through their declared contract.
        if raw and (_contains_any(low, FREEZE_KEYWORDS) or _contains_any(low, LAYER_B_KEYWORDS)
                    or _contains_any(low, LAYER_C_KEYWORDS)):
            raise ValueError("完整模块的归属、接触、遮挡或静止声明需要在 Cindy 对话更新 V2 模块规划；不会自动归零、混权或扩大补全。")
        targets = _detect_target_groups(low)
        if raw and targets and set(targets) != {"global"} and explicit_gain is None:
            raise ValueError("局部模块微调需要在 Cindy 对话明确模块及保持的接触关系，再提交更新后的模块规划。")
        if raw and not targets and explicit_gain is None:
            raise ValueError("这条指令尚不能确定地映射为完整模块修改，请在 Cindy 对话说明要改哪个模块。")
        if requested_gain is None and explicit_gain is None:
            if _contains_any(low, STRONGER_KEYWORDS):
                new_gain = _clamp_gain(float(current_gain) * 1.35)
            elif _contains_any(low, SOFTER_KEYWORDS):
                new_gain = _clamp_gain(float(current_gain) * .7)
        if new_gain > MAX_MODULE_GAIN:
            raise ValueError(f"完整模块 gain 不能超过当前通用规则上限 {MAX_MODULE_GAIN:g}；请重新确定可达的动作幅度。")
        return {
            "raw_instruction": raw,
            "polished_instruction": f"按完整模块规则重算动作，预设 {new_preset}，全局 gain={new_gain:.2f}；保留模块归属与声明接触，再重新检查。",
            "rerun_layer": "motion", "skipped_stages": ["preprocess", "mivo", "vision", "split_parts"],
            "executed_stages": ["build_rig", "audit"],
            "spec_patch": {"preset": new_preset, "gain": new_gain,
                           "rig_motion_overrides": {}, "rig_ik_updates": [], "parts_updates": []},
        }

    # Determine minimal rerun layer (C > B > A)
    if raw and _contains_any(low, LAYER_C_KEYWORDS):
        rerun_layer = "split"
        skipped_stages = ["preprocess", "mivo"]
        executed_stages = ["split_parts", "build_rig", "audit"]
    elif raw and _contains_any(low, LAYER_B_KEYWORDS):
        rerun_layer = "rig"
        skipped_stages = ["preprocess", "mivo", "split_parts"]
        executed_stages = ["build_rig", "audit"]
    else:
        rerun_layer = "motion"
        skipped_stages = ["preprocess", "mivo", "split_parts"]
        executed_stages = ["build_rig", "audit"]

    targets = _detect_target_groups(low)
    if _contains_any(low, FREEZE_KEYWORDS):
        factor = 0.0
        direction_zh = "固定/归零"
    elif _contains_any(low, STRONGER_KEYWORDS):
        factor = 1.35
        direction_zh = "增强（约 1.35x）"
    elif _contains_any(low, SOFTER_KEYWORDS):
        factor = 0.7
        direction_zh = "柔和减弱（约 0.70x）"
    else:
        factor = 1.0
        direction_zh = "保持基准"

    rig_motion_overrides: Dict[str, Dict[str, Any]] = {}
    rig_ik_updates: List[Dict[str, Any]] = []
    parts_updates: List[Dict[str, Any]] = []
    polish_notes: List[str] = []

    if new_preset != current_preset:
        polish_notes.append(f"切换预设为 {PRESET_LABELS.get(new_preset, new_preset)}")

    if abs(new_gain - float(current_gain)) > 1e-4:
        polish_notes.append(f"全局增益 gain 调整为 {new_gain:.2f}x")

    preset_obj = get_preset(new_preset)
    preset_bones = preset_obj.get("bones", {})
    chain_names = list((rig_spec or {}).get("chains", {}).keys())

    if rerun_layer == "split":
        depth_delta = 10
        band_delta = 1 if ("边界" in low or "边缘" in low or "seam" in low) else 0
        target_part = "cloth" if "cloth" in targets else ("hair" if "hair" in targets else "all_soft")
        parts_updates.append(
            {
                "target": target_part,
                "depth_delta": depth_delta,
                "band_delta": band_delta,
                "note": raw or "重新分件与遮挡补全",
            }
        )
        polish_notes.append(
            f"Layer C 分件/补全重跑：扩展遮挡补全深度 +{depth_delta}px"
            + (f"并微调边界过渡 +{band_delta}px" if band_delta else "")
            + "，复用预处理与 Mivo 抠图结果"
        )
    elif rerun_layer == "rig":
        if "arm_L" in targets or "左手" in low or "左臂" in low:
            for bn in ("shoulder_L", "elbow_L", "hand_L", "wrist_target_L"):
                rig_motion_overrides[bn] = {"rotate": [], "tx": [], "ty": []}
            rig_ik_updates.append({"target": "arm_L", "action": "lock_anchor"})
            polish_notes.append("Layer B 绑定微调：锁定左臂/左手锚点位移与旋转，防止滑动")
        elif "arm_R" in targets or "右手" in low or "右臂" in low:
            for bn in ("shoulder_R", "elbow_R", "hand_R", "wrist_target_R"):
                rig_motion_overrides[bn] = {"rotate": [], "tx": [], "ty": []}
            rig_ik_updates.append({"target": "arm_R", "action": "lock_anchor"})
            polish_notes.append("Layer B 绑定微调：锁定右臂/右手锚点位移与旋转")
        elif any(k in low for k in ("反折", "flip", "bend", "膝盖", "手肘")):
            rig_ik_updates.append({"target": "all_2bone", "action": "toggle_bend_positive"})
            polish_notes.append("Layer B 绑定微调：翻转二骨 IK 弯曲方向（bendPositive）")
        else:
            # General anchor stability tightening
            rig_ik_updates.append({"target": "hips", "action": "tighten_anchor_motion"})
            if "hips" in preset_bones:
                hips_ch = copy.deepcopy(preset_bones["hips"])
                for ch_name in ("tx", "ty", "rotate"):
                    if ch_name in hips_ch:
                        hips_ch[ch_name] = _scale_terms(hips_ch[ch_name], 0.75)
                rig_motion_overrides["hips"] = hips_ch
            polish_notes.append("Layer B 绑定微调：收紧根部锚点与关节约束偏移，仅重跑绑定与验收")
    else:
        # Layer A: Motion
        if factor != 1.0:
            if not targets or "global" in targets:
                if requested_gain is None and explicit_gain is None:
                    new_gain = _clamp_gain(float(current_gain) * factor)
                    polish_notes.append(f"Layer A 动画幅度{direction_zh}：全局 gain 调整为 {new_gain:.2f}x")
            if "hair" in targets:
                hair_bones = [bn for bn in ("bangs", "back_hair1", "back_hair2", "back_hair3") if True]
                for cname in chain_names:
                    if "hair" in cname.lower() or "mane" in cname.lower():
                        for i in range(1, 5):
                            hair_bones.append(f"{cname}{i}")
                for bn in sorted(set(hair_bones)):
                    rig_motion_overrides[bn] = {
                        "rotate": [(round(1.6 * factor, 4), 1, 0.45), (round(0.55 * factor, 4), 2, 0.9)],
                    }
                polish_notes.append(f"Layer A 发丝摆动{direction_zh}：调整头发链路摆幅与延迟相位")
            if "breath" in targets:
                for bn in ("chest", "ribcage", "spine"):
                    if bn in preset_bones:
                        ch = copy.deepcopy(preset_bones[bn])
                        for k in ("tx", "ty", "sx", "sy", "rotate"):
                            if k in ch and isinstance(ch[k], list):
                                ch[k] = _scale_terms(ch[k], factor)
                        rig_motion_overrides[bn] = ch
                polish_notes.append(f"Layer A 胸腔呼吸{direction_zh}：缩放 spine/chest/ribcage 呼吸曲线")
            if "hips" in targets:
                if "hips" in preset_bones:
                    ch = copy.deepcopy(preset_bones["hips"])
                    for k in ("tx", "ty", "rotate"):
                        if k in ch and isinstance(ch[k], list):
                            ch[k] = _scale_terms(ch[k], factor)
                    rig_motion_overrides["hips"] = ch
                polish_notes.append(f"Layer A 重心摆动{direction_zh}：调整 hips 平移与旋转振幅")
            if "head" in targets:
                for bn in ("neck", "head"):
                    if bn in preset_bones:
                        ch = copy.deepcopy(preset_bones[bn])
                        if "rotate" in ch:
                            ch["rotate"] = _scale_terms(ch["rotate"], factor)
                        rig_motion_overrides[bn] = ch
                polish_notes.append(f"Layer A 头部微动{direction_zh}：调整 neck/head 旋转振幅")
            if "cloth" in targets:
                for bn in ("skirt_L", "skirt_R", "pouch"):
                    if bn in preset_bones:
                        ch = copy.deepcopy(preset_bones[bn])
                        if "rotate" in ch:
                            ch["rotate"] = _scale_terms(ch["rotate"], factor)
                        rig_motion_overrides[bn] = ch
                polish_notes.append(f"Layer A 布料飘动{direction_zh}：调整衣摆/裙摆跟动幅值")

    if not polish_notes:
        if raw:
            polish_notes.append(f"按指令「{raw}」执行 Layer A 动画层增量重算（跳过抠图与分件）")
        else:
            polish_notes.append("保持当前参数并重新构建动画与门禁验收")

    polished_instruction = "；".join(polish_notes)
    return {
        "raw_instruction": raw,
        "polished_instruction": polished_instruction,
        "rerun_layer": rerun_layer,
        "skipped_stages": skipped_stages,
        "executed_stages": executed_stages,
        "spec_patch": {
            "preset": new_preset,
            "gain": new_gain,
            "rig_motion_overrides": rig_motion_overrides,
            "rig_ik_updates": rig_ik_updates,
            "parts_updates": parts_updates,
        },
    }


def apply_spec_patch(
    parts_spec: Dict[str, Any],
    rig_spec: Dict[str, Any],
    adjustment: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], str, float]:
    """Return patched deep copies of `(parts_spec, rig_spec, preset, gain)`."""
    new_parts = copy.deepcopy(parts_spec)
    new_rig = copy.deepcopy(rig_spec)
    patch = adjustment["spec_patch"]

    # 1. Apply rig motion overrides
    motion_overrides = patch.get("rig_motion_overrides") or {}
    if motion_overrides:
        cur_motion = dict(new_rig.get("motion") or {})
        for bname, ch in motion_overrides.items():
            cur_motion[bname] = copy.deepcopy(ch)
        new_rig["motion"] = cur_motion

    # 2. Apply IK updates
    for upd in patch.get("rig_ik_updates") or []:
        if upd.get("action") == "toggle_bend_positive":
            for c in new_rig.get("ik", []):
                if len(c.get("bones", [])) == 2:
                    c["bendPositive"] = not bool(c.get("bendPositive", True))

    # 3. Apply parts updates (Layer C)
    parts_list = new_parts.get("parts") if isinstance(new_parts, dict) else new_parts
    for upd in patch.get("parts_updates") or []:
        depth_delta = int(upd.get("depth_delta", 0))
        band_delta = int(upd.get("band_delta", 0))
        target = upd.get("target", "all_soft")
        for p in parts_list:
            pname = str(p.get("name", "")).lower()
            is_match = (
                target == "all_soft"
                or (target == "hair" and "hair" in pname)
                or (target == "cloth" and any(k in pname for k in ("cloth", "cape", "skirt", "torso")))
            )
            if is_match:
                p["depth"] = min(120, max(8, int(p.get("depth", 24)) + depth_delta))
                if band_delta:
                    p["band"] = min(20, max(1, int(p.get("band", 4)) + band_delta))

    return new_parts, new_rig, str(patch["preset"]), float(patch["gain"])
