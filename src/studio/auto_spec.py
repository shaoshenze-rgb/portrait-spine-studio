"""Silhouette-aware default parts_spec and rig_spec generator for arbitrary uploaded portraits.

Ensures any RGBA portrait uploaded to the local studio produces a valid multi-part
layered split, Spine 4.2 rig with weighted meshes + IK anchor, and passes all 4 hard gates
(`rest_diff`, `setup_identity`, `no_holes` including 1.6x stress, `anchor_stability`).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from PIL import Image


def sanitize_character_name(filename: str) -> str:
    stem = Path(filename).stem
    clean = re.sub(r"[^a-zA-Z0-9_-]+", "_", stem).strip("_").lower()
    return clean[:32] or "portrait"


def build_default_specs(
    image_path: Union[str, Path],
    *,
    character: str = "portrait",
    preset: str = "agile_full",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Analyze the preprocessed RGBA portrait and generate default `(parts_spec, rig_spec)`."""
    rgba = np.asarray(Image.open(image_path).convert("RGBA"), dtype=np.uint8)
    h, w = rgba.shape[:2]
    alpha = rgba[..., 3]
    nz_y, nz_x = np.nonzero(alpha > 8)

    if nz_y.size > 0:
        x0, x1 = int(nz_x.min()), int(nz_x.max())
        y0, y1 = int(nz_y.min()), int(nz_y.max())
    else:
        x0, y0, x1, y1 = 0, 0, max(1, w - 1), max(1, h - 1)

    bw = max(16.0, float(x1 - x0 + 1))
    bh = max(16.0, float(y1 - y0 + 1))
    cx = (x0 + x1) * 0.5

    # Vertical anatomical landmarks within silhouette bounding box
    if preset == "bust_half":
        y_head_bot = y0 + bh * 0.42
        y_chest = y0 + bh * 0.58
        y_spine = y0 + bh * 0.74
        y_hips = y0 + bh * 0.88
        y_leg_top = y0 + bh * 0.90
    else:
        y_head_bot = y0 + bh * 0.28
        y_chest = y0 + bh * 0.42
        y_spine = y0 + bh * 0.55
        y_hips = y0 + bh * 0.65
        y_leg_top = y0 + bh * 0.68

    y_neck = y_head_bot + bh * 0.04
    y_head_top = y0 + bh * 0.05
    y_foot = min(float(h - 2), y1 - bh * 0.03)
    y_ankle = max(y_hips + 4.0, y_foot - max(8.0, bh * 0.08))

    # Generous hidden fill depth so 1.6x stress never tears interior seams
    fill_depth = max(36, min(140, int(round(max(bw, bh) * 0.18))))
    mesh_cell = max(16, min(48, int(round(min(w, h) / 10.0))))

    # Full-canvas coverage on back_hair ensures every alpha > 0 pixel is owned (rest_diff == 0)
    full_poly = [[0, 0], [w, 0], [w, h], [0, h]]
    legs_poly = [[0, y_leg_top], [w, y_leg_top], [w, h], [0, h]]
    torso_poly = [
        [max(0.0, cx - bw * 0.42), y_head_bot - bh * 0.02],
        [min(float(w), cx + bw * 0.42), y_head_bot - bh * 0.02],
        [min(float(w), cx + bw * 0.42), y_leg_top + bh * 0.02],
        [max(0.0, cx - bw * 0.42), y_leg_top + bh * 0.02],
    ]
    head_poly = [
        [max(0.0, cx - bw * 0.30), max(0.0, y0 - 4.0)],
        [min(float(w), cx + bw * 0.30), max(0.0, y0 - 4.0)],
        [min(float(w), cx + bw * 0.30), y_head_bot + bh * 0.02],
        [max(0.0, cx - bw * 0.30), y_head_bot + bh * 0.02],
    ]

    spread = min(bw, bh)
    hair_centerline = [
        [round(cx + spread * 0.08, 1), round(y0 + bh * 0.18, 1)],
        [round(cx + spread * 0.16, 1), round(y0 + bh * 0.38, 1)],
        [round(cx + spread * 0.22, 1), round(y0 + bh * 0.58, 1)],
    ]

    parts_spec: Dict[str, Any] = {
        "parts": [
            {
                "name": "back_hair",
                "kind": "soft",
                "parent": "torso",
                "pivot": [round(cx, 1), round(y_neck, 1)],
                "own": [full_poly],
                "band": 2,
                "depth": fill_depth,
                "guides": [(None, None, None, "smooth")],
                "centerline": hair_centerline,
            },
            {
                "name": "legs",
                "kind": "static",
                "parent": None,
                "pivot": [round(cx, 1), round(y_leg_top, 1)],
                "own": [legs_poly],
                "band": 2,
                "depth": fill_depth,
                "guides": [(None, None, None, "smooth")],
            },
            {
                "name": "torso",
                "kind": "soft",
                "parent": "legs",
                "pivot": [round(cx, 1), round(y_chest, 1)],
                "own": [torso_poly],
                "band": 2,
                "depth": fill_depth,
                "guides": [(None, None, None, "smooth")],
            },
            {
                "name": "head",
                "kind": "rigid",
                "parent": "torso",
                "pivot": [round(cx, 1), round(y_head_bot, 1)],
                "own": [head_poly],
                "band": 2,
                "depth": max(16, fill_depth // 2),
                "guides": [(None, None, None, "smooth")],
            },
        ],
        "min_island_px": 12,
    }

    # Scale hip translation gently on small portraits so 1.6x stress stays proportional
    size_scale = min(1.0, bh / 720.0)
    motion_overrides: Dict[str, Dict[str, Any]] = {}
    if size_scale < 0.95:
        from src.pipeline.presets import get_preset

        p_bones = get_preset(preset).get("bones", {})
        for bn in ("hips", "chest", "shoulder_L"):
            if bn in p_bones:
                ch = {}
                for k, v in p_bones[bn].items():
                    if k in ("tx", "ty") and isinstance(v, list):
                        ch[k] = [(round(float(a) * size_scale, 5), int(hk), float(ph)) for a, hk, ph in v]
                    else:
                        ch[k] = v
                motion_overrides[bn] = ch

    rig_spec: Dict[str, Any] = {
        "character": sanitize_character_name(character),
        "duration": 8.0,
        "fps": 30,
        "key_step": 0.25,
        "origin": [round(w * 0.5, 2), float(h)],
        "bones": [
            {"name": "root", "parent": None, "pos": [round(cx, 2), float(h)]},
            {"name": "hips", "parent": "root", "pos": [round(cx, 2), round(y_hips, 2)], "aim": [round(cx, 2), round(y_spine, 2)]},
            {"name": "spine", "parent": "hips", "pos": [round(cx, 2), round(y_spine, 2)], "aim": [round(cx, 2), round(y_chest, 2)]},
            {"name": "chest", "parent": "spine", "pos": [round(cx, 2), round(y_chest, 2)], "aim": [round(cx, 2), round(y_neck, 2)]},
            {"name": "ribcage", "parent": "chest", "pos": [round(cx, 2), round(y_chest, 2)]},
            {"name": "neck", "parent": "chest", "pos": [round(cx, 2), round(y_neck, 2)], "aim": [round(cx, 2), round(y_head_bot, 2)]},
            {"name": "head", "parent": "neck", "pos": [round(cx, 2), round(y_head_bot, 2)], "aim": [round(cx, 2), round(y_head_top, 2)]},
            {"name": "foot_L", "parent": "root", "pos": [round(cx - bw * 0.14, 2), round(y_ankle, 2)], "aim": [round(cx - bw * 0.14, 2), round(y_foot, 2)]},
            {"name": "toe_target_L", "parent": "root", "pos": [round(cx - bw * 0.14, 2), round(y_foot, 2)]},
        ],
        "chains": {
            "back_hair": {"parent": "neck", "part": "back_hair", "count": 2},
        },
        "ik": [
            {"name": "left_foot_ik", "bones": ["foot_L"], "target": "toe_target_L"},
        ],
        "skin": {
            "back_hair": {
                "type": "mesh",
                "recipe": "chain",
                "chain": "back_hair",
                "cell": mesh_cell,
            },
            "legs": {"type": "region", "bone": "root"},
            "torso": {
                "type": "mesh",
                "recipe": "linear_y",
                "bones": ["chest", "spine", "hips", "root"],
                "ranges": [
                    [round(y_chest, 1), round(y_spine, 1)],
                    [round(y_spine, 1), round(y_hips, 1)],
                    [round(y_hips, 1), round(y_leg_top, 1)],
                ],
                "cell": mesh_cell,
            },
            "head": {"type": "region", "bone": "head"},
        },
    }
    if motion_overrides:
        rig_spec["motion"] = motion_overrides

    return parts_spec, rig_spec
