"""Unified export profiles (Torchlight official 60fps dual-tier + true 4K) with multi-core parallel rendering."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import zipfile
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

from src.gates.review_binding import ensure_export_approved
from src.render.renderer import SpineRenderer, find_ffmpeg, straight_alpha, write_preview_mp4

LANCZOS = getattr(Image, "Resampling", Image).LANCZOS

SIZE_PRESETS: Dict[str, Dict[str, Any]] = {
    "torchlight_web_60": {
        "label": "官网同款 60帧双档（自动裁透明边 · 4K 硬解线内 ≤3840×2160）",
        "mode": "torchlight_web",
        "max_visible_width": 1920,
        "max_visible_height": 2160,
        "mobile_max_dim": 720,
    },
    "4k_portrait": {
        "label": "4K 竖屏 (2160×3840)",
        "mode": "fixed",
        "width": 2160,
        "height": 3840,
    },
    "4k_landscape": {
        "label": "4K 横屏 (3840×2160)",
        "mode": "fixed",
        "width": 3840,
        "height": 2160,
    },
    "4k_long_edge": {
        "label": "4K 长边上限 (4096px 保比例)",
        "mode": "long_edge",
        "max_long_edge": 4096,
    },
    "2k_portrait": {
        "label": "2K 竖屏 (1440×2560)",
        "mode": "fixed",
        "width": 1440,
        "height": 2560,
    },
    "2k_landscape": {
        "label": "2K 横屏 (2560×1440)",
        "mode": "fixed",
        "width": 2560,
        "height": 1440,
    },
    "1080p_portrait": {
        "label": "1080p 竖屏 (1080×1920)",
        "mode": "fixed",
        "width": 1080,
        "height": 1920,
    },
    "1080p_landscape": {
        "label": "1080p 横屏 (1920×1080)",
        "mode": "fixed",
        "width": 1920,
        "height": 1080,
    },
    "original": {
        "label": "保持原图尺寸",
        "mode": "original",
    },
    "custom": {
        "label": "自定义宽高",
        "mode": "fixed",
        "width": 2160,
        "height": 3840,
    },
}

FORMAT_OPTIONS: Dict[str, str] = {
    "torchlight_web": "官网同款 60帧双档透明视频 (PC + 手机预乘 H.264 + poster.webp + media.json)",
    "transparent_web": "VP9 透明网页视频 (alpha-vp9.webm + packed H.264 + poster.webp)",
    "spine_package": "Spine 4.2 工程包 (.zip)",
    "preview_mp4": "深浅底预览视频 (.mp4)",
    "standalone_html": "单文件交互预览 (.html)",
}

DEFAULT_EXPORT_PROFILE: Dict[str, Any] = {
    "size_preset": "torchlight_web_60",
    "width": 1920,
    "height": 2160,
    "max_long_edge": 4096,
    "align": "bottom_center",
    "bottom_margin_ratio": 0.03,
    "upscale_policy": "no_upscale",
    "formats": ["torchlight_web"],
    "fps": 60,
    "include_av1": False,
    "h264_preset": "slow",
    "high_crf": 15.0,
    "mobile_crf": 17.0,
    "render_workers": 0,
}

_WORKER_RENDERER: Optional[SpineRenderer] = None
_WORKER_RIG_DIR: Optional[str] = None


def _worker_init_renderer(rig_dir_str: str) -> None:
    global _WORKER_RENDERER, _WORKER_RIG_DIR
    _WORKER_RIG_DIR = rig_dir_str
    _WORKER_RENDERER = SpineRenderer(rig_dir_str)


def _worker_render_torchlight_frame(
    task: Tuple[int, float, Tuple[int, int, int, int], int, int, Optional[Tuple[int, int, int, int, int, int]]]
) -> Tuple[int, bytes, bytes, Optional[bytes]]:
    assert _WORKER_RENDERER is not None
    idx, t, high_crop, target_w, target_h, mob_spec = task
    raw_pm = _WORKER_RENDERER.frame(t)
    master_pm = (
        _resize_premultiplied_lanczos(raw_pm, target_w, target_h)
        if (raw_pm.shape[1] != target_w or raw_pm.shape[0] != target_h)
        else raw_pm
    )

    hx, hy, hw, hh = high_crop
    high_crop_pm = master_pm[hy : hy + hh, hx : hx + hw]
    high_sbs = _pack_sbs_premultiplied(high_crop_pm).tobytes()

    if mob_spec is not None:
        m_full_w, m_full_h, mx, my, mw, mh = mob_spec
        mob_full = _resize_premultiplied_lanczos(master_pm, m_full_w, m_full_h)
        mob_crop_pm = mob_full[my : my + mh, mx : mx + mw]
        mob_sbs = _pack_sbs_premultiplied(mob_crop_pm).tobytes()
    else:
        mob_sbs = b""

    f0_bytes = master_pm.tobytes() if idx == 0 else None
    return idx, high_sbs, mob_sbs, f0_bytes


def _worker_render_fitted_frame(task: Tuple[int, float, Dict[str, Any]]) -> Tuple[int, np.ndarray]:
    assert _WORKER_RENDERER is not None
    idx, t, prof = task
    raw_pm = _WORKER_RENDERER.frame(t)
    fitted, _ = fit_frame_to_profile(raw_pm, prof)
    return idx, fitted


def _even_dim(value: float, minimum: int = 2) -> int:
    iv = max(minimum, int(round(value)))
    if iv % 2 != 0:
        iv += 1
    return iv


def _align_down(value: int, mult: int) -> int:
    return (value // mult) * mult


def _align_up(value: int, mult: int) -> int:
    return ((value + mult - 1) // mult) * mult


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_export_profile(raw: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Validate and normalize an export profile dictionary."""
    merged = copy.deepcopy(DEFAULT_EXPORT_PROFILE)
    if isinstance(raw, dict):
        merged.update({k: v for k, v in raw.items() if v is not None})

    preset_key = str(merged.get("size_preset") or "torchlight_web_60")
    if preset_key not in SIZE_PRESETS:
        raise ValueError(f"不支持的导出尺寸预设 {preset_key!r}，可选：{list(SIZE_PRESETS)}")
    preset = SIZE_PRESETS[preset_key]

    if preset_key == "custom":
        w = int(merged.get("width") or 2160)
        h = int(merged.get("height") or 3840)
        if not (64 <= w <= 8192 and 64 <= h <= 8192):
            raise ValueError("自定义导出宽高必须在 64..8192 像素之间。")
        merged["width"] = _even_dim(w)
        merged["height"] = _even_dim(h)
    elif preset["mode"] == "fixed":
        merged["width"] = int(preset["width"])
        merged["height"] = int(preset["height"])
    elif preset["mode"] == "torchlight_web":
        merged["width"] = int(preset["max_visible_width"])
        merged["height"] = int(preset["max_visible_height"])
    elif preset["mode"] == "long_edge":
        edge = int(merged.get("max_long_edge") or preset["max_long_edge"])
        if not (64 <= edge <= 8192):
            raise ValueError("长边上限必须在 64..8192 像素之间。")
        merged["max_long_edge"] = _even_dim(edge)

    align = str(merged.get("align") or "bottom_center")
    if align not in ("bottom_center", "center"):
        raise ValueError("对齐方式仅支持 bottom_center 或 center。")
    merged["align"] = align

    margin = float(merged.get("bottom_margin_ratio", 0.03))
    if not (0.0 <= margin <= 0.25):
        raise ValueError("底部留白比例必须在 0.0..0.25 之间。")
    merged["bottom_margin_ratio"] = round(margin, 4)

    explicit_policy = raw.get("upscale_policy") if isinstance(raw, dict) else None
    policy = str(explicit_policy or ("no_upscale" if preset["mode"] == "torchlight_web" else "allow_upscale"))
    if policy not in ("allow_upscale", "no_upscale"):
        raise ValueError("小图缩放策略仅支持 allow_upscale 或 no_upscale。")
    merged["upscale_policy"] = policy

    raw_formats = merged.get("formats")
    if not isinstance(raw_formats, list) or not raw_formats:
        raise ValueError("至少选择一种导出格式。")
    formats: List[str] = []
    for fmt in raw_formats:
        key = str(fmt)
        if key not in FORMAT_OPTIONS:
            raise ValueError(f"不支持的导出格式 {key!r}，可选：{list(FORMAT_OPTIONS)}")
        if key not in formats:
            formats.append(key)
    merged["formats"] = formats

    fps = int(merged.get("fps", 60))
    if fps not in (15, 24, 30, 60):
        raise ValueError("导出帧率仅支持 15 / 24 / 30 / 60。")
    merged["fps"] = fps
    merged["include_av1"] = bool(merged.get("include_av1", False))

    h264_preset = str(merged.get("h264_preset") or "slow")
    if h264_preset not in ("ultrafast", "veryfast", "fast", "medium", "slow", "veryslow"):
        raise ValueError(f"不支持的 H.264 预设：{h264_preset}")
    merged["h264_preset"] = h264_preset
    merged["high_crf"] = round(float(merged.get("high_crf", 15.0)), 1)
    merged["mobile_crf"] = round(float(merged.get("mobile_crf", 17.0)), 1)
    merged["render_workers"] = max(0, int(merged.get("render_workers", 0)))

    if merged.get("output_dir") is not None:
        out_str = str(merged["output_dir"]).strip()
        merged["output_dir"] = out_str or None

    merged["size_preset"] = preset_key
    return merged


def compute_canvas_transform(src_w: int, src_h: int, profile: Dict[str, Any]) -> Dict[str, Any]:
    """Compute exact uniform scale & placement onto the target export canvas."""
    if src_w <= 0 or src_h <= 0:
        raise ValueError(f"Invalid source dimensions: {src_w}x{src_h}")
    prof = normalize_export_profile(profile)
    preset = SIZE_PRESETS[prof["size_preset"]]
    mode = preset["mode"]

    if mode == "torchlight_web":
        max_w = int(preset["max_visible_width"])
        max_h = int(preset["max_visible_height"])
        scale = min(1.0, max_w / float(src_w), max_h / float(src_h))
        if prof["upscale_policy"] == "allow_upscale":
            scale = min(max_w / float(src_w), max_h / float(src_h))
        scaled_w = max(16, int(round(src_w * scale)))
        scaled_h = max(16, int(round(src_h * scale)))
        out_w = _even_dim(scaled_w, 2)
        out_h = _even_dim(scaled_h, 2)
        return {
            "mode": mode,
            "src_w": int(src_w),
            "src_h": int(src_h),
            "out_w": out_w,
            "out_h": out_h,
            "scaled_w": out_w,
            "scaled_h": out_h,
            "scale": float(out_w / float(src_w)),
            "offset_x": 0,
            "offset_y": 0,
            "upscaled": bool(out_w > src_w + 1 or out_h > src_h + 1),
        }

    if mode == "original":
        out_w = _even_dim(src_w)
        out_h = _even_dim(src_h)
        return {
            "mode": mode,
            "src_w": int(src_w),
            "src_h": int(src_h),
            "out_w": out_w,
            "out_h": out_h,
            "scaled_w": int(src_w),
            "scaled_h": int(src_h),
            "scale": 1.0,
            "offset_x": (out_w - src_w) // 2,
            "offset_y": out_h - src_h if prof["align"] == "bottom_center" else (out_h - src_h) // 2,
            "upscaled": False,
        }

    if mode == "long_edge":
        target_long = int(prof["max_long_edge"])
        cur_long = max(src_w, src_h)
        scale = target_long / float(cur_long)
        if prof["upscale_policy"] == "no_upscale" and scale > 1.0:
            scale = 1.0
        scaled_w = max(2, int(round(src_w * scale)))
        scaled_h = max(2, int(round(src_h * scale)))
        out_w = _even_dim(scaled_w)
        out_h = _even_dim(scaled_h)
        return {
            "mode": mode,
            "src_w": int(src_w),
            "src_h": int(src_h),
            "out_w": out_w,
            "out_h": out_h,
            "scaled_w": scaled_w,
            "scaled_h": scaled_h,
            "scale": float(scale),
            "offset_x": (out_w - scaled_w) // 2,
            "offset_y": out_h - scaled_h if prof["align"] == "bottom_center" else (out_h - scaled_h) // 2,
            "upscaled": bool(scale > 1.001),
        }

    out_w = _even_dim(int(prof["width"]))
    out_h = _even_dim(int(prof["height"]))
    margin_bottom = int(round(out_h * prof["bottom_margin_ratio"])) if prof["align"] == "bottom_center" else 0
    margin_top = int(round(out_h * 0.02))
    margin_side = int(round(out_w * 0.02))
    avail_w = max(2, out_w - 2 * margin_side)
    avail_h = max(2, out_h - margin_top - margin_bottom)

    scale = min(avail_w / float(src_w), avail_h / float(src_h))
    if prof["upscale_policy"] == "no_upscale" and scale > 1.0:
        scale = 1.0

    scaled_w = max(1, min(out_w, int(round(src_w * scale))))
    scaled_h = max(1, min(out_h, int(round(src_h * scale))))
    offset_x = (out_w - scaled_w) // 2
    if prof["align"] == "bottom_center":
        offset_y = max(0, out_h - margin_bottom - scaled_h)
    else:
        offset_y = (out_h - scaled_h) // 2

    return {
        "mode": mode,
        "src_w": int(src_w),
        "src_h": int(src_h),
        "out_w": out_w,
        "out_h": out_h,
        "scaled_w": scaled_w,
        "scaled_h": scaled_h,
        "scale": float(scale),
        "offset_x": int(offset_x),
        "offset_y": int(offset_y),
        "upscaled": bool(scale > 1.001),
    }


def _resize_premultiplied_lanczos(frame_pm: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Resize premultiplied uint8 RGBA via Lanczos without introducing dark/bright edge halos."""
    h, w = frame_pm.shape[:2]
    if w == target_w and h == target_h:
        return frame_pm.copy()
    resized = np.asarray(Image.fromarray(frame_pm).resize((target_w, target_h), resample=LANCZOS), dtype=np.uint8)
    alpha = resized[..., 3:4]
    out = resized.copy()
    out[..., :3] = np.minimum(out[..., :3], alpha)
    return out


def fit_frame_to_profile(frame_pm: np.ndarray, profile: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Scale and align one premultiplied RGBA frame onto the profile's target canvas."""
    h, w = frame_pm.shape[:2]
    tr = compute_canvas_transform(w, h, profile)
    scaled = _resize_premultiplied_lanczos(frame_pm, tr["scaled_w"], tr["scaled_h"])
    if tr["out_w"] == tr["scaled_w"] and tr["out_h"] == tr["scaled_h"] and tr["offset_x"] == 0 and tr["offset_y"] == 0:
        return scaled, tr
    canvas = np.zeros((tr["out_h"], tr["out_w"], 4), dtype=np.uint8)
    oy, ox = tr["offset_y"], tr["offset_x"]
    canvas[oy : oy + tr["scaled_h"], ox : ox + tr["scaled_w"]] = scaled
    return canvas, tr


def compute_rig_union_bbox(
    renderer: SpineRenderer,
    duration: float,
    fps: int,
    total_frames: int,
    *,
    target_w: Optional[int] = None,
    target_h: Optional[int] = None,
) -> Tuple[List[int], Tuple[int, int, int, int]]:
    """Compute exact union alpha bounding box over all poses in <200ms without full rasterization."""
    full_w = int(target_w or renderer.width)
    full_h = int(target_h or renderer.height)
    sx = full_w / float(renderer.width)
    sy = full_h / float(renderer.height)

    slot_tight_uv: Dict[str, Tuple[float, float, float, float]] = {}
    for slot in renderer.sk.slots:
        tex = renderer.tex[slot["name"]]
        th, tw = tex.shape[:2]
        ys, xs = np.nonzero(tex[..., 3] > 0)
        if len(xs):
            slot_tight_uv[slot["name"]] = (
                float(xs.min()) / tw,
                float(ys.min()) / th,
                float(xs.max() + 1) / tw,
                float(ys.max() + 1) / th,
            )

    xmin, ymin = float(full_w), float(full_h)
    xmax, ymax = 0.0, 0.0

    for idx in range(total_frames):
        t = (idx / float(fps)) % duration
        renderer.pose(t)
        for slot in renderer.sk.slots:
            if slot["name"] not in slot_tight_uv:
                continue
            world, uv, _tris, _att = renderer.sk.attachment_vertices(slot)
            vc = renderer.to_canvas(world)
            u0, v0, u1, v1 = slot_tight_uv[slot["name"]]
            vx = vc[:, 0] * sx
            vy = vc[:, 1] * sy
            if len(vc) == 4:
                # Exact 2D affine mapping from texture UV to scaled canvas coords (rotation/flip safe)
                a_mat = np.array(
                    [
                        [float(uv[1, 0] - uv[0, 0]), float(uv[2, 0] - uv[0, 0])],
                        [float(uv[1, 1] - uv[0, 1]), float(uv[2, 1] - uv[0, 1])],
                    ],
                    dtype=np.float64,
                )
                if abs(float(np.linalg.det(a_mat))) > 1e-8:
                    a_inv = np.linalg.inv(a_mat)
                    corners_uv = np.array([[u0, v0], [u1, v0], [u1, v1], [u0, v1]], dtype=np.float64)
                    duv = corners_uv - uv[0].astype(np.float64)
                    ab = (a_inv @ duv.T).T
                    p0 = np.array([vx[0], vy[0]], dtype=np.float64)
                    dp1 = np.array([vx[1] - vx[0], vy[1] - vy[0]], dtype=np.float64)
                    dp2 = np.array([vx[2] - vx[0], vy[2] - vy[0]], dtype=np.float64)
                    pts = p0[None, :] + ab[:, 0:1] * dp1[None, :] + ab[:, 1:2] * dp2[None, :]
                    xmin = min(xmin, float(pts[:, 0].min()))
                    xmax = max(xmax, float(pts[:, 0].max()))
                    ymin = min(ymin, float(pts[:, 1].min()))
                    ymax = max(ymax, float(pts[:, 1].max()))
                    continue
            xmin = min(xmin, float(vx.min()))
            xmax = max(xmax, float(vx.max()))
            ymin = min(ymin, float(vy.min()))
            ymax = max(ymax, float(vy.max()))

    if xmax <= xmin or ymax <= ymin:
        return [0, 0, full_w, full_h], (0, 0, full_w, full_h)

    ux0 = max(0, int(math.floor(xmin)))
    uy0 = max(0, int(math.floor(ymin)))
    ux1 = min(full_w, int(math.ceil(xmax)))
    uy1 = min(full_h, int(math.ceil(ymax)))

    pad = 8
    tx0 = max(0, ux0 - pad)
    ty0 = max(0, uy0 - pad)
    tx1 = min(full_w, ux1 + pad)
    ty1 = min(full_h, uy1 + pad)

    cx0 = _align_down(tx0, 16)
    cw_pref = max(16, _align_up(tx1 - cx0, 16))
    if cx0 + cw_pref <= full_w:
        cw = cw_pref
    else:
        shifted_x0 = _align_down(max(0, full_w - cw_pref), 16)
        if shifted_x0 <= ux0 and shifted_x0 + cw_pref <= full_w:
            cx0, cw = shifted_x0, cw_pref
        else:
            cx0 = _align_down(tx0, 2)
            cw = max(2, _align_up(tx1 - cx0, 2))
            if cx0 + cw > full_w:
                cw = max(2, _align_down(full_w - cx0, 2))
                if cx0 + cw < ux1:
                    cx0 = 0
                    cw = _even_dim(full_w, 2)

    cy0 = _align_down(ty0, 16)
    ch = max(2, _align_up(ty1 - cy0, 2))
    if cy0 + ch > full_h:
        cy0 = _align_down(max(0, full_h - ch), 2)
        if cy0 > uy0:
            cy0 = _align_down(uy0, 2)
        ch = max(2, _align_down(full_h - cy0, 2))
        if cy0 + ch < uy1:
            cy0 = 0
            ch = _even_dim(full_h, 2)

    return [ux0, uy0, ux1, uy1], (int(cx0), int(cy0), int(cw), int(ch))


def _compute_mobile_spec(
    master_w: int,
    master_h: int,
    high_crop: Tuple[int, int, int, int],
    *,
    mobile_max_dim: int = 720,
) -> Tuple[int, int, int, int, int, int]:
    """Compute half-scale mobile dimensions & aligned transparent crop matching official Torchlight mobile tier."""
    hx, hy, hw, hh = high_crop
    scale = min(0.5, mobile_max_dim / float(max(hw, hh)))
    m_full_w = _even_dim(max(16, master_w * scale), 2)
    m_full_h = _even_dim(max(16, master_h * scale), 2)
    sx = m_full_w / float(master_w)
    sy = m_full_h / float(master_h)

    rx0 = max(0, int(math.floor(hx * sx)))
    ry0 = max(0, int(math.floor(hy * sy)))
    rx1 = min(m_full_w, int(math.ceil((hx + hw) * sx)))
    ry1 = min(m_full_h, int(math.ceil((hy + hh) * sy)))

    mx0 = _align_down(rx0, 8)
    mw_pref = max(16, _align_up(rx1 - mx0, 8))
    if mx0 + mw_pref <= m_full_w:
        mw = mw_pref
    else:
        mx0 = _align_down(rx0, 2)
        mw = max(2, _align_up(rx1 - mx0, 2))
        if mx0 + mw > m_full_w:
            mw = max(2, _align_down(m_full_w - mx0, 2))
            if mx0 + mw < rx1:
                mx0, mw = 0, _even_dim(m_full_w, 2)

    my0 = _align_down(ry0, 8)
    mh = max(2, _align_up(ry1 - my0, 2))
    if my0 + mh > m_full_h:
        my0 = _align_down(ry0, 2)
        mh = max(2, _align_down(m_full_h - my0, 2))
        if my0 + mh < ry1:
            my0, mh = 0, _even_dim(m_full_h, 2)
    return int(m_full_w), int(m_full_h), int(mx0), int(my0), int(mw), int(mh)


def _pack_sbs_premultiplied(frame_pm: np.ndarray) -> np.ndarray:
    """Official Torchlight side-by-side packing: left = premultiplied RGB, right = alpha grayscale."""
    rgb = np.minimum(frame_pm[..., :3], frame_pm[..., 3:4])
    alpha = np.repeat(frame_pm[..., 3:4], 3, axis=2)
    h, w = rgb.shape[:2]
    ew, eh = w + (w % 2), h + (h % 2)
    if ew != w or eh != h:
        rgb_p = np.zeros((eh, ew, 3), dtype=np.uint8)
        alp_p = np.zeros((eh, ew, 3), dtype=np.uint8)
        rgb_p[:h, :w] = rgb
        alp_p[:h, :w] = alpha
        rgb, alpha = rgb_p, alp_p
    return np.ascontiguousarray(np.concatenate([rgb, alpha], axis=1))


def _pack_sbs_rgb_alpha(frame_pm: np.ndarray) -> np.ndarray:
    """Straight-alpha side-by-side packing for legacy transparent_web format."""
    st = straight_alpha(frame_pm)
    rgb = st[..., :3]
    alpha = np.repeat(st[..., 3:4], 3, axis=2)
    h, w = rgb.shape[:2]
    ew, eh = w + (w % 2), h + (h % 2)
    if ew != w or eh != h:
        rgb_p = np.zeros((eh, ew, 3), dtype=np.uint8)
        alp_p = np.zeros((eh, ew, 3), dtype=np.uint8)
        rgb_p[:h, :w] = rgb
        alp_p[:h, :w] = alpha
        rgb, alpha = rgb_p, alp_p
    return np.ascontiguousarray(np.concatenate([rgb, alpha], axis=1))


def _h264_bt709_cmd(
    ffmpeg_bin: str,
    out_path: Path,
    *,
    enc_w: int,
    enc_h: int,
    fps: int,
    crf: float,
    preset: str,
) -> List[str]:
    gop = max(24, fps * 2)
    return [
        ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{enc_w}x{enc_h}", "-r", str(fps), "-i", "-",
        "-vf", "scale=out_color_matrix=bt709:out_range=tv,setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv",
        "-c:v", "libx264", "-profile:v", "high", "-preset", preset, "-crf", str(crf),
        "-g", str(gop), "-bf", "2",
        "-pix_fmt", "yuv420p",
        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
        "-movflags", "+faststart", "-an",
        str(out_path),
    ]


def _ensure_repo_in_pythonpath() -> None:
    """Guarantee spawned macOS worker processes can import `src.*` even without external PYTHONPATH."""
    repo_root = str(Path(__file__).resolve().parents[2])
    cur = os.environ.get("PYTHONPATH", "")
    parts = [x for x in cur.split(os.pathsep) if x]
    if repo_root not in parts:
        os.environ["PYTHONPATH"] = os.pathsep.join([repo_root] + parts)


def _resolve_worker_count(requested: int, total_frames: int) -> int:
    import multiprocessing
    if total_frames <= 6 or multiprocessing.current_process().name != "MainProcess":
        return 1
    if requested > 0:
        return min(requested, total_frames)
    cpus = os.cpu_count() or 4
    return max(1, min(12, cpus - 2, total_frames))


def export_torchlight_web_package(
    rig_dir: Union[str, Path],
    out_dir: Union[str, Path],
    *,
    character: Optional[str] = None,
    label: Optional[str] = None,
    profile: Optional[Dict[str, Any]] = None,
    ffmpeg_bin: Optional[str] = None,
    max_frames: Optional[int] = None,
    user_signoff_path: Optional[Union[str, Path]] = None,
    require_approved: bool = True,
) -> Dict[str, Any]:
    """Export official Torchlight web 4-piece package with multi-core parallel 60fps rendering."""
    prof = normalize_export_profile(profile)
    rdir = Path(rig_dir).resolve()
    odir = Path(out_dir).resolve()
    if require_approved:
        ensure_export_approved(rdir, user_signoff_path=user_signoff_path)
    odir.mkdir(parents=True, exist_ok=True)

    ff = ffmpeg_bin or find_ffmpeg()
    if not ff:
        raise RuntimeError("未找到 ffmpeg，无法导出官网同款透明视频。")

    rep = json.loads((rdir / "rig-build-report.json").read_text(encoding="utf-8"))
    char_name = str(character or rep.get("character") or "character")
    display_label = str(label or char_name)
    duration = float(rep.get("duration", 8.0))
    fps = int(prof["fps"])
    total_frames = max(1, int(round(duration * fps)))
    if max_frames is not None:
        total_frames = min(total_frames, int(max_frames))

    renderer = SpineRenderer(rdir)
    tr = compute_canvas_transform(renderer.width, renderer.height, prof)
    master_w = _even_dim(int(tr["scaled_w"]), 2)
    master_h = _even_dim(int(tr["scaled_h"]), 2)

    alpha_union_bbox, high_crop = compute_rig_union_bbox(
        renderer, duration, fps, total_frames, target_w=master_w, target_h=master_h
    )
    hx, hy, hw, hh = high_crop
    if hw > 1920:
        cap_scale = 1920.0 / float(hw)
        master_w = _even_dim(master_w * cap_scale, 2)
        master_h = _even_dim(master_h * cap_scale, 2)
        alpha_union_bbox, high_crop = compute_rig_union_bbox(
            renderer, duration, fps, total_frames, target_w=master_w, target_h=master_h
        )
        hx, hy, hw, hh = high_crop

    mob_max = int(SIZE_PRESETS["torchlight_web_60"]["mobile_max_dim"])
    mob_spec = _compute_mobile_spec(master_w, master_h, high_crop, mobile_max_dim=mob_max)
    m_full_w, m_full_h, mx, my, mw, mh = mob_spec

    high_path = odir / "idle-high-packed.mp4"
    mob_path = odir / "idle-mobile-packed.mp4"
    poster_path = odir / "poster.webp"
    media_path = odir / "media.json"

    high_enc_w = (hw + (hw % 2)) * 2
    high_enc_h = hh + (hh % 2)
    mob_enc_w = (mw + (mw % 2)) * 2
    mob_enc_h = mh + (mh % 2)

    cmd_high = _h264_bt709_cmd(
        ff, high_path, enc_w=high_enc_w, enc_h=high_enc_h, fps=fps,
        crf=float(prof["high_crf"]), preset=str(prof["h264_preset"]),
    )
    cmd_mob = _h264_bt709_cmd(
        ff, mob_path, enc_w=mob_enc_w, enc_h=mob_enc_h, fps=fps,
        crf=float(prof["mobile_crf"]), preset=str(prof["h264_preset"]),
    )

    workers = _resolve_worker_count(int(prof.get("render_workers", 0)), total_frames)
    tasks = [
        (idx, (idx / float(fps)) % duration, high_crop, master_w, master_h, mob_spec)
        for idx in range(total_frames)
    ]

    frame0_master_pm: Optional[np.ndarray] = None
    p_high = subprocess.Popen(cmd_high, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    p_mob = subprocess.Popen(cmd_mob, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        assert p_high.stdin is not None and p_mob.stdin is not None
        if workers <= 1:
            for idx, t, hc, tw, th, ms in tasks:
                raw_pm = renderer.frame(t)
                master_pm = _resize_premultiplied_lanczos(raw_pm, tw, th) if (raw_pm.shape[1] != tw or raw_pm.shape[0] != th) else raw_pm
                if idx == 0:
                    frame0_master_pm = master_pm.copy()
                p_high.stdin.write(_pack_sbs_premultiplied(master_pm[hc[1] : hc[1] + hc[3], hc[0] : hc[0] + hc[2]]).tobytes())
                mob_full = _resize_premultiplied_lanczos(master_pm, ms[0], ms[1])
                p_mob.stdin.write(_pack_sbs_premultiplied(mob_full[ms[3] : ms[3] + ms[5], ms[2] : ms[2] + ms[4]]).tobytes())
        else:
            _ensure_repo_in_pythonpath()
            in_flight = max(workers * 2, 16)
            pending = deque()
            task_iter = iter(tasks)
            with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init_renderer, initargs=(str(rdir),)) as pool:
                for _ in range(min(in_flight, total_frames)):
                    pending.append(pool.submit(_worker_render_torchlight_frame, next(task_iter)))
                while pending:
                    idx, high_bytes, mob_bytes, f0_bytes = pending.popleft().result()
                    next_task = next(task_iter, None)
                    if next_task is not None:
                        pending.append(pool.submit(_worker_render_torchlight_frame, next_task))
                    if idx == 0 and f0_bytes is not None:
                        frame0_master_pm = np.frombuffer(f0_bytes, dtype=np.uint8).reshape((master_h, master_w, 4)).copy()
                    p_high.stdin.write(high_bytes)
                    p_mob.stdin.write(mob_bytes)

        p_high.stdin.close()
        p_mob.stdin.close()
        err_h = p_high.stderr.read().decode("utf-8", errors="replace") if p_high.stderr else ""
        err_m = p_mob.stderr.read().decode("utf-8", errors="replace") if p_mob.stderr else ""
        rc_h = p_high.wait()
        rc_m = p_mob.wait()
        if rc_h != 0:
            raise RuntimeError(f"PC 高清预乘视频编码失败：{err_h}")
        if rc_m != 0:
            raise RuntimeError(f"手机端预乘视频编码失败：{err_m}")
    finally:
        for proc in (p_high, p_mob):
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            if proc.stderr is not None:
                proc.stderr.close()

    assert frame0_master_pm is not None
    Image.fromarray(straight_alpha(frame0_master_pm)).save(poster_path, format="WEBP", lossless=True)

    high_bytes_len = high_path.stat().st_size
    mob_bytes_len = mob_path.stat().st_size
    poster_bytes_len = poster_path.stat().st_size
    gop = max(24, fps * 2)
    last_pts = round((total_frames - 1) / float(fps), 6) if total_frames > 1 else 0.0

    media_json = {
        "role": char_name,
        "label": display_label,
        "duration": duration,
        "fps": fps,
        "frames": total_frames,
        "packing": "side-by-side",
        "rgbAlphaMode": "premultiplied",
        "rgbRect": [0, 0, 0.5, 1],
        "alphaRect": [0.5, 0, 0.5, 1],
        "poster": {
            "src": "poster.webp",
            "width": master_w,
            "height": master_h,
            "bytes": poster_bytes_len,
            "sha256": _sha256_file(poster_path),
            "format": "Lossless WebP, straight alpha; exact Spine animation frame 0",
        },
        "master": {
            "width": master_w,
            "height": master_h,
            "alphaUnionBBox": alpha_union_bbox,
        },
        "crop": None,
        "cropNote": "Full master/poster unchanged. Each video variant.crop=[x,y,w,h,fullW,fullH] trims fully transparent borders only; runtime restores registration.",
        "variants": {
            "high": {
                "src": "idle-high-packed.mp4",
                "visibleWidth": hw,
                "visibleHeight": hh,
                "encodedWidth": high_enc_w,
                "encodedHeight": high_enc_h,
                "fps": fps,
                "frames": total_frames,
                "bytes": high_bytes_len,
                "sha256": _sha256_file(high_path),
                "codec": f"H.264 {fps}fps; independently evaluated Spine pose at i/{fps}",
                "contentType": 'video/mp4; codecs="avc1.640033"',
                "crop": [hx, hy, hw, hh, master_w, master_h],
                "encoding": {
                    "preset": prof["h264_preset"],
                    "crf": prof["high_crf"],
                    "gop": gop,
                    "colorspace": "bt709",
                },
            },
            "mobile": {
                "src": "idle-mobile-packed.mp4",
                "visibleWidth": mw,
                "visibleHeight": mh,
                "encodedWidth": mob_enc_w,
                "encodedHeight": mob_enc_h,
                "fps": fps,
                "frames": total_frames,
                "bytes": mob_bytes_len,
                "sha256": _sha256_file(mob_path),
                "codec": f"H.264 {fps}fps; independently evaluated Spine pose at i/{fps}",
                "contentType": 'video/mp4; codecs="avc1.640028"',
                "scaleFromMaster": m_full_h / float(master_h),
                "scaleFromMasterXY": [m_full_w / float(master_w), m_full_h / float(master_h)],
                "crop": [mx, my, mw, mh, m_full_w, m_full_h],
                "encoding": {
                    "preset": prof["h264_preset"],
                    "crf": prof["mobile_crf"],
                    "gop": gop,
                    "colorspace": "bt709",
                },
            },
        },
        "loop": {
            "frames": total_frames,
            "firstPts": 0,
            "lastPts": last_pts,
            "note": f"{total_frames} independent bone evaluations at {fps}Hz, {duration}s; no duplication, no repeated endpoint.",
        },
    }
    media_path.write_text(json.dumps(media_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return {
        "high_mp4": str(high_path),
        "mobile_mp4": str(mob_path),
        "poster_webp": str(poster_path),
        "media_json": str(media_path),
        "master_size": [master_w, master_h],
        "high_encoded_size": [high_enc_w, high_enc_h],
        "mobile_encoded_size": [mob_enc_w, mob_enc_h],
        "workers_used": workers,
        "media": media_json,
    }


def _scale_mesh_vertices(vertices: List[float], uv_len: int, scale: float) -> List[float]:
    if len(vertices) == uv_len:
        return [round(float(v) * scale, 4) for v in vertices]
    out: List[float] = []
    i = 0
    n = len(vertices)
    while i < n:
        count = int(vertices[i])
        out.append(count)
        i += 1
        for _ in range(count):
            if i + 3 >= n:
                break
            bone_idx = int(vertices[i])
            vx = round(float(vertices[i + 1]) * scale, 4)
            vy = round(float(vertices[i + 2]) * scale, 4)
            weight = float(vertices[i + 3])
            out.extend([bone_idx, vx, vy, weight])
            i += 4
    return out


def scale_spine_rig_to_profile(
    rig_dir: Union[str, Path],
    out_rig_dir: Union[str, Path],
    profile: Dict[str, Any],
) -> Dict[str, Any]:
    """Create a scaled copy of a Spine 4.2 rig matching `profile` exact canvas & alignment."""
    rdir = Path(rig_dir).resolve()
    odir = Path(out_rig_dir).resolve()
    odir.mkdir(parents=True, exist_ok=True)

    rep_path = rdir / "rig-build-report.json"
    rep = json.loads(rep_path.read_text(encoding="utf-8"))
    spine_file = str(rep["spineFile"])
    spine_data = json.loads((rdir / spine_file).read_text(encoding="utf-8"))

    sk_meta = spine_data.setdefault("skeleton", {})
    src_w = int(round(float(rep.get("canvas", [sk_meta.get("width", 1024), sk_meta.get("height", 1024)])[0])))
    src_h = int(round(float(rep.get("canvas", [sk_meta.get("width", 1024), sk_meta.get("height", 1024)])[1])))
    origin_x = float(rep.get("origin", [-float(sk_meta.get("x", -src_w / 2.0)), float(src_h) + float(sk_meta.get("y", 0.0))])[0])
    origin_y = float(rep.get("origin", [-float(sk_meta.get("x", -src_w / 2.0)), float(src_h) + float(sk_meta.get("y", 0.0))])[1])

    tr = compute_canvas_transform(src_w, src_h, profile)
    s = float(tr["scale"])
    ox = float(tr["offset_x"])
    oy = float(tr["offset_y"])
    out_w = int(tr["out_w"])
    out_h = int(tr["out_h"])

    new_origin_x = origin_x * s + ox
    new_origin_y = origin_y * s + oy

    sk_meta["width"] = out_w
    sk_meta["height"] = out_h
    sk_meta["x"] = round(-new_origin_x, 4)
    sk_meta["y"] = round(new_origin_y - out_h, 4)

    if abs(s - 1.0) > 1e-6:
        for bone in spine_data.get("bones", []):
            for key in ("x", "y", "length"):
                if key in bone and isinstance(bone[key], (int, float)):
                    bone[key] = round(float(bone[key]) * s, 4)

        for skin in spine_data.get("skins", []):
            for _slot, atts in (skin.get("attachments") or {}).items():
                for _name, att in (atts or {}).items():
                    if not isinstance(att, dict):
                        continue
                    for key in ("x", "y", "width", "height"):
                        if key in att and isinstance(att[key], (int, float)):
                            att[key] = round(float(att[key]) * s, 4)
                    if "vertices" in att and isinstance(att["vertices"], list):
                        uv_len = len(att.get("uvs") or [])
                        att["vertices"] = _scale_mesh_vertices(att["vertices"], uv_len, s)

        for _anim_name, anim in (spine_data.get("animations") or {}).items():
            for _bname, tracks in (anim.get("bones") or {}).items():
                for key in ("translate",):
                    for kf in tracks.get(key) or []:
                        if "x" in kf:
                            kf["x"] = round(float(kf["x"]) * s, 4)
                        if "y" in kf:
                            kf["y"] = round(float(kf["y"]) * s, 4)
                for key in ("translatex", "translatey"):
                    for kf in tracks.get(key) or []:
                        if "value" in kf:
                            kf["value"] = round(float(kf["value"]) * s, 4)
            for _skin_name, slots in (anim.get("attachments") or {}).items():
                for _slot_name, atts in (slots or {}).items():
                    for _att_name, tracks in (atts or {}).items():
                        for kf in tracks.get("deform") or []:
                            if "vertices" in kf and isinstance(kf["vertices"], list):
                                kf["vertices"] = [round(float(v) * s, 4) for v in kf["vertices"]]

    new_rep = copy.deepcopy(rep)
    new_rep["canvas"] = [out_w, out_h]
    new_rep["origin"] = [round(new_origin_x, 4), round(new_origin_y, 4)]
    new_rep["exportTransform"] = tr

    (odir / spine_file).write_text(json.dumps(spine_data, ensure_ascii=False, indent=2), encoding="utf-8")
    (odir / "rig-build-report.json").write_text(json.dumps(new_rep, ensure_ascii=False, indent=2), encoding="utf-8")
    for extra in ("health-report.json",):
        src_extra = rdir / extra
        if src_extra.is_file():
            shutil.copy2(src_extra, odir / extra)

    img_in = rdir / "images"
    img_out = odir / "images"
    img_out.mkdir(parents=True, exist_ok=True)
    for png in sorted(img_in.glob("*.png")):
        if abs(s - 1.0) <= 1e-6:
            shutil.copy2(png, img_out / png.name)
            continue
        with Image.open(png) as im:
            rgba = np.asarray(im.convert("RGBA"), dtype=np.float32)
            pm = rgba.copy()
            pm[..., :3] *= rgba[..., 3:4] / 255.0
            pm_u8 = np.clip(np.rint(pm), 0, 255).astype(np.uint8)
            nw = max(1, int(round(im.width * s)))
            nh = max(1, int(round(im.height * s)))
            resized_pm = _resize_premultiplied_lanczos(pm_u8, nw, nh)
            Image.fromarray(straight_alpha(resized_pm)).save(img_out / png.name, format="PNG")

    return tr


def _export_transparent_frames(
    frames_pm: Sequence[np.ndarray],
    out_dir: Path,
    *,
    character: str,
    fps: int,
    is_4k: bool,
    include_av1: bool,
    ffmpeg_bin: str,
) -> Dict[str, Optional[str]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    poster_path = out_dir / f"{character}-poster.webp"
    Image.fromarray(straight_alpha(frames_pm[0])).save(poster_path, format="WEBP", lossless=True)

    h, w = frames_pm[0].shape[:2]
    ew, eh = w + (w % 2), h + (h % 2)
    vp9_path = out_dir / f"{character}-alpha-vp9.webm"
    vp9_crf = "18" if is_4k else "22"
    cmd_vp9 = [
        ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{ew}x{eh}", "-r", str(fps), "-i", "-",
        "-vf", "format=yuva420p", "-c:v", "libvpx-vp9", "-b:v", "0", "-crf", vp9_crf,
        "-deadline", "good", "-cpu-used", "4", "-row-mt", "1", "-auto-alt-ref", "0",
        str(vp9_path),
    ]
    with subprocess.Popen(cmd_vp9, stdin=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
        assert proc.stdin is not None
        for frame in frames_pm:
            st = straight_alpha(frame)
            if ew != w or eh != h:
                pad = np.zeros((eh, ew, 4), dtype=np.uint8)
                pad[:h, :w] = st
                st = pad
            proc.stdin.write(st.tobytes())
        proc.stdin.close()
        err = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        if proc.wait() != 0:
            raise RuntimeError(f"VP9 透明视频导出失败：{err}")

    sbs_sample = _pack_sbs_rgb_alpha(frames_pm[0])
    sh, sw = sbs_sample.shape[:2]
    h264_path = out_dir / f"{character}-packed-h264.mp4"
    h264_crf = "14" if is_4k else "16"
    cmd_h264 = [
        ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{sw}x{sh}", "-r", str(fps), "-i", "-",
        "-vf", "format=yuv420p", "-c:v", "libx264", "-preset", "medium", "-crf", h264_crf,
        "-movflags", "+faststart", str(h264_path),
    ]
    with subprocess.Popen(cmd_h264, stdin=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
        assert proc.stdin is not None
        for frame in frames_pm:
            proc.stdin.write(_pack_sbs_rgb_alpha(frame).tobytes())
        proc.stdin.close()
        err = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        if proc.wait() != 0:
            raise RuntimeError(f"H.264 打包透明视频导出失败：{err}")

    av1_path: Optional[Path] = None
    if include_av1:
        enc_list = subprocess.run([ffmpeg_bin, "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
        av1_enc = "libsvtav1" if "libsvtav1" in enc_list else ("libaom-av1" if "libaom-av1" in enc_list else None)
        if av1_enc:
            av1_path = out_dir / f"{character}-packed-av1.mp4"
            cmd_av1 = [
                ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{sw}x{sh}", "-r", str(fps), "-i", "-",
                "-vf", "format=yuv420p", "-c:v", av1_enc, "-crf", "24",
                "-movflags", "+faststart", str(av1_path),
            ]
            if av1_enc == "libaom-av1":
                cmd_av1.insert(-2, "-cpu-used")
                cmd_av1.insert(-2, "8")
            with subprocess.Popen(cmd_av1, stdin=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
                assert proc.stdin is not None
                for frame in frames_pm:
                    proc.stdin.write(_pack_sbs_rgb_alpha(frame).tobytes())
                proc.stdin.close()
                if proc.stderr:
                    proc.stderr.read()
                if proc.wait() != 0:
                    av1_path = None

    return {
        "alpha_vp9_webm": str(vp9_path),
        "packed_h264_mp4": str(h264_path),
        "packed_av1_mp4": str(av1_path) if av1_path else None,
        "poster_webp": str(poster_path),
    }


def export_job_with_profile(
    rig_dir: Union[str, Path],
    out_dir: Union[str, Path],
    profile: Optional[Dict[str, Any]] = None,
    *,
    character: Optional[str] = None,
    label: Optional[str] = None,
    health_report_path: Optional[Union[str, Path]] = None,
    user_signoff_path: Optional[Union[str, Path]] = None,
    ffmpeg_bin: Optional[str] = None,
    max_frames: Optional[int] = None,
    require_approved: bool = True,
) -> Dict[str, Any]:
    """Export one character rig according to a unified Export Profile (default: Torchlight 60fps dual-tier)."""
    from .exporter import export_standalone_html

    prof = normalize_export_profile(profile)
    rdir = Path(rig_dir).resolve()
    odir = Path(out_dir).resolve()
    if require_approved:
        ensure_export_approved(rdir, health_report_path=health_report_path, user_signoff_path=user_signoff_path)
    odir.mkdir(parents=True, exist_ok=True)

    rep = json.loads((rdir / "rig-build-report.json").read_text(encoding="utf-8"))
    char_name = str(character or rep.get("character") or "character")
    duration = float(rep.get("duration", 8.0))
    fps = int(prof["fps"])

    renderer = SpineRenderer(rdir)
    tr = compute_canvas_transform(renderer.width, renderer.height, prof)
    is_4k = max(tr["out_w"], tr["out_h"]) >= 3000

    formats = set(prof["formats"])
    files: List[Dict[str, str]] = []
    notices: List[str] = []
    if tr["upscaled"]:
        notices.append(
            f"原图分辨率 {tr['src_w']}×{tr['src_h']} 小于目标画布，已按 Lanczos 高质量等比缩放至 {tr['scaled_w']}×{tr['scaled_h']}。"
        )

    torchlight_meta: Optional[Dict[str, Any]] = None
    if "torchlight_web" in formats:
        tl_res = export_torchlight_web_package(
            rdir,
            odir,
            character=char_name,
            label=label or char_name,
            profile=prof,
            ffmpeg_bin=ffmpeg_bin,
            max_frames=max_frames,
            user_signoff_path=user_signoff_path,
            require_approved=False,
        )
        torchlight_meta = tl_res
        for flabel, key in (
            ("PC 高清 60帧预乘透明视频 (idle-high-packed.mp4)", "high_mp4"),
            ("手机 60帧预乘透明视频 (idle-mobile-packed.mp4)", "mobile_mp4"),
            ("透明无损首帧封面 (poster.webp)", "poster_webp"),
            ("官网标准元数据 (media.json)", "media_json"),
        ):
            fp = tl_res[key]
            files.append({"format": "torchlight_web", "label": flabel, "path": fp, "filename": Path(fp).name})

    need_frames = bool(formats & {"transparent_web", "preview_mp4"})
    if need_frames:
        ff = ffmpeg_bin or find_ffmpeg()
        if not ff:
            raise RuntimeError("未找到 ffmpeg，无法导出视频格式。")
        total = max(1, int(round(duration * fps)))
        if max_frames is not None:
            total = min(total, int(max_frames))
        workers = _resolve_worker_count(int(prof.get("render_workers", 0)), total)
        if workers <= 1:
            fitted_frames = [
                fit_frame_to_profile(renderer.frame((idx / float(fps)) % duration), prof)[0]
                for idx in range(total)
            ]
        else:
            _ensure_repo_in_pythonpath()
            tasks = [(idx, (idx / float(fps)) % duration, prof) for idx in range(total)]
            fitted_frames = [None] * total  # type: ignore[list-item]
            with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init_renderer, initargs=(str(rdir),)) as pool:
                for idx, fitted in pool.map(_worker_render_fitted_frame, tasks):
                    fitted_frames[idx] = fitted

        if "transparent_web" in formats:
            web_res = _export_transparent_frames(
                fitted_frames,
                odir,
                character=char_name,
                fps=fps,
                is_4k=is_4k,
                include_av1=bool(prof.get("include_av1", False)),
                ffmpeg_bin=ff,
            )
            for flabel, key in (
                ("透明网页视频 VP9 (.webm)", "alpha_vp9_webm"),
                ("打包透明视频 H.264 (.mp4)", "packed_h264_mp4"),
                ("打包透明视频 AV1 (.mp4)", "packed_av1_mp4"),
                ("透明首帧封面 (.webp)", "poster_webp"),
            ):
                fp = web_res.get(key)
                if fp:
                    files.append({"format": "transparent_web", "label": flabel, "path": fp, "filename": Path(fp).name})

        if "preview_mp4" in formats:
            dark_mp4 = write_preview_mp4(
                fitted_frames, odir / f"{char_name}-preview-dark.mp4", bg=(17, 17, 17), fps=fps, ffmpeg_bin=ff
            )
            light_mp4 = write_preview_mp4(
                fitted_frames, odir / f"{char_name}-preview-light.mp4", bg=(221, 221, 221), fps=fps, ffmpeg_bin=ff
            )
            files.append({"format": "preview_mp4", "label": "深底预览视频 (.mp4)", "path": str(dark_mp4), "filename": dark_mp4.name})
            files.append({"format": "preview_mp4", "label": "浅底预览视频 (.mp4)", "path": str(light_mp4), "filename": light_mp4.name})

    if "spine_package" in formats or "standalone_html" in formats:
        with tempfile.TemporaryDirectory() as tmp:
            scaled_rig_dir = Path(tmp) / "rig"
            scale_spine_rig_to_profile(rdir, scaled_rig_dir, prof)
            scaled_rep = json.loads((scaled_rig_dir / "rig-build-report.json").read_text(encoding="utf-8"))
            spine_file = str(scaled_rep["spineFile"])
            hr_path = Path(health_report_path).resolve() if health_report_path else (rdir / "health-report.json")

            if "spine_package" in formats:
                zip_path = odir / f"{char_name}-spine42.zip"
                with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                    zf.write(scaled_rig_dir / spine_file, arcname=spine_file)
                    zf.write(scaled_rig_dir / "rig-build-report.json", arcname="rig-build-report.json")
                    if hr_path.is_file():
                        zf.write(hr_path, arcname="health-report.json")
                    if user_signoff_path is not None and Path(user_signoff_path).is_file():
                        zf.write(user_signoff_path, arcname="user-signoff.json")
                    for img in sorted((scaled_rig_dir / "images").glob("*.png")):
                        zf.write(img, arcname=f"images/{img.name}")
                files.append({"format": "spine_package", "label": "Spine 4.2 工程包 (.zip)", "path": str(zip_path), "filename": zip_path.name})

            if "standalone_html" in formats:
                html_path = odir / f"{char_name}-standalone-preview.html"
                export_standalone_html(
                    scaled_rig_dir,
                    html_path,
                    health_report_path=hr_path if hr_path.is_file() else None,
                    require_approved=False,
                )
                files.append({"format": "standalone_html", "label": "单文件交互预览 (.html)", "path": str(html_path), "filename": html_path.name})

    result: Dict[str, Any] = {
        "character": char_name,
        "output_dir": str(odir),
        "profile": prof,
        "source_size": [tr["src_w"], tr["src_h"]],
        "target_size": torchlight_meta["master_size"] if torchlight_meta else [tr["out_w"], tr["out_h"]],
        "scaled_size": [tr["scaled_w"], tr["scaled_h"]],
        "scale": tr["scale"],
        "upscaled": tr["upscaled"],
        "fps": fps,
        "files": files,
        "notices": notices,
    }
    if torchlight_meta:
        result["torchlight_web"] = {
            "high_encoded_size": torchlight_meta["high_encoded_size"],
            "mobile_encoded_size": torchlight_meta["mobile_encoded_size"],
            "workers_used": torchlight_meta["workers_used"],
            "media_json": torchlight_meta["media_json"],
        }
    return result
