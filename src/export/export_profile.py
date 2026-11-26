"""Unified export profiles (including true 4K) and multi-format batch-ready exporter."""
from __future__ import annotations

import copy
import json
import math
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

from src.gates.review_binding import ensure_export_approved
from src.render.renderer import SpineRenderer, find_ffmpeg, straight_alpha, write_preview_mp4

LANCZOS = getattr(Image, "Resampling", Image).LANCZOS

SIZE_PRESETS: Dict[str, Dict[str, Any]] = {
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
    "transparent_web": "透明网页视频 (VP9 WebM + Packed H.264 MP4 + poster.webp)",
    "spine_package": "Spine 4.2 工程包 (.zip)",
    "preview_mp4": "深浅底预览视频 (.mp4)",
    "standalone_html": "单文件交互预览 (.html)",
}

DEFAULT_EXPORT_PROFILE: Dict[str, Any] = {
    "size_preset": "4k_portrait",
    "width": 2160,
    "height": 3840,
    "max_long_edge": 4096,
    "align": "bottom_center",
    "bottom_margin_ratio": 0.03,
    "upscale_policy": "allow_upscale",
    "formats": ["transparent_web"],
    "fps": 30,
    "include_av1": False,
}


def _even_dim(value: float, minimum: int = 2) -> int:
    iv = max(minimum, int(round(value)))
    if iv % 2 != 0:
        iv += 1
    return iv


def normalize_export_profile(raw: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Validate and normalize an export profile dictionary."""
    merged = copy.deepcopy(DEFAULT_EXPORT_PROFILE)
    if isinstance(raw, dict):
        merged.update({k: v for k, v in raw.items() if v is not None})

    preset_key = str(merged.get("size_preset") or "4k_portrait")
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

    policy = str(merged.get("upscale_policy") or "allow_upscale")
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

    fps = int(merged.get("fps", 30))
    if fps not in (15, 24, 30, 60):
        raise ValueError("导出帧率仅支持 15 / 24 / 30 / 60。")
    merged["fps"] = fps
    merged["include_av1"] = bool(merged.get("include_av1", False))

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

    # Fixed canvas size
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
    # Clamp Lanczos overshoot so premultiplied RGB never exceeds alpha
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


def _pack_sbs_rgb_alpha(frame_pm: np.ndarray) -> np.ndarray:
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
    return np.concatenate([rgb, alpha], axis=1)


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

    # 1. Native VP9 Alpha WebM (yuva420p) — crisp quality for 4K & multi-threaded
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

    # 2. Side-by-side packed H.264 MP4 (universal / Safari fallback)
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
    health_report_path: Optional[Union[str, Path]] = None,
    user_signoff_path: Optional[Union[str, Path]] = None,
    ffmpeg_bin: Optional[str] = None,
    max_frames: Optional[int] = None,
    require_approved: bool = True,
) -> Dict[str, Any]:
    """Export one character rig according to a unified Export Profile (supporting true 4K)."""
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

    need_frames = bool(formats & {"transparent_web", "preview_mp4"})
    fitted_frames: Optional[List[np.ndarray]] = None
    if need_frames:
        ff = ffmpeg_bin or find_ffmpeg()
        if not ff:
            raise RuntimeError("未找到 ffmpeg，无法导出视频格式。")
        total = max(1, int(round(duration * fps)))
        if max_frames is not None:
            total = min(total, int(max_frames))
        fitted_frames = []
        for idx in range(total):
            t = (idx / float(fps)) % duration
            raw_frame = renderer.frame(t)
            fitted, _ = fit_frame_to_profile(raw_frame, prof)
            fitted_frames.append(fitted)

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
            for label, key in (
                ("透明网页视频 VP9 (.webm)", "alpha_vp9_webm"),
                ("Safari 打包透明视频 H.264 (.mp4)", "packed_h264_mp4"),
                ("打包透明视频 AV1 (.mp4)", "packed_av1_mp4"),
                ("透明首帧封面 (.webp)", "poster_webp"),
            ):
                fp = web_res.get(key)
                if fp:
                    files.append({"format": "transparent_web", "label": label, "path": fp, "filename": Path(fp).name})

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

    return {
        "character": char_name,
        "output_dir": str(odir),
        "profile": prof,
        "source_size": [tr["src_w"], tr["src_h"]],
        "target_size": [tr["out_w"], tr["out_h"]],
        "scaled_size": [tr["scaled_w"], tr["scaled_h"]],
        "scale": tr["scale"],
        "upscaled": tr["upscaled"],
        "fps": fps,
        "files": files,
        "notices": notices,
    }
