"""Default Spine 4.2 package exporter and on-demand transparent web video exporter."""
from __future__ import annotations

import json
import subprocess
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from src.render.renderer import SpineRenderer, find_ffmpeg, straight_alpha, write_preview_mp4


def _render_loop_frames(
    renderer: SpineRenderer,
    duration: float,
    fps: int,
    max_frames: Optional[int] = None,
) -> List[np.ndarray]:
    total = max(1, int(round(duration * fps)))
    if max_frames is not None:
        total = min(total, max(1, int(max_frames)))
    return [renderer.frame(duration * i / float(total)) for i in range(total)]


def export_default_package(
    rig_dir: Union[str, Path],
    out_dir: Union[str, Path],
    *,
    health_report_path: Optional[Union[str, Path]] = None,
    render_previews: bool = True,
    preview_fps: int = 15,
    preview_max_frames: Optional[int] = None,
    ffmpeg_bin: Optional[str] = None,
) -> Dict[str, Any]:
    """Export default deliverable: Spine 4.2 project zip + dark/light preview MP4s.

    The Spine 4.2 zip includes:
    - `<character>.spine.json`
    - `images/*.png`
    - `rig-build-report.json`
    - `health-report.json` (if present)
    """
    rdir = Path(rig_dir).resolve()
    odir = Path(out_dir).resolve()
    odir.mkdir(parents=True, exist_ok=True)

    rep_path = rdir / "rig-build-report.json"
    if not rep_path.is_file():
        raise FileNotFoundError(f"rig-build-report.json not found in {rdir}")
    rep = json.loads(rep_path.read_text(encoding="utf-8"))
    character = str(rep.get("character", "character"))
    spine_file = str(rep["spineFile"])
    spine_path = rdir / spine_file
    if not spine_path.is_file():
        raise FileNotFoundError(f"Spine JSON not found: {spine_path}")

    hr_path = Path(health_report_path).resolve() if health_report_path else (rdir / "health-report.json")
    zip_path = odir / f"{character}-spine42.zip"
    zip_entries: List[str] = []

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(spine_path, arcname=spine_file)
        zip_entries.append(spine_file)
        zf.write(rep_path, arcname="rig-build-report.json")
        zip_entries.append("rig-build-report.json")
        if hr_path.is_file():
            zf.write(hr_path, arcname="health-report.json")
            zip_entries.append("health-report.json")
        img_dir = rdir / "images"
        for img in sorted(img_dir.glob("*.png")):
            arc = f"images/{img.name}"
            zf.write(img, arcname=arc)
            zip_entries.append(arc)

    previews: Dict[str, Optional[str]] = {"dark": None, "light": None}
    preview_notice: Optional[str] = None
    if render_previews:
        ff = ffmpeg_bin or find_ffmpeg()
        if ff:
            renderer = SpineRenderer(rdir)
            duration = float(rep.get("duration", 8.0))
            frames = _render_loop_frames(renderer, duration, preview_fps, max_frames=preview_max_frames)
            dark_mp4 = write_preview_mp4(frames, odir / f"{character}-preview-dark.mp4", bg=(17, 17, 17), fps=preview_fps, ffmpeg_bin=ff)
            light_mp4 = write_preview_mp4(frames, odir / f"{character}-preview-light.mp4", bg=(221, 221, 221), fps=preview_fps, ffmpeg_bin=ff)
            previews["dark"] = str(dark_mp4)
            previews["light"] = str(light_mp4)
        else:
            preview_notice = "ffmpeg binary not found; skipped MP4 preview rendering."

    return {
        "character": character,
        "zip_path": str(zip_path),
        "zip_entries": zip_entries,
        "previews": previews,
        "preview_notice": preview_notice,
    }


def _pack_sbs_rgb_alpha(frame_pm: np.ndarray) -> np.ndarray:
    """Pack premultiplied RGBA (H,W,4) side-by-side into RGB (H, 2W, 3): left=RGB, right=Alpha."""
    h, w = frame_pm.shape[:2]
    ew = w + (w % 2)
    eh = h + (h % 2)
    canvas = np.zeros((eh, ew * 2, 3), dtype=np.uint8)
    canvas[:h, :w] = frame_pm[..., :3]
    a = frame_pm[..., 3:4]
    canvas[:h, ew : ew + w] = np.repeat(a, 3, axis=2)
    return canvas


def _has_ffmpeg_encoder(ff: str, encoder: str) -> bool:
    res = subprocess.run([ff, "-hide_banner", "-encoders"], capture_output=True, text=True, check=False)
    return encoder in (res.stdout or "")


def export_transparent_web(
    rig_dir: Union[str, Path],
    out_dir: Union[str, Path],
    *,
    fps: int = 30,
    max_frames: Optional[int] = None,
    ffmpeg_bin: Optional[str] = None,
) -> Dict[str, Any]:
    """On-demand transparent web video export:
    - side-by-side packed H.264 MP4 (`idle-packed-h264.mp4`)
    - side-by-side packed AV1 MP4 (`idle-packed-av1.mp4`, if AV1 encoder is available)
    - native VP9 alpha WebM (`idle-alpha-vp9.webm`, `yuva420p`)
    - frame 0 straight-alpha `poster.webp`
    """
    rdir = Path(rig_dir).resolve()
    odir = Path(out_dir).resolve()
    odir.mkdir(parents=True, exist_ok=True)

    ff = ffmpeg_bin or find_ffmpeg()
    if not ff:
        raise RuntimeError("ffmpeg binary required for export_transparent_web")

    rep = json.loads((rdir / "rig-build-report.json").read_text(encoding="utf-8"))
    duration = float(rep.get("duration", 8.0))
    renderer = SpineRenderer(rdir)
    frames = _render_loop_frames(renderer, duration, fps, max_frames=max_frames)

    # 1. poster.webp (frame 0 straight alpha)
    poster_path = odir / "poster.webp"
    Image.fromarray(straight_alpha(frames[0])).save(poster_path, format="WEBP", lossless=True)

    # 2. Side-by-side packed H.264 MP4
    sbs_sample = _pack_sbs_rgb_alpha(frames[0])
    sh, sw = sbs_sample.shape[:2]
    h264_path = odir / "idle-packed-h264.mp4"
    cmd_h264 = [
        ff, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{sw}x{sh}", "-r", str(fps), "-i", "-",
        "-vf", "format=yuv420p", "-c:v", "libx264", "-preset", "medium", "-crf", "16",
        "-movflags", "+faststart", str(h264_path),
    ]
    with subprocess.Popen(cmd_h264, stdin=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
        assert proc.stdin is not None
        for f in frames:
            proc.stdin.write(_pack_sbs_rgb_alpha(f).tobytes())
        proc.stdin.close()
        err = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        if proc.wait() != 0:
            raise RuntimeError(f"H.264 packed export failed: {err}")

    # 3. Side-by-side packed AV1 MP4 (optional if encoder present)
    av1_path: Optional[Path] = None
    av1_enc = "libsvtav1" if _has_ffmpeg_encoder(ff, "libsvtav1") else ("libaom-av1" if _has_ffmpeg_encoder(ff, "libaom-av1") else None)
    if av1_enc:
        av1_path = odir / "idle-packed-av1.mp4"
        cmd_av1 = [
            ff, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{sw}x{sh}", "-r", str(fps), "-i", "-",
            "-vf", "format=yuv420p", "-c:v", av1_enc, "-crf", "28",
            "-movflags", "+faststart", str(av1_path),
        ]
        if av1_enc == "libaom-av1":
            cmd_av1.insert(-2, "-cpu-used")
            cmd_av1.insert(-2, "8")
        with subprocess.Popen(cmd_av1, stdin=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
            assert proc.stdin is not None
            for f in frames:
                proc.stdin.write(_pack_sbs_rgb_alpha(f).tobytes())
            proc.stdin.close()
            if proc.stderr:
                proc.stderr.read()
            if proc.wait() != 0:
                av1_path = None

    # 4. Native VP9 alpha WebM (yuva420p)
    h, w = frames[0].shape[:2]
    ew, eh = w + (w % 2), h + (h % 2)
    vp9_path = odir / "idle-alpha-vp9.webm"
    cmd_vp9 = [
        ff, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{ew}x{eh}", "-r", str(fps), "-i", "-",
        "-vf", "format=yuva420p", "-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "24",
        "-auto-alt-ref", "0", str(vp9_path),
    ]
    with subprocess.Popen(cmd_vp9, stdin=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
        assert proc.stdin is not None
        for f in frames:
            st = straight_alpha(f)
            if ew != w or eh != h:
                pad = np.zeros((eh, ew, 4), dtype=np.uint8)
                pad[:h, :w] = st
                st = pad
            proc.stdin.write(st.tobytes())
        proc.stdin.close()
        err = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        if proc.wait() != 0:
            raise RuntimeError(f"VP9 alpha WebM export failed: {err}")

    return {
        "packed_h264_mp4": str(h264_path),
        "packed_av1_mp4": str(av1_path) if av1_path else None,
        "alpha_vp9_webm": str(vp9_path),
        "poster_webp": str(poster_path),
        "frames": len(frames),
        "fps": fps,
    }
