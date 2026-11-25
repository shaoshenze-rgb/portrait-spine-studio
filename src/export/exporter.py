"""Default Spine 4.2 package exporter and on-demand transparent web video exporter."""
from __future__ import annotations

import base64
import json
import subprocess
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from src.render.renderer import SpineRenderer, find_ffmpeg, straight_alpha, write_preview_mp4
from src.gates.review_binding import ensure_export_approved

_PLAYER_ROOT = Path(__file__).resolve().parents[1] / "player"


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


def _player_bundle_js() -> str:
    chunks: List[str] = []
    drop_prefixes = (
        "import { Spine42Skeleton } from './spine42-runtime.js';",
        "import { SpineCanvasRenderer } from './canvas-renderer.js';",
    )
    for name in ("spine42-runtime.js", "canvas-renderer.js", "player-widget.js"):
        source = (_PLAYER_ROOT / name).read_text(encoding="utf-8")
        if name == "player-widget.js":
            kept = []
            for line in source.splitlines(True):
                if any(line.startswith(prefix) for prefix in drop_prefixes):
                    continue
                kept.append(line)
            source = "".join(kept)
        chunks.append("\n/* ===== " + name + " ===== */\n" + source)
    return "\n".join(chunks)


def _js_literal(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return payload.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def export_standalone_html(
    rig_dir: Union[str, Path],
    out_path: Union[str, Path],
    *,
    health_report_path: Optional[Union[str, Path]] = None,
    require_approved: bool = True,
    user_signoff_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Write a zero-request single-file HTML preview with inlined runtime, JSON and PNGs."""
    rdir = Path(rig_dir).resolve()
    dest = Path(out_path).resolve()
    if require_approved:
        ensure_export_approved(rdir, health_report_path=health_report_path, user_signoff_path=user_signoff_path)

    rep_path = rdir / "rig-build-report.json"
    if not rep_path.is_file():
        raise FileNotFoundError("rig-build-report.json not found in " + str(rdir))
    rep = json.loads(rep_path.read_text(encoding="utf-8"))
    character = str(rep.get("character", "character"))
    spine_file = str(rep["spineFile"])
    spine_path = rdir / spine_file
    if not spine_path.is_file():
        raise FileNotFoundError("Spine JSON not found: " + str(spine_path))
    spine_json = json.loads(spine_path.read_text(encoding="utf-8"))

    images: Dict[str, str] = {}
    img_dir = rdir / "images"
    for img in sorted(img_dir.glob("*.png")):
        uri = "data:image/png;base64," + base64.b64encode(img.read_bytes()).decode("ascii")
        images[img.name] = uri
        images[img.stem] = uri
    if not images:
        raise FileNotFoundError("No PNG attachments in " + str(img_dir))

    hr_path = Path(health_report_path).resolve() if health_report_path else (rdir / "health-report.json")
    health = None
    if hr_path.is_file():
        try:
            health = json.loads(hr_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            health = None

    safe_title = character.replace("<", "").replace(">", "")
    html = _standalone_html_document(
        safe_title,
        _js_literal(character),
        _js_literal(spine_json),
        _js_literal(images),
        _js_literal(health),
        _player_bundle_js(),
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(html, encoding="utf-8")
    png_names = [k for k in images if k.endswith(".png")]
    return {"character": character, "html_path": str(dest), "image_count": len(png_names)}


def _standalone_html_document(title: str, character_js: str, spine_js: str,
                              images_js: str, health_js: str, bundle_js: str) -> str:
    parts = [
        "<!DOCTYPE html>",
        '<html lang="zh-CN">',
        "<head>",
        '<meta charset="UTF-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">',
        "<title>" + title + " — 单文件预览</title>",
        "<style>",
        ":root { --bg:#101218; --text:#e8edf7; --muted:#93a0b8; --line:#2c3446; --accent:#38bdf8; --btn:#243044; }",
        "* { box-sizing: border-box; }",
        "html,body { margin:0; padding:0; background:var(--bg); color:var(--text); font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; }",
        ".wrap { max-width:960px; margin:0 auto; padding:12px; display:flex; flex-direction:column; gap:10px; min-height:100vh; }",
        "header { display:flex; flex-wrap:wrap; justify-content:space-between; gap:8px; align-items:baseline; }",
        "h1 { font-size:16px; margin:0; }",
        ".muted { color:var(--muted); font-size:12px; }",
        ".stage { background:#0b0d12; border:1px solid var(--line); border-radius:12px; overflow:hidden; display:flex; align-items:center; justify-content:center; min-height:240px; flex:1; }",
        "canvas { width:100%; height:auto; max-height:78vh; display:block; }",
        ".bar { display:flex; flex-wrap:wrap; gap:8px; align-items:center; }",
        "button,label { background:var(--btn); color:var(--text); border:1px solid var(--line); border-radius:8px; padding:8px 10px; font-size:13px; }",
        "button { cursor:pointer; }",
        "button:hover { border-color:var(--accent); }",
        "@media (max-width:430px) { .wrap{padding:8px;} h1{font-size:14px;} canvas{max-height:62vh;} button,label{flex:1 1 calc(50% - 8px); text-align:center;} }",
        "</style>",
        "</head>",
        "<body>",
        '<div class="wrap">',
        "<header>",
        '<h1 id="title">单文件预览</h1>',
        '<div class="muted" id="meta"></div>',
        "</header>",
        '<div class="stage"><canvas id="stage" width="320" height="320"></canvas></div>',
        '<div class="bar">',
        '<button type="button" id="btnPlay">播放 / 暂停</button>',
        '<button type="button" id="btnReset">重置 t=0</button>',
        '<label><input type="checkbox" id="chkBones"> 骨骼线</label>',
        '<label><input type="checkbox" id="chkLight"> 浅底</label>',
        "</div></div>",
        '<script type="module">',
        bundle_js,
        "const CHARACTER = " + character_js + ";",
        "const SPINE_JSON = " + spine_js + ";",
        "const EMBEDDED_IMAGES = " + images_js + ";",
        "const HEALTH = " + health_js + ";",
        "function loadImage(uri) {",
        "  return new Promise((resolve, reject) => {",
        "    const img = new Image();",
        "    img.onload = () => resolve(img);",
        "    img.onerror = () => reject(new Error('贴图解码失败'));",
        "    img.src = uri;",
        "  });",
        "}",
        "const canvas = document.getElementById('stage');",
        "const player = new PortraitSpinePlayer({ canvas, preferWebGL2: true });",
        "const uniqueUris = [...new Set(Object.values(EMBEDDED_IMAGES))];",
        "const uriToImg = new Map();",
        "Promise.all(uniqueUris.map(async (uri) => { uriToImg.set(uri, await loadImage(uri)); })).then(() => {",
        "  const images = new Map();",
        "  for (const [name, uri] of Object.entries(EMBEDDED_IMAGES)) images.set(name, uriToImg.get(uri));",
        "  const info = player.loadRigData({ spineJson: SPINE_JSON, images, healthReport: HEALTH, name: CHARACTER, animationName: 'idle', autoPlay: true });",
        "  document.getElementById('title').textContent = CHARACTER + ' — 单文件预览';",
        "  document.getElementById('meta').textContent = (info.backend || '') + ' · ' + info.width + '×' + info.height;",
        "}).catch((err) => { document.getElementById('meta').textContent = String(err && err.message || err); });",
        "document.getElementById('btnPlay').addEventListener('click', () => player.togglePlay());",
        "document.getElementById('btnReset').addEventListener('click', () => { player.pause(); player.setTime(0); });",
        "document.getElementById('chkBones').addEventListener('change', (ev) => player.setShowBones(ev.target.checked));",
        "document.getElementById('chkLight').addEventListener('change', (ev) => player.setBackground(ev.target.checked ? 'light' : 'dark'));",
        "</script>",
        "</body></html>",
        "",
    ]
    return "\n".join(parts)


def export_default_package(
    rig_dir: Union[str, Path],
    out_dir: Union[str, Path],
    *,
    health_report_path: Optional[Union[str, Path]] = None,
    render_previews: bool = True,
    preview_fps: int = 15,
    preview_max_frames: Optional[int] = None,
    user_signoff_path: Optional[Union[str, Path]] = None,
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
    ensure_export_approved(rdir, health_report_path=health_report_path, user_signoff_path=user_signoff_path)
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
        if user_signoff_path is not None:
            zf.write(user_signoff_path, arcname="user-signoff.json")
            zip_entries.append("user-signoff.json")
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

    html_path = odir / f"{character}-standalone-preview.html"
    export_standalone_html(
        rdir,
        html_path,
        health_report_path=hr_path if hr_path.is_file() else None,
        require_approved=False,
    )

    return {
        "character": character,
        "zip_path": str(zip_path),
        "zip_entries": zip_entries,
        "previews": previews,
        "preview_notice": preview_notice,
        "standalone_html": str(html_path),
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
    user_signoff_path: Optional[Union[str, Path]] = None,
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
    ensure_export_approved(rdir, user_signoff_path=user_signoff_path)
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
