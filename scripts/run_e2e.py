#!/usr/bin/env python3
"""End-to-end orchestrator for Portrait Spine Studio.

Runs:
  preprocess -> split_parts -> build_rig -> audit (4 hard gates + soft hints)
  -> default export -> preview.html

Supports:
  - Minimal-layer partial reruns: --partial-rerun motion | rig | split
  - Balanced intervention switch: --pause-after-split
  - On-demand transparent web video export: --export-web
  - Strict fail-stop with <out_dir>/stage-error.json and preserved intermediates
"""
from __future__ import annotations

import argparse
import html
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.export.exporter import export_default_package, export_transparent_web
from src.gates.audit import GateFailureError, run_audit
from src.mivo.adapter import StageRunner
from src.pipeline.build_rig import build_rig
from src.pipeline.preprocess import MAX_LONG_EDGE, preprocess_portrait
from src.pipeline.presets import list_presets
from src.pipeline.split_parts import split_parts
from src.render.renderer import find_ffmpeg


def _load_parts_spec(spec_or_path: Union[str, Path, Sequence[Dict[str, Any]], Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if isinstance(spec_or_path, (str, Path)):
        raw = json.loads(Path(spec_or_path).read_text(encoding="utf-8"))
    else:
        raw = spec_or_path

    if isinstance(raw, list):
        return [dict(p) for p in raw], {}
    if isinstance(raw, dict):
        parts = raw.get("parts")
        if not isinstance(parts, list) or not parts:
            raise ValueError("parts_spec dict must contain a non-empty 'parts' list")
        opts: Dict[str, Any] = {}
        if "seams" in raw and isinstance(raw["seams"], dict):
            seams_out = {}
            for k, v in raw["seams"].items():
                pair = tuple(k.split("|", 1)) if isinstance(k, str) and "|" in k else tuple(k)
                seams_out[pair] = v
            opts["seams"] = seams_out
        if "pair_bands" in raw and isinstance(raw["pair_bands"], dict):
            bands_out = {}
            for k, v in raw["pair_bands"].items():
                pair = tuple(k.split("|", 1)) if isinstance(k, str) and "|" in k else tuple(k)
                bands_out[pair] = int(v)
            opts["pair_bands"] = bands_out
        for k in ("opaque_threshold", "excl_px", "feather_px", "min_island_px"):
            if k in raw:
                opts[k] = int(raw[k])
        return [dict(p) for p in parts], opts
    raise ValueError("parts_spec must be a JSON list or object with 'parts'")


def _load_rig_spec(spec_or_path: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(spec_or_path, (str, Path)):
        raw = json.loads(Path(spec_or_path).read_text(encoding="utf-8"))
    else:
        raw = spec_or_path
    if not isinstance(raw, dict):
        raise ValueError("rig_spec must be a JSON object")
    return json.loads(json.dumps(raw))


def _relative_posix(target: Path, start: Path) -> Optional[str]:
    try:
        rel = os.path.relpath(target.resolve(), start.resolve())
        return Path(rel).as_posix()
    except ValueError:
        return None


def generate_preview_html(
    out_dir: Path,
    rig_dir: Path,
    rig_report: Dict[str, Any],
    health_report: Dict[str, Any],
) -> Path:
    """Generate a ready-to-open preview.html referencing src/player/ with embedded gate summary."""
    character = str(rig_report.get("character", "character"))
    spine_file = str(rig_report["spineFile"])
    spine_rel = f"./{rig_dir.name}/{spine_file}"
    health_rel = f"./{rig_dir.name}/health-report.json"

    viewer_path = REPO_ROOT / "src" / "player" / "standalone-viewer.html"
    widget_path = REPO_ROOT / "src" / "player" / "player-widget.js"
    viewer_rel = _relative_posix(viewer_path, out_dir)
    widget_rel = _relative_posix(widget_path, out_dir)

    viewer_href = f"{viewer_rel}?rig={html.escape(spine_rel)}" if viewer_rel else ""

    gate_rows: List[str] = []
    for gname, gdata in health_report.get("hardGates", {}).items():
        status = "PASS" if gdata.get("passed") else "FAIL"
        badge_cls = "pass" if gdata.get("passed") else "fail"
        measured = html.escape(json.dumps(gdata.get("measured", {}), ensure_ascii=False))
        gate_rows.append(
            f'<tr><td><code>{html.escape(gname)}</code></td>'
            f'<td><span class="badge {badge_cls}">{status}</span></td>'
            f'<td><code>{measured}</code></td></tr>'
        )

    soft_rows: List[str] = []
    for lm, ldata in health_report.get("softHints", {}).get("landmarks", {}).items():
        st = html.escape(str(ldata.get("status", "na")))
        ptp = ldata.get("ptpPx")
        ptp_str = "--" if ptp is None else f"{ptp:.2f} px"
        rng = ldata.get("refRangePx") or []
        rng_str = f"[{rng[0]}, {rng[1]}] px" if len(rng) == 2 else "--"
        soft_rows.append(
            f'<tr><td><code>{html.escape(lm)}</code></td>'
            f'<td><code>{ptp_str}</code></td>'
            f'<td><code>{rng_str}</code></td>'
            f'<td><span class="badge soft-{st}">{st}</span></td></tr>'
        )

    overall = str(health_report.get("status", "unknown")).upper()
    overall_cls = "pass" if overall == "PASS" else "fail"
    player_module_js = json.dumps(widget_rel or "./src/player/player-widget.js")

    content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Portrait Spine Studio Preview — {html.escape(character)}</title>
<style>
  :root {{
    --bg: #0f1116; --panel: #161922; --elev: #1e222e; --border: #2a3040;
    --text: #e6eaf2; --muted: #939db4; --accent: #38bdf8;
    --pass: #22c55e; --fail: #ef4444; --warn: #eab308;
    --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: var(--bg); color: var(--text); font-family: var(--sans); font-size: 13px; }}
  header {{ display: flex; align-items: center; justify-content: space-between; padding: 12px 18px; background: var(--panel); border-bottom: 1px solid var(--border); gap: 12px; }}
  .title {{ font-weight: 700; font-size: 15px; display: flex; align-items: center; gap: 10px; }}
  .layout {{ display: grid; grid-template-columns: 1fr 380px; min-height: calc(100vh - 52px); }}
  .stage {{ position: relative; display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 16px; background: radial-gradient(circle at 50% 45%, #1c2230 0%, #0f1116 70%); }}
  canvas {{ width: 100%; height: 72vh; max-height: 760px; border: 1px solid var(--border); border-radius: 10px; background: #12151d; }}
  .controls {{ margin-top: 10px; display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }}
  button, a.btn {{ background: var(--elev); color: var(--text); border: 1px solid var(--border); border-radius: 6px; padding: 6px 12px; text-decoration: none; cursor: pointer; font-size: 12px; }}
  button:hover, a.btn:hover {{ border-color: var(--accent); }}
  aside {{ background: var(--panel); border-left: 1px solid var(--border); padding: 16px; overflow-y: auto; }}
  h2 {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); margin: 14px 0 8px; }}
  h2:first-child {{ margin-top: 0; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  th, td {{ text-align: left; padding: 6px 6px; border-bottom: 1px solid var(--border); vertical-align: top; }}
  code {{ font-family: var(--mono); font-size: 11px; color: #cbd5e1; word-break: break-all; }}
  .badge {{ display: inline-block; padding: 2px 7px; border-radius: 999px; font-family: var(--mono); font-size: 11px; font-weight: 600; }}
  .badge.pass, .badge.soft-ok {{ background: rgba(34,197,94,0.16); color: var(--pass); }}
  .badge.fail {{ background: rgba(239,68,68,0.18); color: var(--fail); }}
  .badge.soft-warn_low, .badge.soft-warn_high {{ background: rgba(234,179,8,0.16); color: var(--warn); }}
  .note {{ color: var(--muted); font-size: 12px; line-height: 1.5; margin-top: 10px; }}
  #playerStatus {{ color: var(--muted); font-family: var(--mono); font-size: 11px; }}
  @media (max-width: 960px) {{ .layout {{ grid-template-columns: 1fr; }} aside {{ border-left: none; border-top: 1px solid var(--border); }} }}
</style>
</head>
<body>
<header>
  <div class="title">
    <span>Portrait Spine Studio</span>
    <code>{html.escape(character)}</code>
    <span class="badge {overall_cls}">GATES: {overall}</span>
  </div>
  <div style="display:flex; gap:8px; align-items:center;">
    {f'<a class="btn" href="{viewer_href}">Open Full Standalone Viewer</a>' if viewer_href else ''}
    <a class="btn" href="{html.escape(spine_rel)}">Spine JSON</a>
    <a class="btn" href="{html.escape(health_rel)}">Health Report</a>
  </div>
</header>
<div class="layout">
  <main class="stage">
    <canvas id="previewCanvas" width="720" height="720"></canvas>
    <div class="controls">
      <button id="btnPlayPause" type="button">Pause</button>
      <button id="btnReset" type="button">Reset t=0</button>
      <button id="btnBones" type="button">Toggle Bones</button>
      <span id="playerStatus">Loading Spine 4.2 runtime...</span>
    </div>
    <p class="note">
      Serve this folder over HTTP (for example <code>python3 -m http.server</code>) for live WebGL2/Canvas2D playback.
    </p>
  </main>
  <aside>
    <h2>4 Hard Quality Gates</h2>
    <table>
      <thead><tr><th>Gate</th><th>Status</th><th>Measured</th></tr></thead>
      <tbody>{"".join(gate_rows)}</tbody>
    </table>
    <h2>Soft Displacement Hints (Advisory)</h2>
    <table>
      <thead><tr><th>Landmark</th><th>PTP</th><th>Ref</th><th>Status</th></tr></thead>
      <tbody>{"".join(soft_rows)}</tbody>
    </table>
    <h2>Build Info</h2>
    <p class="note">
      Preset: <code>{html.escape(str(rig_report.get("preset", "")))}</code> |
      Gain: <code>{html.escape(str(rig_report.get("gain", 1.0)))}</code> |
      Duration: <code>{html.escape(str(rig_report.get("duration", 8.0)))}s</code>
    </p>
  </aside>
</div>
<script type="module">
  const statusEl = document.getElementById('playerStatus');
  const canvas = document.getElementById('previewCanvas');
  const spineUrl = {json.dumps(spine_rel)};
  const widgetModuleUrl = {player_module_js};
  try {{
    const mod = await import(widgetModuleUrl);
    const runtimeMod = await import(new URL('./spine42-runtime.js', new URL(widgetModuleUrl, window.location.href)));
    const rendererMod = await import(new URL('./canvas-renderer.js', new URL(widgetModuleUrl, window.location.href)));
    const resp = await fetch(spineUrl);
    if (!resp.ok) throw new Error('HTTP ' + resp.status + ' loading ' + spineUrl);
    const skJson = await resp.json();
    const sk = new runtimeMod.Spine42Skeleton(skJson);
    const renderer = new rendererMod.SpineCanvasRenderer(canvas);
    const baseDir = new URL('./', new URL(spineUrl, window.location.href));
    const images = {{}};
    const names = new Set((skJson.slots || []).map(s => s.attachment).filter(Boolean));
    await Promise.all([...names].map(async (name) => {{
      const img = new Image();
      img.src = new URL('./images/' + name + '.png', baseDir).href;
      await img.decode();
      images[name] = img;
    }}));
    let playing = true;
    let showBones = false;
    let t = 0;
    let last = performance.now();
    const dur = sk.getAnimationDuration('idle') || 8.0;
    document.getElementById('btnPlayPause').onclick = (e) => {{ playing = !playing; e.target.textContent = playing ? 'Pause' : 'Play'; }};
    document.getElementById('btnReset').onclick = () => {{ t = 0; }};
    document.getElementById('btnBones').onclick = () => {{ showBones = !showBones; }};
    statusEl.textContent = 'Live playback (' + renderer.backend + ')';
    function tick(now) {{
      const dt = Math.min(0.1, (now - last) / 1000);
      last = now;
      if (playing) t = (t + dt) % dur;
      sk.apply('idle', t);
      renderer.render(sk, images, {{ showBones }});
      requestAnimationFrame(tick);
    }}
    requestAnimationFrame(tick);
  }} catch (err) {{
    statusEl.textContent = 'Interactive player requires HTTP server (' + err.message + ')';
  }}
</script>
</body>
</html>
"""
    preview_path = out_dir / "preview.html"
    preview_path.write_text(content, encoding="utf-8")
    return preview_path


def run_e2e_pipeline(
    *,
    image_path: Optional[Union[str, Path]],
    parts_spec: Optional[Union[str, Path, Sequence[Dict[str, Any]], Dict[str, Any]]],
    rig_spec: Union[str, Path, Dict[str, Any]],
    out_dir: Union[str, Path],
    preset: str = "agile_full",
    gain: float = 1.0,
    partial_rerun: Optional[str] = None,
    pause_after_split: bool = False,
    max_long_edge: int = MAX_LONG_EDGE,
    sample_count: int = 16,
    stress_gain: float = 1.6,
    stress_samples: int = 8,
    render_previews: bool = True,
    preview_fps: int = 15,
    preview_max_frames: Optional[int] = None,
    export_web: bool = False,
    web_fps: int = 30,
    web_max_frames: Optional[int] = None,
    raise_on_error: bool = True,
) -> Dict[str, Any]:
    """Run the full or minimal-layer partial-rerun pipeline with strict fail-stop semantics."""
    if partial_rerun not in (None, "motion", "rig", "split"):
        raise ValueError(f"Unsupported partial_rerun={partial_rerun!r}; expected None, 'motion', 'rig', or 'split'")

    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    preprocessed_png = out / "preprocessed.png"
    preprocess_report_path = out / "preprocess-report.json"
    parts_dir = out / "parts"
    rig_dir = out / "rig"
    export_dir = out / "export"

    runner = StageRunner(out)
    if preprocessed_png.is_file():
        runner.record_intermediate("preprocess.output_path", preprocessed_png)
    if (parts_dir / "parts.json").is_file():
        runner.record_intermediate("split_parts.parts_json", parts_dir / "parts.json")

    rig_spec_dict = _load_rig_spec(rig_spec)
    executed_stages: List[str] = []
    skipped_stages: List[str] = []
    stages: List[Tuple[str, Callable[[], Any]]] = []

    if partial_rerun in ("motion", "rig"):
        skipped_stages.extend(["preprocess", "split_parts"])

        def _verify_cached_parts() -> Dict[str, Any]:
            pj = parts_dir / "parts.json"
            if not pj.is_file():
                raise FileNotFoundError(
                    f"Partial rerun '{partial_rerun}' requires existing parts/parts.json at {pj}"
                )
            return {"parts_json": str(pj), "reused": True}

        stages.append(("verify_cached_parts", _verify_cached_parts))
    elif partial_rerun == "split":
        skipped_stages.append("preprocess")
        if parts_spec is None:
            raise ValueError("parts_spec is required for --partial-rerun split")
        parts_list, split_opts = _load_parts_spec(parts_spec)

        def _stage_split_incremental() -> Dict[str, Any]:
            src = preprocessed_png if preprocessed_png.is_file() else (Path(image_path).resolve() if image_path else None)
            if src is None or not src.is_file():
                raise FileNotFoundError(f"Preprocessed portrait missing at {preprocessed_png}")
            return split_parts(src, parts_list, parts_dir, **split_opts)

        stages.append(("split_parts", _stage_split_incremental))
    else:
        if image_path is None:
            raise ValueError("image_path is required for a full pipeline run")
        if parts_spec is None:
            raise ValueError("parts_spec is required for a full pipeline run")
        parts_list, split_opts = _load_parts_spec(parts_spec)

        def _stage_preprocess() -> Dict[str, Any]:
            res = preprocess_portrait(image_path, preprocessed_png, max_long_edge=max_long_edge)
            preprocess_report_path.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
            return res

        def _stage_split() -> Dict[str, Any]:
            return split_parts(preprocessed_png, parts_list, parts_dir, **split_opts)

        stages.append(("preprocess", _stage_preprocess))
        stages.append(("split_parts", _stage_split))

    if not pause_after_split:
        def _stage_build_rig() -> Dict[str, Any]:
            return build_rig(parts_dir, rig_spec_dict, rig_dir, preset=preset, gain=gain)

        def _stage_audit() -> Dict[str, Any]:
            return run_audit(
                parts_dir,
                rig_dir,
                sample_count=sample_count,
                stress_gain=stress_gain,
                stress_samples=stress_samples,
                raise_on_fail=True,
            )

        def _stage_export() -> Dict[str, Any]:
            ff = find_ffmpeg() if (render_previews or export_web) else None
            default_res = export_default_package(
                rig_dir,
                export_dir,
                render_previews=bool(render_previews and ff),
                preview_fps=preview_fps,
                preview_max_frames=preview_max_frames,
                ffmpeg_bin=ff,
            )
            web_res = None
            if export_web:
                web_res = export_transparent_web(
                    rig_dir,
                    export_dir / "web",
                    fps=web_fps,
                    max_frames=web_max_frames,
                    ffmpeg_bin=ff,
                )
            return {"default": default_res, "web": web_res}

        def _stage_preview() -> Dict[str, Any]:
            rig_rep = json.loads((rig_dir / "rig-build-report.json").read_text(encoding="utf-8"))
            health_rep = json.loads((rig_dir / "health-report.json").read_text(encoding="utf-8"))
            preview_path = generate_preview_html(out, rig_dir, rig_rep, health_rep)
            return {"output_path": str(preview_path)}

        stages.append(("build_rig", _stage_build_rig))
        stages.append(("audit", _stage_audit))
        stages.append(("export", _stage_export))
        stages.append(("preview", _stage_preview))
    else:
        skipped_stages.extend(["build_rig", "audit", "export", "preview"])

    run_res = runner.run_stages(stages, raise_on_error=raise_on_error)
    if run_res.get("status") != "ok":
        return run_res

    executed_stages = list(run_res["completedStages"])
    mode = f"partial:{partial_rerun}" if partial_rerun else "full"
    status = "paused_after_split" if pause_after_split else "ok"

    summary: Dict[str, Any] = {
        "status": status,
        "mode": mode,
        "preset": preset,
        "gain": float(gain),
        "executedStages": executed_stages,
        "skippedStages": skipped_stages,
        "artifacts": {
            "outDir": str(out),
            "preprocessed": str(preprocessed_png) if preprocessed_png.is_file() else None,
            "partsDir": str(parts_dir),
            "partMap": str(parts_dir / "review" / "part_map.png") if (parts_dir / "review" / "part_map.png").is_file() else None,
            "rigDir": str(rig_dir) if not pause_after_split else None,
            "healthReport": str(rig_dir / "health-report.json") if not pause_after_split else None,
            "rigBuildReport": str(rig_dir / "rig-build-report.json") if not pause_after_split else None,
            "exportDir": str(export_dir) if not pause_after_split else None,
            "previewHtml": str(out / "preview.html") if not pause_after_split else None,
        },
        "results": {
            k: v for k, v in run_res["results"].items()
            if k in ("preprocess", "audit", "export", "preview", "verify_cached_parts")
        },
    }
    if "split_parts" in run_res["results"]:
        sp = run_res["results"]["split_parts"]
        summary["results"]["split_parts"] = {
            "rest_diff": sp["rest_diff"],
            "cache_hits": sp["cache_hits"],
            "part_count": len(sp["parts_json"]["parts"]),
        }
    if "build_rig" in run_res["results"]:
        br = run_res["results"]["build_rig"]
        summary["results"]["build_rig"] = {
            "spine_path": br["spine_path"],
            "report_path": br["report_path"],
        }

    (out / "pipeline-summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Portrait Spine Studio E2E pipeline runner")
    p.add_argument("--image", help="Path to input 2D portrait image (required for full run)")
    p.add_argument("--parts-spec", help="Path to parts_spec JSON (required for full run or --partial-rerun split)")
    p.add_argument("--rig-spec", required=True, help="Path to rig_spec JSON")
    p.add_argument("--out-dir", required=True, help="Output workspace directory")
    p.add_argument("--preset", default="agile_full", choices=list_presets(), help="Core idle preset")
    p.add_argument("--gain", type=float, default=1.0, help="Global animation amplitude multiplier")
    p.add_argument(
        "--partial-rerun",
        choices=("motion", "rig", "split"),
        default=None,
        help="Minimal rerun layer: motion/rig skip preprocess+split_parts; split skips preprocess and uses hash cache",
    )
    p.add_argument("--pause-after-split", action="store_true", help="Pause after split_parts for part/mask inspection")
    p.add_argument("--max-long-edge", type=int, default=MAX_LONG_EDGE, help="Max portrait long edge before downscaling")
    p.add_argument("--sample-count", type=int, default=16, help="Loop frames sampled during audit")
    p.add_argument("--stress-gain", type=float, default=1.6, help="Stress-test gain multiplier")
    p.add_argument("--stress-samples", type=int, default=8, help="Stress-test frames sampled during audit")
    p.add_argument("--skip-previews", action="store_true", help="Skip MP4 preview rendering during default export")
    p.add_argument("--preview-fps", type=int, default=15, help="Preview MP4 FPS")
    p.add_argument("--preview-max-frames", type=int, default=None, help="Optional cap on rendered preview frames")
    p.add_argument("--export-web", action="store_true", help="Also run on-demand transparent web video export")
    p.add_argument("--web-fps", type=int, default=30, help="Transparent web video FPS")
    p.add_argument("--web-max-frames", type=int, default=None, help="Optional cap on transparent web video frames")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        summary = run_e2e_pipeline(
            image_path=args.image,
            parts_spec=args.parts_spec,
            rig_spec=args.rig_spec,
            out_dir=args.out_dir,
            preset=args.preset,
            gain=args.gain,
            partial_rerun=args.partial_rerun,
            pause_after_split=args.pause_after_split,
            max_long_edge=args.max_long_edge,
            sample_count=args.sample_count,
            stress_gain=args.stress_gain,
            stress_samples=args.stress_samples,
            render_previews=not args.skip_previews,
            preview_fps=args.preview_fps,
            preview_max_frames=args.preview_max_frames,
            export_web=args.export_web,
            web_fps=args.web_fps,
            web_max_frames=args.web_max_frames,
            raise_on_error=False,
        )
    except Exception as exc:
        print(json.dumps({"status": "failed", "message": str(exc)}, indent=2), file=sys.stderr)
        return 1

    if summary.get("status") == "failed":
        print(json.dumps(summary, indent=2), file=sys.stderr)
        return 1

    print(json.dumps({
        "status": summary["status"],
        "mode": summary["mode"],
        "executedStages": summary["executedStages"],
        "skippedStages": summary["skippedStages"],
        "artifacts": summary["artifacts"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
