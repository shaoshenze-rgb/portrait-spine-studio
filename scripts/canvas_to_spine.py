#!/usr/bin/env python3
"""Existing Canvas parts -> typed Spine rig -> eight-second offline preview.

No Canvas requests, service process, ownership slicing or automatic signoff.
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.pipeline.canvas_layer_rig import compile_canvas_layer_bindings
from src.pipeline.build_rig import build_rig
from src.render.renderer import SpineRenderer, find_ffmpeg, write_preview_mp4, over_bg
from src.export.exporter import export_standalone_html



_C2S_RENDERER = None


def _c2s_ensure_pythonpath() -> None:
    root_str = str(ROOT)
    cur = os.environ.get('PYTHONPATH', '')
    parts = [x for x in cur.split(os.pathsep) if x]
    if root_str not in parts:
        os.environ['PYTHONPATH'] = os.pathsep.join([root_str] + parts)


def _c2s_worker_count(total_frames: int, long_edge: int) -> int:
    if total_frames <= 12 or long_edge <= 360 or multiprocessing.current_process().name != 'MainProcess':
        return 1
    cpus = os.cpu_count() or 4
    return max(1, min(12, cpus - 2, total_frames))


def _c2s_init_worker(rig_dir_str: str, render_scale: float) -> None:
    global _C2S_RENDERER
    _C2S_RENDERER = SpineRenderer(rig_dir_str, render_scale=render_scale)


def _c2s_render_time_worker(t: float):
    assert _C2S_RENDERER is not None
    return t, _C2S_RENDERER.frame(t)


def _c2s_render_dual_rgb_worker(task):
    idx, t = task
    assert _C2S_RENDERER is not None
    pm = _C2S_RENDERER.frame(t)
    dark = over_bg(pm, (17, 17, 17))
    light = over_bg(pm, (221, 221, 221))
    h, w = dark.shape[:2]
    ew, eh = w + (w % 2), h + (h % 2)
    if ew != w or eh != h:
        d_pad = np.full((eh, ew, 3), (17, 17, 17), dtype=np.uint8)
        l_pad = np.full((eh, ew, 3), (221, 221, 221), dtype=np.uint8)
        d_pad[:h, :w] = dark
        l_pad[:h, :w] = light
        dark, light = d_pad, l_pad
    return idx, np.ascontiguousarray(dark).tobytes(), np.ascontiguousarray(light).tobytes()


def _render_times_parallel(rig_dir: Path, times, *, render_scale: float, workers: int):
    if workers <= 1:
        ren = SpineRenderer(rig_dir, render_scale=render_scale)
        return {t: ren.frame(t) for t in times}
    _c2s_ensure_pythonpath()
    out = {}
    with ProcessPoolExecutor(max_workers=workers, initializer=_c2s_init_worker, initargs=(str(rig_dir), render_scale)) as pool:
        for t, frame in pool.map(_c2s_render_time_worker, times):
            out[t] = frame
    return out


def _preview_ffmpeg_cmd(ffmpeg_bin: str, out_path: Path, ew: int, eh: int, fps: int):
    return [
        ffmpeg_bin, '-hide_banner', '-loglevel', 'error', '-y',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{ew}x{eh}', '-r', str(fps), '-i', '-',
        '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-an',
        str(out_path),
    ]


def _stream_dual_preview_mp4s(rig_dir: Path, dark_path: Path, light_path: Path, *, total: int, fps: int, render_scale: float, ffmpeg_bin: str, workers: int) -> None:
    probe = SpineRenderer(rig_dir, render_scale=render_scale)
    ew = probe.width + (probe.width % 2)
    eh = probe.height + (probe.height % 2)
    p_dark = subprocess.Popen(_preview_ffmpeg_cmd(ffmpeg_bin, dark_path, ew, eh, fps), stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    p_light = subprocess.Popen(_preview_ffmpeg_cmd(ffmpeg_bin, light_path, ew, eh, fps), stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    tasks = [(i, i / float(fps)) for i in range(total)]
    try:
        assert p_dark.stdin is not None and p_light.stdin is not None
        if workers <= 1:
            for idx, t in tasks:
                pm = probe.frame(t)
                dark = over_bg(pm, (17, 17, 17))
                light = over_bg(pm, (221, 221, 221))
                if ew != probe.width or eh != probe.height:
                    d_pad = np.full((eh, ew, 3), (17, 17, 17), dtype=np.uint8)
                    l_pad = np.full((eh, ew, 3), (221, 221, 221), dtype=np.uint8)
                    d_pad[:probe.height, :probe.width] = dark
                    l_pad[:probe.height, :probe.width] = light
                    dark, light = d_pad, l_pad
                p_dark.stdin.write(np.ascontiguousarray(dark).tobytes())
                p_light.stdin.write(np.ascontiguousarray(light).tobytes())
                if idx % max(1, fps) == 0:
                    print(f'render {idx}/{total}', flush=True)
        else:
            _c2s_ensure_pythonpath()
            in_flight = max(workers * 2, 16)
            pending = deque()
            it = iter(tasks)
            with ProcessPoolExecutor(max_workers=workers, initializer=_c2s_init_worker, initargs=(str(rig_dir), render_scale)) as pool:
                for _ in range(min(in_flight, total)):
                    pending.append(pool.submit(_c2s_render_dual_rgb_worker, next(it)))
                while pending:
                    idx, d_bytes, l_bytes = pending.popleft().result()
                    nxt = next(it, None)
                    if nxt is not None:
                        pending.append(pool.submit(_c2s_render_dual_rgb_worker, nxt))
                    p_dark.stdin.write(d_bytes)
                    p_light.stdin.write(l_bytes)
                    if idx % max(1, fps) == 0:
                        print(f'render {idx}/{total}', flush=True)
        p_dark.stdin.close()
        p_light.stdin.close()
        err_d = p_dark.stderr.read().decode('utf-8', errors='replace') if p_dark.stderr else ''
        err_l = p_light.stderr.read().decode('utf-8', errors='replace') if p_light.stderr else ''
        if p_dark.wait() != 0:
            raise RuntimeError(f'preview-dark encode failed: {err_d}')
        if p_light.wait() != 0:
            raise RuntimeError(f'preview-light encode failed: {err_l}')
    finally:
        for proc in (p_dark, p_light):
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            if proc.stderr is not None:
                proc.stderr.close()


def _pm(image):
    rgba = np.asarray(image.convert('RGBA')).astype(np.float32)
    rgba[..., :3] *= rgba[..., 3:4] / 255.
    return np.rint(rgba).astype(np.uint8)


def _diff(a, b):
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return {'maxRGBA': int(diff.max()), 'meanAbsDiff': float(diff.mean()),
            'pixelsOver2': int((diff.max(axis=2) > 2).sum())}


class FrameSequence:
    """The existing encoder's sequence interface, backed by a temporary mmap."""
    def __init__(self, frames): self.frames = frames
    def __len__(self): return len(self.frames)
    def __getitem__(self, index): return self.frames[index]
    def __iter__(self): return iter(self.frames)


def run_preview(parts_dir, plan_path, out_dir, *, evidence_root=None,
                character='canvas_layers', preset='agile_full', fps=30, gain=1.0,
                cleanup=False, tweaks=None, recover_missing=True):
    parts_dir, plan_path = Path(parts_dir).resolve(), Path(plan_path).resolve()
    out = Path(out_dir).resolve()
    if not isinstance(fps, int) or fps <= 0: raise ValueError('fps must be a positive integer')
    if not character or Path(character).name != character or character in ('.', '..'):
        raise ValueError('character must be a single file name')
    ffmpeg = find_ffmpeg()
    if not ffmpeg: raise RuntimeError('ffmpeg is required for the eight-second preview')
    root = Path(evidence_root).resolve() if evidence_root else Path(os.path.commonpath([parts_dir, plan_path.parent]))
    # Existing compiler keeps all binding evidence under its declared root.
    out.relative_to(root)
    if out.exists(): raise ValueError('Use a new output directory; existing previews are preserved')
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    from src.pipeline.module_rig import canvas_motion_policy
    requests, effective = canvas_motion_policy(plan, tweaks)
    material_tweaks = {'layers': (tweaks or {}).get('layers', {})}
    from src.pipeline.canvas_layer_cleanup import inspect_canvas_parts_warnings, prepare_canvas_parts
    if cleanup or tweaks:
        parts_dir, plan, cleanup_report = prepare_canvas_parts(parts_dir, plan, out/'prepared',
            evidence_root=root, cleanup=cleanup, tweaks=material_tweaks, recover_missing=recover_missing)
    else:
        cleanup_report = inspect_canvas_parts_warnings(parts_dir, evidence_root=root)
    material = json.loads((parts_dir / 'parts.json').read_text(encoding='utf-8'))
    rig_spec = compile_canvas_layer_bindings(parts_dir, plan, out / 'rig-spec',
        evidence_root=root, character=character, strict_placement=False)
    baseline_rig = out / 'rig'
    if any(value != 1 for value in effective.values()):
        baseline_rig = out / 'stress-baseline'
        build_rig(parts_dir, rig_spec, baseline_rig, preset=preset, gain=gain)
    rig_spec['canvas_motion_ratios'] = effective
    rig_spec['canvas_motion_requests'] = requests
    (out/'rig-spec/rig-spec.json').write_text(json.dumps(rig_spec, ensure_ascii=False, indent=2))
    made = build_rig(parts_dir, rig_spec, out / 'rig', preset=preset, gain=gain)
    return render_preview(parts_dir, out/'rig', out, fps=fps, rig_spec=rig_spec,
                          plan_path=plan_path, cleanup_report=cleanup_report, stress_rig_dir=baseline_rig)


def render_preview(parts_dir, rig_dir, out_dir, *, fps=30, rig_spec=None, plan_path=None, cleanup_report=None, stress_rig_dir=None):
    """Same preview and diagnostics for Canvas and existing source-sliced rigs."""
    out=Path(out_dir).resolve();parts_dir=Path(parts_dir).resolve();rig_dir=Path(rig_dir).resolve()
    material=json.loads((parts_dir/'parts.json').read_text())
    report=json.loads((rig_dir/'rig-build-report.json').read_text())
    made={'report':report};character=report['character'];ffmpeg=find_ffmpeg()
    if not ffmpeg: raise RuntimeError('ffmpeg required')
    rig_spec=rig_spec or {};cleanup_report=cleanup_report or {}
    renderer = SpineRenderer(rig_dir)
    duration = float(made['report']['duration'])
    total = round(duration * fps)
    key_times = [0., 1.5, 3., 4.5, 6.]
    frames_at = {}
    long_edge = max(renderer.width, renderer.height)
    preview_scale = min(1.0, 1280.0 / float(long_edge)) if long_edge > 1600 else 1.0
    use_workers = _c2s_worker_count(total, long_edge)

    if preview_scale < 1.0 or use_workers > 1:
        # 1. Full native resolution (4K exact) for the 16 diagnostic/keyframe poses
        diag_times = sorted({round(i * 0.5, 6) for i in range(16)} | {round(t, 6) for t in key_times})
        diag_map = _render_times_parallel(rig_dir, diag_times, render_scale=1.0, workers=min(use_workers, len(diag_times)))
        dynamic_frames = {i * 0.5: diag_map[round(i * 0.5, 6)] for i in range(16)}
        for t in key_times:
            frames_at[t] = diag_map[round(t, 6)]
        frame_zero = diag_map[0.0]

        # 2. Stream preview-dark & preview-light MP4 directly into ffmpeg (zero 16GB temp memmap)
        _stream_dual_preview_mp4s(
            rig_dir,
            out / f'{character}-preview-dark.mp4',
            out / f'{character}-preview-light.mp4',
            total=total,
            fps=fps,
            render_scale=preview_scale,
            ffmpeg_bin=ffmpeg,
            workers=use_workers,
        )
    else:
        # Lightweight native path for small test canvases
        with tempfile.TemporaryDirectory(prefix='render-spool-', dir=out) as tmp:
            frames = np.memmap(Path(tmp) / 'frames.rgba', mode='w+', dtype='uint8',
                               shape=(total, renderer.height, renderer.width, 4))
            for i in range(total):
                frames[i] = renderer.frame(i / fps)
                if i % max(1, fps) == 0: print(f'render {i}/{total}', flush=True)
            dynamic_frames = {i*.5: frames[round(i*.5*fps)].copy() for i in range(16)}
            for t in key_times:
                frames_at[t] = (frames[round(t*fps)].copy() if round(t*fps) < total else renderer.frame(t))
            frame_zero = frames[0].copy()
            for theme, bg in [('dark', (17, 17, 17)), ('light', (221, 221, 221))]:
                write_preview_mp4(FrameSequence(frames), out / f'{character}-preview-{theme}.mp4',
                                  bg=bg, fps=fps, ffmpeg_bin=ffmpeg)
            del frames
    html = out / f'{character}-standalone-preview.html'
    export_standalone_html(out / 'rig', html, require_approved=False)
    snapshots = out / 'keyframes'; snapshots.mkdir()
    for t, frame in frames_at.items():
        Image.fromarray(over_bg(frame, (130, 130, 130))).save(snapshots / f't-{t:.1f}.png')
    source = Image.open(material.get('originalSource', material['source'])).convert('RGBA')
    panels = [('Original', over_bg(_pm(source), (130, 130, 130)))]
    panels += [(f't={t:.1f}s', over_bg(frame, (130, 130, 130))) for t, frame in frames_at.items()]
    tile_w, tile_h = 500, 690
    sheet = Image.new('RGB', (tile_w*3, tile_h*2), (130, 130, 130))
    draw = ImageDraw.Draw(sheet)
    for i, (label, array) in enumerate(panels):
        tile = Image.fromarray(array); tile.thumbnail((tile_w, tile_h-30))
        x, y = (i % 3)*tile_w, (i // 3)*tile_h
        sheet.paste(tile, (x+(tile_w-tile.width)//2, y+30))
        draw.text((x+12, y+8), label, fill=(255, 255, 255))
    sheet.save(out / 'contact-sheet-keyframes.jpg', quality=94)
    with Image.open(material['source']) as setup:
        reference = _pm(setup)
    paths = [out/'rig'/'rig-build-report.json', out/'rig'/f'{character}.spine.json', html,
             out/f'{character}-preview-dark.mp4', out/f'{character}-preview-light.mp4']
    paths += list((out/'rig'/'images').glob('*.png'))
    from src.gates.preview_diagnostics import inspect_dynamic
    stress_renderer = (renderer if Path(stress_rig_dir).resolve() == rig_dir else SpineRenderer(stress_rig_dir)) if stress_rig_dir else None
    dynamic = inspect_dynamic(renderer, dynamic_frames, np.asarray(source), out,
                              parts_dir=parts_dir, material=material, stress_renderer=stress_renderer)
    diagnostics = {
        'cleanup': cleanup_report, 'dynamic_checks': dynamic,
        'exposure_stress': dynamic.get('exposure_stress'),
        'character': character, 'material_mode': material.get('materialMode','source_sliced'), 'duration': duration, 'fps': fps,
        'frame_count': total, 'files_complete': all(p.is_file() and p.stat().st_size > 0 for p in paths),
        'loop_seam_max_diff': _diff(frame_zero, renderer.frame(duration))['maxRGBA'],
        'rest_composite_diff': _diff(frame_zero, reference),
        'setup_pose_composite_diff': _diff(renderer.frame(None), reference),
        'placement_warnings': rig_spec.get('placementWarnings', []),
        'advisory_notes': rig_spec.get('moduleContract', {}).get('advisoryNotes', {}),
        'source': str(material.get('originalSource', material['source'])), 'parts_dir': str(parts_dir), 'plan': str(plan_path),
        'new_canvas_calls': 0, 'visual_review': 'pending', 'export_approved': False,
        'files': {str(p.relative_to(out)): p.stat().st_size for p in paths},
    }
    (out/'preview-diagnostics.json').write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding='utf-8')
    return diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parts-dir', required=True, type=Path)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--character', default='canvas_layers')
    parser.add_argument('--preset', default='agile_full', choices=['agile_full', 'heavy_weapon_full', 'bust_half'])
    parser.add_argument('--out-dir', required=True, type=Path)
    parser.add_argument('--evidence-root', type=Path)
    parser.add_argument('--fps', default=30, type=int)
    parser.add_argument('--gain', default=1.0, type=float)
    parser.add_argument('--cleanup', action='store_true')
    parser.add_argument('--tweaks', type=Path, help='JSON layers/module settings')
    args = parser.parse_args()
    result = run_preview(args.parts_dir, args.plan, args.out_dir, evidence_root=args.evidence_root,
                         character=args.character, preset=args.preset, fps=args.fps, gain=args.gain, cleanup=args.cleanup,
                         tweaks=json.loads(args.tweaks.read_text()) if args.tweaks else None)
    print(json.dumps({key: result[key] for key in ('character', 'files_complete', 'frame_count', 'loop_seam_max_diff')}, ensure_ascii=False))


if __name__ == '__main__': main()
