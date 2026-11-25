#!/usr/bin/env python3
"""Existing Canvas parts -> typed Spine rig -> eight-second offline preview.

No Canvas requests, service process, ownership slicing or automatic signoff.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
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
                character='canvas_layers', preset='agile_full', fps=30, gain=1.0):
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
    material = json.loads((parts_dir / 'parts.json').read_text(encoding='utf-8'))
    rig_spec = compile_canvas_layer_bindings(parts_dir, plan, out / 'rig-spec',
        evidence_root=root, character=character, strict_placement=False)
    made = build_rig(parts_dir, rig_spec, out / 'rig', preset=preset, gain=gain)
    renderer = SpineRenderer(out / 'rig')
    duration = float(made['report']['duration'])
    total = round(duration * fps)
    key_times = [0., 1.5, 3., 4.5, 6.]
    frames_at = {}
    # Keep full native resolution without retaining gigabytes of frames in RAM.
    with tempfile.TemporaryDirectory(prefix='render-spool-', dir=out) as tmp:
        frames = np.memmap(Path(tmp) / 'frames.rgba', mode='w+', dtype='uint8',
                           shape=(total, renderer.height, renderer.width, 4))
        for i in range(total):
            frames[i] = renderer.frame(i / fps)
            if i % max(1, fps) == 0: print(f'render {i}/{total}', flush=True)
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
    source = Image.open(material['originalSource']).convert('RGBA')
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
    diagnostics = {
        'character': character, 'material_mode': 'canvas_layers', 'duration': duration, 'fps': fps,
        'frame_count': total, 'files_complete': all(p.is_file() and p.stat().st_size > 0 for p in paths),
        'loop_seam_max_diff': _diff(frame_zero, renderer.frame(duration))['maxRGBA'],
        'rest_composite_diff': _diff(frame_zero, reference),
        'setup_pose_composite_diff': _diff(renderer.frame(None), reference),
        'placement_warnings': rig_spec['placementWarnings'],
        'advisory_notes': rig_spec['moduleContract']['advisoryNotes'],
        'source': str(material['originalSource']), 'parts_dir': str(parts_dir), 'plan': str(plan_path),
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
    args = parser.parse_args()
    result = run_preview(args.parts_dir, args.plan, args.out_dir, evidence_root=args.evidence_root,
                         character=args.character, preset=args.preset, fps=args.fps, gain=args.gain)
    print(json.dumps({key: result[key] for key in ('character', 'files_complete', 'frame_count', 'loop_seam_max_diff')}, ensure_ascii=False))


if __name__ == '__main__': main()
