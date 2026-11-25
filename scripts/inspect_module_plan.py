#!/usr/bin/env python3
"""Render source-vs-module review images. This never grants semantic acceptance."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.studio.module_contract import contract_fingerprint, materialize_module_contract


def inspect_plan(image_path: Path, analysis_path: Path, out_dir: Path) -> dict:
    image_path, analysis_path, out_dir = image_path.resolve(), analysis_path.resolve(), out_dir.resolve()
    plan = json.loads(analysis_path.read_text(encoding='utf-8'))
    if plan.get('version') != 2:
        raise ValueError('Only complete-module V2 plans can be inspected here')
    hints = materialize_module_contract(plan, image_path=image_path, mask_dir=out_dir / 'masks')
    contract = hints['module_contract']
    with Image.open(image_path) as image:
        source = np.asarray(image.convert('RGBA')).copy()
    height, width = source.shape[:2]
    result = {'review_status': 'unreviewed', 'scope': 'declared visible ownership; not proof of semantic completeness',
              'source_input': str(image_path), 'source_sha256': hashlib.sha256(image_path.read_bytes()).hexdigest(),
              'analysis_file': str(analysis_path), 'analysis_sha256': hashlib.sha256(analysis_path.read_bytes()).hexdigest(),
              'contract_sha256': contract_fingerprint(contract), 'image_size': [width, height],
              'module_count': len(contract['modules']), 'modules': []}
    for module in contract['modules']:
        owned = np.zeros((height, width), dtype=bool)
        for piece in module['pieces']:
            with Image.open(hints['module_masks'][piece['id']]) as mask:
                owned |= np.asarray(mask.convert('L')) > 127
        yy, xx = np.nonzero(owned)
        box = [max(0, int(xx.min()) - 12), max(0, int(yy.min()) - 12),
               min(width, int(xx.max()) + 13), min(height, int(yy.max()) + 13)]
        isolated = source.copy(); isolated[~owned] = 0
        isolated_crop = Image.fromarray(isolated).crop(box)
        source_crop = Image.fromarray(source).crop(box)
        isolated_path = out_dir / (module['id'] + '-visible.png')
        isolated_crop.save(isolated_path)
        # Same crop/scale on both sides preserves comparison coordinates. Native PNG
        # remains available above for close inspection, including disconnected pieces.
        left = Image.new('RGBA', source_crop.size, (145, 145, 145, 255)); left.alpha_composite(source_crop)
        right = Image.new('RGBA', isolated_crop.size, (145, 145, 145, 255)); right.alpha_composite(isolated_crop)
        left.thumbnail((640, 720), Image.Resampling.LANCZOS)
        right.thumbnail((640, 720), Image.Resampling.LANCZOS)
        sheet = Image.new('RGB', (left.width * 2, left.height + 28), (145, 145, 145))
        sheet.paste(left.convert('RGB'), (0, 28)); sheet.paste(right.convert('RGB'), (left.width, 28))
        draw = ImageDraw.Draw(sheet)
        draw.text((6, 8), 'SOURCE', fill=(0, 0, 0))
        draw.text((left.width + 6, 8), 'MODULE: ' + module['id'] + ' / ' + module['motion_type'], fill=(0, 0, 0))
        comparison = out_dir / (module['id'] + '-comparison.jpg')
        sheet.save(comparison, quality=95)
        result['modules'].append({'id': module['id'], 'role': module['role'], 'motion_type': module['motion_type'],
            'pieces': [p['id'] for p in module['pieces']], 'visible_pixels': int(owned.sum()), 'source_crop_box': box,
            'isolated_image': str(isolated_path), 'comparison_image': str(comparison), 'review_status': 'unreviewed'})
    (out_dir / 'module-review.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='完整模块 V2 的原图/孤立图对照，不自动判定语义通过。')
    parser.add_argument('--image', required=True, type=Path)
    parser.add_argument('--analysis', required=True, type=Path)
    parser.add_argument('--out-dir', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = inspect_plan(args.image, args.analysis, args.out_dir)
    except Exception as error:
        print(json.dumps({'ok': False, 'errorCode': getattr(error, 'code', type(error).__name__),
                          'message': str(error)}, ensure_ascii=False))
        return 1
    print(json.dumps({'ok': True, 'review_status': 'unreviewed', 'module_count': result['module_count'],
                      'manifest': str(args.out_dir.resolve() / 'module-review.json')}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
