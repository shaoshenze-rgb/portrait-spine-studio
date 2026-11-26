#!/usr/bin/env python3
"""Fit semantic seed masks/contours and split a local, aligned RGBA body."""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.pipeline.local_body_split import BodyLayer, refine_mask_to_edges, snap_contour_band, split_local_body


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--body', type=Path, required=True, help='Aligned transparent body PNG')
    parser.add_argument('--spec', type=Path, required=True, help='JSON semantic layers and draw orders')
    parser.add_argument('--out-dir', type=Path, required=True, help='New output directory')
    args = parser.parse_args(argv)
    spec = json.loads(args.spec.read_text(encoding='utf-8'))
    if spec.get('version') != 1 or not isinstance(spec.get('layers'), list) or not spec['layers']:
        parser.error('Expected version=1 and a nonempty layers array')
    if args.out_dir.exists():
        parser.error('Use a new output directory')
    with Image.open(args.body) as image:
        body = np.array(image.convert('RGBA'))
    definitions, methods = [], {}
    for layer in spec['layers']:
        name = layer.get('id', '')
        order = layer.get('draw_order')
        if not re.fullmatch(r'[a-z][a-z0-9_]{0,47}', name) or type(order) is not int:
            parser.error('Layer id and integer draw_order are required')
        if layer.get('kind') not in ('rigid', 'soft'):
            parser.error(f'{name}: expected kind rigid or soft')
        mask_path, contour = layer.get('seed_mask'), layer.get('contour')
        if (mask_path is None) == (contour is None):
            parser.error(f'{name}: provide exactly one seed_mask or contour')
        if mask_path is not None:
            with Image.open(args.spec.parent / mask_path) as image:
                seed = np.asarray(image.convert('L')) > 127
            mask = refine_mask_to_edges(body, seed, band_px=layer.get('band_px', 12))
            methods[name] = 'edge_refined_seed_mask'
        else:
            mask = snap_contour_band(body, contour['left_guides'], contour['right_guides'],
                                     radius=contour.get('radius', 10))
            methods[name] = 'perceptual_contour_band'
        definitions.append(BodyLayer(name, mask, order, layer['kind'] == 'soft'))
    split = split_local_body(body, definitions, padding_px=spec.get('padding_px', 56),
                             min_island_px=spec.get('min_island_px', 12))
    if any(not owned.any() for owned in split.ownership.values()):
        parser.error('At least one layer has no visible pixels; review its seed and depth')
    args.out_dir.mkdir(parents=True, exist_ok=False)
    for layer in definitions:
        name = layer.name
        image = args.out_dir / f'{name}.png'
        ownership = args.out_dir / f'{name}-ownership.png'
        Image.fromarray(split.rgba[name]).save(image)
        Image.fromarray(split.ownership[name].astype(np.uint8) * 255).save(ownership)
        split.stats['layers'][name].update(image=image.name, sha256=_digest(image), kind='soft' if layer.soft else 'rigid',
            method=methods[name], owner_mask=ownership.name, owner_mask_sha256=_digest(ownership))
    split.stats.update(version=1, parent_aligned_sha256=_digest(args.body), image_size=list(body.shape[:2][::-1]))
    (args.out_dir / 'split-stats.json').write_text(json.dumps(split.stats, ensure_ascii=False, indent=2), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
