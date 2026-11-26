"""Materialize independent Canvas layers as standard parts, without segmentation.

This upstream adapter does not authorize Studio intake, visual approval or export.
The existing full-module review path must still be integrated for this mode.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image

from src.gates.review_binding import read_evidence_rgba, resolve_evidence_path


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r'[a-z][a-z0-9_]{0,47}', value):
        raise ValueError('Layer/module identifiers must be safe lowercase identifiers')
    return value


def materialize_canvas_layer_parts(manifest, out_dir, *, evidence_root):
    """Validate all inputs, then publish a NEW parts directory (manifest last).

    Each piece is an already aligned full-canvas PNG with an explicit canvas-space
    pivot. Overlap and amodal pixels are expected. No owner map is inferred here.
    """
    if type(manifest.get('version')) is not int or manifest['version'] != 1 or manifest.get('material_mode') != 'canvas_layers':
        raise ValueError('Expected explicit version=1 material_mode=canvas_layers')
    size = manifest.get('image_size')
    if (not isinstance(size, list) or len(size) != 2
            or any(type(v) is not int or v <= 0 for v in size)
            or size[0] * size[1] > 64_000_000):
        raise ValueError('Invalid layer canvas size')
    root = Path(evidence_root).resolve()
    source = resolve_evidence_path(manifest.get('source_image'), root=root)
    if _sha(source) != manifest.get('source_sha256'):
        raise ValueError('Original source SHA mismatch')
    with Image.open(source) as image:
        if image.size != tuple(size):
            raise ValueError('Original source size mismatch')
        original = np.asarray(image.convert('RGBA')).copy()
    pieces = manifest.get('pieces')
    if not isinstance(pieces, list) or not 2 <= len(pieces) <= 128:
        raise ValueError('Expected multiple declared layer pieces')
    if len(pieces) * size[0] * size[1] > 128_000_000:
        raise ValueError('Layer canvas batch exceeds materialization budget')
    out = Path(out_dir).absolute()
    try:
        out.resolve().relative_to(root)
    except ValueError as exc:
        raise ValueError('Output must stay within the task evidence root') from exc
    if out.exists() or out.is_symlink():
        raise ValueError('Use a new output directory; existing results are never overwritten')
    names, orders, module_shapes = set(), set(), {}
    records = []
    for piece in pieces:
        name = _identifier(piece.get('id'))
        mid = _identifier(piece.get('module_id'))
        parent = piece.get('parent')
        if parent is not None:
            _identifier(parent)
        order = piece.get('draw_order')
        if type(order) is not int or order in orders or name in names:
            raise ValueError('Duplicate/invalid piece or draw order')
        names.add(name); orders.add(order)
        kind = piece.get('kind')
        if kind not in ('rigid', 'soft'):
            raise ValueError('Expected rigid or soft material')
        pivot = piece.get('pivot')
        if (not isinstance(pivot, list) or len(pivot) != 2
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in pivot)
                or not (0 <= pivot[0] < size[0] and 0 <= pivot[1] < size[1])):
            raise ValueError('Explicit finite canvas-space pivot required')
        shape = (kind, parent, tuple(pivot))
        if mid in module_shapes and module_shapes[mid] != shape:
            raise ValueError('Pieces of one module must share motion kind, parent and pivot')
        module_shapes[mid] = shape
        provenance = piece.get('provenance')
        if not isinstance(provenance, dict):
            raise ValueError('Canvas provenance required')
        op = provenance.get('operation')
        if op == 'local-direct-split':
            derivation = provenance.get('local_derivation')
            if not isinstance(derivation, dict) or not isinstance(derivation.get('method'), str) or not derivation['method'].strip():
                raise ValueError('Local direct split requires local_derivation.method')
            parent_sha = derivation.get('parent_aligned_sha256') or derivation.get('source_sha256')
            if parent_sha != manifest.get('source_sha256'):
                raise ValueError('Local direct split parent SHA mismatch')
        elif op in ('split-layers', 'edit-image', 'remove-background'):
            for key in ('scene_id', 'node_id', 'task_id'):
                if not isinstance(provenance.get(key), str) or not provenance[key].strip():
                    raise ValueError('Canvas identity missing: ' + key)
        else:
            raise ValueError('Unsupported Canvas operation')
        for key in ('source_asset_sha256', 'result_asset_sha256'):
            if not isinstance(provenance.get(key), str) or not re.fullmatch(r'[0-9a-f]{64}', provenance[key]):
                raise ValueError('Canvas asset SHA missing: ' + key)
        if op == 'local-direct-split':
            if provenance['source_asset_sha256'] != manifest.get('source_sha256'):
                raise ValueError('Local direct split source SHA mismatch')
            if provenance['result_asset_sha256'] != piece.get('sha256'):
                raise ValueError('Local direct split result SHA mismatch')
        matrix = np.asarray(provenance.get('transform'), dtype=float)
        if matrix.shape != (2, 3) or not np.isfinite(matrix).all():
            raise ValueError('Invalid registered transform')
        linear = matrix[:, :2]; singular = np.linalg.svd(linear, compute_uv=False)
        if (np.linalg.det(linear) <= 0 or not .03 < singular[0] < 20
                or not np.allclose(singular, singular[0], rtol=1e-6)):
            raise ValueError('Layer registration must be a bounded similarity transform')
        path, digest, rgba = read_evidence_rgba(piece.get('image'), piece.get('sha256'), size, root=root)
        ys, xs = np.nonzero(rgba[..., 3] > 0)
        if not len(xs):
            raise ValueError('Empty Canvas layer')
        x, y = int(xs.min()), int(ys.min())
        w, h = int(xs.max()) + 1 - x, int(ys.max()) + 1 - y
        records.append((order, rgba[y:y+h, x:x+w].copy(), {
            'name': name, 'moduleId': mid, 'pieceId': name, 'drawOrder': order,
            'kind': kind, 'parent': parent, 'pivot': list(pivot), 'enabled': piece.get('enabled', True),
            'bbox': [x, y, w, h], 'image': 'parts/' + name + '.png',
            'label': piece.get('label'),
            'joints': {}, 'notes': 'Independent Canvas RGBA material; not source ownership',
            'canvasLayer': {'image': str(path), 'sha256': digest, 'provenance': provenance},
            'completionVisualReview': 'pending',
        }))
    for mid, (_, parent, _) in module_shapes.items():
        seen = {mid}
        while parent is not None:
            if parent not in module_shapes or parent in seen:
                raise ValueError('Missing/cyclic layer module parent')
            seen.add(parent); parent = module_shapes[parent][1]
    # No output is created before all evidence and structural inputs are valid.
    (out / 'parts').mkdir(parents=True, exist_ok=False)
    composite = Image.new('RGBA', tuple(size))
    metadata = []
    for _, crop, entry in sorted(records, key=lambda item: item[0]):
        cropped = Image.fromarray(crop)
        cropped.save(out / entry['image'])
        entry['imageSHA256'] = _sha(out / entry['image'])
        x, y, _, _ = entry['bbox']
        if entry["enabled"]: composite.alpha_composite(cropped, (x, y))
        metadata.append(entry)
    reference = out / 'setup_composite.png'
    composite.save(reference)
    # Original art comparison is a separate, UNSIGNED visual review requirement.
    delta = np.abs(np.asarray(composite).astype(np.int16) - original.astype(np.int16))
    comparison = {'status': 'pending', 'requiresVisualReview': True,
                  'originalSource': str(source), 'originalSourceSHA256': _sha(source),
                  'maxRGBA': int(delta.max()), 'pixelsOver2': int((delta.max(axis=2) > 2).sum())}
    (out / 'source_comparison.json').write_text(json.dumps(comparison, indent=2))
    result = {'version': '1.0', 'materialMode': 'canvas_layers', 'canvas': size,
              'source': str(reference), 'sourceSHA256': _sha(reference),
              'referenceKind': 'ordered_canvas_setup_not_original_artwork_approval',
              'originalSource': str(source), 'originalSourceSHA256': _sha(source),
              'parts': metadata, 'opaqueThreshold': 256, 'preserveSourceRGBA': False,
              'layerReview': 'pending', 'productionIntakeReady': False, 'exportAllowed': False}
    (out / 'parts.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result
