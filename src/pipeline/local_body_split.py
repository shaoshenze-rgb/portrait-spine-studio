"""Split an aligned body using independently edge-fitted, depth-labelled masks.

This module resolves material ownership, not character semantics. The caller
supplies visible, edge-fitted masks and separates cloth at different depths.
"""
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter


@dataclass(frozen=True)
class BodyLayer:
    name: str
    mask: np.ndarray
    draw_order: int
    soft: bool = False


@dataclass(frozen=True)
class BodySplit:
    rgba: dict[str, np.ndarray]
    ownership: dict[str, np.ndarray]
    padding: dict[str, np.ndarray]
    stats: dict


def _rgba_array(rgba):
    source = np.asarray(rgba)
    if source.dtype != np.uint8 or source.ndim != 3 or source.shape[2] != 4 or not source.size:
        raise ValueError('Expected non-empty uint8 RGBA body')
    return source


def _lab(rgb):
    rgb = rgb.astype(np.float32) / 255
    linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
    xyz = linear @ np.array([[.4124564, .2126729, .0193339], [.3575761, .7151522, .1191920],
                            [.1804375, .0721750, .9503041]], dtype=np.float32)
    xyz /= np.array([.95047, 1., 1.08883], dtype=np.float32)
    lab = np.where(xyz > .008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack((116 * lab[..., 1] - 16, 500 * (lab[..., 0] - lab[..., 1]),
                     200 * (lab[..., 1] - lab[..., 2])), axis=-1)


def _filter_mask(mask, radius, operation):
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    return np.asarray(image.filter(operation(2 * radius + 1))) > 0


def _band_distances(lab, seeds, domain, steps):
    distances = np.where(seeds, 0., np.inf).astype(np.float32)
    vertical = 1 + np.minimum(np.linalg.norm(lab[1:] - lab[:-1], axis=-1), 80.) * .32
    horizontal = 1 + np.minimum(np.linalg.norm(lab[:, 1:] - lab[:, :-1], axis=-1), 80.) * .32
    for _ in range(steps):
        updated = distances.copy()
        np.minimum(updated[1:], distances[:-1] + vertical, out=updated[1:])
        np.minimum(updated[:-1], distances[1:] + vertical, out=updated[:-1])
        np.minimum(updated[:, 1:], distances[:, :-1] + horizontal, out=updated[:, 1:])
        np.minimum(updated[:, :-1], distances[:, 1:] + horizontal, out=updated[:, :-1])
        updated[~domain] = np.inf
        if np.array_equal(updated, distances):
            break
        distances = updated
    return distances


def _gradient_refine(rgb, seed, inner, outer, visible, band):
    colors = _lab(rgb)
    unknown = outer & ~inner
    box = unknown | inner
    back = visible & ~outer
    near_back = back & _filter_mask(unknown, 1, ImageFilter.MaxFilter)
    if not near_back.any():
        return seed
    foreground_dist = _band_distances(colors, inner, box, 4 * band + 4)
    background_dist = _band_distances(colors, near_back, unknown | near_back, 4 * band + 4)
    adjustment = np.where(seed, 1.5, -1.5)
    change = np.isfinite(foreground_dist) & np.isfinite(background_dist)
    result = seed.copy()
    result[change] = foreground_dist[change] <= background_dist[change] + adjustment[change]
    return inner | (result & outer)


def refine_mask_to_edges(rgba, seed_mask, *, band_px=12):
    """Refine visible ownership within a bounded band; preserve hard interior seeds."""
    source = _rgba_array(rgba)
    seed = np.asarray(seed_mask, dtype=bool)
    if seed.shape != source.shape[:2] or type(band_px) is not int or not 1 <= band_px <= 32:
        raise ValueError('Expected a same-size seed mask and band_px in 1..32')
    rows, columns = np.nonzero(seed)
    if not rows.size:
        return seed.copy()
    height, width = seed.shape
    margin = band_px + 2
    roi = np.s_[max(0, rows.min() - margin):min(height, rows.max() + margin + 1),
                max(0, columns.min() - margin):min(width, columns.max() + margin + 1)]
    cropped = seed[roi]
    visible = source[roi][..., 3] > 0
    inner = _filter_mask(cropped, band_px, ImageFilter.MinFilter) & visible
    outer = _filter_mask(cropped, band_px, ImageFilter.MaxFilter) & visible
    rgb = np.ascontiguousarray(source[roi][..., :3])
    cropped = cropped & visible
    result = cropped
    if inner.any() and (~outer).any() and (outer & ~inner).any():
        try:
            import cv2
        except ImportError:
            result = _gradient_refine(rgb, cropped, inner, outer, visible, band_px)
        else:
            cut = np.full(cropped.shape, cv2.GC_BGD, np.uint8)
            cut[outer] = cv2.GC_PR_BGD
            cut[cropped] = cv2.GC_PR_FGD
            cut[inner] = cv2.GC_FGD
            try:
                cv2.grabCut(rgb, cut, None, np.zeros((1, 65)), np.zeros((1, 65)), 3, cv2.GC_INIT_WITH_MASK)
            except cv2.error:
                result = _gradient_refine(rgb, cropped, inner, outer, visible, band_px)
            else:
                result = inner | (outer & ((cut == cv2.GC_FGD) | (cut == cv2.GC_PR_FGD)))
    output = np.zeros(seed.shape, dtype=bool)
    output[roi] = result
    return output


def _snap_edge(source, guides, radius):
    height, width = source.shape[:2]
    points = np.asarray(guides, dtype=float)
    if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all()
            or np.any(points[:, 1] != np.floor(points[:, 1])) or np.any(np.diff(points[:, 1]) <= 0)
            or np.any(points < 0) or np.any(points >= [width, height])):
        raise ValueError('Guides must be in-image (x, integer y) ordered top to bottom')
    rows = np.arange(int(points[0, 1]), int(points[-1, 1]) + 1)
    center = np.rint(np.interp(rows, points[:, 1], points[:, 0])).astype(int)
    offsets = np.arange(-radius, radius + 1)
    columns = center[:, None] + offsets
    valid = (columns >= 0) & (columns < width)
    before = source[rows[:, None], np.clip(columns - 2, 0, width - 1)]
    after = source[rows[:, None], np.clip(columns + 2, 0, width - 1)]
    delta = np.linalg.norm(_lab(before[..., :3]) - _lab(after[..., :3]), axis=-1)
    alpha_delta = np.abs(before[..., 3].astype(float) - after[..., 3]) / 4
    cost = np.where(valid, -np.minimum(delta + alpha_delta, 70) * .42 + np.abs(offsets) * .85, np.inf)
    pointers = np.zeros(cost.shape, dtype=np.int16)
    score = cost[0]
    for index in range(1, len(rows)):
        step = np.abs(offsets[:, None] - offsets[None, :])
        terms = score[None, :] + 1.35 * step + 2.5 * np.maximum(step - 2, 0)
        pointers[index] = terms.argmin(axis=1)
        score = cost[index] + terms[np.arange(len(offsets)), pointers[index]]
    selected = int(score.argmin())
    path = np.empty(len(rows), dtype=int)
    for index in range(len(rows) - 1, -1, -1):
        path[index] = columns[index, selected]
        selected = pointers[index, selected]
    return rows, path


def snap_contour_band(rgba, left_guides, right_guides, *, radius=10):
    """Fit left/right outlines to perceptual edges in a y-monotone band."""
    source = _rgba_array(rgba)
    if type(radius) is not int or not 1 <= radius <= 32:
        raise ValueError('radius must be in 1..32')
    rows, left = _snap_edge(source, left_guides, radius)
    right_rows, right = _snap_edge(source, right_guides, radius)
    if not np.array_equal(rows, right_rows) or np.any(left > right):
        raise ValueError('Contours must span identical rows and may not cross')
    mask = np.zeros(source.shape[:2], dtype=bool)
    columns = np.arange(source.shape[1])[None, :]
    mask[rows] = (columns >= left[:, None]) & (columns <= right[:, None])
    return mask & (source[..., 3] > 0)


def _components(mask):
    parents, runs, previous = [], [], []

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    padded = np.pad(mask, ((0, 0), (1, 1)))
    for row, values in enumerate(padded):
        transitions = np.flatnonzero(values[1:] != values[:-1]).tolist()
        current, offset = [], 0
        for start, stop in zip(transitions[::2], transitions[1::2]):
            index = len(parents)
            parents.append(index)
            while offset < len(previous) and previous[offset][1] < start:
                offset += 1
            adjacent = offset
            while adjacent < len(previous) and previous[adjacent][0] <= stop:
                parents[root(previous[adjacent][2])] = root(index)
                adjacent += 1
            current.append((start, stop, index))
            runs.append((row, start, stop))
        previous = current
    groups = {}
    for index, span in enumerate(runs):
        groups.setdefault(root(index), []).append(span)
    for spans in groups.values():
        yield spans, sum(stop - start for _, start, stop in spans)


def _set_spans(mask, spans, value):
    for row, start, stop in spans:
        mask[row, start:stop] = value


def _bounds(spans):
    return min(s[1] for s in spans), spans[0][0], max(s[2] for s in spans), spans[-1][0] + 1


def _discard_islands(mask, minimum):
    result = mask.copy()
    removed = 0
    for spans, area in _components(mask):
        if area < minimum:
            _set_spans(result, spans, False)
            removed += area
    return result, removed


def _smooth(mask, allowed, max_hole):
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    closed = np.asarray(image.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.MinFilter(3))) > 0
    result = mask | (closed & allowed)
    if max_hole:
        height, width = mask.shape
        for spans, area in _components(~result):
            if area > max_hole:
                continue
            left, top, right, bottom = _bounds(spans)
            if left and top and right < width and bottom < height:
                _set_spans(result, spans, True)
    return mask | (result & allowed)


def _distance_between(first, second):
    return float(np.hypot(max(first[0] - second[2], second[0] - first[2], 0),
                          max(first[1] - second[3], second[1] - first[3], 0)))


def separated_soft_islands(mask, *, min_area=500, min_gap=32):
    """Report large separated components in the tested visible support."""
    islands = [(_bounds(spans), area) for spans, area in _components(mask) if area >= min_area]
    if any(_distance_between(first[0], other[0]) >= min_gap
           for i, first in enumerate(islands) for other in islands[i + 1:]):
        return [{'bbox': list(box), 'pixels': area} for box, area in islands]
    return []


def _nearest(mask):
    """Euclidean nearest-seed field; NumPy jump-flood fallback without OpenCV."""
    try:
        import cv2
    except ImportError:
        row, col = np.indices(mask.shape, dtype=np.int32)
        best_row, best_col = np.where(mask, row, -1), np.where(mask, col, -1)
        best = np.where(mask, 0., np.inf)
        step = 1 << (max(mask.shape) - 1).bit_length()
        height, width = mask.shape
        while step:
            seeds_row, seeds_col = best_row.copy(), best_col.copy()
            for dy in (-step, 0, step):
                for dx in (-step, 0, step):
                    if (not dx and not dy) or abs(dy) >= height or abs(dx) >= width:
                        continue
                    dest = np.s_[max(0, dy):height+min(0, dy), max(0, dx):width+min(0, dx)]
                    origin = np.s_[max(0, -dy):height-max(0, dy), max(0, -dx):width-max(0, dx)]
                    sy, sx = seeds_row[origin], seeds_col[origin]
                    distance = ((row[dest]-sy)**2+(col[dest]-sx)**2)
                    use = (sy >= 0) & (distance < best[dest])
                    np.copyto(best[dest], distance, where=use)
                    np.copyto(best_row[dest], sy, where=use)
                    np.copyto(best_col[dest], sx, where=use)
            step //= 2
        return np.sqrt(best), best_row, best_col
    _, labels = cv2.distanceTransformWithLabels((~mask).astype(np.uint8), cv2.DIST_L2, 5,
                                                 labelType=cv2.DIST_LABEL_PIXEL)
    rows, cols = np.nonzero(mask)
    seed_y, seed_x = rows[labels-1], cols[labels-1]
    grid_y, grid_x = np.ogrid[:mask.shape[0], :mask.shape[1]]
    return np.hypot(grid_y-seed_y, grid_x-seed_x), seed_y, seed_x


def _occluded_spans(core, occluded, limit):
    output = np.zeros_like(core)
    padded = np.pad(core, ((0, 0), (1, 1)))
    for row, values in enumerate(padded):
        crossings = np.flatnonzero(values[1:] != values[:-1])
        for index in range(1, len(crossings)-1, 2):
            left, right = crossings[index:index+2]
            if right-left <= limit and occluded[row, left:right].all():
                output[row, left:right] = True
    return output


def _smooth_extension(colors, support, unknown):
    current = colors.astype(np.float32)
    mask = np.pad(support.astype(np.float32), 1)
    weights = mask[:-2, 1:-1] + mask[2:, 1:-1] + mask[1:-1, :-2] + mask[1:-1, 2:]
    active = unknown & (weights > 0)
    for _ in range(4):
        padded = np.pad(current, ((1, 1), (1, 1), (0, 0)), mode='edge')
        total = (padded[:-2, 1:-1] * mask[:-2, 1:-1, None] + padded[2:, 1:-1] * mask[2:, 1:-1, None]
                 + padded[1:-1, :-2] * mask[1:-1, :-2, None] + padded[1:-1, 2:] * mask[1:-1, 2:, None])
        current[active] = total[active] / weights[active, None]
    return np.rint(current).clip(0, 255).astype(np.uint8)


def _pad(source, core, occluded, radius):
    output = np.zeros_like(source)
    output[core] = source[core]
    padding = np.zeros_like(core)
    bridge = padding.copy()
    if not radius or not core.any() or not occluded.any():
        return output, padding, bridge
    ys, xs = np.nonzero(core)
    margin = radius + 3
    box = np.s_[max(0, ys.min()-margin):min(core.shape[0], ys.max()+margin+1),
                max(0, xs.min()-margin):min(core.shape[1], xs.max()+margin+1)]
    tile, allowed = core[box], occluded[box]
    opaque = tile & (source[box][..., 3] >= 250)
    if not opaque.any():
        return output, padding, bridge
    bridge_limit = max(2 * radius, round(min(core.shape) * .24))
    bridged = (_occluded_spans(opaque, allowed, bridge_limit)
               | _occluded_spans(opaque.T, allowed.T, bridge_limit).T)
    distance, _, _ = _nearest(opaque | bridged)
    support = allowed & (distance <= radius)
    support |= bridged
    for spans, _ in _components(tile | support):
        if not any(tile[row, left:right].any() for row, left, right in spans):
            _set_spans(support, spans, False)
    anchors = _filter_mask(opaque, 2, ImageFilter.MinFilter) & opaque
    if not anchors.any():
        anchors = opaque
    _, rows, cols = _nearest(anchors)
    colors = source[box][rows, cols, :3]
    colors = _smooth_extension(colors, anchors | support, support)
    solid = tile | support
    feather = 3 if radius >= 3 else 1
    inward, _, _ = _nearest(~np.pad(solid, 1))
    alpha = np.minimum(255, np.rint(inward[1:-1, 1:-1] * (255 / feather))).astype(np.uint8)
    output[box][support, :3] = colors[support]
    output[box][support, 3] = alpha[support]
    padding[box] = support
    bridge[box] = bridged & support
    return output, padding, bridge


def split_local_body(rgba, layers, *, padding_px=56, min_island_px=12,
                     max_hole_px=24, soft_island_px=120, soft_island_gap_px=24):
    """Return honest visible ownership and synthesized, strictly hidden underlap.

    Padding advances only through pixels owned by a higher, opaque visible
    layer. RGB propagates from immutable same-layer cores; it never samples the
    source under the actor doing the occluding. Unclaimed pixels are reported.
    """
    source = _rgba_array(rgba)
    definitions = sorted(layers, key=lambda layer: layer.draw_order)
    if not definitions or len({layer.name for layer in definitions}) != len(definitions):
        raise ValueError('Expected uniquely named body layers')
    if len({layer.draw_order for layer in definitions}) != len(definitions):
        raise ValueError('Body layers require unique draw orders')
    if any(type(value) is not int or value < 0 for value in
           (padding_px, min_island_px, max_hole_px, soft_island_px, soft_island_gap_px)):
        raise ValueError('Mask thresholds must be nonnegative integers')
    subject = source[..., 3] > 0
    proposals = []
    for layer in definitions:
        mask = np.asarray(layer.mask, dtype=bool)
        if mask.shape != subject.shape:
            raise ValueError(f'{layer.name}: mask dimensions do not match the body')
        proposals.append(mask & subject)
    count = np.sum(proposals, axis=0, dtype=np.int32)
    owner = np.full(subject.shape, -1, dtype=np.int32)
    cleaned, removed = [], []
    for index, proposal in enumerate(proposals):
        smooth = _smooth(proposal, subject & (count == 0), max_hole_px)
        keep, discarded = _discard_islands(smooth, min_island_px)
        cleaned.append(keep)
        removed.append(discarded)
        owner[keep] = index
    for index in range(len(definitions) - 1, -1, -1):
        core = owner == index
        keep, discarded = _discard_islands(core, min_island_px)
        removed[index] += discarded
        released = core & ~keep
        owner[released] = -1
        if released.any():
            for behind in range(index):
                owner[released & cleaned[behind]] = behind

    images, ownership, padding, stats = {}, {}, {}, {}
    warnings = []
    for index, layer in enumerate(definitions):
        core = owner == index
        occluded = (owner > index) & (source[..., 3] == 255)
        image, underlap, bridge = _pad(source, core, occluded, padding_px)
        images[layer.name], ownership[layer.name], padding[layer.name] = image, core, underlap
        islands = separated_soft_islands(image[..., 3] >= 120, min_area=soft_island_px, min_gap=soft_island_gap_px) if layer.soft else []
        if islands:
            warnings.append({'layer': layer.name, 'code': 'separated_soft_islands',
                             'action': 'Split by depth or into independent left/right pieces',
                             'islands': islands})
        stats[layer.name] = {'draw_order': layer.draw_order, 'owned_px': int(core.sum()),
                             'discarded_island_px': removed[index],
                             'proposed_overlap_px': int((proposals[index] & (count > 1)).sum()),
                             'padding_px': int(underlap.sum()),
                             'occluded_bridge_px': int(bridge.sum()),
                             'padding_under': {front.name: int((underlap & (owner == front_id)).sum())
                                               for front_id, front in enumerate(definitions)
                                               if front_id > index and (underlap & (owner == front_id)).any()}}
    return BodySplit(images, ownership, padding,
                     {'layers': stats, 'unclaimed_px': int((subject & (owner < 0)).sum()),
                      'proposed_overlap_px': int((count > 1).sum()), 'warnings': warnings})
