"""Sixteen-pose visual review aids. Candidates are warnings, not semantic proof."""
import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from src.pipeline.canvas_layer_cleanup import check_missing_original_regions
from src.pipeline.split_parts import _components
from src.render.renderer import over_bg
from src.pipeline.local_body_split import _bounds, _components as _mask_spans, _set_spans


def _material_masks(parts_dir, material, size, native_size):
    width, height = size
    native_width, native_height = native_size
    covered = np.zeros((height, width), dtype=bool)
    masks = []
    for part in sorted(material.get('parts', []), key=lambda part: part['drawOrder'], reverse=True):
        if not part.get('enabled', True):
            continue
        x, y, crop_width, crop_height = part['bbox']
        x0, y0 = x * width // native_width, y * height // native_height
        x1 = math.ceil((x + crop_width) * width / native_width)
        y1 = math.ceil((y + crop_height) * height / native_height)
        with Image.open(Path(parts_dir) / part['image']) as image:
            alpha = image.convert('RGBA').getchannel('A').resize((x1 - x0, y1 - y0), Image.Resampling.BOX)
        opaque = np.zeros_like(covered)
        opaque[y0:y1, x0:x1] = np.asarray(alpha) >= 120
        visible = opaque & ~covered
        covered |= opaque
        if visible.any():
            provenance = part.get('canvasLayer', {}).get('provenance', {})
            band = _dilated(visible, 2)
            masks.append((part['name'], part.get('kind') == 'soft',
                          isinstance(provenance.get('local_derivation'), dict), band, visible))
    return masks


def _frame_change(frames, times, size):
    def features(frame):
        rgb = Image.fromarray(over_bg(frame, (125, 125, 125))).resize(size, Image.Resampling.BOX)
        channels = np.asarray(rgb, dtype=np.int16)
        edges = np.zeros(channels.shape[:2], dtype=np.int16)
        edges[:, 1:] = np.max(np.abs(channels[:, 1:] - channels[:, :-1]), axis=2)
        edges[1:] = np.maximum(edges[1:], np.max(np.abs(channels[1:] - channels[:-1]), axis=2))
        return channels, edges

    base, base_edges = features(frames[0.])
    changes = []
    for time in times:
        channels, edges = features(frames[time])
        differences = np.max(np.abs(channels - base), axis=2)
        changes.append((.55 * differences + .45 * np.abs(edges - base_edges)).astype(np.float32))
    return np.stack(changes)


def _intersection_ratio(first, second):
    intersection = max(0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0, min(first[3], second[3]) - max(first[1], second[1]))
    area = min((box[2] - box[0]) * (box[3] - box[1]) for box in (first, second))
    return intersection / max(1, area)


def _material_seams(frames, times, parts_dir, material, *, limit=4):
    height, width = frames[0.].shape[:2]
    scale = min(1., 660 / max(height, width))
    size = max(1, round(width * scale)), max(1, round(height * scale))
    masks = _material_masks(parts_dir, material, size, (width, height))
    if len(masks) < 2 or not any(soft or local for _, soft, local, _, _ in masks):
        return []
    changes = _frame_change(frames, times, size)
    candidates = []
    tile = max(24, round(min(size) * .12))
    for index, (front, front_soft, front_local, front_band, front_mask) in enumerate(masks):
        for back, back_soft, back_local, back_band, back_mask in masks[index + 1:]:
            soft, local = front_soft or back_soft, front_local or back_local
            if not soft and not local:
                continue
            rows, columns = np.nonzero((front_band & back_band) & (front_mask | back_mask))
            if len(rows) < 5:
                continue
            bins = (rows // tile) * (1 + size[0] // tile) + columns // tile
            best = None
            for group in np.unique(bins):
                members = bins == group
                count = int(members.sum())
                if count < 5:
                    continue
                sample_rows, sample_columns = rows[members], columns[members]
                scores = changes[:, sample_rows, sample_columns].mean(axis=1) * min(1., count / 24.)
                instant = int(np.argmax(scores))
                score = float(scores[instant])
                if best is not None and score <= best['risk_score']:
                    continue
                pad = max(10, tile // 3)
                bbox = [max(0, math.floor((sample_columns.min() - pad) / scale)),
                        max(0, math.floor((sample_rows.min() - pad) / scale)),
                        min(width, math.ceil((sample_columns.max() + 1 + pad) / scale)),
                        min(height, math.ceil((sample_rows.max() + 1 + pad) / scale))]
                basis = [kind for kind, active in [('soft_layer', soft), ('local_split', local)] if active]
                best = {'pair': [back, front], 'time': times[instant], 'bbox': bbox, 'risk_score': score, 'basis': basis}
            if best is not None:
                candidates.append(best)
    selected = []
    for record in sorted(candidates, key=lambda item: item['risk_score'], reverse=True):
        if any(_intersection_ratio(record['bbox'], other['bbox']) >= .55 for other in selected):
            continue
        record['risk_score'] = round(record['risk_score'], 3)
        selected.append(record)
        if len(selected) == limit:
            break
    return selected


_EXCLUDED_STRESS_ROLES = frozenset({'prop', 'weapon', 'effect_group', 'pet', 'ground'})


def _dilated(mask, radius):
    if radius <= 0 or not mask.any():
        return mask
    try:
        import cv2
        k = 2 * int(radius) + 1
        return cv2.dilate(mask.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    except Exception:
        return np.asarray(Image.fromarray(mask.astype(np.uint8) * 255)
                          .filter(ImageFilter.MaxFilter(2 * radius + 1))) > 0


def _eroded_u8(arr, radius):
    if radius <= 0:
        return arr
    try:
        import cv2
        k = 2 * int(radius) + 1
        return cv2.erode(arr, np.ones((k, k), np.uint8))
    except Exception:
        return np.asarray(Image.fromarray(arr).filter(ImageFilter.MinFilter(2 * radius + 1)))


def _rest_contacts(parts_dir, material, part_names, reference):
    height, width = reference.shape
    masks = _material_masks(parts_dir, {'parts': [p for p in material['parts'] if p['name'] in part_names]},
                            (width, height), (width, height))
    reach = max(6, round(min(height, width) * .019))
    contacts = []
    sweep = np.zeros_like(reference)
    for index, (front, _, _, front_near, front_core) in enumerate(masks):
        for back, _, _, back_near, back_core in masks[index + 1:]:
            boundary = (front_near & back_core) | (back_near & front_core)
            if not boundary.any():
                continue
            rows, cols = np.nonzero(boundary)
            box = np.s_[max(0, rows.min()-reach):min(height, rows.max()+reach+1),
                        max(0, cols.min()-reach):min(width, cols.max()+reach+1)]
            band = _dilated(boundary[box], reach) & reference[box]
            sweep[box] |= band
            contacts.append(([back, front], box, band))
    return contacts, sweep


def _exposure_regions(missing, contacts):
    result = []
    for spans, area in _mask_spans(missing):
        if area < 4:
            continue
        left, top, right, bottom = _bounds(spans)
        crop = np.zeros((bottom-top, right-left), bool)
        _set_spans(crop, [(row-top, start-left, stop-left) for row, start, stop in spans], True)
        ranks = []
        for pair, box, band in contacts:
            x0, y0 = max(left, box[1].start), max(top, box[0].start)
            x1, y1 = min(right, box[1].stop), min(bottom, box[0].stop)
            if x0 >= x1 or y0 >= y1:
                continue
            overlap = band[y0-box[0].start:y1-box[0].start, x0-box[1].start:x1-box[1].start]
            score = int((overlap & crop[y0-top:y1-top, x0-left:x1-left]).sum())
            if score:
                ranks.append((score, pair))
        layers = list(dict.fromkeys(name for _, pair in sorted(ranks, reverse=True)[:3] for name in pair))
        result.append({'area_px':area, 'bbox':[left,top,right,bottom], 'layers':layers})
    return sorted(result,key=lambda item:item['area_px'],reverse=True)


def _morph_close(mask, radius):
    """Close open seam notches between moving parts without filling vacated outer silhouette motion."""
    if radius <= 0 or not mask.any():
        return mask
    try:
        import cv2
        k = 2 * int(radius) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0
    except Exception:
        k = 2 * int(radius) + 1
        im = Image.fromarray(mask.astype(np.uint8) * 255)
        closed = im.filter(ImageFilter.MaxFilter(k)).filter(ImageFilter.MinFilter(k))
        return np.asarray(closed) > 0


def _interior_transparency(alpha, threshold=40):
    """Return transparent components enclosed by a frame's opaque silhouette."""
    transparent = np.asarray(alpha) < threshold
    labels = _components(transparent)
    border = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
    return transparent & ~np.isin(labels, border), labels


def _hole_records(holes_mask, labels):
    ids, counts = np.unique(labels[holes_mask], return_counts=True)
    records = []
    for cid, area in zip(ids, counts):
        rows, cols = np.nonzero(labels == cid)
        r0, r1 = int(rows.min()), int(rows.max()) + 1
        c0, c1 = int(cols.min()), int(cols.max()) + 1
        crop = (labels[r0:r1, c0:c1] == cid)
        records.append((int(cid), int(area), r0, r1, c0, c1, float(rows.mean()), float(cols.mean()), crop))
    return records


def _followed_rest_holes(rest_records, current_holes, current_labels, *, max_shift):
    """Match only high-confidence rigid copies of pre-existing enclosed alpha holes.

    Requires similar area/aspect and >=88% overlap after centroid translation so any
    new, widened, or merged crack remains visible to the stress gate.
    """
    if not rest_records or not current_holes.any():
        return np.zeros_like(current_holes)
    current_records = _hole_records(current_holes, current_labels)
    matched = np.zeros_like(current_holes)
    used_rest = set()
    for cid, c_area, ct, cb, cl, cr, crow, ccol, ccrop in current_records:
        best = None
        for r_idx, (_rid, r_area, rt, rb, rl, rr, rrow, rcol, rcrop) in enumerate(rest_records):
            if r_idx in used_rest:
                continue
            area_ratio = c_area / max(1, r_area)
            if not 0.65 <= area_ratio <= 1.55:
                continue
            if not 0.55 <= (cb - ct) / max(1, rb - rt) <= 1.8:
                continue
            if not 0.55 <= (cr - cl) / max(1, rr - rl) <= 1.8:
                continue
            dy = int(round(crow - rrow))
            dx = int(round(ccol - rcol))
            if abs(dy) > max_shift or abs(dx) > max_shift:
                continue
            sr0, sr1 = rt + dy, rb + dy
            sc0, sc1 = rl + dx, rr + dx
            iy0, iy1 = max(ct, sr0), min(cb, sr1)
            ix0, ix1 = max(cl, sc0), min(cr, sc1)
            if iy1 <= iy0 or ix1 <= ix0:
                continue
            sub_c = ccrop[iy0 - ct : iy1 - ct, ix0 - cl : ix1 - cl]
            sub_r = rcrop[iy0 - sr0 : iy1 - sr0, ix0 - sc0 : ix1 - sc0]
            overlap = int((sub_c & sub_r).sum())
            if overlap / max(1, c_area) < 0.88 or overlap / max(1, r_area) < 0.88:
                continue
            score = overlap / max(c_area, r_area)
            if best is None or score > best[0]:
                best = (score, r_idx)
        if best is not None:
            used_rest.add(best[1])
            # Exempt only the predicted old-hole pixels that are still present in
            # the current component. Any expansion or newly attached crack pixels
            # must remain in the stress mask.
            predicted = np.zeros_like(ccrop)
            predicted[iy0 - ct : iy1 - ct, ix0 - cl : ix1 - cl] = sub_r
            matched[ct:cb, cl:cr] |= predicted & ccrop
    return matched


def _build_anchor_unwarper(renderer, modules):
    """Return ``(anchor_bone, unwarp_fn)`` registering pose alpha back to rest anchor frame.

    Eliminates false positives caused by rigid whole-body breathing/floating translation
    of pre-existing artwork holes and concave silhouettes, while preserving 100% of
    relative seam tears between moving child parts.
    """
    sk = getattr(renderer, 'sk', None)
    if (
        sk is None
        or not hasattr(sk, 'by_name')
        or not hasattr(renderer, 'pose')
        or not hasattr(renderer, 'origin_x')
        or not hasattr(renderer, 'origin_y')
        or not hasattr(renderer, '_canvas_sx')
        or not hasattr(renderer, '_canvas_sy')
    ):
        return None, lambda alpha, _t, _g: alpha

    anchor_mod = next(
        (m for m in modules if m.get('parent') is None and m.get('bone') in sk.by_name),
        None,
    )
    if anchor_mod is None:
        anchor_mod = next(
            (m for m in modules if m.get('role') in ('torso', 'body') and m.get('bone') in sk.by_name),
            None,
        )
    if anchor_mod is None:
        return None, lambda alpha, _t, _g: alpha

    anchor_bone = anchor_mod['bone']
    ox, oy = float(renderer.origin_x), float(renderer.origin_y)
    sx, sy = float(renderer._canvas_sx), float(renderer._canvas_sy)
    F = np.array([[sx, 0.0, ox * sx], [0.0, -sy, oy * sy], [0.0, 0.0, 1.0]], dtype=np.float64)
    F_inv = np.linalg.inv(F)

    def _bone_mat(t, g):
        renderer.pose(t, gain=g)
        b = sk.by_name[anchor_bone]
        return np.array([[b.a, b.b, b.worldX], [b.c, b.d, b.worldY], [0.0, 0.0, 1.0]], dtype=np.float64)

    M0 = _bone_mat(0.0, 1.0)

    def _unwarp(alpha_2d, t, g):
        Mt = _bone_mat(t, g)
        try:
            P_inv = F @ M0 @ np.linalg.inv(Mt) @ F_inv
        except np.linalg.LinAlgError:
            return alpha_2d
        if np.allclose(P_inv, np.eye(3), atol=1e-3):
            return alpha_2d
        h, w = alpha_2d.shape[:2]
        try:
            import cv2
            return cv2.warpAffine(
                alpha_2d,
                P_inv[:2, :],
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        except Exception:
            try:
                P_fwd = np.linalg.inv(P_inv)
            except np.linalg.LinAlgError:
                return alpha_2d
            coeffs = (
                float(P_fwd[0, 0]), float(P_fwd[0, 1]), float(P_fwd[0, 2]),
                float(P_fwd[1, 0]), float(P_fwd[1, 1]), float(P_fwd[1, 2]),
            )
            return np.asarray(
                Image.fromarray(alpha_2d).transform((w, h), Image.Transform.AFFINE, coeffs, resample=Image.Resampling.BILINEAR)
            )

    return anchor_bone, _unwarp


def _is_two_sided_exterior_notch(renderer, unwarp_fn, region, time, gain, border_px, enclosed_mask):
    """Verify that an exterior notch > threshold actually borders >=2 layers in the moved pose.

    Enclosed holes are always real and never filtered. For exterior open notches on a real
    SpineRenderer, if <=1 of the matched contact layers borders the region in the moved pose
    (e.g., two free-hanging ribbons swaying the same way leaving vacated exterior air),
    it is vacated outer air rather than a torn seam between two layers.
    """
    x0, y0, x1, y1 = region['bbox']
    if enclosed_mask[y0:y1, x0:x1].any():
        return True
    if getattr(renderer, 'sk', None) is None:
        return True
    layers = region.get('layers') or []
    if len(layers) < 2:
        return True
    h, w = enclosed_mask.shape[:2]
    pad = 12
    rx0, ry0 = max(0, x0 - pad), max(0, y0 - pad)
    rx1, ry1 = min(w, x1 + pad), min(h, y1 + pad)
    touching = 0
    for lname in layers:
        try:
            layer_frame = renderer.frame(time, gain=gain, slot_names={lname})
        except Exception:
            return True
        layer_alpha = unwarp_fn(layer_frame[..., 3], time, gain)
        if int((layer_alpha[ry0:ry1, rx0:rx1] >= 120).sum()) >= 16:
            touching += 1
            if touching >= 2:
                return True
    return False


def inspect_exposure_stress(renderer, parts_dir, material, modules, times, *, threshold=None):
    """Measure missing opaque interior and open seams near material contacts at both gains."""
    roles = {m['id']:m.get('role') for m in modules}
    body = {part['name'] for part in material['parts'] if part.get('enabled', True)
            and roles.get(part.get('moduleId')) not in _EXCLUDED_STRESS_ROLES}
    if not body:
        return {'status':'not_applicable', 'warnings':[], 'levels':{}}, None
    rest = renderer.frame(0., slot_names=body)
    rest_alpha = rest[...,3]
    long_edge = max(rest_alpha.shape)
    if threshold is None:
        scale = max(1.0, long_edge / 1800.0)
        threshold = int(round(120 * scale * scale))
    edge_guard = max(2, round(min(rest_alpha.shape)*.006))
    close_radius = max(12, round(min(rest_alpha.shape)*.02))
    interior = _eroded_u8(rest_alpha, edge_guard) >= 180
    max_hole_shift = max(12, round(min(rest_alpha.shape) * .18))
    rest_holes_mask, rest_holes_labels = _interior_transparency(rest_alpha)
    rest_hole_records = _hole_records(rest_holes_mask, rest_holes_labels)
    contacts, sweep = _rest_contacts(parts_dir, material, body, interior)
    anchor_bone, unwarp_fn = _build_anchor_unwarper(renderer, modules)
    levels, warnings, worst = {}, [], None
    extreme = None
    largest_motion = -1
    for gain in (1., 1.5):
        frames = []
        for time in times:
            if time == 0:
                continue
            pose = renderer.frame(time, gain=gain, slot_names=body)
            aligned_alpha = unwarp_fn(pose[...,3], time, gain)
            transparent = aligned_alpha < 40
            spans = _components(transparent)
            exterior = np.unique(np.concatenate((spans[0], spans[-1], spans[:,0], spans[:,-1])))
            enclosed = transparent & ~np.isin(spans, exterior)
            # Only flag exterior transparent pixels inside sweep when they form a genuine seam notch/gap
            # between moving body parts (filled by closing the moved silhouette), not normal outer drift.
            closed_pose = _morph_close(~transparent, close_radius)
            followed_holes = _followed_rest_holes(rest_hole_records, enclosed, spans, max_shift=max_hole_shift)
            missing = ((transparent & sweep & closed_pose) | (enclosed & interior)) & ~followed_holes
            raw_regions = _exposure_regions(missing, contacts)
            regions = []
            for region in raw_regions:
                if region['area_px'] > threshold and not _is_two_sided_exterior_notch(
                    renderer, unwarp_fn, region, time, gain, edge_guard, enclosed & interior
                ):
                    continue
                region['time'] = time
                regions.append(region)
            area = sum(region['area_px'] for region in regions)
            if gain == 1.5:
                motion = int(((pose[...,3] >= 120) != (rest_alpha >= 120)).sum())
                if motion > largest_motion:
                    largest_motion = motion
                    extreme = time, pose
            if area:
                frames.append({'time':time, 'area_px':area, 'regions':regions})
                if gain == 1.5 and (worst is None or regions[0]['area_px'] > worst[0]['area_px']):
                    worst = regions[0], pose
        key = f'{gain:.1f}x'
        peak = max((frame['area_px'] for frame in frames), default=0)
        largest = max((region['area_px'] for frame in frames for region in frame['regions']), default=0)
        levels[key] = {'peak_exposure_px':peak, 'largest_region_px':largest, 'frames':frames}
        if largest > threshold:
            warnings.append({'code':'max_slider_exposure','gain':gain,'peak_exposure_px':peak,'largest_region_px':largest})
    return {'status':'fail' if warnings else 'pass','threshold_px':threshold,'sample_times':list(times),
            'levels':levels,'warnings':warnings,'contact_pairs':[pair for pair, _, _ in contacts],
            'edge_guard_px':edge_guard,'body_layer_count':len(body),
            'anchor_registered':anchor_bone is not None,'anchor_bone':anchor_bone}, worst or (
                ({'time':extreme[0], 'area_px':0, 'bbox':[0,0,rest.shape[1],rest.shape[0]], 'layers':[]},extreme[1])
                if extreme else None)

def render_layer_sheet(parts_dir, material, out_dir):
    parts = sorted(material['parts'], key=lambda part: part['drawOrder'])
    tile, header, columns = 220, 27, min(5, max(1, len(parts)))
    sheet = Image.new('RGB', (columns*tile, math.ceil(len(parts)/columns)*(tile+header)), (38,38,40))
    tile_bg = Image.new('RGB',(tile,tile),(112,112,112))
    checker = ImageDraw.Draw(tile_bg)
    for y in range(0,tile,14):
        for x in range(0,tile,14):
            if (x//14+y//14)%2:
                checker.rectangle((x,y,x+13,y+13), fill=(150,150,150))
    draw = ImageDraw.Draw(sheet)
    for index, part in enumerate(parts):
        x, y = index%columns*tile, index//columns*(tile+header)
        sheet.paste(tile_bg,(x,y+header))
        with Image.open(Path(parts_dir)/part['image']) as source:
            rgba = source.convert('RGBA');rgba.thumbnail((tile-12,tile-12),Image.Resampling.LANCZOS)
            sheet.paste(rgba,(x+(tile-rgba.width)//2,y+header+(tile-rgba.height)//2),rgba)
        draw.text((x+7,y+7),f"{index+1:02d}  {part['name']}",fill='white')
    sheet.save(Path(out_dir)/'all-layers-sheet.jpg',quality=94)


def inspect_dynamic(renderer, frames, original, out_dir, *, parts_dir=None, material=None, stress_renderer=None):
    out=Path(out_dir);times=sorted(frames);base=frames[0.]
    report={'sample_times':times,'gap_detection_scope':'new enclosed transparent holes; open contacts are shown by angle and material seam crops',
            'new_gap':{'area_px':0,'time':0.,'bbox':None},'relative_angles':[]}
    # Work at quarter resolution for detection, retain native crops for review.
    size=(max(1,base.shape[1]//4),max(1,base.shape[0]//4))
    initial=np.asarray(Image.fromarray(base[...,3]).resize(size))
    interior=np.asarray(Image.fromarray(initial).filter(ImageFilter.MinFilter(5)))>180
    for t,frame in frames.items():
        alpha=np.asarray(Image.fromarray(frame[...,3]).resize(size))
        reference=np.zeros((*interior.shape,4),np.uint8);reference[...,3]=interior*255
        transparent=alpha<40;labels=_components(transparent)
        exterior=np.unique(np.concatenate((labels[0],labels[-1],labels[:,0],labels[:,-1])))
        enclosed=transparent & ~np.isin(labels,exterior)
        current=np.zeros_like(reference);current[...,3]=np.where(enclosed,0,255)
        regions=check_missing_original_regions(reference,current,min_area=4)
        if regions and regions[0]['area_px']*16>report['new_gap']['area_px']:
            report['new_gap']={'area_px':regions[0]['area_px']*16,'time':t,
                               'bbox':[v*4 for v in regions[0]['bbox']]}
    contract=renderer.build_report.get('moduleContract',{}) if hasattr(renderer,'build_report') else {}
    if not contract:
        import json
        contract=json.loads((renderer.rig_dir/'rig-build-report.json').read_text()).get('moduleContract',{})
    modules=contract.get('modules',[])
    arms=[m for m in modules if m.get('role') in ('arm','upper_arm','forearm','hand')]
    anchors=[m for m in modules if m.get('role') in ('head','torso','body')]
    rows=[];rigid_error=0.
    for arm in arms:
        for anchor in anchors:
            record={'pair':[anchor['id'],arm['id']],'max_relative_angle_degrees':0.,'time':0.}
            angles=[]
            for t in times:
                renderer.pose(t)
                bones=[renderer.sk.by_name[m['bone']] for m in (anchor,arm)]
                angles.append(math.degrees(math.atan2(bones[1].c,bones[1].a)-math.atan2(bones[0].c,bones[0].a)))
            values=[abs((angle-angles[0]+180)%360-180) for angle in angles]
            index=int(np.argmax(values));record.update(max_relative_angle_degrees=values[index],time=times[index])
            pivots=np.array([anchor['pivot'],arm['pivot']]);center=pivots.mean(axis=0)
            radius=max(80.,base.shape[0]*.10,float(np.linalg.norm(pivots[0]-pivots[1]))*.65)
            box=[max(0,int(center[0]-radius)),max(0,int(center[1]-radius)),
                 min(base.shape[1],int(center[0]+radius)),min(base.shape[0],int(center[1]+radius))]
            record['bbox']=box;report['relative_angles'].append(record)
            rows.append((anchor['id']+' / '+arm['id'],record))
    for t in times:
        renderer.pose(t)
        for m in modules:
            if m.get('shape')=='rigid':
                matrix=renderer.sk.by_name[m['bone']].matrix()[:,:2]
                rigid_error=max(rigid_error,float(np.abs(np.linalg.svd(matrix,compute_uv=False)-1).max()))
    report['rigid_max_singular_deviation']=rigid_error
    report['material_seams'] = _material_seams(frames, times, parts_dir, material) if parts_dir is not None and material else []
    rows.extend(('Seam: ' + ' / '.join(record['pair']), record) for record in report['material_seams'])
    if stress_renderer is not None and parts_dir is not None and material:
        report['exposure_stress'], worst = inspect_exposure_stress(stress_renderer, parts_dir, material, modules, times)
        render_layer_sheet(parts_dir, material, out)
        if worst:
            region, frame = worst
            Image.fromarray(over_bg(frame,(125,125,125))).save(out/'stress-150-frame.jpg',quality=94)
            left, top, right, bottom = region['bbox']
            border = max(22, round(max(right-left, bottom-top)*.35))
            rows.insert(0, ('150% exposure · '+str(region['area_px'])+' px',
                {**region,'bbox':[max(0,left-border),max(0,top-border),min(base.shape[1],right+border),min(base.shape[0],bottom+border)],'frame':frame}))
    if report['new_gap']['bbox']: rows.insert(0,('Largest new alpha gap',report['new_gap']))
    if not rows:rows=[('Full character',{'time':times[-1],'bbox':[0,0,base.shape[1],base.shape[0]]})]
    tile=420;sheet=Image.new('RGB',(tile*3, (tile+34)*len(rows)),(115,115,115));draw=ImageDraw.Draw(sheet)
    original_image=Image.fromarray(original).convert('RGBA');bg=Image.new('RGBA',original_image.size,(125,125,125,255));bg.alpha_composite(original_image)
    for i,(label,record) in enumerate(rows):
        box=record['bbox'];t=record['time']
        for j,im in enumerate([bg.convert('RGB'),Image.fromarray(over_bg(base,(125,125,125))),Image.fromarray(over_bg(record.get('frame', frames[t]),(125,125,125)))]):
            crop=im.crop(box);scale=tile/max(crop.size);crop=crop.resize((max(1,round(crop.width*scale)),max(1,round(crop.height*scale))),Image.Resampling.LANCZOS);x=j*tile;y=i*(tile+34)
            sheet.paste(crop,(x+(tile-crop.width)//2,y+34))
            draw.text((x+6,y+8),label+' '+['original','t=0',f't={t:g}s'][j],fill='white')
    sheet.save(out/'seam-worst-crops.jpg',quality=94)
    return report
