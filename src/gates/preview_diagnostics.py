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
            band = np.asarray(Image.fromarray(visible.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(5))) > 0
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
    return np.asarray(Image.fromarray(mask.astype(np.uint8) * 255)
                      .filter(ImageFilter.MaxFilter(2 * radius + 1))) > 0


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
    interior = np.asarray(Image.fromarray(rest_alpha).filter(ImageFilter.MinFilter(edge_guard*2+1))) >= 180
    contacts, sweep = _rest_contacts(parts_dir, material, body, interior)
    levels, warnings, worst = {}, [], None
    extreme = None
    largest_motion = -1
    for gain in (1., 1.5):
        frames = []
        for time in times:
            if time == 0:
                continue
            pose = renderer.frame(time, gain=gain, slot_names=body)
            transparent = pose[...,3] < 40
            spans = _components(transparent)
            exterior = np.unique(np.concatenate((spans[0], spans[-1], spans[:,0], spans[:,-1])))
            enclosed = transparent & ~np.isin(spans, exterior)
            # Only flag exterior transparent pixels inside sweep when they form a genuine seam notch/gap
            # between moving body parts (filled by closing the moved silhouette), not normal outer drift.
            closed_pose = _morph_close(~transparent, close_radius)
            missing = (transparent & sweep & closed_pose) | (enclosed & interior)
            regions = _exposure_regions(missing, contacts)
            for region in regions:
                region['time'] = time
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
            'edge_guard_px':edge_guard,'body_layer_count':len(body)}, worst or (
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
