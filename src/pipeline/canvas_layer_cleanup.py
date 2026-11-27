"""Local, source-constrained cleanup of already aligned Canvas layers."""
from __future__ import annotations
import copy
import hashlib
import io
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from src.gates.review_binding import read_evidence_rgba, resolve_evidence_path
from .split_parts import _components
from .local_body_split import separated_soft_islands


def dilate(mask, radius=1):
    try:
        import cv2
        k = int(radius) * 2 + 1
        return cv2.dilate(mask.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    except Exception:
        return np.asarray(Image.fromarray(mask.astype('uint8')*255).filter(ImageFilter.MaxFilter(radius*2+1))) > 0


def defringe_rgba(rgba, *, max_radius=None):
    """Decontaminate white or desaturated mid-gray halo on semi-transparent edges.

    Scales search radius with resolution (3px at <=1600px up to 10px at 4K),
    preserves opaque pixels (alpha >= 245), preserves alpha smoothness, and
    leaves saturated glowing VFX edges untouched.
    """
    layer = np.asarray(rgba, dtype=np.uint8).copy()
    if layer.ndim != 3 or layer.shape[2] != 4 or not layer.size:
        return layer, 0
    h, w = layer.shape[:2]
    alpha = layer[..., 3]
    solid = alpha >= 245
    semi = (alpha > 3) & (alpha < 245)
    if not solid.any() or not semi.any():
        return layer, 0

    steps = (
        max(3, min(10, int(round(3.0 * max(h, w) / 1600.0))))
        if max_radius is None
        else max(1, int(max_radius))
    )

    # Restrict work to the tight ROI around semi-transparent pixels for 4K speed
    ys, xs = np.nonzero(semi)
    margin = steps + 2
    y0, y1 = max(0, int(ys.min()) - margin), min(h, int(ys.max()) + margin + 1)
    x0, x1 = max(0, int(xs.min()) - margin), min(w, int(xs.max()) + margin + 1)

    sub_rgb = layer[y0:y1, x0:x1, :3].astype(np.float32)
    sub_solid = solid[y0:y1, x0:x1]
    sub_semi = semi[y0:y1, x0:x1]
    if not sub_solid.any():
        return layer, 0

    colors = None
    reachable = None
    try:
        import cv2

        dist, labels = cv2.distanceTransformWithLabels(
            (~sub_solid).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
        )
        s_rows, s_cols = np.nonzero(sub_solid)
        sr = s_rows[labels - 1]
        sc = s_cols[labels - 1]
        colors = sub_rgb[sr, sc]
        reachable = dist <= float(steps) + 0.5
    except Exception:
        known = sub_solid.copy()
        colors = sub_rgb.copy()
        sh, sw = sub_solid.shape
        for _ in range(steps):
            sums = np.zeros_like(colors)
            counts = np.zeros((sh, sw), np.float32)
            pad = np.pad(known, 1)
            cp = np.pad(colors, ((1, 1), (1, 1), (0, 0)))
            for dy, dx in ((0, 1), (2, 1), (1, 0), (1, 2)):
                k = pad[dy : dy + sh, dx : dx + sw]
                sums += cp[dy : dy + sh, dx : dx + sw] * k[..., None]
                counts += k
            take_step = (~known) & (counts > 0)
            if not take_step.any():
                break
            colors[take_step] = sums[take_step] / counts[take_step, None]
            known |= take_step
        reachable = known

    edge = sub_semi & reachable
    if not edge.any():
        return layer, 0

    px_rgb = sub_rgb
    px_max = px_rgb.max(axis=2)
    px_min = px_rgb.min(axis=2)
    px_chroma = px_max - px_min
    px_sat = px_chroma / np.maximum(px_max, 1.0)
    px_lum = 0.299 * px_rgb[..., 0] + 0.587 * px_rgb[..., 1] + 0.114 * px_rgb[..., 2]

    in_max = colors.max(axis=2)
    in_min = colors.min(axis=2)
    in_sat = (in_max - in_min) / np.maximum(in_max, 1.0)
    in_lum = 0.299 * colors[..., 0] + 0.587 * colors[..., 1] + 0.114 * colors[..., 2]

    # 1) White/bright backdrop spill (preserving saturated glowing VFX edges)
    white_halo = ((px_rgb - colors).mean(axis=2) > 12.0) & (px_sat <= in_sat + 0.08)
    # 2) Desaturated neutral mid-gray backdrop spill on colored/dark armor & cloth
    gray_halo = (
        (in_sat >= 0.18)
        & (px_sat <= in_sat - 0.14)
        & ((px_lum - in_lum) > 12.0)
        & (px_chroma <= 28.0)
    )
    take = edge & (white_halo | gray_halo)
    count = int(take.sum())
    if count:
        layer[y0:y1, x0:x1, :3][take] = np.rint(colors[take]).clip(0, 255).astype(np.uint8)
    return layer, count


def clean_layer(layer_rgba, original_rgba, *, effect_mode='keep'):
    if effect_mode not in ('keep', 'omit', 'original_extract'):
        raise ValueError('effect_mode must be keep/omit/original_extract')
    layer = np.asarray(layer_rgba, dtype=np.uint8).copy()
    if layer.shape != original_rgba.shape: raise ValueError('Layer and source sizes must match')
    fg = dilate(original_rgba[...,3] > 8)
    occupied = layer[...,3] > 0
    outside = occupied & ~fg
    stats = {'removed_outside_pixels':int(outside.sum()),
             'outside_ratio_before':float(outside.sum()/max(1,occupied.sum())), 'outside_ratio_after':0.,
             'effect_mode':effect_mode}
    layer[~fg] = 0
    layer[layer[...,3] <= 3] = 0
    layer, defringed_px = defringe_rgba(layer)
    stats['defringed_pixels'] = int(defringed_px)
    if effect_mode == 'original_extract':
        rgb=original_rgba[...,:3].astype(float)
        warm=(rgb[...,0]>150)&(rgb[...,0]>rgb[...,1]*1.15)&(rgb[...,1]>rgb[...,2]*1.1)
        take=(layer[...,3]>0)&warm
        layer[:]=0;layer[take]=original_rgba[take]
    # omit retains geometry/material for the rig but disables its drawing slot.
    return layer, stats


def check_missing_original_regions(original_rgba, composite_rgba, min_area=400):
    missing=(original_rgba[...,3]>=180)&(composite_rgba[...,3]<40)
    labels=_components(np.pad(missing,1))[1:-1,1:-1];counts=np.bincount(labels.ravel())
    regions=[]
    for label in np.flatnonzero(counts >= min_area):
        if label==0:continue
        ys,xs=np.nonzero(labels==label)
        regions.append({'area_px':int(counts[label]),'bbox':[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)]})
    return sorted(regions,key=lambda r:r['area_px'],reverse=True)



def _downsample_rgba(rgba, max_edge=1024):
    h, w = rgba.shape[:2]
    if max(h, w) <= max_edge:
        return rgba
    scale = max_edge / float(max(h, w))
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    try:
        import cv2
        return cv2.resize(rgba, (nw, nh), interpolation=cv2.INTER_AREA)
    except Exception:
        return np.asarray(Image.fromarray(rgba).resize((nw, nh), Image.Resampling.BILINEAR))


def _max_connected_area(mask, open_px=0):
    u8 = mask.astype(np.uint8)
    if not u8.any():
        return 0
    try:
        import cv2
        if open_px > 0:
            k = np.ones((open_px * 2 + 1, open_px * 2 + 1), np.uint8)
            u8 = cv2.morphologyEx(u8, cv2.MORPH_OPEN, k)
            if not u8.any():
                return 0
        n, _, stats, _ = cv2.connectedComponentsWithStats(u8, connectivity=8)
        return int(stats[1:, cv2.CC_STAT_AREA].max()) if n > 1 else 0
    except Exception:
        labels = _components(np.pad(u8 > 0, 1))[1:-1, 1:-1]
        counts = np.bincount(labels.ravel())
        return int(counts[1:].max()) if len(counts) > 1 else 0


def check_original_fidelity(original_rgba, composite_rgba, layers=None):
    """Measure visible RGB fidelity of assembled layers against original portrait."""
    orig = _downsample_rgba(original_rgba)
    comp = _downsample_rgba(composite_rgba)
    both = (orig[..., 3] >= 180) & (comp[..., 3] >= 180)
    compared = int(both.sum())
    if compared == 0:
        return {'rgb_mae': 0.0, 'high_diff_ratio': 0.0, 'min_layer_rgb_mae': 0.0,
                'min_layer_high_diff_ratio': 0.0, 'max_drift_patch_ratio': 0.0,
                'compared_pixels': 0, 'drifted': False}
    orig_rgb = orig[..., :3].astype(np.int16)
    comp_diff = np.abs(comp[..., :3].astype(np.int16) - orig_rgb).mean(axis=2)
    comp_vals = comp_diff[both]
    rgb_mae = float(comp_vals.mean())
    high_ratio = float((comp_vals > 40.0).mean())

    min_mae, min_high = rgb_mae, high_ratio
    best_map = np.where(both, comp_diff, np.inf)
    if layers:
        best = np.full(both.shape, np.inf, dtype=np.float32)
        for layer in layers:
            if not layer.get('enabled', True) or layer.get('effect_mode') == 'omit':
                continue
            la = _downsample_rgba(layer['rgba'])
            solid = both & (la[..., 3] >= 160)
            if solid.any():
                d = np.abs(la[..., :3].astype(np.int16) - orig_rgb).mean(axis=2)
                best = np.minimum(best, np.where(solid, d, np.inf))
        cov = both & np.isfinite(best)
        if cov.any():
            bvals = best[cov]
            min_mae = float(bvals.mean())
            min_high = float((bvals > 40.0).mean())
            best_map = best

    drift_mask = both & np.isfinite(best_map) & (best_map > 45.0)
    max_patch = float(_max_connected_area(drift_mask, open_px=1) / max(1, compared))
    drifted = compared >= 200 and (min_mae > 18.0 or min_high > 0.10 or max_patch > 0.03)
    return {
        'rgb_mae': round(rgb_mae, 2),
        'high_diff_ratio': round(high_ratio, 4),
        'min_layer_rgb_mae': round(min_mae, 2),
        'min_layer_high_diff_ratio': round(min_high, 4),
        'max_drift_patch_ratio': round(max_patch, 4),
        'compared_pixels': compared,
        'drifted': bool(drifted),
    }


def _check_duplicate_silhouettes(layers, original_rgba):
    orig = _downsample_rgba(original_rgba)
    orig_fg = orig[..., 3] >= 180
    fg_area = int(orig_fg.sum())
    min_area = max(250, int(0.005 * max(1, fg_area)))
    entries = []
    for layer in sorted(layers, key=lambda item: item['draw_order']):
        if not layer.get('enabled', True) or layer.get('effect_mode') == 'omit':
            continue
        if layer.get('role') == 'effect_group':
            continue
        la = _downsample_rgba(layer['rgba'])
        mask = la[..., 3] >= 160
        if fg_area > 0:
            mask = mask & orig_fg
        if int(mask.sum()) >= min_area:
            entries.append((layer['id'], mask, la[..., :3].astype(np.int16)))
    warnings = []
    for i in range(len(entries)):
        id_a, m_a, rgb_a = entries[i]
        for j in range(i + 1, len(entries)):
            id_b, m_b, rgb_b = entries[j]
            inter_mask = m_a & m_b
            inter = int(inter_mask.sum())
            if not inter:
                continue
            iou = float(inter / max(1, int((m_a | m_b).sum())))
            if iou >= 0.72:
                odiff = np.abs(rgb_a - rgb_b).mean(axis=2)[inter_mask]
                overlap_mae = float(odiff.mean())
                same_ratio = float((odiff <= 20.0).mean())
                if iou >= 0.88 or overlap_mae <= 22.0 or same_ratio >= 0.65:
                    warnings.append({
                        'code': 'duplicate_layer_silhouette',
                        'layers': [id_a, id_b],
                        'iou': round(iou, 3),
                        'overlap_rgb_mae': round(overlap_mae, 1),
                        'suggestion': '两层轮廓高度重叠且像素雷同（重影/重复整图），禁止采用，须剔除重影或回 Canvas 重拆',
                    })
    return warnings


def inspect_local_layer_warnings(layers, original_rgba):
    """Check actual textures, not an upstream splitter's self-reported warnings."""
    warnings = []
    siblings = {}
    for layer in sorted(layers, key=lambda item: item['draw_order']):
        if not layer.get('enabled', True) or layer.get('effect_mode') == 'omit':
            continue
        opaque = layer['rgba'][..., 3] >= 120
        if layer.get('kind') == 'soft':
            islands = separated_soft_islands(opaque)
            if islands:
                warnings.append({'code': 'separated_soft_islands', 'layer': layer['id'],
                                 'islands': islands, 'suggestion': 'split by depth or into independent left/right pieces'})
        provenance = layer.get('provenance') or {}
        derivation = provenance.get('local_derivation')
        if not isinstance(derivation, dict):
            continue
        parent = derivation.get('parent_aligned_sha256') or derivation.get('source_sha256') or provenance.get('source_asset_sha256') or provenance.get('result_asset_sha256')
        if not parent:
            continue
        family = siblings.setdefault(parent, [])
        valid = opaque & (original_rgba[..., 3] >= 180)
        rgb = layer['rgba'][..., :3]
        from_source = np.max(np.abs(rgb.astype(np.int16) - original_rgba[..., :3]), axis=2) <= 3
        for lower, lower_opaque, lower_from_source in family:
            overlap = valid & lower_opaque
            area = int(overlap.sum())
            if area <= 250:
                continue
            shared = overlap & from_source & lower_from_source
            left = lower['rgba'][..., :3][shared].astype(np.int16)
            right = rgb[shared]
            duplicate = int((np.max(np.abs(left - right), axis=1) <= 2).sum())
            ratio = duplicate / area
            if duplicate > 250 and ratio > .60:
                warnings.append({'code': 'local_layer_pixel_bleed', 'layers': [lower['id'], layer['id']],
                                 'duplicate_pixels': duplicate, 'overlap_pixels': area,
                                 'duplicate_ratio': ratio, 'suggestion': 'replace copied colors with lower-layer underlap'})
        family.append((layer, opaque, from_source))
    warnings.extend(_check_duplicate_silhouettes(layers, original_rgba))
    return warnings


def cleanup_layers(layers, original_rgba):
    """Layers have id, rgba, role, draw_order, enabled and effect_mode."""
    result=[];metrics={};warnings=inspect_local_layer_warnings(layers,original_rgba)
    for layer in layers:
        clean,stats=clean_layer(layer['rgba'],original_rgba,effect_mode=layer.get('effect_mode','keep'))
        lower=np.zeros(original_rgba.shape[:2],bool)
        for other in layers:
            if other.get('role')!='effect_group' and other['draw_order']<layer['draw_order'] and other.get('enabled',True):
                lower |= other['rgba'][...,3]>=120
        cover=(layer['rgba'][...,3]>=120)&lower
        stats['solid_overlap_ratio']=float(cover.sum()/max(1,(layer['rgba'][...,3]>=120).sum()))
        if layer.get('role')=='effect_group' and (stats['outside_ratio_before']>.10 or stats['solid_overlap_ratio']>.25):
            warnings.append({'code':'MUDDY_EFFECT_LAYER_DETECTED','layer':layer['id'],
                             'suggestion':'omit or original_extract'})
        enabled=layer.get('enabled',True) and layer.get('effect_mode','keep')!='omit'
        result.append({**layer,'rgba':clean,'enabled':enabled});metrics[layer['id']]=stats
    canvas=Image.new('RGBA',(original_rgba.shape[1],original_rgba.shape[0]))
    for layer in sorted(result,key=lambda v:v['draw_order']):
        if layer['enabled']:canvas.alpha_composite(Image.fromarray(layer['rgba']))
    composite=np.asarray(canvas)
    regions=check_missing_original_regions(original_rgba,composite)
    return result,{'layers':metrics,'warnings':warnings,'missing_original_regions':regions},composite


def _body_envelope(mask):
    """Conservative interior envelope; omit exclusions apply only outside it."""
    points=[]
    for y in np.flatnonzero(mask.any(axis=1)):
        xs=np.flatnonzero(mask[y]);points.extend([(int(xs[0]),int(y)),(int(xs[-1]),int(y))])
    points=sorted(set(points))
    if len(points)<3:return mask.copy()
    def cross(a,b,c):return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    lower=[];upper=[]
    for point in points:
        while len(lower)>1 and cross(lower[-2],lower[-1],point)<=0:lower.pop()
        lower.append(point)
    for point in reversed(points):
        while len(upper)>1 and cross(upper[-2],upper[-1],point)<=0:upper.pop()
        upper.append(point)
    canvas=Image.new('L',(mask.shape[1],mask.shape[0]));ImageDraw.Draw(canvas).polygon(lower[:-1]+upper[:-1],fill=255)
    return np.asarray(canvas)>0


def recover_missing_body_from_original(layers, original_rgba, *, ignore_mask=None, min_area=200):
    """Copy missing ORIGINAL material into the best adjacent enabled body layer.

    This restores coverage, not semantic or dynamic approval. Every recovered
    component records its chosen parent layer for the subsequent visual review.
    Omitted effect-only outskirts stay absent; ambiguous interior overlap is
    kept eligible so an opaque effect cannot hide a missing elbow from recovery.
    """
    result=[{**layer,'rgba':np.asarray(layer['rgba']).copy()} for layer in layers]
    shape=original_rgba.shape[:2]
    def composite():
        canvas=Image.new('RGBA',(shape[1],shape[0]))
        for layer in sorted(result,key=lambda item:item['draw_order']):
            if layer.get('enabled',True) and layer.get('effect_mode','keep')!='omit':
                canvas.alpha_composite(Image.fromarray(layer['rgba']))
        return np.asarray(canvas)
    legal=[layer for layer in result if layer.get('enabled',True)
           and layer.get('effect_mode','keep')!='omit'
           and layer.get('role') not in ('effect_group','weapon','prop','pet','background')
           and (layer['rgba'][...,3]>8).any()]
    supports={layer['id']:layer['rgba'][...,3]>8 for layer in legal}
    body=np.zeros(shape,bool)
    for mask in supports.values():body |= mask
    omitted=np.zeros(shape,bool)
    omitted_rgb_samples=[]
    for layer in result:
        if layer.get('role')=='effect_group' and (layer.get('effect_mode')=='omit' or not layer.get('enabled',True)):
            om=layer['rgba'][...,3]>8
            omitted |= om
            omitted_rgb_samples.append((om, layer['rgba'][...,:3]))
    outer_ignored=omitted & ~dilate(_body_envelope(body),2)
    ignored=outer_ignored.copy()
    if ignore_mask is not None:
        supplied=np.asarray(ignore_mask,dtype=bool)
        if supplied.shape!=shape:raise ValueError('ignore_mask must match the original canvas')
        ignored |= supplied
    before=composite();raw_missing=(original_rgba[...,3]>=180)&(before[...,3]<40)
    # Generic omitted-effect chromatic cleanup: learn the omitted effect's hue
    # direction from its outer (non-body) pixels so interior dangling wisps and
    # effect tendrils inside the body's convex hull are cleanly excluded without
    # dropping neutral/body limbs (e.g. a missing elbow under an opaque effect).
    effect_pixel_mask=np.zeros(shape,bool)
    if omitted.any():
        orig_rgb=original_rgba[...,:3].astype(np.float32)
        cmax=orig_rgb.max(axis=2);cmin=orig_rgb.min(axis=2)
        sat=(cmax-cmin)/np.maximum(cmax,1.0)
        sample_mask=outer_ignored & (original_rgba[...,3]>=120) & (sat>0.40)
        if int(sample_mask.sum())<30:
            # Fallback: sample chromatic pixels directly from the omitted layer itself.
            for om, lrgb in omitted_rgb_samples:
                lr=lrgb.astype(np.float32); lm=lr.max(axis=2); ln=lr.min(axis=2)
                ls=(lm-ln)/np.maximum(lm,1.0)
                sample_mask |= om & (ls>0.40)
        if int(sample_mask.sum())>=16:
            pts=orig_rgb[sample_mask] if (outer_ignored & (original_rgba[...,3]>=120) & (sat>0.40)).sum()>=30 else np.concatenate(
                [lrgb.astype(np.float32)[om & ((lrgb.max(axis=2)-lrgb.min(axis=2))/np.maximum(lrgb.max(axis=2),1.0)>0.40)] for om, lrgb in omitted_rgb_samples], axis=0)
            norm_pts=pts/np.maximum(np.linalg.norm(pts,axis=1,keepdims=True),1.0)
            mean_dir=norm_pts.mean(axis=0); nrm=float(np.linalg.norm(mean_dir))
            if nrm>1e-6:
                mean_dir=mean_dir/nrm
                orig_dir=orig_rgb/np.maximum(np.linalg.norm(orig_rgb,axis=2,keepdims=True),1.0)
                cos_sim=np.tensordot(orig_dir,mean_dir,axes=([2],[0]))
                effect_pixel_mask=omitted & (sat>0.42) & (cos_sim>=0.94)
                # Drop whole raw_missing components that are dominated by omitted effect
                # (including their bright hot cores), and trim clinging wisps on mixed limbs.
                pre_missing=raw_missing & ~ignored
                pre_labels=_components(np.pad(pre_missing,1))[1:-1,1:-1]
                pre_counts=np.bincount(pre_labels.ravel())
                for pl in np.flatnonzero(pre_counts>=20):
                    if pl==0:continue
                    comp=pre_labels==pl
                    in_om=float((comp & omitted).sum())/float(pre_counts[pl])
                    eff_r=float((comp & effect_pixel_mask).sum())/float(pre_counts[pl])
                    if in_om>=0.85 and eff_r>=0.50:
                        ignored |= comp
                # Also strip clinging fiery/glowing wisps (plus 1px halo) inside omitted area
                ignored |= dilate(effect_pixel_mask, 1) & omitted
    missing=raw_missing & ~ignored
    labels=_components(np.pad(missing,1))[1:-1,1:-1];counts=np.bincount(labels.ravel())
    recovered=[];unresolved=[]
    centers={name:np.array(np.nonzero(mask)).mean(axis=1) for name,mask in supports.items()}
    priority={'torso':0,'body':0,'upper_arm':1,'forearm':1,'arm':1,'leg':1,'upper_leg':1,'lower_leg':1,'hand':2,'cloth':3}
    for label in np.flatnonzero(counts>=min_area):
        if label==0:continue
        region=labels==label;ys,xs=np.nonzero(region);center=np.array([ys.mean(),xs.mean()])
        record={'area_px':int(counts[label]),'bbox':[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)]}
        if not legal:
            unresolved.append({**record,'reason':'no_enabled_body_layer'});continue
        boundary=dilate(region,1)&~region
        contacts={layer['id']:int((boundary&supports[layer['id']]).sum()) for layer in legal}
        layer=min(legal,key=lambda item:(-contacts[item['id']],
                  float(np.linalg.norm(centers[item['id']]-center)),priority.get(item.get('role'),4)))
        # Two-pixel feathered ORIGINAL underlap, never paint transparent source
        # holes or intentionally ignored effects. Existing opaque material stays.
        ring1=dilate(region,1)&~region;ring2=dilate(region,2)&~(region|ring1)
        support=(region|ring1|ring2)&(original_rgba[...,3]>8)&~ignored
        patch=np.zeros_like(original_rgba);patch[support]=original_rgba[support]
        patch[...,3][ring1]=np.rint(patch[...,3][ring1]*(2.0/3.0)).astype('uint8')
        patch[...,3][ring2]=np.rint(patch[...,3][ring2]/3).astype('uint8')
        base=Image.fromarray(patch);base.alpha_composite(Image.fromarray(layer['rgba']))
        layer['rgba']=np.asarray(base).copy();layer['rgba'][region]=original_rgba[region]
        recovered.append({**record,'target_layer':layer['id'],'target_role':layer.get('role'),
                          'boundary_contact_pixels':contacts[layer['id']],
                          'selection':'longest_boundary' if contacts[layer['id']] else 'nearest_center',
                          'overlap_feather_px':2,'material_source':'original_rgba'})
    after=composite();body_reference=original_rgba.copy();body_reference[ignored]=0
    report={'recovered_regions':recovered,'recovered_region_count':len(recovered),
            'recovered_pixel_count':sum(r['area_px'] for r in recovered),
            'ignored_omitted_effect_pixels':int((raw_missing&ignored).sum()),
            'unresolved_recovery_regions':unresolved,
            'missing_original_regions':check_missing_original_regions(body_reference,after),
            'missing_original_regions_raw':check_missing_original_regions(original_rgba,after),
            'recovery_review':'pending_semantics_and_full_loop', '_ignore_mask':ignored}
    return result,report,after


def _read_canvas_source(parts, root):
    path = resolve_evidence_path(parts['originalSource'], root=root)
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != parts['originalSourceSHA256']:
        raise ValueError('Original source changed after materialization')
    with Image.open(io.BytesIO(data)) as image:
        if image.size != tuple(parts['canvas']):
            raise ValueError('Original source size mismatch')
        return np.array(image.convert('RGBA'))


def _read_cleanup_layer(part, size, root):
    entry = part['canvasLayer']
    _, _, rgba = read_evidence_rgba(entry['image'], entry['sha256'], size, root=root)
    return {'id': part['name'], 'rgba': rgba, 'kind': part['kind'], 'provenance': entry['provenance'],
            'draw_order': part['drawOrder'], 'enabled': part.get('enabled', True)}


def inspect_canvas_parts_warnings(parts_dir, *, evidence_root):
    manifest = resolve_evidence_path(Path(parts_dir) / 'parts.json', root=evidence_root)
    parts = json.loads(manifest.read_text())
    original = _read_canvas_source(parts, evidence_root)
    layers = [_read_cleanup_layer(part, parts['canvas'], evidence_root) for part in parts['parts']]
    warnings = inspect_local_layer_warnings(layers, original)
    canvas = Image.new('RGBA', tuple(parts['canvas']))
    for layer in sorted(layers, key=lambda item: item['draw_order']):
        if layer.get('enabled', True) and layer.get('effect_mode', 'keep') != 'omit':
            canvas.alpha_composite(Image.fromarray(layer['rgba']))
    fidelity = check_original_fidelity(original, np.asarray(canvas), layers=layers)
    if fidelity['drifted']:
        warnings.append({'code': 'canvas_original_rgb_drift', **fidelity,
                         'suggestion': 'Canvas 拆层合成与原图外观偏差过大，回 Canvas 重拆或降级本机拆分'})
    return {'layers': {}, 'warnings': warnings, 'original_fidelity': fidelity}


def prepare_canvas_parts(parts_dir, plan, out_dir, *, evidence_root, cleanup=True, tweaks=None, recover_missing=True):
    """Re-materialize source layers; always start from base inputs, never compound edits."""
    from .canvas_layer_parts import materialize_canvas_layer_parts
    manifest_path=resolve_evidence_path(Path(parts_dir)/'parts.json',root=evidence_root)
    parts=json.loads(manifest_path.read_text());plan=copy.deepcopy(plan)
    tweaks=tweaks or {}; overrides=tweaks.get('layers',{}); scales=tweaks.get('modules',{})
    byid={m['id']:m for m in plan['modules']}
    for mid,changes in scales.items():
        if mid not in byid:raise ValueError('Unknown module '+mid)
        byid[mid]['motion_scale']=changes.get('motion_scale',byid[mid].get('motion_scale',1.))
    original=_read_canvas_source(parts,evidence_root)
    layers=[];entries={}
    for part in parts['parts']:
        name=part['name'];opt=overrides.get(name,{})
        layers.append({**_read_cleanup_layer(part,parts['canvas'],evidence_root),
                       'role':byid[part['moduleId']]['role'],'draw_order':opt.get('drawOrder',part['drawOrder']),
                       'enabled':opt.get('enabled',part.get('enabled',True)),
                       'effect_mode':opt.get('effect_mode',byid[part['moduleId']].get('effect_mode','keep'))})
        entries[name]=part
    if set(overrides)-set(entries):raise ValueError('Unknown layer')
    for m in plan['modules']:m.pop('effect_mode',None)
    if cleanup:layers,report,composite=cleanup_layers(layers,original)
    else:
        for layer in layers:layer['enabled'] &= layer['effect_mode']!='omit'
        report={'layers':{},'warnings':inspect_local_layer_warnings(layers,original)}
    recovery_ignore=np.zeros(original.shape[:2],bool)
    if cleanup and recover_missing:
        layers,recovery,composite=recover_missing_body_from_original(layers,original)
        recovery_ignore=recovery.pop('_ignore_mask');report.update(recovery)
    out=Path(out_dir).resolve();out.relative_to(Path(evidence_root).resolve());out.mkdir(parents=True,exist_ok=False)
    assets=out/'layers';assets.mkdir();manifest={'version':1,'material_mode':'canvas_layers',
        'source_image':parts['originalSource'],'source_sha256':parts['originalSourceSHA256'],'image_size':parts['canvas'],'pieces':[]}
    for order,layer in enumerate(sorted(layers,key=lambda v:v['draw_order'])):
        part=entries[layer['id']];image=assets/(layer['id']+'.png');rgba=layer['rgba']
        # A fully removed background layer is disabled; its original support is
        # retained solely for stable bone planning, never drawn in the preview.
        if not rgba[...,3].any():
            layer['enabled']=False;rgba=_read_cleanup_layer(part,parts['canvas'],evidence_root)['rgba']
            report['warnings'].append({'code':'EMPTY_LAYER_DISABLED','layer':layer['id']})
        Image.fromarray(rgba).save(image)
        cleaned_sha=hashlib.sha256(image.read_bytes()).hexdigest()
        prov={**part['canvasLayer']['provenance'],'cleanup_source_image':part['canvasLayer']['image']}
        if prov.get('operation')=='local-direct-split':
            prov['cleanup_pre_result_sha256']=prov.get('result_asset_sha256')
            prov['result_asset_sha256']=cleaned_sha
        manifest['pieces'].append({'id':part['name'],'module_id':part['moduleId'],'kind':part['kind'],
            'label':part.get('label'),
            'parent':part['parent'],'pivot':part['pivot'],'draw_order':order,'enabled':layer['enabled'],
            'image':str(image),'sha256':cleaned_sha,
            'provenance':prov})
    result=materialize_canvas_layer_parts(manifest,out/'parts',evidence_root=evidence_root)
    composite=np.asarray(Image.open(result['source']).convert('RGBA'))
    reference=original.copy();reference[recovery_ignore]=0
    report['missing_original_regions']=check_missing_original_regions(reference,composite)
    fidelity=check_original_fidelity(reference,composite,layers=layers)
    report['original_fidelity']=fidelity
    if fidelity['drifted'] and not any(w.get('code')=='canvas_original_rgb_drift' for w in report.get('warnings',[])):
        report.setdefault('warnings',[]).append({'code':'canvas_original_rgb_drift',**fidelity,
            'suggestion':'Canvas 拆层合成与原图外观偏差过大，回 Canvas 重拆或降级本机拆分'})
    marked=Image.fromarray(composite);background=Image.new('RGBA',marked.size,(125,125,125,255));background.alpha_composite(marked)
    draw=ImageDraw.Draw(background)
    for r in report['missing_original_regions']:draw.rectangle(r['bbox'],outline='red',width=4)
    background.convert('RGB').save(out/'missing-original-regions.jpg')
    (out/'cleanup-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    (out/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2))
    return out/'parts',plan,report
