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
    return np.asarray(Image.fromarray(mask.astype('uint8')*255).filter(ImageFilter.MaxFilter(radius*2+1))) > 0


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
    alpha = layer[...,3]
    # Propagate nearby opaque interior colors into white-contaminated fringes.
    known = alpha >= 245; colors = layer[...,:3].astype(np.float32)
    for _ in range(3):
        sums=np.zeros_like(colors); counts=np.zeros(alpha.shape,np.float32)
        pad=np.pad(known,1); cp=np.pad(colors,((1,1),(1,1),(0,0)))
        for dy,dx in ((0,1),(2,1),(1,0),(1,2)):
            k=pad[dy:dy+alpha.shape[0],dx:dx+alpha.shape[1]]
            sums += cp[dy:dy+alpha.shape[0],dx:dx+alpha.shape[1]]*k[...,None];counts+=k
        take=(~known)&(counts>0);colors[take]=sums[take]/counts[take,None];known |= take
    edge=(alpha>3)&(alpha<245)&known
    white_bias=(layer[...,:3].astype(float)-colors).mean(axis=2)>12
    take=edge&white_bias
    layer[...,:3][take]=np.rint(colors[take]).astype('uint8')
    stats['defringed_pixels']=int(take.sum())
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
    return {'layers': {}, 'warnings': inspect_local_layer_warnings(layers, original)}


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
        manifest['pieces'].append({'id':part['name'],'module_id':part['moduleId'],'kind':part['kind'],
            'label':part.get('label'),
            'parent':part['parent'],'pivot':part['pivot'],'draw_order':order,'enabled':layer['enabled'],
            'image':str(image),'sha256':hashlib.sha256(image.read_bytes()).hexdigest(),
            'provenance':{**part['canvasLayer']['provenance'],'cleanup_source_image':part['canvasLayer']['image']}})
    result=materialize_canvas_layer_parts(manifest,out/'parts',evidence_root=evidence_root)
    composite=np.asarray(Image.open(result['source']).convert('RGBA'))
    reference=original.copy();reference[recovery_ignore]=0
    report['missing_original_regions']=check_missing_original_regions(reference,composite)
    marked=Image.fromarray(composite);background=Image.new('RGBA',marked.size,(125,125,125,255));background.alpha_composite(marked)
    draw=ImageDraw.Draw(background)
    for r in report['missing_original_regions']:draw.rectangle(r['bbox'],outline='red',width=4)
    background.convert('RGB').save(out/'missing-original-regions.jpg')
    (out/'cleanup-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    (out/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2))
    return out/'parts',plan,report
