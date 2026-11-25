"""Local, source-constrained cleanup of already aligned Canvas layers."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from .split_parts import _components


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


def cleanup_layers(layers, original_rgba):
    """Layers have id, rgba, role, draw_order, enabled and effect_mode."""
    result=[];metrics={};warnings=[]
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
    for layer in result:
        if layer.get('role')=='effect_group' and (layer.get('effect_mode')=='omit' or not layer.get('enabled',True)):
            omitted |= layer['rgba'][...,3]>8
    ignored=omitted & ~dilate(_body_envelope(body),2)
    if ignore_mask is not None:
        supplied=np.asarray(ignore_mask,dtype=bool)
        if supplied.shape!=shape:raise ValueError('ignore_mask must match the original canvas')
        ignored |= supplied
    before=composite();raw_missing=(original_rgba[...,3]>=180)&(before[...,3]<40)
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


def prepare_canvas_parts(parts_dir, plan, out_dir, *, evidence_root, cleanup=True, tweaks=None, recover_missing=True):
    """Re-materialize source layers; always start from base inputs, never compound edits."""
    from .canvas_layer_parts import materialize_canvas_layer_parts
    parts=json.loads((Path(parts_dir)/'parts.json').read_text());plan=copy.deepcopy(plan)
    tweaks=tweaks or {}; overrides=tweaks.get('layers',{}); scales=tweaks.get('modules',{})
    byid={m['id']:m for m in plan['modules']}
    for mid,changes in scales.items():
        if mid not in byid:raise ValueError('Unknown module '+mid)
        byid[mid]['motion_scale']=changes.get('motion_scale',byid[mid].get('motion_scale',1.))
    original=np.asarray(Image.open(parts['originalSource']).convert('RGBA'))
    layers=[];entries={}
    for part in parts['parts']:
        name=part['name'];opt=overrides.get(name,{})
        layers.append({'id':name,'rgba':np.asarray(Image.open(part['canvasLayer']['image']).convert('RGBA')),
                       'role':byid[part['moduleId']]['role'],'draw_order':opt.get('drawOrder',part['drawOrder']),
                       'enabled':opt.get('enabled',part.get('enabled',True)),
                       'effect_mode':opt.get('effect_mode',byid[part['moduleId']].get('effect_mode','keep'))})
        entries[name]=part
    if set(overrides)-set(entries):raise ValueError('Unknown layer')
    for m in plan['modules']:m.pop('effect_mode',None)
    if cleanup:layers,report,composite=cleanup_layers(layers,original)
    else:
        for layer in layers:layer['enabled'] &= layer['effect_mode']!='omit'
        report={'layers':{},'warnings':[]}
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
            layer['enabled']=False;rgba=np.asarray(Image.open(part['canvasLayer']['image']).convert('RGBA'))
            report['warnings'].append({'code':'EMPTY_LAYER_DISABLED','layer':layer['id']})
        Image.fromarray(rgba).save(image)
        manifest['pieces'].append({'id':part['name'],'module_id':part['moduleId'],'kind':part['kind'],
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
