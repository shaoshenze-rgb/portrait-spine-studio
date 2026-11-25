"""Sixteen-pose visual review aids. Candidates are warnings, not semantic proof."""
import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from src.pipeline.canvas_layer_cleanup import check_missing_original_regions
from src.pipeline.split_parts import _components
from src.render.renderer import over_bg


def inspect_dynamic(renderer, frames, original, out_dir):
    out=Path(out_dir);times=sorted(frames);base=frames[0.]
    report={'sample_times':times,'gap_detection_scope':'new enclosed transparent holes; open contact seams are shown by relative-angle crops',
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
    if report['new_gap']['bbox']: rows.insert(0,('Largest new alpha gap',report['new_gap']))
    if not rows:rows=[('Full character',{'time':times[-1],'bbox':[0,0,base.shape[1],base.shape[0]]})]
    tile=420;sheet=Image.new('RGB',(tile*3, (tile+34)*len(rows)),(115,115,115));draw=ImageDraw.Draw(sheet)
    original_image=Image.fromarray(original).convert('RGBA');bg=Image.new('RGBA',original_image.size,(125,125,125,255));bg.alpha_composite(original_image)
    for i,(label,record) in enumerate(rows):
        box=record['bbox'];t=record['time']
        for j,im in enumerate([bg.convert('RGB'),Image.fromarray(over_bg(base,(125,125,125))),Image.fromarray(over_bg(frames[t],(125,125,125)))]):
            crop=im.crop(box);scale=tile/max(crop.size);crop=crop.resize((max(1,round(crop.width*scale)),max(1,round(crop.height*scale))),Image.Resampling.LANCZOS);x=j*tile;y=i*(tile+34)
            sheet.paste(crop,(x+(tile-crop.width)//2,y+34))
            draw.text((x+6,y+8),label+' '+['original','t=0',f't={t:g}s'][j],fill='white')
    sheet.save(out/'seam-worst-crops.jpg',quality=94)
    return report
