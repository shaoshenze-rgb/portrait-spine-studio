"""Image-bound connection diagnosis. Findings route work; they never move joints."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from src.gates.review_binding import read_evidence_rgba, resolve_evidence_path
from src.pipeline.module_rig import _mask_distance, _placement_tol
from src.studio.module_contract import BODY_ROLES

DIAGNOSTIC_RULE_VERSION='connection-diagnostics-2'


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _point(value, size):
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in value)):
        return None
    return [float(value[0])*size[0], float(value[1])*size[1]]


def inspect_canvas_connections(parts_dir, plan, *, evidence_root, out_dir=None):
    """Collect all connection defects, keeping geometric evidence and uncertainty.

    Low-opacity endpoints are review candidates relative to the layer's opacity,
    not a new universal opacity gate. Material gaps do not automatically authorize
    inpainting: source/structure review must decide whether to replan or repair.
    """
    root = Path(evidence_root).resolve()
    manifest_path = resolve_evidence_path(Path(parts_dir)/'parts.json', root=root)
    material_bytes = manifest_path.read_bytes(); material = json.loads(material_bytes)
    if material.get('materialMode') != 'canvas_layers':
        raise ValueError('Connection diagnostics require Canvas layer parts')
    w, h = material['canvas']
    source = resolve_evidence_path(material['originalSource'], root=root)
    if hashlib.sha256(source.read_bytes()).hexdigest() != material['originalSourceSHA256']:
        raise ValueError('Original source changed')
    with Image.open(source) as image:
        if image.size != (w, h): raise ValueError('Original source canvas differs')
        source_rgba = np.asarray(image.convert('RGBA')).copy()
    _, _, setup = read_evidence_rgba(material['source'], material['sourceSHA256'], (w, h), root=root)
    modules = plan.get('modules')
    connections = plan.get('connections')
    if not isinstance(modules, list) or not isinstance(connections, list):
        raise ValueError('Plan needs modules and connections')
    if any(not isinstance(m,dict) or not isinstance(m.get('id'),str) for m in modules):
        raise ValueError('Invalid module identity')
    if any(not isinstance(c,dict) for c in connections):raise ValueError('Invalid connection declaration')
    by_id = {m['id']: m for m in modules}
    if len(by_id) != len(modules): raise ValueError('Duplicate module identity')
    layers, fingerprints, part_metadata = {}, {}, {}
    composite = Image.new('RGBA', (w, h))
    names, orders = set(), set()
    for part in sorted(material['parts'], key=lambda p:p['drawOrder']):
        if part['name'] in names or part['drawOrder'] in orders: raise ValueError('Duplicate part/order')
        names.add(part['name']); orders.add(part['drawOrder'])
        layer = part['canvasLayer']
        path, sha, rgba = read_evidence_rgba(layer['image'], layer['sha256'], (w, h), root=root)
        if not (rgba[...,3]>0).any(): raise ValueError('Empty layer material')
        x,y,pw,ph=part['bbox']
        if (any(type(v) is not int for v in (x,y,pw,ph)) or min(x,y)<0
                or min(pw,ph)<=0 or x+pw>w or y+ph>h):raise ValueError('Invalid material crop bounds')
        crop_path=resolve_evidence_path(part['image'],root=manifest_path.parent,base=manifest_path.parent)
        _,_,crop=read_evidence_rgba(crop_path,part['imageSHA256'],(pw,ph),root=manifest_path.parent)
        if not np.array_equal(crop,rgba[y:y+ph,x:x+pw]):raise ValueError('Material crop differs from its layer')
        fingerprints[part['name']] = {'path': str(path.relative_to(root)), 'sha256': sha}
        image = Image.fromarray(rgba)
        composite = Image.alpha_composite(composite, image)
        mid = part['moduleId']
        part_metadata.setdefault(mid,[]).append(part)
        layers[mid] = Image.alpha_composite(layers.get(mid, Image.new('RGBA', (w, h))), image)
    if not np.array_equal(np.asarray(composite), setup):
        raise ValueError('Parts layer order/material differs from its setup reference')
    if set(layers)!=set(by_id):
        raise ValueError('Plan must account for every material module; no partial-character substitution')
    arrays = {mid:np.asarray(image) for mid,image in layers.items()}
    masks = {mid:a[...,3]>0 for mid,a in arrays.items()}
    body = np.zeros((h,w), bool)
    for mid,m in by_id.items():
        if mid in masks and m.get('role') in BODY_ROLES: body |= masks[mid]
    ys = np.nonzero(body)[0]; body_h = int(ys.max()-ys.min()+1) if len(ys) else h
    tol = _placement_tol(body_h)
    revision = _digest({'rules_version':DIAGNOSTIC_RULE_VERSION,'parts_path':str(manifest_path.relative_to(root)),
                        'parts_sha256':hashlib.sha256(material_bytes).hexdigest(),
                        'plan':plan,'source_sha256':material['originalSourceSHA256'],'layers':fingerprints})
    defects, rows = [], []
    def add(cid, code, message, action, evidence, blocking=True):
        defects.append({'id':cid+':'+code,'connection_id':cid,'code':code,
                        'message':message,'recommended_action':action,'requires_review':blocking,
                        'evidence':evidence})
    seen = set()
    for c in connections:
        cid = c.get('id')
        if not isinstance(cid,str) or not cid or cid in seen: raise ValueError('Invalid connection identity')
        seen.add(cid); a,b = c.get('a'),c.get('b'); point = _point(c.get('point'),(w,h))
        evidence = {'parent':a,'child':b,'type':c.get('type'),'point_px':point,'tolerance_px':tol}
        row = {'id':cid,'evidence':evidence}; rows.append(row)
        if a not in masks or b not in masks or a not in by_id or b not in by_id:
            add(cid,'missing_module','连接引用的模块材质或规划缺失。','review_connection_plan',evidence);continue
        if point is None:
            add(cid,'invalid_anchor','连接点不是有限二维坐标。','review_connection_plan',evidence);continue
        x,y = point; xi,yi = round(x),round(y)
        if not (0<=xi<w and 0<=yi<h):
            add(cid,'anchor_out_of_canvas','连接点落在画布外。','review_connection_plan',evidence);continue
        child_alpha = arrays[b][...,3]; positive = child_alpha[child_alpha>0]
        reference_alpha = float(np.percentile(positive,95))
        alpha = int(child_alpha[yi,xi]); ratio = alpha/max(1.,reference_alpha)
        parent_distance = _mask_distance(masks[a],x,y);child_distance = _mask_distance(masks[b],x,y)
        core_distance = _mask_distance(child_alpha>=max(1.,reference_alpha*.5),x,y)
        py,px = np.nonzero(masks[a]);nearest = int(np.argmin((px-x)**2+(py-y)**2))
        evidence.update(parent_distance_px=parent_distance,child_distance_px=child_distance,
            child_alpha=alpha,child_reference_alpha_p95=reference_alpha,child_opacity_ratio=ratio,
            child_core_distance_px=core_distance,
            parent_alpha=int(arrays[a][yi,xi,3]),
            nearest_parent_px=[int(px[nearest]),int(py[nearest])],
            source_alpha=int(source_rgba[yi,xi,3]),setup_alpha=int(setup[yi,xi,3]),
            other_covering_modules=[{'module':mid,'role':by_id[mid].get('role'),'alpha':int(rgba[yi,xi,3])}
                                    for mid,rgba in arrays.items() if mid not in (a,b) and rgba[yi,xi,3]>0],
            hidden_references_present=bool(c.get('hidden_by')))
        module_pivot = _point(by_id[b].get('pivot'),(w,h))
        if module_pivot is not None and any(
                part.get('parent')!=by_id[b].get('parent')
                or part.get('kind')!=by_id[b].get('motion_type')
                or math.dist(part['pivot'],module_pivot)>1e-4 for part in part_metadata[b]):
            add(cid,'material_plan_mismatch','规划已改变，但parts仍保存旧父级、类型或枢轴。',
                'rematerialize_reviewed_plan',evidence)
        if module_pivot is None or math.dist(module_pivot,point)>1e-4:
            add(cid,'connection_pivot_mismatch','连接点与子模块枢轴不一致。','review_connection_plan',evidence)
        if alpha==0 or child_distance>tol:
            add(cid,'anchor_outside_child','连接点缺少子模块材质支持。','review_connection_plan',evidence)
        elif ratio<.1:
            weak_connection=parent_distance>tol or core_distance>tol
            add(cid,'fringe_anchor',
                '极淡边缘锚点缺少邻近连接支持，需核对真实结构。' if weak_connection else '边缘锚点邻近完整材质，随正常绑定和循环检查观察。',
                'review_connection_plan' if weak_connection else 'verify_binding_and_cycle',evidence,weak_connection)
        if parent_distance>tol:
            if c.get('hidden_by'):
                add(cid,'declared_hidden_connection','连接声明了遮挡；须由编译核验及全循环覆盖检查证明。',
                    'verify_declared_coverage',evidence,False)
            else:
                add(cid,'parent_support_gap','父侧缺少连接材质支持；先区分规划错误、真实遮挡或缺失材质。',
                    'review_connection_structure',evidence)
    output = None
    if out_dir is not None:
        output = Path(out_dir).absolute();output.resolve().relative_to(root)
        if output.exists() or output.is_symlink():raise ValueError('Use a new diagnostic directory')
        output.mkdir(parents=True)
        for row in rows:
            e=row['evidence']
            if not e.get('point_px') or e['parent'] not in arrays or e['child'] not in arrays:continue
            x,y=e['point_px'];nx,ny=e.get('nearest_parent_px',[x,y]);pad=max(24,round(body_h/60))
            box=(max(0,int(min(x,nx))-pad),max(0,int(min(y,ny))-pad),
                 min(w,int(max(x,nx))+pad+1),min(h,int(max(y,ny))+pad+1))
            if box[2]<=box[0] or box[3]<=box[1]:continue
            stem=hashlib.sha256(row['id'].encode()).hexdigest()[:16];e['crop_box']=list(box);e['images']={}
            for label,rgba in [('source',source_rgba),('parent',arrays[e['parent']]),('child',arrays[e['child']]),('composite',setup)]:
                tile=Image.alpha_composite(Image.new('RGBA',(w,h),(145,145,145,255)),Image.fromarray(rgba)).crop(box).convert('RGB')
                draw=ImageDraw.Draw(tile);cx,cy=x-box[0],y-box[1];draw.ellipse((cx-3,cy-3,cx+3,cy+3),outline=(255,40,70),width=2)
                path=output/(stem+'-'+label+'.png');tile.save(path);e['images'][label]=str(path)
    result={'version':1,'rules_version':DIAGNOSTIC_RULE_VERSION,
            'status':'needs_connection_review' if any(d['requires_review'] for d in defects) else 'ready_for_binding_checks',
            'input_revision':revision,'source_sha256':material['originalSourceSHA256'],
            'parts_sha256':hashlib.sha256(material_bytes).hexdigest(),'plan_sha256':_digest(plan),
            'connections':rows,'defects':defects,'semantic_review':'not_run',
            'notes':'No joint/material/parent changed. Findings are not anatomical truth or permission to generate.'}
    if output:(output/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    return result
