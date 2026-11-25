#!/usr/bin/env python3
"""Import real Canvas receipts, align layers, attach material to existing V2 jobs.

Cindy invokes Canvas through its native tools; this script never calls a model,
uses credentials, invents task receipts, or grants visual approval.
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
import math
import re
import shutil
import sys
import uuid
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw

REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_asset(directory,node):
    ident=node['node_id']
    if not isinstance(ident,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}',ident):
        raise ValueError('Canvas节点身份非法，不能作为文件路径。')
    directory=Path(directory).resolve();p=(directory/(ident+'.png')).resolve()
    p.relative_to(directory)
    if sha(p)!=node['asset_sha256']:raise ValueError('Canvas原件摘要不匹配：'+ident)
    return p


def register_layer(layer,source):
    """Existing optional vision stack, confined to tooling rather than src/."""
    import cv2
    def small(a):
        scale=min(1.,1600./max(a.shape[:2]))
        size=(max(1,round(a.shape[1]*scale)),max(1,round(a.shape[0]*scale)))
        return cv2.resize(a,size,interpolation=cv2.INTER_AREA),scale
    l,ls=small(layer);s,ss=small(source)
    detector=cv2.SIFT_create(nfeatures=6000)
    lk,ld=detector.detectAndCompute(cv2.cvtColor(l[...,:3],cv2.COLOR_RGB2GRAY),(l[...,3]>32).astype('uint8')*255)
    sk,sd=detector.detectAndCompute(cv2.cvtColor(s[...,:3],cv2.COLOR_RGB2GRAY),(s[...,3]>32).astype('uint8')*255)
    if ld is None or sd is None:return None,{'reason':'insufficient source features'}
    pairs=cv2.BFMatcher().knnMatch(ld,sd,k=2)
    matches=[a for pair in pairs if len(pair)==2 for a,b in [pair] if a.distance<.70*b.distance]
    if len(matches)<12:return None,{'reason':'too few reliable correspondences','matches':len(matches)}
    a=np.float32([lk[m.queryIdx].pt for m in matches]);b=np.float32([sk[m.trainIdx].pt for m in matches])
    matrix,inliers=cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=2.,maxIters=3000,confidence=.997)
    if matrix is None:return None,{'reason':'registration failed'}
    keep=inliers[:,0].astype(bool);count=int(keep.sum())
    residual=np.linalg.norm(a@matrix[:,:2].T+matrix[:,2]-b,axis=1)
    angle=math.degrees(math.atan2(matrix[1,0],matrix[0,0]));scale=float(np.hypot(matrix[0,0],matrix[1,0])*ls/ss)
    ys,xs=np.nonzero(l[...,3]>32)
    points=a[keep];extent=float(np.ptp(points[:,0])*np.ptp(points[:,1])/max(1,(xs.max()-xs.min())*(ys.max()-ys.min())))
    metrics={'matches':len(matches),'inliers':count,'inlier_ratio':count/len(matches),
             'median_residual_px':float(np.median(residual[keep])/ss),'rotation_degrees':angle,
             'scale':scale,'matched_extent_fraction':extent}
    if count<12 or count/len(matches)<.25 or abs(angle)>3 or not .03<scale<20 or extent<.12:
        metrics['reason']='alignment confidence insufficient; return to Canvas/refinement, never hand-drag silently'
        return None,metrics
    matrix[:,:2]*=ls/ss;matrix[:,2]/=ss
    return matrix,metrics


def warp_rgba(layer,matrix,size):
    import cv2
    pm=layer.astype(np.float32);pm[...,:3]*=pm[...,3:4]/255.
    out=cv2.warpAffine(pm,matrix,tuple(size),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    alpha=out[...,3:4];rgb=np.divide(out[...,:3]*255,alpha,out=np.zeros_like(out[...,:3]),where=alpha>0)
    result=np.clip(np.rint(np.concatenate((rgb,alpha),axis=2)),0,255).astype('uint8')
    result[result[...,3]==0]=0
    return result


def gray(rgba):
    a=rgba[...,3:4].astype(float)/255
    return Image.fromarray(np.clip(rgba[...,:3]*a+145*(1-a),0,255).astype('uint8'))


def import_layers(receipt_path,source_path,working_path,assets_dir,out_dir,*,parent_report=None):
    out=Path(out_dir).resolve();out.mkdir(parents=True,exist_ok=True)
    pending=out/'CANVAS_IMPORT_PENDING';failed=out/'CANVAS_IMPORT_FAILED'
    pending.write_text('Import in progress; previous result is not consumable.')
    try:
        result=_import_layers(receipt_path,source_path,working_path,assets_dir,out,parent_report=parent_report)
    except Exception as exc:
        failed.write_text(json.dumps({'error':type(exc).__name__,'message':str(exc)},ensure_ascii=False))
        raise
    pending.unlink(missing_ok=True);failed.unlink(missing_ok=True)
    return result


def _import_layers(receipt_path,source_path,working_path,assets_dir,out_dir,*,parent_report=None):
    receipt=json.loads(Path(receipt_path).read_text());out=Path(out_dir).resolve();out.mkdir(parents=True,exist_ok=True)
    operation=receipt.get('operation')
    if receipt.get('version')!=1 or operation not in ('split-layers','edit-image'):
        raise ValueError('需要真实Canvas拆层或改图回执清单。')
    if sha(source_path)!=receipt.get('source_sha256'):raise ValueError('Canvas源图与本次原件不一致。')
    nodes=receipt.get('nodes')or[]
    valid_count=(2<=len(nodes)<=100) if operation=='split-layers' else len(nodes)==1
    if not valid_count or len({n['node_id'] for n in nodes})!=len(nodes):raise ValueError('Canvas返回层清单为空/重复/超限。')
    for key in ('scene_id','source_node_id','task_id'):
        if not isinstance(receipt.get(key),str) or not receipt[key].strip():raise ValueError('Canvas回执缺少'+key)
    if operation=='split-layers':
        if not isinstance(receipt.get('base_node_id'),str) or receipt['base_node_id'] not in {n['node_id'] for n in nodes}:
            raise ValueError('Canvas回执缺少底图身份。')
    elif receipt.get('base_node_id') is not None:
        raise ValueError('Canvas单张改图没有返回底图，不能伪造拆层回执。')
    if any(n.get('task_id',receipt['task_id'])!=receipt['task_id'] for n in nodes):raise ValueError('返回层混入其他Canvas生成任务。')
    with Image.open(source_path) as im:original_size=im.size
    with Image.open(working_path) as im:source=np.asarray(im.convert('RGBA')).copy();working_size=im.size
    lineage=None
    parent_transform=None;registration_source=source
    if parent_report:
        parent=json.loads(Path(parent_report).read_text())
        source_layer=next((l for l in parent['layers'] if l['node_id']==receipt['source_node_id']),None)
        if (not source_layer or source_layer['raw_sha256']!=receipt['source_sha256']
                or parent['input_sha256']!=sha(working_path) or parent['canvas_scene_id']!=receipt['scene_id']):
            raise ValueError('Canvas局部返修不属于当前输入链。')
        if source_layer.get('status')!='aligned_pending_review':raise ValueError('父层尚未对齐，不能猜测子层位置。')
        parent_transform=np.asarray(source_layer.get('transform'),dtype=float)
        if parent_transform.shape!=(2,3) or not np.isfinite(parent_transform).all():raise ValueError('父层变换无效。')
        linear=parent_transform[:,:2];singular=np.linalg.svd(linear,compute_uv=False)
        if np.linalg.det(linear)<=0 or not .03<singular[0]<20 or not np.allclose(singular,singular[0],rtol=1e-6):
            raise ValueError('父层变换不能反射、扭曲或异常缩放。')
        with Image.open(source_path) as im:registration_source=np.asarray(im.convert('RGBA')).copy()
        lineage={'parent_report_sha256':sha(parent_report),'source_node_id':receipt['source_node_id'],
                 'root_source_asset_sha256':parent.get('root_source_asset_sha256',parent['source_asset_sha256'])}
    elif abs(original_size[0]/original_size[1]-working_size[0]/working_size[1])>.003:
        raise ValueError('工作图比例与Canvas源图不一致；子层须携带父交接记录。')
    shutil.copy2(receipt_path,out/'canvas-receipt.json')
    results=[]
    for node in nodes:
        path=checked_asset(assets_dir,node)
        if node['node_id']==receipt.get('base_node_id'):continue
        with Image.open(path) as im:
            if im.mode!='RGBA':raise ValueError('图层原件必须有RGBA，不能用截图或画布导出替代。')
            layer=np.asarray(im).copy()
        matrix,metrics=register_layer(layer,registration_source)
        if matrix is not None and parent_transform is not None:
            # A completed hand may not exist in the flattened original. Match
            # against its actual Canvas parent, then compose the already-checked
            # parent registration. Never invent a location for the new pixels.
            parent_affine=np.vstack((parent_transform,[0,0,1]))
            matrix=(parent_affine@np.vstack((matrix,[0,0,1])))[:2]
            metrics['reference']='immediate_canvas_parent_then_root'
            metrics['composed_rotation_degrees']=math.degrees(math.atan2(matrix[1,0],matrix[0,0]))
            if abs(metrics['composed_rotation_degrees'])>3:
                metrics['reason']='composed layer rotation exceeds source-alignment limit';matrix=None
        entry={'node_id':node['node_id'],'title':node.get('title',''),'raw_sha256':node['asset_sha256'],
               'task_id':node.get('task_id') or receipt['task_id'],'registration':metrics,'semantic_review':'unreviewed'}
        if matrix is None:
            entry['status']='needs_canvas_registration';results.append(entry);continue
        aligned=warp_rgba(layer,matrix,working_size)
        dest=out/(node['node_id']+'-aligned.png');Image.fromarray(aligned).save(dest)
        from src.pipeline.canvas_layer_cleanup import clean_layer
        cleaned,cleanup_stats=clean_layer(aligned,source)
        clean_path=out/(node['node_id']+'-cleaned.png');Image.fromarray(cleaned).save(clean_path)
        entry['cleanup']={**cleanup_stats,'image':str(clean_path),'sha256':sha(clean_path)}
        entry.update(status='aligned_pending_review',image=str(dest),sha256=sha(dest),transform=matrix.tolist())
        ys,xs=np.nonzero(aligned[...,3]>0)
        if not len(xs):raise ValueError('对齐图层落在画布外。')
        box=(max(0,int(xs.min())-8),max(0,int(ys.min())-8),min(working_size[0],int(xs.max())+9),min(working_size[1],int(ys.max())+9))
        left,right=gray(source).crop(box),gray(aligned).crop(box)
        left.thumbnail((520,700));right.thumbnail((520,700))
        sheet=Image.new('RGB',(left.width*2,left.height+32),(145,145,145));sheet.paste(left,(0,32));sheet.paste(right,(left.width,32))
        d=ImageDraw.Draw(sheet);d.text((5,7),'SOURCE',fill='black');d.text((left.width+5,7),'CANVAS ALIGNED - UNREVIEWED',fill='black')
        review=out/(node['node_id']+'-comparison.jpg');sheet.save(review,quality=95);entry['comparison']=str(review)
        results.append(entry)
    report={'version':1,'source_asset_sha256':receipt['source_sha256'],'input_sha256':sha(working_path),
            'image_size':list(working_size),'canvas_scene_id':receipt['scene_id'],'source_node_id':receipt['source_node_id'],
            'operation':operation,'layers':results,'status':'needs_layer_review',
            'next':'Review important module semantics; incorrect grouping returns to Canvas, never becomes a per-character code patch.'}
    if lineage:report['lineage']=lineage;report['root_source_asset_sha256']=lineage['root_source_asset_sha256']
    (out/'import-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    return report


class AttachSetupError(OSError):
    def __init__(self,details):
        self.details=details
        super().__init__('Canvas回接初始化失败，禁止消费输出；隔离未完成；失败状态已落盘='
                         +str(details['persisted'])+'：'+details['message'])


def _attach_setup_error(failed,exc):
    # Without a lock we cannot move or publish an analysis. Report that boundary
    # explicitly, even when the filesystem also refuses the failure marker.
    details={'ok':False,'phase':'attach_setup','analysis':None,'isolation_complete':False,
             'error':type(exc).__name__,'message':str(exc),'persisted':True}
    try:
        failed.write_text(json.dumps(details,ensure_ascii=False))
    except OSError as report_error:
        details.update(persisted=False,status_error=str(report_error))
    return AttachSetupError(details)


def attach_completion(analysis_path,report_path,node_id,front,back,out_path,*,domain_from_layer=False):
    # Remove the consumable name before validation. A failed retry must not
    # leave an older analysis available to a caller that only reads --out.
    out=Path(out_path).absolute()
    if out.is_symlink() or out.resolve() in {Path(analysis_path).resolve(),Path(report_path).resolve()}:
        raise ValueError('Canvas回接输出必须是独立文件，不能覆盖输入或符号链接。')
    pending=out.with_name(out.name+'.CANVAS_ATTACH_PENDING')
    failed=out.with_name(out.name+'.CANVAS_ATTACH_FAILED')
    token=uuid.uuid4().hex
    staged=out.with_name(out.name+'.staging-'+token)
    # Exclusive ownership also prevents another attach from replacing files
    # while this call is validating or publishing. Interrupted calls stay pending.
    try:
        out.parent.mkdir(parents=True,exist_ok=True)
    except OSError as exc:
        raise _attach_setup_error(failed,exc) from exc
    try:
        stream=pending.open('x')
    except FileExistsError:
        # Another owner (or an interrupted operation) holds this output.
        # Do not replace its marker or invalidate its active result.
        raise
    except OSError as exc:
        raise _attach_setup_error(failed,exc) from exc
    try:
        with stream:stream.write(token)
    except OSError as exc:
        # Keep our partially written pending marker: no result was published.
        raise _attach_setup_error(failed,exc) from exc
    archived=None
    try:
        if out.exists():
            archived=out.with_name(out.name+'.prev-success-'+token)
            out.replace(archived)
        analysis=_attach_completion(analysis_path,report_path,node_id,front,back,domain_from_layer=domain_from_layer)
        staged.write_text(json.dumps(analysis,ensure_ascii=False,indent=2))
        failed.unlink(missing_ok=True)
        staged.replace(out)
    except Exception as exc:
        failed.write_text(json.dumps({'ok':False,'error':type(exc).__name__,'message':str(exc),
            'analysis':None,'isolation_complete':not out.exists(),
            'archived_analysis':str(archived) if archived and archived.exists() else None},ensure_ascii=False))
        raise
    finally:
        staged.unlink(missing_ok=True)
        pending.unlink(missing_ok=True)
    return {'ok':True,'analysis':str(out),'next':'inspect_module_plan -> apply-analysis -> current binding review -> generate; not visual approval'}


def _attach_completion(analysis_path,report_path,node_id,front,back,*,domain_from_layer=False):
    folder=Path(report_path).resolve().parent
    if any((folder/name).exists() for name in ('CANVAS_IMPORT_PENDING','CANVAS_IMPORT_FAILED')):
        raise ValueError('Canvas交接未完成或失败，不能消费旧结果。')
    analysis=json.loads(Path(analysis_path).read_text());report=json.loads(Path(report_path).read_text())
    if analysis['input_sha256']!=report['input_sha256'] or analysis['image_size']!=report['image_size']:
        raise ValueError('Canvas交接不属于当前输入，不能跨任务借用。')
    layer=next((l for l in report['layers'] if l['node_id']==node_id),None)
    if not layer or layer['status']!='aligned_pending_review':raise ValueError('Canvas图层尚未可靠对齐。')
    path=Path(layer['image']).resolve();path.relative_to(Path(report_path).resolve().parent)
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=layer['sha256']:raise ValueError('对齐材质已变化。')
    oc=next((o for o in analysis['occlusions'] if o['front']==front and o['back']==back),None)
    if oc is None:raise ValueError('先声明真实遮挡pair与区域，不能按整张图自动乱补。')
    if domain_from_layer:
        import cv2
        with Image.open(path) as im:alpha=np.asarray(im)[...,3]
        contours,_=cv2.findContours((alpha>=128).astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        w,h=analysis['image_size'];polygons=[]
        for contour in contours:
            if cv2.contourArea(contour)<16:continue
            approximation=cv2.approxPolyDP(contour,1.0,True)[:,0,:]
            if len(approximation)<3:continue
            polygons.append([[float(x)/w,float(y)/h]for x,y in approximation])
        if not polygons or len(polygons)>32:raise ValueError('Canvas层轮廓仍碎裂，需要回画布返修。')
        oc['polygons']=polygons;oc.pop('completion_polygons',None)
        oc['notes']='Candidate completion domain follows aligned Canvas layer silhouette; semantic inspection still required.'
    oc.update(completion='provided',provided={'png_base64':base64.b64encode(raw).decode(),'sha256':layer['sha256'],
        'input_sha256':analysis['input_sha256'],'canvas_scene_id':report['canvas_scene_id'],'canvas_node_id':node_id,
        'canvas_task_id':layer['task_id'],'operation':report['operation'],
        'source_asset_sha256':report['source_asset_sha256'],'result_asset_sha256':layer['raw_sha256']})
    oc.pop('derive_band',None);oc.pop('band_depth_px_at_1080',None)
    from src.studio.module_contract import validate_module_contract
    validate_module_contract(analysis,input_sha256=analysis['input_sha256'],image_size=analysis['image_size'])
    return analysis


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    imp=sub.add_parser('import-layers');imp.add_argument('--receipt',required=True);imp.add_argument('--source',required=True)
    imp.add_argument('--working-image',required=True);imp.add_argument('--assets-dir',required=True);imp.add_argument('--out-dir',required=True)
    imp.add_argument('--parent-report',help='Canvas再次拆分某个返回层时，绑定上一级真实交接记录')
    apply=sub.add_parser('attach-completion');apply.add_argument('--analysis',required=True);apply.add_argument('--import-report',required=True)
    apply.add_argument('--node',required=True);apply.add_argument('--front',required=True);apply.add_argument('--back',required=True);apply.add_argument('--out',required=True)
    apply.add_argument('--domain-from-layer',action='store_true',help='以已对齐Canvas层轮廓作为候选补全域，仍需图审；不手描角色特定区域')
    flow=sub.add_parser('workflow',help='连接诊断、返修队列与新输入复查；不直接调用模型')
    flow.add_argument('--work-dir',required=True);flow.add_argument('--evidence-root',required=True)
    flow.add_argument('--action',required=True,choices=['start','inspect','status','record_layer_issue','decide','reserve_submission',
        'record_submission','record_result','apply_input','review_layers','bind'])
    flow.add_argument('--payload',help='当前动作的JSON文件；status可省略')
    a=p.parse_args()
    try:
        if a.command=='workflow':
            from src.studio.canvas_repair_workflow import CanvasRepairWorkflow
            from src.gates.review_binding import resolve_evidence_path
            payload_path=resolve_evidence_path(Path(a.payload).absolute(),root=Path(a.evidence_root).resolve()) if a.payload else None
            payload=json.loads(payload_path.read_text()) if payload_path else {}
            out=CanvasRepairWorkflow(a.work_dir,evidence_root=a.evidence_root).act(a.action,payload)
        elif a.command=='import-layers':out=import_layers(a.receipt,a.source,a.working_image,a.assets_dir,a.out_dir,parent_report=a.parent_report)
        else:out=attach_completion(a.analysis,a.import_report,a.node,a.front,a.back,a.out,domain_from_layer=a.domain_from_layer)
        print(json.dumps(out,ensure_ascii=False,indent=2))
    except Exception as e:
        result={'ok':False,'error':type(e).__name__,'message':str(e),'resubmitted':False}
        if isinstance(e,AttachSetupError):result['details']=e.details
        print(json.dumps(result,ensure_ascii=False));return 1
    return 0


if __name__=='__main__':raise SystemExit(main())
