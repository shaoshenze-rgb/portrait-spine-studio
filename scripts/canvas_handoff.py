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
        from src.pipeline.split_parts import _components
        labels = _components(np.pad(u8 > 0, 1))[1:-1, 1:-1]
        counts = np.bincount(labels.ravel())
        return int(counts[1:].max()) if len(counts) > 1 else 0


def assess_canvas_split_quality(results, aligned_layers, source, *, is_subsplit=False):
    """Balanced intake quality gate for Canvas split-layers outputs.

    Calibrated on real Mivo Canvas 2K splits vs 4K portraits to avoid both
    false rejects (e.g., translucent glow SIFT misses, valid different-RGB
    occluded coat underlap, 1-3 ignorable tiny sparks) and false passes
    (e.g., localized AI face repaint, dropped weapon/limb, ghost clones).
    """
    total_layers = len(results)
    aligned_count = len(aligned_layers)
    failed_count = total_layers - aligned_count
    reg_fail_ratio = float(failed_count / max(1, total_layers))

    src_s = _downsample_rgba(source)
    orig_fg = src_s[..., 3] >= 180
    fg_area = int(orig_fg.sum())
    layers_s = [(nid, _downsample_rgba(rgba)) for nid, rgba in aligned_layers]

    reasons = []
    warnings = []

    usable_thresh = max(64, int(0.005 * max(1, fg_area)))
    sliver_thresh = max(32, int(0.002 * max(1, fg_area)))
    usable_layer_count = sum(1 for _, rgba in layers_s if int((rgba[..., 3] >= 120).sum()) >= usable_thresh)
    tiny_sliver_count = sum(1 for _, rgba in layers_s if int((rgba[..., 3] >= 120).sum()) < sliver_thresh)
    tiny_sliver_ratio = float(tiny_sliver_count / max(1, aligned_count))

    if aligned_count < 2 or usable_layer_count < 2:
        reasons.append('insufficient_aligned_layers')

    uncovered_ratio = 0.0
    max_missing_hole_ratio = 0.0
    min_rgb_mae = 0.0
    high_rgb_diff_ratio = 0.0
    max_drift_patch_ratio = 0.0
    if not is_subsplit and fg_area > 0:
        union_cov = np.zeros(orig_fg.shape, dtype=bool)
        best_diff = np.full(orig_fg.shape, np.inf, dtype=np.float32)
        src_rgb = src_s[..., :3].astype(np.int16)
        for _, rgba in layers_s:
            union_cov |= rgba[..., 3] >= 80
            solid = orig_fg & (rgba[..., 3] >= 160)
            if solid.any():
                diff = np.abs(rgba[..., :3].astype(np.int16) - src_rgb).mean(axis=2)
                best_diff = np.minimum(best_diff, np.where(solid, diff, np.inf))
        missing_mask = orig_fg & ~union_cov
        uncovered_ratio = float(missing_mask.sum() / fg_area)
        max_missing_hole_ratio = float(_max_connected_area(missing_mask, open_px=1) / fg_area)
        if uncovered_ratio > 0.12 or max_missing_hole_ratio > 0.035:
            reasons.append('missing_original_coverage')
        elif uncovered_ratio > 0.06:
            warnings.append('minor_coverage_gap')

        covered_solid = orig_fg & np.isfinite(best_diff)
        if covered_solid.any():
            vals = best_diff[covered_solid]
            min_rgb_mae = float(vals.mean())
            high_rgb_diff_ratio = float((vals > 40.0).mean())
            drift_mask = covered_solid & (best_diff > 45.0)
            max_drift_patch_ratio = float(_max_connected_area(drift_mask, open_px=1) / fg_area)
        elif aligned_count > 0:
            min_rgb_mae = 255.0
            high_rgb_diff_ratio = 1.0
            max_drift_patch_ratio = 1.0
        if aligned_count > 0 and (min_rgb_mae > 18.0 or high_rgb_diff_ratio > 0.10 or max_drift_patch_ratio > 0.03):
            reasons.append('original_rgb_fidelity_drift')

    if total_layers > 0:
        if reg_fail_ratio > 0.50 or (reg_fail_ratio > 0.35 and (is_subsplit or uncovered_ratio > 0.06)):
            reasons.append('high_registration_failure_rate')
        elif reg_fail_ratio > 0.25:
            warnings.append('partial_registration_warning')

    duplicate_pairs = []
    max_pair_iou = 0.0
    masks = []
    for nid, rgba in layers_s:
        m = rgba[..., 3] >= 160
        if not is_subsplit and fg_area > 0:
            m = m & orig_fg
        if int(m.sum()) >= usable_thresh:
            masks.append((nid, m, rgba[..., :3].astype(np.int16)))
    for i in range(len(masks)):
        id_a, m_a, rgb_a = masks[i]
        for j in range(i + 1, len(masks)):
            id_b, m_b, rgb_b = masks[j]
            inter_mask = m_a & m_b
            inter = int(inter_mask.sum())
            if not inter:
                continue
            union = int((m_a | m_b).sum())
            iou = float(inter / max(1, union))
            if iou > max_pair_iou:
                max_pair_iou = iou
            if iou >= 0.72:
                odiff = np.abs(rgb_a - rgb_b).mean(axis=2)[inter_mask]
                overlap_mae = float(odiff.mean())
                same_ratio = float((odiff <= 20.0).mean())
                # Genuine different-color occlusion completion (e.g., dark coat behind bright torso)
                # is allowed up to IoU < 0.88; near-identical clones or same-RGB ghosts are rejected.
                if iou >= 0.88 or overlap_mae <= 22.0 or same_ratio >= 0.65:
                    duplicate_pairs.append([id_a, id_b, round(iou, 3), round(overlap_mae, 1)])
    if duplicate_pairs:
        reasons.append('duplicate_clone_layers')

    if aligned_count > 24:
        reasons.append('excessive_layer_fragmentation')
    if tiny_sliver_count >= 5 and tiny_sliver_ratio > 0.60:
        reasons.append('excessive_tiny_slivers')
    elif tiny_sliver_count >= 2:
        warnings.append('minor_tiny_slivers')

    passed = len(reasons) == 0
    return {
        'passed': passed,
        'verdict': 'pass' if passed else 'reject_fallback_local',
        'reasons': reasons,
        'warnings': warnings,
        'duplicate_pairs': duplicate_pairs,
        'metrics': {
            'total_layers': total_layers,
            'aligned_count': aligned_count,
            'usable_layer_count': usable_layer_count,
            'failed_registration_count': failed_count,
            'registration_fail_ratio': round(reg_fail_ratio, 4),
            'uncovered_original_ratio': round(uncovered_ratio, 4),
            'max_missing_hole_ratio': round(max_missing_hole_ratio, 4),
            'min_rgb_mae': round(min_rgb_mae, 2),
            'high_rgb_diff_ratio': round(high_rgb_diff_ratio, 4),
            'max_drift_patch_ratio': round(max_drift_patch_ratio, 4),
            'max_pair_iou': round(max_pair_iou, 4),
            'tiny_sliver_count': tiny_sliver_count,
            'tiny_sliver_ratio': round(tiny_sliver_ratio, 4),
        },
    }


def _render_split_composite_comparison(aligned_layers, source, out_path):
    src_s = _downsample_rgba(source, max_edge=768)
    h, w = src_s.shape[:2]
    layers_s = [_downsample_rgba(rgba, max_edge=768) for _, rgba in aligned_layers]

    def _compose(seq):
        canvas = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        for arr in seq:
            canvas.alpha_composite(Image.fromarray(arr))
        return np.asarray(canvas)

    fwd = _compose(layers_s)
    rev = _compose(list(reversed(layers_s))) if len(layers_s) > 1 else fwd
    fg = src_s[..., 3] >= 180
    if fg.any():
        err_fwd = float(np.abs(fwd[..., :3].astype(np.int16) - src_s[..., :3].astype(np.int16))[fg].mean())
        err_rev = float(np.abs(rev[..., :3].astype(np.int16) - src_s[..., :3].astype(np.int16))[fg].mean())
        comp = fwd if err_fwd <= err_rev else rev
    else:
        comp = fwd

    diff_rgb = np.clip(np.abs(comp[..., :3].astype(np.int16) - src_s[..., :3].astype(np.int16)) * 4, 0, 255).astype(np.uint8)
    missing = fg & (comp[..., 3] < 80)
    diff_rgb[missing] = [255, 30, 30]
    diff_rgb[~fg & (comp[..., 3] < 20)] = [145, 145, 145]

    p1, p2, p3 = gray(src_s), gray(comp), Image.fromarray(diff_rgb)
    for im in (p1, p2, p3):
        im.thumbnail((520, 700))
    pw, ph = p1.size
    sheet = Image.new('RGB', (pw * 3, ph + 32), (145, 145, 145))
    sheet.paste(p1, (0, 32)); sheet.paste(p2, (pw, 32)); sheet.paste(p3, (pw * 2, 32))
    d = ImageDraw.Draw(sheet)
    d.text((6, 7), 'ORIGINAL', fill='black')
    d.text((pw + 6, 7), 'CANVAS COMPOSITE', fill='black')
    d.text((pw * 2 + 6, 7), 'DIFF x4 (RED=MISSING)', fill='black')
    sheet.save(out_path, quality=92)
    return out_path


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
    aligned_for_gate=[]
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
        aligned_for_gate.append((node['node_id'],cleaned))
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
    if operation=='split-layers':
        qg=assess_canvas_split_quality(results,aligned_for_gate,source,is_subsplit=bool(parent_report))
        report['quality_gate']=qg
        if not parent_report and aligned_for_gate:
            comp_path=_render_split_composite_comparison(aligned_for_gate,source,out/'canvas-composite-comparison.jpg')
            report['composite_comparison']=str(comp_path)
        if not qg['passed']:
            report['next']=('Canvas split quality gate REJECTED ('+','.join(qg['reasons'])+
                            '); do NOT adopt messy layers — retry Canvas once or fall back to local split.')
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



def apply_edit_patch(
    base_layer_path,
    original_path,
    edit_image_path,
    out_path,
    *,
    front_layer_path=None,
    occluded_mask_path=None,
    alpha_mode='auto',
    crop_bbox=None,
    max_underlap_px=0,
    feather_px=4,
    color_match=True,
    check_out=None,
    report_out=None,
    scene_id=None,
    node_id=None,
):
    """Apply an opaque or RGBA Canvas canvas_edit_image backing inpaint strictly behind the front part."""
    from src.pipeline.canvas_edit_patch import compose_canvas_edit_patch

    if not front_layer_path and not occluded_mask_path:
        raise ValueError('Must pass either --front-layer or --occluded-mask to bound the occluded patch.')
    base = np.asarray(Image.open(base_layer_path).convert('RGBA'), dtype=np.uint8)
    orig = np.asarray(Image.open(original_path).convert('RGBA'), dtype=np.uint8)
    edit_sha = hashlib.sha256(Path(edit_image_path).read_bytes()).hexdigest()

    if occluded_mask_path:
        occ_mask = np.asarray(Image.open(occluded_mask_path).convert('L'), dtype=np.uint8) >= 16
    else:
        occ_mask = np.asarray(Image.open(front_layer_path).convert('RGBA'), dtype=np.uint8)[..., 3] >= 16

    bbox = None
    if crop_bbox:
        parts = [int(v.strip()) for v in str(crop_bbox).split(',')]
        if len(parts) != 4:
            raise ValueError('--crop-bbox must be x,y,w,h')
        bbox = tuple(parts)

    res = compose_canvas_edit_patch(
        base, orig, occ_mask, edit_image_path,
        crop_bbox=bbox, alpha_mode=alpha_mode,
        max_underlap_px=int(max_underlap_px), feather_px=int(feather_px),
        color_match=bool(color_match),
    )

    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(res.rgba).save(out_p)

    check_path = None
    if check_out:
        cp = Path(check_out)
        cp.parent.mkdir(parents=True, exist_ok=True)
        res.comparison.save(cp, quality=92)
        check_path = str(cp)

    report = {
        'ok': True,
        'out': str(out_p),
        'out_sha256': sha(out_p),
        'check_image': check_path,
        'provenance': {
            'operation': 'edit-image',
            'canvas_scene_id': scene_id,
            'canvas_node_id': node_id,
            'result_asset_sha256': edit_sha,
            'patch_method': res.stats['method'],
            'patched_px': res.stats['patched_px'],
            'rest_pose_preserved': res.stats['rest_pose_preserved'],
        },
        'stats': res.stats,
    }
    if report_out:
        rp = Path(report_out)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report

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
    ep=sub.add_parser('apply-edit-patch',help='将Canvas不透明/带棋盘格或RGBA定点补图对齐后，仅贴入前景遮挡区，保留4K原画可见区100%无损')
    ep.add_argument('--base-layer',required=True);ep.add_argument('--original',required=True);ep.add_argument('--edit-image',required=True)
    ep.add_argument('--front-layer',help='前景遮挡层PNG（取alpha>=16作为补底上限掩码）')
    ep.add_argument('--occluded-mask',help='显式单通道遮挡掩码PNG（优先于--front-layer）')
    ep.add_argument('--out',required=True);ep.add_argument('--check-out');ep.add_argument('--report-out')
    ep.add_argument('--alpha-mode',default='auto',choices=['auto','inpaint_alpha','clip_original'])
    ep.add_argument('--crop-bbox',help='局部裁图回传时的x,y,w,h坐标（可选）')
    ep.add_argument('--max-underlap-px',type=int,default=0)
    ep.add_argument('--feather-px',type=int,default=4)
    ep.add_argument('--no-color-match',action='store_true')
    ep.add_argument('--scene-id');ep.add_argument('--node-id')
    a=p.parse_args()
    try:
        if a.command=='workflow':
            from src.studio.canvas_repair_workflow import CanvasRepairWorkflow
            from src.gates.review_binding import resolve_evidence_path
            payload_path=resolve_evidence_path(Path(a.payload).absolute(),root=Path(a.evidence_root).resolve()) if a.payload else None
            payload=json.loads(payload_path.read_text()) if payload_path else {}
            out=CanvasRepairWorkflow(a.work_dir,evidence_root=a.evidence_root).act(a.action,payload)
        elif a.command=='import-layers':out=import_layers(a.receipt,a.source,a.working_image,a.assets_dir,a.out_dir,parent_report=a.parent_report)
        elif a.command=='apply-edit-patch':
            out=apply_edit_patch(
                a.base_layer,a.original,a.edit_image,a.out,
                front_layer_path=a.front_layer,occluded_mask_path=a.occluded_mask,
                alpha_mode=a.alpha_mode,crop_bbox=a.crop_bbox,
                max_underlap_px=a.max_underlap_px,feather_px=a.feather_px,
                color_match=not a.no_color_match,check_out=a.check_out,
                report_out=a.report_out,scene_id=a.scene_id,node_id=a.node_id,
            )
        else:out=attach_completion(a.analysis,a.import_report,a.node,a.front,a.back,a.out,domain_from_layer=a.domain_from_layer)
        print(json.dumps(out,ensure_ascii=False,indent=2))
    except Exception as e:
        result={'ok':False,'error':type(e).__name__,'message':str(e),'resubmitted':False}
        if isinstance(e,AttachSetupError):result['details']=e.details
        print(json.dumps(result,ensure_ascii=False));return 1
    return 0


if __name__=='__main__':raise SystemExit(main())
