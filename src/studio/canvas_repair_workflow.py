"""Durable conversation/CLI repair orchestration. Never invokes a model or bills."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from src.gates.review_binding import resolve_evidence_path
from src.pipeline.canvas_layer_parts import materialize_canvas_layer_parts
from src.pipeline.canvas_layer_rig import compile_canvas_layer_bindings
from src.pipeline.build_rig import build_rig
from .connection_diagnostics import inspect_canvas_connections


class WorkflowInputChanged(ValueError):
    pass


def _unresolved_submission(request):
    if request['status'] in ('submission_reserved','submitted','blocked_result'):return True
    if request['status']=='stale_result_not_applied':
        task=request.get('task_id')
        return request.get('result_available') is not True or not isinstance(task,str) or not task.strip()
    return False


class CanvasRepairWorkflow:
    """One task, immutable input revisions and explicit side-effect reservations.

    Source/layer observations and tool receipts are recorded declarations, not
    Host authorization. The Cindy Agent must still use its actual authorized
    Canvas tools. A reservation is not a remote submission or success receipt.
    """
    def __init__(self, work_dir, *, evidence_root):
        self.root=Path(evidence_root).resolve();self.work=Path(work_dir).absolute()
        self.work.resolve().relative_to(self.root)
        self.path=self.work/'workflow.json'

    @contextmanager
    def _locked(self):
        self.work.mkdir(parents=True,exist_ok=True);lock=self.work/'workflow.lock'
        with lock.open('x') as f:f.write('single writer')
        try:yield
        finally:lock.unlink(missing_ok=True)

    def _read(self,path):
        checked=resolve_evidence_path(path,root=self.root,base=self.root)
        return json.loads(checked.read_text())

    def _save(self,state):
        temporary=self.work/('workflow-'+uuid.uuid4().hex+'.pending')
        temporary.write_text(json.dumps(state,ensure_ascii=False,indent=2));temporary.replace(self.path)

    def _diagnose(self,parts,plan):
        folder=self.work/'inspections'/uuid.uuid4().hex
        report=inspect_canvas_connections(parts,plan,evidence_root=self.root,out_dir=folder)
        return report,str(folder/'report.json')

    def _fresh(self,state,expected):
        if expected!=state['input_revision']:raise ValueError('Stale workflow revision')
        try:current=inspect_canvas_connections(state['parts_dir'],self._read(state['plan_path']),evidence_root=self.root)
        except (ValueError,OSError) as exc:raise WorkflowInputChanged('Current inputs no longer validate; preserve results and inspect a corrected revision') from exc
        if current['input_revision']!=expected:raise WorkflowInputChanged('Inputs changed; inspect a new revision before repair/binding')
        return current

    def _requests(self,report):
        groups={}
        for defect in report['defects']:
            if defect['requires_review']:groups.setdefault(defect['connection_id'],[]).append(defect)
        return [{'id':hashlib.sha256((report['input_revision']+cid).encode()).hexdigest()[:20],
                 'connection_id':cid,'input_revision':report['input_revision'],'status':'needs_planning',
                 'defect_ids':[d['id'] for d in rows],'codes':[d['code'] for d in rows],
                 'evidence':rows[0]['evidence'],'notes':'先核对原图结构，再选择重规划或Canvas局部材质返修。',
                 'attempts':0} for cid,rows in groups.items()]

    def _material_fingerprint(self,parts_dir,mid):
        material=self._read(Path(parts_dir)/'parts.json')
        rows=sorted(p['canvasLayer']['sha256'] for p in material['parts'] if p['moduleId']==mid)
        if not rows:raise ValueError('Unknown material module')
        return hashlib.sha256(json.dumps(rows,separators=(',',':')).encode()).hexdigest()

    def _carry_material_requests(self,state,parts,revision):
        material=self._read(Path(parts)/'parts.json');mids={p['moduleId'] for p in material['parts']};carried=[]
        for old in state.get('repairs',[]):
            if (old.get('module_id') in mids and old['codes']==['layer_visual_defect']
                    and old['status'] in ('needs_planning','queued_for_canvas','queued_for_replan')
                    and self._material_fingerprint(parts,old['module_id'])==old.get('material_fingerprint')):
                kept=copy.deepcopy(old);kept['carried_from_revision']=state['input_revision']
                kept['input_revision']=revision;carried.append(kept)
        return carried

    def _readiness(self,payload):
        path=resolve_evidence_path(payload.get('readiness_receipt_path'),root=self.root)
        receipt=self._read(path)
        if receipt.get('isError'):raise ValueError('Canvas readiness failed')
        if 'content' in receipt:
            texts=[c['text'] for c in receipt['content'] if c.get('type')=='text']
            if len(texts)!=1:raise ValueError('Expected one native readiness receipt')
            receipt=json.loads(texts[0])
        result=receipt.get('result') or {};handshake=(result.get('libraryDiagnostics') or {}).get('handshake') or {}
        attempts=handshake.get('attempts') or [];last=attempts[-1] if attempts else {}
        observed=last.get('at',handshake.get('observedAt'));node=payload.get('source_node_id')
        state=(result.get('mediaStates') or {}).get(node) or {}
        ref=state.get('imageRef','');match=re.fullmatch(r'library:assets/([0-9a-f]{2})/([0-9a-f]{64})/blob.[a-zA-Z0-9]+',ref)
        if (receipt.get('ok') is not True or last.get('authorizedReadonly') is not True
                or type(observed) not in (int,float) or not math.isfinite(observed)
                or not -5000<=time.time()*1000-observed<=60000
                or not isinstance(node,str) or not node or state.get('state')!='available'
                or not match or match.group(1)!=match.group(2)[:2]):
            raise ValueError('Fresh authorized Canvas media-read observation is required; nothing reserved')
        return {'source_node_id':node,'readiness_receipt':str(path),
                'readiness_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'observed_at':observed,
                'kind':'recorded_observation_not_host_authorization'}

    def status(self):
        state=self._read(self.path)
        try:self._fresh(state,state['input_revision']);state['inputs_current']=True
        except (ValueError,OSError):state['inputs_current']=False;state['phase']='input_changed'
        return state

    def act(self,action,payload):
        if not isinstance(payload,dict):raise ValueError('Workflow payload must be an object')
        if action=='status':return self.status()
        with self._locked():
            if action in ('start','inspect'):
                if action=='start' and self.path.exists():raise ValueError('Workflow already exists; use status or apply_input')
                if self.path.exists():
                    state=self._read(self.path)
                    if any(_unresolved_submission(q) for q in state['repairs']):
                        raise ValueError('Preserve unresolved submissions; do not replace their input revision')
                else:state={'version':1,'workflow_id':uuid.uuid4().hex[:16],'history':[]}
                if action=='start':
                    material=self._read(payload['material_manifest_path'])
                    parts=self.work/'inputs'/uuid.uuid4().hex/'parts'
                    materialize_canvas_layer_parts(material,parts,evidence_root=self.root)
                else:parts=resolve_evidence_path(Path(payload['parts_dir'])/'parts.json',root=self.root).parent
                plan_path=resolve_evidence_path(payload['plan_path'],root=self.root)
                plan=self._read(plan_path)
                current=inspect_canvas_connections(parts,plan,evidence_root=self.root)
                if state.get('original_source_sha256') and current['source_sha256']!=state['original_source_sha256']:
                    raise ValueError('Use a new workflow for a different portrait')
                if not set(state.get('required_modules',[])).issubset(m['id'] for m in plan['modules']):
                    raise ValueError('Cannot remove unresolved modules to complete the character')
                if state.get('input_revision')==current['input_revision']:
                    return state  # Re-observation does not discard decisions or duplicate requests.
                report,report_path=self._diagnose(parts,plan)
                carried=self._carry_material_requests(state,parts,report['input_revision'])
                if state.get('input_revision'):state['history'].append({k:state.get(k) for k in ('input_revision','repairs','binding_review','binding')})
                state.update(parts_dir=str(parts),plan_path=str(plan_path),input_revision=report['input_revision'],
                    original_source_sha256=report['source_sha256'],required_modules=sorted(m['id'] for m in plan['modules']),
                    diagnostics=report_path,repairs=self._requests(report)+carried,binding_review=None,binding=None,
                    phase='needs_connection_review' if report['status']=='needs_connection_review' else 'awaiting_layer_review')
                if carried:state['phase']='needs_layer_repair'
                self._save(state);return state
            state=self._read(self.path);inputs_current=True
            try:report=self._fresh(state,payload.get('expected_revision'))
            except WorkflowInputChanged:
                if action not in ('record_submission','record_result'):raise
                inputs_current=False;report=None
            requests={q['id']:q for q in state['repairs']}
            if action=='record_layer_issue':
                mid=payload.get('module_id');notes=payload.get('notes')
                if (mid not in state['required_modules'] or payload.get('source_and_affected_layer_viewed') is not True
                        or not isinstance(notes,str) or not notes.strip()):
                    raise ValueError('Current source and affected layer observation required')
                existing=next((q for q in state['repairs'] if q.get('module_id')==mid and q['codes']==['layer_visual_defect']),None)
                rid=existing['id'] if existing else hashlib.sha256((state['input_revision']+'material:'+mid).encode()).hexdigest()[:20]
                if rid not in requests:
                    state['repairs'].append({'id':rid,'module_id':mid,'connection_id':None,
                        'input_revision':state['input_revision'],'status':'needs_planning','codes':['layer_visual_defect'],
                        'defect_ids':[],'attempts':0,'evidence':{'notes':notes,'source_sha256':state['original_source_sha256']},
                        'material_fingerprint':self._material_fingerprint(state['parts_dir'],mid)})
                elif existing:
                    existing['evidence']['latest_observation']=notes
                    if existing['status']=='acknowledged_for_binding_checks':existing['status']='needs_planning'
                state['phase']='needs_layer_repair';state['binding_review']=None;state['binding']=None
            elif action=='decide':
                q=requests[payload['request_id']];decision=payload.get('decision');notes=payload.get('notes')
                if not isinstance(notes,str) or not notes.strip():raise ValueError('Record the source-based structural decision')
                if q['status'] not in ('needs_planning','queued_for_canvas','queued_for_replan'):
                    raise ValueError('A dispatched/result request cannot be silently replaced')
                if decision in ('keep_anchor','keep_current'):
                    if set(q['codes'])-{'fringe_anchor','layer_visual_defect'}:raise ValueError('Structural/data defects cannot be acknowledged away')
                    if payload.get('original_and_layers_viewed') is not True:raise ValueError('Actual source/layer observation required')
                    q['status']='acknowledged_for_binding_checks'
                elif decision in ('replan','canvas_material'):
                    if decision=='canvas_material' and (payload.get('original_and_layers_viewed') is not True
                            or payload.get('diagnosis') not in ('missing_material','visible_material_mismatch')
                            or set(q['codes']) & {'binding_validation','material_plan_mismatch'}):
                        raise ValueError('Canvas repair requires a source-based missing-material diagnosis')
                    q['status']='queued_for_replan' if decision=='replan' else 'queued_for_canvas'
                else:raise ValueError('Unknown repair decision')
                q.update(decision=decision,decision_notes=notes)
            elif action=='reserve_submission':
                q=requests[payload['request_id']]
                if q['status']!='queued_for_canvas' or q['attempts']:
                    raise ValueError('No automatic repeated Canvas submission')
                readiness=self._readiness(payload)
                q.update(status='submission_reserved',attempts=1,source_node_id=readiness['source_node_id'],
                         dispatch_key=uuid.uuid4().hex,readiness_observation=copy.deepcopy(readiness))
            elif action=='record_submission':
                q=requests[payload['request_id']]
                if q['status'] not in ('submission_reserved','submitted','blocked_result','stale_result_not_applied'):
                    raise ValueError('Reserve before submitting')
                task=payload.get('task_id')
                if not isinstance(task,str) or not task.strip():raise ValueError('Actual service task identity required')
                bound_task=q.get('task_id')
                if q['status']!='submission_reserved' and (not isinstance(bound_task,str) or not bound_task.strip()):
                    raise ValueError('Unknown/stale submission has no bound task identity; cannot assign a replacement task')
                receipt=resolve_evidence_path(payload['receipt_path'],root=self.root)
                if q.get('task_id') and q['task_id']!=task:raise ValueError('Request already has a different submitted task')
                next_status='submitted' if q['status']=='submission_reserved' else q['status']
                q.update(status=next_status,task_id=task,receipt_path=str(receipt),
                         receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest())
            elif action=='record_result':
                q=requests[payload['request_id']]
                bound_task=q.get('task_id')
                if (q['status'] not in ('submitted','blocked_result','stale_result_not_applied')
                        or not isinstance(bound_task,str) or not bound_task.strip()):
                    raise ValueError('Bind the service task with record_submission before any result')
                if payload.get('task_id')!=q['task_id']:raise ValueError('Result task mismatch')
                was_stale=q['status']=='stale_result_not_applied'
                if was_stale and q.get('result_available') is True and payload.get('result_available') is not True:
                    raise ValueError('A known stale result cannot be overwritten by an unknown observation')
                receipt=resolve_evidence_path(payload['receipt_path'],root=self.root)
                # Raw failed/unknown receipts remain recoverable; this never resets the attempt.
                q.update(result_receipt=str(receipt),result_receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),
                         result_input_current=inputs_current and not was_stale)
                q['result_available']=payload.get('result_available') is True
                if payload.get('result_available') is not True:q['status']='blocked_result'
                else:
                    q['status']='awaiting_layer_review'
                    if not inputs_current or was_stale:q['status']='stale_result_not_applied'
            elif action=='apply_input':
                if any(_unresolved_submission(q) for q in state['repairs']):
                    raise ValueError('Unresolved submitted work must be recorded before replacing inputs')
                if not isinstance(payload.get('notes'),str) or not payload['notes'].strip():raise ValueError('Explain the input correction')
                material=self._read(payload['material_manifest_path']);plan=self._read(payload['plan_path'])
                if material.get('source_sha256')!=state['original_source_sha256']:raise ValueError('Cannot replace the portrait mid-workflow')
                mids={p['module_id'] for p in material.get('pieces',[])}
                if not set(state['required_modules']).issubset(mids):raise ValueError('Cannot remove unresolved modules to complete the character')
                if {m['id'] for m in plan.get('modules',[])}!=mids:raise ValueError('Plan and material module sets differ')
                revision_dir=self.work/'inputs'/uuid.uuid4().hex
                parts=revision_dir/'parts';materialize_canvas_layer_parts(material,parts,evidence_root=self.root)
                plan_path=revision_dir/'plan.json';plan_path.write_text(json.dumps(plan,ensure_ascii=False,indent=2))
                new_report,report_path=self._diagnose(parts,plan)
                carried=self._carry_material_requests(state,parts,new_report['input_revision'])
                state['history'].append({k:state.get(k) for k in ('input_revision','repairs','binding_review','binding')})
                state.update(parts_dir=str(parts),plan_path=str(plan_path),input_revision=new_report['input_revision'],
                    diagnostics=report_path,repairs=self._requests(new_report)+carried,binding_review=None,binding=None,
                    required_modules=sorted(mids),
                    phase='needs_connection_review' if new_report['status']=='needs_connection_review' else 'awaiting_layer_review',
                    input_change_notes=payload['notes'])
                if carried:state['phase']='needs_layer_repair'
            elif action=='review_layers':
                if any(q['status']!='acknowledged_for_binding_checks' for q in state['repairs']):
                    raise ValueError('Repair or explicitly review current connection defects first')
                observations=payload.get('modules')
                if (payload.get('original_and_layers_viewed') is not True or not payload.get('reviewer')
                        or not isinstance(observations,dict) or set(observations)!=set(state['required_modules'])):
                    raise ValueError('Current full source/layer review required for every module')
                if any(v.get('status')!='pass' or not isinstance(v.get('notes'),str) or not v['notes'].strip()
                       for v in observations.values()):raise ValueError('Unresolved layer observation')
                state.update(binding_review=copy.deepcopy(payload),phase='ready_to_bind')
            elif action=='bind':
                if state['phase']!='ready_to_bind' or not state.get('binding_review'):raise ValueError('Current layer review required')
                token=state['workflow_id']+'-'+uuid.uuid4().hex[:8]
                target=self.root/('canvas-binding-'+token);rig_dir=self.root/('canvas-rig-'+token)
                try:
                    spec=compile_canvas_layer_bindings(state['parts_dir'],self._read(state['plan_path']),target,evidence_root=self.root)
                    made=build_rig(state['parts_dir'],spec,rig_dir,include_animation=False)
                    state.update(binding={'rig_spec':str(target/'rig-spec.json'),'rig_dir':str(rig_dir),
                                          'report_path':made['report_path']},phase='awaiting_rig_confirm')
                except ValueError as exc:
                    state.update(phase='needs_connection_review',last_error={'code':getattr(exc,'code','INPUT_REVIEW_REQUIRED'),'message':str(exc)})
                    rid=hashlib.sha256((state['input_revision']+str(exc)).encode()).hexdigest()[:20]
                    if rid not in requests:state['repairs'].append({'id':rid,'connection_id':None,
                        'input_revision':state['input_revision'],'status':'needs_planning','codes':['binding_validation'],
                        'defect_ids':[],'evidence':state['last_error'],'attempts':0})
            else:raise ValueError('Unknown Canvas workflow action')
            if action=='decide' and all(q['status']=='acknowledged_for_binding_checks' for q in state['repairs']):
                state['phase']='awaiting_layer_review'
            if not inputs_current:state['phase']='input_changed'
            self._save(state);return state
