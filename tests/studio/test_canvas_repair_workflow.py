import copy
import hashlib
import json
import os
import subprocess
import sys
import time
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import tests.pipeline.test_canvas_layer_rig as fixtures
from src.pipeline.canvas_layer_parts import materialize_canvas_layer_parts
from src.studio.connection_diagnostics import inspect_canvas_connections
from src.studio.canvas_repair_workflow import CanvasRepairWorkflow, WorkflowInputChanged


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


class CanvasRepairWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.fx=fixtures.CanvasLayerRigTests('test_typed_rigid_soft_binding_real_renderer_and_export_block')
        self.fx.setUp();self.addCleanup(self.fx.doCleanups)
        self.root=self.fx.root;self.parts=self.fx.parts;self.plan=copy.deepcopy(self.fx.plan)
        self.plan_path=self.root/'plan.json';self.plan_path.write_text(json.dumps(self.plan))
        self.wf=CanvasRepairWorkflow(self.root/'workflow',evidence_root=self.root);self.n=0

    def material(self,kind=None):
        old=json.loads((self.fx.parts/'parts.json').read_text());self.n+=1
        m={'version':1,'material_mode':'canvas_layers','source_image':old['originalSource'],
           'source_sha256':old['originalSourceSHA256'],'image_size':old['canvas'],'pieces':[]}
        for p in old['parts']:
            l=copy.deepcopy(p['canvasLayer']);image=Path(l['image'])
            if (kind=='gap' and p['name']=='body') or (kind in ('fringe','translucent') and p['name']=='head'):
                a=np.asarray(Image.open(image)).copy()
                if kind=='gap':a[:20]=0
                elif kind=='fringe':a[12,32,3]=1
                else:a[a[...,3]>0,3]=32
                image=self.root/(kind+str(self.n)+'.png');Image.fromarray(a).save(image)
                l['image']=str(image);l['sha256']=sha(image);l['provenance']['result_asset_sha256']=sha(image)
            m['pieces'].append({'id':p['name'],'module_id':p['moduleId'],'kind':p['kind'],
                'parent':p['parent'],'pivot':p['pivot'],'draw_order':p['drawOrder'],
                'image':l['image'],'sha256':l['sha256'],'provenance':l['provenance']})
        return m

    def inspect(self,kind=None):
        if kind:
            self.parts=self.root/('parts-'+kind)
            materialize_canvas_layer_parts(self.material(kind),self.parts,evidence_root=self.root)
        return self.wf.act('inspect',{'parts_dir':str(self.parts),'plan_path':str(self.plan_path)})

    def readiness(self,authorized=True,age_seconds=0):
        self.n+=1;path=self.root/('readiness-'+str(self.n)+'.json');at=int(time.time()*1000-age_seconds*1000)
        path.write_text(json.dumps({'ok':True,'result':{'libraryDiagnostics':{'handshake':{
            'observedAt':at,'attempts':[{'at':at,'authorizedReadonly':authorized}]}},
            'mediaStates':{'synthetic':{'state':'available','imageRef':'library:assets/aa/'+'a'*64+'/blob.png'}}}}))
        return {'readiness_receipt_path':str(path),'source_node_id':'synthetic'}

    def test_relative_fringe_warning_does_not_reject_uniform_translucency(self):
        st=self.inspect('fringe');report=json.loads(Path(st['diagnostics']).read_text())
        codes={d['code'] for d in report['defects']};self.assertIn('fringe_anchor',codes)
        self.assertEqual(next(d for d in report['defects'] if d['code']=='fringe_anchor')['evidence']['child_alpha'],1)
        self.parts=self.root/'parts-translucent'
        materialize_canvas_layer_parts(self.material('translucent'),self.parts,evidence_root=self.root)
        d=inspect_canvas_connections(self.parts,self.plan,evidence_root=self.root)
        self.assertNotIn('fringe_anchor',{x['code'] for x in d['defects']})

    def test_gap_is_planning_uncertainty_not_automatic_material_generation(self):
        st=self.inspect('gap');q=st['repairs'][0]
        self.assertIn('parent_support_gap',q['codes']);self.assertEqual(q['status'],'needs_planning')
        self.assertEqual(q['attempts'],0)
        payload={'expected_revision':st['input_revision'],'request_id':q['id'],'decision':'canvas_material','notes':'test'}
        with self.assertRaisesRegex(ValueError,'source-based'):self.wf.act('decide',payload)
        self.assertEqual(self.wf.status()['repairs'][0]['attempts'],0)

    def test_submission_reservation_is_single_use_and_unknown_result_does_not_retry(self):
        st=self.inspect('gap');q=st['repairs'][0];common={'expected_revision':st['input_revision'],'request_id':q['id']}
        self.wf.act('decide',{**common,'decision':'canvas_material','notes':'Synthetic source/layer reviewed: missing material',
                             'original_and_layers_viewed':True,'diagnosis':'missing_material'})
        with self.assertRaises(ValueError):self.wf.act('reserve_submission',{**common,**self.readiness(False)})
        with self.assertRaises(ValueError):self.wf.act('reserve_submission',{**common,**self.readiness(age_seconds=120)})
        reserved=self.wf.act('reserve_submission',{**common,**self.readiness()})
        self.assertEqual(reserved['repairs'][0]['attempts'],1)
        with self.assertRaises(ValueError):self.wf.act('reserve_submission',{**common,**self.readiness()})
        receipt=self.root/'receipt.json';receipt.write_text(json.dumps({'synthetic_only':True,'result':'unknown'}))
        self.wf.act('record_submission',{**common,'task_id':'synthetic-task','receipt_path':str(receipt)})
        blocked=self.wf.act('record_result',{**common,'task_id':'synthetic-task','receipt_path':str(receipt),'result_available':False})
        self.assertEqual(blocked['repairs'][0]['status'],'blocked_result')
        with self.assertRaises(ValueError):self.wf.act('reserve_submission',{**common,**self.readiness()})
        with self.assertRaises(ValueError):self.wf.act('inspect',{'parts_dir':str(self.parts),'plan_path':str(self.plan_path)})

    def test_changed_inputs_keep_old_result_but_never_apply_it(self):
        st=self.inspect('gap');q=st['repairs'][0];common={'expected_revision':st['input_revision'],'request_id':q['id']}
        self.wf.act('decide',{**common,'decision':'canvas_material','notes':'Synthetic missing material',
                             'original_and_layers_viewed':True,'diagnosis':'missing_material'})
        self.wf.act('reserve_submission',{**common,**self.readiness()})
        self.plan['modules'][0]['confidence']=.91;self.plan_path.write_text(json.dumps(self.plan))
        receipt=self.root/'receipt.json';receipt.write_text('{}')
        self.wf.act('record_submission',{**common,'task_id':'synthetic-task','receipt_path':str(receipt)})
        st=self.wf.act('record_result',{**common,'task_id':'synthetic-task','receipt_path':str(receipt),'result_available':True})
        self.assertEqual(st['phase'],'input_changed');self.assertEqual(st['repairs'][0]['status'],'stale_result_not_applied')
        self.assertFalse(st['binding'])
        newer=self.wf.act('inspect',{'parts_dir':str(self.parts),'plan_path':str(self.plan_path)})
        self.assertNotEqual(st['input_revision'],newer['input_revision'])

    def test_repair_then_reinspect_review_and_bind_uses_same_flow(self):
        st=self.inspect('gap');material=self.material();path=self.root/'repaired-material.json';path.write_text(json.dumps(material))
        st2=self.wf.act('apply_input',{'expected_revision':st['input_revision'],'material_manifest_path':str(path),
            'plan_path':str(self.plan_path),'notes':'Synthetic replacement restores parent support; no modules removed'})
        self.assertEqual(st2['phase'],'awaiting_layer_review');self.assertFalse(st2['binding_review'])
        with self.assertRaises(ValueError):self.wf.act('review_layers',{'expected_revision':st['input_revision']})
        reviewed=self.wf.act('review_layers',{'expected_revision':st2['input_revision'],'original_and_layers_viewed':True,
            'reviewer':'synthetic-test-only','modules':{m['id']:{'status':'pass','notes':'Synthetic structure verified'} for m in self.plan['modules']}})
        bound=self.wf.act('bind',{'expected_revision':reviewed['input_revision']})
        self.assertEqual(bound['phase'],'awaiting_rig_confirm',bound.get('last_error'))
        self.assertTrue(Path(bound['binding']['report_path']).is_file())
        before=list((self.root/'workflow/inspections').iterdir())
        same=self.wf.act('inspect',{'parts_dir':bound['parts_dir'],'plan_path':bound['plan_path']})
        self.assertEqual(same['binding'],bound['binding']);self.assertEqual(before,list((self.root/'workflow/inspections').iterdir()))

    def test_partial_character_cannot_be_substituted_to_close_work(self):
        st=self.inspect('gap');material=self.material();material['pieces']=material['pieces'][:-1]
        path=self.root/'partial.json';path.write_text(json.dumps(material))
        with self.assertRaisesRegex(ValueError,'remove unresolved modules'):
            self.wf.act('apply_input',{'expected_revision':st['input_revision'],'material_manifest_path':str(path),
                'plan_path':str(self.plan_path),'notes':'Must not pass'})
        self.assertEqual(self.wf.status()['input_revision'],st['input_revision'])

    def test_plan_cannot_hide_material_modules_from_diagnosis(self):
        plan=copy.deepcopy(self.plan);plan['modules']=plan['modules'][:-1]
        with self.assertRaisesRegex(ValueError,'every material module'):
            inspect_canvas_connections(self.parts,plan,evidence_root=self.root)

    def test_crop_corruption_invalidates_workflow_without_repairing_pixels(self):
        st=self.inspect();p=self.parts/'parts/body.png';a=np.asarray(Image.open(p)).copy();a[0,0,0]^=255;Image.fromarray(a).save(p)
        status=self.wf.status();self.assertFalse(status['inputs_current']);self.assertEqual(status['phase'],'input_changed')
        with self.assertRaises(WorkflowInputChanged):
            self.wf.act('bind',{'expected_revision':st['input_revision']})

    def test_existing_compiler_error_exposes_all_connection_findings(self):
        from src.pipeline.canvas_layer_rig import compile_canvas_layer_bindings
        st=self.inspect('gap')
        with self.assertRaises(ValueError) as caught:
            compile_canvas_layer_bindings(self.parts,self.plan,self.root/'binding-direct',evidence_root=self.root)
        self.assertEqual(caught.exception.code,'CONNECTION_REVIEW_REQUIRED')
        report=caught.exception.details['connection_diagnostics']
        self.assertEqual(len(report['connections']),len(self.plan['connections']))
        self.assertIn('parent_support_gap',{d['code'] for d in report['defects']})

    def test_unchanged_visual_defect_survives_other_input_repair_without_duplication(self):
        st=self.inspect('gap')
        st=self.wf.act('record_layer_issue',{'expected_revision':st['input_revision'],'module_id':'flame',
            'source_and_affected_layer_viewed':True,'notes':'Synthetic visible material mismatch'})
        q=next(q for q in st['repairs'] if q.get('module_id')=='flame')
        self.wf.act('decide',{'expected_revision':st['input_revision'],'request_id':q['id'],'decision':'canvas_material',
            'diagnosis':'visible_material_mismatch','original_and_layers_viewed':True,'notes':'Synthetic source reviewed'})
        path=self.root/'corrected.json';path.write_text(json.dumps(self.material()))
        changed=self.wf.act('apply_input',{'expected_revision':st['input_revision'],'material_manifest_path':str(path),
            'plan_path':str(self.plan_path),'notes':'Restore unrelated parent support'})
        self.assertEqual(changed['phase'],'needs_layer_repair')
        carried=[q for q in changed['repairs'] if q.get('module_id')=='flame'];self.assertEqual(len(carried),1)
        self.assertEqual(carried[0]['attempts'],0);self.assertEqual(carried[0]['status'],'queued_for_canvas')
        repeated=self.wf.act('record_layer_issue',{'expected_revision':changed['input_revision'],'module_id':'flame',
            'source_and_affected_layer_viewed':True,'notes':'Same material, still mismatched'})
        self.assertEqual(len([q for q in repeated['repairs'] if q.get('module_id')=='flame']),1)

    def test_inspect_cannot_reset_required_modules_to_a_partial_character(self):
        st=self.inspect();material=self.material();material['pieces']=material['pieces'][:-1]
        partial=self.root/'partial-parts';materialize_canvas_layer_parts(material,partial,evidence_root=self.root)
        plan=copy.deepcopy(self.plan);plan['modules']=plan['modules'][:-1];plan['connections']=plan['connections'][:-1]
        path=self.root/'partial-plan.json';path.write_text(json.dumps(plan))
        with self.assertRaisesRegex(ValueError,'remove unresolved modules'):
            self.wf.act('inspect',{'parts_dir':str(partial),'plan_path':str(path)})
        self.assertEqual(self.wf.status()['input_revision'],st['input_revision'])

    def test_start_materializes_layers_once_then_enters_review(self):
        path=self.root/'material.json';path.write_text(json.dumps(self.material()))
        st=self.wf.act('start',{'material_manifest_path':str(path),'plan_path':str(self.plan_path)})
        self.assertEqual(st['phase'],'awaiting_layer_review')
        self.assertTrue((Path(st['parts_dir'])/'parts.json').exists())
        with self.assertRaisesRegex(ValueError,'already exists'):
            self.wf.act('start',{'material_manifest_path':str(path),'plan_path':str(self.plan_path)})

    def test_existing_handoff_cli_exposes_same_state_machine(self):
        payload=self.root/'inspect.json';payload.write_text(json.dumps({'parts_dir':str(self.parts),'plan_path':str(self.plan_path)}))
        command=[sys.executable,str(fixtures.ROOT/'scripts/canvas_handoff.py'),'workflow',
                 '--work-dir',str(self.root/'workflow'),'--evidence-root',str(self.root),'--action','inspect','--payload',str(payload)]
        run=subprocess.run(command,cwd=fixtures.ROOT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,text=True)
        self.assertEqual(run.returncode,0,run.stderr+run.stdout)
        result=json.loads(run.stdout);self.assertEqual(result['phase'],'awaiting_layer_review')
        self.assertEqual(result['input_revision'],self.wf.status()['input_revision'])

    def reserve_material_request(self):
        st=self.inspect('gap');q=st['repairs'][0];common={'expected_revision':st['input_revision'],'request_id':q['id']}
        self.wf.act('decide',{**common,'decision':'canvas_material','diagnosis':'missing_material',
            'original_and_layers_viewed':True,'notes':'Synthetic missing-material observation'})
        self.wf.act('reserve_submission',{**common,**self.readiness()})
        receipt=self.root/'receipt.json';receipt.write_text('{"synthetic_only":true}')
        return common,receipt

    def test_results_require_bound_task_and_unknown_cannot_change_task_identity(self):
        common,receipt=self.reserve_material_request();state_path=self.root/'workflow/workflow.json'
        before=state_path.read_bytes()
        for available in (False,True):
            with self.subTest(unbound=available),self.assertRaisesRegex(ValueError,'record_submission'):
                self.wf.act('record_result',{**common,'receipt_path':str(receipt),'result_available':available})
            self.assertEqual(state_path.read_bytes(),before)
        self.wf.act('record_submission',{**common,'task_id':'original-task','receipt_path':str(receipt)})
        blocked=self.wf.act('record_result',{**common,'task_id':'original-task','receipt_path':str(receipt),'result_available':False})
        self.assertEqual(blocked['repairs'][0]['status'],'blocked_result')
        before=state_path.read_bytes()
        for action,task in [('record_submission','new-task-after-unknown'),('record_result','new-task-after-unknown'),('record_result',None)]:
            with self.subTest(action=action,task=task),self.assertRaises(ValueError):
                self.wf.act(action,{**common,'task_id':task,'receipt_path':str(receipt),'result_available':True})
            self.assertEqual(state_path.read_bytes(),before)
        same=self.wf.act('record_submission',{**common,'task_id':'original-task','receipt_path':str(receipt)})
        self.assertEqual(same['repairs'][0]['status'],'blocked_result')
        recovered=self.wf.act('record_result',{**common,'task_id':'original-task','receipt_path':str(receipt),'result_available':True})
        self.assertEqual(recovered['repairs'][0]['status'],'awaiting_layer_review')
        self.assertEqual(recovered['repairs'][0]['attempts'],1)

    def test_legacy_unknown_without_task_id_cannot_adopt_a_new_task(self):
        common,receipt=self.reserve_material_request();path=self.root/'workflow/workflow.json'
        for legacy_status in ('blocked_result','stale_result_not_applied'):
            for invalid_task in (None,'','   ',123):
                state=json.loads(path.read_text());state['repairs'][0]['status']=legacy_status
                state['repairs'][0]['task_id']=invalid_task;path.write_text(json.dumps(state));before=path.read_bytes()
                for action in ('record_submission','record_result'):
                    with self.subTest(status=legacy_status,task=invalid_task,action=action),self.assertRaises(ValueError):
                        self.wf.act(action,{**common,'task_id':'replacement','receipt_path':str(receipt),'result_available':True})
                    self.assertEqual(path.read_bytes(),before)
                with self.assertRaises(ValueError):
                    self.wf.act('inspect',{'parts_dir':str(self.parts),'plan_path':str(self.plan_path)})
                self.assertEqual(self.wf.status()['repairs'][0]['attempts'],1)

    def test_unknown_result_stays_unresolved_even_after_inputs_change(self):
        common,receipt=self.reserve_material_request()
        self.wf.act('record_submission',{**common,'task_id':'original-task','receipt_path':str(receipt)})
        self.plan['modules'][0]['confidence']=.91;self.plan_path.write_text(json.dumps(self.plan))
        blocked=self.wf.act('record_result',{**common,'task_id':'original-task','receipt_path':str(receipt),'result_available':False})
        self.assertEqual(blocked['repairs'][0]['status'],'blocked_result');self.assertFalse(blocked['repairs'][0]['result_input_current'])
        with self.assertRaises(ValueError):self.wf.act('inspect',{'parts_dir':str(self.parts),'plan_path':str(self.plan_path)})
        stale=self.wf.act('record_result',{**common,'task_id':'original-task','receipt_path':str(receipt),'result_available':True})
        self.assertEqual(stale['repairs'][0]['status'],'stale_result_not_applied')
        with self.assertRaises(ValueError):
            self.wf.act('record_result',{**common,'task_id':'original-task','receipt_path':str(receipt),'result_available':False})
        with self.assertRaises(ValueError):
            self.wf.act('record_submission',{**common,'task_id':'replacement','receipt_path':str(receipt)})
        same=self.wf.act('record_submission',{**common,'task_id':'original-task','receipt_path':str(receipt)})
        self.assertEqual(same['repairs'][0]['status'],'stale_result_not_applied');self.assertEqual(same['repairs'][0]['attempts'],1)


if __name__=='__main__':unittest.main()
