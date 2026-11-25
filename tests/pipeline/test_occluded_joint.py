"""Hidden geometry is optional only when named material really covers the joint."""
import base64
import copy
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.studio.module_contract import materialize_module_contract, validate_module_contract
from src.studio.vision_schema import VisionReviewRequired, VisionValidationError
from src.studio.auto_spec import build_default_specs
from src.pipeline.split_parts import split_parts
from src.pipeline.build_rig import build_rig
from src.render.renderer import SpineRenderer
from src.gates.module_quality import ModuleQualityAuditor

ROOT=Path(__file__).resolve().parents[2]


class HiddenJointTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=ROOT/'.studio-work',prefix='hidden-joint-')
        self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.image=self.root/'source.png';self.w,self.h=64,64
        rgba=np.zeros((64,64,4),np.uint8)
        masks={}
        for mid,rect,color in [('body',(10,20,31,52),(80,120,160,255)),
                               ('head',(16,3,26,12),(180,140,100,255)),
                               ('cloth',(10,12,31,20),(100,40,140,255))]:
            x0,y0,x1,y1=rect;mask=np.zeros((64,64),bool);mask[y0:y1,x0:x1]=True
            masks[mid]=mask;rgba[mask]=color
        Image.fromarray(rgba).save(self.image)
        def piece(mid,order):
            b=io.BytesIO();Image.fromarray(masks[mid].astype('uint8')*255).save(b,format='PNG');raw=b.getvalue()
            return {'id':mid+'_p0','draw_order':order,'mask_png_base64':base64.b64encode(raw).decode(),
                    'mask_sha256':hashlib.sha256(raw).hexdigest()}
        modules=[{'id':'body','role':'torso','motion_type':'rigid','parent':None,'pivot':[20/64,35/64],
                  'motion':'breath','pieces':[piece('body',0)],'confidence':1},
                 {'id':'head','role':'head','motion_type':'rigid','parent':'body','pivot':[20/64,13/64],
                  'motion':'sway','pieces':[piece('head',1)],'confidence':1},
                 {'id':'cloth','role':'cloth','motion_type':'soft','parent':'body','pivot':[20/64,19/64],
                  'motion':'soft_bend','centerline':[[20/64,19/64],[20/64,15/64],[20/64,12/64]],
                  'pieces':[piece('cloth',2)],'confidence':1}]
        self.a={'version':2,'input_sha256':hashlib.sha256(self.image.read_bytes()).hexdigest(),
                'image_size':[64,64],'coord_mode':'normalized','pose_type':'airborne','anatomy':'humanoid_bust',
                'modules':modules,'uncertainties':[],
                'connections':[{'id':'neck','type':'joint','a':'body','b':'head','point':[20/64,13/64],
                                'hidden_by':[{'front':'cloth_p0','back':'body_p0'}]},
                               {'id':'cloth_root','type':'attachment','a':'body','b':'cloth','point':[20/64,19/64]}],
                'occlusions':[{'front':'cloth_p0','back':'body_p0','completion':'none',
                               'polygons':[[[18/64,13/64],[22/64,13/64],[22/64,20/64],[18/64,20/64]]]}]}

    def compile(self,a=None):
        a=a or self.a
        hints=materialize_module_contract(a,image_path=self.image,mask_dir=self.root/'masks')
        ps,rs=build_default_specs(self.image,character='unrelated_name',semantic_hints=hints)
        split=split_parts(self.image,ps['parts'],self.root/'parts',min_island_px=0,feather_px=0,module_contract=ps['module_contract'])
        made=build_rig(self.root/'parts',rs,self.root/'rig')
        return hints,ps,rs,split,made

    def auditor(self,mode='steady'):
        hints,ps,rs,split,made=self.compile()
        path=Path(made['spine_path']);sk=json.loads(path.read_text())
        # A real soft mesh occluder, animated independently of its rigid driver.
        sk['animations']['idle']={'bones':{}}
        if mode=='opens':
            sk['animations']['idle']['bones']['module_cloth']={'translate':[{'time':0,'x':0,'y':0},
                {'time':4,'x':25,'y':0},{'time':8,'x':0,'y':0}]}
        elif mode=='drift':
            sk['animations']['idle']['bones']['module_head']={'translate':[{'time':0,'x':0,'y':0},
                {'time':4,'x':4,'y':0},{'time':8,'x':0,'y':0}]}
        path.write_text(json.dumps(sk));ren=SpineRenderer(self.root/'rig')
        qc=ModuleQualityAuditor(ren,self.root/'parts',made['report'],ren.frame(None))
        return ren,qc,split

    def test_hidden_parent_without_fill_can_pass_full_named_coverage(self):
        ren,qc,split=self.auditor()
        self.assertEqual(split['rest_diff']['maxRGBA'],0)
        body=next(p for p in split['parts_json']['parts'] if p['moduleId']=='body')
        self.assertEqual(body['hiddenPixels'],0)
        self.assertFalse(qc.errors,qc.errors)
        qc.audit_hidden_joint_cycle()
        check=qc.finish()['machineQC']['checks']['hiddenJointCoverage']
        self.assertEqual(check['status'],'pass',check)
        self.assertEqual(check['joints']['neck']['requiredFramePoses'],241)
        self.assertGreater(check['joints']['neck']['unfilledReferencePixels'],0)

    def test_missing_reference_keeps_visible_distance_guard(self):
        a=copy.deepcopy(self.a);a['connections'][0].pop('hidden_by')
        with self.assertRaisesRegex(VisionReviewRequired,'neck'):
            self.compile(a)

    def test_unbound_pair_and_grip_are_rejected(self):
        for change in ('pair','grip','foot'):
            a=copy.deepcopy(self.a)
            if change=='pair':a['connections'][0]['hidden_by'][0]['front']='head_p0'
            elif change=='grip':
                a['connections'][0]['type']='grip';a['modules'][1]['motion']='follow'
            else:
                a['connections'][0]['type']='foot_anchor'
            with self.subTest(change=change),self.assertRaises((VisionReviewRequired,VisionValidationError)):
                validate_module_contract(a,input_sha256=a['input_sha256'],image_size=(64,64))

    def test_disconnected_corridor_does_not_authorize_parent(self):
        a=copy.deepcopy(self.a)
        a['occlusions'][0]['polygons']=[[[18/64,13/64],[22/64,13/64],[22/64,16/64],[18/64,16/64]]]
        with self.assertRaisesRegex(VisionReviewRequired,'连通'):
            self.compile(a)

    def test_translucent_source_cannot_authorize_opaque_bridge(self):
        rgba=np.asarray(Image.open(self.image)).copy();rgba[15,20,3]=128;Image.fromarray(rgba).save(self.image)
        self.a['input_sha256']=hashlib.sha256(self.image.read_bytes()).hexdigest()
        with self.assertRaisesRegex(VisionReviewRequired,'不透明'):
            self.compile()

    def test_forged_projection_target_rejected_despite_new_hash(self):
        hints=materialize_module_contract(self.a,image_path=self.image,mask_dir=self.root/'masks')
        oc=next(o for o in hints['module_projection']['occlusions'] if not o.get('autoSeamUnderlap'))
        path=Path(oc['sourceHiddenTargetMask']);mask=np.asarray(Image.open(path)).copy();mask[13,19]=0
        Image.fromarray(mask).save(path);oc['sourceHiddenTargetMaskSHA256']=hashlib.sha256(path.read_bytes()).hexdigest()
        with self.assertRaisesRegex(VisionReviewRequired,'原合同'):
            build_default_specs(self.image,semantic_hints=hints)

    def test_opening_occluder_fails_even_if_composite_frame_is_opaque(self):
        ren,qc,_=self.auditor('opens')
        opaque=ren.frame(4);opaque[...,3]=255
        qc.observe(4,1,opaque);qc.audit_hidden_joint_cycle()
        check=qc.finish()['machineQC']['checks']['hiddenJointCoverage']
        self.assertEqual(check['status'],'fail',check)
        self.assertGreater(check['joints']['neck']['maxUncoveredPixels'],0)

    def test_sparse_samples_cannot_claim_full_loop(self):
        ren,qc,_=self.auditor()
        for t in (0,2,4,6,8):qc.observe(t,1,ren.frame(t))
        self.assertEqual(qc.finish()['machineQC']['checks']['hiddenJointCoverage']['status'],'pending')

    def test_hidden_joint_still_rejects_relative_drift(self):
        ren,qc,_=self.auditor('drift');qc.observe(4,1,ren.frame(4));qc.audit_hidden_joint_cycle()
        q=qc.finish()
        self.assertEqual(q['machineQC']['checks']['hiddenJointCoverage']['status'],'pass')
        self.assertEqual(q['bindingMotion']['checks']['contacts']['status'],'fail')

    def test_dynamic_alpha_loss_is_not_effective_occlusion(self):
        ren,qc,_=self.auditor()
        # The corridor remains geometrically covered, but the named covering
        # material becomes translucent. An opaque unrelated scene cannot help.
        ren.tex['piece_cloth_p0'][...,3]=100
        qc.audit_hidden_joint_cycle()
        self.assertEqual(qc.finish()['machineQC']['checks']['hiddenJointCoverage']['status'],'fail')

    def test_wrong_layer_and_foreground_effect_cannot_be_references(self):
        for kind in ('layer','effect','foreign_parent'):
            a=copy.deepcopy(self.a)
            if kind=='layer':a['modules'][2]['pieces'][0]['draw_order']=-1
            elif kind=='effect':
                a['modules'][2].update(role='effect_group',locked_root=True)
            else:
                a['modules'][2]['parent']='head';a['connections'][1]['a']='head'
            with self.subTest(kind=kind),self.assertRaises((VisionReviewRequired,VisionValidationError)):
                validate_module_contract(a,input_sha256=a['input_sha256'],image_size=(64,64))

    def test_runtime_rejects_changed_reference_and_unrelated_front(self):
        hints,ps,rs,split,made=self.compile()
        report=copy.deepcopy(made['report'])
        hidden=next(c for c in report['moduleContract']['contacts'] if c.get('hiddenBy'))
        hidden['hiddenBy']=[{'front':'piece_head_p0','back':'piece_body_p0'}]
        ren=SpineRenderer(self.root/'rig');qc=ModuleQualityAuditor(ren,self.root/'parts',report,ren.frame(None))
        self.assertTrue(qc.errors)
        self.assertFalse(qc.finish()['overall']['exportAllowed'])

    def test_runtime_forged_hash_cannot_shrink_registered_corridor(self):
        hints,ps,rs,split,made=self.compile()
        report=copy.deepcopy(made['report'])
        oc=next(o for o in report['moduleContract']['occlusions'] if not o.get('autoSeamUnderlap'))
        path=Path(oc['sourceHiddenTargetMask']);mask=np.asarray(Image.open(path)).copy();mask[14,19]=0
        Image.fromarray(mask).save(path);oc['sourceHiddenTargetMaskSHA256']=hashlib.sha256(path.read_bytes()).hexdigest()
        ren=SpineRenderer(self.root/'rig');qc=ModuleQualityAuditor(ren,self.root/'parts',report,ren.frame(None))
        self.assertTrue(qc.errors,qc.finish())
        self.assertFalse(qc.finish()['overall']['exportAllowed'])
        report=copy.deepcopy(made['report'])
        oc=next(o for o in report['moduleContract']['occlusions'] if not o.get('autoSeamUnderlap'))
        Image.new('L',(64,64)).save(oc['sourceHiddenTargetMask'])
        qc=ModuleQualityAuditor(ren,self.root/'parts',report,ren.frame(None))
        self.assertTrue(qc.errors)
        self.assertFalse(qc.finish()['overall']['exportAllowed'])

    def test_run_audit_requests_dense_hidden_coverage(self):
        from src.gates.audit import run_audit
        self.auditor('steady')
        report=run_audit(self.root/'parts',self.root/'rig',sample_count=2,stress_samples=1)
        gate=report['quality']['machineQC']['checks']['hiddenJointCoverage']
        self.assertEqual(gate['status'],'pass',gate)
        self.assertTrue(gate['joints']['neck']['fullCycleSampled'])

    def approved_synthetic_export(self, *, approve=True):
        """Real compiled/audited pass; visual declaration is synthetic test-only."""
        from tests.pipeline.test_complete_modules import fixture
        from src.gates.audit import run_audit
        from src.gates.review_binding import actual_review_binding, ensure_export_approved, VISUAL_DIMENSIONS
        from src.gates.module_quality import finalize_visual_review
        a=fixture(self.image,local_fill=True)
        hints=materialize_module_contract(a,image_path=self.image,mask_dir=self.root/'original-masks')
        for module in a['modules']:
            for piece in module['pieces']:
                raw=Path(hints['module_masks'][piece['id']]).read_bytes()
                piece.update(mask_png_base64=base64.b64encode(raw).decode(),mask_sha256=hashlib.sha256(raw).hexdigest())
        face=next(m for m in a['modules'] if m['id']=='face');face['pivot']=[61/128,49/160]
        a['connections'][0].update(point=face['pivot'],hidden_by=[{'front':'face_p0','back':'body_p0'}])
        a['job_id']='synthetic_hidden_export';self.a=a
        _,_,_,_,made=self.compile()
        plan=self.root/'analysis.json';plan.write_text(json.dumps(a))
        report=run_audit(self.root/'parts',self.root/'rig',sample_count=8,stress_samples=4)
        self.assertEqual(report['status'],'pass')
        self.assertEqual(report['quality']['machineQC']['checks']['hiddenJointCoverage']['status'],'pass')
        if not approve:
            return plan,made,report
        binding=actual_review_binding(self.root/'rig',source_image=self.image,analysis_file=plan,job_id=a['job_id'])
        evidence={'source':'current_cindy_conversation','reviewer':'synthetic regression fixture',
                  'notes':'Test fixture only, not a real portrait approval.',
                  'full_loop_observed':True,'original_and_modules_viewed':True,'review_binding':binding,
                  'paths':{'source_image':str(self.image),'analysis_file':str(plan)},
                  'observations':{k:{'status':'pass','notes':'Synthetic invariant'} for k in VISUAL_DIMENSIONS}}
        report=finalize_visual_review(report,decision='pass',evidence=evidence)
        health=self.root/'rig/health-report.json';health.write_text(json.dumps(report))
        ensure_export_approved(self.root/'rig')
        return plan,made,binding

    def test_audited_external_target_and_remaining_changes_revoke_export(self):
        from src.gates.review_binding import actual_review_binding, ensure_export_approved, ExportApprovalError
        plan,made,binding=self.approved_synthetic_export()
        oc=next(o for o in made['report']['moduleContract']['occlusions'] if not o.get('autoSeamUnderlap'))
        for key in ('sourceHiddenTargetMask','remainingHiddenMask'):
            path=Path(oc[key]);original=path.read_bytes()
            self.assertFalse(path.is_relative_to(self.root/'rig'))
            try:
                mask=np.asarray(Image.open(path)).copy();mask[0,0]=255-mask[0,0];Image.fromarray(mask).save(path)
                with self.subTest(key=key),self.assertRaises(ExportApprovalError):
                    actual_review_binding(self.root/'rig',source_image=self.image,analysis_file=plan,job_id=self.a['job_id'])
                with self.assertRaises(ExportApprovalError):ensure_export_approved(self.root/'rig')
            finally:path.write_bytes(original)
        self.assertEqual(actual_review_binding(self.root/'rig',source_image=self.image,analysis_file=plan,
                                              job_id=self.a['job_id']),binding)

    def test_updated_hash_cannot_replace_expected_target_or_canvas(self):
        from src.gates.review_binding import actual_review_binding, ExportApprovalError
        plan,made,binding=self.approved_synthetic_export()
        report_path=self.root/'rig/rig-build-report.json';original_report=report_path.read_bytes()
        oc=next(o for o in made['report']['moduleContract']['occlusions'] if not o.get('autoSeamUnderlap'))
        target_path=Path(oc['sourceHiddenTargetMask']);original=target_path.read_bytes()
        for wrong in ('content','dimensions'):
            try:
                if wrong=='dimensions':Image.new('L',(8,8)).save(target_path)
                else:
                    pixels=np.asarray(Image.open(io.BytesIO(original))).copy();pixels[49,60]=0
                    Image.fromarray(pixels).save(target_path)
                report=json.loads(original_report)
                ref=next(o for o in report['moduleContract']['occlusions'] if not o.get('autoSeamUnderlap'))
                ref['sourceHiddenTargetMaskSHA256']=hashlib.sha256(target_path.read_bytes()).hexdigest()
                report_path.write_text(json.dumps(report))
                with self.subTest(wrong=wrong),self.assertRaises(ExportApprovalError):
                    actual_review_binding(self.root/'rig',source_image=self.image,analysis_file=plan,job_id=self.a['job_id'])
            finally:
                target_path.write_bytes(original);report_path.write_bytes(original_report)

    def test_moving_same_bytes_changes_binding_and_revokes_old_approval(self):
        from src.gates.review_binding import actual_review_binding, ensure_export_approved, ExportApprovalError
        plan,made,binding=self.approved_synthetic_export()
        report_path=self.root/'rig/rig-build-report.json';report=json.loads(report_path.read_text())
        oc=next(o for o in report['moduleContract']['occlusions'] if not o.get('autoSeamUnderlap'))
        new=self.root/'masks/renamed-target.png';new.write_bytes(Path(oc['sourceHiddenTargetMask']).read_bytes())
        oc['sourceHiddenTargetMask']=str(new);report_path.write_text(json.dumps(report))
        new_binding=actual_review_binding(self.root/'rig',source_image=self.image,analysis_file=plan,job_id=self.a['job_id'])
        self.assertNotEqual(binding['artifact_sha256'],new_binding['artifact_sha256'])
        with self.assertRaises(ExportApprovalError):ensure_export_approved(self.root/'rig')

    def test_external_absolute_parent_traversal_and_symlink_escape_rejected(self):
        from src.gates.review_binding import actual_review_binding, ensure_export_approved, ExportApprovalError
        plan,made,binding=self.approved_synthetic_export()
        outside=tempfile.TemporaryDirectory(dir=ROOT/'.studio-work',prefix='foreign-evidence-')
        self.addCleanup(outside.cleanup)
        report_path=self.root/'rig/rig-build-report.json';raw_report=report_path.read_bytes()
        report=json.loads(raw_report)
        oc=next(o for o in report['moduleContract']['occlusions'] if not o.get('autoSeamUnderlap'))
        external=Path(outside.name)/'target.png';external.write_bytes(Path(oc['sourceHiddenTargetMask']).read_bytes())
        link=self.root/'masks/escape.png';link.symlink_to(external)
        for path in (str(external),str(link),'../'+Path(outside.name).name+'/target.png'):
            report=json.loads(raw_report)
            ref=next(o for o in report['moduleContract']['occlusions'] if not o.get('autoSeamUnderlap'))
            ref['sourceHiddenTargetMask']=path;report_path.write_text(json.dumps(report))
            with self.subTest(path=path),self.assertRaises(ExportApprovalError):
                actual_review_binding(self.root/'rig',source_image=self.image,analysis_file=plan,job_id=self.a['job_id'])
            with self.assertRaises(ExportApprovalError):ensure_export_approved(self.root/'rig')
            ren=SpineRenderer(self.root/'rig');qc=ModuleQualityAuditor(ren,self.root/'parts',report,ren.frame(None))
            self.assertTrue(qc.errors,'runtime must reject the path, not merely lack motion samples')
            self.assertEqual(qc.finish()['inputOwnership']['checks']['contract']['status'],'fail')
            self.assertFalse(qc.finish()['overall']['exportAllowed'])
        report_path.write_bytes(raw_report)

    def test_compile_rejects_foreign_target_before_reading_it(self):
        from unittest.mock import patch
        hints=materialize_module_contract(self.a,image_path=self.image,mask_dir=self.root/'masks')
        oc=next(o for o in hints['module_projection']['occlusions'] if not o.get('autoSeamUnderlap'))
        other=self.root/'outside-mask-dir.png'
        other.write_bytes(Path(oc['sourceHiddenTargetMask']).read_bytes())
        oc['sourceHiddenTargetMask']=str(other)
        read=Path.read_bytes;reads=[]
        def tracked(path):
            reads.append(path.resolve());return read(path)
        with patch.object(Path,'read_bytes',tracked),self.assertRaises(VisionReviewRequired):
            build_default_specs(self.image,semantic_hints=hints)
        self.assertNotIn(other.resolve(),reads)

    def test_source_and_analysis_foreign_paths_rejected_before_read(self):
        from unittest.mock import patch
        from src.gates.review_binding import actual_review_binding, ensure_export_approved, ExportApprovalError
        plan,_,_=self.approved_synthetic_export()
        outside=tempfile.TemporaryDirectory(dir=ROOT/'.studio-work',prefix='foreign-inputs-')
        self.addCleanup(outside.cleanup);foreign_root=Path(outside.name)
        health=self.root/'rig/health-report.json';health_bytes=health.read_bytes()
        directory_link=self.root/'foreign-input-dir';directory_link.symlink_to(foreign_root,target_is_directory=True)
        for field,original in [('source_image',self.image),('analysis_file',plan)]:
            external=foreign_root/original.name;external.write_bytes(original.read_bytes())
            link=self.root/('link-'+original.name);link.symlink_to(external)
            values=[str(external),'../'+foreign_root.name+'/'+original.name,
                    str(link),str(directory_link/original.name)]
            for value in values:
                report=json.loads(health_bytes);report['quality']['visual']['evidence']['paths'][field]=value
                health.write_text(json.dumps(report))
                calls=[];raw_read=Path.read_bytes;raw_text=Path.read_text
                def read_bytes(path):
                    calls.append(path.resolve());return raw_read(path)
                def read_text(path,*args,**kwargs):
                    calls.append(path.resolve());return raw_text(path,*args,**kwargs)
                params={'source_image':self.image,'analysis_file':plan,'job_id':self.a['job_id']}
                params[field]=value
                with self.subTest(field=field,value=value),patch.object(Path,'read_bytes',read_bytes),patch.object(Path,'read_text',read_text):
                    with self.assertRaises(ExportApprovalError):actual_review_binding(self.root/'rig',**params)
                    with self.assertRaises(ExportApprovalError):ensure_export_approved(self.root/'rig')
                self.assertNotIn(external.resolve(),calls,'foreign input must not be read before rejection')
                health.write_bytes(health_bytes)
        ensure_export_approved(self.root/'rig')

    def test_same_bytes_relocated_inputs_revoke_previous_approval(self):
        from src.gates.review_binding import actual_review_binding, ensure_export_approved, ExportApprovalError
        plan,_,binding=self.approved_synthetic_export()
        health=self.root/'rig/health-report.json';old_health=health.read_bytes()
        for field,original in [('source_image',self.image),('analysis_file',plan)]:
            clone=self.root/('copied-'+original.name);clone.write_bytes(original.read_bytes())
            params={'source_image':self.image,'analysis_file':plan,'job_id':self.a['job_id']};params[field]=clone
            current=actual_review_binding(self.root/'rig',**params)
            self.assertEqual(current['input_sha256'],binding['input_sha256'])
            self.assertEqual(current['analysis_sha256'],binding['analysis_sha256'])
            self.assertNotEqual(current['artifact_sha256'],binding['artifact_sha256'])
            report=json.loads(old_health);report['quality']['visual']['evidence']['paths'][field]=str(clone)
            health.write_text(json.dumps(report))
            with self.subTest(field=field),self.assertRaises(ExportApprovalError):ensure_export_approved(self.root/'rig')
            health.write_bytes(old_health)

    def test_task_relative_inputs_resolve_without_changing_binding(self):
        from src.gates.review_binding import actual_review_binding, ensure_export_approved
        plan,_,binding=self.approved_synthetic_export()
        relative=actual_review_binding(self.root/'rig',source_image=self.image.name,
                                      analysis_file=plan.name,job_id=self.a['job_id'])
        self.assertEqual(relative,binding)
        self.assertEqual(binding['source_relative_path'],self.image.name)
        self.assertEqual(binding['analysis_relative_path'],plan.name)
        health=self.root/'rig/health-report.json';report=json.loads(health.read_text())
        report['quality']['visual']['evidence']['paths'].update(source_image=self.image.name,analysis_file=plan.name)
        health.write_text(json.dumps(report))
        self.assertTrue(ensure_export_approved(self.root/'rig')['quality']['overall']['exportAllowed'])

    def test_health_report_override_is_confined_before_reading(self):
        from unittest.mock import patch
        from src.gates.review_binding import ensure_export_approved, ExportApprovalError
        self.approved_synthetic_export();health=self.root/'rig/health-report.json'
        other=tempfile.TemporaryDirectory(dir=ROOT/'.studio-work',prefix='foreign-health-');self.addCleanup(other.cleanup)
        external=Path(other.name)/'health.json';external.write_bytes(health.read_bytes())
        link=self.root/'health-link.json';link.symlink_to(external)
        for path in (external,link,'../'+Path(other.name).name+'/health.json'):
            calls=[];original=Path.read_text
            def read(path,*args,**kwargs):
                calls.append(path.resolve());return original(path,*args,**kwargs)
            with self.subTest(path=str(path)),patch.object(Path,'read_text',read),self.assertRaises(ExportApprovalError):
                ensure_export_approved(self.root/'rig',health_report_path=path)
            self.assertNotIn(external.resolve(),calls)
        own=self.root/'custom-health.json';own.write_bytes(health.read_bytes())
        self.assertTrue(ensure_export_approved(self.root/'rig',health_report_path='custom-health.json')['quality']['overall']['exportAllowed'])

    def test_build_report_escape_is_rejected_before_json_read(self):
        from unittest.mock import patch
        from src.gates.review_binding import actual_review_binding, ExportApprovalError
        plan,_,_=self.approved_synthetic_export();report=self.root/'rig/rig-build-report.json'
        external=self.root/'outside-rig-build-report.json';external.write_bytes(report.read_bytes())
        report.unlink();report.symlink_to(external)
        calls=[];original=Path.read_text
        def read(path,*args,**kwargs):
            calls.append(path.resolve());return original(path,*args,**kwargs)
        with patch.object(Path,'read_text',read),self.assertRaises(ExportApprovalError):
            actual_review_binding(self.root/'rig',source_image=self.image,analysis_file=plan,job_id=self.a['job_id'])
        self.assertNotIn(external.resolve(),calls)

    def test_pre_animation_binding_rejects_source_escape(self):
        from unittest.mock import patch
        from src.gates.review_binding import module_review_binding, ExportApprovalError
        directory=self.root/'job';directory.mkdir()
        (directory/'matted.png').symlink_to(self.image)
        calls=[];original=Path.read_bytes
        def read(path):
            calls.append(path.resolve());return original(path)
        with patch.object(Path,'read_bytes',read),self.assertRaises(ExportApprovalError):
            module_review_binding(directory,self.a)
        self.assertNotIn(self.image.resolve(),calls)

    def test_visual_validation_uses_the_same_checked_analysis_snapshot(self):
        from unittest.mock import patch
        from src.gates import review_binding
        plan,_,_=self.approved_synthetic_export();original=Path.read_bytes;plan_reads=[]
        original_text=Path.read_text
        def read(path):
            if path.resolve()==plan.resolve():plan_reads.append(path)
            return original(path)
        def read_text(path,*args,**kwargs):
            self.assertNotEqual(path.resolve(),plan.resolve(),'analysis must not be reopened as unchecked text')
            return original_text(path,*args,**kwargs)
        with patch.object(Path,'read_bytes',read),patch.object(Path,'read_text',read_text),patch.object(review_binding,'validate_visual_evidence',wraps=review_binding.validate_visual_evidence) as validate:
            review_binding.ensure_export_approved(self.root/'rig')
        self.assertEqual(len(plan_reads),1)
        self.assertEqual(validate.call_args.args[1]['job_id'],self.a['job_id'])

    def test_images_directory_link_is_rejected_before_first_binding(self):
        import shutil
        from unittest.mock import patch
        from src.gates.review_binding import actual_review_binding, ExportApprovalError
        plan,_,report=self.approved_synthetic_export(approve=False)
        self.assertEqual(report['quality']['machineQC']['checks']['hiddenJointCoverage']['status'],'pass')
        foreign=tempfile.TemporaryDirectory(dir=ROOT/'.studio-work',prefix='foreign-images-')
        self.addCleanup(foreign.cleanup);external=Path(foreign.name)/'images'
        images=self.root/'rig/images';shutil.copytree(images,external)
        images.rename(self.root/'original-images');images.symlink_to(external,target_is_directory=True)
        reads=[];original=Path.read_bytes
        def read(path):
            reads.append(path.resolve());return original(path)
        with patch.object(Path,'read_bytes',read),self.assertRaisesRegex(ExportApprovalError,'符号链接'):
            actual_review_binding(self.root/'rig',source_image=self.image,analysis_file=plan,job_id=self.a['job_id'])
        self.assertFalse(any(p.is_relative_to(external) for p in reads))

    def test_external_images_changes_cannot_reuse_approval_or_read_foreign_png(self):
        import shutil
        from unittest.mock import patch
        from src.gates.review_binding import actual_review_binding, ensure_export_approved, ExportApprovalError
        plan,_,_=self.approved_synthetic_export()
        foreign=tempfile.TemporaryDirectory(dir=ROOT/'.studio-work',prefix='foreign-textures-')
        self.addCleanup(foreign.cleanup);external=Path(foreign.name)/'images'
        images=self.root/'rig/images';shutil.copytree(images,external)
        images.rename(self.root/'original-images');images.symlink_to(external,target_is_directory=True)
        texture=next(external.glob('*.png'))
        for changed in (False,True):
            if changed:
                pixels=np.asarray(Image.open(texture)).copy();pixels[0,0,0]^=255
                Image.fromarray(pixels).save(texture)
            reads=[];raw_bytes=Path.read_bytes;raw_text=Path.read_text
            def read_bytes(path):
                reads.append(path.resolve());return raw_bytes(path)
            def read_text(path,*args,**kwargs):
                reads.append(path.resolve());return raw_text(path,*args,**kwargs)
            with self.subTest(changed=changed),patch.object(Path,'read_bytes',read_bytes),patch.object(Path,'read_text',read_text):
                with self.assertRaisesRegex(ExportApprovalError,'符号链接'):
                    actual_review_binding(self.root/'rig',source_image=self.image,analysis_file=plan,job_id=self.a['job_id'])
                with self.assertRaisesRegex(ExportApprovalError,'符号链接'):
                    ensure_export_approved(self.root/'rig')
            self.assertFalse(any(p.is_relative_to(external) for p in reads))

    def test_in_root_and_broken_rig_links_are_not_silently_skipped(self):
        from src.gates.review_binding import actual_review_binding, ensure_export_approved, ExportApprovalError
        plan,_,binding=self.approved_synthetic_export()
        nested=self.root/'rig/nested';nested.mkdir()
        for name,target,directory in [('same-root-images',self.root/'rig/images',True),
                                      ('same-root-file',next((self.root/'rig/images').glob('*.png')),False),
                                      ('missing-directory',self.root/'nonexistent-dir',True)]:
            link=nested/name;link.symlink_to(target,target_is_directory=directory)
            with self.subTest(name=name),self.assertRaisesRegex(ExportApprovalError,'符号链接'):
                actual_review_binding(self.root/'rig',source_image=self.image,analysis_file=plan,job_id=self.a['job_id'])
            with self.assertRaisesRegex(ExportApprovalError,'符号链接'):ensure_export_approved(self.root/'rig')
            link.unlink()
        self.assertEqual(actual_review_binding(self.root/'rig',source_image=self.image,analysis_file=plan,
                                              job_id=self.a['job_id']),binding)
        ensure_export_approved(self.root/'rig')

    def test_renderer_image_and_attachment_paths_cannot_bypass_tree_binding(self):
        import shutil
        from unittest.mock import patch
        from src.gates.review_binding import actual_review_binding, ensure_export_approved, ExportApprovalError
        plan,made,_=self.approved_synthetic_export()
        outside=tempfile.TemporaryDirectory(dir=ROOT/'.studio-work',prefix='external-render-assets-')
        self.addCleanup(outside.cleanup);foreign=Path(outside.name)/'images'
        shutil.copytree(self.root/'rig/images',foreign)
        spine=Path(made['spine_path']);before=spine.read_bytes()
        for case in ('absolute_images','relative_images','absolute_attachment','relative_attachment'):
            data=json.loads(before)
            if case=='absolute_images':data['skeleton']['images']=str(foreign)
            elif case=='relative_images':data['skeleton']['images']='../../'+Path(outside.name).name+'/images'
            elif case=='absolute_attachment':data['slots'][0]['attachment']=str(next(foreign.glob('*.png')).with_suffix(''))
            else:data['slots'][0]['attachment']='../../../'+Path(outside.name).name+'/images/'+next(foreign.glob('*.png')).stem
            spine.write_text(json.dumps(data))
            reads=[];original=Path.read_bytes
            def read(path):
                reads.append(path.resolve());return original(path)
            with self.subTest(case=case),patch.object(Path,'read_bytes',read):
                with self.assertRaises(ExportApprovalError):
                    actual_review_binding(self.root/'rig',source_image=self.image,analysis_file=plan,job_id=self.a['job_id'])
                with self.assertRaises(ExportApprovalError):ensure_export_approved(self.root/'rig')
            self.assertFalse(any(p.is_relative_to(foreign) for p in reads))
            spine.write_bytes(before)
        ensure_export_approved(self.root/'rig')


if __name__=='__main__':unittest.main()
