"""Canvas-to-production material handoff; synthetic fixtures are not real media claims."""
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
from src.studio.module_contract import materialize_module_contract,validate_module_contract
from src.studio.vision_schema import VisionValidationError,VisionReviewRequired
from src.studio.auto_spec import build_default_specs
from src.pipeline.split_parts import split_parts
from src.pipeline.build_rig import build_rig
from src.render.renderer import SpineRenderer
from src.gates.module_quality import ModuleQualityAuditor
from src.gates.review_binding import actual_review_binding,ExportApprovalError
from tests.pipeline.test_complete_modules import fixture

ROOT=Path(__file__).resolve().parents[2]


class CanvasCompletionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=ROOT/'.studio-work',prefix='canvas-completion-')
        self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.image=self.root/'source.png'
        self.a=fixture(self.image,local_fill=True);self.a['job_id']='canvas_test_fixture'
        self.canvas=np.full((160,128,4),[73,41,222,255],np.uint8)
        self.a['occlusions'][0].update(completion='provided',provided=self.payload(self.canvas))

    def payload(self,rgba):
        b=io.BytesIO();Image.fromarray(rgba).save(b,format='PNG');raw=b.getvalue()
        return {'png_base64':base64.b64encode(raw).decode(),'sha256':hashlib.sha256(raw).hexdigest(),
                'input_sha256':self.a['input_sha256'],'canvas_scene_id':'synthetic-test-only',
                'canvas_node_id':'synthetic-layer','canvas_task_id':'synthetic-operation',
                'operation':'edit-image','source_asset_sha256':self.a['input_sha256'],
                'result_asset_sha256':hashlib.sha256(raw).hexdigest()}

    def compile(self,a=None):
        a=a or self.a;hints=materialize_module_contract(a,image_path=self.image,mask_dir=self.root/'masks')
        ps,rs=build_default_specs(self.image,character='arbitrary',semantic_hints=hints)
        s=split_parts(self.image,ps['parts'],self.root/'parts',min_island_px=0,feather_px=0,module_contract=ps['module_contract'])
        made=build_rig(self.root/'parts',rs,self.root/'rig')
        return hints,ps,s,made

    def test_canvas_material_reaches_actual_parts_and_qc_preserving_visible_source(self):
        hints,ps,s,made=self.compile()
        self.assertEqual(s['rest_diff']['maxRGBA'],0)
        body=next(p for p in s['parts_json']['parts'] if p.get('pieceId')=='body_p0')
        layer=np.asarray(Image.open(self.root/'parts'/body['image']))
        x,y,_,_=body['bbox']
        np.testing.assert_array_equal(layer[49-y,60-x],self.canvas[49,60])
        source=np.asarray(Image.open(self.image))
        np.testing.assert_array_equal(layer[70-y,60-x],source[70,60])
        supplied=np.asarray(Image.open(body['providedImage']))
        self.assertEqual(int((supplied[...,3]>0).sum()),29)
        ren=SpineRenderer(self.root/'rig');qc=ModuleQualityAuditor(ren,self.root/'parts',made['report'],ren.frame(None))
        hidden=qc.finish()['machineQC']['checks']['hiddenFill']
        self.assertEqual(hidden['status'],'pass',hidden)
        self.assertEqual(hidden['changedCanvasMaterialPixels'],0)

    def test_provided_can_fill_deep_missing_domain_without_local_smear(self):
        c=copy.deepcopy(self.a)
        for p in c['modules'][0]['pieces'][0]['polygons'][0][:2]:p[1]=.15
        c['occlusions'][0]['polygons']=[[[47/128,.15],[76/128,.15],[76/128,50/160],[47/128,50/160]]]
        _,_,s,_=self.compile(c)
        self.assertEqual(s['rest_diff']['maxRGBA'],0)
        body=next(p for p in s['parts_json']['parts'] if p.get('pieceId')=='body_p0')
        layer=np.asarray(Image.open(self.root/'parts'/body['image']));x,y,_,_=body['bbox']
        np.testing.assert_array_equal(layer[26-y,60-x],self.canvas[26,60])

    def test_wrong_input_hash_size_and_receipt_cannot_enter(self):
        for case in ('source','sha','size','receipt','depth'):
            c=copy.deepcopy(self.a)
            if case=='source':c['occlusions'][0]['provided']['input_sha256']='0'*64
            elif case=='sha':c['occlusions'][0]['provided']['sha256']='0'*64
            elif case=='size':c['occlusions'][0]['provided']=self.payload(np.full((8,8,4),255,np.uint8))
            elif case=='receipt':c['occlusions'][0]['provided']['canvas_node_id']=''
            else:c['occlusions'][0]['band_depth_px_at_1080']=48
            with self.subTest(case=case),self.assertRaises(VisionValidationError):
                validate_module_contract(c,input_sha256=self.a['input_sha256'],image_size=(128,160))

    def test_empty_or_non_covering_canvas_result_requests_rerun(self):
        c=copy.deepcopy(self.a);rgba=np.zeros_like(self.canvas);rgba[90:100,50:60]=[73,41,222,255]
        c['occlusions'][0]['provided']=self.payload(rgba)
        with self.assertRaisesRegex(VisionReviewRequired,'没有覆盖'):
            materialize_module_contract(c,image_path=self.image,mask_dir=self.root/'bad')

    def test_semtransparent_source_not_silently_backfilled(self):
        src=np.asarray(Image.open(self.image)).copy();src[49,60,3]=100;Image.fromarray(src).save(self.image)
        self.a['input_sha256']=hashlib.sha256(self.image.read_bytes()).hexdigest()
        self.a['occlusions'][0]['provided']=self.payload(self.canvas)
        hints,_,s,_=self.compile()
        supplied=np.asarray(Image.open(hints['module_provided_images']['body_p0']))
        self.assertEqual(supplied[49,60,3],0)
        self.assertEqual(s['rest_diff']['maxRGBA'],0)

    def test_mutated_supplied_texture_fails_runtime_and_export_binding(self):
        _,_,s,made=self.compile();self.a['job_id']='canvas_test_fixture'
        analysis=self.root/'analysis.json';analysis.write_text(json.dumps(self.a))
        actual_review_binding(self.root/'rig',source_image=self.image,analysis_file=analysis,job_id=self.a['job_id'])
        body=next(p for p in s['parts_json']['parts'] if p.get('pieceId')=='body_p0')
        path=Path(body['providedImage']);pixels=np.asarray(Image.open(path)).copy();pixels[49,60,0]^=255;Image.fromarray(pixels).save(path)
        ren=SpineRenderer(self.root/'rig');qc=ModuleQualityAuditor(ren,self.root/'parts',made['report'],ren.frame(None))
        self.assertTrue(qc.errors)
        with self.assertRaises(ExportApprovalError):
            actual_review_binding(self.root/'rig',source_image=self.image,analysis_file=analysis,job_id=self.a['job_id'])

    def test_studio_apply_accepts_canvas_handoff_but_stops_for_binding_review(self):
        from src.studio.server import StudioJobManager
        manager=StudioJobManager(self.root/'jobs',audit_samples=4,stress_samples=2)
        job=manager.create_job({'filename':'synthetic.png','preset':'bust_half',
                               'image_base64':base64.b64encode(self.image.read_bytes()).decode()})
        plan=copy.deepcopy(self.a);plan.update(job_id=job['job_id'],input_sha256=job['analysis_request']['input_sha256'])
        plan['occlusions'][0]['provided']['input_sha256']=plan['input_sha256']
        bound=manager.apply_analysis_job(job['job_id'],{'job_id':job['job_id'],'analysis':plan})
        self.assertEqual(bound['status'],'awaiting_rig_confirm',bound.get('stage_error'))
        self.assertIsNone(bound['module_review'])
        self.assertEqual(bound['split_run_count'],1)

    def test_missing_result_has_actionable_canvas_handoff_not_local_fallback(self):
        plan=copy.deepcopy(self.a);plan['occlusions'][0].pop('provided')
        with self.assertRaises(VisionReviewRequired) as error:
            materialize_module_contract(plan,image_path=self.image,mask_dir=self.root/'missing')
        self.assertEqual(error.exception.code,'CANVAS_COMPLETION_REQUIRED')
        self.assertEqual(error.exception.details['back'],'body_p0')
        self.assertFalse((self.root/'missing').exists())


if __name__=='__main__':unittest.main()
