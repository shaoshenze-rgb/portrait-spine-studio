"""Synthetic HTTP handoff tests; no model invocation, plugin or fabricated Host receipt."""
import base64
import copy
import json
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from src.studio.server import StudioJobManager, create_studio_server
from src.studio.vision_provider import ConversationHandoffProvider, CindyVisionProvider
from tests.pipeline.test_complete_modules import fixture

ROOT = Path(__file__).resolve().parents[2]


class TestConversationHandoff(unittest.TestCase):
    def setUp(self):
        work=ROOT/'.studio-work/conversation-handoff-tests'
        work.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=work)
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.contract=fixture(self.root/"source.png")
        self.image_bytes=(self.root/"source.png").read_bytes()
        self.manager=StudioJobManager(self.root,audit_samples=4,stress_samples=2)
        self.server=create_studio_server(port=0,manager=self.manager)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True)
        self.thread.start()
        self.url=f'http://127.0.0.1:{self.server.server_port}'

    def tearDown(self):
        self.server.shutdown();self.server.server_close();self.thread.join()

    def request(self,path,payload=None):
        body=None if payload is None else json.dumps(payload).encode()
        request=urllib.request.Request(self.url+path,body,{'Content-Type':'application/json'})
        try:
            with urllib.request.urlopen(request,timeout=30) as response:return response.status,json.load(response)
        except urllib.error.HTTPError as error:return error.code,json.load(error)

    def prepare(self):
        status,job=self.request('/api/jobs',{'filename':'synthetic.png','image_base64':base64.b64encode(self.image_bytes).decode()})
        self.assertEqual(status,200)
        return job

    def result_for(self,job):
        request=job['analysis_request']
        result=copy.deepcopy(self.contract)
        result.update(input_sha256=request['input_sha256'],image_size=request['image_size'])
        result['job_id']=job['job_id']
        return result

    def apply(self,job,result):
        return self.request('/api/jobs/'+job['job_id']+'/apply-analysis',{'job_id':job['job_id'],'analysis':result})

    def module_review(self,job):
        return {'source':'current_cindy_conversation','reviewer':'synthetic HTTP fixture',
                'original_and_modules_viewed':True,'binding':job['module_review_binding'],
                'modules':{m['id']:{'status':'pass','notes':'Synthetic fixture assertion; not actual portrait approval: '+m['id']} for m in job['modules']}}

    def test_default_prepare_waits_without_model_or_plugin(self):
        with patch.object(CindyVisionProvider,'analyze',side_effect=AssertionError('must not call a bridge')):
            job=self.prepare()
        self.assertIsInstance(self.manager.vision_provider,ConversationHandoffProvider)
        self.assertEqual(job['status'],'awaiting_analysis')
        self.assertEqual(job['semantic_source'],'conversation_handoff')
        self.assertFalse(job['vision_metadata']['application_invoked_model'])
        self.assertEqual((job['split_run_count'],job['rig_build_count']),(0,0))
        self.assertTrue(Path(job['analysis_request']['image_path']).is_file())
        self.assertEqual(job['analysis_request']['job_id'],job['job_id'])
        with urllib.request.urlopen(self.url+job['analysis_request']['image_url']) as response:
            self.assertEqual(response.status,200)
            self.assertTrue(response.read().startswith(bytes([137,80,78,71])))

    def test_generate_before_analysis_is_rejected(self):
        job=self.prepare()
        status,result=self.request('/api/jobs/'+job['job_id']+'/confirm-rig',{})
        self.assertEqual(status,400)
        self.assertEqual(self.manager.get_job(job['job_id'])['status'],'awaiting_analysis')
        self.assertEqual(self.manager.get_job(job['job_id'])['rig_build_count'],0)

    def test_valid_analysis_stops_at_binding_then_explicit_generate(self):
        job=self.prepare();status,bound=self.apply(job,self.result_for(job))
        self.assertEqual(status,200)
        self.assertEqual(bound['status'],'awaiting_rig_confirm',bound.get('stage_error'))
        self.assertEqual(bound['semantic_source'],'conversation_handoff')
        self.assertTrue(bound['vision_metadata']['analysis_applied'])
        self.assertFalse(bound['vision_metadata']['application_invoked_model'])
        self.assertNotIn('execution_receipt',bound['vision_metadata'])
        self.assertEqual(bound['rig_build_count'],0)
        status,generated=self.request('/api/jobs/'+job['job_id']+'/confirm-rig',{'module_review':self.module_review(bound)})
        self.assertEqual(status,200)
        self.assertEqual(generated['rig_build_count'],1)
        self.assertIn(generated['status'],('awaiting_visual_review','needs_review','failed'))
        self.assertIsNotNone(generated['health_report'])

    def test_wrong_job_hash_size_and_low_confidence_stop_before_split(self):
        cases=[('job_id','other-job'),('input_sha256','0'*64),('image_size',[1,1]),('low_confidence',.1)]
        for key,value in cases:
            job=self.prepare();result=self.result_for(job)
            if key=='low_confidence':result['modules'][0]['confidence']=value
            else:result[key]=value
            with self.subTest(key=key):
                status,failed=self.apply(job,result)
                self.assertEqual(status,200)
                self.assertEqual(failed['status'],'failed')
                self.assertEqual(failed['stage_error']['failedStage'],'vision')
                self.assertEqual(failed['split_run_count'],0)
                self.assertFalse(failed['vision_metadata']['analysis_applied'])

    def test_changed_prepared_image_rejects_old_result(self):
        job=self.prepare();result=self.result_for(job)
        Image.new('RGBA',tuple(job['analysis_request']['image_size']),(50,70,90,255)).save(job['analysis_request']['image_path'])
        _,failed=self.apply(job,result)
        self.assertEqual(failed['status'],'failed')
        self.assertIn('图片已变化',failed['stage_error']['message'])
        self.assertEqual(failed['split_run_count'],0)

    def test_cli_prepare_apply_status_generate_uses_same_http_job(self):
        image=self.root/'cli-input.png';image.write_bytes(base64.b64decode(base64.b64encode(self.image_bytes).decode()))
        def cli(*args):
            run=subprocess.run([sys.executable,str(ROOT/'scripts/portrait_task.py'),'--base-url',self.url,*args],capture_output=True,text=True)
            return run.returncode,json.loads(run.stdout)
        code,job=cli('prepare','--image',str(image))
        self.assertEqual(code,0);self.assertEqual(job['status'],'awaiting_analysis')
        vision=self.root/'vision.json';vision.write_text(json.dumps(self.result_for(job)))
        code,bound=cli('apply-analysis','--job',job['job_id'],'--analysis',str(vision))
        self.assertEqual(code,0);self.assertEqual(bound['status'],'awaiting_rig_confirm')
        code,current=cli('status','--job',job['job_id'])
        self.assertEqual(code,0);self.assertEqual(current['job_id'],job['job_id'])
        observed=self.root/'module-observed.json';observed.write_text(json.dumps(self.module_review(bound)))
        _,generated=cli('generate','--job',job['job_id'],'--module-review',str(observed))
        self.assertEqual(generated['rig_build_count'],1)
        self.assertTrue(generated['page_url'].endswith('/?job='+job['job_id']))


if __name__=='__main__':unittest.main()
