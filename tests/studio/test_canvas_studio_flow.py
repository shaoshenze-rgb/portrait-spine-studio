import base64
import hashlib
import json
import threading
import unittest
import urllib.request
import urllib.error
from pathlib import Path
from tests.pipeline import test_canvas_layer_rig as fixtures
from src.studio.server import StudioJobManager,create_studio_server

class CanvasStudioFlowTests(unittest.TestCase):
    def setUp(self):
        f=fixtures.CanvasLayerRigTests();f.setUp();self.addCleanup(f.doCleanups);self.root=f.root
        self.parts=f.parts;self.plan=f.plan
        next(module for module in self.plan['modules'] if module['id']=='head')['label']='角色头部'
        manifest=json.loads((self.parts/'parts.json').read_text());source=Path(manifest['originalSource'])
        source.write_bytes(Path(manifest['source']).read_bytes());manifest['originalSourceSHA256']=hashlib.sha256(source.read_bytes()).hexdigest()
        (self.parts/'parts.json').write_text(json.dumps(manifest));self.source=source
        self.planpath=self.root/'plan.json';self.planpath.write_text(json.dumps(self.plan))
        manager=StudioJobManager(self.root/'studio')
        self.server=create_studio_server(host='127.0.0.1',port=0,work_dir=self.root/'studio',manager=manager)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.addCleanup(self.close)
        self.url='http://127.0.0.1:'+str(self.server.server_address[1])
    def close(self):self.server.shutdown();self.server.server_close();self.thread.join()
    def post(self,path,payload):
        req=urllib.request.Request(self.url+path,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
        try:
            with urllib.request.urlopen(req,timeout=120) as response:return response.status,json.load(response)
        except urllib.error.HTTPError as e:return e.code,json.load(e)
    def test_prepare_apply_tweak_signoff_export(self):
        status,job=self.post('/api/jobs',{'filename':'sample.png','image_base64':base64.b64encode(self.source.read_bytes()).decode()})
        self.assertEqual(status,200,job);self.assertEqual(job['status'],'awaiting_analysis')
        path='/api/jobs/'+job['job_id']
        status,job=self.post(path+'/apply-canvas-layers',{'parts_dir':str(self.parts),'plan':str(self.planpath),'cleanup':True})
        self.assertEqual(status,200,job);self.assertEqual(job['status'],'canvas_preview');self.assertTrue(all(job['canvas_checks'].values()),job['canvas_checks'])
        self.assertEqual(job['canvas_diagnostics']['frame_count'],240)
        with urllib.request.urlopen(self.url+'/?job='+job['job_id']) as response:
            page=response.read().decode('utf-8')
        self.assertIn('id="canvasMotionTab"',page)
        self.assertIn('id="canvasLiveStage"',page)
        self.assertNotIn('<iframe id="canvasPreviewFrame"',page)
        self.assertIn('id="btnCanvasTweaks"',page)
        self.assertIn('id="btnCanvasSignoff"',page)
        self.assertIn('id="canvasSeamImage"',page)
        self.assertIn('id="canvasLayerWarningStatus"',page)
        self.assertIn("new URLSearchParams(window.location.search).get('job')",page)
        with urllib.request.urlopen(self.url+path) as response:
            deep_link_job=json.load(response)
        self.assertEqual(deep_link_job['status'],'canvas_preview')
        self.assertEqual(deep_link_job['canvas_diagnostics']['exposure_stress']['status'],'pass')
        for key in ('rig_url','canvas_preview_html','canvas_seam_preview','canvas_live_rig_url','canvas_diagnostics_url','canvas_all_layers_url'):
            with urllib.request.urlopen(self.url+deep_link_job[key]) as response:
                self.assertEqual(response.status,200)
                self.assertTrue(response.read(1))
        head=next(p for p in job['canvas_layers'] if p['name']=='head')
        self.assertEqual(head['label'],'角色头部');self.assertEqual(head['role'],'head')
        self.assertTrue(head['motion_adjustable']);self.assertEqual(head['seam_crops_url'],job['canvas_seam_preview'])
        body=next(p for p in job['canvas_layers'] if p['name']=='body')
        self.assertRegex(body['label'],r'[\u4e00-\u9fff]')
        with urllib.request.urlopen(self.url+head['thumb_url']) as response:
            self.assertEqual(response.read(8),b'\x89PNG\r\n\x1a\n')
        status,job=self.post(path+'/layer-tweaks',{'layers':{'head':{'drawOrder':0},'body':{'drawOrder':2},'flame':{'effect_mode':'omit'}},'modules':{'head':{'motion_scale':.5}},'global_motion_scale':1.8})
        self.assertEqual(status,200,job);self.assertFalse(next(p for p in job['canvas_layers'] if p['name']=='flame')['enabled'])
        self.assertEqual(next(p for p in job['canvas_layers'] if p['name']=='head')['baked_motion_scale'],.9)
        self.assertEqual(job['canvas_global_motion_scale'],1.8)
        with urllib.request.urlopen(self.url+job['canvas_live_rig_url']) as response:
            self.assertIn('flame',[slot['name'] for slot in json.load(response)['slots']])
        with urllib.request.urlopen(self.url+job['rig_url']) as response:
            self.assertNotIn('flame',[slot['name'] for slot in json.load(response)['slots']])
        status,_=self.post(path+'/user-signoff',{'decision':'pass','expected_rig_version':1})
        self.assertEqual(status,409)
        # Hiding a layer can expose missing original pixels; re-enable before signing.
        status,job=self.post(path+'/layer-tweaks',{'layers':{'flame':{'enabled':True}}})
        self.assertEqual(status,200,job)
        status,job=self.post(path+'/user-signoff',{'decision':'pass','reviewer':'test-human'})
        self.assertEqual(status,200,job);self.assertEqual(job['status'],'ready')
        with urllib.request.urlopen(self.url+path) as response:
            self.assertEqual(json.load(response)['status'],'ready')
        status,record=self.post(path+'/export',{'kind':'spine_package'});self.assertEqual(status,200,record)
        self.assertEqual(len(record['files']),4)
        for file in record['files']:
            with urllib.request.urlopen(self.url+file['url']) as response:self.assertGreater(len(response.read()),0)
    def test_wrong_source_rejected(self):
        from PIL import Image
        import io
        buf=io.BytesIO();Image.new('RGBA',(64,64),(255,0,0,255)).save(buf,format='PNG')
        _,job=self.post('/api/jobs',{'filename':'other.png','image_base64':base64.b64encode(buf.getvalue()).decode()})
        status,error=self.post('/api/jobs/'+job['job_id']+'/apply-canvas-layers',{'parts_dir':str(self.parts),'plan':str(self.planpath)})
        self.assertEqual(status,400,error)
