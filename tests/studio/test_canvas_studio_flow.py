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
        status,job=self.post(path+'/layer-tweaks',{'layers':{'head':{'drawOrder':0},'body':{'drawOrder':2},'flame':{'effect_mode':'omit'}},'modules':{'head':{'motion_scale':.5}}})
        self.assertEqual(status,200,job);self.assertFalse(next(p for p in job['canvas_layers'] if p['name']=='flame')['enabled'])
        self.assertEqual(next(p for p in job['canvas_layers'] if p['name']=='head')['motion_scale'],.5)
        # Hiding a layer can expose missing original pixels; re-enable before signing.
        status,job=self.post(path+'/layer-tweaks',{'layers':{'flame':{'enabled':True}}})
        self.assertEqual(status,200,job)
        status,job=self.post(path+'/user-signoff',{'decision':'pass','reviewer':'test-human'})
        self.assertEqual(status,200,job);self.assertEqual(job['status'],'ready')
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
