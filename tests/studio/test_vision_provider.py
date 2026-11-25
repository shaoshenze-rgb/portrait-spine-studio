"""Explicit fake loopback transport tests; not evidence of live Cindy recognition."""
from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

from src.studio.server import StudioJobManager, create_studio_server
from src.studio.vision_provider import CindyVisionProvider, VisionProviderError
from src.studio.vision_schema import VisionReviewRequired

ROOT = Path(__file__).resolve().parents[2]


def make_result(request):
    return {
        'version': 1, 'input_sha256': request['inputSha256'],
        'image_size': [request['width'], request['height']],
        'coord_mode': 'normalized', 'pose_type': 'airborne', 'uncertainties': [],
        'joints': {'head_top': [.5, .08], 'neck': [.5, .25], 'chest': [.5, .36], 'hips': [.5, .58],
                   'shoulder_R': [.4, .31], 'elbow_R': [.3, .42], 'wrist_R': [.24, .55],
                   'shoulder_L': [.6, .31], 'elbow_L': [.7, .42], 'wrist_L': [.76, .55],
                   'hip_R': [.44, .6], 'knee_R': [.42, .75], 'ankle_R': [.4, .9],
                   'hip_L': [.56, .6], 'knee_L': [.58, .75], 'ankle_L': [.6, .9]},
        'entities': [{'id': 'subject', 'kind': 'subject', 'confidence': .95,
                      'polygon': [[0, 0], [1, 0], [1, 1], [0, 1]]}],
    }


def completed(request):
    return {**{k: request[k] for k in ('jobId', 'inputSha256', 'width', 'height')},
            'source': 'actual_cindy_agent', 'status': 'completed', 'taskId': 'test-task', 'runId': 'test-run',
            'executionReceipt': {'taskId': 'test-task', 'runId': 'test-run', 'outputMessageId': 'test-message',
                                 'acceptedConfig': {'model': 'explicit-test-fixture'},
                                 'execution': {'status': 'completed', 'instanceId': 'test-engine', 'generation': 1}},
            'result': make_result(request)}


class TestVisionProvider(unittest.TestCase):
    def setUp(self):
        tmp_root = ROOT / '.studio-work' / 'test-tmp'
        tmp_root.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=tmp_root)
        self.root = Path(self.tmp.name)
        self.input = self.root / 'input.png'
        im = Image.new('RGBA', (120, 180), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.ellipse((44, 15, 76, 48), fill=(220, 170, 150, 255))
        d.rectangle((42, 45, 78, 110), fill=(40, 140, 160, 255))
        d.line([(47, 58), (28, 98)], fill=(220, 170, 150, 255), width=15)
        d.line([(73, 58), (92, 98)], fill=(220, 170, 150, 255), width=15)
        d.rectangle((43, 105, 55, 165), fill=(70, 80, 120, 255))
        d.rectangle((65, 105, 77, 165), fill=(70, 80, 120, 255))
        im.save(self.input)
        self.posts = []
        self.response_transform = lambda response: response
        self.token = 'explicit-test-token-01234567890123456789'
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                if self.headers.get('Authorization') != 'Bearer ' + owner.token:
                    self.send_error(401)
                    return
                request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                owner.posts.append(request)
                response = owner.response_transform(completed(request))
                data = json.dumps(response).encode()
                self.send_response(202)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.config = self.root / 'vision-bridge' / 'connection.json'
        self.config.parent.mkdir()
        self.config.write_text(json.dumps({'schemaVersion': 1, 'protocol': 'portrait-vision-v1',
                                          'baseUrl': f'http://127.0.0.1:{self.server.server_port}',
                                          'instanceId': 'test-instance', 'token': self.token}))
        self.config.chmod(0o600)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.tmp.cleanup()

    def provider(self):
        return CindyVisionProvider(self.config, timeout_s=1, poll_interval_s=0)

    def analyze(self, provider=None, dirname='vision'):
        return (provider or self.provider()).analyze(job_id='job1', image_path=self.input,
                                                   out_dir=self.root / dirname, preset='agile_full')

    def test_bound_completed_response_preserves_actual_receipt(self):
        result = self.analyze()
        self.assertEqual(result.source, 'actual_cindy_agent')
        self.assertEqual(result.metadata['input_sha256'], hashlib.sha256(self.input.read_bytes()).hexdigest())
        self.assertEqual(result.metadata['output_message_id'], 'test-message')
        self.assertEqual(base64.b64decode(self.posts[0]['imageBase64']), self.input.read_bytes())
        persisted = (self.root / 'vision' / 'response.json').read_text()
        self.assertNotIn(self.token, persisted)

    def test_wrong_identity_or_source_never_passes(self):
        for key, value in [('jobId', 'other'), ('inputSha256', '0' * 64), ('width', 999),
                           ('height', 999), ('source', 'provided_hints')]:
            with self.subTest(key=key):
                self.response_transform = lambda r, k=key, v=value: {**r, k: v}
                with self.assertRaises(VisionProviderError):
                    self.analyze(dirname=key)

    def test_missing_execution_receipt_and_low_confidence_stop(self):
        self.response_transform = lambda r: {**r, 'executionReceipt': {}}
        with self.assertRaises(VisionProviderError) as err:
            self.analyze(dirname='no-receipt')
        self.assertEqual(err.exception.code, 'VISION_RECEIPT_MISSING')
        def uncertain(r):
            r['result']['entities'][0]['confidence'] = .3
            return r
        self.response_transform = uncertain
        with self.assertRaises(VisionReviewRequired):
            self.analyze(dirname='low-confidence')
        self.assertTrue((self.root / 'low-confidence' / 'response.json').is_file())

    def test_unknown_submit_retry_only_reads_same_job(self):
        provider = self.provider()
        methods = []
        digest = hashlib.sha256(self.input.read_bytes()).hexdigest()
        binding = {'jobId': 'job1', 'inputSha256': digest, 'width': 120, 'height': 180}
        def uncertain(cfg, method, path, body=None):
            methods.append((method, path))
            raise VisionProviderError('transport', code='VISION_BRIDGE_TRANSPORT')
        with patch.object(provider, '_request', side_effect=uncertain):
            with self.assertRaises(VisionProviderError):
                self.analyze(provider)
        self.assertEqual([m for m, _ in methods], ['POST', 'GET'])
        methods.clear()
        def recovered(cfg, method, path, body=None):
            methods.append((method, path))
            return completed(binding)
        with patch.object(provider, '_request', side_effect=recovered):
            self.analyze(provider)
        self.assertEqual(methods, [('GET', '/v1/jobs/job1')])

    def test_configuration_must_be_private_and_loopback(self):
        self.config.chmod(0o644)
        with self.assertRaises(VisionProviderError) as err:
            self.analyze()
        self.assertEqual(err.exception.code, 'VISION_CONNECTION_INSECURE')
        self.config.chmod(0o600)
        cfg = json.loads(self.config.read_text())
        cfg['baseUrl'] = 'http://example.com:9000'
        self.config.write_text(json.dumps(cfg))
        with self.assertRaises(VisionProviderError) as err:
            self.analyze()
        self.assertEqual(err.exception.code, 'VISION_CONNECTION_INVALID')
        self.assertEqual(self.posts, [])

    def test_explicit_bridge_injection_missing_connection_fails_before_split(self):
        manager = StudioJobManager(self.root / 'jobs', vision_provider=CindyVisionProvider(self.root / 'missing.json'))
        job = manager.create_job({'image_base64': base64.b64encode(self.input.read_bytes()).decode()})
        self.assertIsInstance(manager.vision_provider, CindyVisionProvider)
        self.assertEqual(job['status'], 'failed')
        self.assertEqual(job['stage_error']['failedStage'], 'vision')
        self.assertEqual(job['split_run_count'], 0)
        self.assertEqual(job['semantic_source'], 'pending')

    def test_manager_uses_bound_live_result_and_keeps_binding_pause(self):
        # The optional transport boundary must now supply complete V2 modules, not
        # an old subject polygon. These are still explicit fake transport fixtures.
        from tests.pipeline.test_complete_modules import fixture
        contract = fixture(self.input)
        def module_response(response):
            value = copy.deepcopy(contract)
            value.update(input_sha256=response['inputSha256'], image_size=[response['width'], response['height']])
            response['result'] = value
            return response
        self.response_transform = module_response
        manager = StudioJobManager(self.root / 'jobs', vision_provider=self.provider())
        job = manager.create_job({'image_base64': base64.b64encode(self.input.read_bytes()).decode()})
        self.assertEqual(job['status'], 'awaiting_rig_confirm', job.get('stage_error'))
        self.assertEqual(job['semantic_source'], 'actual_cindy_agent')
        self.assertEqual(job['vision_metadata']['task_id'], 'test-task')
        self.assertEqual(job['vision_metadata']['input_sha256'], self.posts[0]['inputSha256'])
        self.assertEqual(job['rig_build_count'], 0)

    def test_v1_transport_result_is_not_a_complete_module_plan(self):
        manager = StudioJobManager(self.root / 'jobs', vision_provider=self.provider())
        job = manager.create_job({'image_base64': base64.b64encode(self.input.read_bytes()).decode()})
        self.assertEqual(job['status'], 'failed')
        self.assertEqual(job['stage_error']['errorCode'], 'MODULE_CONTRACT_REQUIRED')
        self.assertEqual(job['split_run_count'], 0)

    def test_opaque_jpeg_reaches_matting_but_never_bypasses_it(self):
        buf = io.BytesIO()
        Image.new('RGB', (60, 90), (100, 120, 140)).save(buf, format='JPEG')
        manager = StudioJobManager(self.root / 'jobs', vision_provider=self.provider())
        job = manager.create_job({'filename': 'portrait.jpeg', 'image_base64': base64.b64encode(buf.getvalue()).decode()})
        self.assertEqual(job['status'], 'failed')
        self.assertEqual(job['stage_error']['failedStage'], 'mivo')
        self.assertEqual(self.posts, [])

    def test_bridge_token_is_not_served_as_job_asset(self):
        manager = StudioJobManager(self.root, vision_provider=self.provider())
        server = create_studio_server(port=0, manager=manager)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with self.assertRaises(urllib.error.HTTPError) as err:
                urllib.request.urlopen(f'http://127.0.0.1:{server.server_port}/jobs/vision-bridge/connection.json')
            self.assertEqual(err.exception.code, 403)
            self.assertNotIn(self.token, err.exception.read().decode())
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == '__main__':
    unittest.main()
