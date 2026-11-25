"""Real V2 pipeline + review/export boundary tests; synthetic observations are not portrait approval."""
import base64
import copy
import json
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image

from src.export.exporter import export_default_package, export_transparent_web
from src.gates.review_binding import ExportApprovalError, VISUAL_DIMENSIONS
from src.studio.server import StudioJobManager, InvalidJobStateError, create_studio_server
from src.studio.nl_adjust import classify_and_polish_adjustment
from src.render.renderer import find_ffmpeg
from tests.pipeline.test_complete_modules import fixture
from scripts.inspect_module_plan import inspect_plan

ROOT = Path(__file__).resolve().parents[2]


class TestCompleteModuleFlow(unittest.TestCase):
    def setUp(self):
        base = ROOT / '.studio-work/module-integration-tests'
        base.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=base)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root / 'source.png'
        self.plan = fixture(self.image)
        self.manager = StudioJobManager(self.root / 'jobs', audit_samples=6, stress_samples=4)

    def prepare(self):
        job = self.manager.create_job({'filename': 'source.png', 'preset': 'bust_half',
                                      'image_base64': base64.b64encode(self.image.read_bytes()).decode()})
        self.plan.update(job_id=job['job_id'], input_sha256=job['analysis_request']['input_sha256'],
                         image_size=job['analysis_request']['image_size'])
        return job

    def generate(self):
        job = self.prepare()
        bound = self.manager.apply_analysis_job(job['job_id'], {'job_id': job['job_id'], 'analysis': self.plan})
        self.assertEqual(bound['status'], 'awaiting_rig_confirm', bound.get('stage_error'))
        generated = self.manager.continue_job(job['job_id'], {'module_review': self.module_decision(bound)})
        self.assertEqual(generated['status'], 'awaiting_visual_review', generated.get('stage_error'))
        for group in ('inputOwnership', 'bindingMotion', 'machineQC'):
            self.assertEqual(generated['quality'][group]['status'], 'pass')
        return generated

    def module_decision(self, job):
        return {'source': 'current_cindy_conversation', 'reviewer': 'synthetic integration fixture',
                'original_and_modules_viewed': True, 'binding': copy.deepcopy(job['module_review_binding']),
                'modules': {m['id']: {'status': 'pass', 'notes': 'Synthetic semantic fixture, not real portrait approval: ' + m['id']}
                            for m in job['modules']}}

    def decision(self, job, status='pass'):
        return {'decision': status, 'source': 'current_cindy_conversation', 'reviewer': 'synthetic integration fixture',
                'notes': 'Acceptance API regression fixture, not a real portrait visual approval.',
                'full_loop_observed': True, 'original_and_modules_viewed': True,
                'observations': {key: {'status': status, 'notes': 'Synthetic fixture invariant: ' + key}
                                 for key in VISUAL_DIMENSIONS},
                'review_binding': copy.deepcopy(job['review_binding'])}

    def test_v2_default_and_v1_never_silently_compile(self):
        job = self.prepare()
        self.assertEqual(job['analysis_request']['required_contract_version'], 2)
        bad = copy.deepcopy(self.plan); bad['version'] = 1
        failed = self.manager.apply_analysis_job(job['job_id'], {'job_id': job['job_id'], 'analysis': bad})
        self.assertEqual(failed['status'], 'failed')
        self.assertEqual(failed['stage_error']['errorCode'], 'MODULE_CONTRACT_REQUIRED')
        self.assertEqual(failed['split_run_count'], 0)
        self.assertIsNone(failed['rig_url'])

    def test_located_input_defect_preserves_reason_and_stops_before_split(self):
        job = self.prepare()
        defect = {'id': 'bottom_line', 'kind': 'background_residue',
                  'polygons': [[[0, .95], [1, .95], [1, 1], [0, 1]]],
                  'notes': 'A separate white border spans the bottom of this input.'}
        self.plan['input_issues'] = [defect]
        failed = self.manager.apply_analysis_job(job['job_id'], {'job_id': job['job_id'], 'analysis': self.plan})
        self.assertEqual(failed['stage_error']['errorCode'], 'INPUT_REVIEW_REQUIRED')
        self.assertEqual(failed['stage_error']['details']['input_issues'][0]['notes'], defect['notes'])
        self.assertEqual(failed['split_run_count'], 0)
        self.assertEqual(failed['rig_build_count'], 0)
        self.assertIsNone(failed['rig_url'])
        self.assertTrue(failed['matted_url'])

    def test_binding_confirmation_requires_all_current_module_observations(self):
        job = self.prepare()
        bound = self.manager.apply_analysis_job(job['job_id'], {'job_id': job['job_id'], 'analysis': self.plan})
        with self.assertRaises(ExportApprovalError): self.manager.continue_job(job['job_id'])
        incomplete = self.module_decision(bound); incomplete['modules'].pop(next(iter(incomplete['modules'])))
        with self.assertRaises(ExportApprovalError): self.manager.continue_job(job['job_id'], {'module_review': incomplete})
        pending = self.module_decision(bound); pending['modules'][next(iter(pending['modules']))]['status'] = 'pending'
        with self.assertRaises(ExportApprovalError): self.manager.continue_job(job['job_id'], {'module_review': pending})
        stale = self.module_decision(bound); stale['binding']['inspection_sha256'] = '0' * 64
        with self.assertRaises(ExportApprovalError): self.manager.continue_job(job['job_id'], {'module_review': stale})
        self.assertEqual(self.manager.get_job(job['job_id'])['rig_build_count'], 0)
        self.assertEqual(self.manager.get_job(job['job_id'])['status'], 'awaiting_rig_confirm')
        good = self.module_decision(bound)
        comparison = self.root / 'jobs' / job['job_id'] / 'review-modules' / (bound['modules'][0]['id'] + '-comparison.jpg')
        comparison.write_bytes(comparison.read_bytes() + b'changed')
        with self.assertRaises(ExportApprovalError): self.manager.continue_job(job['job_id'], {'module_review': good})

    def test_actual_machine_pass_still_requires_structured_visual_review(self):
        job = self.generate()
        self.assertEqual(job['quality']['visual']['status'], 'pending')
        self.assertFalse(job['quality']['overall']['exportAllowed'])
        rig = self.root / 'jobs' / job['job_id'] / 'rig'
        with self.assertRaises(ExportApprovalError): export_default_package(rig, self.root / 'blocked')
        with self.assertRaises(ExportApprovalError): export_transparent_web(rig, self.root / 'blocked-web')
        self.assertFalse((self.root / 'blocked').exists())
        with self.assertRaises(InvalidJobStateError): self.manager.export_job(job['job_id'], {})
        bad = self.decision(job); bad.pop('observations')
        with self.assertRaises(ExportApprovalError): self.manager.review_job(job['job_id'], bad)
        bad = self.decision(job); bad['original_and_modules_viewed'] = False
        with self.assertRaises(ExportApprovalError): self.manager.review_job(job['job_id'], bad)
        approved = self.manager.review_job(job['job_id'], self.decision(job))
        self.assertEqual(approved['status'], 'ready')
        self.assertTrue(approved['quality']['overall']['exportAllowed'])
        result = self.manager.export_job(job['job_id'], {})
        names = [f['filename'] for f in result['files']]
        self.assertTrue(any(n.endswith('.zip') for n in names))
        self.assertTrue(any(n.endswith('.html') for n in names))

    def test_module_comparison_links_require_current_source_and_contract(self):
        job = self.prepare()
        bound = self.manager.apply_analysis_job(job['job_id'], {'job_id': job['job_id'], 'analysis': self.plan})
        directory = self.root / 'jobs' / job['job_id']
        self.assertTrue(all(m['comparison_url'] for m in bound['modules']))
        inspect_plan(directory / 'matted.png', directory / 'vision/vision.json', directory / 'review-modules')
        viewed = self.manager.get_job(job['job_id'])
        self.assertTrue(all(m['comparison_url'] for m in viewed['modules']))
        self.assertIsNone(viewed['visual_review'])
        manifest_path = directory / 'review-modules/module-review.json'
        original = json.loads(manifest_path.read_text())
        for key in ('source_sha256', 'contract_sha256'):
            changed = copy.deepcopy(original); changed[key] = '0' * 64
            manifest_path.write_text(json.dumps(changed))
            stale = self.manager.get_job(job['job_id'])
            self.assertTrue(all(m['comparison_url'] is None for m in stale['modules']))
        manifest_path.write_text(json.dumps(original))
        first = viewed['modules'][0]['id']
        (directory / 'review-modules' / (first + '-comparison.jpg')).unlink()
        missing = self.manager.get_job(job['job_id'])
        self.assertIsNone(next(m['comparison_url'] for m in missing['modules'] if m['id'] == first))
        self.assertTrue(any(m['comparison_url'] for m in missing['modules'] if m['id'] != first))
        source_path = directory / 'matted.png'; original_source = source_path.read_bytes()
        with Image.open(source_path) as im:
            changed = im.convert('RGBA'); changed.putpixel((0, 0), (251, 7, 13, 255)); changed.save(source_path)
        stale = self.manager.get_job(job['job_id'])
        self.assertTrue(all(m['comparison_url'] is None for m in stale['modules']))
        source_path.write_bytes(original_source)
        analysis_path = directory / 'vision/vision.json'
        analysis_path.write_text(analysis_path.read_text() + ' ')
        stale = self.manager.get_job(job['job_id'])
        self.assertTrue(all(m['comparison_url'] is None for m in stale['modules']))

    def test_cli_rejects_stale_service_rules_before_submitting_input(self):
        health = self.manager.health
        self.manager.health = lambda: dict(health(), module_qc_rules_version='stale-service')
        server = create_studio_server(port=0, manager=self.manager)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        self.addCleanup(lambda: (server.shutdown(), server.server_close(), thread.join()))
        url = 'http://127.0.0.1:' + str(server.server_port)
        run = subprocess.run([sys.executable, str(ROOT / 'scripts/portrait_task.py'), '--base-url', url,
                              'prepare', '--image', str(self.image)], capture_output=True, text=True)
        self.assertEqual(run.returncode, 1, run.stdout + run.stderr)
        self.assertIn('旧QC规则', json.loads(run.stdout)['message'])
        self.assertFalse(self.manager._jobs)

    def test_cli_http_generate_structured_review_and_export(self):
        server = create_studio_server(port=0, manager=self.manager)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        self.addCleanup(lambda: (server.shutdown(), server.server_close(), thread.join()))
        url = 'http://127.0.0.1:' + str(server.server_port)
        def cli(*args):
            run = subprocess.run([sys.executable, str(ROOT / 'scripts/portrait_task.py'), '--base-url', url, *args], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            return json.loads(run.stdout)
        job = cli('prepare', '--image', str(self.image), '--preset', 'bust_half')
        self.plan.update(job_id=job['job_id'], input_sha256=job['analysis_request']['input_sha256'], image_size=job['analysis_request']['image_size'])
        analysis = self.root / 'analysis.json'; analysis.write_text(json.dumps(self.plan))
        bound = cli('apply-analysis', '--job', job['job_id'], '--analysis', str(analysis))
        self.assertEqual(bound['status'], 'awaiting_rig_confirm')
        module_review = self.root / 'module-observations.json'; module_review.write_text(json.dumps(self.module_decision(bound)))
        generated = cli('generate', '--job', job['job_id'], '--module-review', str(module_review))
        self.assertEqual(generated['status'], 'awaiting_visual_review')
        target = self.root / 'review-target.json'; target.write_text(json.dumps(generated))
        decision = self.decision(generated)
        notes = self.root / 'observed.txt'; notes.write_text(decision['notes'])
        observations = self.root / 'observations.json'; observations.write_text(json.dumps(decision['observations']))
        reviewed = cli('review', '--job', job['job_id'], '--decision', 'pass', '--reviewer', 'synthetic API fixture',
                       '--notes-file', str(notes), '--binding-file', str(target), '--observations-file', str(observations),
                       '--full-loop-observed', '--original-and-modules-viewed')
        self.assertEqual(reviewed['status'], 'ready')
        req = urllib.request.Request(url + '/api/jobs/' + job['job_id'] + '/export', data=b'{}', headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req) as response: exported = json.load(response)
        with urllib.request.urlopen(url + exported['files'][0]['url']) as response:
            self.assertTrue(response.read().startswith(b'PK'))

    def test_approved_v2_package_and_web_codecs_keep_validation(self):
        job = self.generate(); self.manager.review_job(job['job_id'], self.decision(job))
        rig = self.root / 'jobs' / job['job_id'] / 'rig'
        ff = find_ffmpeg()
        package = export_default_package(rig, self.root / 'package', render_previews=bool(ff),
                                         preview_fps=10, preview_max_frames=4, ffmpeg_bin=ff)
        with zipfile.ZipFile(package['zip_path']) as archive:
            self.assertIn('health-report.json', archive.namelist())
            self.assertTrue(any(name.endswith('.spine.json') for name in archive.namelist()))
        if ff:
            self.assertTrue(Path(package['previews']['dark']).is_file())
            self.assertTrue(Path(package['previews']['light']).is_file())
            self.assertFalse((self.root / 'package/idle-packed-h264.mp4').exists())
            web = export_transparent_web(rig, self.root / 'web', fps=10, max_frames=4, ffmpeg_bin=ff)
            for key in ('packed_h264_mp4', 'alpha_vp9_webm', 'poster_webp'):
                self.assertTrue(Path(web[key]).is_file())

    def test_source_plan_and_rig_change_revoke_direct_export(self):
        for kind in ('source', 'plan', 'rig'):
            with self.subTest(kind=kind):
                job = self.generate(); self.manager.review_job(job['job_id'], self.decision(job))
                directory = self.root / 'jobs' / job['job_id']; rig = directory / 'rig'
                if kind == 'source':
                    path = directory / 'matted.png'
                    with Image.open(path) as image:
                        changed = image.convert('RGBA'); changed.putpixel((0, 0), (255, 0, 0, 255)); changed.save(path)
                elif kind == 'plan':
                    path = directory / 'vision/vision.json'; value = json.loads(path.read_text())
                    value['modules'][0]['notes'] = 'unreviewed alteration'; path.write_text(json.dumps(value))
                else:
                    path = next(rig.glob('*.spine.json')); path.write_text(path.read_text() + ' ')
                with self.assertRaises(ExportApprovalError): export_default_package(rig, self.root / ('bad-' + kind))

    def test_visual_pass_cannot_override_machine_pending_or_fail(self):
        job = self.generate()
        stored = self.manager._jobs[job['job_id']]
        for status in ('pending', 'fail'):
            stored['health_report']['quality']['machineQC']['status'] = status
            with self.assertRaises(InvalidJobStateError): self.manager.review_job(job['job_id'], self.decision(job))

    def test_previous_qc_rules_cannot_reuse_visual_export_approval(self):
        job = self.generate(); self.manager.review_job(job['job_id'], self.decision(job))
        rig = self.root / 'jobs' / job['job_id'] / 'rig'
        report_path = rig / 'health-report.json'
        report = json.loads(report_path.read_text())
        report['quality']['rulesVersion'] = 'old-rules-before-grip-pose-check'
        report_path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ExportApprovalError, 'QC规则版本'):
            export_default_package(rig, self.root / 'stale-qc')

    def test_real_texture_change_invalidates_review_and_both_exporters(self):
        job = self.generate()
        approved = self.manager.review_job(job['job_id'], self.decision(job))
        rig = self.root / 'jobs' / job['job_id'] / 'rig'
        image = next((rig / 'images').glob('*.png'))
        with Image.open(image) as im:
            changed = im.convert('RGBA'); changed.putpixel((0, 0), (251, 7, 13, 255)); changed.save(image)
        with self.assertRaises(InvalidJobStateError): self.manager.export_job(job['job_id'], {})
        with self.assertRaises(ExportApprovalError): export_default_package(rig, self.root / 'changed')
        with self.assertRaises(ExportApprovalError): export_transparent_web(rig, self.root / 'changed-web')
        with self.assertRaises(InvalidJobStateError): self.manager.review_job(job['job_id'], self.decision(approved))

    def test_new_analysis_clears_old_review_without_host_receipt(self):
        job = self.generate()
        self.manager.review_job(job['job_id'], self.decision(job))
        changed = copy.deepcopy(self.plan); changed['modules'][0]['notes'] = 'Fresh semantic plan revision'
        fresh = self.manager.apply_analysis_job(job['job_id'], {'job_id': job['job_id'], 'analysis': changed})
        self.assertEqual(fresh['status'], 'awaiting_rig_confirm')
        self.assertIsNone(fresh['visual_review'])
        self.assertIsNone(fresh['review_binding'])
        self.assertEqual(fresh['exports'], {})
        self.assertNotIn('execution_receipt', fresh['vision_metadata'])

    def test_failed_rebuild_revokes_previous_memory_and_disk_approval(self):
        job = self.generate()
        approved = self.manager.review_job(job['job_id'], self.decision(job))
        self.manager.export_job(job['job_id'], {})
        self.assertTrue(approved['quality']['overall']['exportAllowed'])
        # An accepted public motion change can still encounter a downstream file failure.
        # This changes a temporary fixture, not the production code or QC thresholds.
        directory = self.root / 'jobs' / job['job_id']
        parts_path = directory / 'parts_out/parts.json'
        parts = json.loads(parts_path.read_text())
        parts['parts'][0]['image'] = 'missing-fixture-piece.png'
        parts_path.write_text(json.dumps(parts))
        failed = self.manager.adjust_job(job['job_id'], {'gain': 1.1})
        self.assertEqual(failed['status'], 'failed')
        self.assertIsNone(failed['quality'])
        self.assertIsNone(failed['visual_review'])
        self.assertIsNone(failed['review_binding'])
        self.assertIsNone(failed['rig_url'])
        self.assertEqual(failed['exports'], {})
        report = json.loads((directory / 'rig/health-report.json').read_text())
        self.assertFalse(report['quality']['overall']['exportAllowed'])
        self.assertTrue(list((directory / 'history').glob('health-report-*.json')))
        with self.assertRaises(ExportApprovalError): export_default_package(directory / 'rig', self.root / 'stale')

    def test_module_natural_language_never_freezes_or_inflates_fills(self):
        rig = {'moduleContract': {'sourceVersion': 2}}
        for text in ('左手固定别动', '补全破洞', '尾巴小一点'):
            with self.assertRaises(ValueError): classify_and_polish_adjustment(text, rig_spec=rig)
        result = classify_and_polish_adjustment('整体更柔和', rig_spec=rig)
        self.assertGreater(result['spec_patch']['gain'], 0)
        self.assertEqual(result['spec_patch']['rig_motion_overrides'], {})
        self.assertEqual(result['spec_patch']['parts_updates'], [])


if __name__ == '__main__': unittest.main()
