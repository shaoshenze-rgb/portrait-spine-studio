"""Human signoff exports keep advisories and still reject broken artifacts."""
import json
import unittest
import zipfile
from unittest.mock import patch

from tests.pipeline import test_canvas_layer_rig as fixtures
from src.pipeline.build_rig import build_rig
from src.gates.review_binding import ensure_export_approved, ExportApprovalError
from src.export.exporter import export_default_package, export_standalone_html, export_transparent_web


class UserSignoffExportTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.CanvasLayerRigTests(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        self.root = fixture.root; self.rig = self.root/'rig'
        build_rig(fixture.parts, fixture.compile(), self.rig)
        self.signoff = self.root/'signoff.json'
        self.signoff.write_text(json.dumps({'decision': 'pass', 'user_quote': '验收通过'}))
        self.health = self.rig/'health-report.json'
        self.report = {'hardGates': {'rest_diff': {'passed': True}},
            'quality': {'rulesVersion': 'old', 'overall': {'status': 'pending', 'exportAllowed': False},
                        'bindingMotion': {'checks': {'rigidity': {'status': 'pass'}}},
                        'machineQC': {'checks': {'remainingHiddenCoverage': {'status': 'pending'}}}}}
        self.health.write_text(json.dumps(self.report))

    def test_signed_pending_source_sliced_exports_zip_and_html_without_rewriting_health(self):
        path = self.rig/'rig-build-report.json'; build = json.loads(path.read_text())
        build['moduleContract']['materialMode'] = 'source_sliced'; path.write_text(json.dumps(build))
        before = self.health.read_bytes()
        result = export_default_package(self.rig, self.root/'export', render_previews=False, user_signoff_path=self.signoff)
        with zipfile.ZipFile(result['zip_path']) as archive:
            self.assertIsNone(archive.testzip()); self.assertIn('user-signoff.json', archive.namelist())
        self.assertGreater(__import__('pathlib').Path(result['standalone_html']).stat().st_size, 0)
        self.assertEqual(self.health.read_bytes(), before)
        check = ensure_export_approved(self.rig, user_signoff_path=self.signoff)
        self.assertTrue(check['exportAuthorization']['exportAllowed'])
        self.assertFalse(check['quality']['overall']['exportAllowed'])

    def test_signed_canvas_layers_export_and_direct_html(self):
        result = export_default_package(self.rig, self.root/'export', render_previews=False, user_signoff_path=self.signoff)
        self.assertTrue(result['zip_path'].endswith('.zip'))
        export_standalone_html(self.rig, self.root/'direct.html', user_signoff_path=self.signoff)

    def test_unsigned_failed_or_missing_signoff_rejected(self):
        with self.assertRaises(ExportApprovalError):
            export_default_package(self.rig, self.root/'unsigned', render_previews=False)
        for signoff in ({'decision': 'fail', 'reviewer': 'human'}, {'decision': 'pass', 'reviewer': ' '}):
            self.signoff.write_text(json.dumps(signoff))
            with self.assertRaises(ExportApprovalError):
                export_default_package(self.rig, self.root/'failed', render_previews=False, user_signoff_path=self.signoff)
        with self.assertRaises(ExportApprovalError):
            ensure_export_approved(self.rig, user_signoff_path=self.root/'missing.json')

    def test_broken_rest_or_rigid_still_blocks_signed_export(self):
        for case in ('missing_rest', 'failed_rest', 'rigid_fail'):
            report = json.loads(json.dumps(self.report))
            if case == 'missing_rest': report['hardGates'] = {}
            elif case == 'failed_rest': report['hardGates']['rest_diff']['passed'] = False
            else: report['quality']['bindingMotion']['checks']['rigidity']['status'] = 'fail'
            self.health.write_text(json.dumps(report))
            with self.subTest(case=case), self.assertRaises(ExportApprovalError):
                ensure_export_approved(self.rig, user_signoff_path=self.signoff)

    def test_missing_texture_blocks_signed_export(self):
        next((self.rig/'images').glob('*.png')).write_bytes(b'')
        with self.assertRaises(ExportApprovalError):
            ensure_export_approved(self.rig, user_signoff_path=self.signoff)

    def test_transparent_export_passes_signoff_to_gate(self):
        # Stop at codec discovery: this verifies plumbing without running AV1.
        with patch('src.export.exporter.find_ffmpeg', return_value=None):
            with self.assertRaisesRegex(RuntimeError, 'ffmpeg binary required'):
                export_transparent_web(self.rig, self.root/'web', user_signoff_path=self.signoff)


if __name__ == '__main__': unittest.main()
