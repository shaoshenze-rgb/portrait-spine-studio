import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock,patch

import numpy as np
from PIL import Image
from scripts import canvas_handoff as handoff

ROOT=Path(__file__).resolve().parents[2]


class CanvasHandoffTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=ROOT/'.studio-work',prefix='canvas-handoff-')
        self.addCleanup(self.tmp.cleanup);self.d=Path(self.tmp.name)
        self.source=self.d/'source.png';Image.new('RGBA',(80,80),(30,60,90,255)).save(self.source)
        self.assets=self.d/'raw';self.assets.mkdir();self.layer=self.assets/'layer-one.png'
        Image.new('RGBA',(80,80),(30,60,90,255)).save(self.layer)
        (self.assets/'base.png').write_bytes(self.layer.read_bytes())
        self.receipt=self.d/'receipt.json'
        self.data={'version':1,'operation':'split-layers','scene_id':'synthetic','source_node_id':'root-node',
                   'source_sha256':handoff.sha(self.source),'task_id':'synthetic-only','base_node_id':'base',
                   'nodes':[{'node_id':'base','asset_sha256':handoff.sha(self.layer)},
                            {'node_id':'layer-one','title':'Test layer','asset_sha256':handoff.sha(self.layer)}]}
        self.receipt.write_text(json.dumps(self.data));self.out=self.d/'out'

    def invoke(self,**kwargs):
        return handoff.import_layers(self.receipt,self.source,self.source,self.assets,self.out,**kwargs)

    def test_checked_result_has_receipt_hash_transform_and_unreviewed_status(self):
        identity=np.array([[1.,0,0],[0,1.,0]])
        with patch.object(handoff,'register_layer',return_value=(identity,{'inliers':20})),patch.object(handoff,'warp_rgba',side_effect=lambda layer,matrix,size:layer):
            report=self.invoke()
        self.assertEqual(report['status'],'needs_layer_review')
        self.assertEqual(report['layers'][0]['semantic_review'],'unreviewed')
        self.assertEqual(report['layers'][0]['raw_sha256'],handoff.sha(self.layer))
        self.assertEqual(report['layers'][0]['sha256'],handoff.sha(Path(report['layers'][0]['image'])))
        self.assertFalse((self.out/'CANVAS_IMPORT_PENDING').exists())

    def test_failed_registration_does_not_fabricate_position(self):
        with patch.object(handoff,'register_layer',return_value=(None,{'reason':'low confidence'})):
            report=self.invoke()
        self.assertEqual(report['layers'][0]['status'],'needs_canvas_registration')
        self.assertNotIn('image',report['layers'][0])

    def test_edit_image_single_result_keeps_real_operation_without_fake_base(self):
        self.data.update(operation='edit-image',nodes=[self.data['nodes'][1]])
        self.data.pop('base_node_id');self.receipt.write_text(json.dumps(self.data))
        identity=np.array([[1.,0,0],[0,1.,0]])
        with patch.object(handoff,'register_layer',return_value=(identity,{'inliers':20})),patch.object(handoff,'warp_rgba',side_effect=lambda layer,matrix,size:layer):
            result=self.invoke()
        self.assertEqual(result['operation'],'edit-image');self.assertEqual(len(result['layers']),1)
        self.assertEqual(result['layers'][0]['raw_sha256'],handoff.sha(self.layer))
        self.assertEqual(result['layers'][0]['semantic_review'],'unreviewed')

    def test_import_operation_and_result_shape_must_agree(self):
        for case in ('edit_with_base','edit_two_results','split_without_base','unknown_operation'):
            data=json.loads(json.dumps(self.data))
            if case=='edit_with_base':data.update(operation='edit-image',nodes=[data['nodes'][1]])
            elif case=='edit_two_results':data['operation']='edit-image';data.pop('base_node_id')
            elif case=='split_without_base':data.pop('base_node_id')
            else:data['operation']='invented'
            self.receipt.write_text(json.dumps(data))
            with self.subTest(case=case),self.assertRaises(ValueError):self.invoke()
            self.assertTrue((self.out/'CANVAS_IMPORT_FAILED').exists())

    def test_raw_hash_and_filename_traversal_rejected(self):
        for case in ('hash','path'):
            data=json.loads(self.receipt.read_text())
            data['nodes'][1]['asset_sha256']='0'*64 if case=='hash' else handoff.sha(self.layer)
            if case=='path':data['nodes'][1]['node_id']='../../layer'
            self.receipt.write_text(json.dumps(data))
            with self.subTest(case=case),self.assertRaises(ValueError):self.invoke()
            self.assertTrue((self.out/'CANVAS_IMPORT_FAILED').exists())

    def test_child_alignment_composes_verified_parent_transform(self):
        parent=self.d/'parent.json'
        parent.write_text(json.dumps({'input_sha256':handoff.sha(self.source),'canvas_scene_id':'synthetic',
            'source_asset_sha256':handoff.sha(self.source),'layers':[{'node_id':'root-node','raw_sha256':handoff.sha(self.source),
            'status':'aligned_pending_review','transform':[[.5,0,2],[0,.5,3]]}]}))
        child=np.array([[2.,0,10],[0,2.,20]])
        with patch.object(handoff,'register_layer',return_value=(child,{'inliers':20})),patch.object(handoff,'warp_rgba',side_effect=lambda layer,matrix,size:layer):
            report=self.invoke(parent_report=parent)
        np.testing.assert_allclose(report['layers'][0]['transform'],[[1,0,7],[0,1,13]])
        self.assertEqual(report['lineage']['parent_report_sha256'],handoff.sha(parent))

    def test_failed_import_blocks_stale_report_consumption(self):
        self.out.mkdir();(self.out/'import-report.json').write_text('{}')
        self.receipt.write_text('bad json')
        with self.assertRaises(ValueError):self.invoke()
        with self.assertRaisesRegex(ValueError,'旧结果'):
            handoff.attach_completion(self.d/'absent.json',self.out/'import-report.json','layer-one','a','b',self.d/'result.json')

    def attach_fixture(self):
        from tests.pipeline.test_complete_modules import fixture
        plan=fixture(self.d/'fixture.png',local_fill=True)
        analysis=self.d/'analysis.json';analysis.write_text(json.dumps(plan))
        self.out.mkdir(exist_ok=True)
        material=self.out/'material.png';Image.new('RGBA',(128,160),(73,41,222,255)).save(material)
        report=self.out/'import-report.json'
        report.write_text(json.dumps({'input_sha256':plan['input_sha256'],'image_size':plan['image_size'],
            'canvas_scene_id':'synthetic','source_asset_sha256':plan['input_sha256'],'operation':'split-layers',
            'layers':[{'node_id':'layer-one','status':'aligned_pending_review','image':str(material),
                       'sha256':handoff.sha(material),'raw_sha256':handoff.sha(material),'task_id':'synthetic'}]}))
        oc=plan['occlusions'][0];out=self.d/'attached.json'
        return analysis,report,'layer-one',oc['front'],oc['back'],out

    def test_attach_failure_quarantines_old_output_and_success_recovers(self):
        args=self.attach_fixture();report=args[1];out=args[-1]
        original=report.read_text();data=json.loads(original)
        data['layers'][0]['status']='needs_canvas_registration';report.write_text(json.dumps(data))
        old=b'{"previous":"must not be consumed"}';out.write_bytes(old)
        with self.assertRaisesRegex(ValueError,'尚未可靠对齐'):handoff.attach_completion(*args)
        self.assertFalse(out.exists())
        self.assertEqual(next(self.d.glob('attached.json.prev-success-*')).read_bytes(),old)
        failed=out.with_name(out.name+'.CANVAS_ATTACH_FAILED')
        self.assertTrue(json.loads(failed.read_text())['isolation_complete'])
        report.write_text(original)
        result=handoff.attach_completion(*args)
        self.assertTrue(result['ok']);self.assertTrue(out.exists());self.assertFalse(failed.exists())
        self.assertEqual(json.loads(out.read_text())['occlusions'][0]['completion'],'provided')

    def test_attach_bad_input_and_publish_failure_never_leave_stale_output(self):
        for kind in ('bad_json','write','replace'):
            with self.subTest(kind=kind):
                args=self.attach_fixture();out=args[-1];out.write_text('old '+kind)
                if kind=='bad_json':args[0].write_text('invalid JSON')
                write=Path.write_text;replace=Path.replace
                def guarded_write(path,*a,**kw):
                    if kind=='write' and '.staging-' in path.name:raise OSError('staging write failed')
                    return write(path,*a,**kw)
                def guarded_replace(path,target):
                    if kind=='replace' and '.staging-' in path.name:raise OSError('publish failed')
                    return replace(path,target)
                with patch.object(Path,'write_text',guarded_write),patch.object(Path,'replace',guarded_replace):
                    with self.assertRaises((ValueError,OSError)):handoff.attach_completion(*args)
                self.assertFalse(out.exists())
                self.assertTrue(out.with_name(out.name+'.CANVAS_ATTACH_FAILED').exists())
                self.assertFalse(list(self.d.glob('attached.json.staging-*')))

    def test_attach_busy_does_not_touch_active_output_or_inputs(self):
        args=self.attach_fixture();out=args[-1];out.write_text('active result')
        pending=out.with_name(out.name+'.CANVAS_ATTACH_PENDING');pending.write_text('active-token')
        with self.assertRaises(FileExistsError):handoff.attach_completion(*args)
        self.assertEqual(out.read_text(),'active result');self.assertEqual(pending.read_text(),'active-token')
        self.assertFalse(out.with_name(out.name+'.CANVAS_ATTACH_FAILED').exists())
        with self.assertRaisesRegex(ValueError,'独立文件'):
            handoff.attach_completion(*args[:-1],args[0])
        self.assertIsInstance(json.loads(args[0].read_text()),dict)

    def test_attach_archive_failure_stops_and_reports_incomplete_isolation(self):
        args=self.attach_fixture();out=args[-1];out.write_bytes(b'previous bytes')
        replace=Path.replace
        def blocked(path,target):
            if path==out:raise OSError('archive denied')
            return replace(path,target)
        with patch.object(Path,'replace',blocked),patch.object(handoff,'_attach_completion') as build:
            with self.assertRaisesRegex(OSError,'archive denied'):handoff.attach_completion(*args)
            build.assert_not_called()
        self.assertEqual(out.read_bytes(),b'previous bytes')
        failure=json.loads(out.with_name(out.name+'.CANVAS_ATTACH_FAILED').read_text())
        self.assertFalse(failure['isolation_complete']);self.assertIsNone(failure['analysis'])

    def test_attach_lock_create_or_write_failure_records_incomplete_state(self):
        for phase in ('create','write'):
            with self.subTest(phase=phase):
                args=self.attach_fixture();out=args[-1];out.write_bytes(b'OLD')
                pending=out.with_name(out.name+'.CANVAS_ATTACH_PENDING')
                real_open=Path.open
                def denied(path,mode='r',*a,**kw):
                    if path==pending and mode=='x':
                        if phase=='create':raise OSError('lock create denied')
                        stream=real_open(path,mode,*a,**kw)
                        proxy=MagicMock()
                        proxy.write.side_effect=OSError('lock write denied')
                        proxy.__exit__.side_effect=lambda *unused:stream.close()
                        return proxy
                    return real_open(path,mode,*a,**kw)
                with patch.object(Path,'open',denied),patch.object(handoff,'_attach_completion') as build:
                    with self.assertRaises(handoff.AttachSetupError) as caught:handoff.attach_completion(*args)
                    build.assert_not_called()
                self.assertEqual(out.read_bytes(),b'OLD')
                failure=json.loads(out.with_name(out.name+'.CANVAS_ATTACH_FAILED').read_text())
                self.assertFalse(failure['isolation_complete']);self.assertIsNone(failure['analysis'])
                self.assertTrue(failure['persisted']);self.assertEqual(caught.exception.details,failure)
                self.assertEqual(pending.exists(),phase=='write')
                pending.unlink(missing_ok=True)

    def test_attach_parent_failure_with_unwritable_status_is_reported_by_cli(self):
        args=self.attach_fixture();out=self.d/'cannot-create'/'result.json'
        argv=['canvas_handoff','attach-completion','--analysis',str(args[0]),'--import-report',str(args[1]),
              '--node',args[2],'--front',args[3],'--back',args[4],'--out',str(out)]
        real_mkdir=Path.mkdir
        def denied(path,*a,**kw):
            if path==out.parent:raise PermissionError('parent denied')
            return real_mkdir(path,*a,**kw)
        stdout=io.StringIO()
        with patch.object(Path,'mkdir',denied),patch.object(handoff.sys,'argv',argv),patch('sys.stdout',stdout):
            self.assertEqual(handoff.main(),1)
        payload=json.loads(stdout.getvalue());self.assertFalse(payload['ok'])
        self.assertFalse(payload['details']['persisted']);self.assertFalse(payload['details']['isolation_complete'])
        self.assertIsNone(payload['details']['analysis']);self.assertIn('status_error',payload['details'])
        self.assertFalse(out.exists())


    def _run_multi_split(self, layers_rgba, *, fail_nodes=()):
        """Helper to run import_layers with multiple synthetic aligned RGBA layers."""
        src_arr = np.zeros((100, 100, 4), np.uint8)
        src_arr[10:90, 10:90] = [40, 90, 160, 255]
        src_arr[10:90, 50:90] = [200, 120, 50, 255]
        Image.fromarray(src_arr).save(self.source)
        base_p = self.assets / 'base.png'
        Image.fromarray(src_arr).save(base_p)
        nodes = [{'node_id': 'base', 'asset_sha256': handoff.sha(base_p)}]
        by_id = {}
        for nid, arr in layers_rgba:
            p = self.assets / (nid + '.png')
            Image.fromarray(arr).save(p)
            nodes.append({'node_id': nid, 'title': nid, 'asset_sha256': handoff.sha(p)})
            by_id[nid] = arr
        data = {
            'version': 1, 'operation': 'split-layers', 'scene_id': 'synthetic',
            'source_node_id': 'root-node', 'source_sha256': handoff.sha(self.source),
            'task_id': 'synthetic-multi', 'base_node_id': 'base', 'nodes': nodes,
        }
        self.receipt.write_text(json.dumps(data))
        identity = np.array([[1., 0., 0.], [0., 1., 0.]])
        fail_set = set(fail_nodes)
        call_idx = {'i': 0}

        def fake_register(layer, src):
            nid = layers_rgba[call_idx['i']][0]
            call_idx['i'] += 1
            if nid in fail_set:
                return None, {'reason': 'mock registration fail'}
            return identity, {'inliers': 25}

        with patch.object(handoff, 'register_layer', side_effect=fake_register),              patch.object(handoff, 'warp_rgba', side_effect=lambda layer, matrix, size: layer):
            return handoff.import_layers(self.receipt, self.source, self.source, self.assets, self.out)

    def test_split_quality_gate_passes_clean_two_part_split_and_writes_composite(self):
        left = np.zeros((100, 100, 4), np.uint8); left[10:90, 10:55] = [40, 90, 160, 255]
        right = np.zeros((100, 100, 4), np.uint8); right[10:90, 50:90] = [200, 120, 50, 255]
        report = self._run_multi_split([('left', left), ('right', right)])
        self.assertTrue(report['quality_gate']['passed'], report['quality_gate'])
        self.assertEqual(report['quality_gate']['verdict'], 'pass')
        self.assertTrue(Path(report['composite_comparison']).exists())

    def test_split_quality_gate_rejects_missing_coverage(self):
        left = np.zeros((100, 100, 4), np.uint8); left[10:90, 10:50] = [40, 90, 160, 255]
        # Missing 35% of right side
        right = np.zeros((100, 100, 4), np.uint8); right[10:40, 50:90] = [200, 120, 50, 255]
        report = self._run_multi_split([('left', left), ('right', right)])
        self.assertFalse(report['quality_gate']['passed'])
        self.assertEqual(report['quality_gate']['verdict'], 'reject_fallback_local')
        self.assertIn('missing_original_coverage', report['quality_gate']['reasons'])

    def test_split_quality_gate_rejects_severe_rgb_hallucination(self):
        left = np.zeros((100, 100, 4), np.uint8); left[10:90, 10:50] = [240, 10, 10, 255]
        right = np.zeros((100, 100, 4), np.uint8); right[10:90, 50:90] = [10, 240, 240, 255]
        report = self._run_multi_split([('left', left), ('right', right)])
        self.assertFalse(report['quality_gate']['passed'])
        self.assertIn('original_rgb_fidelity_drift', report['quality_gate']['reasons'])

    def test_split_quality_gate_rejects_duplicate_clone_layers(self):
        full = np.zeros((100, 100, 4), np.uint8)
        full[10:90, 10:50] = [40, 90, 160, 255]; full[10:90, 50:90] = [200, 120, 50, 255]
        clone = full.copy()
        report = self._run_multi_split([('body_a', full), ('body_b', clone)])
        self.assertFalse(report['quality_gate']['passed'])
        self.assertIn('duplicate_clone_layers', report['quality_gate']['reasons'])
        self.assertTrue(report['quality_gate']['duplicate_pairs'])

    def test_split_quality_gate_rejects_high_registration_failure_rate(self):
        left = np.zeros((100, 100, 4), np.uint8); left[10:90, 10:50] = [40, 90, 160, 255]
        right = np.zeros((100, 100, 4), np.uint8); right[10:90, 50:90] = [200, 120, 50, 255]
        report = self._run_multi_split(
            [('left', left), ('right', right), ('bad1', left), ('bad2', right)],
            fail_nodes=('bad1', 'bad2'),
        )
        self.assertFalse(report['quality_gate']['passed'])
        self.assertIn('high_registration_failure_rate', report['quality_gate']['reasons'])

if __name__=='__main__':unittest.main()
