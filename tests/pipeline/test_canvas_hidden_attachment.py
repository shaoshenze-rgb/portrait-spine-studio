import copy
import base64
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from src.pipeline.canvas_layer_parts import materialize_canvas_layer_parts
from src.pipeline.canvas_layer_rig import compile_canvas_layer_bindings
from src.pipeline.build_rig import build_rig
from src.render.renderer import SpineRenderer
from src.gates.module_quality import ModuleQualityAuditor
from src.gates.review_binding import ExportApprovalError, ensure_export_approved
from src.studio.module_contract import validate_module_contract

ROOT=Path(__file__).resolve().parents[2]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


class CanvasHiddenAttachmentTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory(dir=ROOT/'.studio-work',prefix='canvas-hidden-test-')
        self.addCleanup(tmp.cleanup);self.root=Path(tmp.name);self.counter=0

    def fixture(self, low=False):
        self.counter+=1;r=self.root/str(self.counter);r.mkdir()
        source=r/'original.png';Image.new('RGBA',(64,64),(40,50,60,255)).save(source)
        material={'version':1,'material_mode':'canvas_layers','source_image':str(source),
                  'source_sha256':sha(source),'image_size':[64,64],'pieces':[]}
        descriptions=[('body','torso','rigid',None,[18,30],(8,8,23,55)),
                      ('head','head','rigid','body',[18,12],(12,0,25,18)),
                      ('cloth','cloth','soft','body',[32,24],(31,20,45,40)),
                      ('cover','leg','rigid','body',[20,30],(18,15,56,45))]
        modules=[];connections=[]
        for i,(name,role,kind,parent,pivot,box) in enumerate(descriptions):
            a=np.zeros((64,64,4),np.uint8);x0,y0,x1,y1=box;a[y0:y1,x0:x1]=[40+i*40,70,130,255]
            if name=='cover' and low:a[24,29,3]=226
            p=r/(name+'.png');Image.fromarray(a).save(p)
            material['pieces'].append({'id':name,'module_id':name,'parent':parent,'kind':kind,
                'pivot':pivot,'draw_order':i,'image':str(p),'sha256':sha(p),
                'provenance':{'scene_id':'synthetic','node_id':name,'task_id':'synthetic-only',
                    'operation':'split-layers','source_asset_sha256':sha(source),
                    'result_asset_sha256':sha(p),'transform':[[1,0,0],[0,1,0]]}})
            m={'id':name,'role':role,'motion_type':kind,'parent':parent,'pivot':[v/64 for v in pivot],
               'motion':'soft_bend' if kind=='soft' else 'breath' if parent is None else 'sway','confidence':.95}
            if kind=='soft':m['centerline']=[m['pivot'],[38/64,30/64],[42/64,35/64]]
            modules.append(m)
            if parent:
                c={'id':'attach_'+name,'type':'attachment' if kind=='soft' else 'joint',
                   'a':parent,'b':name,'point':m['pivot']}
                if name=='cloth':c['hidden_by']=[{'front':'cover','back':'body'}]
                connections.append(c)
        parts=r/'parts';materialize_canvas_layer_parts(material,parts,evidence_root=r)
        plan={'pose_type':'airborne','anatomy':'humanoid_bust','modules':modules,'connections':connections,
              'occlusions':[{'front':'cover','back':'body','completion':'none',
                  'polygons':[[[22/64,22/64],[34/64,22/64],[34/64,26/64],[22/64,26/64]]]}],
              'uncertainties':[]}
        return r,parts,plan

    def build(self,low=False):
        r,parts,plan=self.fixture(low)
        spec=compile_canvas_layer_bindings(parts,plan,r/'binding',evidence_root=r)
        made=build_rig(parts,spec,r/'rig')
        renderer=SpineRenderer(r/'rig')
        qc=ModuleQualityAuditor(renderer,parts,made['report'],renderer.frame(None))
        self.assertFalse(qc.errors,qc.errors)
        return r,parts,spec,made,renderer,qc

    def test_full_cycle_uses_declared_front_only_and_remains_unexportable(self):
        r,parts,spec,made,renderer,qc=self.build()
        qc.audit_hidden_joint_cycle()
        check=qc.finish()['machineQC']['checks']['hiddenAttachmentCoverage']
        self.assertEqual(check['status'],'pass',check)
        row=check['attachments']['attach_cloth']
        self.assertEqual(row['requiredFramePoses'],241);self.assertTrue(row['fullCycleSampled'])
        self.assertEqual(row['maxUncoveredPixels'],0)
        self.assertEqual(row['allowedOccluders'],['cover'])
        self.assertEqual(spec['moduleContract']['contacts'][1]['hiddenBy'],[{'front':'cover','back':'body'}])
        with self.assertRaises(ExportApprovalError):ensure_export_approved(r/'rig')

    def test_low_alpha_is_pending_not_a_claim_of_large_hole_or_pass(self):
        _,_,_,_,_,qc=self.build(low=True);qc.audit_hidden_joint_cycle()
        check=qc.finish()['machineQC']['checks']['hiddenAttachmentCoverage']
        self.assertEqual(check['status'],'pending',check)
        row=check['attachments']['attach_cloth'];self.assertEqual(row['maxUncoveredPixels'],0)
        self.assertGreater(row['maxLowAlphaPixels'],0)

    def test_opening_at_one_pose_fails_even_if_whole_frame_is_opaque(self):
        _,_,_,_,renderer,qc=self.build()
        renderer.sk.json['animations']['idle']['bones'].setdefault('module_cover',{})['translate']=[
            {'time':0,'x':0,'y':0},{'time':4,'x':45,'y':0},{'time':8,'x':0,'y':0}]
        with patch.object(renderer,'frame',return_value=np.full((64,64,4),255,np.uint8)):
            qc.audit_hidden_joint_cycle()
        check=qc.finish()['machineQC']['checks']['hiddenAttachmentCoverage']
        self.assertEqual(check['status'],'fail',check)
        self.assertGreater(check['attachments']['attach_cloth']['maxUncoveredPixels'],0)

    def test_invalid_types_pair_order_or_changed_material_rejected_before_binding(self):
        for case in ('joint','grip','foot_anchor','effect','back','order','sha','uncovered_zone'):
            r,parts,plan=self.fixture();c=plan['connections'][1]
            if case in ('joint','grip','foot_anchor'):c['type']=case
            elif case=='effect':plan['modules'][3]['role']='effect_group'
            elif case=='back':c['hidden_by'][0]['back']='cloth';plan['occlusions'][0]['back']='cloth'
            elif case=='order':
                p=parts/'parts.json';d=json.loads(p.read_text());d['parts'][3]['drawOrder']=-1;p.write_text(json.dumps(d))
            elif case=='sha':
                p=r/'cover.png';a=np.asarray(Image.open(p)).copy();a[24,29,0]^=255;Image.fromarray(a).save(p)
            else:plan['occlusions'][0]['polygons'][0][2]=[60/64,60/64]
            with self.subTest(case=case),self.assertRaises((ValueError,ExportApprovalError)):
                compile_canvas_layer_bindings(parts,plan,r/'binding',evidence_root=r)
            self.assertFalse((r/'binding').exists())

    def test_target_replacement_with_updated_sha_still_fails_runtime(self):
        _,parts,spec,made,renderer,_=self.build()
        oc=made['report']['moduleContract']['occlusions'][0];path=Path(oc['sourceHiddenTargetMask'])
        a=np.asarray(Image.open(path)).copy();a[24,29]=0;Image.fromarray(a).save(path)
        oc['sourceHiddenTargetMaskSHA256']=sha(path);oc['remainingHiddenMaskSHA256']=sha(path)
        qc=ModuleQualityAuditor(renderer,parts,made['report'],renderer.frame(None))
        self.assertTrue(qc.errors)
        self.assertEqual(qc.finish()['machineQC']['checks']['hiddenAttachmentCoverage']['status'],'fail')

    def test_runtime_foreground_texture_cannot_be_replaced(self):
        _,parts,_,made,renderer,_=self.build()
        renderer.tex['cover'][...,3]=255
        qc=ModuleQualityAuditor(renderer,parts,made['report'],renderer.frame(None))
        self.assertTrue(qc.errors)
        self.assertEqual(qc.finish()['machineQC']['checks']['hiddenAttachmentCoverage']['status'],'fail')

    def test_undeclared_soft_attachment_keeps_parent_distance_gate(self):
        r,parts,plan=self.fixture();plan['connections'][1].pop('hidden_by');plan['occlusions']=[]
        with self.assertRaisesRegex(ValueError,'point.*body.*容差'):
            compile_canvas_layer_bindings(parts,plan,r/'binding',evidence_root=r)
        self.assertFalse((r/'binding').exists())

    def test_json_mode_cannot_enable_exception_on_source_sliced_entry(self):
        _,parts,plan=self.fixture();material=json.loads((parts/'parts.json').read_text())
        shadow=copy.deepcopy(plan);shadow.update(version=2,coord_mode='normalized',image_size=[64,64],
            input_sha256=material['sourceSHA256'],material_mode='canvas_layers')
        for m in shadow['modules']:
            part=next(p for p in material['parts'] if p['moduleId']==m['id'])
            a=np.asarray(Image.open(part['canvasLayer']['image']))[...,3]>0
            buf=io.BytesIO();Image.fromarray(a.astype('uint8')*255).save(buf,format='PNG');raw=buf.getvalue()
            m['pieces']=[{'id':part['name'],'draw_order':part['drawOrder'],
                          'mask_png_base64':base64.b64encode(raw).decode(),'mask_sha256':hashlib.sha256(raw).hexdigest()}]
        with self.assertRaises(ValueError):
            validate_module_contract(shadow,input_sha256=material['sourceSHA256'],image_size=[64,64])


if __name__=='__main__':unittest.main()
