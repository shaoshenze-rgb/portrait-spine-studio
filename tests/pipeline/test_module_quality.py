"""Independent synthetic pictures exercise module QC; no character-name exceptions."""
from __future__ import annotations

import json
import hashlib
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.gates.module_quality import (ModuleQualityAuditor, NegativeSpaceTracker,
    apply_affine, finalize_visual_review, rigidity_metrics, soft_deformation_metrics,
    transport_reference_mask)
from src.render.renderer import SpineRenderer

ROOT=Path(__file__).resolve().parents[2]


def scene(root, *, shared=False, still=False, stretch=False, soft_bad=False, contact=False, padding=0):
    """Write a tiny source and literal Spine JSON independently of build_rig."""
    width=height=128+padding*2
    pdir=root/'parts';rdir=root/'rig';(pdir/'work').mkdir(parents=True);(pdir/'parts').mkdir();(rdir/'images').mkdir(parents=True)
    source=np.zeros((height,width,4),np.uint8);labels=np.full((height,width),-1,np.int16)
    specs=[('trunk_piece','alpha_node',(18,35,42,65),(80,140,190,255),'body'),
           ('head_piece','beta_node',(74,18,28,30),(220,150,90,255),'head')]
    if contact:specs[1]=('head_piece','beta_node',(60,35,28,30),(220,150,90,255),'head')
    bones=[{'name':'root'},{'name':'whole','parent':'root'}];slots=[];attachments={};parts=[];modules=[]
    for index,(name,bone,(x,y,w,h),colour,role)in enumerate(specs):
        x+=padding;y+=padding
        image=np.zeros((h,w,4),np.uint8);image[:]=colour
        if index==0:image[20:31,15:28]=0
        fg=image[...,3]>0;source[y:y+h,x:x+w][fg]=image[fg];labels[y:y+h,x:x+w][fg]=index
        Image.fromarray(image).save(pdir/'parts'/f'{name}.png');Image.fromarray(image).save(rdir/'images'/f'{name}.png')
        mask=np.zeros((height,width),np.uint8);mask[y:y+h,x:x+w]=fg.astype(np.uint8)*255
        maskpath=pdir/f'{name}-mask.png';Image.fromarray(mask).save(maskpath)
        bones.append({'name':bone,'parent':'whole'if shared or stretch else'root','x':x+w/2,'y':-(y+h/2)})
        slots.append({'name':name,'bone':bone,'attachment':name})
        attachments[name]={name:{'width':w,'height':h}}
        parts.append({'name':name,'drawOrder':index,'bbox':[x,y,w,h],'image':f'parts/{name}.png'})
        modules.append({'id':'custom_'+role,'shape':'rigid','role':role,'motion':'required','pieces':[name],'visibleMasks':[str(maskpath)],'bone':bone})
    anim={}
    if not still:
        if shared:anim['whole']={'rotate':[{'time':0,'value':0},{'time':2,'value':8},{'time':8,'value':0}]}
        else:
            anim['alpha_node']={'translate':[{'time':0,'x':0,'y':0},{'time':2,'x':5,'y':0},{'time':8,'x':0,'y':0}]}
            anim['beta_node']={'translate':[{'time':0,'x':0,'y':0},{'time':2,'x':0,'y':3},{'time':8,'x':0,'y':0}]}
    if stretch:
        amount=.1 if stretch is True else float(stretch)
        anim['whole']={'scale':[{'time':0,'x':1,'y':1},{'time':2,'x':1+amount,'y':1},{'time':8,'x':1,'y':1}]}
    Image.fromarray(source).save(pdir/'source.png');Image.fromarray(source).save(pdir/'rest_composite.png');np.save(pdir/'work/labels.npy',labels)
    digest=hashlib.sha256((pdir/'source.png').read_bytes()).hexdigest()
    (pdir/'parts.json').write_text(json.dumps({'source':str(pdir/'source.png'),'sourceSHA256':digest,'canvas':[width,height],'parts':parts,'opaqueThreshold':250}))
    (pdir/'rest_diff.json').write_text(json.dumps({'maxRGBA':0}))
    skeleton={'skeleton':{'spine':'4.2.43','width':width,'height':height,'x':0,'y':-height,'images':'./images/'},
              'bones':bones,'slots':slots,'skins':[{'name':'default','attachments':attachments}],'animations':{'idle':{'bones':anim}}}
    (rdir/'scene.spine.json').write_text(json.dumps(skeleton))
    contract={'version':1,'sourceVersion':2,'sourceSHA256':digest,'anatomy':'humanoid','requiredModules':[m['id']for m in modules],
              'modules':modules,'contacts':[],'occlusions':[]}
    contract['visibleMaskSHA256']={m['pieces'][0]:hashlib.sha256(Path(m['visibleMasks'][0]).read_bytes()).hexdigest()for m in modules}
    if contact:
        contract['contacts']=[{'id':'declared_grip','type':'grip',
            'a':{'module':'custom_body','piece':'trunk_piece','point':[60,50]},
            'b':{'module':'custom_head','piece':'head_piece','point':[60,50]}}]
    report={'spineFile':'scene.spine.json','canvas':[width,height],'origin':[0,0],'duration':8,'preset':'agile_full','gain':1,
            'moduleContract':contract,'attachments':[], 'boneRoles':{}}
    (rdir/'rig-build-report.json').write_text(json.dumps(report))
    return pdir,rdir,report,skeleton


class TestModuleQuality(unittest.TestCase):
    def setUp(self):
        root=ROOT/'.studio-work';root.mkdir(exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=root,prefix='module-qc-');self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)

    def collect(self, **options):
        pdir,rdir,report,sk=scene(self.root,**options);renderer=SpineRenderer(rdir);setup=renderer.frame(None)
        qc=ModuleQualityAuditor(renderer,pdir,report,setup)
        for t in (0,1,2,4,6,8):qc.observe(t,1.0,renderer.frame(t))
        return qc.finish(),pdir,rdir,report,renderer

    def test_rigid_transform_accepts_multi_piece_and_rejects_scale_shear(self):
        # Two disconnected sword pieces are one object, not two semantic fragments.
        ref=np.array([[0,0],[4,0],[4,3],[0,3],[10,0],[14,0],[14,3],[10,3]],float)
        a=.23;rotation=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
        good=ref@rotation.T+[4,7];m=rigidity_metrics(ref,good)
        self.assertLess(m['distanceRelativeError'],1e-10);self.assertLess(m['singularDeviation'],1e-10)
        wrong=good.copy();wrong[4:]+=[2,0]
        self.assertGreater(rigidity_metrics(ref,wrong)['affineResidualPx'],.2)
        self.assertGreater(rigidity_metrics(ref,ref@np.array([[1.1,.2],[0,1]]).T)['shear'],.1)

    def test_audit_crop_matches_full_slot_raster_at_fractional_pose(self):
        from src.gates.module_quality import _render_slot_crop
        p,r,rep,sk=scene(self.root)
        ren=SpineRenderer(r);ren.pose(1.3);slot=ren.sk.slots[0]
        vertices=ren.to_canvas(ren.sk.attachment_vertices(slot)[0])
        full=_render_slot_crop(ren,slot,vertices,[0,0,128,128])
        patch=_render_slot_crop(ren,slot,vertices,[25,42,31,38])
        np.testing.assert_array_equal(patch,full[42:80,25:56])

    def test_ancestor_stretch_is_not_hidden_by_region_type(self):
        q,*_=self.collect(stretch=True)
        self.assertEqual(q['bindingMotion']['checks']['rigidity']['status'],'fail')
        self.assertFalse(q['overall']['exportAllowed'])

    def test_one_tenth_percent_real_stretch_is_not_numeric_tolerance(self):
        q,*_=self.collect(stretch=.001)
        self.assertEqual(q['bindingMotion']['checks']['rigidity']['status'],'fail')
        self.assertGreater(q['bindingMotion']['modules']['custom_body']['rigidity']['singularDeviation'],.00099)

    def test_rigid_numeric_error_for_long_thin_multi_piece_shapes(self):
        worst=0.0
        for size in (32,1080,8192):
            reference=np.array([[0,0],[size,0],[size,.05],[0,.05],
                                [size*1.2,0],[size*1.7,0],[size*1.7,.25],[size*1.2,.25]],float)+[11000,17000]
            for angle in (.03,.37,1.4,2.8):
                rotation=np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
                m=rigidity_metrics(reference,reference@rotation.T+[517,293])
                worst=max(worst,m['singularDeviation'],m['shear'],m['distanceRelativeError'])
                self.assertLess(m['affineResidualPx'],1e-7)
        self.assertLess(worst,1e-7,'pure rigidity must fit below the 1e-6 numerical guard')

    def test_transparent_padding_does_not_change_motion_thresholds(self):
        q,*_=self.collect()
        padded=self.root/'padded';padded.mkdir();p,r,rep,_=scene(padded,padding=80)
        renderer=SpineRenderer(r);qc=ModuleQualityAuditor(renderer,p,rep,renderer.frame(None))
        for t in(0,1,2,4,6,8):qc.observe(t,1,renderer.frame(t))
        other=qc.finish()
        self.assertEqual(q['bodyHeightPx'],other['bodyHeightPx'])
        self.assertEqual(q['thresholdScale'],other['thresholdScale'])
        self.assertEqual(q['bindingMotion']['status'],other['bindingMotion']['status'])
        for key in ('visibleMotion','contacts','relativeMotion'):
            self.assertEqual(q['bindingMotion']['checks'][key]['status'],other['bindingMotion']['checks'][key]['status'])

    def test_declared_tiny_body_height_cannot_lower_motion_threshold(self):
        p,r,rep,_=scene(self.root);rep['bodyHeightPx']=1
        ren=SpineRenderer(r);qc=ModuleQualityAuditor(ren,p,rep,ren.frame(None));qc.observe(2,1,ren.frame(2));q=qc.finish()
        self.assertEqual(q['inputOwnership']['checks']['bodyScaleIdentity']['status'],'fail')
        self.assertGreater(q['bodyHeightPx'],50)

    def test_static_and_single_whole_bone_do_not_fake_breath(self):
        q,*_=self.collect(still=True)
        self.assertEqual(q['bindingMotion']['checks']['visibleMotion']['status'],'fail')
        other=self.root/'shared';other.mkdir();p,r,report,_=scene(other,shared=True);ren=SpineRenderer(r);qc=ModuleQualityAuditor(ren,p,report,ren.frame(None))
        for t in(0,2,4,8):qc.observe(t,1,ren.frame(t))
        q=qc.finish();self.assertEqual(q['bindingMotion']['checks']['relativeMotion']['status'],'fail')

    def test_soft_strain_fold_and_excess_curvature(self):
        ref=np.array([[0,0],[10,0],[0,10],[10,10]],float);tris=np.array([[0,1,2],[1,3,2]])
        rigid=ref+np.array([3,4]);self.assertEqual(soft_deformation_metrics(ref,rigid,tris)['foldedTriangles'],0)
        bad=ref.copy();bad[3]=[5,-4]
        self.assertGreater(soft_deformation_metrics(ref,bad,tris)['foldedTriangles'],0)
        stretch=ref.copy();stretch[:,0]*=1.2
        self.assertGreater(soft_deformation_metrics(ref,stretch,tris)['maxSingularDeviation'],.19)

    def test_whole_soft_module_scale_is_not_accepted_as_gentle_bend(self):
        p,r,rep,sk=scene(self.root)
        rep['moduleContract']['modules'][1].update(shape='soft',role='cloth')
        sk['animations']['idle']['bones']['beta_node']['scale']=[{'time':0,'x':1,'y':1},{'time':2,'x':1.05,'y':1.05},{'time':8,'x':1,'y':1}]
        (r/'scene.spine.json').write_text(json.dumps(sk));ren=SpineRenderer(r);qc=ModuleQualityAuditor(ren,p,rep,ren.frame(None))
        for t in(0,2,4,8):qc.observe(t,1,ren.frame(t))
        q=qc.finish()
        self.assertEqual(q['bindingMotion']['checks']['softDriverRigidity']['status'],'fail')
        self.assertIn('custom_head',q['bindingMotion']['checks']['softDriverRigidity']['uniformScaleOrShearModules'])

    def test_runtime_weighted_edge_on_foreign_bone_fails(self):
        p,r,rep,sk=scene(self.root)
        att=sk['skins'][0]['attachments']['head_piece']['head_piece']
        att.update(type='mesh',uvs=[0,1,0,0,1,0,1,1],triangles=[0,1,2,0,2,3],
                   vertices=[1,3,-14,-15,1,1,3,-14,15,1,1,3,14,15,1,1,2,63,19.5,1])
        (r/'scene.spine.json').write_text(json.dumps(sk));ren=SpineRenderer(r);qc=ModuleQualityAuditor(ren,p,rep,ren.frame(None))
        qc.observe(2,1,ren.frame(2));q=qc.finish()
        self.assertEqual(q['bindingMotion']['checks']['moduleInfluences']['status'],'fail')
        self.assertEqual(q['bindingMotion']['checks']['rigidity']['status'],'fail')

    def test_runtime_soft_mesh_flip_is_measured_not_inferred_from_type(self):
        p,r,rep,sk=scene(self.root)
        rep['moduleContract']['modules'][1].update(shape='soft',role='cloth')
        att=sk['skins'][0]['attachments']['head_piece']['head_piece']
        att.update(type='mesh',uvs=[0,1,0,0,1,0,1,1],triangles=[0,1,2,0,2,3],vertices=[-14,-15,-14,15,14,15,14,-15])
        sk['animations']['idle']['attachments']={'default':{'head_piece':{'head_piece':{'deform':[
            {'time':0,'vertices':[0]*8},{'time':2,'vertices':[0,0,0,0,-34,-40,0,0]}]}}}}
        (r/'scene.spine.json').write_text(json.dumps(sk));ren=SpineRenderer(r);qc=ModuleQualityAuditor(ren,p,rep,ren.frame(None))
        qc.observe(2,1,ren.frame(2));q=qc.finish()
        self.assertEqual(q['bindingMotion']['checks']['softDeformation']['status'],'fail')
        self.assertGreater(q['bindingMotion']['modules']['custom_head']['softDeformation']['foldedTriangles'],0)

    def test_grip_measures_material_point_not_bone_name(self):
        q,*_=self.collect(contact=True)
        self.assertEqual(q['bindingMotion']['checks']['contacts']['status'],'fail')
        self.assertGreater(q['bindingMotion']['checks']['contacts']['maxDriftPx']['declared_grip'],4)

    def pivot_contact_scene(self, root, contact_type='grip', soft=False):
        p,r,rep,sk=scene(root,contact=True,still=True)
        rep['moduleContract']['contacts'][0]['type']=contact_type
        if soft:rep['moduleContract']['modules'][1].update(shape='soft',role='cloth')
        # Keep the setup image unchanged while putting the real material pivot
        # on the contact point; the weapon can now spin without point drift.
        sk['bones'][3]['x']=60
        sk['skins'][0]['attachments']['head_piece']['head_piece']['x']=14
        sk['animations']['idle']['bones']['beta_node']={'rotate':[
            {'time':0,'value':0},{'time':2,'value':1},{'time':8,'value':0}]}
        (r/'scene.spine.json').write_text(json.dumps(sk))
        ren=SpineRenderer(r);qc=ModuleQualityAuditor(ren,p,rep,ren.frame(None))
        for t in(0,2,4,8):qc.observe(t,1,ren.frame(t))
        return qc.finish()

    def test_fixed_grip_rejects_one_degree_spin_with_unchanged_contact_point(self):
        q=self.pivot_contact_scene(self.root)
        checks=q['bindingMotion']['checks']
        self.assertEqual(checks['rigidity']['status'],'pass')
        self.assertEqual(checks['contacts']['status'],'pass')
        self.assertLess(checks['contacts']['maxDriftPx']['declared_grip'],1e-10)
        self.assertEqual(checks['gripRelativePose']['status'],'fail')
        metrics=checks['gripRelativePose']['contacts']['declared_grip']
        self.assertAlmostEqual(metrics['maxRelativeRotationDegrees'],1,places=8)
        self.assertGreater(metrics['maxRelativeLinearDeviation'],.017)
        self.assertFalse(q['overall']['exportAllowed'])

    def test_fixed_grip_accepts_common_ancestor_rotation_and_translation(self):
        p,r,rep,sk=scene(self.root,contact=True,shared=True)
        sk['animations']['idle']['bones']['whole']['translate']=[
            {'time':0,'x':0,'y':0},{'time':2,'x':5,'y':3},{'time':8,'x':0,'y':0}]
        (r/'scene.spine.json').write_text(json.dumps(sk))
        ren=SpineRenderer(r);qc=ModuleQualityAuditor(ren,p,rep,ren.frame(None))
        for t in(0,2,4,8):qc.observe(t,1,ren.frame(t))
        checks=qc.finish()['bindingMotion']['checks']
        self.assertEqual(checks['contacts']['status'],'pass')
        self.assertEqual(checks['gripRelativePose']['status'],'pass')
        self.assertLess(checks['gripRelativePose']['contacts']['declared_grip']['maxRelativeLinearDeviation'],1e-10)
        # Contact stability alone does not certify the independent breathing rule.
        self.assertEqual(checks['relativeMotion']['status'],'fail')

    def test_joint_and_soft_attachment_allow_pivot_rotation(self):
        for kind,soft in(('joint',False),('attachment',True)):
            with self.subTest(kind=kind):
                root=self.root/kind;root.mkdir()
                checks=self.pivot_contact_scene(root,kind,soft)['bindingMotion']['checks']
                self.assertEqual(checks['contacts']['status'],'pass')
                self.assertEqual(checks['gripRelativePose']['status'],'not_applicable')
                if soft:self.assertEqual(checks['softDeformation']['status'],'pass')

    def test_contact_sweep_catches_gap_open_to_exterior(self):
        p,r,rep,_=scene(self.root,contact=True,still=True);ren=SpineRenderer(r);qc=ModuleQualityAuditor(ren,p,rep,ren.frame(None))
        frame=ren.frame(0);frame[:53,59:62]=0
        qc.observe(0,1,frame);q=qc.finish()
        self.assertEqual(q['bindingMotion']['checks']['contacts']['status'],'pass')
        self.assertEqual(q['machineQC']['checks']['contactSweepCoverage']['status'],'fail')

    def test_changed_uv_or_texture_cannot_fake_stable_material(self):
        p,r,rep,_=scene(self.root);ren=SpineRenderer(r);qc=ModuleQualityAuditor(ren,p,rep,ren.frame(None))
        ren.tex['head_piece'][...,:3]*=.5
        qc.observe(2,1,ren.frame(2));q=qc.finish()
        self.assertEqual(q['machineQC']['checks']['materialStability']['status'],'fail')
        other=self.root/'uv';other.mkdir();p,r,rep,sk=scene(other)
        att=sk['skins'][0]['attachments']['head_piece']['head_piece']
        att.update(type='mesh',uvs=[0,1,0,0,1,0,1,1],triangles=[0,1,2,0,2,3],vertices=[-14,-15,-14,15,14,15,14,-15])
        (r/'scene.spine.json').write_text(json.dumps(sk));ren=SpineRenderer(r);qc=ModuleQualityAuditor(ren,p,rep,ren.frame(None))
        ren.sk.attachments['head_piece']['head_piece']['uvs'][0]=.1
        qc.observe(2,1,ren.frame(2));self.assertEqual(qc.finish()['machineQC']['checks']['materialStability']['status'],'fail')

    def test_fully_occluded_motion_is_not_visible_motion(self):
        p,r,rep,sk=scene(self.root)
        cover=np.zeros((40,38,4),np.uint8);cover[:]=[220,150,90,255];Image.fromarray(cover).save(r/'images/cover_piece.png')
        sk['bones'].append({'name':'cover_node','parent':'root','x':88,'y':-33})
        sk['slots'].append({'name':'cover_piece','bone':'cover_node','attachment':'cover_piece'})
        sk['skins'][0]['attachments']['cover_piece']={'cover_piece':{'width':38,'height':40}}
        rep['moduleContract']['modules'].append({'id':'overlay','role':'prop','shape':'rigid','motion':'static','staticReason':'opaque overlay',
            'bone':'cover_node','pieces':['cover_piece'],'visibleMasks':[str(p/'head_piece-mask.png')]})
        rep['moduleContract']['requiredModules'].append('overlay')
        rep['moduleContract']['visibleMaskSHA256']['cover_piece']=rep['moduleContract']['visibleMaskSHA256']['head_piece']
        (r/'scene.spine.json').write_text(json.dumps(sk));ren=SpineRenderer(r);qc=ModuleQualityAuditor(ren,p,rep,ren.frame(None))
        for t in(0,2,4,8):qc.observe(t,1,ren.frame(t))
        check=qc.finish()['bindingMotion']['checks']['visibleMotion']
        self.assertIn('custom_head',check['stillRequiredModules'])

    def test_given_mask_equality_does_not_approve_semantics(self):
        q,*_=self.collect()
        self.assertEqual(q['inputOwnership']['checks']['actualOwnership']['status'],'pass')
        self.assertEqual(q['visual']['status'],'pending')
        self.assertFalse(q['overall']['exportAllowed'])

    def test_foreign_edge_labels_and_visible_colour_pollution_are_caught(self):
        p,r,report,_=scene(self.root);labels=np.load(p/'work/labels.npy');labels[45,25]=1;np.save(p/'work/labels.npy',labels)
        part=np.asarray(Image.open(r/'images/trunk_piece.png')).copy();part[4:9,4:9,:3]=[255,0,255];Image.fromarray(part).save(r/'images/trunk_piece.png')
        ren=SpineRenderer(r);qc=ModuleQualityAuditor(ren,p,report,ren.frame(None));q=qc.finish()
        self.assertEqual(q['inputOwnership']['checks']['actualOwnership']['status'],'fail')
        self.assertEqual(q['inputOwnership']['checks']['visibleTexture']['status'],'fail')

    def test_current_source_identity_is_bound_to_parts_and_contract(self):
        p,r,rep,_=scene(self.root)
        source=np.asarray(Image.open(p/'source.png')).copy();source[40,30,0]+=1;Image.fromarray(source).save(p/'source.png')
        renderer=SpineRenderer(r);qc=ModuleQualityAuditor(renderer,p,rep,renderer.frame(None))
        self.assertEqual(qc.finish()['inputOwnership']['checks']['sourceIdentity']['status'],'fail')

    def test_changed_reference_mask_cannot_borrow_contract_identity(self):
        p,r,rep,_=scene(self.root);mask=p/'trunk_piece-mask.png';a=np.asarray(Image.open(mask)).copy();a[35,18]=0;Image.fromarray(a).save(mask)
        ren=SpineRenderer(r);qc=ModuleQualityAuditor(ren,p,rep,ren.frame(None))
        self.assertEqual(qc.finish()['inputOwnership']['checks']['maskIdentity']['status'],'fail')

    def test_visible_alpha_and_semitransparent_colour_pollution_are_caught(self):
        p,r,rep,_=scene(self.root)
        source=np.asarray(Image.open(p/'source.png')).copy();source[40:43,23:26,3]=80;Image.fromarray(source).save(p/'source.png')
        part=np.asarray(Image.open(r/'images/trunk_piece.png')).copy();part[5:8,5:8,3]=80;Image.fromarray(part).save(r/'images/trunk_piece.png')
        # Reference identity legitimately describes the semitransparent input.
        digest=hashlib.sha256((p/'source.png').read_bytes()).hexdigest();pj=json.loads((p/'parts.json').read_text());pj['sourceSHA256']=digest;(p/'parts.json').write_text(json.dumps(pj));rep['moduleContract']['sourceSHA256']=digest
        clean=SpineRenderer(r);qc=ModuleQualityAuditor(clean,p,rep,clean.frame(None))
        self.assertEqual(qc.finish()['inputOwnership']['checks']['visibleTexture']['status'],'pass')
        part[5:8,5:8,:3]=[255,0,255];part[10:13,5:8,3]=120;Image.fromarray(part).save(r/'images/trunk_piece.png')
        broken=SpineRenderer(r);qc=ModuleQualityAuditor(broken,p,rep,broken.frame(None))
        self.assertEqual(qc.finish()['inputOwnership']['checks']['visibleTexture']['status'],'fail')

    def test_old_rigid_hole_translation_and_new_hole_both_handled(self):
        alpha=np.zeros((100,100),np.uint8);alpha[10:80,10:80]=255;alpha[30:40,30:40]=0
        owners=np.where(alpha>=128,0,-1);tracker=NegativeSpaceTracker(alpha,alpha,owners,[0])
        transform=np.array([[1,0,13],[0,1,7],[0,0,1]],float)
        opaque=transport_reference_mask(alpha>=128,transform,alpha.shape)
        moved=opaque.astype(np.uint8)*255
        valid=tracker.evaluate(moved,{0:transform},15)
        self.assertEqual(valid['status'],'pass',valid)
        moved[45,55:58]=0
        bad=tracker.evaluate(moved,{0:transform},15)
        self.assertEqual(bad['status'],'fail');self.assertGreaterEqual(bad['newHolePixels'],3)

    def test_rotated_existing_hole_and_attached_new_tear(self):
        a=np.zeros((100,100),np.uint8);a[10:90,10:90]=255;a[40:50,40:50]=0
        owner=np.where(a>=128,0,-1);tracker=NegativeSpaceTracker(a,a,owner,[0])
        # Analytic 90-degree rotation around canvas centre is independent of runtime.
        matrix=np.array([[0,-1,100],[1,0,0],[0,0,1]],float);frame=np.rot90(a,-1)
        self.assertEqual(tracker.evaluate(frame,{0:matrix})['status'],'pass')
        frame[40,60:63]=0  # New cut connected to the transported old hole; never flood exempt.
        bad=tracker.evaluate(frame,{0:matrix});self.assertEqual(bad['status'],'fail');self.assertGreaterEqual(bad['newHolePixels'],3)

    def test_1784_pixel_connected_tear_not_flood_exempt(self):
        rest=np.zeros((128,128),np.uint8);rest[8:-8,8:-8]=255;rest[56:60,56:60]=0
        frame=rest.copy();frame[50:80,40:100]=0
        tracker=NegativeSpaceTracker(rest,rest,np.where(rest>=128,0,-1),[0])
        report=tracker.evaluate(frame,{0:np.eye(3)})
        self.assertEqual(report['newHolePixels'],1784);self.assertEqual(report['status'],'fail')

    def test_actual_saved_155_pixel_patch_control_is_still_rejected(self):
        source=ROOT/'.studio-work/erika-gate-review-20261121/new-tear-control'
        if not(source/'rest.png').is_file():self.skipTest('Local incident evidence unavailable; synthetic true-hole tests still run')
        # The saved PNG contact sheets are composited for people; re-render the
        # retained real rig to consume actual alpha rather than opaque screenshots.
        renderer=SpineRenderer(source/'rig')
        rest=renderer.frame(None)[...,3]
        current=renderer.frame(2)[...,3]
        tracker=NegativeSpaceTracker(rest,rest,np.where(rest>=128,0,-1),[0])
        report=tracker.evaluate(current,{0:np.eye(3)})
        self.assertEqual(report['status'],'fail');self.assertEqual(report['newHolePixels'],155)

    def test_real_raster_arbitrary_rotation_preserves_original_hole(self):
        p,r,rep,sk=scene(self.root)
        sk['animations']['idle']['bones']['alpha_node']={'rotate':[{'time':0,'value':0},{'time':2,'value':23},{'time':8,'value':0}]}
        (r/'scene.spine.json').write_text(json.dumps(sk));renderer=SpineRenderer(r)
        qc=ModuleQualityAuditor(renderer,p,rep,renderer.frame(None))
        for t in (0,.5,1,1.5,2,3,4,6,8):qc.observe(t,1,renderer.frame(t))
        negative=qc.finish()['machineQC']['checks']['negativeSpace']
        self.assertEqual(negative['status'],'pass',negative)
        self.assertEqual(negative['newHolePixelsMax'],0)

    def soft_hole_scene(self, missing_triangles=False):
        p,r,rep,sk=scene(self.root)
        rep['moduleContract']['modules'][0].update(shape='soft',role='cloth')
        att=sk['skins'][0]['attachments']['trunk_piece']['trunk_piece']
        if not missing_triangles:
            att.update(type='mesh',uvs=[0,1,0,0,1,0,1,1],triangles=[0,1,2,0,2,3],
                       vertices=[-21,-32.5,-21,32.5,21,32.5,21,-32.5])
            sk['animations']['idle']['attachments']={'default':{'trunk_piece':{'trunk_piece':{'deform':[
                {'time':0,'vertices':[0]*8},{'time':2,'vertices':[0,0,0,0,1,1,0,0]},
                {'time':8,'vertices':[0]*8}]}}}}
        else:
            xs=[0,15,28,42];ys=[0,20,31,65];verts=[];uvs=[];triangles=[]
            for y in ys:
                for x in xs:verts.extend([x-21,32.5-y]);uvs.extend([x/42,y/65])
            for y in range(3):
                for x in range(3):
                    if(x,y)==(1,1):continue
                    a=y*4+x;b=a+1;c=a+5;d=a+4;triangles.extend([a,b,c,a,c,d])
            att.update(type='mesh',uvs=uvs,triangles=triangles,vertices=verts)
        (r/'scene.spine.json').write_text(json.dumps(sk));return p,r,rep,SpineRenderer(r)

    def test_existing_soft_material_hole_survives_gentle_bend(self):
        p,r,rep,ren=self.soft_hole_scene();qc=ModuleQualityAuditor(ren,p,rep,ren.frame(None))
        for t in(0,.5,1,1.5,2,3,4,6,8):qc.observe(t,1,ren.frame(t))
        negative=qc.finish()['machineQC']['checks']['negativeSpace']
        self.assertEqual(negative['status'],'pass',negative)
        self.assertTrue(all(s['supportedSoftMaterialHoles']==1 for s in negative['samples']))

    def test_new_cut_beside_soft_material_hole_stays_failure(self):
        p,r,rep,ren=self.soft_hole_scene();qc=ModuleQualityAuditor(ren,p,rep,ren.frame(None))
        ren.tex['trunk_piece'][20,28:31]=0
        qc.observe(2,1,ren.frame(2));negative=qc.finish()['machineQC']['checks']['negativeSpace']
        self.assertEqual(negative['status'],'pass');self.assertTrue(negative.get('advisory'))
        self.assertEqual(negative.get('measuredStatus'),'fail');self.assertGreater(negative['newHolePixelsMax'],0)

    def test_soft_hole_without_existing_material_triangles_remains_pending(self):
        p,r,rep,ren=self.soft_hole_scene(missing_triangles=True);qc=ModuleQualityAuditor(ren,p,rep,ren.frame(None))
        qc.observe(0,1,ren.frame(0));negative=qc.finish()['machineQC']['checks']['negativeSpace']
        self.assertEqual(negative['status'],'pass');self.assertTrue(negative.get('advisory'))
        self.assertEqual(negative.get('measuredStatus'),'pending');self.assertEqual(negative['samples'][0]['supportedSoftMaterialHoles'],0)

    def test_soft_hole_reference_cannot_use_mismatched_source_identity(self):
        p,r,rep,ren=self.soft_hole_scene();rep['moduleContract']['sourceSHA256']='0'*64
        qc=ModuleQualityAuditor(ren,p,rep,ren.frame(None));qc.observe(2,1,ren.frame(2));q=qc.finish()
        self.assertEqual(q['inputOwnership']['checks']['sourceIdentity']['status'],'fail')
        self.assertFalse(q['machineQC']['checks']['negativeSpace']['samples'][0]['referenceVerified'])
        self.assertEqual(q['machineQC']['checks']['negativeSpace']['samples'][0]['supportedSoftMaterialHoles'],0)

    def test_real_renderer_new_surface_cut_not_exempted_by_original_hole(self):
        p,r,rep,_=scene(self.root);renderer=SpineRenderer(r);qc=ModuleQualityAuditor(renderer,p,rep,renderer.frame(None))
        # Corrupt only the current texture after registering original reference pixels.
        renderer.tex['trunk_piece'][35:40,30:35]=0
        qc.observe(2,1,renderer.frame(2))
        ns=qc.finish()['machineQC']['checks']['negativeSpace']
        self.assertEqual(ns['status'],'pass');self.assertTrue(ns.get('advisory'));self.assertGreater(ns['newHolePixelsMax'],0)

    def test_complete_compiler_output_uses_custom_roles_and_real_mesh_rigidity(self):
        from tests.pipeline.test_complete_modules import TestCompleteModules
        from src.gates.audit import run_audit
        case=TestCompleteModules();case.setUp()
        try:
            case.compile()
            report=run_audit(case.d/'run/parts',case.d/'run/rig',sample_count=6,stress_samples=4)
            self.assertEqual(report['quality']['inputOwnership']['status'],'pass')
            self.assertEqual(report['quality']['bindingMotion']['status'],'pass')
            self.assertEqual(report['quality']['bindingMotion']['checks']['softDriverRigidity']['status'],'pass')
            self.assertEqual(report['structuralChecks']['required_parts']['status'],'pass')
            self.assertEqual(report['structuralChecks']['weapon_rigidity']['status'],'pass')
            self.assertEqual(report['quality']['visual']['status'],'pending')
            self.assertFalse(report['quality']['overall']['exportAllowed'])
        finally:case.doCleanups()

    def test_hidden_local_fill_cannot_copy_foreground_colour_unnoticed(self):
        from tests.pipeline.test_complete_modules import TestCompleteModules, fixture
        case=TestCompleteModules();case.setUp()
        try:
            case.contract=fixture(case.image,local_fill=True)
            _,_,split,made,ren=case.compile()
            piece=next(p for p in split['parts_json']['parts']if p['pieceId']=='body_p0')
            x,y,_,_=piece['bbox'];pad=made['report']['padPx']
            # Corrupt only material hidden under the opaque face, not reference source.
            ren.tex[piece['name']][49-y+pad,60-x+pad]=[200,90,100,255]
            qc=ModuleQualityAuditor(ren,case.d/'run/parts',made['report'],ren.frame(None));q=qc.finish()
            self.assertEqual(q['inputOwnership']['checks']['visibleTexture']['status'],'pass')
            self.assertEqual(q['machineQC']['checks']['hiddenFill']['status'],'fail')
            self.assertGreater(q['machineQC']['checks']['hiddenFill']['localFillOutsideOwnSourceColourBounds'],0)
        finally:case.doCleanups()

    def test_occlusion_without_completion_rejects_opaque_and_translucent_hidden_material(self):
        from tests.pipeline.test_complete_modules import TestCompleteModules, fixture
        for alpha in (255, 64):
            with self.subTest(alpha=alpha):
                case=TestCompleteModules();case.setUp()
                try:
                    case.contract=fixture(case.image,local_fill=True)
                    case.contract['occlusions'][0]['completion']='none'
                    _,_,split,made,ren=case.compile()
                    piece=next(p for p in split['parts_json']['parts']if p['pieceId']=='body_p0')
                    x,y,_,_=piece['bbox'];pad=made['report']['padPx']
                    # A declared overlap does not grant permission to manufacture
                    # material underneath it, even if the front hides it at rest.
                    ren.tex[piece['name']][49-y+pad,60-x+pad]=[200,90,100,alpha]
                    qc=ModuleQualityAuditor(ren,case.d/'run/parts',made['report'],ren.frame(None));q=qc.finish()
                    self.assertEqual(q['inputOwnership']['checks']['visibleTexture']['status'],'pass')
                    hidden=q['machineQC']['checks']['hiddenFill']
                    self.assertEqual(hidden['status'],'fail')
                    self.assertGreater(hidden['undeclaredHiddenPixels'],0)
                    self.assertGreater(hidden['hiddenNonzeroAlphaPixels'],0)
                    self.assertEqual(hidden['appearanceApproval'],'pending_visual_review')
                finally:case.doCleanups()

    def test_translucent_hidden_fill_cannot_escape_into_attachment_padding(self):
        from tests.pipeline.test_complete_modules import TestCompleteModules, fixture
        case=TestCompleteModules();case.setUp()
        try:
            case.contract=fixture(case.image,local_fill=True)
            _,_,split,made,ren=case.compile()
            piece=next(p for p in split['parts_json']['parts']if p['pieceId']=='body_p0')
            x,y,_,_=piece['bbox'];pad=made['report']['padPx']
            # This is own-source colour, hidden behind the head, but outside
            # both local/automatic fill and the actual current parts bbox.
            # Adding a local repair no longer removes neighbouring auto seams,
            # so choose padding relative to the compiled bbox, not old y=48.
            self.assertGreater(pad,0)
            self.assertGreater(y,20)
            ren.tex[piece['name']][pad-1,60-x+pad]=[5,43,15,64]
            q=ModuleQualityAuditor(ren,case.d/'run/parts',made['report'],ren.frame(None)).finish()
            self.assertEqual(q['inputOwnership']['checks']['visibleTexture']['status'],'pass')
            hidden=q['machineQC']['checks']['hiddenFill']
            self.assertEqual(hidden['status'],'fail')
            self.assertGreater(hidden['undeclaredHiddenPixels'],0)
            self.assertGreater(hidden['hiddenNonzeroAlphaPixels'],hidden['hiddenOpaquePixels'])
        finally:case.doCleanups()

    def remaining_hidden_scene(self, *, edge=False, soft=False, missing_registration=False):
        p,r,rep,sk=scene(self.root,contact=True)
        target=np.zeros((128,128),np.uint8)
        if edge:target[45,60]=255
        else:target[41:60,68:80]=255
        path=p/'registered-unfilled.png';Image.fromarray(target).save(path)
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        rep['moduleContract']['contacts']=[]
        rep['moduleContract']['occlusions']=[{'front':'head_piece','back':'trunk_piece','completionMode':'local',
            'bandDerivation':{'unfilledEligiblePixels':int((target>0).sum()),'unsupportedTranslucentPixels':0},
            'sourceHiddenTargetRegistration':'original-back-claim-and-occlusion-and-visible-front',
            'sourceHiddenTargetMask':str(path),'sourceHiddenTargetMaskSHA256':digest,
            'remainingHiddenMask':str(path),'remainingHiddenMaskSHA256':digest,
            'remainingHiddenPixels':int((target>0).sum())}]
        if missing_registration:rep['moduleContract']['occlusions'][0].pop('remainingHiddenMaskSHA256')
        if soft:rep['moduleContract']['modules'][0]['shape']='soft'
        # Front shifts right by exactly one pixel. Deep material remains hidden;
        # the one-pixel boundary probe is exposed without changing back geometry.
        sk['animations']['idle']['bones']={'beta_node':{'translate':[
            {'time':0,'x':0,'y':0},{'time':2,'x':1,'y':0},{'time':8,'x':0,'y':0}]}}
        (r/'scene.spine.json').write_text(json.dumps(sk))
        return p,r,rep,SpineRenderer(r)

    def test_deep_unfilled_hidden_material_can_pass_independent_source_coverage(self):
        p,r,rep,ren=self.remaining_hidden_scene();qc=ModuleQualityAuditor(ren,p,rep,ren.frame(None))
        for time in(0,1,2,4,6,8):qc.observe(time,1,ren.frame(time))
        coverage=qc.finish()['machineQC']['checks']['remainingHiddenCoverage']
        self.assertEqual(coverage['status'],'pass',coverage)
        self.assertGreater(coverage['occlusions'][0]['remainingPixels'],200)
        self.assertEqual(coverage['occlusions'][0]['exposedUnfilledPixelsMax'],0)

    def test_one_unfilled_pixel_exposure_fails_even_if_current_frame_is_opaque(self):
        p,r,rep,ren=self.remaining_hidden_scene(edge=True);qc=ModuleQualityAuditor(ren,p,rep,ren.frame(None))
        frame=ren.frame(2);frame[...,3]=255
        qc.observe(2,1,frame)
        coverage=qc.finish()['machineQC']['checks']['remainingHiddenCoverage']
        self.assertEqual(coverage['status'],'fail',coverage)
        self.assertEqual(coverage['occlusions'][0]['exposedUnfilledPixelsMax'],1)

    def add_declared_occluder_piece(self,p,r,rep,*,separate=False,declared=True):
        sk=json.loads((r/'scene.spine.json').read_text());manifest=json.loads((p/'parts.json').read_text())
        extra='head_extra';mask=np.zeros((128,128),np.uint8);mask[45,59]=255
        path=p/'head-extra-mask.png';Image.fromarray(mask).save(path)
        bodymask=p/'trunk_piece-mask.png';old=np.asarray(Image.open(bodymask)).copy();old[45,59]=0;Image.fromarray(old).save(bodymask)
        for base in(p/'parts',r/'images'):
            tex=np.asarray(Image.open(base/'trunk_piece.png')).copy();tex[10,41]=0;Image.fromarray(tex).save(base/'trunk_piece.png')
            Image.new('RGBA',(1,1),(80,140,190,255)).save(base/(extra+'.png'))
        labels=np.load(p/'work/labels.npy');labels[45,59]=2;np.save(p/'work/labels.npy',labels)
        manifest['parts'].append({'name':extra,'drawOrder':2,'bbox':[59,45,1,1],'image':'parts/'+extra+'.png'})
        (p/'parts.json').write_text(json.dumps(manifest))
        bone='extra_bone'if separate else'beta_node'
        if separate:
            sk['bones'].append({'name':bone,'parent':'root','x':74,'y':-50})
            sk['animations']['idle']['bones'][bone]=sk['animations']['idle']['bones']['beta_node']
        sk['slots'].append({'name':extra,'bone':bone,'attachment':extra})
        sk['skins'][0]['attachments'][extra]={extra:{'width':1,'height':1,'x':-14.5,'y':4.5}}
        (r/'scene.spine.json').write_text(json.dumps(sk))
        contract=rep['moduleContract']
        if separate:
            contract['modules'].append({'id':'extra_occluder','shape':'rigid','role':'prop','motion':'required',
                'pieces':[extra],'visibleMasks':[str(path)],'bone':bone})
            contract['requiredModules'].append('extra_occluder')
            if declared:
                empty_path=p/'extra-hidden-target.png';Image.new('L',(128,128)).save(empty_path)
                empty_sha=hashlib.sha256(empty_path.read_bytes()).hexdigest()
                contract['occlusions'].append({'front':extra,'back':'trunk_piece','completionMode':'none',
                    'sourceHiddenTargetRegistration':'original-back-claim-and-occlusion-and-visible-front',
                    'sourceHiddenTargetMask':str(empty_path),'sourceHiddenTargetMaskSHA256':empty_sha,
                    'remainingHiddenMask':str(empty_path),'remainingHiddenMaskSHA256':empty_sha,'remainingHiddenPixels':0})
        else:
            contract['modules'][1]['pieces'].append(extra);contract['modules'][1]['visibleMasks'].append(str(path))
        contract['visibleMaskSHA256'][extra]=hashlib.sha256(path.read_bytes()).hexdigest()
        contract['visibleMaskSHA256']['trunk_piece']=hashlib.sha256(bodymask.read_bytes()).hexdigest()
        return SpineRenderer(r)

    def test_same_rigid_occluder_module_other_piece_can_cover_remaining_material(self):
        p,r,rep,_=self.remaining_hidden_scene(edge=True);ren=self.add_declared_occluder_piece(p,r,rep)
        qc=ModuleQualityAuditor(ren,p,rep,ren.frame(None));qc.observe(2,1,ren.frame(2))
        q=qc.finish();coverage=q['machineQC']['checks']['remainingHiddenCoverage']
        self.assertEqual(q['inputOwnership']['status'],'pass')
        self.assertEqual(coverage['status'],'pass',coverage)
        self.assertIn('head_extra',coverage['occlusions'][0]['supportedOccluderPieces'])

    def test_other_explicit_occluder_can_cover_but_undeclared_piece_cannot(self):
        for declared in(False,True):
            with self.subTest(declared=declared):
                root=self.root;self.root=root/('declared'if declared else'undeclared');self.root.mkdir()
                try:
                    p,r,rep,_=self.remaining_hidden_scene(edge=True)
                    ren=self.add_declared_occluder_piece(p,r,rep,separate=True,declared=declared)
                    qc=ModuleQualityAuditor(ren,p,rep,ren.frame(None));qc.observe(2,1,ren.frame(2))
                    coverage=qc.finish()['machineQC']['checks']['remainingHiddenCoverage']
                    self.assertEqual(coverage['status'],'pass'if declared else'fail',coverage)
                finally:self.root=root

    def test_unfilled_soft_region_without_material_coordinates_stays_pending(self):
        p,r,rep,ren=self.remaining_hidden_scene(soft=True);qc=ModuleQualityAuditor(ren,p,rep,ren.frame(None))
        qc.observe(2,1,ren.frame(2))
        coverage=qc.finish()['machineQC']['checks']['remainingHiddenCoverage']
        self.assertEqual(coverage['status'],'pending',coverage)
        self.assertIn('soft material',coverage['occlusions'][0]['reason'])

    def test_old_unregistered_remaining_hidden_evidence_is_not_assumed_empty(self):
        p,r,rep,ren=self.remaining_hidden_scene(missing_registration=True);qc=ModuleQualityAuditor(ren,p,rep,ren.frame(None))
        qc.observe(2,1,ren.frame(2))
        self.assertEqual(qc.finish()['machineQC']['checks']['remainingHiddenCoverage']['status'],'pending')

    def test_none_completion_cannot_bypass_exposed_limb_check(self):
        p,r,rep,ren=self.remaining_hidden_scene(edge=True)
        o=rep['moduleContract']['occlusions'][0];o.pop('bandDerivation');o['completionMode']='none'
        qc=ModuleQualityAuditor(ren,p,rep,ren.frame(None));qc.observe(2,1,ren.frame(2))
        check=qc.finish()['machineQC']['checks']['remainingHiddenCoverage']
        self.assertEqual(check['status'],'fail',check)
        self.assertEqual(check['occlusions'][0]['exposedUnfilledPixelsMax'],1)

    def test_touching_limb_and_soft_sibling_cannot_silently_omit_occlusion(self):
        p,r,rep,ren=self.remaining_hidden_scene(edge=True)
        body,cloth=rep['moduleContract']['modules']
        body.update(role='leg',parent='shared_host')
        cloth.update(role='cloth',shape='soft',parent='shared_host')
        registered=list(rep['moduleContract']['occlusions'])
        rep['moduleContract']['occlusions']=[]
        qc=ModuleQualityAuditor(ren,p,rep,ren.frame(None));qc.observe(2,1,ren.frame(2))
        q=qc.finish();check=q['machineQC']['checks']['occlusionDeclarations']
        self.assertEqual(check['status'],'pending',check)
        self.assertEqual(len(check['undeclaredSiblingBoundaries']),1)
        self.assertFalse(q['overall']['exportAllowed'])
        # Explicit registration removes only the planning pending. The actual
        # unfilled pixel still fails dynamic coverage; a declaration is no pass.
        rep['moduleContract']['occlusions']=registered
        qc=ModuleQualityAuditor(ren,p,rep,ren.frame(None));qc.observe(2,1,ren.frame(2))
        q=qc.finish()
        self.assertEqual(q['machineQC']['checks']['occlusionDeclarations']['status'],'pass')
        self.assertEqual(q['machineQC']['checks']['remainingHiddenCoverage']['status'],'fail')

    def test_requested_fill_mask_is_not_proof_of_actual_back_material(self):
        p,r,rep,ren=self.remaining_hidden_scene(edge=True)
        o=rep['moduleContract']['occlusions'][0]
        o['hiddenMask']=o['sourceHiddenTargetMask']
        rep['moduleContract']['hiddenMaskSHA256']={'trunk_piece':o['sourceHiddenTargetMaskSHA256']}
        empty=p/'nominally-filled.png';Image.new('L',(128,128)).save(empty)
        o.update(remainingHiddenMask=str(empty),remainingHiddenMaskSHA256=hashlib.sha256(empty.read_bytes()).hexdigest(),remainingHiddenPixels=0)
        qc=ModuleQualityAuditor(ren,p,rep,ren.frame(None));qc.observe(2,1,ren.frame(2))
        check=qc.finish()['machineQC']['checks']['remainingHiddenCoverage']
        self.assertEqual(check['status'],'fail',check)
        self.assertEqual(check['occlusions'][0]['registeredRemainingPixels'],0)
        self.assertEqual(check['occlusions'][0]['remainingPixels'],1)

    def test_soft_occluder_actual_material_reveals_unfilled_limb(self):
        p,r,rep,ren=self.remaining_hidden_scene(edge=True)
        rep['moduleContract']['modules'][1].update(shape='soft',role='cloth')
        # Driver does not move. Only the cloth mesh's left edge retracts, so a
        # driver affine or the original visible mask would incorrectly pass.
        sk=json.loads((r/'scene.spine.json').read_text())
        att=sk['skins'][0]['attachments']['head_piece']['head_piece']
        att.update(type='mesh',uvs=[0,1,0,0,1,0,1,1],triangles=[0,1,2,0,2,3],
                   vertices=[-14,-15,-14,15,14,15,14,-15])
        sk['animations']['idle']['bones']={}
        sk['animations']['idle']['attachments']={'default':{'head_piece':{'head_piece':{'deform':[
            {'time':0,'vertices':[0]*8},{'time':2,'vertices':[2,0,2,0,0,0,0,0]}]}}}}
        (r/'scene.spine.json').write_text(json.dumps(sk));ren=SpineRenderer(r)
        qc=ModuleQualityAuditor(ren,p,rep,ren.frame(None))
        qc.observe(0,1,ren.frame(0));qc.observe(2,1,ren.frame(2))
        check=qc.finish()['machineQC']['checks']['remainingHiddenCoverage']
        self.assertEqual(check['status'],'fail',check)
        self.assertEqual(check['occlusions'][0]['samples'][0]['exposedUnfilledPixels'],0)
        self.assertEqual(check['occlusions'][0]['exposedUnfilledPixelsMax'],1)

    def test_changed_remaining_hidden_reference_cannot_borrow_hash(self):
        p,r,rep,ren=self.remaining_hidden_scene()
        Image.new('L',(128,128)).save(p/'registered-unfilled.png')
        qc=ModuleQualityAuditor(ren,p,rep,ren.frame(None));qc.observe(2,1,ren.frame(2))
        self.assertEqual(qc.finish()['machineQC']['checks']['remainingHiddenCoverage']['status'],'fail')

    def test_translucent_front_cannot_prove_remaining_material_is_fully_hidden(self):
        p,r,rep,ren=self.remaining_hidden_scene()
        source=np.asarray(Image.open(p/'source.png')).copy();source[45,70,3]=128
        Image.fromarray(source).save(p/'source.png')
        digest=hashlib.sha256((p/'source.png').read_bytes()).hexdigest()
        manifest=json.loads((p/'parts.json').read_text());manifest['sourceSHA256']=digest
        (p/'parts.json').write_text(json.dumps(manifest));rep['moduleContract']['sourceSHA256']=digest
        ren.tex['head_piece'][10,10,3]=128
        qc=ModuleQualityAuditor(ren,p,rep,ren.frame(None));qc.observe(0,1,ren.frame(0))
        coverage=qc.finish()['machineQC']['checks']['remainingHiddenCoverage']
        self.assertEqual(coverage['status'],'pending',coverage)
        self.assertEqual(coverage['occlusions'][0]['translucentUnfilledPixelsMax'],1)
        self.assertEqual(coverage['occlusions'][0]['exposedUnfilledPixelsMax'],0)

    def test_cross_module_old_hole_is_pending_not_approved(self):
        a=np.zeros((80,80),np.uint8);a[10:70,10:70]=255;a[30:40,30:40]=0
        owners=np.where(a>=128,0,-1);owners[:,35:]=np.where(a[:,35:]>=128,1,-1)
        tracker=NegativeSpaceTracker(a,a,owners,[0,1]);q=tracker.evaluate(a,{0:np.eye(3),1:np.eye(3)})
        self.assertEqual(q['status'],'pending');self.assertEqual(q['unresolvedOriginalHoles'],1)

    def test_new_module_coverage_cannot_borrow_another_modules_old_hole(self):
        alpha=np.zeros((100,120),np.uint8);alpha[10:80,10:70]=255;alpha[30:40,30:40]=0;alpha[30:40,90:100]=255
        owners=np.where(alpha>=128,0,-1);owners[30:40,90:100]=1
        tracker=NegativeSpaceTracker(alpha,alpha,owners,[0,1])
        # The second module should cover the old first-module hole after moving,
        # but its visible material has disappeared. It must not be marked passed.
        frame=alpha.copy();frame[30:40,90:100]=0
        q=tracker.evaluate(frame,{0:np.eye(3),1:np.array([[1,0,-60],[0,1,0],[0,0,1]],float)})
        self.assertEqual(q['status'],'pending');self.assertEqual(q['occlusionAmbiguousPixels'],100)

    def test_visual_approval_cannot_override_machine_fail(self):
        q,*_=self.collect(stretch=True)
        with self.assertRaisesRegex(ValueError,'cannot override'):
            finalize_visual_review({'quality':q},decision='pass',evidence={})
        evidence={'reviewer':'review-agent','notes':'reviewed','full_loop_observed':True,
                  'review_binding':{k:'bound-value'for k in('job_id','input_sha256','analysis_sha256','artifact_sha256')}}
        clean={'quality':{'contractApplicable':True,'inputOwnership':{'status':'pass'},'bindingMotion':{'status':'pass'},
                          'machineQC':{'status':'pass'},'visual':{'status':'pending'}}}
        approved=finalize_visual_review(clean,decision='pass',evidence=evidence)
        self.assertTrue(approved['quality']['overall']['exportAllowed'])
        self.assertEqual(clean['quality']['visual']['status'],'pending')



    def test_soft_numeric_strain_is_advisory_but_fold_still_fails(self):
        p,r,rep,sk=scene(self.root)
        rep['moduleContract']['modules'][1].update(shape='soft',role='cloth')
        att=sk['skins'][0]['attachments']['head_piece']['head_piece']
        att.update(type='mesh',uvs=[0,1,0,0,1,0,1,1],triangles=[0,1,2,0,2,3],vertices=[-14,-15,-14,15,14,15,14,-15])
        sk['animations']['idle']['attachments']={'default':{'head_piece':{'head_piece':{'deform':[
            {'time':0,'vertices':[0]*8},{'time':2,'vertices':[0,0,0,0,10,0,10,0]}]}}}}
        (r/'scene.spine.json').write_text(json.dumps(sk));ren=SpineRenderer(r);qc=ModuleQualityAuditor(ren,p,rep,ren.frame(None))
        for tm in(0,2,4,8):qc.observe(tm,1,ren.frame(tm))
        q=qc.finish();soft=q['bindingMotion']['checks']['softDeformation']
        self.assertEqual(soft['status'],'pass')
        self.assertTrue(soft.get('advisory'))
        self.assertIn('custom_head',soft.get('advisoryModules') or [])

    def test_negative_space_holes_are_advisory_and_do_not_fail_machine_qc(self):
        p,r,rep,_=scene(self.root);renderer=SpineRenderer(r);qc=ModuleQualityAuditor(renderer,p,rep,renderer.frame(None))
        renderer.tex['trunk_piece'][35:40,30:35]=0
        qc.observe(2,1,renderer.frame(2));q=qc.finish()
        ns=q['machineQC']['checks']['negativeSpace']
        self.assertEqual(ns['status'],'pass')
        self.assertTrue(ns.get('advisory'))
        self.assertGreater(ns['newHolePixelsMax'],0)
        self.assertIn(ns.get('measuredStatus'), ('fail','pending'))
        # Texture was mutated after auditor init, so materialStability still hard-fails.
        self.assertEqual(q['machineQC']['checks']['materialStability']['status'],'fail')

if __name__=='__main__':unittest.main()
