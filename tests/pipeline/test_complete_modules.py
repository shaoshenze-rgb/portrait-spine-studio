"""Complete-module compiler behavior, independent of character filenames or templates."""
import copy
import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.studio.module_contract import validate_module_contract, materialize_module_contract
from src.studio.vision_schema import VisionValidationError, VisionReviewRequired, validate_vision_result, vision_result_to_semantic_hints
from src.studio.auto_spec import build_default_specs, apply_rig_edits
from src.pipeline.split_parts import split_parts
from src.pipeline.build_rig import build_rig
from src.render.renderer import SpineRenderer

ROOT = Path(__file__).resolve().parents[2]


def fixture(path, local_fill=False):
    w,h=128,160
    rgba=np.zeros((h,w,4),np.uint8)
    def box(x0,y0,x1,y1):
        return [[x0/w,y0/h],[x1/w,y0/h],[x1/w,y1/h],[x0/w,y1/h]]
    def module(mid,role,kind,parent,pivot,rects,color,order,motion=None,**kwargs):
        pieces=[]
        for i,rect in enumerate(rects):
            x0,y0,x1,y1=rect;rgba[y0:y1,x0:x1]=color
            pieces.append({'id':mid+'_p'+str(i),'polygons':[box(*rect)],'draw_order':order+i})
        item={'id':mid,'role':role,'motion_type':kind,'parent':parent,'pivot':[pivot[0]/w,pivot[1]/h],
              'pieces':pieces,'confidence':.95,**kwargs}
        if motion:item['motion']=motion
        return item
    cape=module('cape','cape','soft','body',(44,56),[(10,56,44,82),(10,90,44,129)],(30,70,190,255),0,
                centerline=[[44/w,56/h],[25/w,89/h],[16/w,125/h]])
    body=module('body','torso','rigid',None,(61,104),[(44,49 if local_fill else 50,79,115)],(20,170,60,255),10)
    head=module('face','head','rigid','body',(61,50),[(47,18,76,50)],(200,90,100,255),20)
    hand=module('hand','hand','rigid','body',(79,64),[(79,55,100,73)],(130,100,180,255),30,representation='mesh')
    blade=module('blade','weapon','rigid','hand',(100,61),[(100,57,121,62),(100,67,121,73)],(200,180,40,255),40,motion='follow')
    sparks=module('spark','effect_group','static',None,(7,7),[(5,5,8,8),(115,18,118,22)],(250,160,60,255),50,
                  static_reason='intentional disconnected floating light group')
    Image.fromarray(rgba).save(path)
    modules=[body,head,hand,blade,cape,sparks]
    connections=[{'id':'neck','type':'joint','a':'body','b':'face','point':head['pivot']},
        {'id':'arm','type':'joint','a':'body','b':'hand','point':hand['pivot']},
        {'id':'hold','type':'grip','a':'hand','b':'blade','point':blade['pivot']},
        {'id':'cape_root','type':'attachment','a':'body','b':'cape','point':cape['pivot']}]
    return {'version':2,'input_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'image_size':[w,h],
            'coord_mode':'normalized','pose_type':'airborne','anatomy':'humanoid_bust','modules':modules,
            'connections':connections,'occlusions':([{'front':'face_p0','back':'body_p0',
                'polygons':[box(47,49,76,50)],'completion':'local'}] if local_fill else []),'uncertainties':[]}


class TestCompleteModules(unittest.TestCase):
    def setUp(self):
        work=ROOT/'.studio-work';work.mkdir(exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=work,prefix='complete-module-test-')
        self.addCleanup(self.tmp.cleanup)
        self.d=Path(self.tmp.name);self.image=self.d/'arbitrary.png';self.contract=fixture(self.image)

    def validate(self,c=None):
        return validate_module_contract(c or self.contract,input_sha256=self.contract['input_sha256'],image_size=(128,160))

    def compile(self,c=None, name='run'):
        c=c or self.contract
        hints=vision_result_to_semantic_hints(c,image_path=self.image,mask_dir=self.d/name/'masks')
        ps,rs=build_default_specs(self.image,character='anything',semantic_hints=hints)
        split=split_parts(self.image,ps['parts'],self.d/name/'parts',min_island_px=0,feather_px=0,module_contract=ps['module_contract'])
        made=build_rig(self.d/name/'parts',rs,self.d/name/'rig')
        return ps,rs,split,made,SpineRenderer(self.d/name/'rig')

    def test_end_to_end_rigid_modules_and_shared_piece_motion(self):
        self.assertEqual(validate_vision_result(self.contract,input_sha256=self.contract['input_sha256'],image_size=(128,160))['version'],2)
        ps,rs,split,made,r=self.compile()
        self.assertEqual(split['rest_diff']['maxRGBA'],0)
        self.assertEqual(made['report']['moduleContract']['requiredModules'],['body','face','hand','blade','cape','spark'])
        self.assertNotIn('linear_y',{s.get('recipe') for s in rs['skin'].values()})
        r.pose(None)
        slots={s['name']:s for s in r.sk.slots}
        baseline={k:r.sk.attachment_vertices(v)[0] for k,v in slots.items()}
        max_body_relative=0.0
        for t in np.linspace(0,8,17):
            r.pose(float(t))
            for module in made['report']['moduleContract']['modules']:
                if module['shape']=='rigid':
                    b=r.sk.by_name[module['bone']];mat=b.matrix()[:,:2]
                    np.testing.assert_allclose(mat.T@mat,np.eye(2),atol=1e-9)
                    for piece in module['pieces']:
                        pts=r.sk.attachment_vertices(slots[piece])[0];ref=baseline[piece]
                        np.testing.assert_allclose(np.linalg.norm(pts-pts[0],axis=1),np.linalg.norm(ref-ref[0],axis=1),atol=.001)
            # Disconnected visible sword portions share exactly one moving rigid matrix.
            self.assertEqual(slots['piece_blade_p0']['bone'],slots['piece_blade_p1']['bone'])
            head=r.sk.by_name['module_face'];max_body_relative=max(max_body_relative,abs(head.rotation))
        self.assertGreater(max_body_relative,.1)
        self.assertEqual(int(np.abs(r.frame(0).astype(int)-r.frame(None).astype(int)).max()),0)
        self.assertEqual(int(np.abs(r.frame(8).astype(int)-r.frame(None).astype(int)).max()),0)

    def test_intentional_disconnected_effect_is_one_module_not_island_noise(self):
        _,rs,_,made,r=self.compile()
        group=next(m for m in made['report']['moduleContract']['modules'] if m['id']=='spark')
        self.assertEqual(len(group['pieces']),2)
        self.assertEqual({rs['skin'][p]['bone'] for p in group['pieces']},{'module_spark'})
        r.pose(None);matrix=r.sk.by_name['module_spark'].matrix().copy()
        r.pose(3.5);np.testing.assert_array_equal(r.sk.by_name['module_spark'].matrix(),matrix)
        self.assertEqual(next(m for m in made['report']['moduleContract']['modules'] if m['id']=='cape')['pieces'],['piece_cape_p0','piece_cape_p1'])

    def test_no_color_gradient_split_and_unassigned_pixels_stop(self):
        # A high-contrast tattoo inside body does not become a new module or moving edge.
        im=np.asarray(Image.open(self.image)).copy();im[70:90,55:65,:3]=[255,0,255];Image.fromarray(im).save(self.image)
        self.contract['input_sha256']=hashlib.sha256(self.image.read_bytes()).hexdigest()
        ps,_,_,_,_=self.compile()
        self.assertEqual(len(ps['parts']),9)
        bad=copy.deepcopy(self.contract);bad['modules'][0]['pieces'][0]['polygons'][0]=[[44/128,50/160],[78/128,50/160],[78/128,115/160],[44/128,115/160]]
        with self.assertRaisesRegex(VisionReviewRequired,'未归属'):
            materialize_module_contract(bad,image_path=self.image,mask_dir=self.d/'missing')

    def test_production_cannot_silently_use_old_geometry(self):
        with self.assertRaises(VisionReviewRequired) as ctx:
            build_default_specs(self.image, mask_dir=self.d/'no_plan')
        self.assertEqual(ctx.exception.code,'MODULE_PLAN_REQUIRED')

    def test_changed_masks_cannot_borrow_old_contract_receipt(self):
        hints=vision_result_to_semantic_hints(self.contract,image_path=self.image,mask_dir=self.d/'bound_masks')
        a,b=Path(hints['module_masks']['face_p0']),Path(hints['module_masks']['body_p0'])
        original_a,original_b=a.read_bytes(),b.read_bytes();a.write_bytes(original_b);b.write_bytes(original_a)
        with self.assertRaisesRegex(VisionReviewRequired,'掩码已脱离'):
            build_default_specs(self.image,semantic_hints=hints)

    def test_declared_input_defect_stops_without_becoming_a_module_or_deleting_pixels(self):
        bad=copy.deepcopy(self.contract);bad['input_issues']=[{'id':'border','kind':'background_residue',
            'polygons':[[[0,0],[1,0],[1,.02],[0,.02]]],'notes':'visible unwanted border in the supplied matte'}]
        before=self.image.read_bytes()
        with self.assertRaises(VisionReviewRequired) as ctx:
            materialize_module_contract(bad,image_path=self.image,mask_dir=self.d/'bad_input')
        self.assertEqual(ctx.exception.code,'INPUT_REVIEW_REQUIRED')
        self.assertEqual(ctx.exception.details['input_issues'][0]['id'],'border')
        self.assertFalse((self.d/'bad_input').exists())
        self.assertEqual(self.image.read_bytes(),before)

    def test_overlapping_pieces_require_explicit_occlusion(self):
        bad=copy.deepcopy(self.contract)
        bad['modules'][1]['pieces'][0]['polygons']=copy.deepcopy(bad['modules'][0]['pieces'][0]['polygons'])
        with self.assertRaisesRegex(VisionReviewRequired,'遮挡'):
            materialize_module_contract(bad,image_path=self.image,mask_dir=self.d/'overlap')

    def test_declared_precomposited_effect_requires_independent_source_layers(self):
        bad=copy.deepcopy(self.contract)
        bad['input_issues']=[{'id':'effect_on_surface','kind':'precomposited_overlap',
            'polygons':[[[.4,.4],[.6,.4],[.6,.6],[.4,.6]]],
            'notes':'Effect and a differently moving rigid surface share flattened opaque RGB; clean underlying surface and effect alpha are unavailable.'}]
        before=self.image.read_bytes()
        with self.assertRaises(VisionReviewRequired) as ctx:
            materialize_module_contract(bad,image_path=self.image,mask_dir=self.d/'flattened')
        self.assertEqual(ctx.exception.code,'INPUT_REVIEW_REQUIRED')
        self.assertIn('原分层',str(ctx.exception))
        self.assertEqual(ctx.exception.details['input_issues'][0]['kind'],'precomposited_overlap')
        self.assertFalse((self.d/'flattened').exists())
        self.assertEqual(self.image.read_bytes(),before)
        # Same ordinary V2 input without a declared source-layer ambiguity stays
        # supported. The generator does not hallucinate automatic semantic detection.
        _,_,split,_,_=self.compile(name='clean_layers')
        self.assertEqual(split['rest_diff']['maxRGBA'],0)

    def test_no_zero_gain_cross_weights_scaling_or_old_rig_edit(self):
        _,rs,_,_,_=self.compile()
        for gain in (0,-1,2,float('nan')):
            with self.assertRaises(VisionReviewRequired):build_rig(self.d/'run/parts',rs,self.d/'badgain',gain=gain)
        variants=[]
        scale=copy.deepcopy(rs);scale['bones'][1]['scaleX']=1.01;variants.append(scale)
        mixed=copy.deepcopy(rs);mixed['skin']['piece_body_p0']={'type':'mesh','recipe':'linear_y','module_id':'body'};variants.append(mixed)
        edited=copy.deepcopy(rs);edited['motion']={'module_body':{'sx':[(.1,1,0)]}};variants.append(edited)
        for bad in variants:
            with self.assertRaises(VisionReviewRequired):build_rig(self.d/'run/parts',bad,self.d/'bad')
        with self.assertRaises(VisionReviewRequired) as ctx:apply_rig_edits(rs,{'pose_type':'grounded'})
        self.assertEqual(ctx.exception.code,'MODULE_REPLAN_REQUIRED')

    def test_no_body_static_or_soft_or_per_module_curve_escape(self):
        for patch in ({'motion_type':'static','motion':'static','static_reason':'make pass'},
                      {'motion_type':'soft','motion':'soft_bend'}, {'amplitude':0}):
            bad=copy.deepcopy(self.contract);bad['modules'][0].update(patch)
            with self.assertRaises(VisionValidationError):self.validate(bad)
        bad=copy.deepcopy(self.contract);bad['modules'][1]['motion']='follow';bad['modules'][2]['motion']='follow'
        with self.assertRaisesRegex(VisionReviewRequired,'多个硬'):self.validate(bad)

    def test_no_missing_parent_connection_or_unsupported_dual_end(self):
        bad=copy.deepcopy(self.contract);bad['connections']=bad['connections'][:-1]
        with self.assertRaisesRegex(VisionReviewRequired,'接触'):self.validate(bad)
        bad=copy.deepcopy(self.contract);bad['connections'][2]['type']='dual_grip'
        with self.assertRaisesRegex(VisionReviewRequired,'连接类型'):self.validate(bad)
        bad=copy.deepcopy(self.contract);bad['modules'][2]['parent']='cape'
        with self.assertRaisesRegex(VisionReviewRequired,'软链'):self.validate(bad)

    def test_bounded_local_completion_never_copies_front_texture(self):
        self.contract=fixture(self.image,local_fill=True)
        _,_,split,_,_=self.compile()
        part=next(p for p in split['parts_json']['parts'] if p['pieceId']=='body_p0')
        self.assertEqual(part['completionMode'],'local')
        self.assertGreater(part['hiddenPixels'],0)
        arr=np.asarray(Image.open(self.d/'run/parts'/part['image']))
        x,y,_,_=part['bbox']
        np.testing.assert_array_equal(arr[49-y,60-x,:3],[20,170,60])
        self.assertEqual(split['rest_diff']['maxRGBA'],0)
        # Enlarging a hidden request beyond the common narrow band must not silently clip it.
        bad=copy.deepcopy(self.contract)
        bad['modules'][0]['pieces'][0]['polygons'][0][0][1]=.15
        bad['modules'][0]['pieces'][0]['polygons'][0][1][1]=.15
        bad['occlusions'][0]['polygons']=[[[47/128,.15],[76/128,.15],[76/128,50/160],[47/128,50/160]]]
        with self.assertRaises((VisionReviewRequired,ValueError)):self.compile(bad,'large_hidden')

    def test_large_occlusion_zone_cannot_expand_beyond_claimed_back_shape(self):
        self.contract=fixture(self.image,local_fill=True)
        self.contract['occlusions'][0]['polygons']=[[[0,0],[1,0],[1,1],[0,1]]]
        hints=materialize_module_contract(self.contract,image_path=self.image,mask_dir=self.d/'limited_hidden')
        mask=np.asarray(Image.open(hints['module_hidden_masks']['body_p0']))>0
        self.assertGreaterEqual(int(mask.sum()),29)
        self.assertTrue(mask[49,47:76].all())

    def test_explicit_derived_band_is_local_and_records_unfilled_material(self):
        self.contract=fixture(self.image,local_fill=True)
        c=copy.deepcopy(self.contract)
        for point in c['modules'][0]['pieces'][0]['polygons'][0][:2]:point[1]=.15
        c['occlusions'][0]['polygons']=[[[47/128,.15],[76/128,.15],
                                       [76/128,50/160],[47/128,50/160]]]
        c['occlusions'][0]['derive_band']=True
        hints=materialize_module_contract(c,image_path=self.image,mask_dir=self.d/'auto_band')
        band=hints['module_projection']['occlusions'][0]['bandDerivation']
        self.assertGreater(band['unfilledEligiblePixels'],0)
        self.assertEqual(band['completionStatus'],'partial_pending_visual_review')
        self.assertEqual(band['eligiblePixels'],band['materializedPixels']+band['unfilledEligiblePixels'])
        occ=hints['module_projection']['occlusions'][0]
        target=np.asarray(Image.open(occ['sourceHiddenTargetMask']))>127
        remaining=np.asarray(Image.open(occ['remainingHiddenMask']))>127
        mask=np.asarray(Image.open(hints['module_hidden_masks']['body_p0']))>127
        np.testing.assert_array_equal(remaining,target&~mask)
        self.assertEqual(occ['remainingHiddenPixels'],int(remaining.sum()))
        for label in ('sourceHiddenTarget','remainingHidden'):
            self.assertEqual(occ[label+'MaskSHA256'],hashlib.sha256(Path(occ[label+'Mask']).read_bytes()).hexdigest())
        yy,_=np.nonzero(mask)
        self.assertGreaterEqual(yy.min(),50-band['maxDepthPx'])
        _,_,split,_,_=self.compile(c,'auto_band_compiled')
        self.assertEqual(split['rest_diff']['maxRGBA'],0)
        part=next(p for p in split['parts_json']['parts'] if p['pieceId']=='body_p0')
        tex=np.asarray(Image.open(self.d/'auto_band_compiled/parts'/part['image']))
        x,y,_,_=part['bbox']
        np.testing.assert_array_equal(tex[49-y,60-x,:3],[20,170,60])
        conflict=copy.deepcopy(c)
        conflict['occlusions'][0]['completion_polygons']=conflict['occlusions'][0]['polygons']
        with self.assertRaises(VisionValidationError):self.validate(conflict)
        conflict=copy.deepcopy(c);conflict['occlusions'][0]['completion']='none'
        with self.assertRaises(VisionValidationError):self.validate(conflict)
        rgba=np.asarray(Image.open(self.image)).copy();rgba[49,60,3]=128
        Image.fromarray(rgba).save(self.image)
        translucent=copy.deepcopy(c)
        translucent['input_sha256']=hashlib.sha256(self.image.read_bytes()).hexdigest()
        hints=materialize_module_contract(translucent,image_path=self.image,mask_dir=self.d/'translucent_band')
        self.assertEqual(hints['module_projection']['occlusions'][0]['bandDerivation']['unsupportedTranslucentPixels'],1)
        self.assertEqual(np.asarray(Image.open(hints['module_hidden_masks']['body_p0']))[49,60],0)
        self.assertEqual(np.asarray(Image.open(hints['module_projection']['occlusions'][0]['remainingHiddenMask']))[49,60],255)
        _,_,split,_,_=self.compile(translucent,'translucent_band_compiled')
        self.assertEqual(split['rest_diff']['maxRGBA'],0)

    def test_explicit_completion_subset_is_bound_and_cannot_escape_occlusion(self):
        self.contract=fixture(self.image,local_fill=True)
        occ=self.contract['occlusions'][0]
        occ['completion_polygons']=[[[50/128,49/160],[60/128,49/160],[60/128,50/160],[50/128,50/160]]]
        hints=materialize_module_contract(self.contract,image_path=self.image,mask_dir=self.d/'subset')
        mask=np.asarray(Image.open(hints['module_hidden_masks']['body_p0']))>0
        self.assertGreaterEqual(int(mask.sum()),10)
        self.assertTrue(mask[49,50:60].all())
        self.assertIn('completionPolygons',hints['module_projection']['occlusions'][0])
        self.assertIn('body_p0',hints['module_projection']['hiddenMaskSHA256'])
        bad=copy.deepcopy(self.contract);bad['occlusions'][0]['completion_polygons'][0][0][0]=0
        with self.assertRaisesRegex(VisionReviewRequired,'超出'):
            materialize_module_contract(bad,image_path=self.image,mask_dir=self.d/'escaped_subset')

    def test_provided_without_canvas_material_is_rejected_not_read(self):
        bad=copy.deepcopy(self.contract);bad['occlusions']=[{'front':'face_p0','back':'body_p0',
            'polygons':[[[.3,.3],[.4,.3],[.4,.4]]],'completion':'provided'}]
        with self.assertRaisesRegex(VisionReviewRequired,'provided'):self.validate(bad)

    def test_none_and_subset_completion_register_entire_hidden_domain(self):
        for mode in ('none', 'subset'):
            c=fixture(self.image,local_fill=True)
            if mode=='none':c['occlusions'][0]['completion']='none'
            else:c['occlusions'][0]['completion_polygons']=[[[50/128,49/160],[60/128,49/160],
                                                            [60/128,50/160],[50/128,50/160]]]
            hints=materialize_module_contract(c,image_path=self.image,mask_dir=self.d/mode)
            o=hints['module_projection']['occlusions'][0]
            target=np.asarray(Image.open(o['sourceHiddenTargetMask']))>0
            remaining=np.asarray(Image.open(o['remainingHiddenMask']))>0
            self.assertEqual(int(target.sum()),29,mode)
            self.assertGreater(int(remaining.sum()),0,mode)
            if mode=='none':np.testing.assert_array_equal(remaining,target)
            else:self.assertLess(int(remaining.sum()),int(target.sum()))

    def test_local_occlusion_preserves_auto_seams_outside_its_region(self):
        import base64
        baseline=materialize_module_contract(self.contract,image_path=self.image,mask_dir=self.d/'baseline-seams')
        c=copy.deepcopy(self.contract)
        for module in c['modules']:
            for piece in module['pieces']:
                raw=Path(baseline['module_masks'][piece['id']]).read_bytes()
                piece['mask_png_base64']=base64.b64encode(raw).decode()
                piece['mask_sha256']=hashlib.sha256(raw).hexdigest()
        zone=[[[55/128,47/160],[65/128,47/160],[65/128,50/160],[55/128,50/160]]]
        c['occlusions']=[{'front':'face_p0','back':'body_p0','polygons':zone,'completion':'local'}]
        local=materialize_module_contract(c,image_path=self.image,mask_dir=self.d/'local-plus-auto')
        before=np.asarray(Image.open(baseline['module_hidden_masks']['body_p0']))>0
        after=np.asarray(Image.open(local['module_hidden_masks']['body_p0']))>0
        from src.studio.vision_schema import rasterize_polygons
        region=rasterize_polygons(zone,128,160)
        face=np.asarray(Image.open(local['module_masks']['face_p0']))>0
        # A small repair must not erase safe automatic underlap at the other
        # side of the SAME contact. The explicit region still owns its policy.
        self.assertTrue((before&face&~region).any())
        self.assertFalse((before&face&~region&~after).any())
        self.assertTrue(any(o.get('autoSeamUnderlap') and o['front']=='face_p0' and o['back']=='body_p0'
                            for o in local['module_projection']['occlusions']))
        self.assertEqual(self.compile(c,'local-rest')[2]['rest_diff']['maxRGBA'],0)
        none=copy.deepcopy(c);none['occlusions'][0]['completion']='none'
        disabled=materialize_module_contract(none,image_path=self.image,mask_dir=self.d/'none-optout')
        path=disabled['module_hidden_masks'].get('body_p0')
        hidden=np.asarray(Image.open(path))>0 if path else np.zeros_like(face)
        self.assertFalse((hidden&face).any(),'explicit none must remain an opt-out')

    def test_declared_band_depth_reaches_splitter_without_default_clipping(self):
        # At this scale the declared 48px/1080 band exceeds the old default.
        # The source and visible ownership stay unchanged; only hidden material
        # behind the head is generated. Previously split_parts raised ValueError.
        Image.open(self.image).resize((256,320),Image.Resampling.NEAREST).save(self.image)
        c=copy.deepcopy(self.contract)
        c['image_size']=[256,320]
        c['input_sha256']=hashlib.sha256(self.image.read_bytes()).hexdigest()
        for p in c['modules'][0]['pieces'][0]['polygons'][0][:2]:p[1]=45/160
        c['occlusions']=[{'front':'face_p0','back':'body_p0',
                         'polygons':[[[47/128,45/160],[76/128,45/160],[76/128,50/160],[47/128,50/160]]],
                         'completion':'local','derive_band':True,'band_depth_px_at_1080':48}]
        ps,rs,split,_,_=self.compile(c,'declared-depth')
        body=next(p for p in ps['parts'] if p['piece_id']=='body_p0')
        self.assertGreater(body['depth'],6)
        oc=next(o for o in rs['moduleContract']['occlusions'] if not o.get('autoSeamUnderlap'))
        self.assertEqual(oc['maxLocalDepthPx'],body['depth'])
        self.assertEqual(split['rest_diff']['maxRGBA'],0)
        hidden=next(p for p in split['parts_json']['parts'] if p.get('pieceId')=='body_p0')
        self.assertGreater(hidden['hiddenPixels'],0)

    def test_visible_alpha_and_texture_are_preserved_without_legacy_normalization(self):
        im=np.asarray(Image.open(self.image)).copy()
        im[65,60]=[32,168,79,250];im[66,60]=[120,98,160,80]
        Image.fromarray(im).save(self.image);self.contract['input_sha256']=hashlib.sha256(self.image.read_bytes()).hexdigest()
        _,_,split,_,_=self.compile()
        self.assertEqual(split['rest_diff']['maxRGBA'],0)
        part=next(p for p in split['parts_json']['parts'] if p['pieceId']=='body_p0')
        x,y,_,_=part['bbox'];arr=np.asarray(Image.open(self.d/'run/parts'/part['image']))
        np.testing.assert_array_equal(arr[65-y,60-x],im[65,60])
        np.testing.assert_array_equal(arr[66-y,60-x],im[66,60])
        self.assertTrue(split['parts_json']['preserveSourceRGBA'])
        self.assertEqual(split['parts_json']['opaqueThreshold'],256)

    def test_soft_module_bends_without_flips_and_rigid_js_matches(self):
        _,rs,_,made,r=self.compile()
        slots={s['name']:s for s in r.sk.slots};r.pose(None)
        baseline={p:r.sk.attachment_vertices(slots[p])[0] for p in ('piece_cape_p0','piece_cape_p1')}
        times=np.linspace(0,8,17).tolist();expected=[];max_deformation=0.0
        for time in times:
            r.pose(time);expected.append([[b.worldX,b.worldY,b.a,b.b,b.c,b.d] for b in r.sk.bones])
            for name,ref in baseline.items():
                pts,_uv,tris,_=r.sk.attachment_vertices(slots[name]);triangles=np.asarray(tris).reshape(-1,3)
                for a,b,c in triangles:
                    orig=(ref[b,0]-ref[a,0])*(ref[c,1]-ref[a,1])-(ref[b,1]-ref[a,1])*(ref[c,0]-ref[a,0])
                    moved=(pts[b,0]-pts[a,0])*(pts[c,1]-pts[a,1])-(pts[b,1]-pts[a,1])*(pts[c,0]-pts[a,0])
                    self.assertGreater(orig*moved,0)
                    max_deformation=max(max_deformation,abs(moved/orig-1))
        self.assertGreater(max_deformation,1e-5)
        self.assertLess(max_deformation,.05)
        script="import fs from 'node:fs';import {Spine42Skeleton} from './src/player/spine42-runtime.js';const p=JSON.parse(fs.readFileSync(0,'utf8'));const s=new Spine42Skeleton(p.data);console.log(JSON.stringify(p.times.map(t=>{s.apply('idle',t);return s.bones.map(b=>[b.worldX,b.worldY,b.a,b.b,b.c,b.d]);})));"
        run=subprocess.run(['node','--input-type=module','-e',script],input=json.dumps({'data':r.sk.json,'times':times}),text=True,capture_output=True,cwd=ROOT,check=True)
        np.testing.assert_allclose(json.loads(run.stdout),expected,atol=1e-8,rtol=1e-9)

    def test_grounded_rigid_leg_joint_uses_no_stretch_and_unreachable_stops(self):
        w,h=128,160;rgba=np.zeros((h,w,4),dtype=np.uint8)
        from PIL import ImageDraw
        image=Image.fromarray(rgba);draw=ImageDraw.Draw(image)
        shapes={'body':[(50,40),(78,40),(78,78),(50,78)],
                'face':[(54,10),(74,10),(74,40),(54,40)],
                'upper':[(56,78),(67,78),(52,112),(41,108)],
                'lower':[(41,108),(52,112),(68,145),(56,145)],
                'foot':[(54,145),(74,145),(74,156),(54,156)]}
        # Draw through pixel-center masks, matching the declared partition exactly.
        from src.studio.vision_schema import rasterize_polygons
        modules=[]
        roles={'body':'torso','face':'head','upper':'upper_leg','lower':'lower_leg','foot':'foot'}
        parents={'body':None,'face':'body','upper':'body','lower':'upper','foot':None}
        pivots={'body':(64,75),'face':(64,40),'upper':(61,78),'lower':(47,109),'foot':(62,145)}
        for order,(name,poly) in enumerate(shapes.items()):
            polygon=[[x/w,y/h] for x,y in poly];mask=rasterize_polygons([polygon],w,h)
            rgba[mask]=[30+order*35,70+order*20,140,255]
            m={'id':name,'role':roles[name],'motion_type':'static' if name=='foot' else 'rigid',
               'parent':parents[name],'pivot':[pivots[name][0]/w,pivots[name][1]/h],
               'confidence':.95,'pieces':[{'id':name+'_p0','draw_order':order,'polygons':[polygon]}]}
            if name=='foot':m['static_reason']='declared ground contact'
            modules.append(m)
        Image.fromarray(rgba).save(self.image)
        contract={'version':2,'input_sha256':hashlib.sha256(self.image.read_bytes()).hexdigest(),
            'image_size':[w,h],'coord_mode':'normalized','pose_type':'grounded','anatomy':'humanoid_full',
            'modules':modules,'uncertainties':[],'occlusions':[],'connections':[
                {'id':'neck','type':'joint','a':'body','b':'face','point':[64/w,40/h]},
                {'id':'hip','type':'joint','a':'body','b':'upper','point':[61/w,78/h]},
                {'id':'knee','type':'joint','a':'upper','b':'lower','point':[47/w,109/h]},
                {'id':'anchor','type':'foot_anchor','a':'upper','b':'foot','point':[62/w,145/h],'chain':['upper','lower']}]}
        _,rs,_,made,r=self.compile(contract,'grounded')
        self.assertFalse(made['report']['ik'][0].get('stretch',False))
        self.assertLess(made['report']['moduleMotionPreflight']['rigidAnchorMaxDriftPx'],.001)
        foot_positions=[];hip_positions=[]
        for time in np.linspace(0,8,33):
            r.pose(float(time));b=r.sk.by_name['module_foot'];foot_positions.append([b.worldX,b.worldY]);hip=r.sk.by_name['module_upper'];hip_positions.append([hip.worldX,hip.worldY])
            for bone in r.sk.bones:np.testing.assert_allclose(bone.matrix()[:,:2].T@bone.matrix()[:,:2],np.eye(2),atol=1e-8)
        self.assertLess(np.ptp(foot_positions,axis=0).max(),1e-8)
        self.assertGreater(np.ptp(hip_positions,axis=0).max(),.1)
        bent_png=self.image.read_bytes()
        straight=copy.deepcopy(contract)
        su=[(56,78),(67,78),(67,112),(56,112)]
        sl=[(56,112),(67,112),(67,145),(56,145)]
        straight['modules'][2]['pieces'][0]['polygons']=[[[x/w,y/h] for x,y in su]]
        straight['modules'][3]['pieces'][0]['polygons']=[[[x/w,y/h] for x,y in sl]]
        straight['modules'][3]['pivot']=[61.5/w,111.5/h]
        straight['connections'][2]['point']=[61.5/w,111.5/h]
        srgba=np.zeros((h,w,4),dtype=np.uint8)
        for order,(name,poly) in enumerate({'body':shapes['body'],'face':shapes['face'],'upper':su,'lower':sl,'foot':shapes['foot']}.items()):
            srgba[rasterize_polygons([[[x/w,y/h] for x,y in poly]],w,h)]=[30+order*35,70+order*20,140,255]
        Image.fromarray(srgba).save(self.image)
        straight['input_sha256']=hashlib.sha256(self.image.read_bytes()).hexdigest()
        _,straight_spec,_,straight_build,straight_renderer=self.compile(straight,'straight')
        self.assertLess(straight_build['report']['moduleMotionPreflight']['rigidAnchorMaxDriftPx'],.001)
        self.assertEqual(straight_spec['bones'][4]['pos'],[61.5,111.5])
        body_y=[];head_angle=[]
        for time in np.linspace(0,8,33):
            straight_renderer.pose(float(time))
            body_y.append(straight_renderer.sk.by_name['module_body'].worldY)
            head_angle.append(straight_renderer.sk.by_name['module_face'].rotation)
        self.assertLessEqual(max(body_y),body_y[0]+1e-8)
        self.assertGreater(np.ptp(body_y),.1)
        self.assertGreater(np.ptp(head_angle),.1)
        for time in np.linspace(0,8,65):
            straight_renderer.pose(float(time),gain=1.5)
            foot=straight_renderer.sk.by_name['module_foot']
            tip=straight_renderer.sk.by_name['module_lower'].tip_world()
            np.testing.assert_allclose(tip,[foot.worldX,foot.worldY],atol=.001)
            for bone in straight_renderer.sk.bones:
                np.testing.assert_allclose(bone.matrix()[:,:2].T@bone.matrix()[:,:2],np.eye(2),atol=1e-8)
        # Inner-fold rest: original knee (≈62,150) is now on real upper/lower pixels
        # so placement passes and idle IK still hits MODULE_CONTACT_UNREACHABLE.
        knee=[(61+72/67)/w,150/h]
        bu=[[(56,78),(67,78),(67,145),(56,145)],[(56,149),(67,149),(67,150),(56,150)]]
        bl=[[(56,147),(67,147),(67,148),(56,148)],[(56,150),(67,150),(67,153),(56,153)]]
        bf=[(54,145),(74,145),(74,147),(54,147)]
        bad=copy.deepcopy(contract)
        bad['modules'][2]['pieces'][0]['polygons']=[[[x/w,y/h] for x,y in poly] for poly in bu]
        bad['modules'][3]['pieces'][0]['polygons']=[[[x/w,y/h] for x,y in poly] for poly in bl]
        bad['modules'][4]['pieces'][0]['polygons']=[[[x/w,y/h] for x,y in bf]]
        bad['modules'][3]['pivot']=knee;bad['connections'][2]['point']=knee
        brgba=np.zeros((h,w,4),dtype=np.uint8)
        packed={'body':[shapes['body']],'face':[shapes['face']],'upper':bu,'lower':bl,'foot':[bf]}
        for order,(name,polys) in enumerate(packed.items()):
            mask=np.zeros((h,w),dtype=bool)
            for poly in polys:
                mask|=rasterize_polygons([[[x/w,y/h] for x,y in poly]],w,h)
            brgba[mask]=[30+order*35,70+order*20,140,255]
        Image.fromarray(brgba).save(self.image)
        bad['input_sha256']=hashlib.sha256(self.image.read_bytes()).hexdigest()
        with self.assertRaises(VisionReviewRequired) as ctx:self.compile(bad,'unreachable')
        self.assertEqual(ctx.exception.code,'MODULE_CONTACT_UNREACHABLE')

    def test_semantic_edge_refinement_is_pair_local_and_has_before_after_evidence(self):
        c=copy.deepcopy(self.contract)
        # Deliberate one-pixel analysis error at a known head/body semantic edge.
        c['modules'][0]['pieces'][0]['polygons'][0][0][1]=49/160
        c['modules'][0]['pieces'][0]['polygons'][0][1][1]=49/160
        c['modules'][1]['pieces'][0]['polygons'][0][2][1]=49/160
        c['modules'][1]['pieces'][0]['polygons'][0][3][1]=49/160
        c['boundary_refinements']=[{'id':'neck_edge','a':'body_p0','b':'face_p0',
            'axis':'y','polygons':[[[49/128,47/160],[74/128,47/160],
                                  [74/128,52/160],[49/128,52/160]]],
            'notes':'Confirmed neck/head boundary; only its pixel alignment is uncertain.'}]
        hints=materialize_module_contract(c,image_path=self.image,mask_dir=self.d/'refined')
        e=hints['module_projection']['boundaryRefinements'][0]
        self.assertEqual(e['changedPixels'],25)
        self.assertEqual(e['maxActualDistancePx'],1)
        self.assertEqual(e['visualReview'],'pending')
        self.assertTrue(e['topologyPreserved'] and e['coveragePreserved'])
        before=np.asarray(Image.open(e['beforeMasks']['face_p0']['path']))>127
        after=np.asarray(Image.open(e['afterMasks']['face_p0']['path']))>127
        self.assertTrue(after[49,49:74].all())
        changed=np.asarray(Image.open(e['changedMask']['path']))>127
        np.testing.assert_array_equal(changed,before!=after)
        for kind in ('beforeMasks','afterMasks'):
            for ev in e[kind].values():
                self.assertEqual(ev['sha256'],hashlib.sha256(Path(ev['path']).read_bytes()).hexdigest())
        self.assertEqual(set(hints['module_masks']),{p['id'] for m in c['modules'] for p in m['pieces']})
        repeated=copy.deepcopy(c)
        again=copy.deepcopy(c['boundary_refinements'][0]);again['id']='repeat_neck_edge'
        repeated['boundary_refinements'].append(again)
        with self.assertRaisesRegex(VisionReviewRequired,'走廊不能交叠'):
            materialize_module_contract(repeated,image_path=self.image,mask_dir=self.d/'repeat')
        # Refinement may not move the same edge arbitrarily far to chase color.
        far=copy.deepcopy(c)
        for p in far['modules'][0]['pieces'][0]['polygons'][0][:2]:p[1]=47/160
        for p in far['modules'][1]['pieces'][0]['polygons'][0][2:]:p[1]=47/160
        far['boundary_refinements'][0]['polygons']=[[[49/128,45/160],[74/128,45/160],
                                                   [74/128,52/160],[49/128,52/160]]]
        with self.assertRaises(VisionReviewRequired):
            materialize_module_contract(far,image_path=self.image,mask_dir=self.d/'too_far')

    def test_ambiguous_texture_edges_cannot_choose_module_ownership(self):
        from src.studio.module_contract import _refine_boundaries
        rgba=np.zeros((1080,32,4),np.uint8);rgba[...,3]=255
        rgba[:,:17,:3]=30;rgba[:,17:19,:3]=220;rgba[:,19:,:3]=30
        a=np.zeros((1080,32),bool);a[:,:16]=True;b=~a
        c={'boundary_refinements':[{'id':'known','a':'a','b':'b','axis':'x',
            'polygons':[[[12/32,0],[21/32,0],[21/32,1],[12/32,1]]]}]}
        with self.assertRaises(VisionReviewRequired):_refine_boundaries(c,rgba,{'a':a,'b':b})
        # In a flat image there is no evidence to refine, even when the input
        # explicitly asks for it. No color clustering fallback is available.
        rgba[...,:3]=30
        with self.assertRaises(VisionReviewRequired):_refine_boundaries(c,rgba,{'a':a,'b':b})

    def test_refinement_cannot_cross_a_third_piece_or_change_contract_fields(self):
        bad=copy.deepcopy(self.contract)
        bad['boundary_refinements']=[{'id':'bad','a':'face_p0','b':'hand_p0','axis':'x',
            'polygons':[[[.3,.1],[.9,.1],[.9,.6],[.3,.6]]],'notes':'invalid third-piece corridor'}]
        with self.assertRaises(VisionReviewRequired):
            materialize_module_contract(bad,image_path=self.image,mask_dir=self.d/'third')
        bad['boundary_refinements'][0]['radius']=100
        with self.assertRaises(VisionValidationError):self.validate(bad)



    def test_auto_seam_underlap_stays_behind_opaque_front(self):
        w,h=80,80
        rgba=np.zeros((h,w,4),np.uint8)
        rgba[10:70,10:42]=[40,40,50,255]
        rgba[10:70,42:70]=[180,90,40,255]
        rgba[4:10,36:44]=[200,180,160,255]
        image=self.d/'seam.png';Image.fromarray(rgba).save(image)
        head={'id':'head_p0','polygons':[[[36/w,4/h],[44/w,4/h],[44/w,10/h],[36/w,10/h]]],'draw_order':3}
        back={'id':'back_p0','polygons':[[[10/w,10/h],[42/w,10/h],[42/w,70/h],[10/w,70/h]]],'draw_order':1}
        front={'id':'front_p0','polygons':[[[42/w,10/h],[70/w,10/h],[70/w,70/h],[42/w,70/h]]],'draw_order':2}
        contract={'version':2,'job_id':'seam','input_sha256':hashlib.sha256(image.read_bytes()).hexdigest(),
                  'image_size':[w,h],'coord_mode':'normalized','pose_type':'airborne','anatomy':'humanoid_bust',
                  'modules':[{'id':'back','role':'torso','motion_type':'rigid','parent':None,'pivot':[26/w,40/h],
                              'motion':'sway','pieces':[back],'confidence':0.9},
                             {'id':'head','role':'head','motion_type':'rigid','parent':'back','pivot':[40/w,10/h],
                              'motion':'sway','pieces':[head],'confidence':0.9},
                             {'id':'front','role':'cloth','motion_type':'soft','parent':'back','pivot':[42/w,40/h],
                              'motion':'soft_bend','centerline':[[42/w,40/h],[56/w,60/h]],
                              'pieces':[front],'confidence':0.9}],
                  'connections':[
                      {'id':'neck','type':'joint','a':'back','b':'head','point':[40/w,10/h]},
                      {'id':'sash','type':'attachment','a':'back','b':'front','point':[42/w,40/h]},
                  ],'occlusions':[],'uncertainties':[]}
        hints=materialize_module_contract(contract,image_path=image,mask_dir=self.d/'seam-masks')
        vis_b=np.asarray(Image.open(hints['module_masks']['back_p0']).convert('L'))>127
        vis_f=np.asarray(Image.open(hints['module_masks']['front_p0']).convert('L'))>127
        vis_h=np.asarray(Image.open(hints['module_masks']['head_p0']).convert('L'))>127
        self.assertFalse((vis_b & vis_f).any())
        self.assertTrue(np.array_equal(vis_b | vis_f | vis_h, rgba[:,:,3]>0))
        hidden_path=hints['module_hidden_masks'].get('back_p0')
        self.assertIsNotNone(hidden_path)
        hid=np.asarray(Image.open(hidden_path).convert('L'))>127
        self.assertTrue(hid.any())

        auto=[o for o in hints['module_projection']['occlusions'] if o.get('autoSeamUnderlap')]
        self.assertTrue(auto)
        ps,rs=build_default_specs(image,character='seam',semantic_hints=hints)
        split=split_parts(image,ps['parts'],self.d/'seam-parts',min_island_px=0,feather_px=0,module_contract=ps.get('module_contract'))
        self.assertLessEqual(split['rest_diff']['maxRGBA'],2)

    def test_resolve_module_motion_splits_left_right_and_zeros_loop(self):
        from src.pipeline.module_rig import resolve_module_motion
        from src.pipeline.build_rig import harmonic
        def mod(mid, role, profile, shape, parent, pivot, extra=None):
            item={'id':mid,'bone':'module_'+mid,'role':role,'motionProfile':profile,'motion':'required',
                  'shape':shape,'parent':parent,'pivot':pivot,'pieces':['piece_'+mid]}
            if extra: item.update(extra)
            return item
        modules=[
            mod('torso','torso','breath','rigid',None,[100.0,80.0]),
            mod('head','head','sway','rigid','torso',[100.0,40.0]),
            mod('larm','arm','sway','rigid','torso',[60.0,70.0]),
            mod('rarm','arm','sway','rigid','torso',[140.0,70.0]),
            mod('lleg','leg','sway','rigid','torso',[80.0,130.0]),
            mod('rleg','leg','sway','rigid','torso',[120.0,130.0]),
            mod('lrib','cloth','soft_bend','soft','torso',[50.0,90.0],
                {'centerline':[[50.0,90.0],[20.0,140.0]],'softBones':['bend_lrib1']}),
            mod('rrib','cloth','soft_bend','soft','torso',[150.0,90.0],
                {'centerline':[[150.0,90.0],[180.0,140.0]],'softBones':['bend_rrib1']}),
        ]
        bones=[{'name':'root','parent':None}]
        skin={}; chains={}
        for m in modules:
            bones.append({'name':m['bone'],'parent':('module_'+m['parent']) if m['parent'] else 'root'})
            if m['shape']=='soft':
                cname='bend_'+m['id']
                bones.append({'name':cname+'1','parent':m['bone']})
                chains[cname]={'parent':m['bone'],'count':1,'pts':m['centerline']}
                skin[m['pieces'][0]]={'type':'mesh','recipe':'chain','chain':cname,'module_id':m['id']}
            else:
                skin[m['pieces'][0]]={'type':'region','bone':m['bone'],'module_id':m['id']}
        rig={'duration':8.0,'body_height_px':1080.0,'bones':bones,'ik':[],'skin':skin,'chains':chains,
             'moduleContract':{'sourceVersion':2,'modules':modules}}
        motion=resolve_module_motion(rig,'agile_full',duration=8.0)
        def amp(name, harmonic):
            return next(a for a,k,_ in motion[name]['rotate'] if k==harmonic)
        self.assertLess(amp('module_larm',3)*amp('module_rarm',3),0)
        self.assertLess(amp('module_lleg',3)*amp('module_rleg',3),0)
        self.assertLess(amp('module_lrib',3)*amp('module_rrib',3),0)
        self.assertLess(amp('module_head',2)*amp('module_torso',2),0)
        self.assertEqual({k for a,k,_ in motion['module_torso']['ty']},{2,3})
        self.assertEqual({k for a,k,_ in motion['bend_lrib1']['rotate']},{2,3,4})
        self.assertIn(3, {k for a,k,_ in motion['module_larm']['rotate']})
        self.assertIn(2, {k for a,k,_ in motion['module_larm']['rotate']})
        for name,ch in motion.items():
            for key in ('rotate','tx','ty'):
                spec=ch.get(key)
                if not spec: continue
                self.assertAlmostEqual(harmonic(spec,0.0,8.0),0.0,places=9,msg=name+' '+key)
                self.assertAlmostEqual(harmonic(spec,8.0,8.0),0.0,places=9,msg=name+' '+key)

    def test_auto_seam_underlap_requires_topological_link(self):
        """Unconnected layer crossings (e.g. weapon over cloth) must not get
        auto seam underlap; only parent/child or contract-connected pairs may."""
        w,h=80,80
        rgba=np.zeros((h,w,4),np.uint8)
        rgba[10:70,10:40]=[40,40,50,255]
        rgba[10:70,40:60]=[180,90,40,255]
        rgba[20:60,60:70]=[200,60,60,255]
        rgba[2:10,22:32]=[230,200,180,255]
        image=self.d/'seam_link.png';Image.fromarray(rgba).save(image)
        back={'id':'back_p0','polygons':[[[10/w,10/h],[40/w,10/h],[40/w,70/h],[10/w,70/h]]],'draw_order':1}
        cloth={'id':'cloth_p0','polygons':[[[40/w,10/h],[60/w,10/h],[60/w,70/h],[40/w,70/h]]],'draw_order':2}
        blade={'id':'blade_p0','polygons':[[[60/w,20/h],[70/w,20/h],[70/w,60/h],[60/w,60/h]]],'draw_order':3}
        head={'id':'head_p0','polygons':[[[22/w,2/h],[32/w,2/h],[32/w,10/h],[22/w,10/h]]],'draw_order':4}
        contract={'version':2,'job_id':'seamlink','input_sha256':hashlib.sha256(image.read_bytes()).hexdigest(),
                  'image_size':[w,h],'coord_mode':'normalized','pose_type':'airborne','anatomy':'humanoid_bust',
                  'modules':[{'id':'torso','role':'torso','motion_type':'rigid','parent':None,'pivot':[26/w,40/h],
                              'motion':'breath','pieces':[back],'confidence':0.9},
                             {'id':'head','role':'head','motion_type':'rigid','parent':'torso','pivot':[27/w,6/h],
                              'motion':'sway','pieces':[head],'confidence':0.9},
                             {'id':'sash','role':'cloth','motion_type':'soft','parent':'torso','pivot':[42/w,40/h],
                              'motion':'soft_bend','centerline':[[42/w,40/h],[56/w,60/h]],
                              'pieces':[cloth],'confidence':0.9},
                             {'id':'blade','role':'weapon','motion_type':'rigid','parent':'torso','pivot':[40/w,40/h],
                              'motion':'follow','pieces':[blade],'confidence':0.9}],
                  'connections':[
                      {'id':'neck','type':'joint','a':'torso','b':'head','point':[27/w,6/h]},
                      {'id':'sashj','type':'attachment','a':'torso','b':'sash','point':[42/w,40/h]},
                      {'id':'grip','type':'grip','a':'torso','b':'blade','point':[40/w,40/h]},
                  ],'occlusions':[],'uncertainties':[]}
        hints=materialize_module_contract(contract,image_path=image,mask_dir=self.d/'seamlink-masks')
        auto=[o for o in hints['module_projection']['occlusions'] if o.get('autoSeamUnderlap')]
        pairs={(o['back'],o['front']) for o in auto}
        # torso <-> sash (parent/child + connection) may underlap
        self.assertIn(('back_p0','cloth_p0'),pairs)
        # torso <-> head (parent/child + joint connection) may underlap
        self.assertIn(('back_p0','head_p0'),pairs)
        # sash <-> blade: no topological link -> must NOT underlap
        self.assertNotIn(('cloth_p0','blade_p0'),pairs)
        self.assertNotIn(('blade_p0','cloth_p0'),pairs)

if __name__=='__main__':unittest.main()

