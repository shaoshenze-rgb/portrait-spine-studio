"""Synthetic contract tests; these fixtures are not claimed as live vision results."""
import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.studio.auto_spec import build_default_specs
from src.studio.vision_schema import (
    VisionReviewRequired, VisionValidationError, validate_vision_result,
    vision_result_to_semantic_hints, rasterize_polygons,
)

ROOT = Path(__file__).resolve().parents[2]


def synthetic_result(image_path, width=120, height=140):
    rgba = np.zeros((height,width,4), dtype=np.uint8)
    rgba[10:125,30:80] = [100,150,190,255]
    rgba[85:125,10:30] = [160,90,70,255]
    rgba[50:135,75:95] = [180,140,100,255]
    Image.fromarray(rgba).save(image_path)
    def rect(x0,y0,x1,y1):
        return [[x0/width,y0/height],[x1/width,y0/height],[x1/width,y1/height],[x0/width,y1/height]]
    return {
        "version":1, "input_sha256":hashlib.sha256(Path(image_path).read_bytes()).hexdigest(),
        "image_size":[width,height], "coord_mode":"normalized", "pose_type":"airborne",
        "joints":{
            "head_top":[.45,.1], "neck":[.45,.23], "chest":[.45,.35], "hips":[.45,.6],
            "hip_L":[.40,.6], "knee_L":[.36,.75], "ankle_L":[.35,.85],
            "hip_R":[.50,.6], "knee_R":[.54,.75], "ankle_R":[.56,.85],
        },
        "entities":[
            {"id":"person","kind":"subject","polygon":rect(0,0,width,height),"confidence":.95},
            {"id":"companion","kind":"pet","polygon":rect(10,85,30,125),"confidence":.94},
            {"id":"tail","kind":"tail","polygon":rect(75,50,95,135),"confidence":.91,
             "centerline":[[.66,.40],[.72,.65],[.70,.94]]},
        ], "uncertainties":[],
    }


class TestVisionSchema(unittest.TestCase):
    def setUp(self):
        work = ROOT / '.studio-work'
        work.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=work, prefix='vision-schema-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root / 'arbitrary-filename.png'
        self.result = synthetic_result(self.image)

    def validate(self, result=None):
        return validate_vision_result(result or self.result,
                                      input_sha256=self.result['input_sha256'], image_size=(120,140))

    def test_subject_excludes_pet_and_tail_at_original_resolution(self):
        normalized = self.validate()
        self.assertNotIn('static', self.result['entities'][1], 'validation must not mutate provider data')
        hints = vision_result_to_semantic_hints(normalized,image_path=self.image,mask_dir=self.root/'masks')
        subject = np.asarray(Image.open(hints['subject_mask'])) > 0
        masks = {k:np.asarray(Image.open(p)) > 0 for k,p in hints['ownership_masks'].items()}
        self.assertEqual(subject.shape,(140,120))
        self.assertFalse((subject & masks['entity_companion']).any())
        self.assertFalse((subject & masks['entity_tail']).any())
        self.assertEqual(int(masks['entity_companion'].sum()),20*40)
        self.assertEqual(int(masks['entity_tail'].sum()),20*85)
        parts, rig = build_default_specs(self.image,mask_dir=self.root/'parts-masks',semantic_hints=hints,legacy_diagnostic=True)
        self.assertEqual(rig['skin']['entity_companion']['type'],'region')
        self.assertEqual(rig['skin']['entity_companion']['bone'],'root')
        self.assertEqual(rig['chains']['entity_tail']['parent'],'hips')
        self.assertEqual(rig['skin']['entity_tail']['recipe'],'chain')
        byname = {p['name']:p for p in parts['parts']}
        for name in masks:
            self.assertTrue(byname[name]['ownership_locked'])
            np.testing.assert_array_equal(np.asarray(Image.open(byname[name]['mask']))>0,masks[name])

    def test_bad_hash_size_coordinates_and_incomplete_limb_rejected(self):
        for key,value in [('input_sha256','0'*64),('image_size',[121,140]),('coord_mode','canvas')]:
            bad=copy.deepcopy(self.result);bad[key]=value
            with self.subTest(key=key), self.assertRaises(VisionValidationError):self.validate(bad)
        bad=copy.deepcopy(self.result);bad['joints']['neck']=[float('nan'),.2]
        with self.assertRaises(VisionValidationError):self.validate(bad)
        bad=copy.deepcopy(self.result);del bad['joints']['knee_L']
        with self.assertRaises(VisionReviewRequired):self.validate(bad)

    def test_overlapping_independent_entities_and_unassigned_are_reviewed(self):
        bad=copy.deepcopy(self.result)
        bad['entities'][2]['polygon']=bad['entities'][1]['polygon']
        with self.assertRaisesRegex(VisionReviewRequired,'重叠'):
            vision_result_to_semantic_hints(self.validate(bad),image_path=self.image,mask_dir=self.root/'overlap')
        self.assertFalse((self.root/'overlap').exists())
        bad=copy.deepcopy(self.result);bad['entities'][0]['polygon']=[[.3,.1],[.4,.1],[.4,.3],[.3,.3]]
        with self.assertRaisesRegex(VisionReviewRequired,'未覆盖'):
            vision_result_to_semantic_hints(self.validate(bad),image_path=self.image,mask_dir=self.root/'missing')

    def test_low_confidence_uncertainty_and_wrong_semantic_parent_reviewed(self):
        variants=[]
        low=copy.deepcopy(self.result);low['entities'][1]['confidence']=.01;variants.append(low)
        unsure=copy.deepcopy(self.result);unsure['uncertainties']=['尾巴边界不确定'];variants.append(unsure)
        pet=copy.deepcopy(self.result);pet['entities'][1]['parent_bone']='thigh_L';variants.append(pet)
        tail=copy.deepcopy(self.result);tail['entities'][2]['static']=True;variants.append(tail)
        for bad in variants:
            with self.assertRaises(VisionReviewRequired):self.validate(bad)

    def test_self_touching_polygon_rejected_and_shared_edge_is_disjoint(self):
        bad=copy.deepcopy(self.result)
        bad['entities'][1]['polygon']=[[.1,.1],[.8,.1],[.8,.8],[.4,.1],[.1,.8]]
        with self.assertRaises(VisionValidationError):self.validate(bad)
        a=rasterize_polygons([[[0,0],[.5,0],[.5,1],[0,1]]],120,140)
        b=rasterize_polygons([[[.5,0],[1,0],[1,1],[.5,1]]],120,140)
        self.assertFalse((a & b).any())
        self.assertTrue((a | b).all())

    def test_changed_input_cannot_reuse_analysis(self):
        valid=self.validate()
        Image.new('RGBA',(120,140),(30,60,90,255)).save(self.image)
        with self.assertRaisesRegex(VisionValidationError,'SHA'):
            vision_result_to_semantic_hints(valid,image_path=self.image,mask_dir=self.root/'stale')

    def test_one_pixel_high_resolution_tail_is_not_lost_at_260px(self):
        image=Image.open(self.image).resize((1200,1400),Image.Resampling.NEAREST)
        image.save(self.image)
        result=copy.deepcopy(self.result)
        result['image_size']=[1200,1400]
        result['input_sha256']=hashlib.sha256(self.image.read_bytes()).hexdigest()
        result['entities'][2]['polygon']=[[.625,.4],[.625+1/1200,.4],[.625+1/1200,.8],[.625,.8]]
        normalized=validate_vision_result(result,input_sha256=result['input_sha256'],image_size=(1200,1400))
        hints=vision_result_to_semantic_hints(normalized,image_path=self.image,mask_dir=self.root/'thin-masks')
        expected=np.asarray(Image.open(hints['ownership_masks']['entity_tail']))>0
        self.assertEqual(int(expected.sum()),560)
        parts,rig=build_default_specs(self.image,semantic_hints=hints,mask_dir=self.root/'thin-parts',legacy_diagnostic=True)
        tail=next(p for p in parts['parts'] if p['name']=='entity_tail')
        self.assertTrue(tail['ownership_locked'])
        np.testing.assert_array_equal(np.asarray(Image.open(tail['mask']))>0,expected)


if __name__ == '__main__':
    unittest.main()
