import unittest
import numpy as np
from src.pipeline.canvas_layer_cleanup import clean_layer,cleanup_layers,check_missing_original_regions,dilate

class CanvasCleanupTests(unittest.TestCase):
    def test_outside_and_background_holes_removed_hidden_interior_kept(self):
        original=np.zeros((80,80,4),np.uint8);original[10:70,10:70]=[20,30,40,255];original[30:40,30:40]=0
        layer=np.full_like(original,255);clean,stats=clean_layer(layer,original)
        self.assertTrue((clean[~dilate(original[...,3]>8)]==0).all())
        np.testing.assert_array_equal(clean[20:25,20:25],layer[20:25,20:25])
        self.assertEqual(stats['outside_ratio_after'],0);self.assertGreater(stats['removed_outside_pixels'],0)
    def test_white_fringe_replaced_with_local_interior_color_without_alpha_change(self):
        original=np.full((20,20,4),255,np.uint8);layer=np.zeros_like(original)
        layer[5:15,5:15]=[25,40,55,255];layer[4,5:15]=[220,220,220,100];layer[0,0]=[255,255,255,2]
        clean,stats=clean_layer(layer,original)
        self.assertEqual(clean[4,8,3],100);self.assertLess(clean[4,8,0],50);self.assertEqual(clean[0,0,3],0)
        self.assertGreater(stats['defringed_pixels'],0)
    def test_muddy_effect_omit_reveals_hand(self):
        original=np.zeros((50,50,4),np.uint8);original[10:40,10:40]=[40,80,110,255]
        flame=np.full_like(original,[240,70,0,255])
        layers=[{'id':'hand','rgba':original,'role':'hand','draw_order':0},
                {'id':'effect','rgba':flame,'role':'effect_group','draw_order':1,'effect_mode':'omit'}]
        _,report,composite=cleanup_layers(layers,original)
        np.testing.assert_array_equal(composite[20,20],original[20,20])
        self.assertEqual(report['warnings'][0]['code'],'MUDDY_EFFECT_LAYER_DETECTED')
    def test_original_extract_uses_source_pixels_and_missing_component_threshold(self):
        original=np.zeros((60,60,4),np.uint8);original[5:30,5:30]=[250,120,15,255];original[35:55,35:55]=[80,90,110,255]
        clean,_=clean_layer(np.full_like(original,255),original,effect_mode='original_extract')
        np.testing.assert_array_equal(clean[10,10],original[10,10]);self.assertEqual(clean[40,40,3],0)
        regions=check_missing_original_regions(original,clean)
        self.assertEqual(regions,[{'area_px':400,'bbox':[35,35,55,55]}])
        full=np.full((32,32,4),255,np.uint8)
        self.assertEqual(check_missing_original_regions(full,np.zeros_like(full))[0]['area_px'],1024)
