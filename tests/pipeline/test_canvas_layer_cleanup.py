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

class OriginalBodyRecoveryTests(unittest.TestCase):
    def fixture(self):
        original=np.zeros((100,150,4),np.uint8)
        original[10:90,10:100]=[40,70,110,255]
        arm=original.copy();arm[30:60,40:60]=0
        return original,[{'id':'arm','rgba':arm,'role':'arm','draw_order':0,'enabled':True}]

    def test_600px_elbow_is_restored_with_exact_original_material(self):
        from src.pipeline.canvas_layer_cleanup import recover_missing_body_from_original
        original,layers=self.fixture();before=layers[0]['rgba'].copy()
        restored,report,composite=recover_missing_body_from_original(layers,original)
        self.assertEqual(report['recovered_region_count'],1)
        self.assertEqual(report['recovered_pixel_count'],600)
        self.assertEqual(report['recovered_regions'][0]['target_layer'],'arm')
        self.assertEqual(report['missing_original_regions'],[])
        np.testing.assert_array_equal(restored[0]['rgba'][30:60,40:60],original[30:60,40:60])
        np.testing.assert_array_equal(layers[0]['rgba'],before)
        self.assertTrue((composite[original[...,3]==0]==0).all())

    def test_omitted_effect_outskirts_stay_hidden_but_overlap_cannot_erase_elbow(self):
        from src.pipeline.canvas_layer_cleanup import recover_missing_body_from_original
        original,layers=self.fixture();original[30:60,115:140]=[250,110,10,255]
        effect=np.zeros_like(original);effect[30:60,115:140]=original[30:60,115:140]
        effect[30:60,40:60]=[240,80,0,255]
        layers.append({'id':'flame','role':'effect_group','rgba':effect,'draw_order':1,'effect_mode':'omit','enabled':False})
        _,report,composite=recover_missing_body_from_original(layers,original)
        self.assertEqual(report['recovered_pixel_count'],600)
        self.assertEqual(report['ignored_omitted_effect_pixels'],750)
        self.assertEqual(report['missing_original_regions'],[])
        self.assertTrue((composite[30:60,115:140]==0).all())
        np.testing.assert_array_equal(composite[30:60,40:60],original[30:60,40:60])

    def test_explicit_ignore_and_no_legal_parent_never_invent_body_material(self):
        from src.pipeline.canvas_layer_cleanup import recover_missing_body_from_original
        original,layers=self.fixture();ignore=np.zeros(original.shape[:2],bool);ignore[30:60,40:60]=True
        _,report,composite=recover_missing_body_from_original(layers,original,ignore_mask=ignore)
        self.assertEqual(report['recovered_pixel_count'],0);self.assertEqual(composite[40,50,3],0)
        layers[0]['role']='weapon'
        _,report,composite=recover_missing_body_from_original(layers,original)
        self.assertEqual(report['recovered_region_count'],0)
        self.assertEqual(report['unresolved_recovery_regions'][0]['reason'],'no_enabled_body_layer')

    def test_small_gaps_below_200px_are_not_promoted_to_repairs(self):
        from src.pipeline.canvas_layer_cleanup import recover_missing_body_from_original
        original,layers=self.fixture();layers[0]['rgba']=original.copy();layers[0]['rgba'][30:40,40:59]=0
        _,report,_=recover_missing_body_from_original(layers,original)
        self.assertEqual(report['recovered_region_count'],0)

    def test_feather_overlap_preserves_fractional_alpha_without_byte_overflow(self):
        from src.pipeline.canvas_layer_cleanup import recover_missing_body_from_original
        original,layers=self.fixture()
        original[30:60,60]=[40,70,110,180]
        # alpha 180 belongs to core; a non-core translucent edge is feathered.
        original[30:60,60,3]=170;layers[0]['rgba'][30:60,60]=0
        restored,report,_=recover_missing_body_from_original(layers,original)
        self.assertEqual(report['recovered_pixel_count'],600)
        self.assertEqual(int(restored[0]['rgba'][45,60,3]),113)

    def test_interior_omitted_effect_tendril_is_removed_while_body_elbow_is_recovered(self):
        from src.pipeline.canvas_layer_cleanup import recover_missing_body_from_original
        original,layers=self.fixture()
        # Add an extended limb so convex hull encloses y=70..95, x=40..70
        layers[0]['rgba'][80:100,10:90]=[50,60,70,255]
        original[80:100,10:90]=[50,60,70,255]
        # Pure fiery tendril INSIDE body convex hull (must be excluded when flame is omitted)
        layers[0]['rgba'][62:78,40:70]=0
        original[62:78,40:70]=[230,85,15,255]
        # Outer flame pixels outside convex hull to define effect chromatic signature
        original[30:60,115:140]=[235,80,12,255]
        effect=np.zeros_like(original)
        effect[30:60,115:140]=[235,80,12,255]
        effect[62:78,40:70]=[230,85,15,255]
        effect[30:60,40:60]=[240,80,0,255]  # overlaps neutral blue-gray missing elbow!
        layers.append({'id':'flame','role':'effect_group','rgba':effect,'draw_order':1,'effect_mode':'omit','enabled':False})
        _,report,composite=recover_missing_body_from_original(layers,original)
        # Neutral elbow (600px) IS recovered; fiery interior tendril (480px) is NOT recovered!
        self.assertEqual(report['recovered_pixel_count'],600)
        self.assertTrue((composite[62:78,40:70,3]==0).all())
        np.testing.assert_array_equal(composite[30:60,40:60],original[30:60,40:60])
