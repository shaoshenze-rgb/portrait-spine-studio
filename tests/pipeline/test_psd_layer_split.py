import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from src.pipeline.canvas_layer_cleanup import clean_layer, defringe_rgba
from src.pipeline.psd_layer_split import PsdPartSpec, split_from_psd


class _FakePsdLayer:
    def __init__(self, name, rgba, left=0, top=0, *, blend_mode="normal", kind="pixel", clipping=False, visible=True):
        self.name = name
        self._rgba = np.asarray(rgba, dtype=np.uint8)
        self.left = int(left)
        self.top = int(top)
        self.right = int(left + self._rgba.shape[1])
        self.bottom = int(top + self._rgba.shape[0])
        self.blend_mode = blend_mode
        self.kind = kind
        self.clipping = clipping
        self.visible = visible

    def is_group(self):
        return False

    def topil(self):
        return Image.fromarray(self._rgba)


class Defringe4KTests(unittest.TestCase):
    def test_4k_wide_white_and_midgray_fringe_cleaned_while_preserving_vfx_glow(self):
        # Simulate a large canvas edge (max_radius=8) with:
        # 1) 6px wide bright white halo on dark purple armor
        # 2) 5px wide desaturated mid-gray (#969696) backdrop halo on dark purple armor
        # 3) saturated glowing purple VFX fringe (MUST be preserved!)
        h, w = 60, 120
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        # Solid dark purple armor core (sat ~0.58, lum ~50)
        rgba[20:45, 20:100] = [70, 35, 95, 255]
        # 1) White halo 6px above left segment (x=25..45)
        rgba[14:20, 25:45] = [235, 235, 235, 120]
        # 2) Neutral mid-gray halo 5px above middle segment (x=50..70)
        rgba[15:20, 50:70] = [145, 145, 145, 140]
        # 3) Bright saturated neon violet glow above right segment (x=75..95) — sat=0.80 > interior
        rgba[15:20, 75:95] = [220, 44, 255, 150]

        cleaned, count = defringe_rgba(rgba, max_radius=8)
        self.assertGreater(count, 150)
        # Alpha channel strictly untouched
        np.testing.assert_array_equal(cleaned[..., 3], rgba[..., 3])
        # Opaque interior strictly untouched
        np.testing.assert_array_equal(cleaned[20:45, 20:100], rgba[20:45, 20:100])
        # White halo 6px out replaced with purple interior
        self.assertLess(int(cleaned[14, 35, 0]), 90)
        self.assertLess(int(cleaned[14, 35, 1]), 55)
        # Neutral mid-gray halo replaced with purple interior
        self.assertLess(int(cleaned[16, 60, 1]), 55)
        # Saturated glowing neon violet VFX edge strictly preserved!
        np.testing.assert_array_equal(cleaned[16, 85, :3], np.array([220, 44, 255], dtype=np.uint8))


class PsdLayerSplitTests(unittest.TestCase):
    def test_cross_part_highlight_overlay_stays_bound_to_underlying_part_and_auto_underlaps(self):
        # Canvas 100x100:
        # - scarf_back (order 0): x=10..60, y=20..80 (blue), artist painted hidden under arm only up to x=60
        # - rear_arm (order 1): x=45..90, y=20..80 (purple), overlaps scarf_back at x=45..60
        # - highlight_fx (psd idx 2, blend='screen'): rim-light across BOTH scarf_back (x=20..44) and rear_arm (x=45..65)
        h, w = 100, 100
        scarf_ly = np.zeros((h, w, 4), dtype=np.uint8)
        scarf_ly[20:80, 10:50] = [40, 90, 200, 255]  # Artist stopped scarf right at x=50 (5px overlap with arm)
        arm_ly = np.zeros((h, w, 4), dtype=np.uint8)
        arm_ly[20:80, 45:90] = [140, 50, 170, 255]
        # Cross-part rim light painted over BOTH scarf (x=25..44) and arm (x=45..60)
        rim_ly = np.zeros((h, w, 4), dtype=np.uint8)
        rim_ly[25:75, 25:60] = [245, 240, 255, 200]

        # Final composited artwork (source_rgba): scarf + arm + rim light baked in
        comp = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        comp.alpha_composite(Image.fromarray(scarf_ly))
        comp.alpha_composite(Image.fromarray(arm_ly))
        comp.alpha_composite(Image.fromarray(rim_ly))
        source_rgba = np.asarray(comp, dtype=np.uint8)

        fake_layers = [
            _FakePsdLayer("scarf", scarf_ly),
            _FakePsdLayer("arm", arm_ly),
            _FakePsdLayer("图层 118", rim_ly, blend_mode="screen"),
        ]
        # Caller mistakenly includes the cross-part highlight layer (idx 2) inside rear_arm!
        specs = [
            PsdPartSpec(id="scarf_back", psd_indices=(0,), draw_order=0, kind="soft"),
            PsdPartSpec(id="rear_arm", psd_indices=(1, 2), draw_order=1, kind="rigid"),
        ]
        res = split_from_psd(
            source_rgba,
            Path("fake.psd"),
            specs,
            underlap_px=16,
            defringe=False,
            psd_layers_override=fake_layers,
            psd_size_override=(w, h),
        )

        # 1. Cross-part highlight over scarf (x=25..44) MUST stay owned by scarf_back, NOT rear_arm!
        self.assertGreater(res.stats["overlay_rebound_px"], 500)
        self.assertTrue(res.ownership["scarf_back"][40, 30])
        self.assertFalse(res.ownership["rear_arm"][40, 30])
        self.assertEqual(int(res.rgba["rear_arm"][40, 30, 3]), 0)
        # And scarf_back carries the exact baked highlight pixels from source_rgba!
        np.testing.assert_array_equal(res.rgba["scarf_back"][40, 30], source_rgba[40, 30])

        # 2. Genuine artist hidden pixels (x=45..49) preserved AND auto-bridge extended behind arm (x=50..60)!
        self.assertGreater(res.stats["layers"]["scarf_back"]["artist_hidden_px"], 200)
        self.assertGreater(res.stats["layers"]["scarf_back"]["auto_bridged_px"], 300)
        self.assertGreater(int(res.rgba["scarf_back"][40, 56, 3]), 200)

        # 3. Rest-pose composite of the split parts MUST be pixel-identical to original source_rgba!
        rebuilt = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        rebuilt.alpha_composite(Image.fromarray(res.rgba["scarf_back"]))
        rebuilt.alpha_composite(Image.fromarray(res.rgba["rear_arm"]))
        np.testing.assert_array_equal(np.asarray(rebuilt, dtype=np.uint8), source_rgba)


if __name__ == "__main__":
    unittest.main()
