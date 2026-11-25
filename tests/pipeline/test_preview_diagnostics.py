import tempfile
import unittest
from pathlib import Path
import numpy as np
from src.gates.preview_diagnostics import inspect_dynamic
class DynamicChecksTests(unittest.TestCase):
    def test_silhouette_motion_is_not_an_internal_hole(self):
        class Renderer:
            build_report={'moduleContract':{'modules':[]}}
            def pose(self,t):pass
        base=np.zeros((96,96,4),np.uint8);base[16:80,16:80]=255
        moved=np.zeros_like(base);moved[16:80,24:88]=255
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]/'.studio-work') as tmp:
            report=inspect_dynamic(Renderer(),{0.:base,.5:moved},base,tmp)
            self.assertEqual(report['new_gap']['area_px'],0)
            hole=base.copy();hole[36:60,36:60]=0
            report=inspect_dynamic(Renderer(),{0.:base,.5:hole},base,tmp)
            self.assertGreater(report['new_gap']['area_px'],0)
            self.assertEqual(report['new_gap']['time'],.5)
