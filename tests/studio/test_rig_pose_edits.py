"""Pose edits must change foot constraints, not just their display label."""
from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from src.pipeline.build_rig import build_rig
from src.pipeline.split_parts import split_parts
from src.render.renderer import SpineRenderer
from src.studio.auto_spec import apply_rig_edits


def airborne_spec():
    return {
        "character": "pose_test", "pose_type": "airborne", "body_height_px": 100,
        "bones": [
            {"name": "root", "parent": None, "pos": [64, 128]},
            {"name": "hips", "parent": "root", "pos": [64, 40]},
            {"name": "thigh_L", "parent": "hips", "pos": [60, 45], "aim": [90, 75], "role": "leg"},
            {"name": "shin_L", "parent": "thigh_L", "pos": [90, 75], "aim": [60, 105], "role": "leg"},
            {"name": "foot_L", "parent": "shin_L", "pos": [60, 105], "aim": [50, 110], "role": "leg"},
        ],
        "ik": [], "chains": {},
        "skin": {"legs": {"type": "mesh", "recipe": "limb", "start_bone": "hips",
                            "mid_bone": "thigh_L", "end_bone": "shin_L", "cell": 12}},
    }


class TestRigPoseEdits(unittest.TestCase):
    def test_round_trip_changes_real_foot_motion(self):
        original = airborne_spec()
        grounded = apply_rig_edits(original, {"pose_type": "grounded"})
        airborne = apply_rig_edits(grounded, {"pose_type": "airborne"})
        self.assertEqual(len(grounded["ik"]), 1)
        self.assertEqual(airborne["ik"], [])
        self.assertNotIn("toe_target_L", [b["name"] for b in airborne["bones"]])
        self.assertEqual(original, airborne_spec(), "edits must not mutate the caller's rig")

        work = Path(__file__).resolve().parents[2] / ".studio-work"
        work.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=work, prefix="pose-edit-test-") as tmp:
            root = Path(tmp)
            source = root / "source.png"
            im = Image.new("RGBA", (128, 128))
            ImageDraw.Draw(im).line([(60, 45), (90, 75), (60, 105)], fill=(90, 140, 180, 255), width=16)
            im.save(source)
            split_parts(source, [{"name": "legs", "own": [[(0, 0), (127, 0), (127, 127), (0, 127)]],
                                  "pivot": [60, 45], "depth": 0}], root / "parts")
            paths = {}
            for name, spec in (("grounded", grounded), ("airborne", airborne)):
                rig = root / name
                build_rig(root / "parts", spec, rig)
                renderer = SpineRenderer(rig)
                animation = renderer.sk.json["animations"]["idle"]["bones"]
                self.assertEqual("thigh_L" in animation, name == "airborne")
                points = []
                for t in np.linspace(0, 8, 33):
                    renderer.pose(float(t))
                    foot = renderer.sk.by_name["foot_L"]
                    points.append((foot.worldX, foot.worldY))
                paths[name] = np.asarray(points)
            self.assertLess(float(np.ptp(paths["grounded"], axis=0).max()), 0.02)
            self.assertGreater(float(np.ptp(paths["airborne"], axis=0).max()), 0.5)

    def test_switch_preserves_user_joint_positions(self):
        original = airborne_spec()
        edited = copy.deepcopy(original["bones"])
        next(b for b in edited if b["name"] == "shin_L")["pos"] = [96, 72]
        updated = apply_rig_edits(original, {"bones": edited, "pose_type": "grounded"})
        positions = {b["name"]: b["pos"] for b in updated["bones"]}
        for b in edited:
            self.assertEqual(positions[b["name"]], b["pos"])
        self.assertEqual(positions["toe_target_L"], positions["foot_L"])
        thigh = next(b for b in updated["bones"] if b["name"] == "thigh_L")
        self.assertEqual(thigh["aim"], positions["shin_L"])

    def test_invalid_standing_chain_fails_without_mutating_input(self):
        cases = []
        missing = airborne_spec()
        missing["bones"] = [b for b in missing["bones"] if b["name"] != "shin_L"]
        cases.append(missing)
        straight = airborne_spec()
        next(b for b in straight["bones"] if b["name"] == "shin_L")["pos"] = [60, 75]
        cases.append(straight)
        no_legs = airborne_spec()
        no_legs["bones"] = no_legs["bones"][:2]
        cases.append(no_legs)
        for rig in cases:
            before = copy.deepcopy(rig)
            with self.subTest(bones=[b["name"] for b in rig["bones"]]):
                with self.assertRaisesRegex(ValueError, "无法切换站立姿态"):
                    apply_rig_edits(rig, {"pose_type": "grounded"})
                self.assertEqual(rig, before)

    def test_release_preserves_unrelated_ik_and_referenced_target(self):
        grounded = apply_rig_edits(airborne_spec(), {"pose_type": "grounded"})
        other = {"name": "hand_constraint", "bones": ["hand_custom"], "target": "hand_target"}
        grounded["ik"].append(other)
        grounded["skin"]["custom_marker"] = {"type": "region", "bone": "toe_target_L"}
        airborne = apply_rig_edits(grounded, {"pose_type": "airborne"})
        self.assertEqual(airborne["ik"], [other])
        self.assertIn("toe_target_L", [b["name"] for b in airborne["bones"]])


if __name__ == "__main__":
    unittest.main()
