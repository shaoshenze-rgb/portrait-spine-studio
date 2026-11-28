import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.pipeline.canvas_layer_cleanup import (
    check_canvas_fallback_policy,
    check_per_character_override_policy,
    inspect_canvas_parts_warnings,
)
from src.pipeline.canvas_layer_parts import materialize_canvas_layer_parts
from src.pipeline.dead_crossing_rebind import rebind_dead_crossings
from src.pipeline.sibling_variant_transfer import transfer_sibling_variant


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_sample_grounded_material_and_plan():
    material = {
        "version": 1,
        "material_mode": "canvas_layers",
        "source_image": "/tmp/orig.png",
        "source_sha256": "a" * 64,
        "image_size": [256, 256],
        "pieces": [
            {"id": "ground", "module_id": "ground", "parent": None, "kind": "rigid", "pivot": [128.0, 220.0], "draw_order": 0},
            {"id": "scarf", "module_id": "scarf", "parent": "torso", "kind": "rigid", "pivot": [128.0, 110.0], "draw_order": 10},
            {"id": "lower_body", "module_id": "lower_body", "parent": None, "kind": "rigid", "pivot": [128.0, 180.0], "draw_order": 20},
            {"id": "torso", "module_id": "torso", "parent": "lower_body", "kind": "rigid", "pivot": [128.0, 130.0], "draw_order": 21},
            {"id": "contact_fx", "module_id": "contact_fx", "parent": "torso", "kind": "rigid", "pivot": [128.0, 130.0], "draw_order": 30},
        ],
    }
    plan = {
        "pose_type": "grounded",
        "anatomy": "humanoid_full",
        "modules": [
            {"id": "ground", "role": "ground", "parent": None, "pivot": [0.5, 0.859375], "motion": "static", "motion_type": "static"},
            {"id": "scarf", "role": "cloth", "parent": "torso", "pivot": [0.5, 0.4296875], "motion": "sway", "motion_type": "rigid"},
            {"id": "lower_body", "role": "lower_body", "parent": None, "pivot": [0.5, 0.703125], "motion": "static", "motion_type": "static"},
            {"id": "torso", "role": "torso", "parent": "lower_body", "pivot": [0.5, 0.5078125], "motion": "breath", "motion_type": "rigid"},
            {"id": "contact_fx", "role": "effect_group", "parent": "torso", "pivot": [0.5, 0.5078125], "motion": "follow", "motion_type": "rigid"},
        ],
        "connections": [
            {"id": "at_scarf", "type": "attachment", "a": "torso", "b": "scarf", "point": [0.5, 0.43]},
            {"id": "at_torso", "type": "joint", "a": "lower_body", "b": "torso", "point": [0.5, 0.51]},
            {"id": "at_fx", "type": "attachment", "a": "torso", "b": "contact_fx", "point": [0.5, 0.51]},
        ],
        "occlusions": [{"front": "scarf", "back": "torso"}],
    }
    return material, plan


class DeadCrossingRebindTests(unittest.TestCase):
    def test_grounded_scarf_torso_exposure_absorbs_scarf_preserving_draw_order(self):
        mat, plan = _make_sample_grounded_material_and_plan()
        diag = {
            "exposure_stress": {
                "status": "fail",
                "threshold_px": 120,
                "levels": {
                    "1.0x": {"frames": [{"time": 1.0, "area_px": 1500, "regions": [{"area_px": 1500, "layers": ["scarf", "torso"]}]}]},
                    "1.5x": {"frames": [{"time": 1.0, "area_px": 2400, "regions": [{"area_px": 2400, "layers": ["scarf", "torso"]}]}]},
                },
            }
        }
        res = rebind_dead_crossings(mat, plan, diagnostics=diag)
        self.assertEqual(res.status, "rebound")
        self.assertEqual(len(res.report["rebinds"]), 1)
        rb = res.report["rebinds"][0]
        self.assertEqual(rb["secondary_module"], "scarf")
        self.assertEqual(rb["anchor_module"], "torso")

        scarf_piece = next(p for p in res.material["pieces"] if p["id"] == "scarf")
        torso_piece = next(p for p in res.material["pieces"] if p["id"] == "torso")
        self.assertEqual(scarf_piece["module_id"], "torso")
        self.assertEqual(scarf_piece["parent"], torso_piece["parent"])
        self.assertEqual(scarf_piece["pivot"], torso_piece["pivot"])
        self.assertEqual(scarf_piece["draw_order"], 10)  # original draw_order untouched!

        mod_ids = [m["id"] for m in res.plan["modules"]]
        self.assertNotIn("scarf", mod_ids)
        self.assertIn("torso", mod_ids)
        self.assertIn("lower_body", mod_ids)
        self.assertEqual(len(res.plan["connections"]), 2)  # at_scarf self-loop pruned
        self.assertEqual(res.plan["occlusions"], [])  # self-occlusion pruned

    def test_grounded_scarf_lower_body_exposure_anchors_to_static_lower_body(self):
        mat, plan = _make_sample_grounded_material_and_plan()
        res = rebind_dead_crossings(mat, plan, failing_pairs=[("scarf", "lower_body")])
        self.assertEqual(res.status, "rebound")
        scarf_piece = next(p for p in res.material["pieces"] if p["id"] == "scarf")
        self.assertEqual(scarf_piece["module_id"], "lower_body")
        self.assertIsNone(scarf_piece["parent"])
        self.assertEqual(scarf_piece["draw_order"], 10)

    def test_refuses_absorbing_protected_torso_into_static_lower_body(self):
        mat, plan = _make_sample_grounded_material_and_plan()
        res = rebind_dead_crossings(mat, plan, failing_pairs=[("torso", "lower_body")])
        self.assertEqual(res.status, "requires_human_review")
        self.assertEqual(res.report["rebinds"], [])
        self.assertIn("both_primary_structural_roles", res.report["blocked"][0]["reason"])

    def test_refuses_collapsing_two_module_airborne_to_single_slab(self):
        mat = {
            "version": 1,
            "material_mode": "canvas_layers",
            "image_size": [256, 256],
            "pieces": [
                {"id": "torso", "module_id": "torso", "parent": None, "kind": "rigid", "pivot": [128.0, 128.0], "draw_order": 0},
                {"id": "air_fx", "module_id": "air_fx", "parent": None, "kind": "rigid", "pivot": [128.0, 100.0], "draw_order": 10},
            ],
        }
        plan = {
            "pose_type": "airborne",
            "modules": [
                {"id": "torso", "role": "torso", "parent": None, "motion": "breath", "motion_type": "rigid"},
                {"id": "air_fx", "role": "effect_group", "parent": None, "motion": "sway", "motion_type": "rigid"},
            ],
            "connections": [],
            "occlusions": [],
        }
        res = rebind_dead_crossings(mat, plan, failing_pairs=[("air_fx", "torso")])
        self.assertEqual(res.status, "requires_human_review")
        self.assertEqual(res.report["blocked"][0]["reason"], "would_collapse_to_single_slab")


class SiblingVariantTransferAndGovernanceTests(unittest.TestCase):
    def _build_donor_bundle(self, td: Path):
        h, w = 128, 128
        torso = np.zeros((h, w, 4), dtype=np.uint8)
        torso[20:110, 35:95] = [120, 80, 160, 255]
        bottles = np.zeros((h, w, 4), dtype=np.uint8)
        bottles[15:45, 95:118] = [40, 210, 120, 255]
        swirl = np.zeros((h, w, 4), dtype=np.uint8)
        swirl[48:90, 96:116] = [50, 230, 110, 180]

        donor_src = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        for arr in (torso, swirl, bottles):
            donor_src.alpha_composite(Image.fromarray(arr))
        donor_src_p = td / "donor_src.png"
        donor_src.save(donor_src_p)
        donor_sha = _sha(donor_src_p)

        # Target sibling: identical bottles & swirl, localized censored patch on torso chest (y=40..55, x=50..75)
        target_arr = np.asarray(donor_src, dtype=np.uint8).copy()
        target_arr[40:55, 50:75] = [30, 30, 45, 255]
        target_src_p = td / "target_src.png"
        Image.fromarray(target_arr).save(target_src_p)

        donor_parts = td / "donor_parts"
        donor_parts.mkdir()
        pieces = []
        for pid, order, arr in (("torso", 0, torso), ("swirl_fx", 10, swirl), ("bottles", 20, bottles)):
            p = donor_parts / f"{pid}.png"
            Image.fromarray(arr).save(p)
            psha = _sha(p)
            pieces.append(
                {
                    "id": pid,
                    "module_id": pid,
                    "label": pid,
                    "parent": None,
                    "kind": "rigid",
                    "pivot": [64.0, 64.0],
                    "draw_order": order,
                    "image": str(p),
                    "sha256": psha,
                    "provenance": {
                        "operation": "split-layers",
                        "scene_id": "s1",
                        "node_id": f"n-{pid}",
                        "task_id": "t1",
                        "source_asset_sha256": donor_sha,
                        "result_asset_sha256": psha,
                        "transform": [[1, 0, 0], [0, 1, 0]],
                    },
                }
            )
        mat = {
            "version": 1,
            "material_mode": "canvas_layers",
            "source_image": str(donor_src_p),
            "source_sha256": donor_sha,
            "image_size": [w, h],
            "pieces": pieces,
        }
        plan = {
            "pose_type": "airborne",
            "anatomy": "humanoid_full",
            "modules": [
                {"id": "torso", "role": "torso", "parent": None, "pivot": [0.5, 0.5], "motion": "breath", "motion_type": "rigid"},
                {"id": "swirl_fx", "role": "effect_group", "parent": None, "pivot": [0.5, 0.5], "motion": "sway", "motion_type": "rigid"},
                {"id": "bottles", "role": "prop", "parent": None, "pivot": [0.5, 0.5], "motion": "sway", "motion_type": "rigid"},
            ],
            "connections": [],
            "occlusions": [],
        }
        mat_p = td / "donor_material.json"
        plan_p = td / "donor_plan.json"
        mat_p.write_text(json.dumps(mat))
        plan_p.write_text(json.dumps(plan))
        return mat_p, plan_p, donor_src_p, target_src_p

    def test_sibling_variant_transfer_lossless_and_passes_policy_and_materializer(self):
        with tempfile.TemporaryDirectory() as tmp:
            td = Path(tmp)
            mat_p, plan_p, _, target_src_p = self._build_donor_bundle(td)
            out_dir = td / "transferred"
            proc = subprocess.run(
                [
                    sys.executable, "scripts/canvas_handoff.py", "transfer-sibling-variant",
                    "--donor-material", str(mat_p),
                    "--donor-plan", str(plan_p),
                    "--target-source", str(target_src_p),
                    "--out-dir", str(out_dir),
                ],
                capture_output=True, text=True, check=True,
            )
            rep = json.loads(proc.stdout)
            self.assertTrue(rep["ok"])
            self.assertEqual(rep["rest_composite_max_diff"], 0)
            self.assertLess(rep["diff_ratio"], 0.10)
            self.assertEqual(rep["diff_component_assignments"][0]["assigned_piece"], "torso")

            new_mat = json.loads((out_dir / "material.json").read_text())
            parts_dir = td / "materialized_parts"
            materialize_canvas_layer_parts(new_mat, parts_dir, evidence_root=td)
            inspected = inspect_canvas_parts_warnings(parts_dir, evidence_root=td)
            pol = inspected["canvas_fallback_policy"]
            self.assertTrue(pol["passed"], pol)
            self.assertFalse(pol["bypassed"], pol)
            self.assertEqual(sorted(pol["sibling_transfer_layers"]), ["bottles", "swirl_fx", "torso"])

    def test_sibling_transfer_rejects_unrelated_image_and_self_transfer(self):
        with tempfile.TemporaryDirectory() as tmp:
            td = Path(tmp)
            mat_p, plan_p, donor_src_p, _ = self._build_donor_bundle(td)
            # Self transfer rejected
            with self.assertRaises(ValueError):
                transfer_sibling_variant(mat_p, plan_p, donor_src_p, td / "self_out")

            # Unrelated image (>15% diff) rejected
            unrelated = np.zeros((128, 128, 4), dtype=np.uint8)
            unrelated[10:115, 10:115] = [250, 40, 40, 255]
            unrel_p = td / "unrelated.png"
            Image.fromarray(unrelated).save(unrel_p)
            with self.assertRaises(ValueError):
                transfer_sibling_variant(mat_p, plan_p, unrel_p, td / "unrel_out")

    def test_spoofed_sibling_transfer_blocked_by_policy(self):
        same_sha = "a" * 64
        pol = check_canvas_fallback_policy(
            [
                {
                    "name": "torso",
                    "enabled": True,
                    "provenance": {
                        "operation": "local-direct-split",
                        "local_derivation": {
                            "method": "sibling_variant_transfer",
                            "parent_aligned_sha256": same_sha,
                            "donor_source_sha256": same_sha,  # spoofed self-transfer!
                            "donor_manifest_sha256": "b" * 64,
                            "diff_ratio": 0.01,
                        },
                    },
                }
            ]
        )
        self.assertFalse(pol["passed"])
        self.assertTrue(pol["bypassed"])

    def test_unapproved_per_character_override_blocked_by_policy_and_materialized_gate(self):
        bad = check_per_character_override_policy({"per_character_override": {"approved_by_user": False}})
        self.assertFalse(bad["passed"])
        self.assertEqual(bad["code"], "unapproved_per_character_override")

        good = check_per_character_override_policy(
            {"per_character_override": {"approved_by_user": True, "approval_note": "用户已审核同意针对17号微调腿甲"}}
        )
        self.assertTrue(good["passed"])
        self.assertTrue(good["approved_by_user"])

        with tempfile.TemporaryDirectory() as tmp:
            td = Path(tmp)
            mat_p, plan_p, _, target_src_p = self._build_donor_bundle(td)
            out_dir = td / "transferred"
            transfer_sibling_variant(mat_p, plan_p, target_src_p, out_dir)
            new_mat = json.loads((out_dir / "material.json").read_text())
            new_mat["per_character_override"] = {"approved_by_user": False, "note": "unauthorized"}
            parts_dir = td / "override_parts"
            materialize_canvas_layer_parts(new_mat, parts_dir, evidence_root=td)
            inspected = inspect_canvas_parts_warnings(parts_dir, evidence_root=td)
            codes = [w.get("code") for w in inspected["warnings"]]
            self.assertIn("unapproved_per_character_override", codes)

    def test_studio_traffic_light_blocks_unapproved_per_character_override(self):
        from src.studio.server import StudioJobManager

        root_dir = Path(__file__).resolve().parents[2] / ".studio-work"
        with tempfile.TemporaryDirectory(dir=root_dir) as tmp:
            mgr = StudioJobManager(Path(tmp))
            job = {
                "status": "canvas_preview",
                "canvas_mode": True,
                "canvas_checks": {
                    "files_complete": True,
                    "loop_seam": True,
                    "original_coverage": True,
                    "rigid_no_stretch": True,
                },
                "canvas_diagnostics": {
                    "exposure_stress": {"status": "pass"},
                    "cleanup": {"recovered_pixel_count": 0, "recovered_region_count": 0},
                    "cleanup_warnings": [{"code": "unapproved_per_character_override"}],
                },
            }
            light, reason = mgr._compute_item_traffic_light({"status": "done"}, job)
            self.assertEqual(light, "red")
            self.assertIn("未经用户审核批准", reason)


if __name__ == "__main__":
    unittest.main()
