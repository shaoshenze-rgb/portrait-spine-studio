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
from src.pipeline.canvas_edit_patch import inpaint_translucent_fx_underlap, plan_occluded_edit_crop
from src.pipeline.dead_crossing_rebind import rebind_dead_crossings, trim_hidden_silhouette_pokeout
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

    def test_partial_rebind_never_hides_blocked_pairs_and_cli_exits_2(self):
        mat, plan = _make_sample_grounded_material_and_plan()
        res = rebind_dead_crossings(mat, plan, failing_pairs=[("scarf", "torso"), ("torso", "lower_body")])
        self.assertEqual(res.status, "partial_requires_human_review")
        self.assertEqual(len(res.report["rebinds"]), 1)
        self.assertEqual(len(res.report["blocked"]), 1)

        with tempfile.TemporaryDirectory() as tmp:
            td = Path(tmp)
            (td / "mat.json").write_text(json.dumps(mat))
            (td / "plan.json").write_text(json.dumps(plan))
            proc = subprocess.run(
                [
                    sys.executable, "scripts/canvas_handoff.py", "rebind-dead-crossings",
                    "--material", str(td / "mat.json"),
                    "--plan", str(td / "plan.json"),
                    "--failing-pair", "scarf,torso",
                    "--failing-pair", "torso,lower_body",
                    "--out-dir", str(td / "out"),
                ],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
            out = json.loads(proc.stdout)
            self.assertFalse(out["ok"])
            self.assertEqual(out["status"], "partial_requires_human_review")

    def test_prepare_edit_crop_flags_too_wide_near_full_frame_and_cli_exits_2(self):
        h, w = 256, 256
        orig = np.zeros((h, w, 4), dtype=np.uint8)
        orig[20:236, 20:236] = [100, 120, 140, 255]
        front = np.zeros((h, w, 4), dtype=np.uint8)
        front[20:236, 20:236] = [200, 180, 90, 255]  # wraps almost entire canvas
        base = orig.copy()
        empty_diag = {"exposure_stress": {"status": "pass", "levels": {"1.0x": {"frames": []}, "1.5x": {"frames": []}}}}

        plan = plan_occluded_edit_crop(orig, front_layer_rgba=front, base_layer_rgba=base, diagnostics=empty_diag, min_pad_px=20, min_crop_edge=64)
        self.assertFalse(plan.meta["usable_for_local_edit"])
        self.assertEqual(plan.meta["status"], "crop_too_wide_requires_human_review")
        self.assertGreater(plan.meta["crop_area_ratio"], 0.55)
        self.assertFalse(plan.meta["diagnostics_matched"])
        self.assertTrue(any("diagnostics_no_matching" in w_msg for w_msg in plan.meta["warnings"]))

        with tempfile.TemporaryDirectory() as tmp:
            td = Path(tmp)
            Image.fromarray(orig).save(td / "orig.png")
            Image.fromarray(front).save(td / "front.png")
            Image.fromarray(base).save(td / "base.png")
            (td / "diag.json").write_text(json.dumps(empty_diag))
            proc = subprocess.run(
                [
                    sys.executable, "scripts/canvas_handoff.py", "prepare-edit-crop",
                    "--original", str(td / "orig.png"),
                    "--front-layer", str(td / "front.png"),
                    "--base-layer", str(td / "base.png"),
                    "--diagnostics", str(td / "diag.json"),
                    "--min-pad-px", "20", "--min-crop-edge", "64",
                    "--out-crop", str(td / "c.png"), "--out-mask", str(td / "m.png"), "--out-meta", str(td / "meta.json"),
                ],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(proc.returncode, 2)
            out = json.loads(proc.stdout)
            self.assertFalse(out["ok"])
            self.assertEqual(out["status"], "crop_too_wide_requires_human_review")

    def test_trim_hidden_silhouette_pokeout_removes_hidden_boot_edge_ghost_losslessly(self):
        h, w = 256, 256
        # Static lower_body boot occupies x=80..170, y=100..230; open air is outside.
        lower = np.zeros((h, w, 4), dtype=np.uint8)
        lower[100:230, 80:170] = [60, 70, 90, 255]
        # Moving scarf behind lower_body: visible only near top (y=40..105, x=90..150),
        # plus hidden cyan tail buried under boot right next to open air at y=180..228, x=80..120!
        scarf = np.zeros((h, w, 4), dtype=np.uint8)
        scarf[40:130, 90:150] = [0, 120, 140, 255]
        scarf[180:228, 80:120] = [0, 110, 125, 255]  # hidden poke-out risk along boot bottom-left!

        comp = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        comp.alpha_composite(Image.fromarray(scarf))
        comp.alpha_composite(Image.fromarray(lower))

        with tempfile.TemporaryDirectory() as tmp:
            td = Path(tmp)
            orig_p = td / "orig.png"
            comp.save(orig_p)
            scarf_p = td / "scarf.png"
            lower_p = td / "lower.png"
            Image.fromarray(scarf).save(scarf_p)
            Image.fromarray(lower).save(lower_p)
            orig_sha = _sha(orig_p)
            mat = {
                "version": 1, "material_mode": "canvas_layers",
                "source_image": str(orig_p), "source_sha256": orig_sha, "image_size": [w, h],
                "pieces": [
                    {"id": "scarf", "module_id": "torso", "parent": "lower_body", "kind": "rigid", "pivot": [128.0, 100.0],
                     "draw_order": 10, "image": str(scarf_p), "sha256": _sha(scarf_p),
                     "provenance": {"operation": "local-direct-split", "source_asset_sha256": orig_sha, "result_asset_sha256": _sha(scarf_p), "transform": [[1, 0, 0], [0, 1, 0]], "local_derivation": {"method": "psd_artist_layer_split", "parent_aligned_sha256": orig_sha}}},
                    {"id": "lower_body", "module_id": "lower_body", "parent": None, "kind": "rigid", "pivot": [128.0, 160.0],
                     "draw_order": 20, "image": str(lower_p), "sha256": _sha(lower_p),
                     "provenance": {"operation": "local-direct-split", "source_asset_sha256": orig_sha, "result_asset_sha256": _sha(lower_p), "transform": [[1, 0, 0], [0, 1, 0]], "local_derivation": {"method": "psd_artist_layer_split", "parent_aligned_sha256": orig_sha}}},
                ],
            }
            plan = {
                "pose_type": "grounded",
                "modules": [
                    {"id": "lower_body", "role": "lower_body", "parent": None, "motion": "static", "motion_type": "static"},
                    {"id": "torso", "role": "torso", "parent": "lower_body", "motion": "breath", "motion_type": "rigid"},
                ],
            }
            res = trim_hidden_silhouette_pokeout(mat, plan, td / "trimmed", keep_underlap_px=24, pokeout_margin_px=24)
            self.assertEqual(res.status, "trimmed")
            self.assertGreater(res.report["total_trimmed_px"], 1000)
            new_scarf = np.asarray(Image.open(res.material["pieces"][0]["image"]).convert("RGBA"))
            # Hidden poke-out pixels at boot edge are gone!
            self.assertEqual(int(new_scarf[210, 88, 3]), 0)
            # Genuine underlap right behind top visible emergence (y=108, within keep_underlap_px=24 of y=99) is KEPT!
            self.assertEqual(int(new_scarf[108, 120, 3]), 255)
            # Visible scarf pixels outside boot are 100% untouched!
            np.testing.assert_array_equal(new_scarf[40:100, 90:150], scarf[40:100, 90:150])

    def test_inpaint_translucent_fx_underlap_fills_cookie_cutter_hole_losslessly(self):
        h, w = 160, 160
        yy, xx = np.indices((h, w))
        dist = np.hypot(xx - 80, yy - 80)
        glow_a = np.clip((1.0 - dist / 65.0) * 190.0, 0, 190).astype(np.uint8)
        fx = np.zeros((h, w, 4), dtype=np.uint8)
        fx[..., 0] = 55
        fx[..., 1] = 225
        fx[..., 2] = 90
        fx[..., 3] = glow_a

        bottle = np.zeros((h, w, 4), dtype=np.uint8)
        bmask = (xx >= 66) & (xx < 94) & (yy >= 66) & (yy < 94)
        bottle[bmask] = [220, 200, 180, 255]
        # Hard cookie-cutter cutout in fx under bottle!
        fx_cut = fx.copy()
        fx_cut[bmask] = 0

        res = inpaint_translucent_fx_underlap(fx_cut, bottle, max_underlap_px=36, blur_sigma=3.0)
        self.assertTrue(res.stats["rest_pose_preserved"])
        self.assertEqual(res.stats["max_outside_diff"], 0)
        self.assertEqual(res.stats["composite_max_diff"], 0)
        self.assertEqual(res.stats["composite_changed_px"], 0)
        np.testing.assert_array_equal(res.rgba[~bmask], fx_cut[~bmask])
        self.assertGreater(int(res.rgba[80, 80, 3]), 110)
        self.assertGreater(int(res.rgba[80, 80, 1]), 180)  # clean green glow, no black bleed!

    def test_inpaint_translucent_fx_underlap_does_not_change_semitransparent_rim_composite(self):
        h, w = 160, 160
        yy, xx = np.indices((h, w))
        dist = np.hypot(xx - 80, yy - 80)
        glow_a = np.clip((1.0 - dist / 65.0) * 180.0, 0, 180).astype(np.uint8)
        fx = np.zeros((h, w, 4), dtype=np.uint8)
        fx[..., :3] = [50, 230, 85]
        fx[..., 3] = glow_a

        bottle = np.zeros((h, w, 4), dtype=np.uint8)
        outer = (xx >= 62) & (xx < 98) & (yy >= 62) & (yy < 98)
        core = (xx >= 66) & (xx < 94) & (yy >= 66) & (yy < 94)
        bottle[outer] = [210, 195, 175, 110]  # semi-transparent glass rim!
        bottle[core] = [210, 195, 175, 255]   # opaque bottle core
        fx_cut = fx.copy()
        fx_cut[outer] = 0

        res = inpaint_translucent_fx_underlap(fx_cut, bottle, max_underlap_px=36, blur_sigma=3.0)
        self.assertTrue(res.stats["rest_pose_preserved"])
        self.assertEqual(res.stats["composite_max_diff"], 0)
        self.assertEqual(res.stats["composite_changed_px"], 0)
        self.assertGreater(res.stats["filled_px"], 500)
        # Semi-transparent rim pixels remain untouched so rest composite never shifts!
        rim = outer & (~core)
        np.testing.assert_array_equal(res.rgba[rim], fx_cut[rim])

    def test_trim_hidden_pokeout_absorbs_static_fringe_slivers_with_zero_composite_diff(self):
        h, w = 256, 256
        lower = np.zeros((h, w, 4), dtype=np.uint8)
        lower[100:230, 80:170] = [60, 70, 90, 255]
        scarf = np.zeros((h, w, 4), dtype=np.uint8)
        scarf[40:130, 90:150] = [0, 120, 140, 255]   # substantial visible top cloak
        scarf[180:228, 80:120] = [0, 110, 125, 255]  # hidden cyan behind boot
        scarf[190:215, 77:80] = [0, 115, 130, 255]   # tiny 3x25 px sliver sticking out along boot edge!

        comp = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        comp.alpha_composite(Image.fromarray(scarf))
        comp.alpha_composite(Image.fromarray(lower))

        with tempfile.TemporaryDirectory() as tmp:
            td = Path(tmp)
            orig_p, scarf_p, lower_p = td / "orig.png", td / "scarf.png", td / "lower.png"
            comp.save(orig_p)
            Image.fromarray(scarf).save(scarf_p)
            Image.fromarray(lower).save(lower_p)
            orig_sha = _sha(orig_p)
            mat = {
                "version": 1, "material_mode": "canvas_layers",
                "source_image": str(orig_p), "source_sha256": orig_sha, "image_size": [w, h],
                "pieces": [
                    {"id": "scarf", "module_id": "torso", "parent": "lower_body", "kind": "rigid", "pivot": [128.0, 100.0],
                     "draw_order": 10, "image": str(scarf_p), "sha256": _sha(scarf_p),
                     "provenance": {"operation": "local-direct-split", "source_asset_sha256": orig_sha, "result_asset_sha256": _sha(scarf_p), "transform": [[1, 0, 0], [0, 1, 0]], "local_derivation": {"method": "psd_artist_layer_split", "parent_aligned_sha256": orig_sha}}},
                    {"id": "lower_body", "module_id": "lower_body", "parent": None, "kind": "rigid", "pivot": [128.0, 160.0],
                     "draw_order": 20, "image": str(lower_p), "sha256": _sha(lower_p),
                     "provenance": {"operation": "local-direct-split", "source_asset_sha256": orig_sha, "result_asset_sha256": _sha(lower_p), "transform": [[1, 0, 0], [0, 1, 0]], "local_derivation": {"method": "psd_artist_layer_split", "parent_aligned_sha256": orig_sha}}},
                ],
            }
            plan = {
                "pose_type": "grounded",
                "modules": [
                    {"id": "lower_body", "role": "lower_body", "parent": None, "motion": "static", "motion_type": "static"},
                    {"id": "torso", "role": "torso", "parent": "lower_body", "motion": "breath", "motion_type": "rigid"},
                ],
            }
            res = trim_hidden_silhouette_pokeout(mat, plan, td / "trimmed", keep_underlap_px=24, pokeout_margin_px=24)
            self.assertEqual(res.status, "trimmed")
            self.assertEqual(res.report["full_composite_max_diff"], 0)
            self.assertEqual(res.report["full_composite_changed_px"], 0)
            self.assertGreater(res.report["total_trimmed_px"], 1000)
            self.assertGreaterEqual(res.report["total_absorbed_fringe_sliver_px"], 75)
            new_scarf = np.asarray(Image.open(res.material["pieces"][0]["image"]).convert("RGBA"))
            new_lower = np.asarray(Image.open(res.material["pieces"][1]["image"]).convert("RGBA"))
            # Sliver is removed from moving scarf and cleanly preserved on static lower_body!
            self.assertEqual(int(new_scarf[200, 78, 3]), 0)
            self.assertEqual(int(new_lower[200, 78, 3]), 255)
            # Hidden boot interior was NOT shielded by the tiny sliver!
            self.assertEqual(int(new_scarf[200, 90, 3]), 0)

    def test_material_seams_skips_same_module_and_ghost_detector_catches_pokeout(self):
        from types import SimpleNamespace
        from src.gates.preview_diagnostics import _material_seams, inspect_dynamic_silhouette_ghost

        h, w = 200, 200
        with tempfile.TemporaryDirectory() as tmp:
            td = Path(tmp)
            pdir = td / "parts"
            pdir.mkdir()
            # Two pieces welded to same module `torso` (`scarf` + `torso`) plus static `lower_body`
            lower = np.zeros((h, w, 4), dtype=np.uint8)
            lower[100:180, 60:140] = [60, 70, 90, 255]
            torso = np.zeros((h, w, 4), dtype=np.uint8)
            torso[30:105, 70:130] = [150, 90, 80, 255]
            scarf = np.zeros((h, w, 4), dtype=np.uint8)
            scarf[40:102, 110:145] = [0, 120, 140, 255]
            Image.fromarray(lower).save(pdir / "lower_body.png")
            Image.fromarray(torso).save(pdir / "torso.png")
            Image.fromarray(scarf).save(pdir / "scarf.png")
            mat = {
                "parts": [
                    {"name": "scarf", "moduleId": "torso", "drawOrder": 10, "kind": "rigid", "enabled": True, "bbox": [0, 0, w, h], "image": "parts/scarf.png", "canvasLayer": {"provenance": {"operation": "local-direct-split"}}},
                    {"name": "lower_body", "moduleId": "lower_body", "drawOrder": 20, "kind": "rigid", "enabled": True, "bbox": [0, 0, w, h], "image": "parts/lower_body.png", "canvasLayer": {"provenance": {"operation": "local-direct-split"}}},
                    {"name": "torso", "moduleId": "torso", "drawOrder": 21, "kind": "rigid", "enabled": True, "bbox": [0, 0, w, h], "image": "parts/torso.png", "canvasLayer": {"provenance": {"operation": "local-direct-split"}}},
                ]
            }
            f0 = np.zeros((h, w, 4), dtype=np.uint8)
            f1 = np.zeros((h, w, 4), dtype=np.uint8)
            f0[30:180, 60:145] = 255
            f1[25:175, 65:150] = 255
            seams = _material_seams({0.0: f0, 1.0: f1}, [0.0, 1.0], td, mat)
            pairs = [set(s["pair"]) for s in seams]
            self.assertNotIn({"scarf", "torso"}, pairs)

            # Synthetic renderer to test dynamic_silhouette_ghost detector (fail when ghost pokes out, pass when clean)
            class _FakeRenderer:
                def __init__(self, poke_out: bool):
                    self.poke_out = poke_out
                    self.t = 0.0
                    self.sk = SimpleNamespace(by_name={
                        "module_lower_body": SimpleNamespace(a=1.0, b=0.0, c=0.0, d=1.0, worldX=100.0, worldY=50.0),
                        "module_torso": SimpleNamespace(a=1.0, b=0.0, c=0.0, d=1.0, worldX=100.0, worldY=100.0),
                    })
                def pose(self, t, gain=1.0):
                    self.t = t
                    self.sk.by_name["module_torso"].worldY = 100.0 + (8.0 * gain if t > 0 else 0.0)
                def frame(self, t, gain=1.0, slot_names=None):
                    self.pose(t, gain=gain)
                    slots = set(slot_names) if slot_names is not None else {"lower_body", "torso", "scarf"}
                    out = np.zeros((h, w, 4), dtype=np.uint8)
                    if "lower_body" in slots:
                        out[100:180, 60:140] = [60, 70, 90, 255]
                    if "torso" in slots or "scarf" in slots:
                        out[30:105, 70:130] = [150, 90, 80, 255]
                        if self.poke_out and t > 0:
                            # Hidden cyan scarf poking 10px left of static boot into open air (x=50..60, y=135..175)!
                            out[135:175, 50:60] = [0, 120, 140, 255]
                    return out

            mods = [
                {"id": "lower_body", "role": "lower_body", "bone": "module_lower_body"},
                {"id": "torso", "role": "torso", "bone": "module_torso"},
            ]
            g_fail, _ = inspect_dynamic_silhouette_ghost(_FakeRenderer(poke_out=True), td, mat, mods, [0.0, 1.0], threshold=60)
            self.assertEqual(g_fail["status"], "fail")
            self.assertGreaterEqual(g_fail["levels"]["1.5x"]["peak_ghost_px"], 300)

            g_pass, _ = inspect_dynamic_silhouette_ghost(_FakeRenderer(poke_out=False), td, mat, mods, [0.0, 1.0], threshold=60)
            self.assertEqual(g_pass["status"], "pass")
            self.assertEqual(g_pass["levels"]["1.5x"]["peak_ghost_px"], 0)


if __name__ == "__main__":
    unittest.main()
