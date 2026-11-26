import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.pipeline.canvas_layer_cleanup import cleanup_layers, prepare_canvas_parts
from src.pipeline.canvas_layer_parts import materialize_canvas_layer_parts
from src.pipeline.local_body_split import BodyLayer, split_local_body


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestLocalDirectSplit(unittest.TestCase):
    def _build_fixture(self, root: Path):
        h, w = 80, 60
        src = np.zeros((h, w, 4), dtype=np.uint8)
        src[10:40, 15:45] = (200, 140, 90, 255)   # head / upper
        src[35:70, 12:48] = (70, 95, 140, 255)    # torso / lower
        src_path = root / "matted.png"
        Image.fromarray(src, mode="RGBA").save(src_path)

        m_body = np.zeros((h, w), dtype=bool)
        m_body[35:70, 12:48] = True
        m_head = np.zeros((h, w), dtype=bool)
        m_head[10:40, 15:45] = True

        split = split_local_body(
            src,
            [BodyLayer("torso", m_body, 0, False), BodyLayer("head", m_head, 1, False)],
            padding_px=12,
        )
        p_torso = root / "torso.png"
        p_head = root / "head.png"
        Image.fromarray(split.rgba["torso"], mode="RGBA").save(p_torso)
        Image.fromarray(split.rgba["head"], mode="RGBA").save(p_head)
        src_sha = _sha(src_path)

        def prov(img_path: Path):
            return {
                "operation": "local-direct-split",
                "source_asset_sha256": src_sha,
                "result_asset_sha256": _sha(img_path),
                "transform": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                "local_derivation": {
                    "method": "edge_refined_seed_mask",
                    "parent_aligned_sha256": src_sha,
                },
            }

        manifest = {
            "version": 1,
            "material_mode": "canvas_layers",
            "source_image": str(src_path),
            "source_sha256": src_sha,
            "image_size": [w, h],
            "pieces": [
                {
                    "id": "torso",
                    "module_id": "torso",
                    "parent": None,
                    "kind": "rigid",
                    "pivot": [30, 52],
                    "draw_order": 0,
                    "label": "躯干",
                    "image": str(p_torso),
                    "sha256": _sha(p_torso),
                    "provenance": prov(p_torso),
                },
                {
                    "id": "head",
                    "module_id": "head",
                    "parent": "torso",
                    "kind": "rigid",
                    "pivot": [30, 35],
                    "draw_order": 1,
                    "label": "头部",
                    "image": str(p_head),
                    "sha256": _sha(p_head),
                    "provenance": prov(p_head),
                },
            ],
        }
        return src, manifest

    def test_local_direct_split_materializes_without_fake_canvas_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, manifest = self._build_fixture(root)
            out = materialize_canvas_layer_parts(manifest, root / "parts", evidence_root=root)
            self.assertEqual(len(out["parts"]), 2)
            self.assertEqual(out["parts"][0]["canvasLayer"]["provenance"]["operation"], "local-direct-split")
            self.assertNotIn("scene_id", out["parts"][0]["canvasLayer"]["provenance"])
            self.assertTrue((root / "parts/parts.json").is_file())

    def test_local_direct_split_rejects_mismatched_sha_or_missing_method(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, manifest = self._build_fixture(root)

            bad1 = json.loads(json.dumps(manifest))
            bad1["pieces"][0]["provenance"]["result_asset_sha256"] = "0" * 64
            with self.assertRaises(ValueError):
                materialize_canvas_layer_parts(bad1, root / "parts_bad1", evidence_root=root)

            bad2 = json.loads(json.dumps(manifest))
            bad2["pieces"][0]["provenance"]["local_derivation"] = {}
            with self.assertRaises(ValueError):
                materialize_canvas_layer_parts(bad2, root / "parts_bad2", evidence_root=root)

    def test_local_direct_split_bleed_check_catches_copied_pixels_and_allows_clean_underlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src, manifest = self._build_fixture(root)
            clean_layers = []
            for piece in manifest["pieces"]:
                rgba = np.asarray(Image.open(piece["image"]).convert("RGBA"))
                clean_layers.append({
                    "id": piece["id"],
                    "rgba": rgba,
                    "role": piece["id"],
                    "kind": piece["kind"],
                    "draw_order": piece["draw_order"],
                    "enabled": True,
                    "effect_mode": "keep",
                    "provenance": piece["provenance"],
                })
            _, report, _ = cleanup_layers(clean_layers, src)
            bleeds = [w for w in report["warnings"] if w.get("code") == "local_layer_pixel_bleed"]
            self.assertEqual(bleeds, [])

            # Deliberately copy >250 identical source pixels onto both layers -> must trigger bleed warning!
            leaky = [dict(layer, rgba=layer["rgba"].copy()) for layer in clean_layers]
            leaky[0]["rgba"][20:40, 15:45] = src[20:40, 15:45]
            _, bad_report, _ = cleanup_layers(leaky, src)
            bad_bleeds = [w for w in bad_report["warnings"] if w.get("code") == "local_layer_pixel_bleed"]
            self.assertEqual(len(bad_bleeds), 1)

    def test_local_direct_split_prepare_canvas_parts_cleanup_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, manifest = self._build_fixture(root)
            # Inject an out-of-silhouette stray pixel so cleanup genuinely rewrites torso.png
            torso_path = Path(manifest["pieces"][0]["image"])
            torso_rgba = np.asarray(Image.open(torso_path).convert("RGBA")).copy()
            torso_rgba[0:3, 0:3] = (240, 240, 240, 200)
            Image.fromarray(torso_rgba, mode="RGBA").save(torso_path)
            pre_sha = _sha(torso_path)
            manifest["pieces"][0]["sha256"] = pre_sha
            manifest["pieces"][0]["provenance"]["result_asset_sha256"] = pre_sha

            materialize_canvas_layer_parts(manifest, root / "parts_raw", evidence_root=root)
            plan = {
                "version": 1,
                "modules": [
                    {"id": "torso", "role": "torso", "kind": "rigid", "parent": None, "pivot": [30, 52]},
                    {"id": "head", "role": "head", "kind": "rigid", "parent": "torso", "pivot": [30, 35]},
                ],
            }
            cleaned_parts_dir, _, _ = prepare_canvas_parts(
                root / "parts_raw",
                plan,
                root / "cleaned",
                evidence_root=root,
                cleanup=True,
            )
            cleaned = json.loads((cleaned_parts_dir / "parts.json").read_text())
            torso_entry = next(p for p in cleaned["parts"] if p["name"] == "torso")
            prov = torso_entry["canvasLayer"]["provenance"]
            self.assertEqual(prov["operation"], "local-direct-split")
            self.assertEqual(prov["cleanup_pre_result_sha256"], pre_sha)
            self.assertEqual(prov["result_asset_sha256"], torso_entry["canvasLayer"]["sha256"])
            self.assertNotEqual(prov["result_asset_sha256"], pre_sha)


if __name__ == "__main__":
    unittest.main()
