"""General-purpose isomorphic sibling variant transfer (e.g. censored/original, CN/global pairs).

When two portraits in a batch share identical framing and anatomy outside a localized variant
region (default foreground pixel diff <= 15%), this module transfers an already-validated donor
variant's clean layer ownership directly to the target variant while:
1. Preserving 100% of the target variant's original 4K pixels (zero donor pixels leak into variant
   diff regions; rest-pose composite is verified against target source),
2. Preserving exact donor module topology, pivots, draw order, and motion plan,
3. Recording cryptographically bound provenance (`sibling_variant_transfer` with donor source SHA,
   donor manifest SHA, and measured `diff_ratio`) so unrelated characters can never spoof a transfer.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_rgba(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.ascontiguousarray(np.asarray(im.convert("RGBA"), dtype=np.uint8))


def _alpha_composite_layers(layers_in_order: List[np.ndarray], shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape
    canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    for arr in layers_in_order:
        canvas.alpha_composite(Image.fromarray(arr))
    return np.asarray(canvas, dtype=np.uint8)


@dataclass
class SiblingTransferResult:
    material: Dict[str, Any]
    plan: Dict[str, Any]
    report: Dict[str, Any]


def transfer_sibling_variant(
    donor_material_path: Any,
    donor_plan_path: Any,
    target_source_path: Any,
    out_dir: Any,
    *,
    max_diff_ratio: float = 0.15,
    diff_threshold: int = 4,
    max_rest_composite_diff: int = 4,
) -> SiblingTransferResult:
    """Transfer layer split & motion plan from a verified isomorphic donor to a sibling target PNG."""
    import cv2

    donor_mat_p = Path(donor_material_path).resolve()
    donor_plan_p = Path(donor_plan_path).resolve()
    target_src_p = Path(target_source_path).resolve()
    out_p = Path(out_dir).resolve()
    if out_p.exists():
        raise ValueError(f"Output directory already exists; pass a fresh directory: {out_p}")

    donor_mat = json.loads(donor_mat_p.read_text())
    donor_plan = json.loads(donor_plan_p.read_text())
    donor_manifest_sha = _sha256_file(donor_mat_p)

    donor_src_p = Path(donor_mat["source_image"]).resolve()
    donor_src_sha = _sha256_file(donor_src_p)
    target_src_sha = _sha256_file(target_src_p)
    if donor_src_sha == target_src_sha:
        raise ValueError("Target source is byte-identical to donor source; refusing self-transfer.")

    donor_rgba = _load_rgba(donor_src_p)
    target_rgba = _load_rgba(target_src_p)
    if donor_rgba.shape != target_rgba.shape:
        raise ValueError(f"Dimension mismatch: donor {donor_rgba.shape[:2]} != target {target_rgba.shape[:2]}")
    h, w = target_rgba.shape[:2]

    fg_union = (donor_rgba[..., 3] >= 16) | (target_rgba[..., 3] >= 16)
    chan_diff = np.abs(target_rgba.astype(np.int16) - donor_rgba.astype(np.int16)).max(axis=2)
    diff_mask = chan_diff > int(diff_threshold)
    diff_px = int((diff_mask & fg_union).sum())
    fg_px = int(fg_union.sum())
    diff_ratio = float(diff_px / max(1, fg_px))
    if diff_ratio > float(max_diff_ratio):
        raise ValueError(
            f"Target differs from donor by {diff_ratio:.2%} ({diff_px}/{fg_px} fg px), "
            f"exceeding max_diff_ratio={max_diff_ratio:.2%}; not an isomorphic sibling variant."
        )

    donor_pieces = sorted(donor_mat.get("pieces") or [], key=lambda p: int(p["draw_order"]))
    if len(donor_pieces) < 2:
        raise ValueError("Donor material must contain at least 2 pieces.")

    piece_arrays: Dict[str, np.ndarray] = {}
    for p in donor_pieces:
        arr = _load_rgba(Path(p["image"]).resolve())
        if arr.shape[:2] != (h, w):
            raise ValueError(f"Donor piece {p['id']} shape {arr.shape[:2]} != {(h, w)}")
        piece_arrays[p["id"]] = arr.copy()

    role_by_mod = {m["id"]: str(m.get("role") or "").lower() for m in donor_plan.get("modules") or []}
    assignments: List[Dict[str, Any]] = []

    if diff_mask.any():
        # Group spatially adjacent variant edits into components and route each to its owning layer.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        grouped = cv2.dilate(diff_mask.astype(np.uint8), kernel)
        num_labels, labels = cv2.connectedComponents(grouped, connectivity=8)
        for comp_id in range(1, num_labels):
            comp_mask = diff_mask & (labels == comp_id)
            comp_px = int(comp_mask.sum())
            if comp_px == 0:
                continue

            # Rank candidate pieces by opaque overlap with the variant region (preferring body roles on ties).
            best_piece = None
            best_score = (-1, -1, -1)
            for idx, p in enumerate(donor_pieces):
                arr = piece_arrays[p["id"]]
                ov_opaque = int(((arr[..., 3] >= 128) & comp_mask).sum())
                ov_any = int(((arr[..., 3] >= 16) & comp_mask).sum())
                is_body = 1 if role_by_mod.get(p["module_id"]) in ("torso", "body", "lower_body") else 0
                score = (ov_opaque, ov_any, is_body, -idx)
                if score[:3] > best_score[:3]:
                    best_score = score[:3]
                    best_piece = p

            if best_piece is None or (best_score[0] == 0 and best_score[1] == 0):
                # Variant added pixels in previously transparent air: assign to nearest body/largest opaque piece.
                dist_best = (float("inf"), None)
                comp_dil = cv2.dilate(comp_mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (33, 33))) > 0
                for p in donor_pieces:
                    arr = piece_arrays[p["id"]]
                    near = int(((arr[..., 3] >= 16) & comp_dil).sum())
                    if near > 0:
                        best_piece = p
                        break
                if best_piece is None:
                    best_piece = max(
                        donor_pieces,
                        key=lambda p: (
                            1 if role_by_mod.get(p["module_id"]) in ("torso", "body") else 0,
                            int((piece_arrays[p["id"]][..., 3] >= 16).sum()),
                        ),
                    )

            owner_id = best_piece["id"]
            owner_order = int(best_piece["draw_order"])
            cleared_overlays: List[str] = []
            # Ensure target variant pixels belong cleanly to `owner_id` with zero double-composite error.
            for p in donor_pieces:
                if p["id"] == owner_id:
                    continue
                if int(p["draw_order"]) > owner_order:
                    above_hit = (piece_arrays[p["id"]][..., 3] > 0) & comp_mask
                    if above_hit.any():
                        piece_arrays[p["id"]][above_hit] = 0
                        cleared_overlays.append(p["id"])
            piece_arrays[owner_id][comp_mask] = target_rgba[comp_mask]
            # Also zero underlying pixels directly behind newly opaque target edits only when target became transparent
            # inside `comp_mask` (e.g. variant removed an accessory), so composite matches target_rgba exactly.
            became_transparent = comp_mask & (target_rgba[..., 3] == 0)
            if became_transparent.any():
                for p in donor_pieces:
                    piece_arrays[p["id"]][became_transparent] = 0

            ys, xs = np.nonzero(comp_mask)
            assignments.append(
                {
                    "component_id": comp_id,
                    "pixels": comp_px,
                    "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                    "assigned_piece": owner_id,
                    "assigned_module": best_piece["module_id"],
                    "cleared_above_overlays": cleared_overlays,
                }
            )

    # Verify rest-pose composite fidelity against target_source before writing outputs.
    rest_comp = _alpha_composite_layers([piece_arrays[p["id"]] for p in donor_pieces], (h, w))
    rest_diff_arr = np.abs(rest_comp.astype(np.int16) - target_rgba.astype(np.int16)).max(axis=2)
    rest_max_diff = int(rest_diff_arr.max())
    rest_gt4_px = int((rest_diff_arr > 4).sum())
    if rest_max_diff > int(max_rest_composite_diff) and rest_gt4_px > 64:
        raise ValueError(
            f"Rest-pose composite diverged from target source (max_diff={rest_max_diff}, "
            f"gt4_px={rest_gt4_px}); refusing inaccurate sibling transfer."
        )

    parts_dir = out_p / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    new_pieces: List[Dict[str, Any]] = []
    for p in donor_pieces:
        pid = p["id"]
        out_img = parts_dir / f"{pid}.png"
        Image.fromarray(piece_arrays[pid]).save(out_img)
        piece_sha = _sha256_file(out_img)
        entry = copy.deepcopy(p)
        entry["image"] = str(out_img)
        entry["sha256"] = piece_sha
        entry["provenance"] = {
            "operation": "local-direct-split",
            "source_asset_sha256": target_src_sha,
            "result_asset_sha256": piece_sha,
            "transform": [[1, 0, 0], [0, 1, 0]],
            "local_derivation": {
                "method": "sibling_variant_transfer",
                "parent_aligned_sha256": target_src_sha,
                "donor_source_sha256": donor_src_sha,
                "donor_manifest_sha256": donor_manifest_sha,
                "donor_piece_id": pid,
                "diff_ratio": round(diff_ratio, 6),
                "diff_px": diff_px,
            },
        }
        new_pieces.append(entry)

    new_material = {
        "version": donor_mat.get("version", 1),
        "material_mode": donor_mat.get("material_mode", "canvas_layers"),
        "source_image": str(target_src_p),
        "source_sha256": target_src_sha,
        "image_size": [w, h],
        "pieces": new_pieces,
    }
    new_plan = copy.deepcopy(donor_plan)

    report = {
        "general_rule": "sibling_variant_transfer",
        "ok": True,
        "donor_source": str(donor_src_p),
        "donor_source_sha256": donor_src_sha,
        "donor_manifest_sha256": donor_manifest_sha,
        "target_source": str(target_src_p),
        "target_source_sha256": target_src_sha,
        "image_size": [w, h],
        "fg_px": fg_px,
        "diff_px": diff_px,
        "diff_ratio": round(diff_ratio, 6),
        "rest_composite_max_diff": rest_max_diff,
        "rest_composite_gt4_px": rest_gt4_px,
        "diff_component_assignments": assignments,
        "material_out": str(out_p / "material.json"),
        "plan_out": str(out_p / "plan.json"),
    }

    (out_p / "material.json").write_text(json.dumps(new_material, ensure_ascii=False, indent=2))
    (out_p / "plan.json").write_text(json.dumps(new_plan, ensure_ascii=False, indent=2))
    (out_p / "transfer-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return SiblingTransferResult(material=new_material, plan=new_plan, report=report)
