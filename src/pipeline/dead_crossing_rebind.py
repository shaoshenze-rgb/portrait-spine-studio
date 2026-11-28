"""General-purpose dead-crossing rebind fallback for unresolvable seam exposure pairs.

When exposure stress testing identifies a layer pair whose occluded backing cannot be obtained
cleanly (e.g. flat overlapping scarves, skirts, straps, or props with no continuous artwork behind
them), this module deterministically rebinds the secondary crossing module onto its anchor module
while preserving:
1. 100% original piece `draw_order` (never flips front/back layering),
2. 100% original 4K piece pixels, SHA-256 hashes, and provenance,
3. Grounded static lower-body / foot anchors (never introduces foot drift),
4. Non-monolithic animation safety (never collapses an airborne portrait to a single rigid slab or
   freezes the primary `torso` breathing module).

Zero character IDs or hardcoded coordinates exist in this module; every decision comes strictly
from `preview-diagnostics.json` / `exposure_stress`, `material.json`, and `plan.json`.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Anatomical anchor priority: lower tier number = stronger anchor (absorbed INTO, not absorbed away).
ROLE_ANCHOR_TIER: Dict[str, int] = {
    "ground": 0,
    "foot": 0,
    "lower_body": 0,
    "torso": 1,
    "body": 1,
    "head": 2,
    "arm": 2,
    "hand": 2,
    "weapon": 2,
    "cloth": 3,
    "scarf": 3,
    "skirt": 3,
    "cape": 3,
    "cloak": 3,
    "hair": 3,
    "tail": 3,
    "ribbon": 3,
    "prop": 3,
    "accessory": 3,
    "effect_group": 3,
    "fx": 3,
}

# Primary structural roles that must NEVER be automatically absorbed into another module without
# explicit human approval (absorbing `torso` into static `lower_body` freezes character breathing).
PROTECTED_PRIMARY_ROLES = frozenset({"ground", "foot", "lower_body", "torso", "body"})


@dataclass
class DeadCrossingRebindResult:
    status: str  # "rebound" | "noop_already_passing" | "requires_human_review"
    material: Dict[str, Any]
    plan: Dict[str, Any]
    report: Dict[str, Any]


def extract_failing_exposure_pairs(
    diagnostics: Dict[str, Any],
    *,
    min_area_px: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Aggregate failing layer pairs from `preview-diagnostics.json` or an `exposure_stress` dict."""
    stress = diagnostics.get("exposure_stress") if isinstance(diagnostics.get("exposure_stress"), dict) else diagnostics
    if not isinstance(stress, dict):
        return []
    threshold = int(stress.get("threshold_px") or 0) if min_area_px is None else int(min_area_px)
    levels = stress.get("levels") or {}
    pair_stats: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for level_name, lvl in levels.items():
        for frame in (lvl or {}).get("frames") or []:
            for reg in frame.get("regions") or []:
                area = int(reg.get("area_px") or 0)
                if area <= threshold:
                    continue
                layers = sorted({str(x) for x in (reg.get("layers") or []) if str(x)})
                if len(layers) < 2:
                    continue
                for i in range(len(layers)):
                    for j in range(i + 1, len(layers)):
                        key = (layers[i], layers[j])
                        rec = pair_stats.setdefault(
                            key,
                            {"layers": [key[0], key[1]], "max_area_px": 0, "total_area_px": 0, "hit_count": 0, "levels": set()},
                        )
                        rec["max_area_px"] = max(rec["max_area_px"], area)
                        rec["total_area_px"] += area
                        rec["hit_count"] += 1
                        rec["levels"].add(str(level_name))
    ranked = sorted(pair_stats.values(), key=lambda r: (r["max_area_px"], r["total_area_px"]), reverse=True)
    for rec in ranked:
        rec["levels"] = sorted(rec["levels"])
    return ranked


def _module_role_tier(module: Dict[str, Any], pose_type: Optional[str]) -> int:
    role = str(module.get("role") or "").strip().lower()
    if pose_type == "grounded" and module.get("motion_type") == "static" and role in ("ground", "foot", "lower_body"):
        return 0
    return ROLE_ANCHOR_TIER.get(role, 3)


def _is_animated_module(module: Dict[str, Any]) -> bool:
    motion = str(module.get("motion") or "").strip().lower()
    mtype = str(module.get("motion_type") or "").strip().lower()
    return motion != "static" and mtype != "static"


def _independent_motion_count(modules: Sequence[Dict[str, Any]], pose_type: Optional[str]) -> int:
    """Count distinct non-static motion groups after removing passive `follow` children."""
    animated = [m for m in modules if _is_animated_module(m)]
    if not animated:
        return 0
    independent = [m for m in animated if str(m.get("motion") or "").lower() != "follow"]
    if pose_type == "grounded":
        # Grounded pose with a static lower body and breathing upper torso has genuine relative
        # body motion even if `torso` is the sole independent moving module.
        has_static_anchor = any(not _is_animated_module(m) for m in modules)
        if has_static_anchor and len(independent) >= 1:
            return len(independent) + 1
    return len(independent)


def _choose_anchor_and_secondary(
    mod_a: Dict[str, Any],
    mod_b: Dict[str, Any],
    pose_type: Optional[str],
    piece_counts: Dict[str, int],
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[str]]:
    """Return `(anchor, secondary, block_reason)` for a failing module pair."""
    role_a = str(mod_a.get("role") or "").strip().lower()
    role_b = str(mod_b.get("role") or "").strip().lower()
    tier_a = _module_role_tier(mod_a, pose_type)
    tier_b = _module_role_tier(mod_b, pose_type)

    if role_a in PROTECTED_PRIMARY_ROLES and role_b in PROTECTED_PRIMARY_ROLES:
        return None, None, f"both_primary_structural_roles:{mod_a['id']}({role_a})_vs_{mod_b['id']}({role_b})"

    if tier_a < tier_b:
        anchor, secondary = mod_a, mod_b
    elif tier_b < tier_a:
        anchor, secondary = mod_b, mod_a
    else:
        # Tie-breaker: keep module with more pieces (or lexicographically earlier id) as anchor.
        ca, cb = piece_counts.get(mod_a["id"], 0), piece_counts.get(mod_b["id"], 0)
        if ca > cb or (ca == cb and mod_a["id"] <= mod_b["id"]):
            anchor, secondary = mod_a, mod_b
        else:
            anchor, secondary = mod_b, mod_a

    sec_role = str(secondary.get("role") or "").strip().lower()
    if sec_role in PROTECTED_PRIMARY_ROLES:
        return None, None, f"protected_primary_role:{secondary['id']}({sec_role})"
    return anchor, secondary, None


def rebind_dead_crossings(
    material: Dict[str, Any],
    plan: Dict[str, Any],
    *,
    diagnostics: Optional[Dict[str, Any]] = None,
    failing_pairs: Optional[Iterable[Tuple[str, str]]] = None,
    min_area_px: Optional[int] = None,
    max_rebinds: int = 4,
) -> DeadCrossingRebindResult:
    """Absorb unresolvable secondary crossing modules into their anchor modules without losing draw order.

    Returns a new deep-copied `(material, plan)` pair plus a structured audit report. Never mutates
    the caller's input dicts.
    """
    mat = copy.deepcopy(material)
    pl = copy.deepcopy(plan)
    pose_type = pl.get("pose_type")

    candidate_pairs: List[Tuple[str, str, Dict[str, Any]]] = []
    if failing_pairs is not None:
        for a, b in failing_pairs:
            candidate_pairs.append((str(a), str(b), {"layers": [str(a), str(b)], "source": "explicit_pairs"}))
    elif diagnostics is not None:
        for rec in extract_failing_exposure_pairs(diagnostics, min_area_px=min_area_px):
            candidate_pairs.append((rec["layers"][0], rec["layers"][1], rec))
    else:
        raise ValueError("Provide either diagnostics or failing_pairs.")

    if not candidate_pairs:
        return DeadCrossingRebindResult(
            status="noop_already_passing",
            material=mat,
            plan=pl,
            report={
                "general_rule": "dead_crossing_rebind",
                "status": "noop_already_passing",
                "rebinds": [],
                "blocked": [],
                "remaining_modules": [m["id"] for m in pl.get("modules") or []],
            },
        )

    rebinds: List[Dict[str, Any]] = []
    blocked: List[Dict[str, Any]] = []

    for piece_a, piece_b, meta in candidate_pairs:
        if len(rebinds) >= int(max_rebinds):
            break

        pieces = mat.get("pieces") or []
        modules = pl.get("modules") or []
        by_piece = {p["id"]: p for p in pieces}
        by_mod = {m["id"]: m for m in modules}

        mod_id_a = by_piece[piece_a]["module_id"] if piece_a in by_piece else piece_a
        mod_id_b = by_piece[piece_b]["module_id"] if piece_b in by_piece else piece_b
        if mod_id_a == mod_id_b:
            continue
        if mod_id_a not in by_mod or mod_id_b not in by_mod:
            blocked.append({"pair": [piece_a, piece_b], "reason": f"unknown_module:{mod_id_a},{mod_id_b}"})
            continue

        piece_counts: Dict[str, int] = {}
        for p in pieces:
            piece_counts[p["module_id"]] = piece_counts.get(p["module_id"], 0) + 1

        anchor, secondary, block_reason = _choose_anchor_and_secondary(
            by_mod[mod_id_a], by_mod[mod_id_b], pose_type, piece_counts
        )
        if block_reason or anchor is None or secondary is None:
            blocked.append({"pair": [piece_a, piece_b], "modules": [mod_id_a, mod_id_b], "reason": block_reason})
            continue

        sec_id = secondary["id"]
        anc_id = anchor["id"]

        # Non-single-slab guard: check remaining independent motion if `sec_id` is removed.
        hypothetical_modules = [m for m in modules if m["id"] != sec_id]
        if _independent_motion_count(hypothetical_modules, pose_type) < 2:
            blocked.append(
                {
                    "pair": [piece_a, piece_b],
                    "modules": [mod_id_a, mod_id_b],
                    "candidate_rebind": {"secondary": sec_id, "anchor": anc_id},
                    "reason": "would_collapse_to_single_slab",
                }
            )
            continue

        # Find representative piece of `anchor` to copy exact canvas-space pivot / parent / kind
        # (guaranteed compatible with `materialize_canvas_layer_parts` module shape invariants).
        anc_piece = next((p for p in pieces if p["module_id"] == anc_id), None)
        if anc_piece is None:
            blocked.append({"pair": [piece_a, piece_b], "reason": f"anchor_has_no_pieces:{anc_id}"})
            continue

        moved_pieces: List[str] = []
        reparented_children: List[str] = []
        for p in pieces:
            if p["module_id"] == sec_id:
                p["module_id"] = anc_id
                p["parent"] = anc_piece["parent"]
                p["kind"] = anc_piece["kind"]
                p["pivot"] = list(anc_piece["pivot"])
                # Strictly preserve original draw_order, image, sha256, and provenance!
                moved_pieces.append(p["id"])
            elif p.get("parent") == sec_id:
                p["parent"] = anc_id

        new_modules = []
        for m in modules:
            if m["id"] == sec_id:
                continue
            if m.get("parent") == sec_id:
                m["parent"] = anc_id
                reparented_children.append(m["id"])
            new_modules.append(m)
        pl["modules"] = new_modules

        # Rewrite/clean connections referencing `sec_id`.
        new_conns = []
        seen_conn_pairs = set()
        for conn in pl.get("connections") or []:
            ca = anc_id if conn.get("a") == sec_id else conn.get("a")
            cb = anc_id if conn.get("b") == sec_id else conn.get("b")
            if ca == cb:
                continue
            edge = (ca, cb, conn.get("type"))
            if edge in seen_conn_pairs:
                continue
            seen_conn_pairs.add(edge)
            updated = copy.deepcopy(conn)
            updated["a"] = ca
            updated["b"] = cb
            new_conns.append(updated)
        pl["connections"] = new_conns

        # Rewrite/clean occlusions referencing `sec_id`.
        if isinstance(pl.get("occlusions"), list):
            new_occs = []
            for occ in pl["occlusions"]:
                if isinstance(occ, dict):
                    fr = anc_id if occ.get("front") == sec_id else occ.get("front")
                    bk = anc_id if occ.get("back") == sec_id else occ.get("back")
                    if fr == bk:
                        continue
                    u = copy.deepcopy(occ)
                    if "front" in u:
                        u["front"] = fr
                    if "back" in u:
                        u["back"] = bk
                    new_occs.append(u)
                else:
                    new_occs.append(occ)
            pl["occlusions"] = new_occs

        rebinds.append(
            {
                "trigger_pair": [piece_a, piece_b],
                "secondary_module": sec_id,
                "secondary_role": secondary.get("role"),
                "anchor_module": anc_id,
                "anchor_role": anchor.get("role"),
                "moved_pieces": moved_pieces,
                "reparented_children": reparented_children,
                "draw_order_preserved": True,
                "trigger_meta": meta,
            }
        )

    status = "rebound" if rebinds else "requires_human_review"
    report = {
        "general_rule": "dead_crossing_rebind",
        "status": status,
        "pose_type": pose_type,
        "rebinds": rebinds,
        "blocked": blocked,
        "remaining_modules": [
            {"id": m["id"], "role": m.get("role"), "parent": m.get("parent"), "motion": m.get("motion")}
            for m in pl.get("modules") or []
        ],
        "independent_motion_groups": _independent_motion_count(pl.get("modules") or [], pose_type),
    }
    return DeadCrossingRebindResult(status=status, material=mat, plan=pl, report=report)
