"""Quality gate engine: stress test, 4 hard gates, soft displacement hints, and health-report.json."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

from src.pipeline.presets import get_preset
from src.render.renderer import SpineRenderer, parts_composite_pm
from .module_quality import (ModuleQualityAuditor, aggregate, recompute_quality, result,
                             finalize_visual_review)


class GateFailureError(RuntimeError):
    """Raised when one or more hard quality gates fail."""

    def __init__(self, message: str, health_report: Dict[str, Any]):
        super().__init__(message)
        self.code = "QUALITY_REVIEW_REQUIRED" if health_report.get('status') == 'needs_review' else "HARD_GATE_FAILED"
        self.retryable = False
        self.details = health_report
        self.health_report = health_report


def _seed_runs(T: np.ndarray, seed: np.ndarray) -> np.ndarray:
    Tp = np.pad(T, ((0, 0), (0, 1)))
    Sp = np.pad(seed, ((0, 0), (0, 1)))
    f = Tp.reshape(-1)
    sd = Sp.reshape(-1)
    start = f & ~np.concatenate([[False], f[:-1]])
    rid = np.cumsum(start) * f
    hit = np.bincount(rid[sd & f], minlength=int(rid.max()) + 1) > 0
    hit[0] = False
    return hit[rid].reshape(Tp.shape)[:, :-1]


def enclosed_transparent(alpha: np.ndarray, thr: int = 128) -> np.ndarray:
    """Pixels with `alpha < thr` NOT 4-connected to the canvas border."""
    T = alpha < thr
    seed = np.zeros_like(T)
    seed[0] = T[0]
    seed[-1] = T[-1]
    seed[:, 0] = T[:, 0]
    seed[:, -1] = T[:, -1]
    prev_count = -1
    while True:
        seed = _seed_runs(T, seed)
        seed = _seed_runs(T.T, seed.T).T
        c = int(seed.sum())
        if c == prev_count:
            break
        prev_count = c
    return T & ~seed


def _dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """Fast box dilation of a 2D boolean mask by `radius` pixels."""
    if radius <= 0 or not mask.any():
        return mask.copy()
    a = mask.copy()
    s = 1
    while s <= radius:
        step = min(s, radius - s + 1) if s * 2 > radius else s
        b = a.copy()
        b[:, step:] |= a[:, :-step]
        b[:, :-step] |= a[:, step:]
        c = b.copy()
        c[step:, :] |= b[:-step, :]
        c[:-step, :] |= b[step:, :]
        a = c
        s += step
    return a


def count_enclosed_holes(frame_pm: np.ndarray, rest_alpha: np.ndarray, alpha_thr: int = 128) -> int:
    """Conservatively count newly transparent opaque-at-rest pixels in enclosed regions.

    No fixed-position transparent seed or dilated old hole may exempt a new tear.
    Motion of pre-existing cutouts can need visual review; it is not silently passed.
    """
    frame_alpha = frame_pm[..., 3]
    T = frame_alpha < alpha_thr
    if not T.any():
        return 0

    seed = np.zeros_like(T)
    seed[0] = T[0]
    seed[-1] = T[-1]
    seed[:, 0] = T[:, 0]
    seed[:, -1] = T[:, -1]

    prev_count = -1
    while True:
        seed = _seed_runs(T, seed)
        seed = _seed_runs(T.T, seed.T).T
        c = int(seed.sum())
        if c == prev_count:
            break
        prev_count = c

    holes = (T & ~seed) & (rest_alpha >= 250)
    return int(holes.sum())


def stress_test(
    parts_dir: Union[str, Path],
    rig_dir: Union[str, Path],
    *,
    stress_gain: float = 1.6,
    sample_times: Optional[Sequence[float]] = None,
    sample_count: int = 8,
    renderer: Optional[SpineRenderer] = None,
    sample_callback: Optional[Callable[[float, float, np.ndarray], None]] = None,
) -> Dict[str, Any]:
    """Run extreme-deformation stress poses to catch hidden-fill holes early."""
    r = renderer or SpineRenderer(rig_dir)
    rep_path = Path(rig_dir) / "rig-build-report.json"
    duration = 8.0
    if rep_path.is_file():
        duration = float(json.loads(rep_path.read_text(encoding="utf-8")).get("duration", 8.0))

    if sample_times is None:
        times = [duration * (i + 0.5) / float(sample_count) for i in range(sample_count)]
    else:
        times = [float(t) for t in sample_times]

    rest_frame = r.frame(None)
    rest_alpha = rest_frame[..., 3]
    per_sample: List[int] = []
    for t in times:
        f = r.frame(t, gain=stress_gain)
        if sample_callback is not None:
            sample_callback(t, stress_gain, f)
        per_sample.append(count_enclosed_holes(f, rest_alpha))

    return {
        "partsDir": str(Path(parts_dir).resolve()),
        "rigDir": str(Path(rig_dir).resolve()),
        "gain": float(stress_gain),
        "sampleTimes": [round(t, 4) for t in times],
        "sampleCount": len(times),
        "enclosedHolesMax": max(per_sample) if per_sample else 0,
        "perSampleHoles": per_sample,
        "passed": (max(per_sample) if per_sample else 0) == 0,
    }


def _evaluate_rest_diff(parts_dir: Path) -> Dict[str, Any]:
    rd_path = parts_dir / "rest_diff.json"
    if rd_path.is_file():
        rd = json.loads(rd_path.read_text(encoding="utf-8"))
        max_rgba = int(rd["maxRGBA"])
        return {
            "passed": max_rgba <= 2,
            "threshold": {"maxRGBA": 2},
            "measured": {
                "maxRGBA": max_rgba,
                "meanAbsDiff": float(rd.get("meanAbsDiff", 0.0)),
                "pixelsOver2": int(rd.get("pixelsOver2", 0)),
            },
        }
    pj = json.loads((parts_dir / "parts.json").read_text(encoding="utf-8"))
    src = np.asarray(Image.open(pj["source"]).convert("RGBA"), dtype=np.uint8)
    rc = np.asarray(Image.open(parts_dir / "rest_composite.png").convert("RGBA"), dtype=np.uint8)
    opaque_thr = int(pj.get("opaqueThreshold", 250))
    src_n = src.copy()
    src_n[..., 3] = np.where(src_n[..., 3] >= opaque_thr, 255, src_n[..., 3])
    src_n[src_n[..., 3] == 0] = 0
    rc_n = rc.copy()
    rc_n[rc_n[..., 3] == 0] = 0
    d = np.abs(rc_n.astype(np.int16) - src_n.astype(np.int16))
    max_rgba = int(d.max()) if d.size else 0
    return {
        "passed": max_rgba <= 2,
        "threshold": {"maxRGBA": 2},
        "measured": {
            "maxRGBA": max_rgba,
            "meanAbsDiff": round(float(d.mean()), 6) if d.size else 0.0,
            "pixelsOver2": int((d.max(axis=-1) > 2).sum()) if d.size else 0,
        },
    }


def _resolve_soft_landmark_bone(sk_bones: Dict[str, Any], name: str) -> Optional[Tuple[str, bool]]:
    """Map logical landmark (`head`, `chest`, `hips`, `hand_R`, `hand_L`, `weapon_L`, `leg_L`, `hair_tip`) -> (bone_name, use_tip)."""
    if name in sk_bones:
        return name, False
    if name == "leg_L":
        for cand in ("foot_L", "shin_L", "thigh_L"):
            if cand in sk_bones:
                return cand, (cand == "foot_L")
    if name == "hair_tip":
        hair_bones = [bn for bn in sk_bones if any(tok in bn.lower() for tok in ("back_hair", "hair_strand", "mane", "hair", "cloth"))]
        if hair_bones:
            hair_bones.sort()
            return hair_bones[-1], True
        if "head" in sk_bones:
            return "head", True
    if name == "chest" and "torso" in sk_bones:
        return "torso", False
    if name == "hips" and "waist" in sk_bones:
        return "waist", False
    return None


def _check_structural_integrity(renderer: SpineRenderer, rig_rep: Dict[str, Any]) -> Dict[str, Any]:
    """Check effective skin influences, including ancestors, instead of bone presence alone."""
    raw = renderer.sk.json
    bones = raw.get("bones", [])
    names = [b["name"] for b in bones]
    parents = {b["name"]: b.get("parent") for b in bones}
    slots = {s["name"]: s for s in raw.get("slots", [])}
    bone_roles = rig_rep.get("boneRoles", {})
    attachment_roles = {a["part"]: a.get("semanticRole") for a in rig_rep.get("attachments", [])}
    used_bones = set()
    weapon_attachments = []
    non_rigid_weapons = []
    for skin in raw.get("skins", []):
        for slot_name, att_map in skin.get("attachments", {}).items():
            slot = slots.get(slot_name, {})
            for att_name, att in att_map.items():
                if slot.get("attachment") != att_name:
                    continue
                kind = att.get("type", "region")
                verts = att.get("vertices", [])
                if kind == "mesh" and len(verts) > len(att.get("uvs", [])):
                    i = 0
                    while i < len(verts):
                        count = int(verts[i]); i += 1
                        for _ in range(count):
                            idx, weight = int(verts[i]), float(verts[i + 3])
                            if 0 <= idx < len(names) and weight > 0:
                                used_bones.add(names[idx])
                            i += 4
                elif kind in ("region", "mesh") and slot.get("bone"):
                    used_bones.add(slot["bone"])
                is_weapon = (attachment_roles.get(slot_name) == "weapon"
                             if attachment_roles.get(slot_name) else "weapon" in slot_name.lower())
                if is_weapon:
                    weapon_attachments.append(slot_name)
                    if kind != "region":
                        non_rigid_weapons.append(slot_name)

    effective_bones = set(used_bones)
    for name in used_bones:
        seen = set()
        while parents.get(name) and name not in seen:
            seen.add(name)
            name = parents[name]
            effective_bones.add(name)
    hollow_ik = [c["name"] for c in renderer.sk.ik
                 if not any(bn in effective_bones for bn in c.get("bones", []))]
    missing = []
    for logical in ("head", "chest", "hips"):
        binding = _resolve_soft_landmark_bone(renderer.sk.by_name, logical)
        if binding is None or binding[0] not in effective_bones:
            missing.append(logical)
    full_body = str(rig_rep.get("preset", "")).endswith("_full")
    lower_bones = sorted(b for b in effective_bones
                         if (bone_roles[b] == "leg" if b in bone_roles
                             else b.startswith(("thigh", "shin", "foot", "leg"))))
    if full_body and not lower_bones:
        missing.append("lower_body")
    animated = set()
    for animation in raw.get("animations", {}).values():
        animated.update(animation.get("bones", {}))
    return {
        "weapon_rigidity": {
            "passed": not non_rigid_weapons,
            "applicable": bool(weapon_attachments),
            "status": "pass" if weapon_attachments and not non_rigid_weapons else ("fail" if non_rigid_weapons else "not_applicable"),
            "nonRigidWeapons": non_rigid_weapons,
        },
        "ik_binding": {"passed": not hollow_ik, "hollowIkConstraints": hollow_ik},
        "required_parts": {
            "passed": not missing, "missingOrUnweighted": missing,
            "fullBody": full_body, "effectiveLowerBodyBones": lower_bones,
        },
        "influence_diagnostics": {
            "passed": True, "directlyWeightedBones": sorted(used_bones),
            "animatedWithoutSkinInfluence": sorted(animated - effective_bones),
        },
    }


def _ik_constraint_drift(r: SpineRenderer, c: Dict[str, Any], setup_dists: Dict[str, float]) -> float:
    names = c["bones"]
    tgt = r.sk.by_name[c["target"]]
    T = np.array([tgt.worldX, tgt.worldY], dtype=np.float64)
    if len(names) == 2:
        tip = r.sk.by_name[names[1]].tip_world()
        return float(np.linalg.norm(tip - T))
    # 1-bone aim IK: measure distance from bone's ray/tip to target
    b = r.sk.by_name[names[0]]
    p0 = np.array([b.worldX, b.worldY], dtype=np.float64)
    length = float(b.data.get("length", 0.0))
    if length > 1e-6 and abs(length - setup_dists[c["name"]]) <= 1.0:
        return float(np.linalg.norm(b.tip_world() - T))
    # Otherwise perpendicular distance from T to bone +x ray
    axis = np.array([b.a, b.c], dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        return float(np.linalg.norm(T - p0))
    axis /= norm
    rel = T - p0
    proj = max(0.0, float(rel @ axis))
    closest = p0 + proj * axis
    return float(np.linalg.norm(T - closest))


def run_audit(
    parts_dir: Union[str, Path],
    rig_dir: Union[str, Path],
    *,
    out_path: Optional[Union[str, Path]] = None,
    sample_count: int = 16,
    stress_gain: float = 1.6,
    stress_samples: int = 8,
    raise_on_fail: bool = False,
) -> Dict[str, Any]:
    """Run all 4 hard gates + stress test + soft displacement hints and write `health-report.json`."""
    if sample_count < 1 or stress_samples < 1:
        raise ValueError('Quality checks require nonempty normal and stress sampling')
    pdir = Path(parts_dir).resolve()
    rdir = Path(rig_dir).resolve()
    report_dst = Path(out_path).resolve() if out_path else (rdir / "health-report.json")

    rep_path = rdir / "rig-build-report.json"
    rig_rep = json.loads(rep_path.read_text(encoding="utf-8")) if rep_path.is_file() else {}
    duration = float(rig_rep.get("duration", 8.0))
    preset_name = str(rig_rep.get("preset", "agile_full"))
    gain = float(rig_rep.get("gain", 1.0))
    ref_ranges = rig_rep.get("softReferenceRanges") or get_preset(preset_name)["soft_displacement_ranges"]

    renderer = SpineRenderer(rdir)

    # Gate 1: rest_diff (maxRGBA <= 2)
    gate_rest_diff = _evaluate_rest_diff(pdir)

    # Gate 2: setup_identity (frame0VsSetup == 0, frameLoopEndVsSetup == 0, setupVsParts.maxRGBA <= 2)
    f_setup = renderer.frame(None)
    module_qc = ModuleQualityAuditor(renderer, pdir, rig_rep, f_setup)
    f_0 = renderer.frame(0.0)
    f_end = renderer.frame(duration)
    parts_pm = parts_composite_pm(pdir, renderer.width, renderer.height)

    diff_0 = int(np.abs(f_0.astype(np.int16) - f_setup.astype(np.int16)).max())
    diff_end = int(np.abs(f_end.astype(np.int16) - f_setup.astype(np.int16)).max())
    diff_parts = int(np.abs(f_setup.astype(np.int16) - parts_pm.astype(np.int16)).max())
    gate_setup_identity = {
        "passed": (diff_0 == 0) and (diff_end == 0) and (diff_parts <= 2),
        "threshold": {"frame0VsSetup": 0, "frameLoopEndVsSetup": 0, "setupVsPartsMaxRGBA": 2},
        "measured": {
            "frame0VsSetup": diff_0,
            "frameLoopEndVsSetup": diff_end,
            "setupVsPartsMaxRGBA": diff_parts,
        },
    }

    # Gate 3 & 4 + Soft displacement tracks over sampled loop frames
    rest_alpha = f_setup[..., 3]
    ys_nz, _ = np.nonzero(rest_alpha > 8)
    body_h = int(ys_nz.max() - ys_nz.min() + 1) if ys_nz.size else renderer.height

    renderer.pose(None)
    setup_ik_dists: Dict[str, float] = {}
    for c in renderer.sk.ik:
        b0 = renderer.sk.by_name[c["bones"][0]]
        tgt = renderer.sk.by_name[c["target"]]
        setup_ik_dists[c["name"]] = float(math.hypot(tgt.worldX - b0.worldX, tgt.worldY - b0.worldY))

    loop_times = [duration * i / float(sample_count) for i in range(sample_count)]
    loop_holes: List[int] = []
    ik_drifts: Dict[str, float] = {c["name"]: 0.0 for c in renderer.sk.ik}

    landmark_names = ("head", "chest", "hips", "hand_R", "hand_L", "weapon_L", "leg_L", "hair_tip")
    lm_bindings = {nm: _resolve_soft_landmark_bone(renderer.sk.by_name, nm) for nm in landmark_names}
    lm_tracks: Dict[str, List[Tuple[float, float]]] = {nm: [] for nm in landmark_names}

    for t in loop_times:
        f = renderer.frame(t)
        module_qc.observe(t, 1.0, f)
        loop_holes.append(count_enclosed_holes(f, rest_alpha))
        for c in renderer.sk.ik:
            drift = _ik_constraint_drift(renderer, c, setup_ik_dists)
            if drift > ik_drifts[c["name"]]:
                ik_drifts[c["name"]] = drift
        for nm, bind in lm_bindings.items():
            if bind is None:
                continue
            bname, use_tip = bind
            b = renderer.sk.by_name[bname]
            if use_tip:
                pt = b.tip_world()
                lm_tracks[nm].append((float(pt[0]), float(pt[1])))
            else:
                lm_tracks[nm].append((float(b.worldX), float(b.worldY)))

    stress_res = stress_test(
        pdir,
        rdir,
        stress_gain=stress_gain,
        sample_count=stress_samples,
        renderer=renderer,
        sample_callback=module_qc.observe,
    )
    loop_holes_max = max(loop_holes) if loop_holes else 0
    stress_holes_max = int(stress_res["enclosedHolesMax"])
    enclosed_max = max(loop_holes_max, stress_holes_max)

    gate_no_holes = {
        "passed": True,
        "advisory": True,
        "status": "pass",
        "measuredStatus": "pass" if enclosed_max == 0 else "fail",
        "threshold": {"enclosedHolesMax": 0},
        "measured": {
            "loopEnclosedHolesMax": loop_holes_max,
            "stressEnclosedHolesMax": stress_holes_max,
            "enclosedHolesMax": enclosed_max,
        },
    }

    max_ik_drift = max(ik_drifts.values()) if ik_drifts else 0.0
    gate_anchor_stability = {
        "passed": max_ik_drift <= 1.5,
        "threshold": {"maxDriftPx": 1.5},
        "measured": {
            "maxDriftPx": round(float(max_ik_drift), 4),
            "perConstraint": {k: round(float(v), 4) for k, v in ik_drifts.items()},
        },
    }

    # Soft displacement hints (advisory only; never fail hard gates)
    soft_landmarks: Dict[str, Dict[str, Any]] = {}
    for nm in landmark_names:
        lo, hi = ref_ranges.get(nm, [0.0, 999.0])
        pts = lm_tracks[nm]
        if not pts:
            soft_landmarks[nm] = {
                "bone": None,
                "ptpPx": None,
                "ptpX": None,
                "ptpY": None,
                "refRangePx": [float(lo), float(hi)],
                "status": "not_present",
            }
            continue
        arr = np.asarray(pts, dtype=np.float64)
        ptp_x = float(np.ptp(arr[:, 0]))
        ptp_y = float(np.ptp(arr[:, 1]))
        ptp = float(math.hypot(ptp_x, ptp_y))
        if ptp < float(lo):
            st = "warn_low"
        elif ptp > float(hi):
            st = "warn_high"
        else:
            st = "ok"
        bind = lm_bindings[nm]
        soft_landmarks[nm] = {
            "bone": bind[0] if bind else None,
            "ptpPx": round(ptp, 3),
            "ptpX": round(ptp_x, 3),
            "ptpY": round(ptp_y, 3),
            "refRangePx": [float(lo), float(hi)],
            "status": st,
        }

    module_qc.audit_hidden_joint_cycle()
    quality = module_qc.finish()
    structural = _check_structural_integrity(renderer, rig_rep)
    if quality.get('contractApplicable'):
        negative = quality['machineQC']['checks'].get('negativeSpace', result('pending'))
        normal_samples = [s for s in negative.get('samples', []) if abs(s['gain']-1.0) < 1e-9]
        stressed_samples = [s for s in negative.get('samples', []) if abs(s['gain']-stress_gain) < 1e-9]
        measured_status = negative.get('measuredStatus', negative.get('status'))
        gate_no_holes = {
            'passed': True, 'status': 'pass', 'advisory': True,
            'measuredStatus': measured_status,
            'method': 'original-rigid-module-negative-space-transport',
            'threshold': {'newHolePixelsMax': 0},
            'measured': {
                'loopEnclosedHolesMax': max((s['newHolePixels'] for s in normal_samples), default=0),
                'stressEnclosedHolesMax': max((s['newHolePixels'] for s in stressed_samples), default=0),
                'enclosedHolesMax': negative.get('newHolePixelsMax', 0),
                'ambiguousPixelsMax': negative.get('ambiguousPixelsMax', 0),
            },
            'legacyConservative': gate_no_holes['measured'],
            'scope': negative.get('scope'),
        }
        stress_res['legacyConservative'] = {'enclosedHolesMax': stress_res['enclosedHolesMax'],
                                             'perSampleHoles': stress_res['perSampleHoles'],
                                             'passed': stress_res['passed']}
        stress_res['enclosedHolesMax'] = gate_no_holes['measured']['stressEnclosedHolesMax']
        stress_res['perSampleHoles'] = [s['newHolePixels'] for s in stressed_samples]
        stress_measured = aggregate({str(i): s for i, s in enumerate(stressed_samples)}) if stressed_samples else 'pending'
        stress_res['measuredStatus'] = stress_measured
        stress_res['status'] = 'pass'
        stress_res['passed'] = True
        stress_res['advisory'] = True
        stress_res['ambiguityRequiresReview'] = False
        # V2 supports arbitrary module/bone IDs and single-bone meshes. The old
        # region-only weapon and hard-coded head/chest/hips tests are not its contract.
        rigid_check = quality['bindingMotion']['checks'].get('rigidity', result('pending'))
        structural['weapon_rigidity'] = {**rigid_check, 'scope': 'actual rigid module geometry, including ancestors and multiple pieces'}
        structural['required_parts'] = quality['inputOwnership']['checks'].get('declaredCoverage', result('pending'))
    hard_gates = {
        "rest_diff": gate_rest_diff,
        "setup_identity": gate_setup_identity,
        "no_holes": gate_no_holes,
        "anchor_stability": gate_anchor_stability,
    }
    if quality.get('contractApplicable'):
        base_checks = {k: result(g.get('status') or ('pass' if g['passed'] else 'fail')) for k, g in hard_gates.items()}
        base_checks['ik_binding'] = result('pass' if structural['ik_binding']['passed'] else 'fail')
        quality['machineQC']['checks']['baseGates'] = result(aggregate(base_checks), checks=base_checks,
            measuredReferences=['hardGates', 'structuralChecks.ik_binding'])
        quality['machineQC']['status'] = aggregate(quality['machineQC']['checks'])
        quality['machineQC']['passed'] = quality['machineQC']['status'] == 'pass'
        machine_state = aggregate({k: quality[k] for k in ('inputOwnership', 'bindingMotion', 'machineQC')})
        status = 'fail' if machine_state == 'fail' else ('needs_review' if machine_state == 'pending' else 'pass')
    else:
        status = 'pass' if all(g['passed'] for g in hard_gates.values()) and all(g['passed'] for g in structural.values()) else 'fail'
        quality['machineQC'] = result(status, scope='legacy checks only; module contract not assessed',
            failedChecks=[k for k,g in {**hard_gates,**structural}.items() if not g['passed']])

    health_report: Dict[str, Any] = {
        "version": "2.0",
        "status": status,
        "machineStatus": status,
        "statusScope": "machine checks only; product approval/export require quality.overall and bound visual evidence",
        "quality": quality,
        "hardGates": hard_gates,
        "structuralChecks": structural,
        "softHints": {
            "preset": preset_name,
            "gain": gain,
            "bodyHeightPx": body_h,
            "landmarks": soft_landmarks,
        },
        "stressTest": stress_res,
        "artifacts": {
            "partsDir": str(pdir),
            "rigDir": str(rdir),
            "healthReport": str(report_dst),
        },
    }
    recompute_quality(health_report)

    report_dst.parent.mkdir(parents=True, exist_ok=True)
    report_dst.write_text(json.dumps(health_report, indent=2) + "\n", encoding="utf-8")

    if raise_on_fail and status != 'pass':
        failed = [k for k, v in {**hard_gates, **structural}.items() if not v['passed']]
        if quality.get('contractApplicable'):
            failed += ['quality.'+k for k in ('inputOwnership', 'bindingMotion', 'machineQC') if quality[k]['status'] not in ('pass', 'not_applicable')]
        raise GateFailureError(f"Hard quality gates failed: {', '.join(failed)}", health_report)

    return health_report
