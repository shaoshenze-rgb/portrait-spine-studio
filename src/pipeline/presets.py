"""Core idle animation presets calibrated from Erika v2 and Rain rigs.

Each harmonic term is stored as (amplitude, harmonic_k, phase_rad):
  f(t) = gain * sum(amp * (sin(2*pi*k*t/duration - phase) - sin(-phase)))
so every channel is guaranteed 0 at t = 0 and t = duration (zero-at-start seamless loop).
"""
from __future__ import annotations

import copy
import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

HALF_PI = math.pi / 2.0
DEFAULT_DURATION = 8.0
DEFAULT_FPS = 30
DEFAULT_KEY_STEP = 0.125


def _p2p_terms(terms: List[Tuple[int, float, float]], duration: float = DEFAULT_DURATION) -> List[Tuple[float, int, float]]:
    """Convert (harmonic_k, peak_to_peak, lag_s) tuples to (amp, k, phase_rad)."""
    return [(p2p * 0.5, k, 2.0 * math.pi * k * lag_s / duration) for k, p2p, lag_s in terms]


PRESETS: Dict[str, Dict[str, Any]] = {
    "agile_full": {
        "name": "agile_full",
        "description": "Nimble full-body idle calibrated from Erika v2: 8s sway, 4s breath rise, lively hair/ear/tail follow-through.",
        "duration": DEFAULT_DURATION,
        "fps": DEFAULT_FPS,
        "key_step": DEFAULT_KEY_STEP,
        "bones": {
            "hips": {
                "tx": _p2p_terms([(1, 10.0, 0.20), (2, 1.8, 0.15)]),
                "ty": _p2p_terms([(2, 4.4, 0.15)]),
                "rotate": _p2p_terms([(1, 0.9, 0.20)]),
            },
            "spine": {
                "rotate": [(0.60, 1, 0.0)],
            },
            "chest": {
                "rotate": [(0.48, 1, 0.25)],
                "ty": [(2.4, 2, HALF_PI)],
            },
            "ribcage": {
                "sx": [(0.0022, 2, HALF_PI + 0.10)],
                "sy": [(0.0045, 2, HALF_PI + 0.10)],
            },
            "neck": {
                "rotate": [(0.30, 1, 0.50)],
            },
            "head": {
                "rotate": [(0.80, 1, 0.75), (0.22, 2, HALF_PI + 0.60)],
            },
            "ear_L": {
                "rotate": [(0.60, 1, 1.30), (0.20, 2, HALF_PI + 0.90)],
                "twitch": (6.1, 6.0),
            },
            "ear_R": {
                "rotate": [(0.60, 1, 1.10), (0.20, 2, HALF_PI + 0.80)],
                "twitch": (2.9, 8.0),
            },
            "bangs": {
                "rotate": [(0.40, 1, 1.10), (0.12, 2, HALF_PI + 0.90)],
            },
            "shoulder_R": {
                "rotate": _p2p_terms([(1, -2.6, 0.45), (2, 1.4, 0.40)]),
            },
            "elbow_R": {
                "rotate": _p2p_terms([(1, -3.0, 0.75), (2, 1.6, 0.70)]),
            },
            "hand_R": {
                "rotate": _p2p_terms([(1, 3.6, 1.00), (2, 1.4, 0.95)]),
            },
            "weapon_R": {
                "rotate": _p2p_terms([(1, 2.2, 1.25), (2, 0.9, 1.15)]),
            },
            "shoulder_L": {
                "rotate": _p2p_terms([(1, 2.4, 0.55), (2, -1.3, 0.45)]),
                "ty": [(-1.70, 2, HALF_PI)],
            },
            "elbow_L": {
                "rotate": _p2p_terms([(1, 2.8, 0.85), (2, -1.5, 0.75)]),
            },
            "hand_L": {
                "rotate": _p2p_terms([(1, -3.2, 1.10), (2, 1.3, 1.00)]),
            },
            "weapon_L": {
                "rotate": _p2p_terms([(1, -2.0, 1.35), (2, 0.8, 1.20)]),
            },
            "thigh_L": {
                "rotate": _p2p_terms([(1, 2.8, 0.30), (2, 1.2, 0.25)]),
            },
            "shin_L": {
                "rotate": _p2p_terms([(1, -3.4, 0.65), (2, 1.5, 0.55)]),
            },
            "foot_L": {
                "rotate": _p2p_terms([(1, 3.0, 0.95), (2, 1.2, 0.85)]),
            },
            "thigh_R": {
                "rotate": _p2p_terms([(1, -2.6, 0.40), (2, 1.1, 0.30)]),
            },
            "shin_R": {
                "rotate": _p2p_terms([(1, 3.2, 0.75), (2, -1.4, 0.60)]),
            },
            "foot_R": {
                "rotate": _p2p_terms([(1, -2.8, 1.05), (2, 1.1, 0.90)]),
            },
            "pouch": {
                "rotate": [(0.30, 1, 1.00), (0.10, 2, HALF_PI + 0.80)],
            },
            "skirt_L": {
                "rotate": [(0.45, 1, 0.85), (0.15, 2, HALF_PI + 0.70)],
            },
            "skirt_R": {
                "rotate": [(0.40, 1, 0.95), (0.15, 2, HALF_PI + 0.80)],
            },
        },
        "chain_waves": {
            "back_hair": {"root_p2p": 1.6, "growth": 1.38, "base_lag_s": 0.45, "lag_per_link_s": 0.32, "k": [(1, 1.0), (2, 0.35)]},
            "hair_strand": {"root_p2p": 1.8, "growth": 1.35, "base_lag_s": 0.40, "lag_per_link_s": 0.30, "k": [(1, 1.0), (2, 0.30)]},
            "cloth": {"root_p2p": 1.4, "growth": 1.32, "base_lag_s": 0.35, "lag_per_link_s": 0.30, "k": [(1, 1.0), (2, 0.40)]},
            "tail": {"root_p2p": 2.0, "growth": 1.35, "base_lag_s": 0.50, "lag_per_link_s": 0.35, "k": [(1, 1.0), (2, 0.30)]},
            "default": {"root_p2p": 1.6, "growth": 1.30, "base_lag_s": 0.40, "lag_per_link_s": 0.30, "k": [(1, 1.0), (2, 0.35)]},
        },
        "soft_displacement_ranges": {
            "head": [4.0, 28.0],
            "chest": [3.0, 20.0],
            "hips": [2.0, 18.0],
            "hair_tip": [8.0, 65.0],
        },
    },
    "heavy_weapon_full": {
        "name": "heavy_weapon_full",
        "description": "Heavy-weapon full-body idle calibrated from Rain: grounded hip/weapon sway, slower counter-rotation, rich mane/sash waves.",
        "duration": DEFAULT_DURATION,
        "fps": DEFAULT_FPS,
        "key_step": DEFAULT_KEY_STEP,
        "bones": {
            "hips": {
                "tx": _p2p_terms([(1, 22.0, 0.25), (2, 2.4, 0.15)]),
                "ty": _p2p_terms([(2, 8.8, 0.20)]),
                "rotate": _p2p_terms([(1, 1.2, 0.25)]),
            },
            "spine": {
                "rotate": _p2p_terms([(1, -2.4, -0.20)]),
            },
            "chest": {
                "rotate": _p2p_terms([(1, -1.0, -0.20)]),
                "ty": _p2p_terms([(2, 9.6, 0.0)]),
            },
            "ribcage": {
                "sx": _p2p_terms([(2, 0.008, 0.15)]),
                "sy": _p2p_terms([(2, 0.014, 0.15)]),
            },
            "neck": {
                "rotate": _p2p_terms([(1, 3.2, -0.80)]),
            },
            "head": {
                "rotate": _p2p_terms([(1, 0.8, -0.80), (2, -0.8, 0.30)]),
            },
            "weapon_ctrl": {
                "tx": _p2p_terms([(1, 32.0, 0.35)]),
                "ty": _p2p_terms([(2, 13.0, 0.35)]),
                "rotate": _p2p_terms([(1, -2.2, 0.45), (2, -0.7, 0.50)]),
            },
            "shoulder_R": {
                "rotate": _p2p_terms([(1, -2.0, 0.45), (2, 1.0, 0.40)]),
            },
            "elbow_R": {
                "rotate": _p2p_terms([(1, -2.4, 0.80), (2, 1.2, 0.70)]),
            },
            "hand_R": {
                "rotate": _p2p_terms([(1, 2.6, 1.05), (2, 1.0, 0.95)]),
            },
            "weapon_R": {
                "rotate": _p2p_terms([(1, 1.8, 1.30), (2, 0.7, 1.20)]),
            },
            "shoulder_L": {
                "rotate": _p2p_terms([(1, 1.9, 0.55), (2, -1.0, 0.45)]),
            },
            "elbow_L": {
                "rotate": _p2p_terms([(1, 2.2, 0.85), (2, -1.1, 0.75)]),
            },
            "hand_L": {
                "rotate": _p2p_terms([(1, -2.4, 1.15), (2, 1.0, 1.00)]),
            },
            "weapon_L": {
                "rotate": _p2p_terms([(1, -1.6, 1.35), (2, 0.6, 1.25)]),
            },
            "thigh_L": {
                "rotate": _p2p_terms([(1, 2.0, 0.30), (2, 0.9, 0.25)]),
            },
            "shin_L": {
                "rotate": _p2p_terms([(1, -2.4, 0.65), (2, 1.1, 0.55)]),
            },
            "foot_L": {
                "rotate": _p2p_terms([(1, 2.2, 0.95), (2, 0.9, 0.85)]),
            },
            "thigh_R": {
                "rotate": _p2p_terms([(1, -1.9, 0.40), (2, 0.8, 0.30)]),
            },
            "shin_R": {
                "rotate": _p2p_terms([(1, 2.2, 0.75), (2, -1.0, 0.60)]),
            },
            "foot_R": {
                "rotate": _p2p_terms([(1, -2.0, 1.05), (2, 0.8, 0.90)]),
            },
            "earring": {
                "rotate": _p2p_terms([(1, 7.0, 0.60), (2, 2.8, 0.65)]),
            },
        },
        "chain_waves": {
            "back_hair": {"root_p2p": 3.8, "growth": 1.45, "base_lag_s": 0.35, "lag_per_link_s": 0.35, "k": [(1, 1.0), (2, 0.48)]},
            "hair_strand": {"root_p2p": 4.2, "growth": 1.45, "base_lag_s": 0.32, "lag_per_link_s": 0.35, "k": [(1, 1.0), (2, 0.45)]},
            "cloth": {"root_p2p": 2.4, "growth": 1.32, "base_lag_s": 0.38, "lag_per_link_s": 0.35, "k": [(1, 1.0), (2, 0.55)]},
            "tail": {"root_p2p": 2.8, "growth": 1.35, "base_lag_s": 0.40, "lag_per_link_s": 0.35, "k": [(1, 1.0), (2, 0.50)]},
            "default": {"root_p2p": 2.6, "growth": 1.35, "base_lag_s": 0.35, "lag_per_link_s": 0.35, "k": [(1, 1.0), (2, 0.50)]},
        },
        "soft_displacement_ranges": {
            "head": [6.0, 38.0],
            "chest": [6.0, 34.0],
            "hips": [8.0, 36.0],
            "hair_tip": [12.0, 85.0],
        },
    },
    "bust_half": {
        "name": "bust_half",
        "description": "Half-body / bust idle: anchored lower torso/hips, refined chest breath expansion, gentle head and hair drift.",
        "duration": DEFAULT_DURATION,
        "fps": DEFAULT_FPS,
        "key_step": DEFAULT_KEY_STEP,
        "bones": {
            "hips": {
                "tx": _p2p_terms([(1, 1.6, 0.15)]),
                "ty": _p2p_terms([(2, 1.0, 0.10)]),
            },
            "spine": {
                "rotate": [(0.35, 1, 0.0)],
            },
            "chest": {
                "rotate": [(0.30, 1, 0.25)],
                "ty": [(2.2, 2, HALF_PI)],
            },
            "ribcage": {
                "sx": [(0.0028, 2, HALF_PI + 0.10)],
                "sy": [(0.0052, 2, HALF_PI + 0.10)],
            },
            "neck": {
                "rotate": [(0.24, 1, 0.45)],
            },
            "head": {
                "rotate": [(0.65, 1, 0.70), (0.18, 2, HALF_PI + 0.55)],
            },
            "bangs": {
                "rotate": [(0.35, 1, 1.05), (0.10, 2, HALF_PI + 0.85)],
            },
            "shoulder_L": {
                "rotate": [(0.22, 1, 0.55), (0.28, 2, HALF_PI + 0.45)],
            },
            "shoulder_R": {
                "rotate": [(-0.22, 1, 0.55), (-0.28, 2, HALF_PI + 0.45)],
            },
        },
        "chain_waves": {
            "back_hair": {"root_p2p": 1.3, "growth": 1.32, "base_lag_s": 0.45, "lag_per_link_s": 0.30, "k": [(1, 1.0), (2, 0.30)]},
            "hair_strand": {"root_p2p": 1.5, "growth": 1.30, "base_lag_s": 0.40, "lag_per_link_s": 0.28, "k": [(1, 1.0), (2, 0.28)]},
            "cloth": {"root_p2p": 1.1, "growth": 1.25, "base_lag_s": 0.35, "lag_per_link_s": 0.28, "k": [(1, 1.0), (2, 0.32)]},
            "default": {"root_p2p": 1.3, "growth": 1.28, "base_lag_s": 0.40, "lag_per_link_s": 0.28, "k": [(1, 1.0), (2, 0.30)]},
        },
        "soft_displacement_ranges": {
            "head": [2.5, 22.0],
            "chest": [2.0, 14.0],
            "hips": [0.0, 8.0],
            "hair_tip": [5.0, 48.0],
        },
    },
}


def list_presets() -> List[str]:
    return sorted(PRESETS.keys())


def get_preset(name: str) -> Dict[str, Any]:
    if name not in PRESETS:
        raise KeyError(f"Unknown preset {name!r}; expected one of {list_presets()}")
    return copy.deepcopy(PRESETS[name])


def _classify_chain(chain_name: str) -> str:
    low = chain_name.lower()
    if "back_hair" in low or "mane" in low:
        return "back_hair"
    if "hair" in low or "lock" in low or "bang" in low:
        return "hair_strand"
    if "tail" in low:
        return "tail"
    if any(tok in low for tok in ("cloth", "skirt", "sash", "cape", "ribbon", "tassel", "fur")):
        return "cloth"
    return "default"


ROLE_DEFAULT_MOTION: Dict[str, Dict[str, Any]] = {
    "head": {"rotate": _p2p_terms([(1, 1.6, 0.60), (2, 0.45, 0.55)])},
    "body": {"rotate": _p2p_terms([(1, 1.0, 0.25)])},
    "arm": {"rotate": _p2p_terms([(1, 2.4, 0.55), (2, 1.2, 0.50)])},
    "hand": {"rotate": _p2p_terms([(1, 3.0, 0.95), (2, 1.2, 0.90)])},
    "weapon": {"rotate": _p2p_terms([(1, 1.8, 1.25), (2, 0.7, 1.15)])},
    "leg": {"rotate": _p2p_terms([(1, 2.4, 0.45), (2, 1.0, 0.40)])},
    "soft": {"rotate": _p2p_terms([(1, 3.6, 0.70), (2, 1.4, 0.65)])},
}


def resolve_preset_motion(
    preset_name: str,
    bone_names: Iterable[str],
    chains: Optional[Dict[str, List[str]]] = None,
    overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    duration: Optional[float] = None,
    bone_roles: Optional[Dict[str, str]] = None,
    ik_driven_bones: Optional[Iterable[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Build per-bone harmonic motion table from a preset, chain waves, role fallbacks, and overrides."""
    preset = get_preset(preset_name)
    dur = float(duration or preset["duration"])
    valid_bones = set(bone_names)
    ik_set = set(ik_driven_bones or ())
    motion: Dict[str, Dict[str, Any]] = {}

    for bname, channels in preset["bones"].items():
        if bname in valid_bones and bname not in ik_set:
            motion[bname] = copy.deepcopy(channels)

    if chains:
        waves = preset["chain_waves"]
        for cname, link_bones in chains.items():
            wv = waves.get(cname) or waves.get(_classify_chain(cname)) or waves["default"]
            for i, bn in enumerate(link_bones):
                if bn not in valid_bones or bn in ik_set:
                    continue
                lag = wv["base_lag_s"] + i * wv["lag_per_link_s"]
                p2p = wv["root_p2p"] * (wv["growth"] ** i)
                motion[bn] = {
                    "rotate": [
                        (p2p * factor * 0.5, k, 2.0 * math.pi * k * lag / dur)
                        for k, factor in wv["k"]
                    ]
                }

    # Give user-added custom bones a sensible role-matched rhythm automatically
    if bone_roles:
        for bname, role in bone_roles.items():
            if bname in valid_bones and bname not in motion and bname not in ik_set and role in ROLE_DEFAULT_MOTION:
                motion[bname] = copy.deepcopy(ROLE_DEFAULT_MOTION[role])

    if overrides:
        for bname, ch in overrides.items():
            if bname not in valid_bones:
                continue
            merged = dict(motion.get(bname, {}))
            merged.update(copy.deepcopy(ch))
            motion[bname] = merged

    return motion
