# Minimal Rerun & Troubleshooting Reference

## 1. Minimal-Rerun Decision Table

Always classify a user's natural-language feedback into Layer A, B, or C before re-running. Never discard `parts/` or re-run Mivo matting when only animation or rigging parameters changed.

| Layer | Natural-Language Signals | Files / Fields Modified | Command Flag | Executed Stages | Cache / Reuse Guarantee |
| --- | --- | --- | --- | --- | --- |
| **A (Motion)** | Amplitude, speed/phase feel, softer breathing, stronger hair/tail wave, global `gain`, preset switch | `rig_spec.motion`, `--gain`, `--preset` | `--partial-rerun motion` | `build_rig` -> `audit` -> `export` -> `preview.html` | Reuses existing `preprocessed.png` and `parts/` untouched |
| **B (Rig / IK / Weights)** | Anchor lock ("左手固定别动"), IK bend flip ("膝盖/手肘反折"), seam sliding ("袖口拉扯"), bone pivot tweak | `rig_spec.bones`, `rig_spec.chains`, `rig_spec.ik`, `rig_spec.skin` (`contact_dist`, `cell`, `ranges`) | `--partial-rerun rig` | `build_rig` -> `audit` -> `export` -> `preview.html` | Reuses existing `preprocessed.png` and `parts/` untouched |
| **C (Split / Inpaint)** | Split new part ("把右侧披风拆出来"), fix occlusion tear/hole, refine boundary seam or colour guide | `parts_spec` (`own`, `band`, `depth`, `guides`, `centerline`) + matching `rig_spec` entries | `--partial-rerun split` | `split_parts` -> `build_rig` -> `audit` -> `export` -> `preview.html` | Unaffected parts hit SHA-256 fill cache in `parts/work/fill_<name>_<hash>.npy` |

## 2. Three Core Preset Baselines

All presets generate zero-at-start 8.0s periodic loops (`fps=30`, `key_step=0.125s` by default) scaled by global `gain`:

| Preset | Best For | Key Motion Traits | Soft Displacement Ranges (`px`, advisory) |
| --- | --- | --- | --- |
| `agile_full` | Nimble full-body characters (e.g. Erika) | 8s hip sway (`10px`), 4s breath (`4.4px`), lively ear twitch & hair/tail waves | `head`: 4-28, `chest`: 3-20, `hips`: 2-18, `hair_tip`: 8-65 |
| `heavy_weapon_full` | Grounded heavy-weapon full-body characters (e.g. Rain) | Larger hip & `weapon_ctrl` sway (`22px`/`32px`), counter-rotated spine/neck, deep mane/sash waves | `head`: 6-38, `chest`: 6-34, `hips`: 8-36, `hair_tip`: 12-85 |
| `bust_half` | Half-body / bust portraits without legs | Anchored lower torso/hips (`1.6px`), refined ribcage breath expansion, gentle head & hair drift | `head`: 2.5-22, `chest`: 2-14, `hips`: 0-8, `hair_tip`: 5-48 |

## 3. Troubleshooting the 4 Hard Gates

| Gate | Threshold | Common Root Cause | Targeted Fix (Minimal Layer) |
| --- | --- | --- | --- |
| **`rest_diff`** | `maxRGBA <= 2` | Semi-transparent pixels dropped by `own` polygons, or overlapping feather ramp gaps | **Layer C**: Expand `own` coverage so every `alpha > 0` pixel is owned; verify `drawOrder` back-to-front. |
| **`setup_identity`** | `frame0VsSetup == 0`, `frameLoopEndVsSetup == 0`, `setupVsPartsMaxRGBA <= 2` | Non-zero animation offset at `t=0` / `t=duration`, or custom keyframe breaking periodic zero-at-start formula | **Layer A/B**: Ensure all motion channels use harmonic zero-at-start terms (or explicit `0` at `t=0` and `t=duration`), and re-check IK target setup position equals bone tip at rest. |
| **`no_holes`** | `enclosedHolesMax == 0` (loop + `1.6x` stress) | Foreground part shifts beyond underlying part's `depth` fill, or missing occluded fill under neck/arm/hair | **Layer C** (increase `depth` / add `guides` or Mivo `inpaint_occluded`) or **Layer A** (reduce excessive translation/rotation amplitude). |
| **`anchor_stability`** | `maxDriftPx <= 1.5` | IK target drifts beyond `L1 + L2` reach of 2-bone chain (e.g. hip translation too large for arm length) or wrong `bendPositive` | **Layer B** (adjust IK target parent/position or `bendPositive`) or **Layer A** (reduce parent bone `tx`/`ty` so target stays within reach). |

## 4. Mivo Canvas & Fail-Stop Recovery

- On any failure, `StageRunner` writes `<out_dir>/stage-error.json` with `failedStage`, `errorCode`, `message`, `preservedIntermediates`, and `recoveryActions: ["retry", "discard"]`.
- Never delete prior intermediates automatically; inspect `preservedIntermediates`, fix the failing input or spec, and rerun from the appropriate layer.
