---
name: portrait-spine-idle
description: >
  Turn a 2D character portrait into a Spine 4.2.43 idle animation rig with Mivo Canvas matting/inpainting,
  deterministic part splitting, 4 hard quality gates, real-time offline preview, and natural-language minimal-layer reruns.
  Use in Cindy + Mivo Canvas when rigging a 2D portrait to Spine idle, tuning idle motion/IK/weights,
  re-splitting parts, or exporting Spine 4.2 packages and transparent web idle videos.
---

# Portrait Spine Idle Workflow

Convert a 2D character portrait into a production-ready **Spine 4.2.43** seamless 8-second idle rig (`<character>.spine.json` + 4px-padded `images/*.png`), verified by 4 hard quality gates and an offline WebGL2/Canvas2D preview player.

## 5-Stage Pipeline

1. **Visual Diagnosis & Spec Planning**
   - Run `preprocess_portrait` (`src/pipeline/preprocess.py`): converts to RGBA and automatically downscales proportionally via LANCZOS if long edge `> 2400px` (emitting an explicit `notice`).
   - Inspect the portrait silhouette, framing, and contact seams (hair over shoulders, arm resting on hip, weapon grip, cape/tail).
   - Select one of the **3 core presets** (`src/pipeline/presets.py`):
     - `agile_full`: full-body nimble character (8s sway, 4s breath rise, lively hair/ear/tail waves).
     - `heavy_weapon_full`: full-body heavy-weapon character (grounded hip/weapon sway, slower counter-rotation, rich mane/sash waves).
     - `bust_half`: half-body or bust portrait (anchored lower torso/hips, refined chest breath, gentle head/hair drift).
   - Draft `parts_spec` (back-to-front `drawOrder`, `own` polygons, `depth`, `guides`, optional `centerline`) and `rig_spec` (`bones`, `chains`, 1-bone / 2-bone `ik`, `skin` recipes including `contact_dist`, and optional `motion` overrides).

2. **Mivo Canvas Auto-Matting & Occluded Inpaint + `split_parts`**
   - If the input portrait has a background, run Mivo Canvas `auto_matting` (`src/mivo/adapter.py`).
   - For large occluded areas that exceed local PatchMatch/Laplace completion, run Mivo Canvas `inpaint_occluded` with targeted masks and prompts.
   - Run `split_parts` (`src/pipeline/split_parts.py`): watershed seam refinement (`band`), island cleanup, deterministic multi-scale PatchMatch / multigrid Laplace hidden fill (`guides`), and per-part SHA-256 hash caching in `parts/work/`.
   - Outputs: `parts/parts.json`, `parts/parts/<name>.png`, `parts/rest_composite.png`, `parts/rest_diff.json`, `parts/review/part_map.png`.

3. **`build_rig` with 3 Core Presets + `gain`**
   - Run `build_rig` (`src/pipeline/build_rig.py`) with `--preset <agile_full|heavy_weapon_full|bust_half>` and global amplitude `--gain` (default `1.0`).
   - Builds `region` and weighted `mesh` attachments (`chain`, `linear_y`, `linear_x`, `limb`, `radial`, `contact_dist`), 1-bone / 2-bone IK (auto-computing `bendPositive` when omitted), and zero-at-start periodic cubic-bezier timelines (`t = 0s` and `t = 8.0s` identical to Setup Pose).
   - Outputs: `rig/<character>.spine.json`, `rig/images/*.png` (4px-padded), `rig/rig-build-report.json`.

4. **4 Hard Gates + Soft Hints + Real-Time Player Review**
   - Run `run_audit` (`src/gates/audit.py`, `raise_on_fail=True`) to verify all **4 hard gates**:
     - `rest_diff`: `maxRGBA <= 2` (`rest_composite.png` vs normalized source).
     - `setup_identity`: `frame0VsSetup == 0`, `frameLoopEndVsSetup == 0`, `setupVsPartsMaxRGBA <= 2`.
     - `no_holes`: `enclosedHolesMax == 0` across sampled loop frames **and** `1.6x` stress-test frames.
     - `anchor_stability`: `maxDriftPx <= 1.5` across all IK constraints.
   - Compute advisory **soft displacement hints** (`head`, `chest`, `hips`, `hair_tip`) against preset reference ranges (`ok` / `warn_low` / `warn_high`); soft hints never fail hard gates.
   - Generate `preview.html` (backed by `src/player/`) for real-time interactive review of playback, timeline scrubbing, bone/mesh overlay, and gate badges.

5. **Default Export & On-Demand Transparent Web Video Export**
   - **Default export** (`export_default_package` in `src/export/exporter.py`): packages `export/<character>-spine42.zip` (`<character>.spine.json`, `images/*.png`, `rig-build-report.json`, `health-report.json`) plus dark/light preview MP4s when `ffmpeg` is available.
   - **On-demand transparent web video export** (`--export-web` / `export_transparent_web`): run **only** when the user explicitly requests web video assets. Produces `idle-packed-h264.mp4`, optional `idle-packed-av1.mp4`, `idle-alpha-vp9.webm` (`yuva420p`), and frame-0 `poster.webp`.

## Balanced Intervention Flow (折中介入流)

- **Default (One-Click to Preview)**: Run `scripts/run_e2e.py` end-to-end straight to `preview.html` and `health-report.json` without interrupting for routine confirmations.
- **Auto-Pause on Anomaly (Strict Fail-Stop)**: If any stage fails—Mivo matting/inpaint error, missing/empty part, excessive islands, stress-test holes, or any hard-gate failure—`StageRunner` halts immediately, writes `stage-error.json` with `failedStage`, `errorCode`, `recoveryActions` (`retry` / `discard`), and **preserves all prior intermediates**. Present the anomaly and `review/part_map.png` / `health-report.json` to the user at part/bone confirmation before retrying.
- **Manual Pause Switch**: Pass `--pause-after-split` when the user wants to inspect `review/part_map.png` and `rest_composite.png` before running `build_rig`.

## Minimal Rerun Layer Matrix (自然语言局部重跑分层契约)

When the user asks for an adjustment in natural language, rerun **only the lowest necessary layer**:

| Layer | Trigger Examples | Edit Target | Rerun Scope (`scripts/run_e2e.py`) | Skipped Stages |
| --- | --- | --- | --- | --- |
| **Layer A: Motion** | "右边长发幅度加大" / "呼吸更柔和" / "整体动作放大到 1.2x" | `rig_spec.motion`, `gain`, or `preset` | `--partial-rerun motion` (`build_rig` -> `audit` -> refresh `preview.html` & export) | `preprocess`, Mivo, `split_parts` |
| **Layer B: Rig / IK / Weights** | "左手固定别动" / "膝盖反折" / "手臂和身体交界处滑动" | `rig_spec.bones`, `ik`, `skin` (`contact_dist`, `bendPositive`, etc.) | `--partial-rerun rig` (`build_rig` -> `audit` -> refresh `preview.html` & export) | `preprocess`, Mivo, `split_parts` |
| **Layer C: Part Split / Inpaint** | "把右侧披风单独拆成软体链" / "脖子后面露洞补深一点" | Affected entries in `parts_spec` (`own`, `depth`, `guides`, `centerline`) | `--partial-rerun split` (incremental `split_parts` with SHA-256 cache -> `build_rig` -> `audit` -> refresh `preview.html` & export) | `preprocess`, unaffected part fills (cache hits) |

## Entry Point & Reference

- CLI orchestrator: `python3 scripts/run_e2e.py --image <portrait> --parts-spec <parts.json> --rig-spec <rig.json> --out-dir <dir>`
- For the full decision table, preset reference ranges, and hard-gate troubleshooting, read `references/rerun-and-troubleshooting.md`.
