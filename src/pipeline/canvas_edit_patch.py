"""Consume opaque or RGBA Mivo Canvas `canvas_edit_image` occluded-backing inpaints losslessly.

Generative edit models (e.g., GPT Image / Seedream via `canvas_edit_image`) typically return opaque
RGB images (often with solid or fake gray-white checkerboard backdrops) and minor global color drift.
Rather than replacing the whole 4K artwork or rejecting opaque returns outright, this module:

1. Registers the Canvas edit onto the 4K original artwork coordinate frame (SIFT + RANSAC, with
   optional `crop_bbox` support for local detail crops).
2. Strips solid or fake gray-white checkerboard exterior backdrops (or smoothly inpaints translucent
   alpha for semi-transparent FX layers like glows/mist, or uses native alpha when already matted).
3. Matches local RGB tone across the visible boundary ring around the occluded patch.
4. Pastes ONLY the Canvas-inpainted backing pixels strictly inside `occluded_mask` (behind the front
   moving part), keeping 100% of visible 4K original `base_layer_rgba` pixels bit-for-bit lossless.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw


@dataclass
class CanvasEditPatchResult:
    rgba: np.ndarray
    stats: Dict[str, Any]
    comparison: Image.Image


@dataclass
class OccludedEditCropPlan:
    crop_rgba: np.ndarray
    occluded_mask_u8: np.ndarray
    crop_bbox: Tuple[int, int, int, int]
    meta: Dict[str, Any]


def _to_rgba_array(img: Any) -> np.ndarray:
    if isinstance(img, np.ndarray):
        arr = img
    elif isinstance(img, Image.Image):
        arr = np.asarray(img.convert("RGBA"), dtype=np.uint8)
    else:
        with Image.open(img) as im:
            arr = np.asarray(im.convert("RGBA"), dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError(f"Expected HxWx3/4 image array, got {arr.shape}")
    if arr.shape[2] == 3:
        alpha = np.full(arr.shape[:2] + (1,), 255, dtype=np.uint8)
        arr = np.concatenate([arr.astype(np.uint8), alpha], axis=2)
    return np.ascontiguousarray(arr.astype(np.uint8))


def _has_real_transparency(rgba: np.ndarray) -> bool:
    alpha = rgba[..., 3]
    transparent_ratio = float((alpha < 16).mean())
    opaque_ratio = float((alpha >= 200).mean())
    return transparent_ratio >= 0.02 and opaque_ratio >= 0.01


def _dilate_bool(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or not mask.any():
        return mask.copy()
    import cv2

    ksize = int(radius) * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.dilate(mask.astype(np.uint8), kernel) > 0


def _erode_bool(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or not mask.any():
        return mask.copy()
    import cv2

    ksize = int(radius) * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.erode(mask.astype(np.uint8), kernel) > 0


def register_edit_to_source(
    edit_rgba: np.ndarray,
    source_rgba: np.ndarray,
    *,
    crop_bbox: Optional[Tuple[int, int, int, int]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Align an opaque or transparent Canvas edit image onto the full 4K `source_rgba` canvas."""
    import cv2

    sh, sw = source_rgba.shape[:2]
    eh, ew = edit_rgba.shape[:2]

    sx0, sy0, sx1, sy1 = 0, 0, sw, sh
    if crop_bbox is not None:
        cx, cy, cw, ch = (int(v) for v in crop_bbox)
        if cw <= 0 or ch <= 0:
            raise ValueError(f"Invalid crop_bbox: {crop_bbox}")
        mx, my = int(round(cw * 0.15)), int(round(ch * 0.15))
        sx0 = max(0, cx - mx)
        sy0 = max(0, cy - my)
        sx1 = min(sw, cx + cw + mx)
        sy1 = min(sh, cy + ch + my)
    src_roi = source_rgba[sy0:sy1, sx0:sx1]

    def _downscale(arr: np.ndarray, max_edge: float = 1600.0):
        scale = min(1.0, max_edge / float(max(arr.shape[:2])))
        nsz = (max(1, int(round(arr.shape[1] * scale))), max(1, int(round(arr.shape[0] * scale))))
        return cv2.resize(arr, nsz, interpolation=cv2.INTER_AREA), scale

    el, es = _downscale(edit_rgba)
    sl, ss = _downscale(src_roi)

    detector = cv2.SIFT_create(nfeatures=6000)
    e_mask = (el[..., 3] > 32).astype(np.uint8) * 255
    s_mask = (sl[..., 3] > 32).astype(np.uint8) * 255
    ek, ed = detector.detectAndCompute(cv2.cvtColor(el[..., :3], cv2.COLOR_RGB2GRAY), e_mask)
    sk, sd = detector.detectAndCompute(cv2.cvtColor(sl[..., :3], cv2.COLOR_RGB2GRAY), s_mask)

    matrix = None
    metrics: Dict[str, Any] = {}
    if ed is not None and sd is not None and len(ek) >= 12 and len(sk) >= 12:
        pairs = cv2.BFMatcher().knnMatch(ed, sd, k=2)
        matches = [a for pair in pairs if len(pair) == 2 for a, b in [pair] if a.distance < 0.72 * b.distance]
        if len(matches) >= 12:
            pts_e = np.float32([ek[m.queryIdx].pt for m in matches])
            pts_s = np.float32([sk[m.trainIdx].pt for m in matches])
            est, inliers = cv2.estimateAffinePartial2D(
                pts_e,
                pts_s,
                method=cv2.RANSAC,
                ransacReprojThreshold=2.5,
                maxIters=3000,
                confidence=0.997,
            )
            if est is not None and inliers is not None:
                keep = inliers[:, 0].astype(bool)
                count = int(keep.sum())
                ratio = float(count / max(1, len(matches)))
                angle = math.degrees(math.atan2(est[1, 0], est[0, 0]))
                scale = float(math.hypot(est[0, 0], est[1, 0]) * es / ss)
                residual = np.linalg.norm(pts_e @ est[:, :2].T + est[:, 2] - pts_s, axis=1)
                med_res = float(np.median(residual[keep]) / ss) if count > 0 else 999.0
                metrics = {
                    "method": "sift_affine",
                    "matches": len(matches),
                    "inliers": count,
                    "inlier_ratio": round(ratio, 4),
                    "median_residual_px": round(med_res, 3),
                    "rotation_degrees": round(angle, 4),
                    "scale": round(scale, 6),
                }
                if count >= 12 and ratio >= 0.22 and abs(angle) <= 4.0 and 0.05 < scale < 20.0:
                    m = est.copy().astype(np.float64)
                    m[:, :2] *= es / ss
                    m[:, 2] /= ss
                    m[0, 2] += float(sx0)
                    m[1, 2] += float(sy0)
                    matrix = m

    if matrix is None:
        if crop_bbox is not None:
            cx, cy, cw, ch = (float(v) for v in crop_bbox)
            matrix = np.array(
                [[cw / float(ew), 0.0, cx], [0.0, ch / float(eh), cy]],
                dtype=np.float64,
            )
            metrics = {
                "method": "crop_bbox_direct",
                "crop_bbox": [int(v) for v in crop_bbox],
                "scale_x": round(cw / float(ew), 6),
                "scale_y": round(ch / float(eh), 6),
            }
        else:
            raise ValueError(f"Canvas edit SIFT registration failed and no crop_bbox fallback provided: {metrics}")

    has_native = _has_real_transparency(edit_rgba)
    metrics["source_has_native_alpha"] = has_native
    if has_native:
        pm = edit_rgba.astype(np.float32)
        pm[..., :3] *= pm[..., 3:4] / 255.0
        warped = cv2.warpAffine(pm, matrix, (sw, sh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        a = warped[..., 3:4]
        rgb = np.divide(warped[..., :3] * 255.0, a, out=np.zeros_like(warped[..., :3]), where=a > 0)
        out = np.clip(np.rint(np.concatenate([rgb, a], axis=2)), 0, 255).astype(np.uint8)
        out[out[..., 3] == 0] = 0
    else:
        warped = cv2.warpAffine(
            edit_rgba.astype(np.float32),
            matrix,
            (sw, sh),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
        out = np.clip(np.rint(warped), 0, 255).astype(np.uint8)
    return out, metrics


def estimate_edit_foreground_alpha(
    aligned_edit_rgba: np.ndarray,
    base_layer_rgba: np.ndarray,
    original_rgba: np.ndarray,
    occluded_mask: np.ndarray,
    *,
    mode: str = "auto",
    source_has_native_alpha: Optional[bool] = None,
    replace_occluded: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Derive a clean alpha channel for an aligned Canvas edit image inside the occluded zone."""
    import cv2

    h, w = original_rgba.shape[:2]
    if mode == "clip_original":
        valid = aligned_edit_rgba[..., 3] > 0
        alpha = np.where(valid, original_rgba[..., 3], 0).astype(np.uint8)
        return alpha, {"alpha_source": "clip_original"}

    if mode == "inpaint_alpha":
        base_a = base_layer_rgba[..., 3].astype(np.uint8)
        if replace_occluded:
            hole = (occluded_mask & (aligned_edit_rgba[..., 3] > 0)).astype(np.uint8) * 255
        else:
            hole = (occluded_mask & (base_a < 250) & (aligned_edit_rgba[..., 3] > 0)).astype(np.uint8) * 255
        if hole.any():
            inpainted = cv2.inpaint(base_a, hole, inpaintRadius=12, flags=cv2.INPAINT_TELEA)
            blurred = cv2.GaussianBlur(inpainted, (0, 0), sigmaX=4.0)
            alpha = np.where(hole > 0, blurred, base_a).astype(np.uint8)
        else:
            alpha = base_a.copy()
        return alpha, {"alpha_source": "inpaint_alpha", "inpainted_alpha_px": int((hole > 0).sum())}

    if mode != "auto":
        raise ValueError(f"Unsupported alpha_mode: {mode}")

    use_native = _has_real_transparency(aligned_edit_rgba) if source_has_native_alpha is None else bool(source_has_native_alpha)
    if use_native:
        return aligned_edit_rgba[..., 3].copy(), {"alpha_source": "native_rgba"}

    valid_warp = aligned_edit_rgba[..., 3] >= 128
    exterior = (original_rgba[..., 3] == 0) & (~occluded_mask) & valid_warp
    exterior_core = _erode_bool(exterior, 6)
    if int(exterior_core.sum()) < 64:
        exterior_core = exterior

    if int(exterior_core.sum()) < 16:
        alpha = np.where(valid_warp, original_rgba[..., 3], 0).astype(np.uint8)
        return alpha, {"alpha_source": "fallback_clip_original_no_exterior"}

    rgb = aligned_edit_rgba[..., :3].astype(np.float32)
    ext_rgb = rgb[exterior_core]
    ext_chroma = ext_rgb.max(axis=1) - ext_rgb.min(axis=1)
    ext_luma = ext_rgb.mean(axis=1)
    neutral_ratio = float((ext_chroma <= 24.0).mean())
    luma_p10, luma_p90 = float(np.percentile(ext_luma, 10)), float(np.percentile(ext_luma, 90))
    luma_spread = luma_p90 - luma_p10

    full_chroma = rgb.max(axis=2) - rgb.min(axis=2)
    full_luma = rgb.mean(axis=2)

    if neutral_ratio >= 0.85 and luma_spread >= 18.0:
        lo = max(0.0, luma_p10 - 20.0)
        hi = min(255.0, luma_p90 + 20.0)
        bg_candidate = (full_chroma <= 26.0) & (full_luma >= lo) & (full_luma <= hi) & valid_warp
        alpha_source = "exterior_checkerboard_strip"
        bg_meta = {"neutral_ratio": round(neutral_ratio, 4), "luma_band": [round(lo, 1), round(hi, 1)]}
    else:
        med_rgb = np.median(ext_rgb, axis=0)
        dist = np.linalg.norm(rgb - med_rgb[None, None, :], axis=2)
        ext_p95 = float(np.percentile(np.linalg.norm(ext_rgb - med_rgb[None, :], axis=1), 95))
        tol = max(18.0, min(48.0, ext_p95 + 14.0))
        bg_candidate = (dist <= tol) & valid_warp
        alpha_source = "exterior_solid_strip"
        bg_meta = {"bg_median_rgb": [round(float(v), 1) for v in med_rgb], "tol": round(tol, 1)}

    seed = exterior & bg_candidate
    if not seed.any():
        seed = exterior
    _, labels = cv2.connectedComponents((bg_candidate | exterior).astype(np.uint8), connectivity=8)
    seed_labels = np.unique(labels[seed])
    seed_labels = seed_labels[seed_labels > 0]
    connected_bg = np.isin(labels, seed_labels) if len(seed_labels) > 0 else np.zeros((h, w), dtype=bool)

    fg_mask = valid_warp & (~connected_bg)
    fg_u8 = fg_mask.astype(np.uint8) * 255
    soft = cv2.GaussianBlur(fg_u8, (0, 0), sigmaX=1.2)
    fg_core = _erode_bool(fg_mask, 2)
    alpha = np.where(fg_core, 255, np.where(fg_mask, np.maximum(soft, 180), 0)).astype(np.uint8)
    return alpha, {"alpha_source": alpha_source, "stripped_bg_px": int(connected_bg.sum()), **bg_meta}


def _render_patch_comparison(
    base_rgba: np.ndarray,
    aligned_edit_rgba: np.ndarray,
    patched_rgba: np.ndarray,
    patch_mask: np.ndarray,
    max_panel: int = 720,
) -> Image.Image:
    h, w = base_rgba.shape[:2]
    scale = min(1.0, float(max_panel) / float(max(h, w)))
    pw, ph = max(1, int(round(w * scale))), max(1, int(round(h * scale)))

    def _on_gray(rgba: np.ndarray) -> Image.Image:
        a = rgba[..., 3:4].astype(np.float32) / 255.0
        comp = np.clip(rgba[..., :3].astype(np.float32) * a + 128.0 * (1.0 - a), 0, 255).astype(np.uint8)
        return Image.fromarray(comp).resize((pw, ph), Image.Resampling.BILINEAR)

    p1 = _on_gray(base_rgba)
    p2 = _on_gray(aligned_edit_rgba)
    p3 = _on_gray(patched_rgba)

    overlay = patched_rgba.copy()
    a = overlay[..., 3:4].astype(np.float32) / 255.0
    comp4 = np.clip(overlay[..., :3].astype(np.float32) * a + 128.0 * (1.0 - a), 0, 255).astype(np.float32)
    comp4[patch_mask] = comp4[patch_mask] * 0.55 + np.array([255.0, 50.0, 50.0], dtype=np.float32) * 0.45
    p4 = Image.fromarray(np.clip(comp4, 0, 255).astype(np.uint8)).resize((pw, ph), Image.Resampling.BILINEAR)

    sheet = Image.new("RGB", (pw * 4 + 12, ph + 32), (32, 32, 36))
    draw = ImageDraw.Draw(sheet)
    labels = ["1. Base (4K orig)", "2. Aligned Canvas Edit", "3. Patched Result", "4. Patch Mask (Red)"]
    for idx, (panel, lbl) in enumerate(zip((p1, p2, p3, p4), labels)):
        x = idx * (pw + 4)
        sheet.paste(panel, (x, 28))
        draw.text((x + 8, 6), lbl, fill=(240, 240, 240))
    return sheet


def compose_canvas_edit_patch(
    base_layer_rgba: Any,
    original_rgba: Any,
    occluded_mask: np.ndarray,
    edit_image: Any,
    *,
    crop_bbox: Optional[Tuple[int, int, int, int]] = None,
    alpha_mode: str = "auto",
    max_underlap_px: int = 0,
    feather_px: int = 4,
    color_match: bool = True,
    replace_occluded: bool = False,
    aligned_edit_rgba: Optional[np.ndarray] = None,
) -> CanvasEditPatchResult:
    """Losslessly patch occluded backing pixels from a Canvas edit onto `base_layer_rgba`.

    Visible pixels of `base_layer_rgba` outside `occluded_mask` are always guaranteed bit-for-bit
    preserved so rest-pose fidelity stays 100% identical to original 4K art. When
    `replace_occluded=True`, existing opaque or semi-transparent pixels strictly inside the occluded
    zone (hidden behind the front part at rest) may be replaced by the Canvas-inpainted backing.
    """
    import cv2

    base = _to_rgba_array(base_layer_rgba)
    orig = _to_rgba_array(original_rgba)
    h, w = base.shape[:2]
    if orig.shape[:2] != (h, w):
        raise ValueError(f"base_layer_rgba {base.shape[:2]} != original_rgba {orig.shape[:2]}")

    occ = np.asarray(occluded_mask, dtype=bool)
    if occ.shape != (h, w):
        raise ValueError(f"occluded_mask shape {occ.shape} != {(h, w)}")

    if aligned_edit_rgba is not None:
        aligned = _to_rgba_array(aligned_edit_rgba)
        reg_metrics: Dict[str, Any] = {"method": "pre_aligned"}
    else:
        edit_arr = _to_rgba_array(edit_image)
        aligned, reg_metrics = register_edit_to_source(edit_arr, orig, crop_bbox=crop_bbox)

    est_alpha, alpha_meta = estimate_edit_foreground_alpha(
        aligned,
        base,
        orig,
        occ,
        mode=alpha_mode,
        source_has_native_alpha=reg_metrics.get("source_has_native_alpha"),
        replace_occluded=replace_occluded,
    )
    aligned_with_alpha = aligned.copy()
    aligned_with_alpha[..., 3] = est_alpha

    patch_zone = occ.copy()
    if max_underlap_px > 0:
        base_support = base[..., 3] >= 32
        if base_support.any():
            patch_zone &= _dilate_bool(base_support, int(max_underlap_px))

    if replace_occluded:
        fill_candidates = patch_zone & (aligned[..., 3] > 0)
    else:
        fill_candidates = patch_zone & (base[..., 3] < 250) & (est_alpha > 0)

    shift_rgb = np.zeros(3, dtype=np.float32)
    if color_match and fill_candidates.any():
        ring = _dilate_bool(fill_candidates, 18) & (~patch_zone) & (base[..., 3] >= 180) & (est_alpha >= 180)
        if int(ring.sum()) < 32:
            if replace_occluded:
                ring = _dilate_bool(fill_candidates, 32) & (~patch_zone) & (base[..., 3] >= 160) & (est_alpha >= 160)
            else:
                ring = _dilate_bool(fill_candidates, 28) & (~fill_candidates) & (base[..., 3] >= 160) & (est_alpha >= 160)
        if int(ring.sum()) >= 32:
            diff = base[..., :3][ring].astype(np.float32) - aligned[..., :3][ring].astype(np.float32)
            shift_rgb = np.clip(np.median(diff, axis=0), -30.0, 30.0).astype(np.float32)

    patched_rgb = np.clip(aligned[..., :3].astype(np.float32) + shift_rgb[None, None, :], 0.0, 255.0)
    base_f = base.astype(np.float32)
    ba = base_f[..., 3:4] / 255.0

    if replace_occluded:
        if feather_px > 0 and fill_candidates.any():
            zone_u8 = fill_candidates.astype(np.uint8) * 255
            dist_in = cv2.distanceTransform(zone_u8, cv2.DIST_L2, 5)
            w_in = np.where(fill_candidates, np.clip(dist_in / float(feather_px), 0.0, 1.0), 0.0)[..., None]
        else:
            w_in = fill_candidates.astype(np.float32)[..., None]
        ta = (est_alpha.astype(np.float32) / 255.0)[..., None]
        out_a = ba * (1.0 - w_in) + ta * w_in
        out_rgb = np.divide(
            base_f[..., :3] * ba * (1.0 - w_in) + patched_rgb * ta * w_in,
            out_a,
            out=np.zeros_like(base_f[..., :3]),
            where=out_a > 1e-6,
        )
        out = np.concatenate([np.clip(np.rint(out_rgb), 0, 255), np.clip(np.rint(out_a * 255.0), 0, 255)], axis=2).astype(
            np.uint8
        )
        sacred = ~patch_zone
        out[sacred] = base[sacred]
        patched_mask = (~sacred) & (np.abs(out.astype(np.int16) - base.astype(np.int16)).max(axis=2) > 0)
    else:
        patch_alpha_f = np.where(fill_candidates, est_alpha.astype(np.float32) / 255.0, 0.0)
        if feather_px > 0 and fill_candidates.any():
            zone_u8 = patch_zone.astype(np.uint8) * 255
            dist_in = cv2.distanceTransform(zone_u8, cv2.DIST_L2, 5)
            ramp = np.clip(dist_in / float(feather_px), 0.0, 1.0)
            near_base = _dilate_bool(base[..., 3] >= 200, int(feather_px) + 2)
            ramp = np.where(near_base, 1.0, ramp)
            patch_alpha_f *= ramp

        pa = patch_alpha_f[..., None]
        out_a = ba + pa * (1.0 - ba)
        out_rgb = np.divide(
            base_f[..., :3] * ba + patched_rgb * pa * (1.0 - ba),
            out_a,
            out=base_f[..., :3].copy(),
            where=out_a > 1e-6,
        )
        out = np.concatenate([np.clip(np.rint(out_rgb), 0, 255), np.clip(np.rint(out_a * 255.0), 0, 255)], axis=2).astype(
            np.uint8
        )
        sacred = (~patch_zone) | (base[..., 3] == 255)
        out[sacred] = base[sacred]
        patched_mask = (~sacred) & (out[..., 3] > base[..., 3])

    outside_diff = int(np.abs(out[~occ].astype(np.int16) - base[~occ].astype(np.int16)).max()) if (~occ).any() else 0
    patch_zone_px = int(patch_zone.sum())
    edit_fg_in_patch_px = int((patch_zone & (est_alpha >= 32)).sum())
    edit_fg_ratio = float(edit_fg_in_patch_px / max(1, patch_zone_px))

    if patch_zone_px == 0:
        diagnosis, next_action = "empty_patch_zone", "check_occluded_mask_or_underlap"
    elif patch_zone_px >= 64 and edit_fg_ratio < 0.03:
        diagnosis, next_action = "open_air_no_backing", "clean_open_air_halo"
    elif edit_fg_ratio < 0.12 or int(patched_mask.sum()) == 0:
        diagnosis, next_action = "sparse_or_shifted_backing", "retry_local_crop_or_rebind_dead_crossing"
    else:
        diagnosis, next_action = "usable_backing", "inspect_check_image_and_probe"

    stats: Dict[str, Any] = {
        "method": "aligned_occluded_backing_patch",
        "registration": reg_metrics,
        **alpha_meta,
        "replace_occluded": bool(replace_occluded),
        "occluded_mask_px": int(occ.sum()),
        "patch_zone_px": patch_zone_px,
        "patched_px": int(patched_mask.sum()),
        "edit_fg_in_patch_zone_px": edit_fg_in_patch_px,
        "edit_fg_ratio_in_patch_zone": round(edit_fg_ratio, 4),
        "backing_diagnosis": diagnosis,
        "recommended_next_action": next_action,
        "color_shift_rgb": [round(float(v), 2) for v in shift_rgb],
        "max_outside_diff": outside_diff,
        "rest_pose_preserved": outside_diff == 0,
    }
    comp = _render_patch_comparison(base, aligned_with_alpha, out, patched_mask)
    return CanvasEditPatchResult(rgba=out, stats=stats, comparison=comp)


def plan_occluded_edit_crop(
    original_rgba: Any,
    *,
    occluded_mask: Optional[np.ndarray] = None,
    front_layer_rgba: Optional[Any] = None,
    base_layer_rgba: Optional[Any] = None,
    diagnostics: Optional[Dict[str, Any]] = None,
    layer_pair: Optional[Tuple[str, str]] = None,
    pad_ratio: float = 0.45,
    min_pad_px: int = 128,
    min_crop_edge: int = 512,
    region_expand_px: Optional[int] = None,
    contact_radius_px: Optional[int] = None,
    max_crop_area_ratio: float = 0.55,
    require_diagnostics_match: bool = False,
) -> OccludedEditCropPlan:
    """Automatically derive an anchor-rich local crop and occluded mask for `canvas_edit_image`.

    Eliminates manual per-hero coordinate recipes and prevents full-body geometry drift when
    generative edit models remove an occluding strap, braid, scarf, or floating prop.
    """
    orig = _to_rgba_array(original_rgba)
    h, w = orig.shape[:2]
    long_edge = max(h, w)
    front = _to_rgba_array(front_layer_rgba) if front_layer_rgba is not None else None
    base = _to_rgba_array(base_layer_rgba) if base_layer_rgba is not None else None
    if front is not None and front.shape[:2] != (h, w):
        raise ValueError(f"front_layer_rgba {front.shape[:2]} != original {orig.shape[:2]}")
    if base is not None and base.shape[:2] != (h, w):
        raise ValueError(f"base_layer_rgba {base.shape[:2]} != original {orig.shape[:2]}")

    source_kind = "explicit_mask"
    matched_regions = 0
    if occluded_mask is not None:
        mask = np.asarray(occluded_mask, dtype=bool)
        if mask.shape != (h, w):
            raise ValueError(f"occluded_mask shape {mask.shape} != {(h, w)}")
    else:
        diag_roi = np.zeros((h, w), dtype=bool)
        if diagnostics is not None:
            stress = diagnostics.get("exposure_stress") if isinstance(diagnostics.get("exposure_stress"), dict) else diagnostics
            levels = stress.get("levels") if isinstance(stress, dict) else None
            exp = int(region_expand_px if region_expand_px is not None else max(48, round(96 * long_edge / 1800.0)))
            target_pair = set(layer_pair) if layer_pair else None
            if isinstance(levels, dict):
                for lvl in levels.values():
                    for frame in (lvl or {}).get("frames") or []:
                        for reg in frame.get("regions") or []:
                            layers = set(reg.get("layers") or [])
                            if target_pair is not None and not target_pair.issubset(layers):
                                continue
                            bbox = reg.get("bbox")
                            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                                rx0, ry0, rx1, ry1 = (int(v) for v in bbox)
                                x0, y0 = max(0, rx0 - exp), max(0, ry0 - exp)
                                x1, y1 = min(w, rx1 + exp), min(h, ry1 + exp)
                                if x1 > x0 and y1 > y0:
                                    diag_roi[y0:y1, x0:x1] = True
                                    matched_regions += 1
        if diag_roi.any():
            source_kind = "diagnostics_exposure_regions"
            if front is not None:
                mask = diag_roi & (front[..., 3] >= 16)
                if not mask.any():
                    mask = diag_roi & (orig[..., 3] >= 16)
            else:
                mask = diag_roi & (orig[..., 3] >= 16)
        elif front is not None:
            front_mask = front[..., 3] >= 16
            if base is not None:
                c_rad = int(contact_radius_px if contact_radius_px is not None else max(120, round(long_edge * 0.04)))
                near_base = _dilate_bool(base[..., 3] >= 16, c_rad)
                contact = front_mask & near_base
                if contact.any():
                    mask = contact
                    source_kind = "front_base_contact"
                else:
                    mask = front_mask
                    source_kind = "front_layer_full"
            else:
                mask = front_mask
                source_kind = "front_layer_full"
        else:
            raise ValueError("Provide occluded_mask, diagnostics with exposure regions, or front_layer_rgba.")

    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError(f"Derived occluded mask is empty (source={source_kind})")

    mx0, my0, mx1, my1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    mw, mh = mx1 - mx0, my1 - my0
    pad = max(int(min_pad_px), int(round(max(mw, mh) * float(pad_ratio))))
    cx0, cy0 = max(0, mx0 - pad), max(0, my0 - pad)
    cx1, cy1 = min(w, mx1 + pad), min(h, my1 + pad)

    min_edge_w = min(int(min_crop_edge), w)
    min_edge_h = min(int(min_crop_edge), h)
    if (cx1 - cx0) < min_edge_w:
        mid_x = (cx0 + cx1) // 2
        cx0 = max(0, min(w - min_edge_w, mid_x - min_edge_w // 2))
        cx1 = cx0 + min_edge_w
    if (cy1 - cy0) < min_edge_h:
        mid_y = (cy0 + cy1) // 2
        cy0 = max(0, min(h - min_edge_h, mid_y - min_edge_h // 2))
        cy1 = cy0 + min_edge_h

    crop_bbox = (int(cx0), int(cy0), int(cx1 - cx0), int(cy1 - cy0))
    crop_rgba = np.ascontiguousarray(orig[cy0:cy1, cx0:cx1].copy())
    mask_u8 = (mask.astype(np.uint8) * 255)
    crop_area_ratio = round(float(crop_bbox[2] * crop_bbox[3]) / float(max(1, w * h)), 4)
    diagnostics_supplied = diagnostics is not None
    diagnostics_matched = matched_regions > 0
    warnings = []
    if diagnostics_supplied and not diagnostics_matched:
        warnings.append("diagnostics_no_matching_exposure_regions_fell_back_to_contact")
    if crop_area_ratio > float(max_crop_area_ratio):
        warnings.append(f"crop_too_wide_near_full_frame:{crop_area_ratio:.2%}>{float(max_crop_area_ratio):.2%}")

    if require_diagnostics_match and diagnostics_supplied and not diagnostics_matched:
        crop_status = "diagnostics_unmatched_requires_human_review"
        usable = False
    elif crop_area_ratio > float(max_crop_area_ratio):
        crop_status = "crop_too_wide_requires_human_review"
        usable = False
    else:
        crop_status = "ready"
        usable = True

    meta = {
        "status": crop_status,
        "usable_for_local_edit": usable,
        "source": source_kind,
        "diagnostics_supplied": diagnostics_supplied,
        "diagnostics_matched": diagnostics_matched,
        "matched_exposure_regions": matched_regions,
        "image_size": [w, h],
        "occluded_bbox_xywh": [mx0, my0, mw, mh],
        "occluded_mask_px": int(mask.sum()),
        "crop_bbox": list(crop_bbox),
        "crop_area_ratio": crop_area_ratio,
        "max_crop_area_ratio": float(max_crop_area_ratio),
        "pad_px": pad,
        "warnings": warnings,
    }
    return OccludedEditCropPlan(crop_rgba=crop_rgba, occluded_mask_u8=mask_u8, crop_bbox=crop_bbox, meta=meta)


def inpaint_translucent_fx_underlap(
    fx_layer_rgba: Any,
    occluder_rgba: Any,
    *,
    max_underlap_px: int = 72,
    blur_sigma: float = 4.0,
    inpaint_radius: int = 14,
) -> CanvasEditPatchResult:
    """Smoothly extend a translucent FX layer strictly behind an occluding foreground prop.

    Eliminates sharp cookie-cutter silhouette cutouts on translucent glow/swirl/mist layers when a
    foreground prop moves away, while keeping 100% of visible pixels outside `occluder_rgba`
    bit-for-bit identical (`max_outside_diff == 0`).
    """
    import cv2

    fx = _to_rgba_array(fx_layer_rgba)
    occ_arr = _to_rgba_array(occluder_rgba)
    if fx.shape[:2] != occ_arr.shape[:2]:
        raise ValueError(f"fx_layer_rgba {fx.shape[:2]} != occluder_rgba {occ_arr.shape[:2]}")
    h, w = fx.shape[:2]

    occ = occ_arr[..., 3] >= 16
    fx_support = fx[..., 3] >= 16
    fill_zone = occ & (fx[..., 3] < 250) & _dilate_bool(fx_support, int(max_underlap_px))
    if not fill_zone.any():
        stats = {
            "method": "inpaint_translucent_fx_underlap",
            "filled_px": 0,
            "fill_zone_px": 0,
            "max_outside_diff": 0,
            "rest_pose_preserved": True,
        }
        return CanvasEditPatchResult(rgba=fx.copy(), stats=stats, comparison=_render_patch_comparison(fx, occ_arr, fx, fill_zone))

    ys, xs = np.nonzero(fill_zone)
    pad = max(int(max_underlap_px) + 32, int(inpaint_radius) * 3)
    ry0, ry1 = max(0, int(ys.min()) - pad), min(h, int(ys.max()) + 1 + pad)
    rx0, rx1 = max(0, int(xs.min()) - pad), min(w, int(xs.max()) + 1 + pad)

    roi_fx = fx[ry0:ry1, rx0:rx1]
    roi_fill = fill_zone[ry0:ry1, rx0:rx1]
    roi_a = roi_fx[..., 3]
    roi_hole_u8 = roi_fill.astype(np.uint8) * 255

    # 1. Smoothly inpaint alpha across the occluded cutout.
    inpainted_a = cv2.inpaint(roi_a, roi_hole_u8, inpaintRadius=int(inpaint_radius), flags=cv2.INPAINT_TELEA)
    if float(blur_sigma) > 0:
        inpainted_a = cv2.GaussianBlur(inpainted_a, (0, 0), sigmaX=float(blur_sigma))

    # Taper alpha smoothly if fill_zone extends all the way to the open-air far side of max_underlap_px.
    if int(max_underlap_px) > 0:
        dist_from_fx = cv2.distanceTransform((~(roi_a >= 16)).astype(np.uint8), cv2.DIST_L2, 5)
        taper = np.clip(1.0 - np.maximum(0.0, dist_from_fx - float(max_underlap_px) * 0.7) / max(1.0, float(max_underlap_px) * 0.35), 0.0, 1.0)
        inpainted_a = np.clip(np.rint(inpainted_a.astype(np.float32) * taper), 0, 255).astype(np.uint8)

    # 2. Inpaint RGB strictly from visible non-empty glow pixels (mark transparent air as unknown too so black never bleeds in).
    rgb_unknown = ((roi_a < 24) | roi_fill).astype(np.uint8) * 255
    roi_rgb_bgr = cv2.cvtColor(roi_fx[..., :3], cv2.COLOR_RGB2BGR)
    inpainted_bgr = cv2.inpaint(roi_rgb_bgr, rgb_unknown, inpaintRadius=int(inpaint_radius), flags=cv2.INPAINT_TELEA)
    inpainted_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
    if float(blur_sigma) > 0:
        inpainted_rgb = cv2.GaussianBlur(inpainted_rgb, (0, 0), sigmaX=max(1.0, float(blur_sigma) * 0.6))

    out = fx.copy()
    roi_out = out[ry0:ry1, rx0:rx1]
    # Keep higher alpha where base already had partial glow, or fill smoothly inside roi_fill.
    new_a = np.where(roi_fill, np.maximum(roi_a, inpainted_a), roi_a)
    blend_rgb_mask = roi_fill & (new_a > roi_a)
    roi_out[blend_rgb_mask, :3] = inpainted_rgb[blend_rgb_mask]
    roi_out[roi_fill, 3] = new_a[roi_fill]
    out[ry0:ry1, rx0:rx1] = roi_out

    # Enforce strict bit-for-bit preservation outside occluder silhouette.
    sacred = (~occ) | (fx[..., 3] == 255)
    out[sacred] = fx[sacred]

    filled_mask = (~sacred) & (out[..., 3] > fx[..., 3])
    outside_diff = int(np.abs(out[~occ].astype(np.int16) - fx[~occ].astype(np.int16)).max()) if (~occ).any() else 0
    stats: Dict[str, Any] = {
        "method": "inpaint_translucent_fx_underlap",
        "max_underlap_px": int(max_underlap_px),
        "blur_sigma": float(blur_sigma),
        "fill_zone_px": int(fill_zone.sum()),
        "filled_px": int(filled_mask.sum()),
        "max_outside_diff": outside_diff,
        "rest_pose_preserved": outside_diff == 0,
    }
    comp = _render_patch_comparison(fx, occ_arr, out, filled_mask)
    return CanvasEditPatchResult(rgba=out, stats=stats, comparison=comp)


def clean_open_air_halo(
    base_layer_rgba: Any,
    front_layer_rgba: Any,
    *,
    radius_px: int = 14,
    max_halo_alpha: int = 215,
    min_solid_alpha: int = 245,
) -> CanvasEditPatchResult:
    """Strip soft front-prop halo/glow fringe left on `base_layer_rgba` when backing is open air.

    Preserves 100% of solid opaque body core pixels and all pixels outside the immediate halo ring
    around `front_layer_rgba`.
    """
    base = _to_rgba_array(base_layer_rgba)
    front = _to_rgba_array(front_layer_rgba)
    if base.shape[:2] != front.shape[:2]:
        raise ValueError(f"base_layer_rgba {base.shape[:2]} != front_layer_rgba {front.shape[:2]}")

    front_mask = front[..., 3] >= 16
    solid_core = _erode_bool(base[..., 3] >= int(min_solid_alpha), 2)
    halo_zone = _dilate_bool(front_mask, int(radius_px)) & (~solid_core)
    strip_mask = halo_zone & (base[..., 3] > 0) & (
        (base[..., 3] <= int(max_halo_alpha)) | (front_mask & (base[..., 3] < int(min_solid_alpha)))
    )

    out = base.copy()
    out[strip_mask] = 0
    out[solid_core] = base[solid_core]
    out[~halo_zone] = base[~halo_zone]

    core_diff = int(np.abs(out[solid_core].astype(np.int16) - base[solid_core].astype(np.int16)).max()) if solid_core.any() else 0
    outside_diff = int(np.abs(out[~halo_zone].astype(np.int16) - base[~halo_zone].astype(np.int16)).max()) if (~halo_zone).any() else 0

    stats: Dict[str, Any] = {
        "method": "clean_open_air_halo",
        "radius_px": int(radius_px),
        "max_halo_alpha": int(max_halo_alpha),
        "halo_zone_px": int(halo_zone.sum()),
        "stripped_halo_px": int(strip_mask.sum()),
        "max_solid_core_diff": core_diff,
        "max_outside_halo_diff": outside_diff,
        "solid_core_preserved": core_diff == 0,
    }
    comp = _render_patch_comparison(base, front, out, strip_mask)
    return CanvasEditPatchResult(rgba=out, stats=stats, comparison=comp)
