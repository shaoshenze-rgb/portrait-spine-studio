"""General PSD multi-layer inspection and artist-layer body splitting.

Turns multi-layer artist PSD files directly into standard Canvas-layer parts
without throwaway per-character scripts:

1. ``inspect_psd_layers`` scans any PSD (read-only), assigns stable descendant
   indices, classifies structural vs background vs adjustment vs clipping vs
   cross-part overlay/FX layers, and optionally renders an indexed contact sheet.
2. ``split_from_psd`` builds pixel ownership strictly from structural layers
   (preventing cross-part rim-light/highlight overlays from sticking to one wrong
   part), preserves genuine artist-painted hidden pixels under front occluders,
   bridges any remaining occluded contact seams strictly behind opaque front
   layers (zero rest-pose pixel distortion), and applies 4K-aware edge defringe.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from .canvas_layer_cleanup import defringe_rgba
from .local_matte import _is_psd_bg_layer

_ADJUSTMENT_KINDS = frozenset(
    {
        "levels",
        "curves",
        "hue_saturation",
        "huesaturation",
        "color_balance",
        "colorbalance",
        "brightness_contrast",
        "brightnesscontrast",
        "exposure",
        "gradient_map",
        "gradientmap",
        "selective_color",
        "selectivecolor",
        "vibrance",
        "photo_filter",
        "photofilter",
        "invert",
        "posterize",
        "threshold",
        "black_and_white",
        "blackwhite",
        "channel_mixer",
        "channelmixer",
        "color_lookup",
        "colorlookup",
        "solidcolorfill",
        "gradientfill",
        "patternfill",
    }
)

_OVERLAY_BLEND_TOKENS = (
    "screen",
    "linear_dodge",
    "lineardodge",
    "color_dodge",
    "colordodge",
    "lighter_color",
    "lighten",
    "overlay",
    "soft_light",
    "softlight",
    "hard_light",
    "hardlight",
    "vivid_light",
    "linear_light",
    "pin_light",
    "multiply",
    "color_burn",
    "linear_burn",
    "darken",
    "divide",
    "subtract",
    "difference",
    "exclusion",
    "hue",
    "saturation",
    "color",
    "luminosity",
)

_OVERLAY_NAME_HINTS = (
    "光源",
    "高光",
    "光效",
    "外发光",
    "氛围",
    "环境光",
    "rim",
    "glow",
    "highlight",
    "lighting",
    "light",
    "fx",
)

_SAFE_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,47}$")


@dataclass(frozen=True)
class PsdPartSpec:
    id: str
    psd_indices: Tuple[int, ...]
    draw_order: int
    kind: str = "rigid"
    label: Optional[str] = None
    role: Optional[str] = None
    parent: Optional[str] = None
    pivot: Optional[Tuple[float, float]] = None


@dataclass(frozen=True)
class PsdSplitResult:
    rgba: Dict[str, np.ndarray]
    ownership: Dict[str, np.ndarray]
    stats: Dict[str, Any]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(Path(path).read_bytes())


def _is_effectively_visible(layer: Any) -> bool:
    cur = layer
    while cur is not None:
        if hasattr(cur, "visible") and not bool(cur.visible):
            return False
        cur = getattr(cur, "parent", None)
        if cur is not None and not hasattr(cur, "parent"):
            break
    return True


def _layer_path(layer: Any) -> str:
    parts: List[str] = []
    cur = layer
    while cur is not None and hasattr(cur, "name"):
        name = str(getattr(cur, "name", "") or "").strip()
        if name:
            parts.append(name)
        parent = getattr(cur, "parent", None)
        if parent is None or not hasattr(parent, "parent"):
            break
        cur = parent
    return " / ".join(reversed(parts))


def _is_overlay_blend(blend: str) -> bool:
    low = str(blend or "").strip().lower()
    if not low or "normal" in low or "pass" in low:
        return False
    return any(tok in low for tok in _OVERLAY_BLEND_TOKENS)


def _is_adjustment_kind(kind: str) -> bool:
    low = str(kind or "").strip().lower().replace(" ", "").replace("-", "_")
    if low in _ADJUSTMENT_KINDS:
        return True
    return any(tok in low for tok in ("adjustment", "fill", "levels", "curves", "brightness", "exposure", "gradient"))


def _classify_psd_leaf(
    layer: Any,
    psd_width: int,
    psd_height: int,
    *,
    is_bottom_top_level: bool,
) -> str:
    kind = str(getattr(layer, "kind", "") or "").lower()
    if _is_psd_bg_layer(layer, psd_width, psd_height, is_bottom=is_bottom_top_level):
        return "background"
    if _is_adjustment_kind(kind):
        return "adjustment"
    if bool(getattr(layer, "clipping", False)):
        return "clipping"
    blend = str(getattr(layer, "blend_mode", "") or "")
    if _is_overlay_blend(blend):
        return "overlay_fx"
    lname = str(getattr(layer, "name", "") or "").strip().lower()
    if any(hint in lname for hint in _OVERLAY_NAME_HINTS):
        return "overlay_fx"
    return "structural"


def _render_psd_contact_sheet(
    psd_layers: Sequence[Any],
    records: Sequence[Dict[str, Any]],
    out_path: Path,
    *,
    columns: int = 5,
    tile: int = 220,
) -> Path:
    drawable = [r for r in records if r["category"] not in ("background", "adjustment")]
    if not drawable:
        drawable = list(records)
    cols = min(max(1, columns), max(1, len(drawable)))
    rows = max(1, math.ceil(len(drawable) / cols))
    header = 34
    sheet = Image.new("RGB", (cols * tile, rows * (tile + header)), (30, 32, 38))
    tile_bg = Image.new("RGB", (tile, tile), (96, 98, 104))
    checker = ImageDraw.Draw(tile_bg)
    for y in range(0, tile, 14):
        for x in range(0, tile, 14):
            if (x // 14 + y // 14) % 2:
                checker.rectangle((x, y, x + 13, y + 13), fill=(132, 134, 140))
    draw = ImageDraw.Draw(sheet)
    for pos, rec in enumerate(drawable):
        x = (pos % cols) * tile
        y = (pos // cols) * (tile + header)
        sheet.paste(tile_bg, (x, y + header))
        layer = psd_layers[rec["index"]]
        try:
            pil = layer.topil()
        except Exception:
            pil = None
        if pil is not None:
            thumb = pil.convert("RGBA")
            thumb.thumbnail((tile - 12, tile - 12), Image.Resampling.LANCZOS)
            sheet.paste(thumb, (x + (tile - thumb.width) // 2, y + header + (tile - thumb.height) // 2), thumb)
        tag = {"structural": "S", "overlay_fx": "FX", "clipping": "CL"}.get(rec["category"], "?")
        label = f"#{rec['index']} [{tag}] {rec['name'][:18]}"
        color = (140, 235, 175) if tag == "S" else (255, 208, 120) if tag == "FX" else (160, 200, 255)
        draw.text((x + 6, y + 8), label, fill=color)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=92)
    return out


def inspect_psd_layers(
    psd_path: Path,
    *,
    canvas_size: Optional[Tuple[int, int]] = None,
    contact_sheet_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Inspect a PSD file (read-only) and return indexed layer metadata for Spine splitting."""
    from psd_tools import PSDImage

    src = Path(psd_path).resolve()
    psd = PSDImage.open(str(src))
    pw, ph = int(psd.width), int(psd.height)
    cw, ch = (int(canvas_size[0]), int(canvas_size[1])) if canvas_size else (pw, ph)

    top_ids = {id(ly): idx for idx, ly in enumerate(list(psd))}
    descendants = list(psd.descendants())
    records: List[Dict[str, Any]] = []
    last_base_by_parent: Dict[int, int] = {}

    for idx, layer in enumerate(descendants):
        if layer.is_group():
            continue
        if not _is_effectively_visible(layer):
            continue
        bbox = list(layer.bbox) if layer.bbox else [0, 0, 0, 0]
        bw, bh = max(0, bbox[2] - bbox[0]), max(0, bbox[3] - bbox[1])
        kind = str(getattr(layer, "kind", "") or "")
        if bw <= 0 or bh <= 0:
            if not _is_adjustment_kind(kind):
                continue

        is_bottom_top = id(layer) in top_ids and top_ids[id(layer)] <= 2
        category = _classify_psd_leaf(layer, pw, ph, is_bottom_top_level=is_bottom_top)
        parent_id = id(getattr(layer, "parent", None))
        clip_base = None
        if category == "clipping":
            clip_base = last_base_by_parent.get(parent_id)
        elif category in ("structural", "overlay_fx"):
            last_base_by_parent[parent_id] = idx

        scaled_bbox = [
            int(round(bbox[0] * cw / max(1, pw))),
            int(round(bbox[1] * ch / max(1, ph))),
            int(round(bbox[2] * cw / max(1, pw))),
            int(round(bbox[3] * ch / max(1, ph))),
        ]
        records.append(
            {
                "index": idx,
                "name": str(getattr(layer, "name", "") or f"layer_{idx}"),
                "path": _layer_path(layer),
                "kind": kind,
                "category": category,
                "blend_mode": str(getattr(layer, "blend_mode", "") or ""),
                "opacity": int(getattr(layer, "opacity", 255)),
                "clipping": bool(getattr(layer, "clipping", False)),
                "clip_base_index": clip_base,
                "psd_bbox": bbox,
                "canvas_bbox": scaled_bbox,
                "area_ratio": round(float(bw * bh) / float(max(1, pw * ph)), 4),
            }
        )

    structural_count = sum(1 for r in records if r["category"] == "structural")
    overlay_count = sum(1 for r in records if r["category"] == "overlay_fx")
    clipping_count = sum(1 for r in records if r["category"] == "clipping")
    bg_count = sum(1 for r in records if r["category"] == "background")
    usable_multi_layer = structural_count >= 3

    report: Dict[str, Any] = {
        "psd_source": str(src),
        "psd_size": [pw, ph],
        "canvas_size": [cw, ch],
        "usable_multi_layer": usable_multi_layer,
        "counts": {
            "visible_leaves": len(records),
            "structural": structural_count,
            "overlay_fx": overlay_count,
            "clipping": clipping_count,
            "background": bg_count,
        },
        "layers": records,
    }
    if contact_sheet_path is not None:
        saved = _render_psd_contact_sheet(descendants, records, Path(contact_sheet_path))
        report["contact_sheet"] = str(saved)
    return report


def _nearest_labels(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (distance, seed_rows, seed_cols) to nearest True pixel in ``mask``."""
    try:
        import cv2

        dist, labels = cv2.distanceTransformWithLabels(
            (~mask).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
        )
        rows, cols = np.nonzero(mask)
        return dist, rows[labels - 1], cols[labels - 1]
    except Exception:
        row, col = np.indices(mask.shape, dtype=np.int32)
        best_r = np.where(mask, row, -1)
        best_c = np.where(mask, col, -1)
        best = np.where(mask, 0.0, np.inf)
        h, w = mask.shape
        step = 1 << max(0, (max(h, w) - 1).bit_length() - 1)
        while step:
            sr, sc = best_r.copy(), best_c.copy()
            for dy in (-step, 0, step):
                for dx in (-step, 0, step):
                    if (not dx and not dy) or abs(dy) >= h or abs(dx) >= w:
                        continue
                    dst = np.s_[max(0, dy) : h + min(0, dy), max(0, dx) : w + min(0, dx)]
                    org = np.s_[max(0, -dy) : h - max(0, dy), max(0, -dx) : w - max(0, dx)]
                    ry, rx = sr[org], sc[org]
                    d2 = (row[dst] - ry) ** 2 + (col[dst] - rx) ** 2
                    use = (ry >= 0) & (d2 < best[dst])
                    np.copyto(best[dst], d2, where=use)
                    np.copyto(best_r[dst], ry, where=use)
                    np.copyto(best_c[dst], rx, where=use)
            step //= 2
        return np.sqrt(best), best_r, best_c


def _extend_occluded_underlap(
    part_rgba: np.ndarray,
    occluded_allowed: np.ndarray,
    radius: int,
) -> Tuple[np.ndarray, int]:
    """Extend solid same-part pixels behind opaque higher layers (zero rest-pose impact)."""
    if radius <= 0 or not occluded_allowed.any():
        return part_rgba, 0
    core = part_rgba[..., 3] >= 245
    if not core.any():
        return part_rgba, 0

    ys, xs = np.nonzero(core)
    h, w = core.shape
    margin = radius + 4
    y0, y1 = max(0, int(ys.min()) - margin), min(h, int(ys.max()) + margin + 1)
    x0, x1 = max(0, int(xs.min()) - margin), min(w, int(xs.max()) + margin + 1)

    sub_img = part_rgba[y0:y1, x0:x1]
    sub_core = core[y0:y1, x0:x1]
    sub_allowed = occluded_allowed[y0:y1, x0:x1]
    if not sub_allowed.any():
        return part_rgba, 0

    dist, seed_r, seed_c = _nearest_labels(sub_core)
    domain = sub_allowed & (dist <= float(radius)) & (~sub_core)
    if not domain.any():
        return part_rgba, 0

    # Smooth edge feather at the outer boundary of the underlap strip
    try:
        import cv2

        inner_dist = cv2.distanceTransform((sub_core | domain).astype(np.uint8), cv2.DIST_L2, 5)
        fill_alpha = np.minimum(255, np.rint(inner_dist * 85.0)).astype(np.uint8)
    except Exception:
        fill_alpha = np.full(sub_core.shape, 255, dtype=np.uint8)

    fill = domain & (fill_alpha > sub_img[..., 3])
    if not fill.any():
        return part_rgba, 0

    out = part_rgba.copy()
    out_sub = out[y0:y1, x0:x1]
    out_sub[fill, :3] = sub_img[seed_r[fill], seed_c[fill], :3]
    out_sub[fill, 3] = fill_alpha[fill]
    return out, int(fill.sum())


def _rasterize_psd_layer(
    layer: Any,
    psd_size: Tuple[int, int],
    canvas_size: Tuple[int, int],
) -> Optional[Tuple[int, int, np.ndarray]]:
    """Render one PSD leaf layer scaled into ``canvas_size`` returning ``(left, top, rgba)``."""
    pil = layer.topil()
    if pil is None:
        return None
    pw, ph = psd_size
    cw, ch = canvas_size
    left = int(round(layer.left * cw / max(1, pw)))
    top = int(round(layer.top * ch / max(1, ph)))
    right = int(round(layer.right * cw / max(1, pw)))
    bottom = int(round(layer.bottom * ch / max(1, ph)))
    tw, th = max(1, right - left), max(1, bottom - top)
    tile = pil.convert("RGBA")
    if tile.size != (tw, th):
        tile = tile.resize((tw, th), Image.Resampling.LANCZOS)
    arr = np.asarray(tile, dtype=np.uint8)

    # Clip to canvas bounds cleanly
    sx0, sy0 = max(0, -left), max(0, -top)
    dx0, dy0 = max(0, left), max(0, top)
    dx1, dy1 = min(cw, left + tw), min(ch, top + th)
    if dx1 <= dx0 or dy1 <= dy0:
        return None
    cropped = arr[sy0 : sy0 + (dy1 - dy0), sx0 : sx0 + (dx1 - dx0)]
    return dx0, dy0, cropped


def split_from_psd(
    source_rgba: np.ndarray,
    psd_path: Path,
    part_specs: Sequence[PsdPartSpec],
    *,
    overlay_indices: Optional[Iterable[int]] = None,
    auto_exclude_overlays: bool = True,
    underlap_px: Optional[int] = None,
    aa_boundary_px: int = 9,
    max_unclaimed_ratio: float = 0.015,
    defringe: bool = True,
    psd_layers_override: Optional[Sequence[Any]] = None,
    psd_size_override: Optional[Tuple[int, int]] = None,
) -> PsdSplitResult:
    """Split an aligned transparent portrait using its sibling multi-layer PSD.

    Guarantees:
    - Cross-part overlay/highlight layers (explicitly listed or non-normal blend)
      never steal pixel ownership from underlying structural parts.
    - Visible pixels are copied verbatim from ``source_rgba`` (preserving all
      lighting/color-grade/blend effects baked into the final artwork).
    - Genuine artist-painted hidden pixels behind front layers are preserved
      first, then bounded occluded underlap fills any remaining contact gap
      strictly behind opaque front layers (0 pixels changed in rest composite).
    """
    art = np.asarray(source_rgba, dtype=np.uint8)
    if art.ndim != 3 or art.shape[2] != 4 or not art.size:
        raise ValueError("Expected non-empty uint8 RGBA source_rgba")
    height, width = art.shape[:2]
    subject = art[..., 3] > 0
    subject_px = int(subject.sum())
    if subject_px == 0:
        raise ValueError("source_rgba contains no visible subject pixels")

    ordered = sorted(part_specs, key=lambda spec: spec.draw_order)
    if not ordered:
        raise ValueError("part_specs must be non-empty")
    if len({s.id for s in ordered}) != len(ordered):
        raise ValueError("Duplicate part id in part_specs")
    if len({s.draw_order for s in ordered}) != len(ordered):
        raise ValueError("Duplicate draw_order in part_specs")
    for s in ordered:
        if not _SAFE_ID_RE.fullmatch(s.id):
            raise ValueError(f"Invalid part id: {s.id!r}")
        if s.kind not in ("rigid", "soft"):
            raise ValueError(f"{s.id}: kind must be 'rigid' or 'soft'")

    if psd_layers_override is not None and psd_size_override is not None:
        psd_layers = list(psd_layers_override)
        psd_size = (int(psd_size_override[0]), int(psd_size_override[1]))
    else:
        from psd_tools import PSDImage

        psd = PSDImage.open(str(Path(psd_path).resolve()))
        psd_layers = list(psd.descendants())
        psd_size = (int(psd.width), int(psd.height))

    explicit_overlays = set(int(i) for i in (overlay_indices or ()))
    memberships: Dict[int, int] = {}
    for order_idx, spec in enumerate(ordered):
        for psd_idx in spec.psd_indices:
            if not (0 <= int(psd_idx) < len(psd_layers)):
                raise ValueError(f"{spec.id}: psd index {psd_idx} out of range (0..{len(psd_layers)-1})")
            memberships[int(psd_idx)] = order_idx

    # Classify each referenced layer as structural vs overlay/highlight so cross-part
    # highlights never overwrite another part's structural ownership.
    overlay_set = set(explicit_overlays)
    if auto_exclude_overlays:
        for psd_idx in memberships:
            ly = psd_layers[psd_idx]
            blend = str(getattr(ly, "blend_mode", "") or "")
            kind = str(getattr(ly, "kind", "") or "")
            if _is_overlay_blend(blend) or _is_adjustment_kind(kind):
                overlay_set.add(psd_idx)

    artist_canvases = {i: Image.new("RGBA", (width, height)) for i in range(len(ordered))}
    ownership = np.full((height, width), -1, dtype=np.int16)
    faint_owner = np.full((height, width), -1, dtype=np.int16)
    faint_alpha = np.zeros((height, width), dtype=np.uint8)
    overlay_support = np.zeros((height, width), dtype=bool)
    overlay_rebound_px = 0

    # Pass 1: rasterize structural layers (back-to-front by part draw_order, then PSD index)
    structural_items = sorted(
        [(order_idx, psd_idx) for psd_idx, order_idx in memberships.items() if psd_idx not in overlay_set]
    )
    for order_idx, psd_idx in structural_items:
        layer = psd_layers[psd_idx]
        if getattr(layer, "is_group", lambda: False)():
            continue
        rendered = _rasterize_psd_layer(layer, psd_size, (width, height))
        if rendered is None:
            continue
        x0, y0, tile = rendered
        th, tw = tile.shape[:2]
        artist_canvases[order_idx].alpha_composite(Image.fromarray(tile), (x0, y0))
        alpha = tile[..., 3]
        solid = alpha >= 96
        ownership[y0 : y0 + th, x0 : x0 + tw][solid] = order_idx
        faint_slice = faint_alpha[y0 : y0 + th, x0 : x0 + tw]
        better_faint = (alpha > 0) & (alpha >= faint_slice)
        faint_slice[better_faint] = alpha[better_faint]
        faint_owner[y0 : y0 + th, x0 : x0 + tw][better_faint] = order_idx

    # Pass 2: track cross-part overlay layers WITHOUT letting them overwrite structural_owner.
    # Any overlay pixel sitting on top of another part stays bound to that underlying part!
    for psd_idx in sorted(overlay_set):
        if not (0 <= psd_idx < len(psd_layers)):
            continue
        layer = psd_layers[psd_idx]
        if getattr(layer, "is_group", lambda: False)():
            continue
        rendered = _rasterize_psd_layer(layer, psd_size, (width, height))
        if rendered is None:
            continue
        x0, y0, tile = rendered
        th, tw = tile.shape[:2]
        alpha = tile[..., 3]
        ov_mask = alpha >= 16
        if psd_idx in memberships:
            assigned_order = memberships[psd_idx]
            rebound = ov_mask & (ownership[y0 : y0 + th, x0 : x0 + tw] >= 0) & (
                ownership[y0 : y0 + th, x0 : x0 + tw] != assigned_order
            )
            overlay_rebound_px += int(rebound.sum())
        overlay_support[y0 : y0 + th, x0 : x0 + tw] |= ov_mask

    # Assign remaining faint structural pixels
    faint_assign = (ownership < 0) & (faint_owner >= 0) & subject
    ownership[faint_assign] = faint_owner[faint_assign]

    # Bounded AA edge assignment (for antialiased silhouette fringes & thin overlay halos)
    missing_before_aa = (ownership < 0) & subject
    boundary_assigned_px = 0
    if missing_before_aa.any() and (ownership >= 0).any() and aa_boundary_px > 0:
        dist, seed_r, seed_c = _nearest_labels(ownership >= 0)
        assign = missing_before_aa & (dist <= float(aa_boundary_px))
        ownership[assign] = ownership[seed_r[assign], seed_c[assign]]
        boundary_assigned_px = int(assign.sum())

    unclaimed = (ownership < 0) & subject
    unclaimed_px = int(unclaimed.sum())
    unclaimed_ratio = float(unclaimed_px / max(1, subject_px))
    unclaimed_solid_px = int((unclaimed & (art[..., 3] >= 96)).sum())
    if unclaimed_ratio > max_unclaimed_ratio:
        raise ValueError(
            f"PSD layer coverage below threshold: unclaimed={unclaimed_px}px ({unclaimed_ratio:.4%} > {max_unclaimed_ratio:.2%})"
        )

    pad_px = (
        max(24, min(48, int(round(32.0 * max(height, width) / 4096.0))))
        if underlap_px is None
        else int(underlap_px)
    )

    rgba_out: Dict[str, np.ndarray] = {}
    owner_out: Dict[str, np.ndarray] = {}
    layer_stats: Dict[str, Dict[str, Any]] = {}
    total_artist_hidden = 0
    total_auto_bridged = 0
    total_defringed = 0

    for order_idx, spec in enumerate(ordered):
        authored = np.asarray(artist_canvases[order_idx], dtype=np.uint8)
        del artist_canvases[order_idx]
        owned = (ownership == order_idx) & subject
        image = np.zeros_like(art)

        # 1. Genuine artist-painted hidden pixels under opaque front parts
        occluded = (ownership > order_idx) & (art[..., 3] == 255)
        artist_hidden = (authored[..., 3] > 30) & occluded
        image[artist_hidden] = authored[artist_hidden]
        artist_hidden_px = int(artist_hidden.sum())

        # 2. Exact final artwork pixels for every visible owned pixel
        image[owned] = art[owned]

        # 3. Auto-bridge any remaining occluded contact seams strictly behind opaque front layers
        image, bridged_px = _extend_occluded_underlap(image, occluded, pad_px)
        # Re-stamp owned pixels to guarantee exact artwork fidelity
        image[owned] = art[owned]

        # 4. General 4K-aware semi-transparent edge defringe (never alters opaque pixels or alpha)
        defringed_px = 0
        if defringe:
            image, defringed_px = defringe_rgba(image)

        rgba_out[spec.id] = image
        owner_out[spec.id] = owned
        total_artist_hidden += artist_hidden_px
        total_auto_bridged += bridged_px
        total_defringed += defringed_px
        layer_stats[spec.id] = {
            "draw_order": spec.draw_order,
            "kind": spec.kind,
            "psd_indices": list(spec.psd_indices),
            "owned_px": int(owned.sum()),
            "artist_hidden_px": artist_hidden_px,
            "auto_bridged_px": bridged_px,
            "defringed_px": defringed_px,
        }

    stats: Dict[str, Any] = {
        "method": "psd_artist_layer_split",
        "psd_source": str(psd_path),
        "image_size": [width, height],
        "subject_px": subject_px,
        "unclaimed_px": unclaimed_px,
        "unclaimed_ratio": round(unclaimed_ratio, 6),
        "unclaimed_solid_px": unclaimed_solid_px,
        "faint_psd_match_px": int(faint_assign.sum()),
        "boundary_assignment_px": boundary_assigned_px,
        "overlay_excluded_indices": sorted(overlay_set),
        "overlay_rebound_px": overlay_rebound_px,
        "underlap_px": pad_px,
        "total_artist_hidden_px": total_artist_hidden,
        "total_auto_bridged_px": total_auto_bridged,
        "total_defringed_px": total_defringed,
        "layers": layer_stats,
    }
    return PsdSplitResult(rgba=rgba_out, ownership=owner_out, stats=stats)


def write_psd_split_bundle(
    source_path: Path,
    psd_path: Path,
    part_specs: Sequence[PsdPartSpec],
    out_dir: Path,
    *,
    overlay_indices: Optional[Iterable[int]] = None,
    auto_exclude_overlays: bool = True,
    underlap_px: Optional[int] = None,
    aa_boundary_px: int = 9,
    max_unclaimed_ratio: float = 0.015,
    defringe: bool = True,
) -> Dict[str, Any]:
    """Run ``split_from_psd`` and write part PNGs, audit sheet, stats, and ``material.json``."""
    src = Path(source_path).resolve()
    psd = Path(psd_path).resolve()
    out = Path(out_dir).resolve()
    if out.exists():
        raise FileExistsError(f"Output directory already exists: {out}")

    with Image.open(src) as im:
        art = np.asarray(im.convert("RGBA"), dtype=np.uint8)
    h, w = art.shape[:2]
    result = split_from_psd(
        art,
        psd,
        part_specs,
        overlay_indices=overlay_indices,
        auto_exclude_overlays=auto_exclude_overlays,
        underlap_px=underlap_px,
        aa_boundary_px=aa_boundary_px,
        max_unclaimed_ratio=max_unclaimed_ratio,
        defringe=defringe,
    )

    out.mkdir(parents=True, exist_ok=False)
    source_sha = _sha256_file(src)
    ordered = sorted(part_specs, key=lambda s: s.draw_order)

    # Save ownership audit image (highlights any unclaimed subject pixels in red)
    claimed = np.zeros((h, w), dtype=bool)
    for mask in result.ownership.values():
        claimed |= mask
    unclaimed = (art[..., 3] > 0) & (~claimed)
    marked = art.copy()
    marked[unclaimed] = [255, 40, 50, 255]
    audit = Image.new("RGBA", (w, h), (125, 125, 125, 255))
    audit.alpha_composite(Image.fromarray(marked))
    audit.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
    audit.convert("RGB").save(out / "ownership-audit.png")

    pieces: List[Dict[str, Any]] = []
    for spec in ordered:
        layer_path = out / f"{spec.id}.png"
        owner_path = out / f"{spec.id}-owner.png"
        Image.fromarray(result.rgba[spec.id]).save(layer_path)
        Image.fromarray(result.ownership[spec.id].astype(np.uint8) * 255).save(owner_path)
        layer_sha = _sha256_file(layer_path)
        owner_sha = _sha256_file(owner_path)
        pivot = list(spec.pivot) if spec.pivot is not None else [w / 2.0, h / 2.0]
        pieces.append(
            {
                "id": spec.id,
                "module_id": spec.id,
                "label": spec.label or spec.id,
                "parent": spec.parent,
                "kind": spec.kind,
                "pivot": [float(pivot[0]), float(pivot[1])],
                "draw_order": spec.draw_order,
                "image": str(layer_path),
                "sha256": layer_sha,
                "provenance": {
                    "operation": "local-direct-split",
                    "source_asset_sha256": source_sha,
                    "result_asset_sha256": layer_sha,
                    "transform": [[1, 0, 0], [0, 1, 0]],
                    "local_derivation": {
                        "method": "psd_artist_layer_split",
                        "parent_aligned_sha256": source_sha,
                        "psd_source": str(psd),
                        "psd_indices": list(spec.psd_indices),
                        "owner_mask": str(owner_path),
                        "owner_mask_sha256": owner_sha,
                        "padding_px": result.stats["underlap_px"],
                    },
                },
            }
        )

    manifest = {
        "version": 1,
        "material_mode": "canvas_layers",
        "source_image": str(src),
        "source_sha256": source_sha,
        "image_size": [w, h],
        "pieces": pieces,
    }
    (out / "material.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "split-stats.json").write_text(json.dumps(result.stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"manifest": manifest, "stats": result.stats, "out_dir": str(out)}
