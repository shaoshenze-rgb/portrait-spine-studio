#!/usr/bin/env python3
"""Inspect or split a multi-layer artist PSD directly into standard Spine layer parts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.psd_layer_split import PsdPartSpec, inspect_psd_layers, write_psd_split_bundle


def _parse_size(raw: str | None) -> tuple[int, int] | None:
    if not raw:
        return None
    parts = [int(x.strip()) for x in raw.replace("x", ",").split(",") if x.strip()]
    if len(parts) != 2 or parts[0] <= 0 or parts[1] <= 0:
        raise argparse.ArgumentTypeError("Expected W,H or WxH positive integers")
    return (parts[0], parts[1])


def _cmd_inspect(args: argparse.Namespace) -> int:
    size = _parse_size(args.size)
    report = inspect_psd_layers(
        args.psd,
        canvas_size=size,
        contact_sheet_path=args.sheet,
    )
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "psd_source": report["psd_source"],
        "psd_size": report["psd_size"],
        "canvas_size": report["canvas_size"],
        "usable_multi_layer": report["usable_multi_layer"],
        "counts": report["counts"],
        "contact_sheet": report.get("contact_sheet"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _cmd_split(args: argparse.Namespace) -> int:
    spec_data = json.loads(args.spec.read_text(encoding="utf-8"))
    if spec_data.get("version") != 1 or not isinstance(spec_data.get("parts"), list) or not spec_data["parts"]:
        raise SystemExit("Expected version=1 and a non-empty parts list in --spec JSON")

    part_specs = []
    for idx, row in enumerate(spec_data["parts"]):
        pivot_raw = row.get("pivot")
        pivot = (float(pivot_raw[0]), float(pivot_raw[1])) if isinstance(pivot_raw, list) and len(pivot_raw) == 2 else None
        part_specs.append(
            PsdPartSpec(
                id=str(row["id"]),
                psd_indices=tuple(int(i) for i in row["psd_indices"]),
                draw_order=int(row.get("draw_order", idx)),
                kind=str(row.get("kind", "rigid")),
                label=row.get("label"),
                role=row.get("role"),
                parent=row.get("parent"),
                pivot=pivot,
            )
        )

    bundle = write_psd_split_bundle(
        args.body,
        args.psd,
        part_specs,
        args.out_dir,
        overlay_indices=spec_data.get("overlay_indices"),
        auto_exclude_overlays=bool(spec_data.get("auto_exclude_overlays", True)),
        underlap_px=spec_data.get("underlap_px"),
        aa_boundary_px=int(spec_data.get("aa_boundary_px", 9)),
        max_unclaimed_ratio=float(spec_data.get("max_unclaimed_ratio", 0.015)),
        defringe=bool(spec_data.get("defringe", True)),
    )
    print(json.dumps(bundle["stats"], ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ins = sub.add_parser("inspect", help="Inspect PSD layer tree and optionally render an indexed sheet")
    p_ins.add_argument("--psd", type=Path, required=True, help="Source PSD file (read-only)")
    p_ins.add_argument("--size", type=str, default=None, help="Optional target canvas size WxH or W,H")
    p_ins.add_argument("--out-json", type=Path, default=None, help="Output JSON path for indexed layers")
    p_ins.add_argument("--sheet", type=Path, default=None, help="Optional indexed visual contact sheet JPG")
    p_ins.set_defaults(func=_cmd_inspect)

    p_spl = sub.add_parser("split", help="Split transparent body portrait using PSD layer mapping spec")
    p_spl.add_argument("--body", type=Path, required=True, help="Aligned transparent portrait PNG (matted.png)")
    p_spl.add_argument("--psd", type=Path, required=True, help="Sibling source PSD file (read-only)")
    p_spl.add_argument("--spec", type=Path, required=True, help="JSON mapping spec (version=1, parts=[...])")
    p_spl.add_argument("--out-dir", type=Path, required=True, help="New output directory")
    p_spl.set_defaults(func=_cmd_split)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
