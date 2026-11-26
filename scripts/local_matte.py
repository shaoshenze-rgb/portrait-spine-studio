#!/usr/bin/env python3
"""Local portrait background matting CLI (transparent passthrough or offline matting)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.pipeline.local_matte import matte_portrait_file


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Source portrait PNG/JPG")
    parser.add_argument("--out", type=Path, required=True, help="Output transparent RGBA PNG")
    parser.add_argument("--check-out", type=Path, default=None, help="Optional 3-panel check JPG")
    parser.add_argument("--fg-box", type=str, default=None, help="Optional x1,y1,x2,y2 (px or 0..1)")
    parser.add_argument("--sam-weight", type=Path, default=None, help="Optional SAM 2 weight path")
    parser.add_argument("--no-sam", action="store_true", help="Disable SAM 2 even if installed")
    args = parser.parse_args(argv)

    report = matte_portrait_file(
        args.input,
        args.out,
        check_out=args.check_out,
        fg_box=args.fg_box,
        sam_weight=args.sam_weight,
        prefer_sam=not args.no_sam,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
