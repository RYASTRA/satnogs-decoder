# scripts/infer.py — SHIPS. Frozen model + a satellite's frames -> a structural .ksy.
# Run in-container: docker compose run --rm app python scripts/infer.py --norad 47959 \
#   --start 2022-06-15T00:00:00Z --end 2022-06-15T06:00:00Z --validate
from __future__ import annotations
import argparse
import dataclasses
import json
import pathlib
import sys

from satnogs_decoder.infer import infer_ksy, load_model
from satnogs_decoder.shared.kaitai import compile_ksy
from satnogs_decoder.shared.satnogs_db import fetch_frames
from satnogs_decoder.validate.engine import validate


def _to_dict(obj):
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_dict(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, list):
        return [_to_dict(i) for i in obj]
    return obj


def main() -> None:
    ap = argparse.ArgumentParser(description="Infer a structural .ksy decoder from raw frames.")
    ap.add_argument("--norad", type=int, required=True)
    ap.add_argument("--start", required=True, metavar="ISO")
    ap.add_argument("--end", required=True, metavar="ISO")
    ap.add_argument("--out", default=None, metavar="PATH", help="Write .ksy here (default stdout).")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--endian", default="be", choices=("be", "le"))
    ap.add_argument("--validate", action="store_true",
                    help="Compile the inferred decoder and report coverage over the same frames.")
    args = ap.parse_args()

    frames = fetch_frames(args.norad, start=args.start, end=args.end, limit=args.limit)
    print(f"Fetched {len(frames)} frames for NORAD {args.norad}", file=sys.stderr)
    if not frames:
        sys.exit("No frames in that window.")

    ksy = infer_ksy([f.data for f in frames], load_model(), f"sat_{args.norad}", endian=args.endian)
    if args.out:
        pathlib.Path(args.out).write_text(ksy)
    else:
        print(ksy)

    if args.validate:
        report = validate(compile_ksy(ksy), frames)
        print(json.dumps(_to_dict(report), indent=2))


if __name__ == "__main__":
    main()
