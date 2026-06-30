"""CLI driver: validate a .ksy parser against a SatNOGS anchor's live frame corpus.

Usage
-----
    python scripts/validate.py --ksy path/to/my.ksy --anchor grbalpha

Anchors are read from tests/fixtures/anchors.json.  The ``--anchor`` argument
matches on the ``module`` field.  Frames are fetched with the anchor's date
window (``start``/``end``) so unbounded queries are avoided.

The full ``ValidationReport`` is printed as JSON to stdout.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys

from satnogs_decoder.shared.kaitai import compile_ksy
from satnogs_decoder.shared.satnogs_db import fetch_frames
from satnogs_decoder.validate.engine import validate

ANCHORS_PATH = pathlib.Path(__file__).parents[1] / "tests" / "fixtures" / "anchors.json"


def _report_to_dict(obj: object) -> object:
    """Recursively convert dataclass instances to plain dicts for JSON serialisation."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _report_to_dict(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, list):
        return [_report_to_dict(i) for i in obj]
    return obj


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Validate a .ksy parser against a windowed SatNOGS frame corpus "
            "and print the ValidationReport as JSON."
        )
    )
    ap.add_argument(
        "--ksy",
        required=True,
        metavar="PATH",
        help="Path to the .ksy file to validate.",
    )
    ap.add_argument(
        "--anchor",
        required=True,
        metavar="MODULE",
        help="Anchor module name (e.g. grbalpha).  Must match an entry in anchors.json.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Maximum frames to fetch (default: 200).",
    )
    ap.add_argument(
        "--no-ref",
        action="store_true",
        help="Skip cross-check against the canonical reference decoder.",
    )
    ap.add_argument(
        "--import-dir",
        action="append",
        dest="import_dirs",
        metavar="DIR",
        default=[],
        help="Extra .ksy import search directory (may be repeated).",
    )
    args = ap.parse_args()

    anchors = json.loads(ANCHORS_PATH.read_text())
    matches = [a for a in anchors if a["module"] == args.anchor]
    if not matches:
        sys.exit(
            f"No anchor named '{args.anchor}' in anchors.json. "
            f"Available: {[a['module'] for a in anchors]}"
        )
    anchor = matches[0]

    ksy_path = pathlib.Path(args.ksy)
    ksy_text = ksy_path.read_text()

    # Default import dir: the directory containing the .ksy file.
    import_dirs: list[str] = list(args.import_dirs)
    if str(ksy_path.parent) not in import_dirs:
        import_dirs.insert(0, str(ksy_path.parent))

    parser_cls = compile_ksy(ksy_text, import_dirs=import_dirs)

    frames = fetch_frames(
        anchor["norad"],
        start=anchor["start"],
        end=anchor["end"],
        limit=args.limit,
    )
    print(f"Fetched {len(frames)} frames for {anchor['name']} (NORAD {anchor['norad']})", file=sys.stderr)

    ref_module: str | None = None if args.no_ref else anchor["module"]

    report = validate(parser_cls, frames, ref_module=ref_module)

    print(json.dumps(_report_to_dict(report), indent=2))


if __name__ == "__main__":
    main()
