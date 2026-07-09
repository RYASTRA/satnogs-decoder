"""CLI driver: turn a v1 field-table spec into complete SatNOGS .ksy text.

Usage
-----
    python scripts/generate.py --spec path/to/spec.yaml
    python scripts/generate.py --spec path/to/spec.yaml --out path/to/out.ksy
    python scripts/generate.py --spec path/to/spec.yaml --validate --anchor grbalpha

With ``--out`` the generated .ksy text is written to that path; otherwise it
is printed to stdout. With ``--validate`` the generated decoder is compiled
and run against the anchor's live frame corpus (window read from
``tests/fixtures/anchors.json``, matched on ``module``) and the resulting
``ValidationReport`` is printed as JSON to stdout. Mirrors
``scripts/validate.py`` / ``scripts/build_corpus.py``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys

from satnogs_decoder.generate import generate
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
        description="Generate a complete SatNOGS .ksy decoder from a v1 field-table spec."
    )
    ap.add_argument(
        "--spec",
        required=True,
        metavar="PATH",
        help="Path to the v1 field-table spec YAML.",
    )
    ap.add_argument(
        "--out",
        default=None,
        metavar="PATH",
        help="Write the generated .ksy text here (default: stdout).",
    )
    ap.add_argument(
        "--validate",
        action="store_true",
        help="Compile the generated decoder and validate it against --anchor's frame corpus.",
    )
    ap.add_argument(
        "--anchor",
        default=None,
        metavar="MODULE",
        help="Anchor module name (e.g. grbalpha). Required with --validate; "
        "must match an entry in anchors.json.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Maximum frames to fetch when validating (default: 200).",
    )
    args = ap.parse_args()

    if args.validate and not args.anchor:
        ap.error("--validate requires --anchor MODULE")

    ksy_text = generate(pathlib.Path(args.spec))

    if args.out:
        pathlib.Path(args.out).write_text(ksy_text)
    else:
        print(ksy_text)

    if args.validate:
        anchors = json.loads(ANCHORS_PATH.read_text())
        matches = [a for a in anchors if a["module"] == args.anchor]
        if not matches:
            sys.exit(
                f"No anchor named '{args.anchor}' in anchors.json. "
                f"Available: {[a['module'] for a in anchors]}"
            )
        anchor = matches[0]

        parser_cls = compile_ksy(ksy_text)

        frames = fetch_frames(
            anchor["norad"],
            start=anchor["start"],
            end=anchor["end"],
            limit=args.limit,
        )
        print(
            f"Fetched {len(frames)} frames for {anchor['name']} (NORAD {anchor['norad']})",
            file=sys.stderr,
        )

        report = validate(parser_cls, frames, ref_module=anchor["module"])

        print(json.dumps(_report_to_dict(report), indent=2))


if __name__ == "__main__":
    main()
