"""CLI driver: fetch raw frames for each anchor satellite and build a HF Dataset.

Usage
-----
    python scripts/build_corpus.py --limit 200
    python scripts/build_corpus.py --limit 200 --push owner/my-dataset

Anchors are read from tests/fixtures/anchors.json.  Each anchor must supply
``norad``, ``module``, ``start``, and ``end`` (ISO-8601 strings) so that
fetch_frames can issue a bounded date-window query (unbounded queries time out
for high-volume satellites).
"""

from __future__ import annotations

import argparse
import json
import pathlib

from satnogs_decoder.shared.satnogs_db import fetch_frames
from satnogs_decoder.data.build_dataset import build_dataset, push

ANCHORS_PATH = pathlib.Path(__file__).parents[1] / "tests" / "fixtures" / "anchors.json"


def build(
    anchors: list[dict],
    *,
    limit: int,
    token: str | None = None,
):
    """Fetch frames for each anchor and assemble them into a HF Dataset.

    Parameters
    ----------
    anchors:
        List of anchor dicts, each with keys ``norad``, ``module``,
        ``start``, and ``end``.
    limit:
        Maximum number of frames to fetch per satellite.
    token:
        SatNOGS DB API token.  Falls back to the ``satnogs_db_api_key``
        environment variable when None.

    Returns
    -------
    datasets.Dataset
    """
    frames_by_norad = {
        a["norad"]: fetch_frames(
            a["norad"],
            start=a["start"],
            end=a["end"],
            token=token,
            limit=limit,
        )
        for a in anchors
    }
    modules = {a["norad"]: a["module"] for a in anchors}
    return build_dataset(frames_by_norad, modules)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fetch SatNOGS frames for anchor satellites and build a HF Dataset."
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Maximum frames to fetch per satellite (default: 200).",
    )
    ap.add_argument(
        "--push",
        default=None,
        metavar="REPO_ID",
        help="Hugging Face dataset repo_id to push to (e.g. owner/my-dataset).",
    )
    args = ap.parse_args()

    anchors = json.loads(ANCHORS_PATH.read_text())
    ds = build(anchors, limit=args.limit)
    print(f"Built {ds.num_rows} frames")

    if args.push:
        push(ds, args.push)
        print(f"Pushed to {args.push}")


if __name__ == "__main__":
    main()
