# scripts/build_corpus_db.py — TEMPORARY. Run as a MODULE:
# docker compose run --rm app python -m scripts.build_corpus_db
from __future__ import annotations

import json
import pathlib
import sys

import requests
from scripts.audit_harness import RAW
from satnogs_decoder.infer import corpus
from satnogs_decoder.infer.qualify import qualify, qualify_frames
from satnogs_decoder.shared.satnogs_db import fetch_frames

SATS = pathlib.Path("tests/infer/corpus_sats.json")
DB = "corpus/satnogs_corpus.db"
PROBE_CAP, FULL_CAP = 50, 500


def main() -> int:
    pathlib.Path("corpus").mkdir(exist_ok=True)
    sats = json.loads(SATS.read_text())
    conn = corpus.open_corpus(DB)
    kept, dropped = 0, 0
    for idx, s in enumerate(sats, start=1):
        module, norad = s["module"], s["norad"]
        print(f"[{idx}/{len(sats)}] probe {module} (norad {norad})", file=sys.stderr, flush=True)
        try:
            probe = fetch_frames(norad, start=s["start"], end=s["end"], limit=PROBE_CAP)
        except Exception as e:  # noqa: BLE001
            print(f"drop {module}: fetch failed: {e}", file=sys.stderr, flush=True)
            dropped += 1
            continue
        if len(probe) < 5:
            print(f"drop {module}: only {len(probe)} probe frames", file=sys.stderr, flush=True)
            dropped += 1
            continue
        try:
            text = requests.get(RAW + f"{module}.ksy", timeout=60).text
        except Exception as e:  # noqa: BLE001
            print(f"drop {module}: ksy fetch failed: {e}", file=sys.stderr, flush=True)
            dropped += 1
            continue
        layout, reason = qualify(text, [f.data for f in probe])
        if layout is None:
            print(f"drop {module}: {reason}", file=sys.stderr, flush=True)
            dropped += 1
            continue
        # Full pull, then keep only frames matching the dominant parsed layout.
        print(f"[{idx}/{len(sats)}] full pull {module}", file=sys.stderr, flush=True)
        try:
            full = fetch_frames(norad, start=s["start"], end=s["end"], limit=FULL_CAP)
        except Exception as e:  # noqa: BLE001
            print(f"drop {module}: full fetch failed: {e}", file=sys.stderr, flush=True)
            dropped += 1
            continue
        layout, keep_frames, reason = qualify_frames(text, [f.data for f in full])
        if layout is None:
            print(
                f"drop {module}: full qualification failed: {reason}",
                file=sys.stderr,
                flush=True,
            )
            dropped += 1
            continue
        modal = len(keep_frames[0])
        corpus.insert_frames(conn, norad, keep_frames)
        corpus.insert_layout(conn, norad, layout)
        kept += 1
        print(
            f"keep {module} (norad {norad}): "
            f"{len(keep_frames)} frames @ {modal}B, {len(layout)} fields",
            flush=True,
        )
    corpus.set_meta(conn, "corpus_version", "2")
    corpus.set_meta(conn, "n_sats", str(kept))
    print(f"\ncorpus built: {kept} kept, {dropped} dropped -> {DB}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
