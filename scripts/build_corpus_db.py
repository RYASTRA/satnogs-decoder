# scripts/build_corpus_db.py — TEMPORARY. Run as a MODULE:
# docker compose run --rm app python -m scripts.build_corpus_db
from __future__ import annotations
import json, pathlib, sys
import requests
from scripts.audit_harness import RAW
from satnogs_decoder.infer import corpus
from satnogs_decoder.infer.qualify import qualify, modal_length
from satnogs_decoder.shared.satnogs_db import fetch_frames

SATS = pathlib.Path("tests/infer/corpus_sats.json")
DB = "corpus/satnogs_corpus.db"
PROBE_CAP, FULL_CAP = 50, 500


def main() -> int:
    pathlib.Path("corpus").mkdir(exist_ok=True)
    sats = json.loads(SATS.read_text())
    conn = corpus.open_corpus(DB)
    kept, dropped = 0, 0
    for s in sats:
        module, norad = s["module"], s["norad"]
        try:
            probe = fetch_frames(norad, start=s["start"], end=s["end"], limit=PROBE_CAP)
        except Exception as e:  # noqa: BLE001
            print(f"drop {module}: fetch failed: {e}", file=sys.stderr); dropped += 1; continue
        if len(probe) < 5:
            print(f"drop {module}: only {len(probe)} probe frames", file=sys.stderr); dropped += 1; continue
        try:
            text = requests.get(RAW + f"{module}.ksy", timeout=60).text
        except Exception as e:  # noqa: BLE001
            print(f"drop {module}: ksy fetch failed: {e}", file=sys.stderr); dropped += 1; continue
        layout, reason = qualify(text, [f.data for f in probe])
        if layout is None:
            print(f"drop {module}: {reason}", file=sys.stderr); dropped += 1; continue
        # full pull, keep only the dominant (modal) length frames
        try:
            full = fetch_frames(norad, start=s["start"], end=s["end"], limit=FULL_CAP)
        except Exception as e:  # noqa: BLE001
            print(f"drop {module}: full fetch failed: {e}", file=sys.stderr); dropped += 1; continue
        modal, _ = modal_length([f.data for f in full])
        keep_frames = [f.data for f in full if len(f.data) == modal]
        corpus.insert_frames(conn, norad, keep_frames)
        corpus.insert_layout(conn, norad, layout)
        kept += 1
        print(f"keep {module} (norad {norad}): {len(keep_frames)} frames @ {modal}B, {len(layout)} fields")
    corpus.set_meta(conn, "corpus_version", "2")
    corpus.set_meta(conn, "n_sats", str(kept))
    print(f"\ncorpus built: {kept} kept, {dropped} dropped -> {DB}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
