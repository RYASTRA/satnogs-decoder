# scripts/eval_infer.py — TEMPORARY. Emits the held-out scoreboard.
# Run in-container: docker compose run --rm app python -m scripts.eval_infer
from __future__ import annotations
import json
import sys
from satnogs_decoder.infer import corpus
from satnogs_decoder.infer.eval import evaluate_holdout
from satnogs_decoder.infer.model import InferModel

DB = "corpus/satnogs_corpus.db"


DEGRADATION_CAPS = [30, 50, 100, 200, None]  # None = all available frames


def main() -> int:
    conn = corpus.open_corpus(DB)
    if not corpus.list_norads(conn):
        print("empty corpus", file=sys.stderr)
        return 1
    scores = evaluate_holdout(conn, InferModel)
    print(json.dumps(scores, indent=2))
    print(
        f"\nLearned boundary F1={scores['boundary_f1']:.3f} vs "
        f"u8-baseline={scores['baseline_u8_f1']:.3f}, "
        f"entropy-baseline={scores['baseline_entropy_f1']:.3f}",
        file=sys.stderr,
    )
    # Degradation curve (spec §9): boundary F1 vs frames-per-sat.
    print("\nframes/sat -> boundary_f1 (degradation curve):", file=sys.stderr)
    curve = {}
    for cap in DEGRADATION_CAPS:
        s = evaluate_holdout(conn, InferModel, max_frames=cap)
        curve[cap if cap is not None else "all"] = round(s["boundary_f1"], 3)
        print(f"  {cap if cap is not None else 'all':>4} -> {s['boundary_f1']:.3f}", file=sys.stderr)
    print(json.dumps({"degradation_curve": curve}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
