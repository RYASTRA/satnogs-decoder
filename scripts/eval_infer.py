# scripts/eval_infer.py — TEMPORARY. Emits the held-out scoreboard.
# Run in-container: docker compose run --rm app python -m scripts.eval_infer
from __future__ import annotations

import argparse
import json
import sys

from satnogs_decoder.infer import corpus
from satnogs_decoder.infer.eval import evaluate_holdout
from satnogs_decoder.infer.model import InferModel
from satnogs_decoder.infer.review import build_review_report, model_artifacts, write_json

DB = "corpus/satnogs_corpus.db"
MODEL_DIR = "satnogs_decoder/infer/model"


DEGRADATION_CAPS = [30, 50, 100, 200, None]  # None = all available frames


def main() -> int:
    parser = argparse.ArgumentParser(description="Emit the held-out inference scoreboard.")
    parser.add_argument("--db", default=DB, help=f"Corpus DB path (default: {DB}).")
    parser.add_argument(
        "--write-report",
        default=None,
        help="Optional path for the full JSON review report.",
    )
    parser.add_argument(
        "--write-model-metadata",
        default=None,
        help="Optional path for tracked model metadata JSON.",
    )
    args = parser.parse_args()

    conn = corpus.open_corpus(args.db)
    if not corpus.list_norads(conn):
        print("empty corpus", file=sys.stderr)
        return 1
    scores = evaluate_holdout(conn, InferModel, include_per_sat=True)
    per_sat = scores.pop("per_sat", [])
    print(json.dumps(scores, indent=2))
    print(json.dumps({"per_sat": per_sat}, indent=2))
    print(
        f"\nLearned boundary F1={scores['boundary_f1']:.3f} vs "
        f"u8-baseline={scores['baseline_u8_f1']:.3f}, "
        f"entropy-baseline={scores['baseline_entropy_f1']:.3f}",
        file=sys.stderr,
    )
    print(
        f"Exact span F1={scores['span_f1']:.3f}, "
        f"width_all={scores['width_acc_all']:.3f}, "
        f"field_count_mae={scores['field_count_mae']:.3f}",
        file=sys.stderr,
    )
    # Degradation curve (spec §9): boundary F1 vs frames-per-sat.
    print("\nframes/sat -> boundary_f1 (degradation curve):", file=sys.stderr)
    curve: dict[str, float] = {}
    for cap in DEGRADATION_CAPS:
        s = evaluate_holdout(conn, InferModel, max_frames=cap)
        curve[str(cap if cap is not None else "all")] = round(s["boundary_f1"], 3)
        print(
            f"  {cap if cap is not None else 'all':>4} -> {s['boundary_f1']:.3f}", file=sys.stderr
        )
    print(json.dumps({"degradation_curve": curve}, indent=2))
    report = build_review_report(
        {**scores, "per_sat": per_sat},
        degradation_curve=curve,
        corpus=corpus.corpus_summary(conn),
        model=model_artifacts(MODEL_DIR),
    )
    if args.write_report:
        write_json(args.write_report, report)
        print(f"wrote review report -> {args.write_report}", file=sys.stderr)
    if args.write_model_metadata:
        write_json(args.write_model_metadata, report)
        print(f"wrote model metadata -> {args.write_model_metadata}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
