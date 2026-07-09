"""TEMPORARY trainer driver. Freezes the shipped model.
Deleted at finalization (Task 14).
"""

from __future__ import annotations

import sys

from satnogs_decoder.infer import corpus
from satnogs_decoder.infer.model import InferModel
from satnogs_decoder.infer.training import build_training_rows

DB = "corpus/satnogs_corpus.db"
MODEL_DIR = "satnogs_decoder/infer/model"


def main() -> int:
    conn = corpus.open_corpus(DB)
    norads = corpus.list_norads(conn)
    if not norads:
        print("empty corpus — run `python -m scripts.build_corpus_db` first", file=sys.stderr)
        return 1
    Xb, yb, Xf, ys, ye = build_training_rows(conn, norads)
    print(
        f"training rows: {Xb.shape[0]} positions ({yb.mean():.2%} boundaries), "
        f"{Xf.shape[0]} fields ({ys.mean():.2%} signed, {ye.mean():.2%} enum)"
    )
    model = InferModel()
    model.fit_boundary(Xb, yb)
    model.fit_field(Xf, ys, ye)
    model.save(MODEL_DIR)
    print(f"frozen model -> {MODEL_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
