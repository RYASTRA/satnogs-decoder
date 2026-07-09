"""Assemble supervised training rows from the corpus. Single source of row
assembly, imported by both scripts/train.py and infer/eval.py so the
trained model and the held-out eval see identical features/labels."""

from __future__ import annotations

from typing import Any, Literal, overload

import numpy as np

from satnogs_decoder.infer import corpus
from satnogs_decoder.infer.features import common_length, position_features
from satnogs_decoder.infer.model import field_features


Rows = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
RowsWithWidth = tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]


@overload
def build_training_rows(conn: Any, norads: Any, *, include_width: Literal[True]) -> RowsWithWidth:
    ...


@overload
def build_training_rows(conn: Any, norads: Any, *, include_width: Literal[False] = False) -> Rows:
    ...


def build_training_rows(conn: Any, norads: Any, *, include_width: bool = False) -> Rows | RowsWithWidth:
    Xb, yb, Xf, ys, ye, Xw, yw = [], [], [], [], [], [], []
    for norad in norads:
        frames = corpus.query_frames(conn, norad)
        layout = corpus.query_layout(conn, norad)
        if not frames or not layout:
            continue
        L = common_length(frames)
        feats = position_features(frames)  # (L, F)
        # ONE definition of "a field this sat has": in-frame and non-degenerate.
        # Boundary labels and field rows must derive from the same set.
        valid = [f for f in layout if f.start < L and f.end > f.start]
        true_starts = {f.start for f in valid}
        yb.extend(1 if i in true_starts else 0 for i in range(L))
        Xb.append(feats)
        spans = [(f.start, min(f.end, L)) for f in valid]
        Xf.append(field_features(feats, spans))
        ys.extend(int(f.signed) for f in valid)
        ye.extend(int(f.is_enum) for f in valid)
        if include_width:
            Xw.append(field_features(feats, spans, include_width=False))
            yw.extend(min(f.end, L) - f.start for f in valid)
    if not Xb:
        raise ValueError("no usable sats in corpus (all empty or fully filtered)")
    rows = (
        np.vstack(Xb),
        np.array(yb),
        np.vstack(Xf),
        np.array(ys),
        np.array(ye),
    )
    if include_width:
        return (*rows, np.vstack(Xw), np.array(yw))
    return rows
