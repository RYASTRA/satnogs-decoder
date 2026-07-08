"""Assemble supervised training rows from the corpus. Single source of row
assembly, imported by both scripts/train.py and infer/eval.py so the
trained model and the held-out eval see identical features/labels."""
from __future__ import annotations

import numpy as np

from satnogs_decoder.infer import corpus
from satnogs_decoder.infer.features import common_length, position_features
from satnogs_decoder.infer.model import field_features


def build_training_rows(conn, norads):
    Xb, yb, Xf, ys, ye = [], [], [], [], []
    for norad in norads:
        frames = corpus.query_frames(conn, norad)
        layout = corpus.query_layout(conn, norad)
        if not frames or not layout:
            continue
        L = common_length(frames)
        feats = position_features(frames)          # (L, F)
        true_starts = {f.start for f in layout if f.start < L}
        yb.extend(1 if i in true_starts else 0 for i in range(L))
        Xb.append(feats)
        spans = [(f.start, min(f.end, L)) for f in layout if f.start < L and f.end > f.start]
        Xf.append(field_features(feats, spans))
        ys.extend(int(f.signed) for f in layout if f.start < L and f.end > f.start)
        ye.extend(int(f.is_enum) for f in layout if f.start < L and f.end > f.start)
    return (
        np.vstack(Xb), np.array(yb),
        np.vstack(Xf), np.array(ys), np.array(ye),
    )
