"""Held-out evaluation — the scoreboard (spec §9).

All metrics compare a TRUE Layout (from the canonical .ksy) against a
PREDICTED Layout on satellites the model never trained on. Boundary metrics
are on field-START positions; width/sign accuracy are conditioned on a
correctly-located start (you can only judge a field's width once you found
where it begins).
"""
from __future__ import annotations

import numpy as np

from satnogs_decoder.infer.layout import Layout


def _byte_fields(layout: Layout) -> Layout:
    return [f for f in layout if f.end > f.start]


def _starts(layout: Layout) -> set[int]:
    return {f.start for f in _byte_fields(layout)}


def boundary_prf(true: Layout, pred: Layout) -> tuple[float, float, float]:
    t, p = _starts(true), _starts(pred)
    if not p:
        return 0.0, 0.0, 0.0
    tp = len(t & p)
    precision = tp / len(p)
    recall = tp / len(t) if t else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def width_accuracy(true: Layout, pred: Layout) -> float:
    tw = {f.start: f.width for f in _byte_fields(true)}
    hits = [f for f in _byte_fields(pred) if f.start in tw]
    if not hits:
        return 0.0
    return sum(1 for f in hits if f.width == tw[f.start]) / len(hits)


def sign_accuracy(true: Layout, pred: Layout) -> float:
    ts = {f.start: f.signed for f in _byte_fields(true)}
    hits = [f for f in _byte_fields(pred) if f.start in ts]
    if not hits:
        return 0.0
    return sum(1 for f in hits if f.signed == ts[f.start]) / len(hits)


def _baseline_every_u8(length: int) -> Layout:
    from satnogs_decoder.infer.layout import FieldSpan
    return [FieldSpan(i, i + 1, 1, False, False) for i in range(length)]


def _baseline_entropy_threshold(pos_feats: np.ndarray, length: int, feat_names: list[str]) -> Layout:
    """No learning: start a new field wherever entropy jumps above the median."""
    from satnogs_decoder.infer.layout import FieldSpan
    ent = pos_feats[:, feat_names.index("entropy")]
    thr = float(np.median(ent))
    spans, s = [], 0
    for i in range(1, length + 1):
        if i == length or ent[i] > thr:
            spans.append((s, i))
            s = i
    return [FieldSpan(a, b, b - a, False, False) for a, b in spans]


def evaluate_holdout(conn, model_factory, *, max_frames: int | None = None) -> dict:
    """Leave-one-sat-out CV. `model_factory()` returns a fresh untrained InferModel.

    `max_frames`, when set, caps the held-out sat's frame count before feature
    extraction — the driver sweeps it to trace the frames-per-sat degradation
    curve (spec §9: "how data-hungry it is").
    """
    from satnogs_decoder.infer import corpus
    from satnogs_decoder.infer.features import FEATURE_NAMES, common_length, position_features
    from satnogs_decoder.infer.infer import predict_layout
    from satnogs_decoder.infer.training import build_training_rows

    norads = corpus.list_norads(conn)
    rows = {"boundary_f1": [], "width_acc": [], "sign_acc": [],
            "baseline_u8_f1": [], "baseline_entropy_f1": []}
    for held in norads:
        train_ns = [n for n in norads if n != held]
        if not train_ns:
            continue
        Xb, yb, Xf, ys, ye = build_training_rows(conn, train_ns)
        model = model_factory()
        model.fit_boundary(Xb, yb)
        model.fit_field(Xf, ys, ye)

        frames = corpus.query_frames(conn, held)
        if max_frames is not None:
            frames = frames[:max_frames]
        true = corpus.query_layout(conn, held)
        L = common_length(frames)
        pred = predict_layout(frames, model)
        _, _, f1 = boundary_prf(true, pred)
        rows["boundary_f1"].append(f1)
        rows["width_acc"].append(width_accuracy(true, pred))
        rows["sign_acc"].append(sign_accuracy(true, pred))
        rows["baseline_u8_f1"].append(boundary_prf(true, _baseline_every_u8(L))[2])
        feats = position_features(frames)
        rows["baseline_entropy_f1"].append(
            boundary_prf(true, _baseline_entropy_threshold(feats, L, FEATURE_NAMES))[2])
    return {k: (float(np.mean(v)) if v else 0.0) for k, v in rows.items()}
