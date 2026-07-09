"""Held-out evaluation — the scoreboard (spec §9).

All metrics compare a TRUE Layout (from the canonical .ksy) against a
PREDICTED Layout on satellites the model never trained on. Boundary metrics
are on field-START positions; width/sign accuracy are conditioned on a
correctly-located start (you can only judge a field's width once you found
where it begins).
"""

from __future__ import annotations

from typing import Any

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


def exact_span_prf(true: Layout, pred: Layout) -> tuple[float, float, float]:
    t = {(f.start, f.end) for f in _byte_fields(true)}
    p = {(f.start, f.end) for f in _byte_fields(pred)}
    if not p:
        return 0.0, 0.0, 0.0
    tp = len(t & p)
    precision = tp / len(p)
    recall = tp / len(t) if t else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def width_accuracy_all(true: Layout, pred: Layout) -> float:
    true_fields = _byte_fields(true)
    if not true_fields:
        return 0.0
    pw = {f.start: f.width for f in _byte_fields(pred)}
    return sum(1 for f in true_fields if pw.get(f.start) == f.width) / len(true_fields)


def enum_prf(true: Layout, pred: Layout) -> tuple[float, float, float]:
    t = {f.start for f in _byte_fields(true) if f.is_enum}
    p = {f.start for f in _byte_fields(pred) if f.is_enum}
    if not p:
        return 0.0, 0.0, 0.0
    tp = len(t & p)
    precision = tp / len(p)
    recall = tp / len(t) if t else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def segmentation_rates(true: Layout, pred: Layout) -> tuple[float, float]:
    """Return over- and under-segmentation rates.

    Over-segmentation: fraction of true fields split by an interior predicted
    boundary. Under-segmentation: fraction of predicted fields that skip over an
    interior true boundary.
    """
    true_fields = _byte_fields(true)
    pred_fields = _byte_fields(pred)
    pred_starts = _starts(pred)
    true_starts = _starts(true)
    over = sum(1 for f in true_fields if any(f.start < s < f.end for s in pred_starts))
    under = sum(1 for f in pred_fields if any(f.start < s < f.end for s in true_starts))
    over_rate = over / len(true_fields) if true_fields else 0.0
    under_rate = under / len(pred_fields) if pred_fields else 0.0
    return over_rate, under_rate


def field_count_error(true: Layout, pred: Layout) -> int:
    return len(_byte_fields(pred)) - len(_byte_fields(true))


def _baseline_every_u8(length: int) -> Layout:
    from satnogs_decoder.infer.layout import FieldSpan

    return [FieldSpan(i, i + 1, 1, False, False) for i in range(length)]


def _baseline_entropy_threshold(
    pos_feats: np.ndarray, length: int, feat_names: list[str]
) -> Layout:
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


def evaluate_holdout(
    conn,
    model_factory,
    *,
    max_frames: int | None = None,
    include_per_sat: bool = False,
) -> dict[str, Any]:
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
    rows: dict[str, list[float]] = {
        "boundary_precision": [],
        "boundary_recall": [],
        "boundary_f1": [],
        "span_precision": [],
        "span_recall": [],
        "span_f1": [],
        "width_acc": [],
        "width_acc_all": [],
        "sign_acc": [],
        "enum_precision": [],
        "enum_recall": [],
        "enum_f1": [],
        "oversegmentation_rate": [],
        "undersegmentation_rate": [],
        "field_count_mae": [],
        "baseline_u8_f1": [],
        "baseline_entropy_f1": [],
    }
    per_sat: list[dict[str, float | int]] = []
    for held in norads:
        train_ns = [n for n in norads if n != held]
        if not train_ns:
            continue
        Xb, yb, Xf, ys, ye, Xw, yw = build_training_rows(conn, train_ns, include_width=True)
        model = model_factory()
        model.fit_boundary(Xb, yb)
        model.fit_field(Xf, ys, ye, X_width=Xw, y_width=yw)

        frames = corpus.query_frames(conn, held)
        if max_frames is not None:
            frames = frames[:max_frames]
        true = corpus.query_layout(conn, held)
        L = common_length(frames)
        pred = predict_layout(frames, model)
        boundary_p, boundary_r, boundary_f1 = boundary_prf(true, pred)
        span_p, span_r, span_f1 = exact_span_prf(true, pred)
        enum_p, enum_r, enum_f1 = enum_prf(true, pred)
        overseg, underseg = segmentation_rates(true, pred)
        width_acc = width_accuracy(true, pred)
        width_all = width_accuracy_all(true, pred)
        sign_acc = sign_accuracy(true, pred)
        count_error = field_count_error(true, pred)
        u8_f1 = boundary_prf(true, _baseline_every_u8(L))[2]
        feats = position_features(frames)
        entropy_f1 = boundary_prf(true, _baseline_entropy_threshold(feats, L, FEATURE_NAMES))[2]
        rows["boundary_precision"].append(boundary_p)
        rows["boundary_recall"].append(boundary_r)
        rows["boundary_f1"].append(boundary_f1)
        rows["span_precision"].append(span_p)
        rows["span_recall"].append(span_r)
        rows["span_f1"].append(span_f1)
        rows["width_acc"].append(width_acc)
        rows["width_acc_all"].append(width_all)
        rows["sign_acc"].append(sign_acc)
        rows["enum_precision"].append(enum_p)
        rows["enum_recall"].append(enum_r)
        rows["enum_f1"].append(enum_f1)
        rows["oversegmentation_rate"].append(overseg)
        rows["undersegmentation_rate"].append(underseg)
        rows["field_count_mae"].append(abs(float(count_error)))
        rows["baseline_u8_f1"].append(u8_f1)
        rows["baseline_entropy_f1"].append(entropy_f1)
        if include_per_sat:
            per_sat.append(
                {
                    "norad": held,
                    "n_frames": len(frames),
                    "frame_len": L,
                    "n_true_fields": len(_byte_fields(true)),
                    "n_pred_fields": len(_byte_fields(pred)),
                    "field_count_error": count_error,
                    "boundary_precision": boundary_p,
                    "boundary_recall": boundary_r,
                    "boundary_f1": boundary_f1,
                    "span_precision": span_p,
                    "span_recall": span_r,
                    "span_f1": span_f1,
                    "width_acc": width_acc,
                    "width_acc_all": width_all,
                    "sign_acc": sign_acc,
                    "enum_precision": enum_p,
                    "enum_recall": enum_r,
                    "enum_f1": enum_f1,
                    "oversegmentation_rate": overseg,
                    "undersegmentation_rate": underseg,
                    "baseline_u8_f1": u8_f1,
                    "baseline_entropy_f1": entropy_f1,
                }
            )
    out: dict[str, Any] = {k: (float(np.mean(v)) if v else 0.0) for k, v in rows.items()}
    if include_per_sat:
        out["per_sat"] = per_sat
    return out
