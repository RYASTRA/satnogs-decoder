"""The shipped inference path: raw frames -> structural .ksy decoder.

frames -> features.position_features -> model (boundary + optional width +
signed/enum heads) -> Layout -> KsySpec (Phase-3 IR) -> .ksy text. Field ids are synthetic
(`field_0`, ...): only STRUCTURE is inferred, never names/units (spec §2).
"""

from __future__ import annotations

import pathlib

import numpy as np

from satnogs_decoder.generate.fields import field_block
from satnogs_decoder.infer.features import common_length, position_features
from satnogs_decoder.infer.layout import FieldSpan, Layout
from satnogs_decoder.infer.model import InferModel, field_features
from satnogs_decoder.shared.ksy import KsyField, KsySpec

_VALID_WIDTHS = (1, 2, 4, 8)
_MODEL_DIR = pathlib.Path(__file__).parent / "model"
_MIN_PROBA = 1e-9
_WIDTH_BOUNDARY_WEIGHT = 0.75


def boundaries_to_spans(boundary: np.ndarray, length: int) -> list[tuple[int, int]]:
    """Tile [0, length) with contiguous, gap-free spans of valid Kaitai scalar
    widths {1,2,4,8}, respecting predicted field starts.

    Position 0 is always a start. Walking a running cursor, each span takes the
    WIDEST valid width not exceeding the distance to the next predicted boundary
    (nor the frame end), so every boundary lands on a span start and the spans
    cover the whole frame with no gaps or overlaps. A predicted field whose
    width is not a single scalar width (e.g. 3) is TILED into several valid-width
    fields (e.g. 2 + 1) rather than dropping bytes — Kaitai has no u3.
    """
    starts = sorted({0} | {i for i in range(1, length) if boundary[i]})
    spans: list[tuple[int, int]] = []
    cursor = 0
    while cursor < length:
        next_start = next((s for s in starts if s > cursor), length)
        room = min(next_start - cursor, length - cursor)  # >= 1
        w = max(v for v in _VALID_WIDTHS if v <= room)  # widest valid <= room
        spans.append((cursor, cursor + w))
        cursor += w
    return spans


def _width_model_candidates(model: InferModel, remaining: int) -> list[int]:
    classes = {w for w in model.width_classes() if 0 < w <= remaining}
    if not classes:
        classes = {w for w in _VALID_WIDTHS if w <= remaining}
    if not classes:
        classes = {remaining}
    return sorted(classes)


def width_model_spans(
    pos_feats: np.ndarray,
    boundary_proba: np.ndarray,
    model: InferModel,
) -> list[tuple[int, int]]:
    """Select gap-free spans with a trained width head.

    The boundary head is still useful, but as a soft end-of-span signal rather
    than a hard split point. This lets the model ignore low-value interior
    boundary blips that previously forced over-segmentation.
    """
    length = len(pos_feats)
    if length == 0:
        return []
    class_to_idx = {w: i for i, w in enumerate(model.width_classes())}
    dp = np.full(length + 1, -np.inf, dtype=np.float64)
    choice = np.zeros(length, dtype=int)
    dp[length] = 0.0

    for pos in range(length - 1, -1, -1):
        candidates = _width_model_candidates(model, length - pos)
        spans = [(pos, pos + w) for w in candidates]
        width_scores = model.predict_width_proba(
            field_features(pos_feats, spans, include_width=False)
        )
        best_score = -np.inf
        best_width = candidates[0]
        for row_idx, width in enumerate(candidates):
            end = pos + width
            width_idx = class_to_idx.get(width)
            width_proba = (
                float(width_scores[row_idx, width_idx])
                if width_idx is not None
                else _MIN_PROBA
            )
            boundary = 1.0 if end == length else float(boundary_proba[end])
            score = (
                np.log(max(width_proba, _MIN_PROBA))
                + _WIDTH_BOUNDARY_WEIGHT * np.log(max(boundary, _MIN_PROBA))
                + dp[end]
            )
            if score > best_score:
                best_score = score
                best_width = width
        dp[pos] = best_score
        choice[pos] = best_width

    spans: list[tuple[int, int]] = []
    cursor = 0
    while cursor < length:
        width = int(choice[cursor])
        if width <= 0:
            width = 1
        end = min(cursor + width, length)
        spans.append((cursor, end))
        cursor = end
    return spans


def _width_to_type(w: int, signed: bool) -> str:
    # boundaries_to_spans guarantees w in {1,2,4,8}.
    return f"{'s' if signed else 'u'}{w}"


def predict_layout(frames: list[bytes], model: InferModel) -> Layout:
    L = common_length(frames)
    feats = position_features(frames)
    if model.has_width_head:
        spans = width_model_spans(feats, model.predict_boundary_proba(feats), model)
    else:
        boundary = model.predict_boundary(feats)
        spans = boundaries_to_spans(boundary, L)
    Xf = field_features(feats, spans)
    signed, is_enum = model.predict_field(Xf)
    layout: Layout = []
    for i, (s, e) in enumerate(spans):
        layout.append(
            FieldSpan(
                start=s,
                end=e,
                width=e - s,
                signed=bool(signed[i]),
                is_enum=bool(is_enum[i]),
                name=f"field_{i}",
            )
        )
    return layout


def layout_to_ksyspec(layout: Layout, sat_id: str, endian: str = "be") -> KsySpec:
    seq: list[KsyField] = []
    for i, f in enumerate(layout):
        field_id = f.name or f"field_{i}"
        if f.width in _VALID_WIDTHS:
            seq.append(KsyField(id=field_id, type=_width_to_type(f.width, f.signed)))
        else:
            seq.append(KsyField(id=field_id, size=f.width, doc="raw bytes"))
    spec = KsySpec(id=sat_id, endian=endian, seq=seq, ks_version="0.10")
    spec.doc = field_block(spec)
    return spec


def infer_ksy(frames: list[bytes], model: InferModel, sat_id: str, endian: str = "be") -> str:
    layout = predict_layout(frames, model)
    return layout_to_ksyspec(layout, sat_id, endian).to_yaml()


def load_model(path: str | None = None) -> InferModel:
    return InferModel.load(str(path) if path else str(_MODEL_DIR))
