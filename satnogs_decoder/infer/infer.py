"""The shipped inference path: raw frames -> structural .ksy decoder.

frames -> features.position_features -> model (boundary + signed/enum heads)
-> Layout -> KsySpec (Phase-3 IR) -> .ksy text. Field ids are synthetic
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
        room = min(next_start - cursor, length - cursor)   # >= 1
        w = max(v for v in _VALID_WIDTHS if v <= room)      # widest valid <= room
        spans.append((cursor, cursor + w))
        cursor += w
    return spans


def _width_to_type(w: int, signed: bool) -> str:
    # boundaries_to_spans guarantees w in {1,2,4,8}.
    return f"{'s' if signed else 'u'}{w}"


def predict_layout(frames: list[bytes], model: InferModel) -> Layout:
    L = common_length(frames)
    feats = position_features(frames)
    boundary = model.predict_boundary(feats)
    spans = boundaries_to_spans(boundary, L)
    Xf = field_features(feats, spans)
    signed, is_enum = model.predict_field(Xf)
    layout: Layout = []
    for i, (s, e) in enumerate(spans):
        layout.append(FieldSpan(
            start=s, end=e, width=e - s,
            signed=bool(signed[i]), is_enum=bool(is_enum[i]),
            name=f"field_{i}",
        ))
    return layout


def layout_to_ksyspec(layout: Layout, sat_id: str, endian: str = "be") -> KsySpec:
    seq = [
        KsyField(id=f.name or f"field_{i}", type=_width_to_type(f.width, f.signed))
        for i, f in enumerate(layout)
    ]
    spec = KsySpec(id=sat_id, endian=endian, seq=seq, ks_version="0.10")
    spec.doc = field_block(spec)
    return spec


def infer_ksy(frames: list[bytes], model: InferModel, sat_id: str, endian: str = "be") -> str:
    layout = predict_layout(frames, model)
    return layout_to_ksyspec(layout, sat_id, endian).to_yaml()


def load_model(path: str | None = None) -> InferModel:
    return InferModel.load(str(path) if path else str(_MODEL_DIR))
