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


def _snap_width(w: int) -> int:
    return min(_VALID_WIDTHS, key=lambda v: (abs(v - w), v))


def boundaries_to_spans(boundary: np.ndarray, length: int) -> list[tuple[int, int]]:
    starts = [i for i in range(length) if (i == 0 or boundary[i])]
    starts = sorted(set(starts))
    spans: list[tuple[int, int]] = []
    cursor = 0
    for idx, s in enumerate(starts):
        nxt = starts[idx + 1] if idx + 1 < len(starts) else length
        if s < cursor:  # previous snap overran this start; skip it
            continue
        raw_w = nxt - s
        w = _snap_width(raw_w)
        w = min(w, length - s)               # never run past the frame
        w = _snap_width(w) if w in _VALID_WIDTHS else max(1, w)
        spans.append((s, s + w))
        cursor = s + w
    return spans


def _width_to_type(w: int, signed: bool) -> str:
    if w in (1, 2, 4, 8):
        return f"{'s' if signed else 'u'}{w}"
    return f"b{w * 8}"  # fallback: raw bit field (should not occur after snapping)


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
