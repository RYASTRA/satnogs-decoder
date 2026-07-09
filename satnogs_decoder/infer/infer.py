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
    """Convert predicted starts into contiguous, gap-free field spans.

    Position 0 is always a start. Each predicted start opens a field that runs
    until the next predicted start or the frame end. The generated KSY layer can
    represent non-scalar widths as fixed-size byte fields, so the inference
    layer should not invent extra internal boundaries just to tile with scalar
    integer widths.
    """
    starts = sorted({0} | {i for i in range(1, length) if boundary[i]})
    if length == 0:
        return []
    return [(start, end) for start, end in zip(starts, [*starts[1:], length]) if end > start]


def _width_to_type(w: int, signed: bool) -> str:
    # Only called for Kaitai scalar widths.
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
