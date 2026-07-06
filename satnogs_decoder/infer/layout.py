"""The shared field-layout vocabulary spoken by every infer module.

A `Layout` is an ordered list of `FieldSpan`s covering a frame's bytes
left-to-right. The same type carries both TRUE layouts (mined from a
canonical .ksy in `labels`) and INFERRED layouts (predicted in `infer`),
so truth and prediction are always comparable field-for-field.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FieldSpan:
    start: int          # inclusive byte offset of the field's first byte
    end: int            # exclusive byte offset just past the field
    width: int          # end - start (in bytes)
    signed: bool        # signed integer interpretation
    is_enum: bool       # low-cardinality / fixed-code candidate
    name: str | None = None   # synthetic id at inference; declared id in truth

    def __post_init__(self) -> None:
        if self.width != self.end - self.start:
            raise ValueError(
                f"FieldSpan width={self.width} != end-start={self.end - self.start}"
            )


Layout = list  # list[FieldSpan]; alias kept loose to avoid runtime generics churn
