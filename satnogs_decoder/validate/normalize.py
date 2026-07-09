from __future__ import annotations

import enum
import math
import re
from dataclasses import dataclass, field
from numbers import Real
from typing import Mapping, cast


_INDEX_RE = re.compile(r"\[\d+\]")


@dataclass(frozen=True)
class NormalizedFields:
    values: dict[str, object]
    originals: dict[str, str]
    collisions: dict[str, list[str]] = field(default_factory=dict)


def normalize_key(name: str) -> str:
    """Normalize common Kaitai/SatNOGS field-path variants to a comparison key."""
    compact = _INDEX_RE.sub("", name)
    leaf = compact.split(".")[-1]
    while leaf.endswith("_raw"):
        leaf = leaf[:-4]
    return leaf.lower()


def normalize_fields(fields: Mapping[str, object]) -> NormalizedFields:
    values: dict[str, object] = {}
    originals: dict[str, str] = {}
    collisions: dict[str, list[str]] = {}
    for original, value in fields.items():
        key = normalize_key(original)
        if key in values:
            existing = originals[key]
            collisions.setdefault(key, [existing]).append(original)
            values.pop(key, None)
            originals.pop(key, None)
            continue
        if key in collisions:
            collisions[key].append(original)
            continue
        values[key] = value
        originals[key] = original
    return NormalizedFields(values=values, originals=originals, collisions=collisions)


def comparable_value(value: object) -> object:
    if isinstance(value, enum.Enum):
        return value.value
    return value


def values_equal(left: object, right: object, *, abs_tol: float = 1e-9) -> bool:
    a = comparable_value(left)
    b = comparable_value(right)
    numeric = (
        isinstance(a, Real)
        and isinstance(b, Real)
        and not isinstance(a, bool)
        and not isinstance(b, bool)
    )
    if numeric:
        return math.isclose(
            float(cast(Real, a)),
            float(cast(Real, b)),
            rel_tol=0.0,
            abs_tol=abs_tol,
        )
    return a == b
