from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CrossCheck:
    n_compared: int
    agreement: float
    mismatched_fields: dict[str, int] = field(default_factory=dict)


def cross_check(ours: list[dict], refs: list[dict | None]) -> CrossCheck:
    compared = 0
    matched = 0
    mism: dict[str, int] = {}
    for a, b in zip(ours, refs):
        if not a or not b:
            continue
        for k in a.keys() & b.keys():
            compared += 1
            if a[k] == b[k]:
                matched += 1
            else:
                mism[k] = mism.get(k, 0) + 1
    agreement = (matched / compared) if compared else 1.0
    return CrossCheck(n_compared=compared, agreement=agreement, mismatched_fields=mism)
