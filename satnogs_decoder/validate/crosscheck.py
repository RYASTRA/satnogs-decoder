from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from satnogs_decoder.validate.normalize import normalize_fields, values_equal

CrossCheckStatus = Literal["passed", "failed", "inconclusive", "skipped"]


@dataclass
class CrossCheck:
    n_compared: int
    agreement: float
    status: CrossCheckStatus
    reason: str
    mismatched_fields: dict[str, int] = field(default_factory=dict)
    unmatched_ours: dict[str, int] = field(default_factory=dict)
    unmatched_refs: dict[str, int] = field(default_factory=dict)
    normalization_collisions: dict[str, int] = field(default_factory=dict)
    n_reference_failures: int = 0
    n_ours_empty: int = 0
    n_ref_empty: int = 0


def _bump_counts(target: dict[str, int], keys: set[str]) -> None:
    for key in keys:
        target[key] = target.get(key, 0) + 1


def _status(
    *,
    compared: int,
    agreement: float,
    n_pairs: int,
    n_reference_failures: int,
    agreement_threshold: float,
    min_compared: int,
) -> tuple[CrossCheckStatus, str]:
    if not n_pairs:
        return "skipped", "no frame pairs supplied"
    if n_reference_failures == n_pairs:
        return "skipped", "reference decoder failed for every frame"
    if compared == 0:
        return "inconclusive", "no shared fields after normalization"
    if compared < min_compared:
        return "inconclusive", f"only {compared} field pair(s) compared (< {min_compared})"
    if agreement >= agreement_threshold:
        return "passed", f"agreement {agreement:.4f} >= {agreement_threshold:.4f}"
    return "failed", f"agreement {agreement:.4f} < {agreement_threshold:.4f}"


def cross_check(
    ours: Sequence[Mapping[str, object]],
    refs: Sequence[Mapping[str, object] | None],
    *,
    agreement_threshold: float = 0.95,
    min_compared: int = 1,
) -> CrossCheck:
    compared = 0
    matched = 0
    mism: dict[str, int] = {}
    unmatched_ours: dict[str, int] = {}
    unmatched_refs: dict[str, int] = {}
    collisions: dict[str, int] = {}
    n_reference_failures = 0
    n_ours_empty = 0
    n_ref_empty = 0

    for a, b in zip(ours, refs):
        if b is None:
            n_reference_failures += 1
            continue
        if not a:
            n_ours_empty += 1
            continue
        if not b:
            n_ref_empty += 1
            continue

        left = normalize_fields(a)
        right = normalize_fields(b)
        for key, names in left.collisions.items():
            collisions[key] = collisions.get(key, 0) + len(names)
        for key, names in right.collisions.items():
            collisions[key] = collisions.get(key, 0) + len(names)

        shared = left.values.keys() & right.values.keys()
        _bump_counts(unmatched_ours, set(left.values) - shared)
        _bump_counts(unmatched_refs, set(right.values) - shared)

        for k in shared:
            compared += 1
            if values_equal(left.values[k], right.values[k]):
                matched += 1
            else:
                mism[k] = mism.get(k, 0) + 1

    # agreement is field-pair-weighted: a frame with more shared fields contributes
    # more to the numerator/denominator than a frame with fewer shared fields.
    agreement = (matched / compared) if compared else 1.0
    status, reason = _status(
        compared=compared,
        agreement=agreement,
        n_pairs=min(len(ours), len(refs)),
        n_reference_failures=n_reference_failures,
        agreement_threshold=agreement_threshold,
        min_compared=min_compared,
    )
    return CrossCheck(
        n_compared=compared,
        agreement=agreement,
        status=status,
        reason=reason,
        mismatched_fields=mism,
        unmatched_ours=unmatched_ours,
        unmatched_refs=unmatched_refs,
        normalization_collisions=collisions,
        n_reference_failures=n_reference_failures,
        n_ours_empty=n_ours_empty,
        n_ref_empty=n_ref_empty,
    )
