"""Coverage, decode-rate, and field-range statistics over ParseResult lists."""

from __future__ import annotations

from dataclasses import dataclass

from satnogs_decoder.shared.kaitai import ParseResult


@dataclass
class FieldStat:
    name: str
    n: int
    min: float | None
    max: float | None
    constant: bool
    all_zero: bool


@dataclass
class Coverage:
    decode_rate: float
    mean_consumed_frac: float
    full_coverage_rate: float


def coverage(results: list[ParseResult]) -> Coverage:
    """Aggregate decode-rate, mean byte-consumed fraction, and full-coverage rate."""
    n = len(results)
    if n == 0:
        return Coverage(decode_rate=0.0, mean_consumed_frac=0.0, full_coverage_rate=0.0)

    ok = [r for r in results if r.ok]
    decode_rate = len(ok) / n

    fracs = [r.consumed / r.total for r in ok if r.total]
    mean_frac = sum(fracs) / len(fracs) if fracs else 0.0

    full = sum(1 for r in ok if r.total and r.consumed == r.total) / n

    return Coverage(
        decode_rate=decode_rate,
        mean_consumed_frac=mean_frac,
        full_coverage_rate=full,
    )


def field_stats(results: list[ParseResult]) -> list[FieldStat]:
    """Compute per-field stats (numeric min/max, constant, all_zero) from successful parses."""
    values: dict[str, list[object]] = {}
    for r in results:
        if not r.ok:
            continue
        for k, v in r.fields.items():
            values.setdefault(k, []).append(v)

    out: list[FieldStat] = []
    for name, vals in sorted(values.items()):
        nums = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
        mn: float | None = min(nums) if nums else None  # type: ignore[arg-type]
        mx: float | None = max(nums) if nums else None  # type: ignore[arg-type]
        out.append(
            FieldStat(
                name=name,
                n=len(vals),
                min=mn,
                max=mx,
                constant=len(set(map(repr, vals))) == 1,
                all_zero=bool(nums) and all(v == 0 for v in nums),
            )
        )
    return out
