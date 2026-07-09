"""Orchestrator: run a Kaitai parser over a frame corpus and produce a ValidationReport."""

from __future__ import annotations

from dataclasses import dataclass

from satnogs_decoder.shared.kaitai import parse
from satnogs_decoder.shared.reference import decode_reference
from satnogs_decoder.validate.crosscheck import CrossCheck, cross_check
from satnogs_decoder.validate.report import Coverage, FieldStat, coverage, field_stats

VALIDATION_REPORT_SCHEMA_VERSION = 2


@dataclass
class ValidationReport:
    n_frames: int
    coverage: Coverage
    field_stats: list[FieldStat]
    crosscheck: CrossCheck | None
    report_schema_version: int = VALIDATION_REPORT_SCHEMA_VERSION


def validate(
    parser_cls: type,
    frames: list,
    *,
    ref_module: str | None = None,
) -> ValidationReport:
    """Parse every frame with *parser_cls*, aggregate stats, and optionally cross-check.

    Parameters
    ----------
    parser_cls:
        A Kaitai-compiled class (from ``compile_ksy`` or a pre-generated fixture).
    frames:
        List of ``Frame`` objects (from ``satnogs_decoder.shared.satnogs_db``).
    ref_module:
        When given, calls ``decode_reference(ref_module, frame.data)`` for every
        frame and computes agreement between our parse and the canonical decode.

    Returns
    -------
    ValidationReport
    """
    results = [parse(parser_cls, f.data) for f in frames]

    cc: CrossCheck | None = None
    if ref_module is not None:
        ours = [r.fields if r.ok else {} for r in results]
        refs = [decode_reference(ref_module, f.data) for f in frames]
        cc = cross_check(ours, refs)

    return ValidationReport(
        n_frames=len(frames),
        coverage=coverage(results),
        field_stats=field_stats(results),
        crosscheck=cc,
    )
