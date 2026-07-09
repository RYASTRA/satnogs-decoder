"""Structured inference evaluation reports.

The console scoreboard is useful while iterating, but the improvement loop needs
stable JSON that can be diffed across runs and attached to model artifacts.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

INFER_REVIEW_SCHEMA_VERSION = 1


def _without_per_sat(scores: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in scores.items() if k != "per_sat"}


def _sort_per_sat(
    per_sat: list[dict[str, Any]],
    key: str,
    *,
    reverse: bool = False,
    limit: int = 5,
) -> list[dict[str, Any]]:
    return sorted(per_sat, key=lambda row: float(row.get(key, 0.0)), reverse=reverse)[:limit]


def model_artifacts(model_dir: str | pathlib.Path) -> dict[str, Any]:
    path = pathlib.Path(model_dir)
    artifacts = {}
    for name in ("boundary.joblib", "signed.joblib", "enum.joblib", "width.joblib"):
        artifact = path / name
        artifacts[name] = {
            "present": artifact.exists(),
            "bytes": artifact.stat().st_size if artifact.exists() else 0,
        }
    return {"model_dir": str(path), "artifacts": artifacts}


def build_review_report(
    scores: dict[str, Any],
    *,
    degradation_curve: dict[str, float] | None = None,
    corpus: dict[str, Any] | None = None,
    model: dict[str, Any] | None = None,
) -> dict[str, Any]:
    per_sat = list(scores.get("per_sat", []))
    return {
        "schema_version": INFER_REVIEW_SCHEMA_VERSION,
        "aggregate": _without_per_sat(scores),
        "degradation_curve": degradation_curve or {},
        "review": {
            "worst_span_f1": _sort_per_sat(per_sat, "span_f1"),
            "worst_boundary_f1": _sort_per_sat(per_sat, "boundary_f1"),
            "highest_oversegmentation": _sort_per_sat(
                per_sat, "oversegmentation_rate", reverse=True
            ),
            "largest_field_count_error": sorted(
                per_sat,
                key=lambda row: abs(float(row.get("field_count_error", 0.0))),
                reverse=True,
            )[:5],
        },
        "per_sat": per_sat,
        "corpus": corpus or {},
        "model": model or {},
    }


def write_json(path: str | pathlib.Path, payload: dict[str, Any]) -> None:
    out = pathlib.Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
