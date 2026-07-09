"""Per-byte-position statistics across a satellite's frames — the model inputs.

For each byte offset `i`, aggregate the value of byte `i` across every
fixed-length frame into features that expose field structure: constancy and
entropy (constant/padding bytes), sign-bit behaviour (signedness), positive
monotonicity (counters/timestamps), and correlation with the neighbouring
byte (multi-byte field grouping). Position itself is included so the model
can learn header-region priors.
"""

from __future__ import annotations

import collections

import numpy as np

FEATURE_NAMES: list[str] = [
    "pos_abs",
    "pos_frac",
    "mean",
    "std",
    "entropy",
    "distinct_frac",
    "is_constant",
    "frac_zero",
    "frac_ff",
    "high_bit_frac",
    "monotonic",
    "corr_prev",
    "corr_next",
]


def common_length(frames: list[bytes]) -> int:
    lengths = collections.Counter(len(f) for f in frames)
    return lengths.most_common(1)[0][0]


def _entropy(col: np.ndarray) -> float:
    counts = np.bincount(col, minlength=256)
    p = counts[counts > 0] / col.size
    return float(-(p * np.log2(p)).sum())


def _monotonic(col: np.ndarray) -> float:
    """Fraction of consecutive steps that are positive (counter signal)."""
    if col.size < 2:
        return 0.0
    diffs = np.diff(col.astype(np.int16))
    return float((diffs > 0).mean())


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(abs(np.corrcoef(a, b)[0, 1]))


def position_features(frames: list[bytes]) -> np.ndarray:
    L = common_length(frames)
    mat = np.array([list(f) for f in frames if len(f) == L], dtype=np.uint8)
    n, _ = mat.shape
    rows: list[list[float]] = []
    for i in range(L):
        col = mat[:, i]
        prev = mat[:, i - 1] if i > 0 else col
        nxt = mat[:, i + 1] if i + 1 < L else col
        distinct = len(np.unique(col))
        rows.append(
            [
                float(i),
                i / max(L - 1, 1),
                float(col.mean()),
                float(col.std()),
                _entropy(col),
                distinct / max(n, 1),
                1.0 if distinct == 1 else 0.0,
                float((col == 0).mean()),
                float((col == 0xFF).mean()),
                float((col >= 0x80).mean()),
                _monotonic(col),
                _corr(col.astype(float), prev.astype(float)) if i > 0 else 0.0,
                _corr(col.astype(float), nxt.astype(float)) if i + 1 < L else 0.0,
            ]
        )
    return np.array(rows, dtype=np.float64)
