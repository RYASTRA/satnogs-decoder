"""The trained model: three lightweight gradient-boosted classifiers.

  * boundary head — per byte position: does a field START here?
  * signed head   — per field: signed vs unsigned integer?
  * enum head     — per field: low-cardinality / fixed-code candidate?

Widths are NOT modelled — a field's width is `end - start`, determined by
the boundary head's predictions (spec §8). Escalate to a neural sequence
model ONLY if the held-out eval (Task 12) shows this plateauing below
target — that decision is made from the scoreboard, not assumed.
"""
from __future__ import annotations

import pathlib

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


def field_features(pos_feats: np.ndarray, spans: list[tuple[int, int]]) -> np.ndarray:
    """Pool per-position features into one row per field span.

    Row = [mean-pool over the span] ++ [first-byte row] ++ [last-byte row] ++ [width].
    Gives the signed/enum heads both the field's aggregate behaviour and its
    boundary-byte behaviour (the sign bit lives in the field's MSB).
    """
    rows: list[np.ndarray] = []
    for (s, e) in spans:
        block = pos_feats[s:e]
        pooled = block.mean(axis=0)
        rows.append(np.concatenate([pooled, pos_feats[s], pos_feats[e - 1], [e - s]]))
    return np.array(rows, dtype=np.float64)


class InferModel:
    def __init__(self) -> None:
        self._boundary = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.1)
        self._signed = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.1)
        self._enum = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.1)
        self._has_field_heads = False

    def fit_boundary(self, X: np.ndarray, y: np.ndarray) -> None:
        self._boundary.fit(X, y)

    def predict_boundary(self, X: np.ndarray) -> np.ndarray:
        pred = self._boundary.predict(X)
        return np.asarray(pred, dtype=int)

    def predict_boundary_proba(self, X: np.ndarray) -> np.ndarray:
        proba = self._boundary.predict_proba(X)
        return np.asarray(proba[:, 1], dtype=np.float64)

    def fit_field(self, Xf: np.ndarray, y_signed: np.ndarray, y_enum: np.ndarray) -> None:
        self._signed.fit(Xf, y_signed)
        self._enum.fit(Xf, y_enum)
        self._has_field_heads = True

    def predict_field(self, Xf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        signed_pred = self._signed.predict(Xf)
        enum_pred = self._enum.predict(Xf)
        return np.asarray(signed_pred, dtype=int), np.asarray(enum_pred, dtype=int)

    def save(self, path: str) -> None:
        p = pathlib.Path(path)
        p.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._boundary, p / "boundary.joblib")
        if self._has_field_heads:
            joblib.dump(self._signed, p / "signed.joblib")
            joblib.dump(self._enum, p / "enum.joblib")

    @classmethod
    def load(cls, path: str) -> "InferModel":
        p = pathlib.Path(path)
        m = cls()
        m._boundary = joblib.load(p / "boundary.joblib")
        if (p / "signed.joblib").exists():
            m._signed = joblib.load(p / "signed.joblib")
            m._enum = joblib.load(p / "enum.joblib")
            m._has_field_heads = True
        return m
