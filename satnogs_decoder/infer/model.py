"""The trained model: three lightweight gradient-boosted classifiers.

  * boundary head — per byte position: does a field START here?
  * signed head   — per field: signed vs unsigned integer?
  * enum head     — per field: low-cardinality / fixed-code candidate?

The width head is optional for backwards compatibility with older frozen
artifacts. When present, it lets inference choose field spans directly instead
of hard-splitting on every boundary threshold crossing.
"""

from __future__ import annotations

import pathlib

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

DEFAULT_BOUNDARY_THRESHOLD = 0.20


def field_features(
    pos_feats: np.ndarray,
    spans: list[tuple[int, int]],
    *,
    include_width: bool = True,
) -> np.ndarray:
    """Pool per-position features into one row per field span.

    Row = [mean-pool over the span] ++ [first-byte row] ++ [last-byte row] ++ [width].
    Gives the signed/enum heads both the field's aggregate behaviour and its
    boundary-byte behaviour (the sign bit lives in the field's MSB).
    """
    rows: list[np.ndarray] = []
    n_features = pos_feats.shape[1] * 3 + (1 if include_width else 0)
    for s, e in spans:
        if not 0 <= s < e <= len(pos_feats):
            # Fail loudly: a degenerate/out-of-range span would mean-pool an
            # empty slice into NaNs and silently corrupt the training matrix.
            raise ValueError(
                f"degenerate/out-of-range span ({s}, {e}) for {len(pos_feats)} positions"
            )
        block = pos_feats[s:e]
        pooled = block.mean(axis=0)
        chunks = [pooled, pos_feats[s], pos_feats[e - 1]]
        if include_width:
            chunks.append(np.array([e - s], dtype=np.float64))
        rows.append(np.concatenate(chunks))
    if not rows:
        return np.empty((0, n_features), dtype=np.float64)
    return np.array(rows, dtype=np.float64)


class InferModel:
    def __init__(self, *, boundary_threshold: float = DEFAULT_BOUNDARY_THRESHOLD) -> None:
        self._boundary = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.1)
        self._signed = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.1)
        self._enum = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.1)
        self._width = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.1)
        self.boundary_threshold = boundary_threshold
        self._has_field_heads = False
        self._has_width_head = False
        self._constant_width: int | None = None
        self._width_classes = np.array([], dtype=int)

    def fit_boundary(self, X: np.ndarray, y: np.ndarray) -> None:
        self._boundary.fit(X, y)

    def predict_boundary(self, X: np.ndarray) -> np.ndarray:
        pred = self.predict_boundary_proba(X) >= self.boundary_threshold
        return np.asarray(pred, dtype=int)

    def predict_boundary_proba(self, X: np.ndarray) -> np.ndarray:
        proba = self._boundary.predict_proba(X)
        return np.asarray(proba[:, 1], dtype=np.float64)

    @property
    def has_width_head(self) -> bool:
        return self._has_width_head

    def fit_field(
        self,
        Xf: np.ndarray,
        y_signed: np.ndarray,
        y_enum: np.ndarray,
        *,
        X_width: np.ndarray | None = None,
        y_width: np.ndarray | None = None,
    ) -> None:
        self._signed.fit(Xf, y_signed)
        self._enum.fit(Xf, y_enum)
        self._has_field_heads = True
        if X_width is not None and y_width is not None:
            self.fit_width(X_width, y_width)

    def fit_width(self, X_width: np.ndarray, y_width: np.ndarray) -> None:
        widths = np.asarray(y_width, dtype=int)
        if widths.size == 0:
            return
        classes = np.unique(widths)
        self._width_classes = np.asarray(classes, dtype=int)
        self._has_width_head = True
        if classes.size == 1:
            self._constant_width = int(classes[0])
            return
        self._constant_width = None
        self._width.fit(X_width, widths)

    def predict_field(self, Xf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        signed_pred = self._signed.predict(Xf)
        enum_pred = self._enum.predict(Xf)
        return np.asarray(signed_pred, dtype=int), np.asarray(enum_pred, dtype=int)

    def width_classes(self) -> tuple[int, ...]:
        return tuple(int(w) for w in self._width_classes)

    def predict_width_proba(self, X_width: np.ndarray) -> np.ndarray:
        if not self._has_width_head:
            raise RuntimeError("width head is not available")
        if self._constant_width is not None:
            return np.ones((X_width.shape[0], 1), dtype=np.float64)
        return np.asarray(self._width.predict_proba(X_width), dtype=np.float64)

    def predict_width(self, X_width: np.ndarray) -> np.ndarray:
        if not self._has_width_head:
            raise RuntimeError("width head is not available")
        if self._constant_width is not None:
            return np.full(X_width.shape[0], self._constant_width, dtype=int)
        return np.asarray(self._width.predict(X_width), dtype=int)

    def save(self, path: str) -> None:
        p = pathlib.Path(path)
        p.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._boundary, p / "boundary.joblib")
        if self._has_field_heads:
            joblib.dump(self._signed, p / "signed.joblib")
            joblib.dump(self._enum, p / "enum.joblib")
        if self._has_width_head:
            joblib.dump(
                {
                    "classifier": None if self._constant_width is not None else self._width,
                    "constant": self._constant_width,
                    "classes": self._width_classes,
                },
                p / "width.joblib",
            )

    @classmethod
    def load(cls, path: str) -> "InferModel":
        p = pathlib.Path(path)
        m = cls()
        m._boundary = joblib.load(p / "boundary.joblib")
        if (p / "signed.joblib").exists():
            m._signed = joblib.load(p / "signed.joblib")
            m._enum = joblib.load(p / "enum.joblib")
            m._has_field_heads = True
        if (p / "width.joblib").exists():
            payload = joblib.load(p / "width.joblib")
            if isinstance(payload, dict):
                m._constant_width = payload.get("constant")
                classifier = payload.get("classifier")
                if classifier is not None:
                    m._width = classifier
                m._width_classes = np.asarray(payload.get("classes", []), dtype=int)
            else:
                m._width = payload
                m._constant_width = None
                m._width_classes = np.asarray(getattr(payload, "classes_", []), dtype=int)
            m._has_width_head = True
        return m
