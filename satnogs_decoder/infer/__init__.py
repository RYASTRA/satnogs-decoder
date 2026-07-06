"""Phase 4: automatic structural decoder inference from raw payload bytes."""
from __future__ import annotations

from satnogs_decoder.infer.infer import infer_ksy, load_model

__all__ = ["infer_ksy", "load_model"]
