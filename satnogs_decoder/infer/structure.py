"""Cheap static pre-filter for the widened `infer` target class.

`has_fixed_layout_case` admits a decoder if *any* candidate sequence — the
top-level seq or any declared sub-type — is a flat, fixed-width scalar
sequence. Switched decoders qualify as long as one case is fixed-layout;
only decoders where no sequence is fixed-layout (every candidate uses
repeat/size-eos/nested switch) are skipped. Final fixedness is confirmed
frame-aware in infer.qualify. This is deliberately permissive — over-
admitting only costs a qualify probe; under-admitting silently loses sats.
"""
from __future__ import annotations

import io
import re

from ruamel.yaml import YAML

_SCALAR = re.compile(r"^(u[1248]|s[1248]|f[48])(le|be)?$")
_DISALLOWED = ("repeat", "repeat-expr", "repeat-until", "if")


def _seq_is_flat(seq: object, types: dict, seen: frozenset) -> bool:
    if not isinstance(seq, list) or not seq:
        return False
    for f in seq:
        if not isinstance(f, dict) or "id" not in f:
            return False
        if any(k in f for k in _DISALLOWED):
            return False
        if f.get("size") == "eos" or "size-eos" in f:
            return False
        if "contents" in f:
            continue
        ftype = f.get("type")
        if isinstance(ftype, dict):      # switch inside a candidate seq -> not flat
            return False
        if ftype is None:
            if isinstance(f.get("size"), int):
                continue
            return False
        if _SCALAR.match(ftype):
            continue
        if ftype == "str":
            if isinstance(f.get("size"), int):
                continue
            return False
        if ftype in types:               # fixed sub-type: recurse
            if ftype in seen:
                return False
            if not _seq_is_flat(types[ftype].get("seq"), types, seen | {ftype}):
                return False
            continue
        return False
    return True


def has_fixed_layout_case(ksy_text: str) -> bool:
    # Batch pre-filter over external canonical .ksy in an unattended loop: any
    # malformed/odd input is SKIPPED (return False), never allowed to crash the
    # discovery of the other ~160 candidates.
    try:
        doc = YAML(typ="safe").load(io.StringIO(ksy_text))
        if not isinstance(doc, dict):
            return False
        types = doc.get("types") or {}
        if not isinstance(types, dict):
            return False
        if _seq_is_flat(doc.get("seq"), types, frozenset()):
            return True
        return any(_seq_is_flat(t.get("seq"), types, frozenset()) for t in types.values()
                   if isinstance(t, dict))
    except Exception:  # noqa: BLE001 — a bad candidate is skipped, not fatal
        return False
