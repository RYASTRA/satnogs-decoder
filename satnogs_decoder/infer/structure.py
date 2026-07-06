"""Decide whether a canonical .ksy is in the v1 `infer` target class:
a flat, single-packet, fixed-layout beacon (spec §6).

Rejection is conservative and reason-bearing: any construct whose byte
layout is not a static left-to-right concatenation of fixed-width leaves
(switch/repeat/conditional/size-eos/computed-size/instances-with-pos)
excludes the decoder. A fixed transport header (itself flat scalars or
`size`d fields, e.g. an AX.25 header) is allowed because its byte width is
still static.
"""
from __future__ import annotations

import io
import re

from ruamel.yaml import YAML

_SCALAR = re.compile(r"^(u[1248]|s[1248]|f[48])(le|be)?$")
_DISALLOWED_KEYS = ("repeat", "repeat-expr", "repeat-until", "if")


def _load(ksy_text: str) -> dict:
    return YAML(typ="safe").load(io.StringIO(ksy_text))


def _leaf_ok(field: dict, types: dict, reason: list[str], seen: frozenset) -> bool:
    if not isinstance(field, dict) or "id" not in field:
        reason.append(f"malformed seq entry: {field!r}")
        return False
    for k in _DISALLOWED_KEYS:
        if k in field:
            reason.append(f"field {field['id']!r} uses '{k}' (not fixed-layout)")
            return False
    ftype = field.get("type")
    if isinstance(ftype, dict):  # switch-on
        reason.append(f"field {field['id']!r} is a switch (not single-packet)")
        return False
    if field.get("size") == "eos" or "size-eos" in field:
        reason.append(f"field {field['id']!r} is size-eos (unbounded)")
        return False
    if ftype is None:
        # a raw byte blob with a fixed `size:` int is allowed
        if isinstance(field.get("size"), int):
            return True
        reason.append(f"field {field['id']!r} has no type and no fixed size")
        return False
    if _SCALAR.match(ftype):
        return True
    if ftype == "str":
        if isinstance(field.get("size"), int):
            return True
        reason.append(f"str field {field['id']!r} has no fixed size")
        return False
    if ftype in types:  # a fixed transport-header sub-type: recurse
        if ftype in seen:
            reason.append(f"recursive sub-type {ftype!r}")
            return False
        sub = types[ftype]
        for sf in sub.get("seq", []):
            if not _leaf_ok(sf, types, reason, seen | {ftype}):
                return False
        return True
    reason.append(f"field {field['id']!r} unknown/complex type {ftype!r}")
    return False


def is_flat_beacon(ksy_text: str) -> tuple[bool, str]:
    try:
        doc = _load(ksy_text)
    except Exception as e:  # noqa: BLE001
        return False, f"unparseable ksy: {e}"
    if not isinstance(doc, dict) or "seq" not in doc:
        return False, "no top-level seq"
    types = doc.get("types") or {}
    # Reject nested switches anywhere in `types` (a sub-type could hide one).
    for tname, tdef in types.items():
        for sf in (tdef.get("seq") or []):
            if isinstance(sf, dict) and isinstance(sf.get("type"), dict):
                return False, f"sub-type {tname!r} contains a switch"
    reason: list[str] = []
    for f in doc["seq"]:
        if not _leaf_ok(f, types, reason, frozenset()):
            return False, reason[-1] if reason else "non-flat seq field"
    return True, ""
