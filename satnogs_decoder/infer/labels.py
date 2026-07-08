"""Mine the TRUE field layout of a frame from the FULL canonical .ksy.

Parses one real frame in ksc --debug mode and walks the parsed object in
parallel with the declared seq, following the object's ACTUAL switch/sub-type
selections. This handles switched/transport-framed decoders (the common case)
by labeling exactly the fields Kaitai read for that frame — the transport
header + discriminator + the selected frame-type case. Byte spans come from
the object's `_debug`; signedness/enum come from the declared type of the
field actually taken.
"""
from __future__ import annotations

import io
import re

from kaitaistruct import KaitaiStream, KaitaiStruct
from ruamel.yaml import YAML

from satnogs_decoder.infer.layout import FieldSpan, Layout
from satnogs_decoder.shared.kaitai import compile_ksy

_SIGNED = re.compile(r"^s[1248](le|be)?$")


def _cap(t: str) -> str:
    return "".join(p.capitalize() for p in t.split("_"))


def _resolve_type(child: object, ftype: object) -> str | None:
    """Map a parsed sub-object to its declared .ksy type name.

    For a plain user-type field ftype is the type-name string. For a switch,
    ftype is a {switch-on, cases} dict; the selected case is identified by
    matching the sub-object's generated class name to the capitalized case
    type name.
    """
    if isinstance(ftype, str):
        return ftype
    if isinstance(ftype, dict):
        cls_name = type(child).__name__
        for tname in ftype.get("cases", {}).values():
            if _cap(tname) == cls_name:
                return tname
    return None


def _walk(obj: object, seq: list, types: dict, prefix: str,
          out: "list[tuple[str, str, bool, int, int]]") -> None:
    debug = getattr(obj, "_debug", {}) or {}
    for f in seq:
        if not isinstance(f, dict) or "id" not in f:
            continue
        fid = f["id"]
        span = debug.get(fid)
        child = getattr(obj, fid, None)
        ftype = f.get("type")
        if isinstance(child, list):
            # a repeat/array field has no fixed layout; a supposedly-flat case
            # must not contain one -> fail loud rather than emit a bogus span.
            raise ValueError(f"unexpected repeat/array field {prefix}{fid!r} in a flat case")
        if isinstance(child, KaitaiStruct):
            tname = _resolve_type(child, ftype)
            sub = types.get(tname) if tname else None
            sub_seq = sub.get("seq", []) if isinstance(sub, dict) else []
            if not sub_seq and getattr(child, "_debug", None):
                # the sub-object read fields we cannot type (unresolved switch
                # case or imported type) -> labels would be silently incomplete.
                raise ValueError(
                    f"cannot resolve declared type of {prefix}{fid!r} "
                    f"(class {type(child).__name__}); labels would be incomplete"
                )
            _walk(child, sub_seq, types, f"{prefix}{fid}.", out)
        elif span is not None:  # a scalar/str/contents leaf actually read
            t = ftype if isinstance(ftype, str) else ""
            out.append((f"{prefix}{fid}", t, "enum" in f,
                        int(span["start"]), int(span["end"])))


def extract_layout(ksy_text: str, frame: bytes, *, import_dirs: list[str] | None = None) -> Layout:
    cls = compile_ksy(ksy_text, import_dirs=import_dirs, debug=True)
    obj = cls(KaitaiStream(io.BytesIO(frame)))
    obj._read()  # --debug mode has no auto-read
    doc = YAML(typ="safe").load(io.StringIO(ksy_text))
    out: "list[tuple[str, str, bool, int, int]]" = []
    _walk(obj, doc.get("seq", []), doc.get("types") or {}, "", out)
    return [
        FieldSpan(start=s, end=e, width=e - s,
                  signed=bool(_SIGNED.match(t)), is_enum=en, name=name)
        for name, t, en, s, e in out
    ]
