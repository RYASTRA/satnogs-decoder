"""Mine the TRUE field layout of a canonical .ksy — the training labels.

Two independent sources, zipped by leaf name:
  * byte spans  — from a ksc --debug parse of a real frame (exact, handles
                  process:/str/sub-types transparently; spec §5).
  * field types — from a static walk of the .ksy seq/types (gives signedness
                  and enum-presence, which --debug offsets do not carry).

Restricted to the flat-beacon target class (structure.is_flat_beacon), so
the two orderings align one-to-one.
"""
from __future__ import annotations

import io

from kaitaistruct import KaitaiStream, KaitaiStruct
from ruamel.yaml import YAML

from satnogs_decoder.infer.layout import FieldSpan, Layout
from satnogs_decoder.shared.kaitai import compile_ksy


def _leaf_spans(obj: object, prefix: str, out: list[tuple[str, int, int]]) -> None:
    """Walk a --debug-parsed object, emitting (dotted_name, start, end) per scalar leaf."""
    debug = getattr(obj, "_debug", {}) or {}
    for name, span in debug.items():
        if name.startswith("_"):
            continue
        child = getattr(obj, name, None)
        dotted = f"{prefix}{name}"
        if isinstance(child, KaitaiStruct):
            _leaf_spans(child, dotted + ".", out)
        else:
            out.append((dotted, int(span["start"]), int(span["end"])))


def debug_leaf_spans(
    ksy_text: str, frame: bytes, *, import_dirs: list[str] | None = None
) -> list[tuple[str, int, int]]:
    cls = compile_ksy(ksy_text, import_dirs=import_dirs, debug=True)
    obj = cls(KaitaiStream(io.BytesIO(frame)))
    # ksc --debug mode does NOT auto-read (verified by the Task-0 spike):
    # __init__ only sets up _io/_debug; _read() must be called explicitly, or
    # _debug stays empty and every field attribute is missing. Sub-type _read()
    # is invoked by the parent's _read(), so one top-level call populates all.
    obj._read()
    out: list[tuple[str, int, int]] = []
    _leaf_spans(obj, "", out)
    return out


def _walk_declared(
    seq: list, types: dict, prefix: str, out: list[tuple[str, str, bool]]
) -> None:
    for f in seq:
        if not isinstance(f, dict) or "id" not in f:
            continue
        ftype = f.get("type")
        dotted = f"{prefix}{f['id']}"
        if isinstance(ftype, str) and ftype in types:
            _walk_declared(types[ftype].get("seq", []), types, dotted + ".", out)
        else:
            out.append((dotted, ftype or "", "enum" in f))


def declared_leaves(ksy_text: str) -> list[tuple[str, str, bool]]:
    doc = YAML(typ="safe").load(io.StringIO(ksy_text))
    out: list[tuple[str, str, bool]] = []
    _walk_declared(doc.get("seq", []), doc.get("types") or {}, "", out)
    return out


def extract_layout(
    ksy_text: str, frame: bytes, *, import_dirs: list[str] | None = None
) -> Layout:
    spans = debug_leaf_spans(ksy_text, frame, import_dirs=import_dirs)
    decls = {name: (t, e) for name, t, e in declared_leaves(ksy_text)}
    # The two leaf sources must align one-to-one (guaranteed for the flat-beacon
    # class). A span whose name is absent from the declared seq means the two
    # walkers diverged — fail LOUDLY rather than emit a plausible-but-wrong
    # label (this is the ground-truth label miner; a silent bad label is worse
    # than a crash).
    missing = [name for name, _, _ in spans if name not in decls]
    if missing:
        raise ValueError(
            f"debug spans reference leaves absent from the declared seq "
            f"(name divergence — labels untrustworthy): {missing}"
        )
    layout: Layout = []
    for name, start, end in spans:
        ftype, has_enum = decls[name]
        signed = bool(ftype) and ftype[0] == "s"
        layout.append(FieldSpan(
            start=start, end=end, width=end - start,
            signed=signed, is_enum=has_enum, name=name,
        ))
    return layout
