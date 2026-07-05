"""Derive the `:field` dashboard doc-block from a built `KsySpec` IR.

The SatNOGS dashboard (and `satnogsdecoders.decoder.get_fields()`) discover
decoded telemetry fields by scanning `:field <leaf>: <path>` lines out of a
compiled Kaitai struct's docstring. This module walks the IR that
`generate.build.build_ir` produces and derives those lines mechanically, so
every scalar leaf reachable from the parsed struct — header callsign/ssid
fields, top-level computed instances, and every frame_type's own seq +
instances + one level of nested sub-types — gets a matching `:field` line.

`get_fields()` resolves each path with `functools.reduce(getattr, path.split("."),
struct)` starting from the root parsed object itself — there is NO leading
`<meta.id>` attribute on that object in general (canonical upstream .ksy
files confirm this: e.g. vzlusat2.ksy's doc block starts `csp_header.crc`,
not `vzlusat2.csp_header.crc`). Paths must therefore start from the
top-level seq field names directly, not be prefixed with `ir.id`.

Walk rule:
- Header fields (everything in `ir.seq` other than `payload`) are walked
  under `<field_id>...<leaf>`.
- Top-level instances (e.g. the discriminator `kind`) are emitted as
  `<instance_id>`.
- The `payload` field is walked under `payload.<leaf>` in both cases: when
  the payload is a switch (multiple frame_types) and when there is a
  single, unswitched frame_type. Kaitai's `switch-on` exposes the selected
  case's fields directly on the switch field, so the case/frame_type name
  never appears in the path.
- Within a frame_type (or any sub-type), every seq leaf and every instance
  is a field; a seq field whose type names another declared sub-type is
  walked recursively (one level per the spec, but the walk itself is
  generic/recursive so header types — which nest two levels deep — are
  fully covered too).
"""
from __future__ import annotations

from satnogs_decoder.shared.ksy import KsyField, KsyInstance, KsySpec, KsySwitch, KsyType


def _type_fields(t: "KsyType | list[KsyField]") -> "tuple[list[KsyField], list[KsyInstance]]":
    if isinstance(t, KsyType):
        return t.seq, t.instances or []
    return list(t), []


def _walk_type(
    type_name: str,
    prefix: str,
    types: "dict[str, KsyType | list[KsyField]]",
    visited: frozenset[str],
    lines: list[str],
) -> None:
    t = types.get(type_name)
    if t is None or type_name in visited:
        return
    visited = visited | {type_name}
    seq, instances = _type_fields(t)
    for f in seq:
        path = f"{prefix}.{f.id}"
        if isinstance(f.type, str) and f.type in types:
            _walk_type(f.type, path, types, visited, lines)
        else:
            lines.append(f":field {f.id}: {path}")
    for inst in instances:
        lines.append(f":field {inst.id}: {prefix}.{inst.id}")


def field_block(ir: KsySpec) -> str:
    """Derive the `:field <leaf>: <path>.<leaf>` dashboard doc-block.

    Walks every frame_type sub-type's seq leaves and instances (plus nested
    sub-types one level down), the header seq's own leaves, and any
    top-level instances (e.g. the discriminator). Returns the block as a
    single string, one `:field` line per decoded leaf, deduped and in
    deterministic (first-seen) order, terminated by a trailing newline.

    Paths start from the top-level seq field names directly (NOT prefixed
    with `ir.id`) — see module docstring for why.
    """
    types = ir.types or {}
    lines: list[str] = []

    payload: KsyField | None = None
    header_fields: list[KsyField] = []
    for f in ir.seq:
        if f.id == "payload":
            payload = f
        else:
            header_fields.append(f)

    for f in header_fields:
        path = f.id
        if isinstance(f.type, str) and f.type in types:
            _walk_type(f.type, path, types, frozenset(), lines)
        else:
            lines.append(f":field {f.id}: {path}")

    for inst in ir.instances or []:
        lines.append(f":field {inst.id}: {inst.id}")

    if payload is not None:
        if isinstance(payload.type, KsySwitch):
            for frame_type_id in dict.fromkeys(payload.type.cases.values()):
                _walk_type(frame_type_id, "payload", types, frozenset(), lines)
        elif isinstance(payload.type, str):
            _walk_type(payload.type, "payload", types, frozenset(), lines)

    seen: set[str] = set()
    deduped: list[str] = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            deduped.append(line)

    return "".join(f"{line}\n" for line in deduped)
