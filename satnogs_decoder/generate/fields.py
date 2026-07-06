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

`get_fields()` also keys its result by the bare `:field` NAME (before the
colon) in a plain `dict` — a second `:field foo: ...` line silently
overwrites the first. So besides being a valid *path*, every emitted NAME
must be unique across the whole block, or one of the two values becomes
unrecoverable. The clearest real case: the default `ax25` transport header
declares both `ax25_dest_callsign_raw` and `ax25_src_callsign_raw` as the
same sub-type (`ax25_callsign_raw`), so a naive walk emits `:field
callsign:` (and `:field ssid:`) twice — dest clobbers src, or vice versa.
The same shape of collision can happen whenever a sub-type is reused by
two sibling fields, whether that's the ax25 header or two sibling
frame_type seq fields sharing one sub-type.

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

Name disambiguation (two-pass):
- Pass 1 walks the IR and collects `(name, path)` pairs, exactly as before,
  but without formatting them into `:field` lines yet. Each collected pair
  also carries the chain of path segments belonging to the field it came
  from (excluding the fixed `payload`/root prefix), so we know which
  segments are available to disambiguate with.
- Pass 2 groups the collected entries by `name` (after de-duplicating exact
  `(name, path)` repeats — the same leaf reached the same way twice, e.g.
  two frame_types sharing a sub-type, is not a collision). Any name left
  with more than one distinct path is disambiguated by prepending
  successive path segments (innermost distinguishing segment first, e.g.
  the reused sub-type's own field id such as `dest`/`src`) until every
  variant is unique. This is mechanical and deterministic — same input
  spec always yields the same names — and only touches the NAME half of
  the `:field NAME: path` line; the path is untouched.
"""
from __future__ import annotations

from dataclasses import dataclass, field as _dc_field

from satnogs_decoder.shared.ksy import KsyField, KsyInstance, KsySpec, KsySwitch, KsyType


@dataclass
class _Entry:
    name: str
    path: str
    # Path segments available to disambiguate `name` with, ordered from the
    # outermost (the top-level field id the walk started from) to the
    # innermost (closest to the leaf). Never includes the leaf's own
    # segment (that IS `name`, pre-disambiguation) or the fixed `payload`
    # root segment (disambiguating on that would be meaningless — every
    # payload-rooted field shares it).
    segments: "list[str]" = _dc_field(default_factory=list)


def _type_fields(t: "KsyType | list[KsyField]") -> "tuple[list[KsyField], list[KsyInstance]]":
    if isinstance(t, KsyType):
        return t.seq, t.instances or []
    return list(t), []


def _walk_type(
    type_name: str,
    prefix: str,
    path_segments: "list[str]",
    types: "dict[str, KsyType | list[KsyField]]",
    visited: frozenset[str],
    entries: "list[_Entry]",
) -> None:
    t = types.get(type_name)
    if t is None or type_name in visited:
        return
    visited = visited | {type_name}
    seq, instances = _type_fields(t)
    for f in seq:
        path = f"{prefix}.{f.id}"
        if isinstance(f.type, str) and f.type in types:
            _walk_type(f.type, path, path_segments + [f.id], types, visited, entries)
        else:
            entries.append(_Entry(name=f.id, path=path, segments=list(path_segments)))
    for inst in instances:
        entries.append(_Entry(
            name=inst.id, path=f"{prefix}.{inst.id}", segments=list(path_segments),
        ))


def _disambiguate(entries: "list[_Entry]") -> "list[tuple[str, str]]":
    """Assign a unique NAME to each entry, preserving first-seen order.

    Entries that are exact `(name, path)` duplicates collapse to one. Among
    entries that share a `name` but have distinct `path`s, the name is
    disambiguated by prepending path segments — outermost (the top-level
    field id, e.g. `dest`/`src`) first, since that is normally the
    semantically meaningful distinguisher — one at a time until unique;
    ties that persist after all segments are exhausted fall back to the
    full path, which is unique by construction.
    """
    # De-dupe exact (name, path) repeats first (e.g. same sub-type reached
    # via two frame_types) while preserving first-seen order.
    deduped: "list[_Entry]" = []
    seen_pairs: set[tuple[str, str]] = set()
    for e in entries:
        key = (e.name, e.path)
        if key not in seen_pairs:
            seen_pairs.add(key)
            deduped.append(e)

    by_name: "dict[str, list[_Entry]]" = {}
    for e in deduped:
        by_name.setdefault(e.name, []).append(e)

    resolved: "dict[int, str]" = {}
    for name, group in by_name.items():
        if len(group) == 1:
            resolved[id(group[0])] = name
            continue
        # Collision: disambiguate by prepending segments one at a time,
        # outermost-first. A segment that is identical across every member
        # of the group adds no distinguishing information, so it is
        # skipped rather than prepended to all names.
        current = {id(e): name for e in group}
        depth = 0
        max_depth = max((len(e.segments) for e in group), default=0)
        while len(set(current.values())) < len(group) and depth < max_depth:
            values_at_depth = [e.segments[depth] if depth < len(e.segments) else None for e in group]
            if len(set(values_at_depth)) > 1:
                for e, v in zip(group, values_at_depth):
                    if v is not None:
                        current[id(e)] = f"{v}_{current[id(e)]}"
            depth += 1
        # Still colliding (segments exhausted, e.g. identical prefixes) —
        # fall back to the full path, guaranteed unique.
        if len(set(current.values())) < len(group):
            counts: "dict[str, int]" = {}
            for e in group:
                counts[current[id(e)]] = counts.get(current[id(e)], 0) + 1
            for e in group:
                if counts[current[id(e)]] > 1:
                    current[id(e)] = e.path.replace(".", "_")
        resolved.update(current)

    return [(resolved[id(e)], e.path) for e in deduped]


def field_block(ir: KsySpec) -> str:
    """Derive the `:field <leaf>: <path>.<leaf>` dashboard doc-block.

    Walks every frame_type sub-type's seq leaves and instances (plus nested
    sub-types one level down), the header seq's own leaves, and any
    top-level instances (e.g. the discriminator). Returns the block as a
    single string, one `:field` line per decoded leaf, in deterministic
    (first-seen) order, terminated by a trailing newline.

    Every emitted NAME (before the colon) is unique across the whole block
    — see module docstring "Name disambiguation" for why and how.

    Paths start from the top-level seq field names directly (NOT prefixed
    with `ir.id`) — see module docstring for why.
    """
    types = ir.types or {}
    entries: "list[_Entry]" = []

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
            _walk_type(f.type, path, [f.id], types, frozenset(), entries)
        else:
            entries.append(_Entry(name=f.id, path=path))

    for inst in ir.instances or []:
        entries.append(_Entry(name=inst.id, path=inst.id))

    if payload is not None:
        if isinstance(payload.type, KsySwitch):
            for frame_type_id in dict.fromkeys(payload.type.cases.values()):
                _walk_type(frame_type_id, "payload", [], types, frozenset(), entries)
        elif isinstance(payload.type, str):
            _walk_type(payload.type, "payload", [], types, frozenset(), entries)

    named = _disambiguate(entries)

    seen: set[str] = set()
    deduped_lines: list[str] = []
    for name, path in named:
        line = f":field {name}: {path}"
        if line not in seen:
            seen.add(line)
            deduped_lines.append(line)

    return "".join(f"{line}\n" for line in deduped_lines)
