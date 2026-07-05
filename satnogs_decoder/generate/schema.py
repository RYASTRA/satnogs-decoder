"""v1 field-table spec loader.

Parses and validates a hand-authored field-table YAML spec (the input to
`generate.build.build_ir`) into typed, frozen dataclasses. Every validation
failure raises `SpecError` with the offending token embedded in the message.
"""
from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass

from ruamel.yaml import YAML

_INT_RE = re.compile(r"^(u[1248]|s[1248]|f[48])(le|be)?$")
_BIT_RE = re.compile(r"^b([1-9]|[1-5][0-9]|6[0-4])$")   # b1..b64
_UINT_RE = re.compile(r"^u[1248](le|be)?$")


def _known(t: str, types: set[str]) -> bool:
    return t == "str" or t in types or bool(_INT_RE.match(t)) or bool(_BIT_RE.match(t))


class SpecError(ValueError):
    """Raised for any structurally or semantically invalid field-table spec."""


@dataclass(frozen=True)
class FieldSpec:
    id: str
    type: str
    enum: str | None = None
    doc: str | None = None
    doc_ref: str | None = None


@dataclass(frozen=True)
class InstanceSpec:
    id: str
    value: str
    unit: str | None = None
    enum: str | None = None
    doc: str | None = None


@dataclass(frozen=True)
class FrameType:
    id: str
    match: "int | str"
    seq: "tuple[FieldSpec, ...]"
    instances: "tuple[InstanceSpec, ...]"


@dataclass(frozen=True)
class SubType:
    seq: "tuple[FieldSpec, ...]"
    instances: "tuple[InstanceSpec, ...]"


@dataclass(frozen=True)
class Discriminator:
    pos: int
    type: str


@dataclass(frozen=True)
class Spec:
    id: str
    endian: str
    transport: str
    frame_types: "tuple[FrameType, ...]"
    types: "dict[str, SubType]"
    enums: "dict[str, dict[int, str]]"
    discriminator: "Discriminator | None"
    title: str | None
    ks_version: str | None
    doc_ref: str | None


def _load_yaml(source: "str | pathlib.Path") -> dict:
    yaml = YAML(typ="safe")
    if isinstance(source, pathlib.Path) or (
        isinstance(source, str) and "\n" not in source and source.endswith((".yaml", ".yml"))
    ):
        text = pathlib.Path(source).read_text()
    else:
        text = source
    data = yaml.load(text)
    if not isinstance(data, dict):
        raise SpecError("spec root must be a mapping (got empty/invalid YAML)")
    return data


def _parse_fields(raw_seq: "list | None", where: str, known_types: set[str]) -> "tuple[FieldSpec, ...]":
    raw_seq = raw_seq or []
    out: list[FieldSpec] = []
    seen: set[str] = set()
    for raw in raw_seq:
        if not isinstance(raw, dict) or "id" not in raw:
            raise SpecError(f"{where}: field missing required key 'id': {raw!r}")
        fid = raw["id"]
        if fid in seen:
            raise SpecError(f"{where}: duplicate field id {fid!r}")
        seen.add(fid)
        if "type" not in raw:
            raise SpecError(f"{where}: field {fid!r} missing required key 'type'")
        ftype = raw["type"]
        if not _known(ftype, known_types):
            raise SpecError(f"{where}: field {fid!r} has unknown type {ftype!r}")
        fenum = raw.get("enum")
        out.append(FieldSpec(
            id=fid, type=ftype, enum=fenum,
            doc=raw.get("doc"), doc_ref=raw.get("doc-ref") or raw.get("doc_ref"),
        ))
    return tuple(out)


def _parse_instances(raw_insts: "list | None", where: str) -> "tuple[InstanceSpec, ...]":
    raw_insts = raw_insts or []
    out: list[InstanceSpec] = []
    seen: set[str] = set()
    for raw in raw_insts:
        if not isinstance(raw, dict) or "id" not in raw:
            raise SpecError(f"{where}: instance missing required key 'id': {raw!r}")
        iid = raw["id"]
        if iid in seen:
            raise SpecError(f"{where}: duplicate instance id {iid!r}")
        seen.add(iid)
        if "value" not in raw:
            raise SpecError(f"{where}: instance {iid!r} missing required key 'value'")
        ivalue = raw["value"]
        if not isinstance(ivalue, str) or not ivalue:
            raise SpecError(f"{where}: instance {iid!r} 'value' must be a non-empty string, got {ivalue!r}")
        out.append(InstanceSpec(
            id=iid, value=raw["value"], unit=raw.get("unit"),
            enum=raw.get("enum"), doc=raw.get("doc"),
        ))
    return tuple(out)


def _check_enum_refs(fields: "tuple[FieldSpec, ...]", enums: dict, where: str) -> None:
    for f in fields:
        if f.enum is not None and f.enum not in enums:
            raise SpecError(f"{where}: field {f.id!r} references unknown enum {f.enum!r}")


def load_spec(source: "str | pathlib.Path") -> Spec:
    data = _load_yaml(source)

    meta = data.get("meta") or {}
    if not isinstance(meta, dict) or "id" not in meta:
        raise SpecError("meta.id is required")
    spec_id = meta["id"]
    if "endian" not in meta:
        raise SpecError("meta.endian is required")
    endian = meta["endian"]
    if endian not in ("le", "be"):
        raise SpecError(f"meta.endian must be 'le' or 'be', got {endian!r}")

    transport = data.get("transport", "none")
    if transport not in ("ax25", "csp", "none"):
        raise SpecError(f"transport must be one of 'ax25', 'csp', 'none', got {transport!r}")

    raw_enums = data.get("enums") or {}
    if not isinstance(raw_enums, dict):
        raise SpecError("enums must be a mapping")
    enums: dict[str, dict[int, str]] = {}
    for name, mapping in raw_enums.items():
        if not isinstance(mapping, dict):
            raise SpecError(f"enum {name!r} must be a mapping of int -> name")
        enums[name] = dict(mapping)

    raw_types = data.get("types") or {}
    if not isinstance(raw_types, dict):
        raise SpecError("types must be a mapping")
    known_type_names = set(raw_types)

    # one-level nesting only: a sub-type's own seq may not reference another sub-type
    types: dict[str, SubType] = {}
    for name, tdef in raw_types.items():
        if not isinstance(tdef, dict):
            raise SpecError(f"types.{name}: must be a mapping with 'seq'")
        where = f"types.{name}"
        fields = _parse_fields(tdef.get("seq"), where, known_type_names - {name})
        for f in fields:
            if f.type in known_type_names:
                raise SpecError(
                    f"types.{name}: field {f.id!r} references sub-type {f.type!r} — "
                    f"nested sub-types may only be referenced from frame_types (one-level nesting)"
                )
        _check_enum_refs(fields, enums, where)
        insts = _parse_instances(tdef.get("instances"), where)
        types[name] = SubType(seq=fields, instances=insts)

    raw_frame_types = data.get("frame_types")
    if not raw_frame_types:
        raise SpecError("frame_types is required and must be non-empty")
    frame_types: list[FrameType] = []
    seen_ft_ids: set[str] = set()
    seen_matches: set = set()
    for raw in raw_frame_types:
        if not isinstance(raw, dict) or "id" not in raw:
            raise SpecError(f"frame_types: entry missing required key 'id': {raw!r}")
        ft_id = raw["id"]
        if ft_id in seen_ft_ids:
            raise SpecError(f"frame_types: duplicate frame_type id {ft_id!r}")
        if ft_id in known_type_names:
            raise SpecError(
                f"frame_types.{ft_id}: frame_type id {ft_id!r} collides with a declared "
                f"types.{ft_id!r} sub-type — dispatch and sub-type names must be distinct"
            )
        seen_ft_ids.add(ft_id)
        where = f"frame_types.{ft_id}"
        fields = _parse_fields(raw.get("seq"), where, known_type_names)
        _check_enum_refs(fields, enums, where)
        for f in fields:
            if f.type in known_type_names and f.type not in types:
                raise SpecError(f"{where}: field {f.id!r} references undeclared sub-type {f.type!r}")
        insts = _parse_instances(raw.get("instances"), where)
        match = raw.get("match", "default")
        if match in seen_matches:
            if match == "default":
                raise SpecError(
                    f"{where}: more than one frame_type has match: default — "
                    f"only one default dispatch case is allowed"
                )
            raise SpecError(
                f"{where}: duplicate discriminator match value {match!r} — "
                f"each frame_type must have a unique match"
            )
        seen_matches.add(match)
        frame_types.append(FrameType(id=ft_id, match=match, seq=fields, instances=insts))

    discriminator: Discriminator | None = None
    raw_disc = data.get("discriminator")
    if raw_disc is not None:
        if not isinstance(raw_disc, dict) or "pos" not in raw_disc or "type" not in raw_disc:
            raise SpecError("discriminator must be a mapping with 'pos' and 'type'")
        disc_type = raw_disc["type"]
        if not isinstance(disc_type, str) or not _UINT_RE.match(disc_type):
            raise SpecError(
                f"discriminator.type must be an unsigned integer type "
                f"(u1/u2/u4/u8, optional le/be suffix), got {disc_type!r}"
            )
        discriminator = Discriminator(pos=raw_disc["pos"], type=disc_type)

    if len(frame_types) > 1 and discriminator is None:
        raise SpecError(
            "discriminator is required when more than one frame_type is declared "
            f"(got {len(frame_types)}: {[ft.id for ft in frame_types]})"
        )

    return Spec(
        id=spec_id, endian=endian, transport=transport,
        frame_types=tuple(frame_types), types=types, enums=enums,
        discriminator=discriminator,
        title=meta.get("title"), ks_version=meta.get("ks-version") or meta.get("ks_version"),
        doc_ref=data.get("doc-ref") or data.get("doc_ref"),
    )
