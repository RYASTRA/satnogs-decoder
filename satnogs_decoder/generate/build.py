"""Spec -> .ksy IR builder.

Composes the v1 field-table `Spec` (schema.py) with the canonical transport
header IR (headers.py) into one complete `KsySpec`: transport header wrapper
+ frame-type `switch-on` dispatch + scaling instances + one-level nesting +
enums. This is the crux of `generate`.
"""
from __future__ import annotations

from .schema import FieldSpec, InstanceSpec, Spec, SpecError
from .headers import header
from satnogs_decoder.shared.ksy import KsyField, KsyInstance, KsySpec, KsySwitch, KsyType


def _fields(seq: "tuple[FieldSpec, ...]") -> list[KsyField]:
    return [KsyField(id=f.id, type=f.type, enum=f.enum, doc=f.doc, doc_ref=f.doc_ref) for f in seq]


def _insts(insts: "tuple[InstanceSpec, ...]") -> "list[KsyInstance] | None":
    out = [KsyInstance(id=i.id, value=i.value, unit=i.unit, enum=i.enum, doc=i.doc) for i in insts]
    return out or None


def build_ir(spec: Spec) -> KsySpec:
    hdr_seq, hdr_types = header(spec.transport)
    types: dict[str, KsyType | list[KsyField]] = {}
    for _hname, _htype in hdr_types.items():
        types[_hname] = _htype
    for name, st in spec.types.items():
        if name in types:
            raise SpecError(
                f"types.{name}: name collides with a reserved '{spec.transport}' transport "
                f"header type {name!r} — rename this sub-type"
            )
        types[name] = KsyType(seq=_fields(st.seq), instances=_insts(st.instances))
    for ft in spec.frame_types:
        if ft.id in types:
            raise SpecError(
                f"frame_types.{ft.id}: frame_type id collides with a reserved "
                f"'{spec.transport}' transport header type {ft.id!r} — rename this frame_type"
            )
        types[ft.id] = KsyType(seq=_fields(ft.seq), instances=_insts(ft.instances))
    top_instances: list[KsyInstance] = []
    if spec.discriminator:
        cases = {("_" if ft.match == "default" else ft.match): ft.id for ft in spec.frame_types}
        payload = KsyField(id="payload", type=KsySwitch(on="kind", cases=cases))
        top_instances.append(KsyInstance(id="kind", pos=spec.discriminator.pos, type=spec.discriminator.type))
    else:
        payload = KsyField(id="payload", type=spec.frame_types[0].id)
    return KsySpec(
        id=spec.id, endian=spec.endian, seq=hdr_seq + [payload],
        title=spec.title, ks_version=spec.ks_version,
        types=types, enums=spec.enums or None,
        instances=top_instances or None,
    )
