from __future__ import annotations
import io
from dataclasses import dataclass
from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import DoubleQuotedScalarString as DQ
from ruamel.yaml.scalarstring import LiteralScalarString as LS


@dataclass
class KsySwitch:
    on: str                          # switch-on expression
    cases: dict                      # {int_or_"_": subtype_name}


@dataclass
class KsyField:
    id: str
    type: "str | KsySwitch"
    doc: str | None = None
    doc_ref: str | None = None
    enum: str | None = None
    scale: float | None = None
    offset: float | None = None
    unit: str | None = None
    process: str | None = None
    size: int | None = None
    encoding: str | None = None


@dataclass
class KsyInstance:
    id: str
    value: str | None = None        # computed expression  → {value: ...}
    pos: int | None = None          # OR positional read   → {pos: N, type: ...}
    type: str | None = None
    enum: str | None = None
    doc: str | None = None
    unit: str | None = None
    scale: float | None = None
    offset: float | None = None


@dataclass
class KsyType:
    seq: list[KsyField]
    instances: list[KsyInstance] | None = None
    doc_ref: str | None = None


@dataclass
class KsySpec:
    id: str
    endian: str
    seq: list[KsyField]
    title: str | None = None
    ks_version: str | None = None
    imports: list[str] | None = None
    types: "dict[str, KsyType | list[KsyField]] | None" = None
    enums: dict[str, dict[int, str]] | None = None
    doc: str | None = None
    instances: list[KsyInstance] | None = None

    def _field(self, f: KsyField) -> dict:
        e: dict = {"id": f.id}
        if isinstance(f.type, KsySwitch):
            e["type"] = {"switch-on": f.type.on, "cases": dict(f.type.cases)}
        else:
            e["type"] = f.type
        if f.enum:
            e["enum"] = f.enum
        extra = [x for x in (
            f"unit={f.unit}" if f.unit else "",
            f"scale={f.scale}" if f.scale is not None else "",
            f"offset={f.offset}" if f.offset is not None else "",
        ) if x]
        doc = f.doc
        if extra:  # scale/offset/unit have no native .ksy key; keep them in doc (lossless)
            doc = (doc + " " if doc else "") + "[" + ", ".join(extra) + "]"
        if doc:
            e["doc"] = doc
        if f.doc_ref:
            e["doc-ref"] = f.doc_ref
        if f.size is not None:
            e["size"] = f.size
        if f.encoding:
            e["encoding"] = f.encoding
        if f.process:
            e["process"] = f.process
        return e

    def _instances(self, insts: list[KsyInstance]) -> dict:
        out: dict = {}
        for it in insts:
            e: dict = {}
            if it.pos is not None:
                e["pos"] = it.pos
                if it.type:
                    e["type"] = it.type
            else:
                e["value"] = it.value
                if it.type:
                    e["type"] = it.type
            if it.enum:
                e["enum"] = it.enum
            extra = [x for x in (f"unit={it.unit}" if it.unit else "",
                                 f"scale={it.scale}" if it.scale is not None else "",
                                 f"offset={it.offset}" if it.offset is not None else "") if x]
            doc = it.doc
            if extra:
                doc = (doc + " " if doc else "") + "[" + ", ".join(extra) + "]"
            if doc:
                e["doc"] = doc
            out[it.id] = e
        return out

    def to_dict(self) -> dict:
        meta: dict = {"id": self.id, "endian": self.endian}
        if self.title:
            meta["title"] = self.title
        if self.ks_version:
            meta["ks-version"] = self.ks_version
        if self.imports:
            meta["imports"] = list(self.imports)
        if self.doc:
            meta["doc"] = LS(self.doc)
        out: dict = {"meta": meta, "seq": [self._field(f) for f in self.seq]}
        if self.instances:
            out["instances"] = self._instances(self.instances)
        if self.types:
            types_out: dict = {}
            for n, t in self.types.items():
                if isinstance(t, KsyType):
                    td: dict = {"seq": [self._field(f) for f in t.seq]}
                    if t.instances:
                        td["instances"] = self._instances(t.instances)
                    if t.doc_ref:
                        td["doc-ref"] = t.doc_ref
                    types_out[n] = td
                else:
                    types_out[n] = {"seq": [self._field(f) for f in t]}
            out["types"] = types_out
        if self.enums:
            out["enums"] = {
                en: {k: DQ(v) for k, v in mp.items()}
                for en, mp in self.enums.items()
            }
        return out

    def to_yaml(self) -> str:
        yaml = YAML()
        yaml.default_flow_style = False
        buf = io.StringIO()
        yaml.dump(self.to_dict(), buf)
        return buf.getvalue()
