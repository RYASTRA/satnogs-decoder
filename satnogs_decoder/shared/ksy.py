from __future__ import annotations
import io
from dataclasses import dataclass
from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import DoubleQuotedScalarString as DQ


@dataclass
class KsyField:
    id: str
    type: str
    doc: str | None = None
    doc_ref: str | None = None
    enum: str | None = None
    scale: float | None = None
    offset: float | None = None
    unit: str | None = None


@dataclass
class KsySpec:
    id: str
    endian: str
    seq: list[KsyField]
    title: str | None = None
    ks_version: str | None = None
    imports: list[str] | None = None
    types: dict[str, list[KsyField]] | None = None
    enums: dict[str, dict[int, str]] | None = None

    def _field(self, f: KsyField) -> dict:
        e: dict = {"id": f.id, "type": f.type}
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
        return e

    def to_dict(self) -> dict:
        meta: dict = {"id": self.id, "endian": self.endian}
        if self.title:
            meta["title"] = self.title
        if self.ks_version:
            meta["ks-version"] = self.ks_version
        if self.imports:
            meta["imports"] = list(self.imports)
        out: dict = {"meta": meta, "seq": [self._field(f) for f in self.seq]}
        if self.types:
            out["types"] = {
                n: {"seq": [self._field(f) for f in fs]}
                for n, fs in self.types.items()
            }
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
