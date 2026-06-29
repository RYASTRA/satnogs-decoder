from __future__ import annotations

import importlib.util
import io
import pathlib
import subprocess
import tempfile
from dataclasses import dataclass

from kaitaistruct import KaitaiStream, KaitaiStruct
from ruamel.yaml import YAML

_SCALAR = (int, float, str, bytes, bool)


@dataclass
class ParseResult:
    ok: bool
    consumed: int
    total: int
    fields: dict[str, object]
    error: str | None = None


def _meta_id(ksy_text: str) -> str:
    doc = YAML(typ="safe").load(io.StringIO(ksy_text))
    try:
        return doc["meta"]["id"]
    except (KeyError, TypeError) as e:
        raise ValueError("ksy_text has no meta/id") from e


def compile_ksy(
    ksy_text: str,
    *,
    ksc: str | list[str] = "ksc",
    import_dirs: list[str] | None = None,
) -> type:
    """Compile a .ksy spec string to a Python class via ksc.

    Returns the generated class (id parts joined with capitalize()).
    Raises RuntimeError if ksc exits non-zero.
    """
    ks_id = _meta_id(ksy_text)
    tmp = pathlib.Path(tempfile.mkdtemp())
    (tmp / f"{ks_id}.ksy").write_text(ksy_text)

    cmd = [ksc] if isinstance(ksc, str) else list(ksc)
    for d in import_dirs or []:
        cmd += ["--import-path", d]
    cmd += ["--target", "python", "--outdir", str(tmp), str(tmp / f"{ks_id}.ksy")]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ksc failed for id={ks_id}:\n{proc.stderr}")

    spec = importlib.util.spec_from_file_location(ks_id, tmp / f"{ks_id}.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load generated module for id={ks_id}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return getattr(module, "".join(p.capitalize() for p in ks_id.split("_")))


def _flatten(obj: object, prefix: str = "") -> dict[str, object]:
    """Recursively extract scalar fields from a KaitaiStruct instance.

    Walks vars(obj), keeps scalars, recurses into KaitaiStruct sub-objects
    and lists thereof. Skips private attributes (starting with '_') and
    instances (@property), which are not present in vars().
    """
    out: dict[str, object] = {}
    for k, v in vars(obj).items():
        if k.startswith("_"):
            continue
        name = f"{prefix}{k}"
        if isinstance(v, _SCALAR):
            out[name] = v
        elif isinstance(v, KaitaiStruct):
            out.update(_flatten(v, name + "."))
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, KaitaiStruct):
                    out.update(_flatten(item, f"{name}[{i}]."))
                elif isinstance(item, _SCALAR):
                    out[f"{name}[{i}]"] = item
    return out


def parse(parser_cls: type, data: bytes) -> ParseResult:
    """Parse `data` with `parser_cls` (a compiled Kaitai class).

    The generated __init__ already calls _read(), so we do NOT call it again.
    Returns ParseResult(ok=True) on success, or ok=False with error on any exception.
    """
    stream = KaitaiStream(io.BytesIO(data))
    try:
        obj = parser_cls(stream)  # __init__ runs _read() internally
        return ParseResult(ok=True, consumed=stream.pos(), total=len(data), fields=_flatten(obj))
    except Exception as e:  # noqa: BLE001 — parse failure is a result, not a crash
        return ParseResult(
            ok=False,
            consumed=stream.pos(),
            total=len(data),
            fields={},
            error=str(e),
        )
