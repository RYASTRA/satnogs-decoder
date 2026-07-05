"""Public entry point: v1 field-table spec -> complete, compiling SatNOGS .ksy text.

Pipeline: `load_spec` (schema.py) -> `build_ir` (build.py) -> attach the
`:field` dashboard doc-block (fields.py) -> `KsySpec.to_yaml()`.
"""
from __future__ import annotations

import pathlib

from .build import build_ir
from .fields import field_block
from .schema import load_spec


def generate(source: "str | pathlib.Path") -> str:
    """Turn a v1 field-table spec (path or YAML text) into complete .ksy text."""
    ir = build_ir(load_spec(source))
    ir.doc = field_block(ir)
    return ir.to_yaml()
