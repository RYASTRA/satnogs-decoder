from satnogs_decoder.infer.structure import has_fixed_layout_case

FLAT = "meta: {id: f, endian: be}\nseq:\n  - {id: a, type: u1}\n  - {id: b, type: u2}\n"
SWITCHED_WITH_FLAT_CASE = """
meta: {id: sw, endian: be}
seq:
  - {id: kind, type: u1}
  - id: body
    type: {switch-on: kind, cases: {1: case_a}}
types:
  case_a: {seq: [{id: x, type: u2}, {id: y, type: s1}]}
"""
ALL_VARIABLE = """
meta: {id: v, endian: be}
seq:
  - {id: n, type: u1}
  - {id: xs, type: u2, repeat: expr, repeat-expr: n}
types:
  blob: {seq: [{id: raw, size-eos: true}]}
"""

def test_flat_admitted():
    assert has_fixed_layout_case(FLAT) is True

def test_switched_with_flat_case_admitted():
    # the switch case `case_a` is flat -> admit (frame-aware qualify confirms later)
    assert has_fixed_layout_case(SWITCHED_WITH_FLAT_CASE) is True

def test_all_variable_rejected():
    assert has_fixed_layout_case(ALL_VARIABLE) is False

def test_malformed_ksy_skipped_not_crash():
    # non-dict types, non-string field type, and garbage YAML -> False, never raise
    assert has_fixed_layout_case("meta: {id: x}\ntypes: [not, a, dict]\nseq: []") is False
    assert has_fixed_layout_case(": : bad : :") is False
    assert has_fixed_layout_case("meta: {id: x}\nseq:\n  - {id: a, type: [1, 2]}") is False
