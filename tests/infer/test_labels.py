import pytest
from satnogs_decoder.infer.labels import extract_layout

SWITCHED = """
meta: {id: sw, endian: be}
seq:
  - {id: kind, type: u1}
  - id: body
    type: {switch-on: kind, cases: {1: case_a, 2: case_b}}
types:
  case_a: {seq: [{id: a1, type: u2}, {id: a2, type: s1}]}
  case_b: {seq: [{id: b1, type: u1}]}
"""

STR_KSY = """
meta: {id: st, endian: be}
seq:
  - {id: call, type: str, size: 3, encoding: ASCII}
  - {id: v, type: s1}
"""

@pytest.mark.slow
def test_switched_case_a_layout():
    # kind=1 -> case_a: a1=u2[1,3), a2=s1[3,4)
    layout = extract_layout(SWITCHED, bytes([0x01, 0x11, 0x22, 0xFF]))
    assert [(f.name, f.start, f.end, f.signed) for f in layout] == [
        ("kind", 0, 1, False),
        ("body.a1", 1, 3, False),
        ("body.a2", 3, 4, True),
    ]

@pytest.mark.slow
def test_switched_case_b_layout():
    # kind=2 -> case_b: b1=u1[1,2)
    layout = extract_layout(SWITCHED, bytes([0x02, 0x55]))
    assert [(f.name, f.start, f.end) for f in layout] == [("kind", 0, 1), ("body.b1", 1, 2)]

@pytest.mark.slow
def test_str_field_not_signed():
    # 'str' starts with 's' but must NOT be marked signed
    layout = extract_layout(STR_KSY, b"ABC\xff")
    assert layout[0].name == "call" and layout[0].signed is False and layout[0].width == 3
    assert layout[1].name == "v" and layout[1].signed is True
