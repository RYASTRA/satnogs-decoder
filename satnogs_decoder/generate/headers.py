from satnogs_decoder.shared.ksy import KsyField, KsyInstance, KsyType


def _ax25() -> tuple[list[KsyField], dict[str, KsyType]]:
    seq = [
        KsyField(id="ax25_dest_callsign_raw", type="ax25_callsign_raw"),
        KsyField(id="ax25_dest_ssid_raw", type="ax25_ssid_mask"),
        KsyField(id="ax25_src_callsign_raw", type="ax25_callsign_raw"),
        KsyField(id="ax25_src_ssid_raw", type="ax25_ssid_mask"),
        KsyField(id="ax25_ctl", type="u1"),
        KsyField(id="ax25_pid", type="u1"),
    ]
    types = {
        "ax25_callsign_raw": KsyType(
            seq=[
                KsyField(id="callsign_ror", type="ax25_callsign", process="ror(1)", size=6),
            ]
        ),
        "ax25_callsign": KsyType(
            seq=[
                KsyField(id="callsign", type="str", size=6, encoding="ASCII"),
            ]
        ),
        "ax25_ssid_mask": KsyType(
            seq=[KsyField(id="ssid_mask", type="u1")],
            instances=[KsyInstance(id="ssid", value="(ssid_mask & 0x1e) >> 1")],
        ),
    }
    return seq, types


def _csp() -> tuple[list[KsyField], dict[str, KsyType]]:
    seq = [
        KsyField(id="csp_header", type="u4be", doc="CSP 32-bit header (prio/src/dst/ports/flags)")
    ]
    return seq, {}


def header(transport: str) -> tuple[list[KsyField], dict[str, KsyType]]:
    if transport == "ax25":
        return _ax25()
    if transport == "csp":
        return _csp()
    if transport == "none":
        return [], {}
    raise ValueError(f"unknown transport: {transport}")
