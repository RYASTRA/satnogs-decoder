from __future__ import annotations


def decode_reference(sat_module: str, frame: bytes) -> dict | None:
    """Decode `frame` via the upstream satnogs-decoders package (ground truth).

    Mirrors the package's own decode_frame.py verbatim:
    getattr(decoder, name.capitalize()).from_bytes(...) -> get_fields(empty=False).

    `sat_module` e.g. 'grbalpha'. Returns a flat field dict, or None on any failure.
    """
    try:
        from satnogsdecoders import decoder  # re-exports every per-sat class

        cls = getattr(decoder, sat_module.capitalize())
        return decoder.get_fields(cls.from_bytes(frame), empty=False)
    except Exception:
        return None
