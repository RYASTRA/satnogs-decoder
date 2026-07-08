"""Frame-aware candidate qualification (spec §4). Dev-only; deleted at finalization.

Decides whether a satellite has a fixed-layout DOMINANT frame-type case,
using real frames: the modal frame-length group must dominate (a variable/
repeat/size-eos case fails this), be long enough, have enough frames, and a
representative modal-length frame must label cleanly via the full canonical
.ksy. `activity_window` derives a pull window from the sat's launched/decayed
metadata so we never guess.
"""
from __future__ import annotations

import collections

from satnogs_decoder.infer.labels import extract_layout
from satnogs_decoder.infer.layout import Layout

DOMINANCE_MIN = 0.80
MIN_LEN = 8
MIN_FRAMES = 30


def _year_before(iso_end: str) -> str:
    # crude: subtract 1 from the year field of an ISO date; good enough for a window start
    y = int(iso_end[:4]) - 1
    return f"{y}{iso_end[4:]}"


def activity_window(launched: str | None, decayed: str | None, status: str, *, now: str) -> tuple[str, str]:
    end = decayed[:10] if decayed else now[:10]
    start = _year_before(end)
    if launched and launched[:10] > start:
        start = launched[:10]
    return f"{start}T00:00:00Z", f"{end}T00:00:00Z"


def modal_length(frames: list[bytes]) -> tuple[int, float]:
    counts = collections.Counter(len(f) for f in frames)
    modal, n = counts.most_common(1)[0]
    return modal, n / len(frames)


def qualify(ksy_text: str, frames: list[bytes], *, import_dirs: list[str] | None = None
            ) -> tuple[Layout | None, str]:
    if not frames:
        return None, "no frames"
    modal, frac = modal_length(frames)
    at_modal = [f for f in frames if len(f) == modal]
    if frac < DOMINANCE_MIN:
        return None, f"no dominant length (modal {modal} at {frac:.0%} < {DOMINANCE_MIN:.0%})"
    if modal < MIN_LEN:
        return None, f"modal length {modal} too short (< {MIN_LEN})"
    if len(at_modal) < MIN_FRAMES:
        return None, f"only {len(at_modal)} frames at modal length (< {MIN_FRAMES})"
    try:
        layout = extract_layout(ksy_text, at_modal[0], import_dirs=import_dirs)
    except Exception as e:  # noqa: BLE001
        return None, f"label extraction failed: {e}"
    if not layout:
        return None, "empty layout"
    return layout, ""
