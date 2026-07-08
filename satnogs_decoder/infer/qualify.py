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
import datetime

from satnogs_decoder.infer.labels import extract_layout
from satnogs_decoder.infer.layout import Layout

DOMINANCE_MIN = 0.80
MIN_LEN = 8
MIN_FRAMES = 30
COVERAGE_MIN = 0.85


def _year_before(iso_end: str) -> str:
    """One year before an ISO date (YYYY-MM-DD...), calendar-safe.

    Naive year-minus-1 breaks on Feb 29 (the prior year is never leap), so use
    real date arithmetic and clamp Feb 29 -> Feb 28.
    """
    y, m, d = int(iso_end[:4]), int(iso_end[5:7]), int(iso_end[8:10])
    try:
        return datetime.date(y - 1, m, d).isoformat()
    except ValueError:  # Feb 29 in a non-leap prior year
        return datetime.date(y - 1, m, d - 1).isoformat()


def activity_window(
    launched: str | None, decayed: str | None, status: str, *, now: str
) -> tuple[str, str]:
    # An alive sat gets a recent window (near now); a decayed one ends at decay.
    end = now[:10] if (status.lower() == "alive" or not decayed) else decayed[:10]
    start = _year_before(end)
    if launched and launched[:10] > start:
        start = launched[:10]
    return f"{start}T00:00:00Z", f"{end}T00:00:00Z"


def modal_length(frames: list[bytes]) -> tuple[int, float]:
    if not frames:
        return 0, 0.0
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
    # Coverage gate: the .ksy parse must STRUCTURE most of the frame. A switched
    # decoder can read only the header + a minimal case, leaving most bytes
    # unmodeled -> those positions have no ground-truth layout, so labeling them
    # as "no boundary" would be wrong. Require the parse to reach >= COVERAGE_MIN
    # of the modal frame, else drop (untrustworthy labels).
    covered = max(f.end for f in layout)
    if covered < COVERAGE_MIN * modal:
        return None, f"layout covers only {covered}/{modal}B ({covered / modal:.0%} < {COVERAGE_MIN:.0%})"
    return layout, ""
