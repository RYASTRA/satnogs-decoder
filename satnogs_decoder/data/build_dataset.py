"""Benchmark dataset builder.

Assembles raw telemetry frames (optionally paired with reference-decoded
ground-truth JSON) into a Hugging Face ``datasets.Dataset``.
"""
from __future__ import annotations

import json

from datasets import Dataset

from satnogs_decoder.shared.satnogs_db import Frame
from satnogs_decoder.shared.reference import decode_reference


def _bytes_safe_json(d: dict) -> str | None:
    """Serialize *d* to a JSON string, encoding bytes values as hex.

    Returns None if serialisation fails for any reason.
    """
    try:
        return json.dumps(
            d,
            default=lambda b: b.hex() if isinstance(b, (bytes, bytearray)) else str(b),
        )
    except (TypeError, ValueError):
        return None


def rows_from_frames(
    frames: list[Frame],
    sat_module: str | None = None,
) -> list[dict]:
    """Convert a list of *frames* into dataset row dicts.

    Parameters
    ----------
    frames:
        Raw telemetry frames (typically for a single NORAD ID).
    sat_module:
        If given, call ``decode_reference(sat_module, frame.data)`` for each
        frame and store the result (bytes-safe JSON) in ``decoded_json``.
        When the decoder returns an empty dict or raises, ``decoded_json``
        is set to None.

    Returns
    -------
    list[dict]
        One dict per frame with keys:
        ``norad``, ``frame_hex``, ``timestamp``, ``n_bytes``,
        ``transmitter``, ``observation_id``, ``decoded_json``.
    """
    rows: list[dict] = []
    successful_decodes = 0
    for f in frames:
        decoded_json: str | None = None
        if sat_module is not None:
            d = decode_reference(sat_module, f.data)
            if d:
                decoded_json = _bytes_safe_json(d)
                successful_decodes += 1
        rows.append(
            {
                "norad": f.norad,
                "frame_hex": f.data.hex(),
                "timestamp": f.timestamp,
                "n_bytes": len(f.data),
                "transmitter": f.transmitter,
                "observation_id": f.observation_id,
                "decoded_json": decoded_json,
            }
        )
    if sat_module is not None and frames and successful_decodes == 0:
        raise RuntimeError(
            f"sat_module={sat_module!r} produced zero successful decodes across "
            f"{len(frames)} frame(s) — check the module name or frame data"
        )
    return rows


def build_dataset(
    frames_by_norad: dict[int, list[Frame]],
    modules: dict[int, str] | None = None,
) -> Dataset:
    """Flatten frames across all NORAD IDs into a single HF Dataset.

    Parameters
    ----------
    frames_by_norad:
        Mapping from NORAD ID to list of frames.
    modules:
        Optional mapping from NORAD ID to sat_module name used for
        reference decoding.  Missing keys → no decode for that NORAD.

    Returns
    -------
    datasets.Dataset

    Raises
    ------
    ValueError
        When there are no frames at all (empty input or all lists empty).
    """
    modules = modules or {}
    rows: list[dict] = []
    for norad, frames in frames_by_norad.items():
        rows.extend(rows_from_frames(frames, modules.get(norad)))
    if not rows:
        raise ValueError("no frames to build a dataset from")
    return Dataset.from_list(rows)


def push(ds: Dataset, repo_id: str) -> None:
    """Push *ds* to the Hugging Face Hub at *repo_id*."""
    ds.push_to_hub(repo_id)
