"""SatNOGS DB telemetry client.

Provides authenticated, date-windowed, cursor-paginated access to
https://db.satnogs.org/api/telemetry/.  Every page fetch is throttled to
respect the ~6 req/min rate limit, and transient 429/5xx errors are retried
with growing backoff.

⚠ REQUIRED DEVIATION from the v3 plan (verified go/no-go finding):
  Unbounded queries timeout for high-volume satellites (>90 s for ELFIN-A).
  fetch_frames therefore requires explicit `start`/`end` date-window params.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import requests
from dotenv import load_dotenv

load_dotenv()

DB_API = "https://db.satnogs.org/api/telemetry/"
USER_AGENT = "satnogs-decoder (gitlab.com/RYSATNOGS/satnogs-decoder)"
PAGE_DELAY = 10  # seconds between pages (~6 req/min good-citizen throttle)


@dataclass(frozen=True)
class Frame:
    """A single raw telemetry frame from the SatNOGS DB."""

    norad: int
    data: bytes
    timestamp: str
    transmitter: str | None = None
    observation_id: int | None = None


def _frame_from_dict(d: dict) -> Frame:
    return Frame(
        norad=int(d["norad_cat_id"]),
        data=bytes.fromhex(d["frame"].replace(" ", "").strip()),
        timestamp=d["timestamp"],
        transmitter=d.get("transmitter"),
        observation_id=d.get("observation_id"),
    )


def parse_frames(results: list[dict]) -> list[Frame]:
    """Parse a list of raw API result dicts into Frame objects."""
    return [_frame_from_dict(d) for d in results]


def fetch_frames(
    norad: int,
    *,
    start: str,
    end: str,
    token: str | None = None,
    limit: int | None = None,
    session: requests.Session | None = None,
) -> list[Frame]:
    """Fetch frames for *norad* within the ISO-8601 date window [start, end].

    Parameters
    ----------
    norad:
        NORAD catalogue ID.
    start:
        ISO-8601 datetime string (inclusive lower bound on `timestamp`).
    end:
        ISO-8601 datetime string (inclusive upper bound on `timestamp`).
    token:
        SatNOGS DB API token.  Falls back to the ``satnogs_db_api_key``
        environment variable (set in ``.env``).  Raises ``RuntimeError``
        when neither is set.
    limit:
        If given, stop after collecting this many frames (across all pages).
    session:
        Optional ``requests.Session`` to reuse (useful for testing).

    Returns
    -------
    list[Frame]
        Collected frames, in the order returned by the API.
    """
    token = token or os.getenv("satnogs_db_api_key")
    if not token:
        raise RuntimeError(
            "SatNOGS DB API token not found.  "
            "Set satnogs_db_api_key in .env (SatNOGS DB → Settings → API Key)."
        )

    sess = session or requests.Session()
    sess.headers.update({"User-Agent": USER_AGENT, "Authorization": f"Token {token}"})

    frames: list[Frame] = []
    url: str | None = f"{DB_API}?satellite={norad}&start={start}&end={end}&format=json"

    while url:
        resp = _get_with_backoff(sess, url)
        frames.extend(parse_frames(resp.json()["results"]))
        if limit is not None and len(frames) >= limit:
            return frames[:limit]
        url = resp.links.get("next", {}).get("url")
        if url:
            time.sleep(PAGE_DELAY)  # good-citizen throttle on every page turn

    return frames


def _get_with_backoff(
    sess: requests.Session,
    url: str,
    *,
    retries: int = 6,
) -> requests.Response:
    """GET *url* with retry + growing backoff on 429/5xx."""
    delay = 10.0
    last_resp: requests.Response | None = None
    last_exc: requests.RequestException | None = None

    for attempt in range(retries):
        try:
            resp = sess.get(url, timeout=30)
        except requests.RequestException as exc:
            last_exc = exc
            if attempt + 1 == retries:
                raise
            time.sleep(delay)
            delay *= 2
            continue
        last_resp = resp

        if resp.status_code == 429:
            wait = float(resp.headers.get("Retry-After", delay))
            time.sleep(wait)
            delay *= 2
            continue

        if resp.status_code in (500, 502, 503, 504):
            time.sleep(delay)
            delay *= 2
            continue

        resp.raise_for_status()
        return resp

    # All retries exhausted — surface the last response or request error.
    if last_resp is not None:
        last_resp.raise_for_status()
        return last_resp  # unreachable, but satisfies type checkers
    assert last_exc is not None
    raise last_exc
