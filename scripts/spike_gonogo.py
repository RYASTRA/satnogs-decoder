"""Task 1 go/no-go spike: confirm DB auth + reference-decoder ground truth.

Runs INSIDE the container (deps + token injected via compose env_file / mounted .env).
Never touches the host Python. Run with:  docker compose run --rm app python scripts/spike_gonogo.py

Note: the SatNOGS DB telemetry endpoint's latency scales with the matching-frame
count, and high-volume sats (ELFIN-A ~8M frames, GRBAlpha) time out on wide/unbounded
queries. So we bound to a NARROW historical window when the sat was active.
"""

from __future__ import annotations
import os
import sys

from dotenv import load_dotenv
import requests

load_dotenv()  # compose env_file already injects these; fallback for the mounted .env
TOKEN = os.getenv("satnogs_db_api_key")
NORAD = 47959  # GRBAlpha — very active in 2022
MODULE = "grbalpha"  # satnogsdecoders.<MODULE>; class = MODULE.capitalize()
START = "2022-06-15T00:00:00Z"
END = "2022-06-15T06:00:00Z"  # 6-hour window -> small, fast result set
UA = "satnogs-decoder-dev (research; gitlab.com/RYSATNOGS/satnogs-decoder)"


def main() -> int:
    if not TOKEN:
        print("NO-GO: satnogs_db_api_key not present in the container environment")
        return 1

    url = (
        f"https://db.satnogs.org/api/telemetry/"
        f"?satellite={NORAD}&start={START}&end={END}&format=json"
    )
    r = requests.get(url, headers={"Authorization": f"Token {TOKEN}", "User-Agent": UA}, timeout=60)
    print("API status:", r.status_code)
    if r.status_code != 200:
        print("NO-GO: telemetry API did not return 200")
        print(r.text[:300])
        return 1

    results = r.json().get("results", [])
    print(f"n_results ({START}..{END}):", len(results))
    if not results:
        print("NO-GO: no frames in that window — widen/shift START/END")
        return 1

    first = results[0]
    print("frame keys:", list(first)[:14])
    frame = bytes.fromhex(first["frame"].replace(" ", "").strip())
    print("frame bytes:", len(frame), "first16:", frame[:16].hex())

    # Reference decode via the upstream package — the canonical decode_frame.py path.
    from satnogsdecoders import decoder

    cls_name = MODULE.capitalize()
    if not hasattr(decoder, cls_name):
        cand = [n for n in dir(decoder) if MODULE[:4] in n.lower()]
        print(f"NO-GO: decoder.{cls_name} not found; candidates: {cand}")
        return 1
    klass = getattr(decoder, cls_name)
    fields = decoder.get_fields(klass.from_bytes(frame), empty=False)
    print("decoded field count:", len(fields))
    print("sample:", list(fields.items())[:8])

    if len(fields) >= 1:
        print("\nGO: reference decoder produced non-empty fields from a live frame")
        return 0
    print("\nNO-GO: decoder returned empty fields")
    return 1


if __name__ == "__main__":
    sys.exit(main())
