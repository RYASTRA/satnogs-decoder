"""Task 1 go/no-go spike: confirm DB auth + reference-decoder ground truth.

Runs INSIDE the container (deps + token injected via compose env_file / mounted .env).
Never touches the host Python. Run with:  docker compose run --rm app python scripts/spike_gonogo.py
"""
from __future__ import annotations
import os
import sys

from dotenv import load_dotenv
import requests

load_dotenv()  # compose env_file already injects these; this is a fallback for the mounted .env
TOKEN = os.getenv("satnogs_db_api_key")
NORAD = 43617  # ELFIN-A
UA = "satnogs-decoder-dev (research; gitlab.com/RYASTRA/satnogs-decoder)"


def main() -> int:
    if not TOKEN:
        print("NO-GO: satnogs_db_api_key not present in the container environment")
        return 1

    r = requests.get(
        f"https://db.satnogs.org/api/telemetry/?satellite={NORAD}&format=json",
        headers={"Authorization": f"Token {TOKEN}", "User-Agent": UA},
        timeout=30,
    )
    print("API status:", r.status_code)
    if r.status_code != 200:
        print("NO-GO: telemetry API did not return 200")
        print(r.text[:300])
        return 1

    body = r.json()
    print("body keys:", list(body)[:6])
    results = body.get("results", [])
    print("n_results:", len(results))
    if not results:
        print(f"NO-GO: no frames returned for NORAD {NORAD}")
        return 1

    first = results[0]
    print("frame keys:", list(first)[:14])
    frame = bytes.fromhex(first["frame"].replace(" ", "").strip())
    print("frame bytes:", len(frame), "first16:", frame[:16].hex())

    # Reference decode via the upstream package — the canonical decode_frame.py path.
    from satnogsdecoders import decoder

    klass = getattr(decoder, "Elfin")
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
