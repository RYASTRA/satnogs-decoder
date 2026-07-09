# scripts/build_corpus_db.py — TEMPORARY. Run as a MODULE:
# docker compose run --rm app python -m scripts.build_corpus_db
from __future__ import annotations

import base64
import hashlib
import json
import pathlib
import sys
from dataclasses import dataclass
from urllib.parse import quote

import requests
from scripts.audit_harness import RAW
from satnogs_decoder.infer import corpus
from satnogs_decoder.infer.qualify import qualify, qualify_frame_indices
from satnogs_decoder.shared.satnogs_db import fetch_frames

SATS = pathlib.Path("tests/infer/corpus_sats.json")
DB = "corpus/satnogs_corpus.db"
PROBE_CAP, FULL_CAP = 50, 500
FILE_API = (
    "https://gitlab.com/api/v4/projects/"
    "librespacefoundation%2Fsatnogs%2Fsatnogs-decoders/repository/files/"
)


@dataclass(frozen=True)
class KsySource:
    text: str
    url: str
    sha256: str
    upstream_revision: str | None


def fetch_ksy_source(module: str) -> KsySource:
    raw_url = RAW + f"{module}.ksy"
    file_path = quote(f"ksy/{module}.ksy", safe="")
    try:
        resp = requests.get(FILE_API + file_path, params={"ref": "master"}, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        text = base64.b64decode(payload["content"]).decode()
        revision = (
            payload.get("last_commit_id") or payload.get("commit_id") or payload.get("blob_id")
        )
    except Exception:  # noqa: BLE001
        text = requests.get(raw_url, timeout=60).text
        revision = None
    return KsySource(
        text=text,
        url=raw_url,
        sha256=hashlib.sha256(text.encode()).hexdigest(),
        upstream_revision=revision,
    )


def main() -> int:
    pathlib.Path("corpus").mkdir(exist_ok=True)
    sats = json.loads(SATS.read_text())
    conn = corpus.open_corpus(DB)
    kept, dropped = 0, 0
    for idx, s in enumerate(sats, start=1):
        module, norad = s["module"], s["norad"]
        print(f"[{idx}/{len(sats)}] probe {module} (norad {norad})", file=sys.stderr, flush=True)
        try:
            probe = fetch_frames(norad, start=s["start"], end=s["end"], limit=PROBE_CAP)
        except Exception as e:  # noqa: BLE001
            print(f"drop {module}: fetch failed: {e}", file=sys.stderr, flush=True)
            dropped += 1
            continue
        if len(probe) < 5:
            print(f"drop {module}: only {len(probe)} probe frames", file=sys.stderr, flush=True)
            dropped += 1
            continue
        try:
            source = fetch_ksy_source(module)
        except Exception as e:  # noqa: BLE001
            print(f"drop {module}: ksy fetch failed: {e}", file=sys.stderr, flush=True)
            dropped += 1
            continue
        layout, reason = qualify(source.text, [f.data for f in probe])
        if layout is None:
            print(f"drop {module}: {reason}", file=sys.stderr, flush=True)
            dropped += 1
            continue
        # Full pull, then keep only frames matching the dominant parsed layout.
        print(f"[{idx}/{len(sats)}] full pull {module}", file=sys.stderr, flush=True)
        try:
            full = fetch_frames(norad, start=s["start"], end=s["end"], limit=FULL_CAP)
        except Exception as e:  # noqa: BLE001
            print(f"drop {module}: full fetch failed: {e}", file=sys.stderr, flush=True)
            dropped += 1
            continue
        full_data = [f.data for f in full]
        layout, keep_indexes, reason = qualify_frame_indices(source.text, full_data)
        if layout is None:
            print(
                f"drop {module}: full qualification failed: {reason}",
                file=sys.stderr,
                flush=True,
            )
            dropped += 1
            continue
        keep_frames = [full[i] for i in keep_indexes]
        modal = len(keep_frames[0].data)
        corpus.insert_frames(
            conn,
            norad,
            keep_frames,
            source_module=module,
            fetch_start=s["start"],
            fetch_end=s["end"],
        )
        corpus.insert_layout(conn, norad, layout)
        corpus.insert_satellite_metadata(
            conn,
            corpus.SatelliteMetadata(
                norad=norad,
                module=module,
                fetch_start=s["start"],
                fetch_end=s["end"],
                source_ksy_url=source.url,
                source_ksy_sha256=source.sha256,
                upstream_revision=source.upstream_revision,
                probe_cap=PROBE_CAP,
                full_cap=FULL_CAP,
                accepted_frame_count=len(keep_frames),
                modal_frame_length=modal,
                layout_field_count=len(layout),
                qualification_reason=reason,
            ),
        )
        kept += 1
        print(
            f"keep {module} (norad {norad}): "
            f"{len(keep_frames)} frames @ {modal}B, {len(layout)} fields",
            flush=True,
        )
    corpus.set_meta(conn, "corpus_version", "3")
    corpus.set_meta(conn, "corpus_schema_version", str(corpus.SCHEMA_VERSION))
    corpus.set_meta(conn, "n_sats", str(kept))
    print(f"\ncorpus built: {kept} kept, {dropped} dropped -> {DB}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
