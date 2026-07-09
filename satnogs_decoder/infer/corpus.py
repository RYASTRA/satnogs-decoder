"""The temporary SQLite training corpus (spec §4). Deleted at finalization.

Three tables: `frames` (raw bytes per sat), `layouts` (true field spans per
sat — the labels), `meta` (key/value provenance). Portable, serverless,
git-ignored.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from satnogs_decoder.infer.layout import FieldSpan, Layout

SCHEMA_VERSION = 3


@dataclass(frozen=True)
class FrameRecord:
    norad: int
    idx: int
    data: bytes
    timestamp: str | None = None
    transmitter: str | None = None
    observation_id: int | None = None
    source_module: str | None = None
    fetch_start: str | None = None
    fetch_end: str | None = None


@dataclass(frozen=True)
class SatelliteMetadata:
    norad: int
    module: str
    fetch_start: str
    fetch_end: str
    source_ksy_url: str
    source_ksy_sha256: str
    upstream_revision: str | None
    probe_cap: int
    full_cap: int
    accepted_frame_count: int
    modal_frame_length: int
    layout_field_count: int
    qualification_reason: str = ""


_SCHEMA = """
CREATE TABLE IF NOT EXISTS frames (
    norad          INTEGER NOT NULL,
    idx            INTEGER NOT NULL,
    data           BLOB    NOT NULL,
    timestamp      TEXT,
    transmitter    TEXT,
    observation_id INTEGER,
    source_module  TEXT,
    fetch_start    TEXT,
    fetch_end      TEXT,
    PRIMARY KEY (norad, idx)
);
CREATE TABLE IF NOT EXISTS layouts (
    norad     INTEGER NOT NULL,
    field_idx INTEGER NOT NULL,
    start     INTEGER NOT NULL,
    end       INTEGER NOT NULL,
    width     INTEGER NOT NULL,
    signed    INTEGER NOT NULL,
    is_enum   INTEGER NOT NULL,
    name      TEXT,
    PRIMARY KEY (norad, field_idx)
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS satellites (
    norad                INTEGER PRIMARY KEY,
    module               TEXT    NOT NULL,
    fetch_start          TEXT    NOT NULL,
    fetch_end            TEXT    NOT NULL,
    source_ksy_url       TEXT    NOT NULL,
    source_ksy_sha256    TEXT    NOT NULL,
    upstream_revision    TEXT,
    probe_cap            INTEGER NOT NULL,
    full_cap             INTEGER NOT NULL,
    accepted_frame_count INTEGER NOT NULL,
    modal_frame_length   INTEGER NOT NULL,
    layout_field_count   INTEGER NOT NULL,
    qualification_reason TEXT    NOT NULL DEFAULT ''
);
"""


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(r[1]) for r in conn.execute(f"PRAGMA table_info({table})")}


def _migrate(conn: sqlite3.Connection) -> None:
    frame_additions = {
        "timestamp": "ALTER TABLE frames ADD COLUMN timestamp TEXT",
        "transmitter": "ALTER TABLE frames ADD COLUMN transmitter TEXT",
        "observation_id": "ALTER TABLE frames ADD COLUMN observation_id INTEGER",
        "source_module": "ALTER TABLE frames ADD COLUMN source_module TEXT",
        "fetch_start": "ALTER TABLE frames ADD COLUMN fetch_start TEXT",
        "fetch_end": "ALTER TABLE frames ADD COLUMN fetch_end TEXT",
    }
    frame_columns = _columns(conn, "frames")
    for column, statement in frame_additions.items():
        if column not in frame_columns:
            conn.execute(statement)
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
        ("corpus_schema_version", str(SCHEMA_VERSION)),
    )


def open_corpus(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.executescript(_SCHEMA)
    _migrate(conn)
    conn.commit()
    return conn


def _frame_data(frame: object) -> bytes:
    if isinstance(frame, bytes):
        return frame
    if isinstance(frame, bytearray | memoryview):
        return bytes(frame)
    data = getattr(frame, "data", None)
    if data is None:
        raise TypeError(f"frame object {type(frame).__name__} has no data attribute")
    return bytes(data)


def insert_frames(
    conn: sqlite3.Connection,
    norad: int,
    frames: Sequence[object],
    *,
    source_module: str | None = None,
    fetch_start: str | None = None,
    fetch_end: str | None = None,
) -> None:
    conn.executemany(
        "INSERT OR REPLACE INTO frames "
        "(norad, idx, data, timestamp, transmitter, observation_id, "
        "source_module, fetch_start, fetch_end) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                norad,
                i,
                _frame_data(f),
                getattr(f, "timestamp", None),
                getattr(f, "transmitter", None),
                getattr(f, "observation_id", None),
                source_module,
                fetch_start,
                fetch_end,
            )
            for i, f in enumerate(frames)
        ],
    )
    conn.commit()


def insert_satellite_metadata(conn: sqlite3.Connection, metadata: SatelliteMetadata) -> None:
    values = asdict(metadata)
    conn.execute(
        "INSERT OR REPLACE INTO satellites "
        "(norad, module, fetch_start, fetch_end, source_ksy_url, source_ksy_sha256, "
        "upstream_revision, probe_cap, full_cap, accepted_frame_count, "
        "modal_frame_length, layout_field_count, qualification_reason) "
        "VALUES (:norad, :module, :fetch_start, :fetch_end, :source_ksy_url, "
        ":source_ksy_sha256, :upstream_revision, :probe_cap, :full_cap, "
        ":accepted_frame_count, :modal_frame_length, :layout_field_count, "
        ":qualification_reason)",
        values,
    )
    conn.commit()


def insert_layout(conn: sqlite3.Connection, norad: int, layout: Layout) -> None:
    conn.executemany(
        "INSERT OR REPLACE INTO layouts "
        "(norad, field_idx, start, end, width, signed, is_enum, name) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (norad, i, f.start, f.end, f.width, int(f.signed), int(f.is_enum), f.name)
            for i, f in enumerate(layout)
        ],
    )
    conn.commit()


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))
    conn.commit()


def get_meta(conn: sqlite3.Connection, key: str, default: str | None = None) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return str(row[0]) if row else default


def list_norads(conn: sqlite3.Connection) -> list[int]:
    return [r[0] for r in conn.execute("SELECT DISTINCT norad FROM frames ORDER BY norad")]


def query_frames(conn: sqlite3.Connection, norad: int) -> list[bytes]:
    return [
        bytes(r[0])
        for r in conn.execute("SELECT data FROM frames WHERE norad = ? ORDER BY idx", (norad,))
    ]


def query_frame_records(conn: sqlite3.Connection, norad: int) -> list[FrameRecord]:
    return [
        FrameRecord(
            norad=int(r[0]),
            idx=int(r[1]),
            data=bytes(r[2]),
            timestamp=r[3],
            transmitter=r[4],
            observation_id=r[5],
            source_module=r[6],
            fetch_start=r[7],
            fetch_end=r[8],
        )
        for r in conn.execute(
            "SELECT norad, idx, data, timestamp, transmitter, observation_id, "
            "source_module, fetch_start, fetch_end FROM frames "
            "WHERE norad = ? ORDER BY idx",
            (norad,),
        )
    ]


def query_layout(conn: sqlite3.Connection, norad: int) -> Layout:
    return [
        FieldSpan(
            start=r[0], end=r[1], width=r[2], signed=bool(r[3]), is_enum=bool(r[4]), name=r[5]
        )
        for r in conn.execute(
            "SELECT start, end, width, signed, is_enum, name FROM layouts "
            "WHERE norad = ? ORDER BY field_idx",
            (norad,),
        )
    ]


def corpus_summary(conn: sqlite3.Connection) -> dict[str, Any]:
    meta = {str(k): str(v) for k, v in conn.execute("SELECT key, value FROM meta ORDER BY key")}
    n_frames = int(conn.execute("SELECT COUNT(*) FROM frames").fetchone()[0])
    sats = [
        {
            "norad": int(r[0]),
            "module": r[1],
            "fetch_start": r[2],
            "fetch_end": r[3],
            "upstream_revision": r[4],
            "accepted_frame_count": int(r[5]),
            "modal_frame_length": int(r[6]),
            "layout_field_count": int(r[7]),
        }
        for r in conn.execute(
            "SELECT norad, module, fetch_start, fetch_end, upstream_revision, "
            "accepted_frame_count, modal_frame_length, layout_field_count "
            "FROM satellites ORDER BY norad"
        )
    ]
    return {
        "schema_version": int(meta.get("corpus_schema_version", SCHEMA_VERSION)),
        "n_sats": len(list_norads(conn)),
        "n_frames": n_frames,
        "meta": meta,
        "satellites": sats,
        "frame_columns": sorted(_columns(conn, "frames")),
    }
