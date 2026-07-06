"""The temporary SQLite training corpus (spec §4). Deleted at finalization.

Three tables: `frames` (raw bytes per sat), `layouts` (true field spans per
sat — the labels), `meta` (key/value provenance). Portable, serverless,
git-ignored.
"""
from __future__ import annotations

import sqlite3

from satnogs_decoder.infer.layout import FieldSpan, Layout

_SCHEMA = """
CREATE TABLE IF NOT EXISTS frames (
    norad INTEGER NOT NULL,
    idx   INTEGER NOT NULL,
    data  BLOB    NOT NULL,
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
"""


def open_corpus(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def insert_frames(conn: sqlite3.Connection, norad: int, frames: list[bytes]) -> None:
    conn.executemany(
        "INSERT OR REPLACE INTO frames (norad, idx, data) VALUES (?, ?, ?)",
        [(norad, i, f) for i, f in enumerate(frames)],
    )
    conn.commit()


def insert_layout(conn: sqlite3.Connection, norad: int, layout: Layout) -> None:
    conn.executemany(
        "INSERT OR REPLACE INTO layouts "
        "(norad, field_idx, start, end, width, signed, is_enum, name) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [(norad, i, f.start, f.end, f.width, int(f.signed), int(f.is_enum), f.name)
         for i, f in enumerate(layout)],
    )
    conn.commit()


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))
    conn.commit()


def list_norads(conn: sqlite3.Connection) -> list[int]:
    return [r[0] for r in conn.execute("SELECT DISTINCT norad FROM frames ORDER BY norad")]


def query_frames(conn: sqlite3.Connection, norad: int) -> list[bytes]:
    return [bytes(r[0]) for r in conn.execute(
        "SELECT data FROM frames WHERE norad = ? ORDER BY idx", (norad,))]


def query_layout(conn: sqlite3.Connection, norad: int) -> Layout:
    return [
        FieldSpan(start=r[0], end=r[1], width=r[2], signed=bool(r[3]),
                  is_enum=bool(r[4]), name=r[5])
        for r in conn.execute(
            "SELECT start, end, width, signed, is_enum, name FROM layouts "
            "WHERE norad = ? ORDER BY field_idx", (norad,))
    ]
