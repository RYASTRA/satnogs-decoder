"""Audit our Kaitai harness against EVERY canonical satnogs-decoders .ksy.

Lists every file in the upstream repo's ksy/ directory, downloads them all into
one dir (so `imports:` resolve), and tries to compile each through our harness
(`compile_ksy`). Prints a JSON summary: how many compile, and a categorized
breakdown of every failure. The installed `satnogs-decoders` package provides the
`process:` routines.

Run in-container:  docker compose run --rm app python scripts/audit_harness.py
"""
from __future__ import annotations
import json
import pathlib
import sys
import tempfile
from collections import Counter

import requests

from satnogs_decoder.shared.kaitai import compile_ksy

TREE = ("https://gitlab.com/api/v4/projects/"
        "librespacefoundation%2Fsatnogs%2Fsatnogs-decoders/repository/tree")
RAW = "https://gitlab.com/librespacefoundation/satnogs/satnogs-decoders/-/raw/master/ksy/"


def list_ksy() -> list[str]:
    names: list[str] = []
    page = 1
    while True:
        r = requests.get(TREE, params={"path": "ksy", "per_page": 100, "page": page}, timeout=60)
        r.raise_for_status()
        items = r.json()
        if not items:
            break
        names += [i["name"] for i in items if i["name"].endswith(".ksy")]
        page += 1
    return sorted(names)


def categorize(err: str) -> str:
    e = err.lower()
    if "unable to find" in e or ("imports" in e and "ksy" in e):
        return "unresolved import"
    if "no module named" in e or "modulenotfound" in e:
        return "missing process/python module"
    if "api_version" in e or "incompatible" in e:
        return "kaitaistruct/ksc version"
    if "ksc failed" in e:
        return "ksc compile error"
    if "no meta/id" in e:
        return "no meta/id"
    return "other"


def main() -> int:
    names = list_ksy()
    tmp = pathlib.Path(tempfile.mkdtemp())
    for n in names:
        try:
            (tmp / n).write_text(requests.get(RAW + n, timeout=60).text)
        except Exception as exc:  # noqa: BLE001
            print(f"fetch fail {n}: {exc}", file=sys.stderr)

    ok: list[str] = []
    failures: list[dict] = []
    cats: Counter[str] = Counter()
    for n in names:
        f = tmp / n
        if not f.exists():
            failures.append({"file": n, "cat": "fetch failed", "error": "download failed"})
            cats["fetch failed"] += 1
            continue
        try:
            compile_ksy(f.read_text(), import_dirs=[str(tmp)])
            ok.append(n)
        except Exception as exc:  # noqa: BLE001
            msg = (str(exc).strip().splitlines() or [type(exc).__name__])[-1][:240]
            cat = categorize(str(exc))
            failures.append({"file": n, "cat": cat, "error": msg})
            cats[cat] += 1

    summary = {
        "total": len(names),
        "compiled_ok": len(ok),
        "failed": len(failures),
        "pct_ok": round(100 * len(ok) / max(len(names), 1), 1),
        "failure_categories": dict(cats),
        "failures": failures,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
