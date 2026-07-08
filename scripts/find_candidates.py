# scripts/find_candidates.py — THROWAWAY. Run as a MODULE (imports scripts.audit_harness):
# docker compose run --rm app python -m scripts.find_candidates > tests/infer/corpus_sats.json
from __future__ import annotations
import json, re, sys
import requests
from scripts.audit_harness import list_ksy, RAW
from satnogs_decoder.infer.structure import has_fixed_layout_case
from satnogs_decoder.infer.qualify import activity_window

SATS_API = "https://db.satnogs.org/api/satellites/?format=json"
NOW = "2026-07-07"


def _norm(s):
    return "".join(c for c in (s or "").lower() if c.isalnum())


def sat_index():
    recs = {}
    for s in requests.get(SATS_API, timeout=90).json():
        nid = s.get("norad_cat_id")
        if not nid:
            continue
        keys = [_norm(s.get("name"))] + [_norm(x) for x in re.split(r"[\n,;]+", s.get("names") or "") if x.strip()]
        for k in keys:
            if k:
                recs.setdefault(k, s)
    return recs


def main() -> int:
    names = list_ksy()
    idx = sat_index()
    out, n_flat, n_nomap = [], 0, 0
    for fn in names:
        module = fn[:-4]
        try:
            text = requests.get(RAW + fn, timeout=60).text
        except Exception:
            continue
        if not has_fixed_layout_case(text):
            continue
        n_flat += 1
        s = idx.get(_norm(module))
        if not s:
            n_nomap += 1
            continue
        start, end = activity_window(
            (s.get("launched") or "")[:10] or None,
            s.get("decayed"), s.get("status") or "", now=NOW)
        out.append({"norad": int(s["norad_cat_id"]), "name": s.get("name") or module,
                    "module": module, "start": start, "end": end})
    print(json.dumps(out, indent=2))
    print(f"candidates={len(out)} fixed_layout_case={n_flat} no_norad={n_nomap} of {len(names)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
