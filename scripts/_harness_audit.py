"""Harness robustness audit: stress-test compile_ksy + parse across all canonical
satnogs-decoders .ksy files from the GitLab repo, then run parse against real frames
for the 4 anchor satellites.

Run inside the container:
    docker compose run --rm app python scripts/_harness_audit.py
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import time
import traceback
from collections import defaultdict

import requests

from satnogs_decoder.shared.kaitai import compile_ksy, parse
from satnogs_decoder.shared.satnogs_db import fetch_frames
from satnogs_decoder.shared.reference import decode_reference

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GITLAB_BASE = "https://gitlab.com/api/v4"
GITLAB_RAW = "https://gitlab.com/librespacefoundation/satnogs/satnogs-decoders/-/raw/master/ksy"
PROJECT_ID = "librespacefoundation%2Fsatnogs%2Fsatnogs-decoders"
KSY_LOCAL = pathlib.Path("/tmp/ksy")
KSY_LOCAL.mkdir(parents=True, exist_ok=True)

ANCHORS_PATH = pathlib.Path(__file__).parents[1] / "tests" / "fixtures" / "anchors.json"
REAL_FRAME_PATH = pathlib.Path(__file__).parents[1] / "tests" / "fixtures" / "real_frame_grbalpha.hex"

# Per-anchor, the module name maps to the .ksy filename in the repo
ANCHOR_KSY_MAP = {
    "grbalpha":  "grbalpha.ksy",
    "vzlusat2":  "vzlusat_2.ksy",
    "duchifat3": "duchifat_3.ksy",
    "ledsat":    "ledsat.ksy",
}

# ---------------------------------------------------------------------------
# 1. List all .ksy files via the GitLab API
# ---------------------------------------------------------------------------

def list_ksy_files() -> list[str]:
    """Return list of all .ksy filenames from the satnogs-decoders ksy/ directory."""
    names: list[str] = []
    page = 1
    session = requests.Session()
    session.headers.update({"User-Agent": "satnogs-decoder-audit"})

    while True:
        url = (
            f"{GITLAB_BASE}/projects/{PROJECT_ID}/repository/tree"
            f"?path=ksy&per_page=100&page={page}"
        )
        r = session.get(url, timeout=30)
        r.raise_for_status()
        items = r.json()
        if not items:
            break
        names.extend(item["name"] for item in items if item["name"].endswith(".ksy"))
        page += 1

    print(f"  Found {len(names)} .ksy files in GitLab repo")
    return names


# ---------------------------------------------------------------------------
# 2. Download each .ksy file
# ---------------------------------------------------------------------------

def download_ksy_files(names: list[str]) -> dict[str, str]:
    """Download each .ksy and return {name: content}. Also write to KSY_LOCAL/."""
    session = requests.Session()
    session.headers.update({"User-Agent": "satnogs-decoder-audit"})
    contents: dict[str, str] = {}

    for i, name in enumerate(names, 1):
        local = KSY_LOCAL / name
        if local.exists():
            # Use cached version from a previous run
            contents[name] = local.read_text()
            if i % 20 == 0:
                print(f"  [{i}/{len(names)}] cached {name}")
            continue

        url = f"{GITLAB_RAW}/{name}"
        r = session.get(url, timeout=30)
        if r.status_code != 200:
            print(f"  [{i}/{len(names)}] DOWNLOAD FAILED {name}: HTTP {r.status_code}")
            continue
        local.write_text(r.text)
        contents[name] = r.text

        if i % 20 == 0:
            print(f"  [{i}/{len(names)}] downloaded {name}")
        time.sleep(0.05)  # gentle rate-limit

    print(f"  Downloaded/cached {len(contents)} files")
    return contents


# ---------------------------------------------------------------------------
# 3. Attempt compile_ksy for each file and categorize failures
# ---------------------------------------------------------------------------

_CATEGORIES = {
    "ok": "Compiled OK",
    "unresolved_import": "Unresolved import (missing dependency)",
    "process_module": "Missing process: module",
    "ksc_syntax": "ksc compiler syntax/feature error",
    "ksc_not_found": "ksc not found / not installed",
    "meta_id": "No meta/id in .ksy",
    "other": "Other / unknown",
}

def categorize_error(error: str) -> str:
    low = error.lower()
    if "ksc: not found" in low or "no such file" in low and "ksc" in low:
        return "ksc_not_found"
    if "ksy_text has no meta/id" in low or "no meta/id" in low:
        return "meta_id"
    # ksc stderr patterns
    if "error" in low and ("import" in low or "unknown type" in low or "undefined type" in low):
        return "unresolved_import"
    if "process" in low and ("not found" in low or "no module" in low or "import" in low):
        return "process_module"
    if "ksc failed" in low:
        # Try to determine subcategory from stderr content
        if "process" in low:
            return "process_module"
        if "import" in low or "undefined" in low or "unknown type" in low:
            return "unresolved_import"
        return "ksc_syntax"
    if "modulenotfounderror" in low or "importerror" in low:
        return "process_module"
    return "other"


def compile_all(contents: dict[str, str]) -> dict:
    """Try to compile each .ksy. Return results dict."""
    results: dict[str, dict] = {}
    import_dir = str(KSY_LOCAL)
    names = sorted(contents.keys())

    for i, name in enumerate(names, 1):
        text = contents[name]
        try:
            cls = compile_ksy(text, import_dirs=[import_dir])
            results[name] = {"status": "ok", "cls": cls, "error": None, "category": "ok"}
        except Exception as e:
            err = str(e)
            cat = categorize_error(err)
            results[name] = {"status": "fail", "cls": None, "error": err, "category": cat}

        if i % 20 == 0 or i == len(names):
            n_ok = sum(1 for v in results.values() if v["status"] == "ok")
            print(f"  [{i}/{len(names)}] OK={n_ok} fail={i - n_ok}  (just did: {name})")

    return results


# ---------------------------------------------------------------------------
# 4. Parse real frames for anchor satellites
# ---------------------------------------------------------------------------

def parse_anchors(compile_results: dict) -> list[dict]:
    """For each anchor, fetch frames, parse with compiled class, record coverage."""
    anchors = json.loads(ANCHORS_PATH.read_text())
    anchor_results = []

    for anchor in anchors:
        norad = anchor["norad"]
        name = anchor["name"]
        module = anchor["module"]
        ksy_name = ANCHOR_KSY_MAP.get(module)
        start = anchor["start"]
        end = anchor["end"]

        print(f"\n  Anchor: {name} (NORAD {norad}, module={module})")

        # Get the compiled class (if it compiled)
        cls = None
        compile_ok = False
        if ksy_name and ksy_name in compile_results:
            entry = compile_results[ksy_name]
            if entry["status"] == "ok":
                cls = entry["cls"]
                compile_ok = True
                print(f"    .ksy compiled OK")
            else:
                print(f"    .ksy compile FAILED: {entry['error'][:120]}")
        else:
            print(f"    .ksy={ksy_name} not found in compile_results (keys: {list(compile_results.keys())[:5]})")

        # For grbalpha, use the pre-existing real frame; for others fetch from DB
        frames: list[bytes] = []
        if module == "grbalpha" and REAL_FRAME_PATH.exists():
            hex_str = REAL_FRAME_PATH.read_text().strip()
            frames = [bytes.fromhex(hex_str)]
            print(f"    Using pre-cached real frame ({len(frames[0])} bytes)")
        else:
            print(f"    Fetching frames from SatNOGS DB (window: {start} to {end})...")
            try:
                fetched = fetch_frames(norad, start=start, end=end, limit=30)
                frames = [f.data for f in fetched]
                print(f"    Fetched {len(frames)} frames from DB")
            except Exception as e:
                print(f"    Frame fetch failed: {e}")

        if not frames:
            anchor_results.append({
                "name": name, "norad": norad, "module": module, "ksy": ksy_name,
                "compile_ok": compile_ok,
                "frames_fetched": 0, "parse_ok": 0, "parse_fail": 0,
                "ref_ok": 0, "mean_coverage": None, "error": "no frames"
            })
            continue

        # Parse each frame
        parse_ok = 0
        parse_fail = 0
        coverages: list[float] = []

        for frame_bytes in frames:
            if cls is not None:
                result = parse(cls, frame_bytes)
                if result.ok:
                    parse_ok += 1
                    # coverage = fraction of bytes consumed
                    cov = result.consumed / result.total if result.total > 0 else 0.0
                    coverages.append(cov)
                else:
                    parse_fail += 1
            else:
                parse_fail += 1

        # Reference decode check
        ref_ok = 0
        for frame_bytes in frames:
            ref = decode_reference(module, frame_bytes)
            if ref is not None and len(ref) > 0:
                ref_ok += 1

        mean_cov = (sum(coverages) / len(coverages)) if coverages else None

        print(f"    Parse: {parse_ok}/{len(frames)} OK, ref-decode {ref_ok}/{len(frames)}")
        if mean_cov is not None:
            print(f"    Mean byte coverage: {mean_cov:.1%}")

        anchor_results.append({
            "name": name, "norad": norad, "module": module, "ksy": ksy_name,
            "compile_ok": compile_ok,
            "frames_fetched": len(frames),
            "parse_ok": parse_ok,
            "parse_fail": parse_fail,
            "ref_ok": ref_ok,
            "mean_coverage": mean_cov,
            "error": None,
        })

    return anchor_results


# ---------------------------------------------------------------------------
# 5. Print summary report
# ---------------------------------------------------------------------------

def print_summary(compile_results: dict, anchor_results: list[dict]) -> None:
    total = len(compile_results)
    n_ok = sum(1 for v in compile_results.values() if v["status"] == "ok")
    n_fail = total - n_ok
    pct = 100.0 * n_ok / total if total else 0

    print("\n" + "=" * 70)
    print(f"HARNESS ROBUSTNESS AUDIT — SUMMARY")
    print("=" * 70)
    print(f"Total .ksy files:   {total}")
    print(f"Compiled OK:        {n_ok}  ({pct:.1f}%)")
    print(f"Compile failures:   {n_fail}  ({100-pct:.1f}%)")
    print()

    # Group failures by category
    failures_by_cat: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for name, v in compile_results.items():
        if v["status"] != "ok":
            failures_by_cat[v["category"]].append((name, v["error"] or ""))

    if failures_by_cat:
        print("FAILURE CATEGORIES:")
        for cat, items in sorted(failures_by_cat.items(), key=lambda x: -len(x[1])):
            print(f"\n  [{cat}]  count={len(items)}")
            print(f"  Description: {_CATEGORIES.get(cat, cat)}")
            for fname, err in items[:3]:
                snippet = err.replace("\n", " ").strip()[:200]
                print(f"    - {fname}: {snippet}")

    print("\n" + "-" * 70)
    print("ANCHOR PARSE RESULTS:")
    for ar in anchor_results:
        name = ar["name"]
        frames = ar["frames_fetched"]
        compile_ok = "OK" if ar["compile_ok"] else "FAIL"
        if frames == 0:
            print(f"  {name}: compile={compile_ok} frames=0 (no data)")
        else:
            parse_rate = ar["parse_ok"] / frames if frames else 0
            cov_str = f"{ar['mean_coverage']:.1%}" if ar["mean_coverage"] is not None else "N/A"
            print(f"  {name}: compile={compile_ok} parse={ar['parse_ok']}/{frames} ({parse_rate:.0%})"
                  f"  ref_ok={ar['ref_ok']}/{frames}  mean_coverage={cov_str}")

    print()

    return {
        "total": total,
        "n_ok": n_ok,
        "pct": pct,
        "failures_by_cat": {
            cat: [{"name": name, "error": err[:300]} for name, err in items]
            for cat, items in failures_by_cat.items()
        },
        "anchor_results": anchor_results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n[1] Listing .ksy files from GitLab...")
    names = list_ksy_files()

    print(f"\n[2] Downloading {len(names)} .ksy files to {KSY_LOCAL}...")
    contents = download_ksy_files(names)

    print(f"\n[3] Compiling {len(contents)} .ksy files...")
    compile_results = compile_all(contents)

    print(f"\n[4] Parsing anchor frames...")
    anchor_results = parse_anchors(compile_results)

    # Print and collect summary data
    summary = print_summary(compile_results, anchor_results)

    # Save full results for doc generation
    output_path = pathlib.Path("/tmp/audit_results.json")
    save_data = {
        "total": len(compile_results),
        "n_ok": sum(1 for v in compile_results.values() if v["status"] == "ok"),
        "compile_results": {
            name: {
                "status": v["status"],
                "category": v["category"],
                "error": (v["error"] or "")[:500],
            }
            for name, v in compile_results.items()
        },
        "anchor_results": anchor_results,
    }
    output_path.write_text(json.dumps(save_data, indent=2, default=str))
    print(f"\nFull results saved to {output_path}")


if __name__ == "__main__":
    main()
