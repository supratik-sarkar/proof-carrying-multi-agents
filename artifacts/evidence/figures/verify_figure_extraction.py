#!/usr/bin/env python3
"""Figure acceptance gate: every figure must yield extractable vector text.

Figure 3 of the submission carried the entire R1-R4 result set as four rasters at
~233 ppi with zero extractable words above the caption. This gate makes that class
of defect fail the build: a figure region yielding no extractable text is rejected.

Pure file inspection — no network, no model, no rendering of new content.
Returns 0 when every inspected figure passes, 1 otherwise.
"""
from __future__ import annotations
import re, sys, zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SEARCH_DIRS = [ROOT / "results/figures/manuscript", ROOT / "results/figures/supplementary"]
MIN_WORDS = 3


def extract_words_from_pdf(path: Path) -> int:
    """Count text-showing operators in a PDF without external tooling."""
    raw = path.read_bytes()
    text_ops = 0
    for m in re.finditer(rb"stream\r?\n(.*?)endstream", raw, re.S):
        chunk = m.group(1)
        try:
            chunk = zlib.decompress(chunk)
        except Exception:
            pass
        text_ops += len(re.findall(rb"\((?:[^()\\]|\\.)+\)\s*Tj", chunk))
        text_ops += len(re.findall(rb"\[[^\]]*\]\s*TJ", chunk))
    return text_ops


def inspect(path: Path) -> dict:
    n = extract_words_from_pdf(path)
    return {"figure": f"{path.parent.name}/{path.name}", "text_ops": n,
            "status": "PASS" if n >= MIN_WORDS else "FAIL_NO_VECTOR_TEXT"}


def main(argv=None) -> int:
    results, missing = [], []
    for d in SEARCH_DIRS:
        if not d.exists():
            missing.append(str(d)); continue
        for p in sorted(d.rglob("*.pdf")):
            results.append(inspect(p))
    if missing and not results:
        print(f"[BLOCKED] no figure directory found: {missing}")
        return 1
    bad = [r for r in results if r["status"] != "PASS"]
    for r in results:
        print(f"  [{r['status']}] {r['figure']}  text_ops={r['text_ops']}")
    print(f"\nfigures inspected: {len(results)} | failing: {len(bad)}")
    return 0 if results and not bad else 1


if __name__ == "__main__":
    sys.exit(main())
