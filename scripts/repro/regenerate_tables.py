#!/usr/bin/env python3
"""Deterministically regenerate tables from the authoritative 56-cell records."""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from canonical_metrics import (load, by_cell, cell_row, SYSTEMS, ABLATIONS,
                               ablation_harm, PROVENANCE_CLASS)

ROOT = Path(__file__).resolve().parents[2]
RECS = ROOT / "artifacts/evidence/source_records/per_example_records.jsonl"
OUT  = ROOT / "results/tables/generated"

HEADLINE = ["phi-3.5-mini__FEVER","qwen2.5-7B__HotpotQA","Llama-3.1-8B__PubMedQA",
            "Gemma-2-9b-it__TAT-QA","Llama-3.3-70B__ToolBench","deepseek-v3__WebLINX"]

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", type=Path, default=RECS)
    ap.add_argument("--outdir",  type=Path, default=OUT)
    a = ap.parse_args()
    if not a.records.exists():
        raise SystemExit(f"BLOCKED: records absent at {a.records}")
    recs = load(a.records)
    a.outdir.mkdir(parents=True, exist_ok=True)
    groups = by_cell(recs)
    written = []

    # all 56 cells, clean + adversarial
    for cond in ("clean", "adversarial"):
        rows = []
        for cid in sorted(groups):
            r = cell_row(groups[cid], cond); r["cell_id"] = cid
            rows.append(r)
        p = a.outdir / f"cells_56_{cond}.csv"
        cols = ["cell_id"] + [c for c in rows[0] if c != "cell_id"]
        with p.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
        written.append(p)

    # six headline cells
    rows = []
    for cid in HEADLINE:
        if cid not in groups:      # fail closed, never substitute
            rows.append({"cell_id": cid, "status": "BLOCKED_MISSING_CELL"}); continue
        r = cell_row(groups[cid], "clean"); r["cell_id"] = cid; rows.append(r)
    p = a.outdir / "headline_six.csv"
    cols = ["cell_id"] + sorted({k for r in rows for k in r if k != "cell_id"})
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    written.append(p)

    # ablations
    rows = [{"ablation": ab, "harm_clean": ablation_harm([r for r in recs if r["condition"]=="clean"], ab),
             "harm_adversarial": ablation_harm([r for r in recs if r["condition"]=="adversarial"], ab),
             "provenance_class": PROVENANCE_CLASS} for ab in ABLATIONS]
    p = a.outdir / "ablations.csv"
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    written.append(p)

    manifest = {"provenance_class": PROVENANCE_CLASS,
                "safe_for_empirical_manuscript_use": False,
                "source_records": str(a.records.relative_to(ROOT)),
                "tables": [str(p.relative_to(ROOT)) for p in written]}
    (a.outdir / "TABLE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for p in written: print(f"  wrote {p.relative_to(ROOT)}")
    print(f"\nprovenance_class = {PROVENANCE_CLASS}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
