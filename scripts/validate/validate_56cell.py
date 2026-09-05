#!/usr/bin/env python3
"""Deterministic structural validation of the authoritative 56-cell records.

Validates STRUCTURE and INTEGRITY only. It makes no claim about native execution
provenance or empirical authenticity; see reports/ARTIFACT_56_CELL_LINEAGE.md.
"""
from __future__ import annotations
import argparse, collections, hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "artifacts/evidence/source_records/per_example_records.jsonl"
EXPECT_SHA = "c63babd59efe28b52797379d19e2dffbc500416fd53fc0d45e0e953ebd3d3cef"
EXPECT = dict(records=13440, cells=56, seeds=5, conditions=2, per_tuple=24, tuples=560)

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def validate(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"BLOCKED: authoritative records not found at {path}")
    digest = sha256(path)
    recs = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    tup = collections.Counter((r["cell_id"], r["seed"], r["condition"]) for r in recs)
    ids = collections.Counter(r["record_id"] for r in recs)
    schemas = {tuple(sorted(r.keys())) for r in recs}
    per_tuple = sorted(set(tup.values()))

    checks = {
        "file_sha256_matches_frozen": digest == EXPECT_SHA,
        "record_count":               len(recs) == EXPECT["records"],
        "unique_cells":               len({r["cell_id"] for r in recs}) == EXPECT["cells"],
        "seed_count":                 len({r["seed"] for r in recs}) == EXPECT["seeds"],
        "condition_count":            len({r["condition"] for r in recs}) == EXPECT["conditions"],
        "records_per_tuple_uniform":  per_tuple == [EXPECT["per_tuple"]],
        "tuple_count":                len(tup) == EXPECT["tuples"],
        "no_duplicate_record_ids":    all(v == 1 for v in ids.values()),
        "single_schema":              len(schemas) == 1,
    }
    report = {
        "path": str(path.relative_to(ROOT)),
        "sha256": digest,
        "observed": {
            "records": len(recs),
            "cells": len({r["cell_id"] for r in recs}),
            "seeds": sorted({r["seed"] for r in recs}),
            "conditions": sorted({r["condition"] for r in recs}),
            "records_per_tuple": per_tuple,
            "tuples": len(tup),
            "duplicate_record_ids": sum(1 for v in ids.values() if v > 1),
            "distinct_schemas": len(schemas),
        },
        "checks": checks,
        "structural_completeness": "PASS" if all(checks.values()) else "FAIL",
        "internal_consistency": "PASS" if all(checks.values()) else "FAIL",
        "native_execution_provenance": "NOT_AVAILABLE",
        "empirical_authenticity": "NOT_ESTABLISHED",
        "origin": "UNKNOWN_WITH_GENERATIVE_INDICATORS",
        "safe_for_empirical_manuscript_claims": "NO",
    }
    return report

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", type=Path, default=DEFAULT)
    ap.add_argument("--output", type=Path, default=ROOT / "reports/validate_56cell_report.json")
    a = ap.parse_args()
    rep = validate(a.records)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    for k, v in rep["checks"].items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"\nSTRUCTURAL_COMPLETENESS = {rep['structural_completeness']}")
    print(f"NATIVE_EXECUTION_PROVENANCE = {rep['native_execution_provenance']}")
    print(f"EMPIRICAL_AUTHENTICITY = {rep['empirical_authenticity']}")
    return 0 if rep["structural_completeness"] == "PASS" else 1

if __name__ == "__main__":
    sys.exit(main())
