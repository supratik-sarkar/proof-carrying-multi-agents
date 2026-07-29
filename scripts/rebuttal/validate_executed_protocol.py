#!/usr/bin/env python3
"""Executed Protocol Validator (Validates against executed seeds {0, 1, 2} and N=240 records per cell)."""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_REC = REPO_ROOT / "artifacts" / "rebuttal" / "source_records"

per_cell_file = SRC_REC / "per_cell_metrics.jsonl"
per_ex_file = SRC_REC / "per_example_records.jsonl"

per_cell_rows = [json.loads(l) for l in per_cell_file.read_text().splitlines() if l.strip()]
per_ex_rows = [json.loads(l) for l in per_ex_file.read_text().splitlines() if l.strip()]

executed_seeds_expected = {0, 1, 2}
cell_ex_map = {}
for r in per_ex_rows:
    cid = r["cell_id"]
    cell_ex_map.setdefault(cid, []).append(r)

errors = []
for cid, ex_list in cell_ex_map.items():
    if len(ex_list) != 240:
        errors.append(f"Cell {cid} has {len(ex_list)} records, expected 240")
    seeds_found = set(r.get("seed", 0) for r in ex_list)
    if seeds_found != executed_seeds_expected:
        errors.append(f"Cell {cid} has seeds {seeds_found}, expected {executed_seeds_expected}")

print("--- VALIDATING EXECUTED PROTOCOL ---")
print(f"Executed Protocol Cell Count: {len(cell_ex_map)} / 56")
print(f"Executed Protocol Total Records: {len(per_ex_rows)} (56 * 240 = 13,440)")

if not errors:
    print("Result: PASS")
    print("Status: EXECUTED_PROTOCOL_DOCUMENTED = true")
    sys.exit(0)
else:
    print(f"Result: FAIL ({len(errors)} errors found)")
    for err in errors[:5]:
        print("  -", err)
    sys.exit(1)
