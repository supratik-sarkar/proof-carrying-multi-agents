#!/usr/bin/env python3
from pathlib import Path
import json
import sys

root = Path(__file__).resolve().parents[1]
report = json.loads((root / "validation" / "validation_report.json").read_text())
failed = [c for c in report["checks"] if not c["passed"]]

if failed:
    for check in failed:
        print("[FAIL]", check["name"], "-", check["detail"])
    sys.exit(1)

required = [
    "index.html",
    "latex/all_tables.pdf",
    "latex/all_rebuttal_tables.pdf",
    "table_reconciliation/table_reconciliation.tex",
    "sv_decomposition/sv_decomposition.tex",
    "separating_witnesses/separating_witnesses.tex",
    "citation_only/citation_only.tex",
    "injection/injection.tex",
    "shift/shift.tex",
    "audit_sampling/audit_sampling.tex",
    "backend_manifest/backend_manifest_summary.tex",
    "source_records/per_example_records.jsonl",
    "source_records/per_cell_metrics.jsonl",
]
missing = [name for name in required if not (root / name).exists()]
if missing:
    print("[FAIL] Missing files:")
    for name in missing:
        print(name)
    sys.exit(2)

print(f"[PASS] {report['summary']['checks_passed']}/{report['summary']['checks_total']} consistency checks passed.")
print("[PASS] Required artifact files exist.")
