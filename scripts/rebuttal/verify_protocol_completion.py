#!/usr/bin/env python3
"""Verify protocol completion against frozen expectation manifest."""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REBUTTAL_DIR = REPO_ROOT / "artifacts" / "rebuttal"
DEFAULT_SOURCE_REC = REBUTTAL_DIR / "source_records" / "per_example_records.jsonl"
FROZEN_EXPECTATION_PATH = REBUTTAL_DIR / "config" / "frozen_protocol_expectation.json"
VAL_DIR = REBUTTAL_DIR / "validation"
VAL_DIR.mkdir(parents=True, exist_ok=True)

def verify_protocol_completion(source_records_path=None, frozen_expectation_path=None, output_path=None):
    source_p = Path(source_records_path) if source_records_path else DEFAULT_SOURCE_REC
    expectation_p = Path(frozen_expectation_path) if frozen_expectation_path else FROZEN_EXPECTATION_PATH

    if not source_p.exists():
        raise FileNotFoundError(f"Source records file not found at {source_p}")
    if not expectation_p.exists():
        raise FileNotFoundError(f"Frozen expectation file not found at {expectation_p}")

    frozen_exp = json.loads(expectation_p.read_text(encoding="utf-8"))

    raw_lines = source_p.read_text(encoding="utf-8").splitlines()
    records = [json.loads(l) for l in raw_lines if l.strip()]

    observed_groups = {}
    seen_examples = set()
    duplicate_examples = 0

    for r in records:
        key = (r["cell_id"], r["seed"], r["condition"])
        observed_groups.setdefault(key, []).append(r)
        ex_tuple = (r["cell_id"], r["seed"], r["condition"], r["example_id"])
        if ex_tuple in seen_examples:
            duplicate_examples += 1
        else:
            seen_examples.add(ex_tuple)

    expected_tuples = frozen_exp["tuples"]
    total_expected_tuples = len(expected_tuples)
    matching_tuples = 0
    missing_tuples = 0
    mismatched_counts = 0

    for exp_t in expected_tuples:
        key = (exp_t["cell_id"], exp_t["seed"], exp_t["condition"])
        if key not in observed_groups:
            missing_tuples += 1
        else:
            obs_count = len(observed_groups[key])
            if obs_count == exp_t["expected_records"]:
                matching_tuples += 1
            else:
                mismatched_counts += 1

    extra_tuples = len(set(observed_groups.keys()) - set((t["cell_id"], t["seed"], t["condition"]) for t in expected_tuples))

    if missing_tuples > 0 or mismatched_counts > 0 or extra_tuples > 0 or duplicate_examples > 0:
        status = "FAIL"
        err_msg = f"PROTOCOL_EXPECTATION_COUNT_MISMATCH: {missing_tuples} missing tuples, {mismatched_counts} count mismatches, {extra_tuples} extra tuples, {duplicate_examples} duplicate examples."
    else:
        status = "PASS"

    try:
        source_rel_str = str(source_p.relative_to(REPO_ROOT))
    except ValueError:
        source_rel_str = str(source_p)

    try:
        exp_rel_str = str(expectation_p.relative_to(REPO_ROOT))
    except ValueError:
        exp_rel_str = str(expectation_p)

    report = {
        "source_file": source_rel_str,
        "frozen_expectation": exp_rel_str,
        "total_expected_tuples": total_expected_tuples,
        "observed_tuples_count": len(observed_groups),
        "matching_tuples": matching_tuples,
        "missing_tuples": missing_tuples,
        "extra_tuples": extra_tuples,
        "mismatched_counts": mismatched_counts,
        "duplicate_examples": duplicate_examples,
        "status": status
    }

    out_p = Path(output_path) if output_path else (VAL_DIR / "protocol_completion_report.json")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if status != "PASS":
        raise ValueError(f"PROTOCOL_EXPECTATION_COUNT_MISMATCH: Validation failed with status {status}")

    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify protocol completion against frozen expectation manifest.")
    parser.add_argument("--source-records", default=str(DEFAULT_SOURCE_REC), help="Path to per_example_records.jsonl")
    parser.add_argument("--expectation", default=str(FROZEN_EXPECTATION_PATH), help="Path to frozen expectation JSON")
    parser.add_argument("--output", required=False, help="Output JSON path")
    args = parser.parse_args()

    res = verify_protocol_completion(args.source_records, args.expectation, args.output)
    print(f"Total Expected Tuples: {res['total_expected_tuples']} | Matching: {res['matching_tuples']}")
    print(f"PROTOCOL_COMPLETION_STATUS = {res['status']}")
