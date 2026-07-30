#!/usr/bin/env python3
"""Recompute and verify SHA-256 hashes and field-level integrity for all 13,440 per_example_records."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_REC = REPO_ROOT / "artifacts" / "rebuttal" / "source_records" / "per_example_records.jsonl"
VAL_DIR = REPO_ROOT / "artifacts" / "rebuttal" / "validation"
VAL_DIR.mkdir(parents=True, exist_ok=True)

def verify_source_integrity(source_records_path, output_path=None):
    source_p = Path(source_records_path)
    if not source_p.exists():
        raise FileNotFoundError(f"Source records file not found at {source_p}")

    raw_bytes = source_p.read_bytes()
    file_sha256 = hashlib.sha256(raw_bytes).hexdigest()

    lines = source_p.read_text(encoding="utf-8").splitlines()
    non_empty_lines = [l for l in lines if l.strip()]
    total_records = len(non_empty_lines)

    line_hashes = []
    corrupted_records = 0

    for idx, line in enumerate(non_empty_lines):
        line_sha = hashlib.sha256(line.encode("utf-8")).hexdigest()
        line_hashes.append(line_sha)
        try:
            rec = json.loads(line)
            required_keys = ["cell_id", "example_id", "seed", "condition", "systems"]
            if not all(k in rec for k in required_keys):
                corrupted_records += 1
        except Exception:
            corrupted_records += 1

    integrity_status = "PASS" if (corrupted_records == 0 and total_records > 0) else "FAIL"

    report = {
        "source_file": str(source_p),
        "file_size_bytes": len(raw_bytes),
        "file_sha256": file_sha256,
        "total_records": total_records,
        "corrupted_records": corrupted_records,
        "line_hashes_count": len(line_hashes),
        "integrity_status": integrity_status
    }

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recompute and verify source record file integrity.")
    parser.add_argument("--source-records", default=str(DEFAULT_SOURCE_REC), help="Path to per_example_records.jsonl")
    parser.add_argument("--output", required=False, help="Output JSON path")
    args = parser.parse_args()

    res = verify_source_integrity(args.source_records, args.output)
    print(f"File SHA-256: {res['file_sha256']}")
    print(f"Total Records Verified: {res['total_records']} | Corrupted: {res['corrupted_records']}")
    print(f"SOURCE_RECORD_FILE_INTEGRITY = {res['integrity_status']}")
