#!/usr/bin/env python3
"""Recompute and verify SHA-256 hashes and field-level integrity against frozen manifest."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REBUTTAL_DIR = REPO_ROOT / "artifacts" / "rebuttal"
DEFAULT_SOURCE_REC = REBUTTAL_DIR / "source_records" / "per_example_records.jsonl"
FROZEN_MANIFEST_PATH = REBUTTAL_DIR / "config" / "source_record_integrity_manifest.json"
VAL_DIR = REBUTTAL_DIR / "validation"
VAL_DIR.mkdir(parents=True, exist_ok=True)

def verify_source_integrity(source_records_path=None, frozen_manifest_path=None, output_path=None):
    source_p = Path(source_records_path) if source_records_path else DEFAULT_SOURCE_REC
    manifest_p = Path(frozen_manifest_path) if frozen_manifest_path else FROZEN_MANIFEST_PATH

    if not source_p.exists():
        raise FileNotFoundError(f"Source records file not found at {source_p}")
    if not manifest_p.exists():
        raise FileNotFoundError(f"Frozen manifest file not found at {manifest_p}")

    frozen_manifest = json.loads(manifest_p.read_text(encoding="utf-8"))

    raw_bytes = source_p.read_bytes()
    file_sha256 = hashlib.sha256(raw_bytes).hexdigest()

    lines = source_p.read_text(encoding="utf-8").splitlines()
    non_empty_lines = [l for l in lines if l.strip()]
    total_records = len(non_empty_lines)

    line_hashes = []
    seen_composite_keys = set()
    corrupted_records = 0
    duplicate_composite_keys = 0

    for idx, line in enumerate(non_empty_lines):
        line_sha = hashlib.sha256(line.encode("utf-8")).hexdigest()
        line_hashes.append(line_sha)
        try:
            rec = json.loads(line)
            required_keys = ["cell_id", "example_id", "seed", "condition", "systems"]
            if not all(k in rec for k in required_keys):
                corrupted_records += 1
            key_tuple = (rec.get("cell_id"), rec.get("example_id"), rec.get("seed"), rec.get("condition"))
            if key_tuple in seen_composite_keys:
                duplicate_composite_keys += 1
            else:
                seen_composite_keys.add(key_tuple)
        except Exception:
            corrupted_records += 1

    sha_matches = (file_sha256 == frozen_manifest["expected_file_sha256"])
    count_matches = (total_records == frozen_manifest["expected_record_count"])
    line_hashes_match = (line_hashes == frozen_manifest["expected_ordered_line_hashes"])
    no_duplicates = (duplicate_composite_keys == 0)

    integrity_status = "PASS" if (sha_matches and count_matches and line_hashes_match and corrupted_records == 0 and no_duplicates) else "FAIL"

    report = {
        "source_file": str(source_p.relative_to(REPO_ROOT)),
        "frozen_manifest": str(manifest_p.relative_to(REPO_ROOT)),
        "file_size_bytes": len(raw_bytes),
        "file_sha256": file_sha256,
        "expected_file_sha256": frozen_manifest["expected_file_sha256"],
        "sha256_match": sha_matches,
        "total_records": total_records,
        "expected_record_count": frozen_manifest["expected_record_count"],
        "count_match": count_matches,
        "line_hashes_match": line_hashes_match,
        "unique_identity_keys_count": len(seen_composite_keys),
        "duplicate_identity_keys": duplicate_composite_keys,
        "corrupted_records": corrupted_records,
        "integrity_status": integrity_status
    }

    out_p = Path(output_path) if output_path else (VAL_DIR / "source_record_integrity_report.json")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if integrity_status != "PASS":
        raise ValueError(f"SOURCE_RECORD_FILE_INTEGRITY_MISMATCH: Validation failed with status {integrity_status}")

    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify source record file integrity against frozen manifest.")
    parser.add_argument("--source-records", default=str(DEFAULT_SOURCE_REC), help="Path to per_example_records.jsonl")
    parser.add_argument("--manifest", default=str(FROZEN_MANIFEST_PATH), help="Path to frozen manifest JSON")
    parser.add_argument("--output", required=False, help="Output JSON path")
    args = parser.parse_args()

    res = verify_source_integrity(args.source_records, args.manifest, args.output)
    print(f"File SHA-256: {res['file_sha256']} | SHA Match: {res['sha256_match']}")
    print(f"Total Records: {res['total_records']} | Unique Composite Keys: {res['unique_identity_keys_count']}")
    print(f"SOURCE_RECORD_FILE_INTEGRITY = {res['integrity_status']}")
