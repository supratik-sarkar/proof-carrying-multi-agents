#!/usr/bin/env python3
"""Non-circular clean-room reproduction validator using static frozen manifest."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REBUTTAL_DIR = REPO_ROOT / "artifacts" / "rebuttal"
DEFAULT_SOURCE_REC = REBUTTAL_DIR / "source_records" / "per_example_records.jsonl"
FROZEN_MANIFEST_PATH = REBUTTAL_DIR / "config" / "clean_room_expected_outputs.json"
VAL_DIR = REBUTTAL_DIR / "validation"
PYTHON_BIN = sys.executable

VAL_DIR.mkdir(parents=True, exist_ok=True)

SUBDIRS = [
    "table_reconciliation", "sv_decomposition", "separating_witnesses",
    "citation_only", "injection", "shift", "audit_sampling", "backend_manifest"
]

def run_clean_room_validation(source_records_path=None, frozen_manifest_path=None, output_path=None):
    source_p = Path(source_records_path) if source_records_path else DEFAULT_SOURCE_REC
    manifest_p = Path(frozen_manifest_path) if frozen_manifest_path else FROZEN_MANIFEST_PATH

    if not source_p.exists():
        raise FileNotFoundError(f"Source records file not found at {source_p}")
    if not manifest_p.exists():
        raise FileNotFoundError(f"Frozen manifest file not found at {manifest_p}")

    manifest_bytes_before = manifest_p.read_bytes()
    frozen_manifest = json.loads(manifest_bytes_before.decode("utf-8"))
    expected_hashes = frozen_manifest["expected_deterministic_outputs"]

    results = {}
    total_files_checked = 0
    mismatches = []
    missing_files = []
    extra_files = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_root = Path(tmp_dir)

        # Execute reproduce_all.py for each subdirectory into tmp_root
        for sub in SUBDIRS:
            sub_script = REBUTTAL_DIR / sub / "scripts" / "reproduce_all.py"
            sub_tmp_out = tmp_root / sub
            sub_tmp_out.mkdir(parents=True, exist_ok=True)

            cmd = [
                PYTHON_BIN, str(sub_script),
                "--source-records", str(source_p),
                "--output-dir", str(sub_tmp_out)
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                raise RuntimeError(f"reproduce_all.py failed for {sub} with exit code {res.returncode}: {res.stderr}")

        # Recursively enumerate all generated files under tmp_root
        generated_files = {}
        for f in sorted(tmp_root.rglob("*")):
            if f.is_file() and f.name not in ["reproduction_manifest.json"]:
                rel_clean = str(f.relative_to(tmp_root))
                generated_files[rel_clean] = hashlib.sha256(f.read_bytes()).hexdigest()

        # Compare against expected_hashes and committed canonical files
        for rel_p, exp_hash in expected_hashes.items():
            total_files_checked += 1
            canonical_file = REBUTTAL_DIR / rel_p
            if not canonical_file.exists():
                missing_files.append(f"Canonical file missing: {rel_p}")
                continue
            canonical_hash = hashlib.sha256(canonical_file.read_bytes()).hexdigest()
            if canonical_hash != exp_hash:
                mismatches.append(f"CLEAN_ROOM_HASH_MISMATCH: Canonical file {rel_p} hash {canonical_hash} != expected {exp_hash}")
                continue

            if rel_p not in generated_files:
                missing_files.append(f"Regenerated output missing: {rel_p}")
                continue

            gen_hash = generated_files[rel_p]
            if gen_hash != exp_hash or gen_hash != canonical_hash:
                mismatches.append(f"CLEAN_ROOM_HASH_MISMATCH: Regenerated {rel_p} hash {gen_hash} != expected {exp_hash}")

        for gen_rel in generated_files.keys():
            if gen_rel not in expected_hashes:
                extra_files.append(f"Unexpected extra file generated: {gen_rel}")

    manifest_bytes_after = manifest_p.read_bytes()
    if manifest_bytes_before != manifest_bytes_after:
        raise RuntimeError("CLEAN_ROOM_HASH_MISMATCH: Frozen clean-room manifest was modified during execution!")

    clean_room_status = "PASS" if (len(mismatches) == 0 and len(missing_files) == 0 and len(extra_files) == 0) else "FAIL"

    try:
        manifest_rel_str = str(manifest_p.relative_to(REPO_ROOT))
    except ValueError:
        manifest_rel_str = str(manifest_p)

    report = {
        "frozen_manifest": manifest_rel_str,
        "total_files_expected": len(expected_hashes),
        "total_files_checked": total_files_checked,
        "mismatches_count": len(mismatches),
        "missing_files_count": len(missing_files),
        "extra_files_count": len(extra_files),
        "mismatches": mismatches,
        "missing_files": missing_files,
        "extra_files": extra_files,
        "status": clean_room_status
    }

    out_p = Path(output_path) if output_path else (VAL_DIR / "clean_room_reproduction.json")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if clean_room_status != "PASS":
        raise ValueError(f"CLEAN_ROOM_HASH_MISMATCH: Clean-room validation failed with status {clean_room_status}")

    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run non-circular clean-room reproduction validator.")
    parser.add_argument("--source-records", default=str(DEFAULT_SOURCE_REC), help="Path to per_example_records.jsonl")
    parser.add_argument("--manifest", default=str(FROZEN_MANIFEST_PATH), help="Path to frozen manifest JSON")
    parser.add_argument("--output", required=False, help="Output JSON path")
    args = parser.parse_args()

    res = run_clean_room_validation(args.source_records, args.manifest, args.output)
    print(f"Clean-Room Verification: Total Expected Files: {res['total_files_expected']} | Mismatches: {res['mismatches_count']}")
    print(f"CLEAN_ROOM_STATUS = {res['status']}")
