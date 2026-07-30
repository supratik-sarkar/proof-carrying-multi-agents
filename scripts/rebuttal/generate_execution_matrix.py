#!/usr/bin/env python3
"""Execution Smoke Test for artifact Python scripts."""

import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REBUTTAL_DIR = REPO_ROOT / "artifacts" / "rebuttal"
VAL_DIR = REBUTTAL_DIR / "validation"
SOURCE_REC = REBUTTAL_DIR / "source_records" / "per_example_records.jsonl"
PYTHON_BIN = sys.executable

VAL_DIR.mkdir(parents=True, exist_ok=True)

print("=================================================================")
print("=== GENERATING ARTIFACT PYTHON EXECUTION SMOKE TEST & REPORT ===")
print("=================================================================\n")

py_scripts = sorted([
    f for f in REBUTTAL_DIR.rglob("*.py")
    if f.is_file() and "tests" not in f.parts and f.name not in ["audit_artifact_python.py", "generate_execution_matrix.py", "run_mutation_tests.py", "execute_full_workflow.py"]
])

execution_records = []
failures_count = 0

for script in py_scripts:
    rel_path = str(script.relative_to(REPO_ROOT))
    content = script.read_text(encoding="utf-8")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_out = Path(tmp_dir) / "output"
        tmp_out.mkdir(parents=True, exist_ok=True)

        # 1. Valid execution smoke test
        cmd_valid = [PYTHON_BIN, str(script)]
        if "--source-records" in content or "source_records" in content:
            cmd_valid.extend(["--source-records", str(SOURCE_REC)])
        if "--output-dir" in content:
            cmd_valid.extend(["--output-dir", str(tmp_out)])
        elif "--output" in content:
            cmd_valid.extend(["--output", str(tmp_out / "result.json")])

        res_v = subprocess.run(cmd_valid, capture_output=True, text=True)
        created_files = [str(f.name) for f in tmp_out.rglob("*") if f.is_file()]

        valid_passed = (res_v.returncode == 0)

        # 2. Invalid execution smoke test (bad CLI input)
        cmd_invalid = [PYTHON_BIN, str(script), "--source-records", "/tmp/nonexistent_file_xyz.jsonl"]
        res_inv = subprocess.run(cmd_invalid, capture_output=True, text=True)

        invalid_passed = (res_inv.returncode != 0)
        obs_inv_code = res_inv.returncode
        obs_inv_err = (res_inv.stderr.strip() or res_inv.stdout.strip()).splitlines()[-1] if (res_inv.stderr or res_inv.stdout) else "Exit code 1"

        overall_smoke_pass = valid_passed and invalid_passed
        if not overall_smoke_pass:
            failures_count += 1

        execution_records.append({
            "script_path": rel_path,
            "valid_cmd_exit_code": res_v.returncode,
            "valid_cmd_passed": valid_passed,
            "files_created_count": len(created_files),
            "files_created": ", ".join(created_files) if created_files else "none",
            "invalid_cmd_exit_code": obs_inv_code,
            "invalid_cmd_passed": invalid_passed,
            "invalid_cmd_observed_error": obs_inv_err[:80],
            "smoke_test_status": "PASS" if overall_smoke_pass else "FAIL"
        })

csv_path = VAL_DIR / "python_execution_matrix.csv"
md_path = VAL_DIR / "python_execution_report.md"

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(execution_records[0].keys()), lineterminator="\n")
    writer.writeheader()
    writer.writerows(execution_records)

md_content = f"""# Artifact Python Execution Smoke Test Report

## Summary: {'PASS' if failures_count == 0 else 'FAIL'} ({failures_count} Failures out of {len(py_scripts)} Scripts Executed)

> [!NOTE]
> This is a CLI smoke test verifying command-line execution and exit code responses on valid and invalid CLI parameters.

| Script Path | Valid Exit Code | Created Files | Invalid Exit Code | Invalid Error Output | Smoke Test Status |
|---|---|---|---|---|---|
""" + "\n".join([f"| `{r['script_path']}` | {r['valid_cmd_exit_code']} | `{r['files_created']}` | {r['invalid_cmd_exit_code']} | `{r['invalid_cmd_observed_error']}` | **{r['smoke_test_status']}** |" for r in execution_records])

md_path.write_text(md_content.strip() + "\n", encoding="utf-8")

print(f"Execution Smoke Test Generated: {len(py_scripts)} scripts verified with valid and invalid commands.")
print(f"Global Execution Failures: {failures_count}")

if failures_count > 0:
    sys.exit(1)
else:
    sys.exit(0)
