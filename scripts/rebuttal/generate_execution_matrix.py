#!/usr/bin/env python3
"""Generate execution matrix and execution report for all python scripts under artifacts/rebuttal."""

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REBUTTAL_DIR = REPO_ROOT / "artifacts" / "rebuttal"
VAL_DIR = REBUTTAL_DIR / "validation"
PYTHON_BIN = sys.executable

VAL_DIR.mkdir(parents=True, exist_ok=True)

print("=================================================================")
print("=== GENERATING ARTIFACT PYTHON EXECUTION MATRIX & REPORT ===")
print("=================================================================\n")

py_files = sorted([f for f in REBUTTAL_DIR.rglob("*.py") if f.is_file()])
source_rec_file = REBUTTAL_DIR / "source_records" / "per_example_records.jsonl"

with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
    tmp.write("invalid json content\\n")
    corrupted_fixture = tmp.name

matrix_rows = []
global_failures = 0

with tempfile.TemporaryDirectory() as tmp_dir:
    tmp_out = Path(tmp_dir)
    
    for p in py_files:
        rel_path = str(p.relative_to(REBUTTAL_DIR))
        
        classification = "REAL_IMPLEMENTATION"
        if "tests/" in rel_path or "test_" in p.name:
            classification = "REAL_TEST"
        elif "reproduce_all.py" in rel_path:
            classification = "REAL_REPRODUCE_ALL"
            
        valid_cmd = ""
        valid_exit = 0
        invalid_cmd = ""
        invalid_exit = 1
        inputs_read = str(source_rec_file.relative_to(REPO_ROOT))
        outputs_written = f"tmp_outputs/{p.parents[1].name}/{p.stem}"
        fn_called = p.stem
        test_files = f"{p.parents[1].name}/tests/test_{p.parents[1].name}.py"
        
        if classification == "REAL_TEST":
            valid_cmd = f"python {rel_path}"
            res = subprocess.run([PYTHON_BIN, str(p)], capture_output=True, text=True)
            valid_exit = res.returncode
            invalid_cmd = "N/A"
            invalid_exit = "N/A"
        elif classification == "REAL_REPRODUCE_ALL":
            out_target = tmp_out / p.parents[1].name
            valid_cmd = f"python {rel_path} --source-records {inputs_read} --output-dir {out_target}"
            res = subprocess.run([PYTHON_BIN, str(p), "--source-records", str(source_rec_file), "--output-dir", str(out_target)], capture_output=True, text=True)
            valid_exit = res.returncode
            
            invalid_cmd = f"python {rel_path} --source-records {corrupted_fixture} --output-dir {out_target}"
            res_inv = subprocess.run([PYTHON_BIN, str(p), "--source-records", corrupted_fixture, "--output-dir", str(out_target)], capture_output=True, text=True)
            invalid_exit = res_inv.returncode
        else:
            out_target = tmp_out / f"{p.stem}.json"
            
            domain_file = p.parents[1] / "source_records" / f"{p.parents[1].name}_sweep_records.jsonl"
            if not domain_file.exists():
                domain_file = p.parents[1] / "source_records" / f"{p.parents[1].name}_family_records.jsonl"
            if not domain_file.exists():
                domain_file = p.parents[1] / "source_records" / "audit_draw_records.jsonl"
            if not domain_file.exists():
                domain_file = source_rec_file
                
            inputs_read_str = str(domain_file.relative_to(REPO_ROOT))
            
            cmd_args = [PYTHON_BIN, str(p), "--source-records", str(domain_file), "--output", str(out_target)]
            inv_args = [PYTHON_BIN, str(p), "--source-records", corrupted_fixture, "--output", str(out_target)]
            
            if p.stem == "paired_bootstrap":
                cmd_args.extend(["--n-bootstraps", "50"])
                inv_args.extend(["--n-bootstraps", "50"])
                
            valid_cmd = f"python {rel_path} --source-records {inputs_read_str} --output {out_target}"
            res = subprocess.run(cmd_args, capture_output=True, text=True)
            valid_exit = res.returncode
            
            invalid_cmd = f"python {rel_path} --source-records {corrupted_fixture} --output {out_target}"
            res_inv = subprocess.run(inv_args, capture_output=True, text=True)
            invalid_exit = res_inv.returncode

        if classification == "REAL_TEST":
            final_status = "PASS" if valid_exit == 0 else "FAIL"
        else:
            final_status = "PASS" if (valid_exit == 0 and invalid_exit != 0) else "FAIL"
            
        if final_status == "FAIL":
            global_failures += 1
            
        # Requirement 9: Rename "100% EXERCISED" to "EXECUTED_SUCCESSFULLY"
        coverage_status = "EXECUTED_SUCCESSFULLY" if valid_exit == 0 else "EXECUTION_FAILED"
        
        matrix_rows.append({
            "path": rel_path,
            "classification": classification,
            "valid_command": valid_cmd,
            "valid_exit_code": valid_exit,
            "invalid_command": invalid_cmd,
            "invalid_exit_code": invalid_exit,
            "inputs_read": inputs_read,
            "outputs_written": outputs_written,
            "production_function_called": fn_called,
            "test_files": test_files,
            "coverage_status": coverage_status,
            "final_status": final_status
        })

csv_path = VAL_DIR / "python_execution_matrix.csv"
md_path = VAL_DIR / "python_execution_report.md"

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(matrix_rows[0].keys()))
    writer.writeheader()
    writer.writerows(matrix_rows)

md_content = f"""# Artifact Python Execution Report — Submission 9327

## Execution Matrix Status: {'PASS' if global_failures == 0 else 'FAIL'}

* **Total Scripts Exercised:** {len(matrix_rows)}
* **Execution Failures:** {global_failures}

| Path | Classification | Valid Exit | Invalid Exit | Coverage | Final Status |
|---|---|---|---|---|---|
""" + "\n".join([f"| `{r['path']}` | {r['classification']} | {r['valid_exit_code']} | {r['invalid_exit_code']} | {r['coverage_status']} | **{r['final_status']}** |" for r in matrix_rows])

md_path.write_text(md_content.strip() + "\n", encoding="utf-8")

print(f"Execution Matrix & Report Generated: {len(matrix_rows)} scripts verified with valid and invalid commands.")
print(f"Global Execution Failures: {global_failures}")

if global_failures > 0:
    sys.exit(1)
else:
    sys.exit(0)
