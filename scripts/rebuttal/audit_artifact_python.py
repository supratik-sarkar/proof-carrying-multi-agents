#!/usr/bin/env python3
"""AST-based Meta-Validator for Python scripts under artifacts/rebuttal and scripts/rebuttal."""

import ast
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REBUTTAL_DIR = REPO_ROOT / "artifacts" / "rebuttal"
SCRIPTS_DIR = REPO_ROOT / "scripts" / "rebuttal"
VAL_DIR = REBUTTAL_DIR / "validation"

VAL_DIR.mkdir(parents=True, exist_ok=True)

print("=================================================================")
print("=== AST-BASED META-VALIDATOR FOR ALL REBUTTAL PYTHON SCRIPTS ===")
print("=================================================================\n")

py_files = sorted([f for f in (list(REBUTTAL_DIR.rglob("*.py")) + list(SCRIPTS_DIR.rglob("*.py"))) if f.is_file()])

audit_results = []
failures_count = 0

tautology_pattern = "assertTrue" + "(True)"
tautology_pattern_2 = "assertEqual" + "(True, True)"
abs_path_pattern = "/" + "Users/"

for p in py_files:
    try:
        rel_path = str(p.relative_to(REPO_ROOT))
    except ValueError:
        rel_path = p.name
        
    content = p.read_text(encoding="utf-8", errors="ignore")
    
    classification = "REAL_IMPLEMENTATION"
    if "tests/" in rel_path or "test_" in p.name:
        classification = "REAL_TEST"
    elif "reproduce_all.py" in rel_path:
        classification = "REAL_REPRODUCE_ALL"
        
    is_print_only = False
    is_tautological_test = False
    has_absolute_paths = False
    
    # Requirement 8: Do not treat Path(...) alone as evidence of reading data
    reads_files = ("read_text" in content or "read_bytes" in content or "open(" in content or "json.loads(" in content)
    
    try:
        tree = ast.parse(content)
        
        non_print_nodes = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.With, ast.For, ast.While, ast.Import, ast.ImportFrom))]
        if not non_print_nodes and "print(" in content:
            is_print_only = True
            classification = "PLACEHOLDER"
            
        if p.name != "audit_artifact_python.py" and (tautology_pattern in content or tautology_pattern_2 in content):
            is_tautological_test = True
            classification = "TAUTOLOGICAL_TEST"
            
    except SyntaxError as e:
        # Requirement 8: Make AST parse failures FATAL
        classification = "AST_PARSE_FAILURE"
        
    if p.name != "audit_artifact_python.py" and abs_path_pattern in content:
        has_absolute_paths = True
        classification = "CONTAINS_ABSOLUTE_PATHS"
        
    if "reproduce_all.py" in rel_path and not ("run_" in content or "compute_" in content or "evaluate_" in content or "verify_" in content or "apply_" in content or "reconcile_" in content):
        classification = "DUMMY_REPRODUCE_ALL"
        
    status = "PASS"
    failure_reason = ""
    
    if classification in ["PLACEHOLDER", "TAUTOLOGICAL_TEST", "ALWAYS_PASS_VALIDATOR", "CONTAINS_ABSOLUTE_PATHS", "DUMMY_REPRODUCE_ALL", "AST_PARSE_FAILURE"]:
        status = "FAIL"
        failure_reason = f"Flagged as {classification}"
        failures_count += 1
        
    audit_results.append({
        "path": rel_path,
        "classification": classification,
        "reads_files": reads_files,
        "has_absolute_paths": has_absolute_paths,
        "status": status,
        "failure_reason": failure_reason
    })

csv_path = VAL_DIR / "python_file_audit.csv"
json_path = VAL_DIR / "python_file_audit.json"
md_path = VAL_DIR / "python_file_audit.md"

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(audit_results[0].keys()))
    writer.writeheader()
    writer.writerows(audit_results)

json_path.write_text(json.dumps({
    "total_py_files": len(py_files),
    "failures_count": failures_count,
    "audit_status": "PASS" if failures_count == 0 else "FAIL",
    "files": audit_results
}, indent=2) + "\n", encoding="utf-8")

md_content = f"""# Python File Meta-Audit Report — Submission 9327

## Overall Meta-Audit Status: {'PASS' if failures_count == 0 else 'FAIL'}

* **Total Python Files Analyzed:** {len(py_files)}
* **Failed / Placeholder Files:** {failures_count}
* **Valid Functional Files:** {len(py_files) - failures_count}

| Relative Path | Classification | Reads Files | Absolute Paths | Status | Failure Reason |
|---|---|---|---|---|---|
""" + "\n".join([f"| `{r['path']}` | {r['classification']} | {r['reads_files']} | {r['has_absolute_paths']} | **{r['status']}** | {r['failure_reason']} |" for r in audit_results])

md_path.write_text(md_content.strip() + "\n", encoding="utf-8")

print(f"Meta-Audit Complete: Total Files = {len(py_files)} | Failures = {failures_count}")
print(f"Audit Status: [{'PASS' if failures_count == 0 else 'FAIL'}]")

if failures_count > 0:
    sys.exit(1)
else:
    sys.exit(0)
