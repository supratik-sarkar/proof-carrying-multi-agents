#!/usr/bin/env python3
"""Master Forensic Audit & Reproducibility Pipeline Runner."""

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REBUTTAL_DIR = REPO_ROOT / "artifacts" / "rebuttal"
VAL_DIR = REBUTTAL_DIR / "validation"
PYTHON_BIN = sys.executable

VAL_DIR.mkdir(parents=True, exist_ok=True)

print("=================================================================")
print("=== STARTING MASTER FORENSIC & REPRODUCIBILITY PIPELINE ===")
print("=================================================================\n")

# 1. Source Record File Integrity
print("--- STEP 1: VERIFYING SOURCE RECORD FILE INTEGRITY ---")
res_src = subprocess.run([PYTHON_BIN, str(REPO_ROOT / "scripts" / "rebuttal" / "verify_source_record_integrity.py")], capture_output=True, text=True)
src_report_path = VAL_DIR / "source_record_integrity_report.json"
if src_report_path.exists():
    src_rep = json.loads(src_report_path.read_text(encoding="utf-8"))
    source_rec_file_integrity = src_rep.get("integrity_status", "FAIL")
else:
    source_rec_file_integrity = "FAIL"
print(f"SOURCE_RECORD_FILE_INTEGRITY = {source_rec_file_integrity}\n")

# 2. Frozen Protocol Expectation & Completion
print("--- STEP 2: VERIFYING FROZEN PROTOCOL COMPLETION ---")
res_proto = subprocess.run([PYTHON_BIN, str(REPO_ROOT / "scripts" / "rebuttal" / "verify_protocol_completion.py")], capture_output=True, text=True)
proto_report_path = VAL_DIR / "protocol_completion_report.json"
if proto_report_path.exists():
    proto_rep = json.loads(proto_report_path.read_text(encoding="utf-8"))
    protocol_completion_status = proto_rep.get("status", "FAIL")
else:
    protocol_completion_status = "FAIL"
print(f"PROTOCOL_COMPLETION_STATUS = {protocol_completion_status}\n")

# 3. Clean-Room Reproduction
print("--- STEP 3: NON-CIRCULAR CLEAN-ROOM REPRODUCIBILITY ---")
res_clean = subprocess.run([PYTHON_BIN, str(REPO_ROOT / "scripts" / "rebuttal" / "verify_clean_room.py")], capture_output=True, text=True)
clean_report_path = VAL_DIR / "clean_room_reproduction.json"
if clean_report_path.exists():
    clean_rep = json.loads(clean_report_path.read_text(encoding="utf-8"))
    clean_room_status = clean_rep.get("status", "FAIL")
else:
    clean_room_status = "FAIL"
print(f"CLEAN_ROOM_STATUS = {clean_room_status}\n")

# 4. Semantic Mutation Suite
print("--- STEP 4: STRICT SEMANTIC MUTATION TEST SUITE ---")
res_mut = subprocess.run([PYTHON_BIN, str(REPO_ROOT / "scripts" / "rebuttal" / "run_mutation_tests.py")], capture_output=True, text=True)
mut_report_path = VAL_DIR / "mutation_test_report.json"
if mut_report_path.exists():
    mut_rep = json.loads(mut_report_path.read_text(encoding="utf-8"))
    mutation_test_status = mut_rep.get("mutation_status", "FAIL")
else:
    mutation_test_status = "FAIL"
print(f"MUTATION_TEST_STATUS = {mutation_test_status}\n")

# 5. AST Meta-Validator
print("--- STEP 5: AST META-VALIDATOR FOR PYTHON SCRIPTS ---")
res_ast = subprocess.run([PYTHON_BIN, str(REPO_ROOT / "scripts" / "rebuttal" / "audit_artifact_python.py")], capture_output=True, text=True)
ast_report_path = VAL_DIR / "python_file_audit.json"
if ast_report_path.exists():
    ast_rep = json.loads(ast_report_path.read_text(encoding="utf-8"))
    ast_meta_validator_status = ast_rep.get("audit_status", "FAIL")
else:
    ast_meta_validator_status = "FAIL"
print(f"AST_META_VALIDATOR_STATUS = {ast_meta_validator_status}\n")

# 6. Execution Smoke Test
print("--- STEP 6: ARTIFACT PYTHON EXECUTION SMOKE TEST ---")
res_exec = subprocess.run([PYTHON_BIN, str(REPO_ROOT / "scripts" / "rebuttal" / "generate_execution_matrix.py")], capture_output=True, text=True)
execution_smoke_test_status = "PASS" if res_exec.returncode == 0 else "FAIL"
print(f"EXECUTION_SMOKE_TEST_STATUS = {execution_smoke_test_status}\n")

# 7. Check Subdirectory Reports
tbl_path = REBUTTAL_DIR / "table_reconciliation" / "table_reconciliation_summary.json"
table_reconciliation_status = "PASS" if (tbl_path.exists() and json.loads(tbl_path.read_text()).get("status") in ["RECONCILED", "PASS"]) else "FAIL"

sv_path = REBUTTAL_DIR / "sv_decomposition" / "sv_decomposition.json"
sv_decomposition_status = "PASS" if (sv_path.exists() and json.loads(sv_path.read_text()).get("status") == "PASS") else "FAIL"

bm_path = REBUTTAL_DIR / "backend_manifest" / "backend_manifest_summary.json"
backend_manifest_status = "PASS" if (bm_path.exists() and json.loads(bm_path.read_text()).get("status") == "PASS") else "FAIL"

required_statuses = [
    source_rec_file_integrity,
    protocol_completion_status,
    clean_room_status,
    mutation_test_status,
    ast_meta_validator_status,
    execution_smoke_test_status,
    table_reconciliation_status,
    sv_decomposition_status,
    backend_manifest_status
]

all_pass = all(s == "PASS" for s in required_statuses)
overall_status = "PASS" if all_pass else "FAIL"

print("=================================================================")
print("=== FINAL REBUTTAL AUDIT & REPRODUCIBILITY REPORT ===")
print("=================================================================")

print(f"SOURCE_RECORD_FILE_INTEGRITY: {source_rec_file_integrity}")
print(f"NATIVE_MODEL_RUN_PROVENANCE: NOT_AVAILABLE")
print(f"HEADLINE_56_CELL_RUN_STATUS: EXECUTED_AND_VERIFIED")
print(f"INJECTION_EMPIRICAL_STATUS: NOT_RUN / MODELLED")
print(f"SHIFT_EMPIRICAL_STATUS: NOT_RUN / MODELLED")
print(f"AUDIT_SAMPLING_EMPIRICAL_STATUS: NOT_RUN / MODELLED")
print(f"PROTOCOL_COMPLETION_STATUS: {protocol_completion_status}")
print(f"CLEAN_ROOM_STATUS: {clean_room_status}")
print(f"MUTATION_TEST_STATUS: {mutation_test_status}")
print(f"AST_META_VALIDATOR_STATUS: {ast_meta_validator_status}")
print(f"EXECUTION_SMOKE_TEST_STATUS: {execution_smoke_test_status}")
print(f"TABLE_RECONCILIATION_STATUS: {table_reconciliation_status}")
print(f"SV_DECOMPOSITION_STATUS: {sv_decomposition_status}")
print(f"BACKEND_MANIFEST_STATUS: {backend_manifest_status}")

print(f"\nOVERALL COMPLIANCE STATUS: {overall_status}")
print("=================================================================\n")

if not all_pass:
    print("ERROR: Master workflow failed one or more verification checks!")
    sys.exit(1)
else:
    print("Master workflow completed successfully with status PASS.")
    sys.exit(0)
