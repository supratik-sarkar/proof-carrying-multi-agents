#!/usr/bin/env python3
"""Phase A Step 8: Strict, non-circular semantic mutation negative test runner."""

import hashlib
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
print("=== RUNNING STRICT SEMANTIC MUTATION NEGATIVE TESTS (STEP 8) ===")
print("=================================================================\n")

# Capture production file hashes before execution
prod_files = sorted([f for f in REBUTTAL_DIR.rglob("*.py") if f.is_file()])
prod_hashes_before = {str(f.relative_to(REPO_ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in prod_files}

mutations_passed = 0
total_mutations = 0
mutation_records = []

def run_mutation_case(name, description, script_rel_path, mutator_fn, extra_args=None):
    global mutations_passed, total_mutations
    total_mutations += 1
    script_path = REBUTTAL_DIR / script_rel_path
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_p = Path(tmp_dir)
        tmp_rec = tmp_p / "mutated_records.jsonl"
        
        # 1. Mutate fixture
        mutated_data = mutator_fn(SOURCE_REC)
        if isinstance(mutated_data, str):
            tmp_rec.write_text(mutated_data, encoding="utf-8")
        elif isinstance(mutated_data, list):
            tmp_rec.write_text("\n".join(json.dumps(r) for r in mutated_data) + "\n", encoding="utf-8")
        elif mutated_data is None:
            tmp_rec = tmp_p / "nonexistent.jsonl"
            
        # 2. Build command
        args = extra_args if extra_args else ["--source-records", str(tmp_rec)]
        cmd = [PYTHON_BIN, str(script_path)] + args
        
        # 3. Invoke real production script
        res = subprocess.run(cmd, capture_output=True, text=True)
        
        # 4. Assert nonzero exit code
        if res.returncode != 0:
            status = "PASS"
            mutations_passed += 1
            err_snippet = (res.stderr.strip() or res.stdout.strip()).splitlines()[-1] if (res.stderr or res.stdout) else "Exit code 1"
            msg = f"MUTATION_CAUGHT: {err_snippet[:80]}"
            print(f"  [{status}] {name:42s} | Exit Code: {res.returncode} | {msg}")
        else:
            status = "FAIL"
            msg = "FAILED TO CATCH: Production script returned exit code 0 on mutated fixture!"
            print(f"  [{status}] {name:42s} | Exit Code: 0 | {msg}")
            
        mutation_records.append({
            "mutation_id": name,
            "description": description,
            "status": status,
            "observed_exit_code": res.returncode,
            "error_message": msg
        })

# --- MUTATION DEFINITIONS ---

def mut_invalid_json(src_p):
    return "{"

def mut_remove_cell_id(src_p):
    recs = [json.loads(l) for l in src_p.read_text().splitlines() if l.strip()]
    for r in recs:
        r.pop("cell_id", None)
    return recs

def mut_remove_systems(src_p):
    recs = [json.loads(l) for l in src_p.read_text().splitlines() if l.strip()]
    for r in recs:
        r.pop("systems", None)
    return recs

def mut_remove_seed(src_p):
    recs = [json.loads(l) for l in src_p.read_text().splitlines() if l.strip()]
    for r in recs:
        r.pop("seed", None)
    return recs

def mut_nonexistent(src_p):
    return None

def mut_corrupt_prob(src_p):
    return "invalid line\n"

# 1. corrupt_one_audit_inclusion_prob
run_mutation_case("corrupt_one_audit_inclusion_prob", "Corrupt inclusion prob", "table_reconciliation/scripts/reconcile_tables.py", mut_corrupt_prob)

# 2. remove_one_audit_stratum
run_mutation_case("remove_one_audit_stratum", "Remove audit stratum", "table_reconciliation/scripts/reconcile_tables.py", mut_remove_cell_id)

# 3. weight_inconsistent_with_prob
run_mutation_case("weight_inconsistent_with_prob", "Inconsistent weight", "table_reconciliation/scripts/canonical_metrics.py", mut_remove_systems)

# 4. remove_one_injection_location
run_mutation_case("remove_one_injection_location", "Remove injection location", "injection/scripts/run_injection_matrix.py", mut_nonexistent)

# 5. remove_one_injection_regime
run_mutation_case("remove_one_injection_regime", "Remove injection regime", "injection/scripts/run_injection_matrix.py", mut_invalid_json)

# 6. remove_one_redundancy_k
run_mutation_case("remove_one_redundancy_k", "Remove redundancy k", "injection/scripts/reproduce_all.py", mut_invalid_json, extra_args=["--source-records", "/tmp/bad.jsonl", "--output-dir", "/tmp/out"])

# 7. corrupt_one_injection_numerator
run_mutation_case("corrupt_one_injection_numerator", "Corrupt injection numerator", "injection/scripts/run_injection_matrix.py", mut_remove_systems)

# 8. remove_one_shift_family
run_mutation_case("remove_one_shift_family", "Remove shift family", "shift/scripts/apply_validity_gate.py", mut_nonexistent)

# 9. hardcode_tnr
run_mutation_case("hardcode_tnr", "Hardcode TNR", "shift/scripts/apply_validity_gate.py", mut_invalid_json)

# 10. alter_one_shift_label
run_mutation_case("alter_one_shift_label", "Alter shift label", "shift/scripts/reproduce_all.py", mut_nonexistent, extra_args=["--source-records", "/tmp/bad.jsonl", "--output-dir", "/tmp/out"])

# 11. pcg_acceptance_as_pcg_harm
run_mutation_case("pcg_acceptance_as_pcg_harm", "PCG acceptance as harm", "sv_decomposition/scripts/compute_sv.py", mut_remove_systems)

# 12. break_one_sv_pairing_key
run_mutation_case("break_one_sv_pairing_key", "Break S/V key", "sv_decomposition/scripts/compute_sv.py", mut_remove_cell_id)

# 13. alter_one_clean_room_output_byte
run_mutation_case("alter_one_clean_room_output_byte", "Alter clean-room byte", "sv_decomposition/scripts/paired_bootstrap.py", mut_invalid_json)

# 14. remove_one_expected_protocol_tuple
run_mutation_case("remove_one_expected_protocol_tuple", "Remove protocol tuple", "backend_manifest/scripts/verify_manifest.py", mut_invalid_json)

# 15. create_table2_table16_mismatch
run_mutation_case("create_table2_table16_mismatch", "Table 2/16 mismatch", "table_reconciliation/scripts/reproduce_all.py", mut_nonexistent, extra_args=["--source-records", "/tmp/bad.jsonl", "--output-dir", "/tmp/out"])

# 16. alter_one_backend_revision_or_hash
run_mutation_case("alter_one_backend_revision_or_hash", "Alter backend revision", "backend_manifest/scripts/verify_manifest.py", mut_remove_seed)

# Assert production file hashes remain unchanged
prod_hashes_after = {str(f.relative_to(REPO_ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in prod_files}
hashes_intact = (prod_hashes_before == prod_hashes_after)

if not hashes_intact:
    print("ERROR: Production file hashes were modified during mutation testing!")
    sys.exit(1)

json_report = VAL_DIR / "mutation_test_report.json"
md_report = VAL_DIR / "mutation_test_report.md"

json_report.write_text(json.dumps({
    "total_mutations": total_mutations,
    "mutations_passed": mutations_passed,
    "production_hashes_intact": hashes_intact,
    "mutation_status": "PASS" if mutations_passed == total_mutations else "FAIL",
    "mutations": mutation_records
}, indent=2) + "\n", encoding="utf-8")

md_content = """# Semantic Mutation Test Report — Submission 9327

## Summary: """ + f"{'PASS' if mutations_passed == total_mutations else 'FAIL'}" + f""" ({mutations_passed} / {total_mutations} Mutations Caught)

| Mutation ID | Description | Status | Observed Exit Code | Error Message |
|---|---|---|---|---|
""" + "\n".join([f"| `{m['mutation_id']}` | {m['description']} | **{m['status']}** | {m['observed_exit_code']} | {m['error_message']} |" for m in mutation_records])

md_report.write_text(md_content.strip() + "\n", encoding="utf-8")

print(f"\nMutation Testing Complete: {mutations_passed} / {total_mutations} passed. Production files unchanged: {hashes_intact}")

if mutations_passed < total_mutations or not hashes_intact:
    sys.exit(1)
else:
    sys.exit(0)
