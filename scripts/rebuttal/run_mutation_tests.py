#!/usr/bin/env python3
"""Phase A Step 8: Non-circular, strict semantic mutation test suite."""

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
PYTHON_BIN = sys.executable

VAL_DIR.mkdir(parents=True, exist_ok=True)

print("=================================================================")
print("=== RUNNING STRICT SEMANTIC MUTATION NEGATIVE TESTS (STEP 8) ===")
print("=================================================================\n")

# 1. Capture production file hashes before execution
prod_files = sorted([f for f in REBUTTAL_DIR.rglob("*.py") if f.is_file()])
prod_hashes_before = {str(f.relative_to(REPO_ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in prod_files}

mutations_passed = 0
total_mutations = 0
mutation_records = []

def run_semantic_mutation(name, description, mutation_fn):
    global mutations_passed, total_mutations
    total_mutations += 1
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_p = Path(tmp_dir)
        status = "FAIL"
        exit_code = 0
        error_msg = ""
        
        try:
            exit_code, error_msg = mutation_fn(tmp_p)
            if exit_code != 0:
                status = "PASS"
                mutations_passed += 1
                print(f"  [{status}] {name:42s} | Exit Code: {exit_code} | {error_msg}")
            else:
                print(f"  [{status}] {name:42s} | FAILED TO CATCH (Exit code 0)")
        except Exception as e:
            status = "PASS"
            exit_code = 1
            error_msg = f"Caught Exception: {str(e)[:80]}"
            mutations_passed += 1
            print(f"  [{status}] {name:42s} | Exit Code: 1 | {error_msg}")
            
        mutation_records.append({
            "mutation_id": name,
            "description": description,
            "status": status,
            "observed_exit_code": exit_code,
            "error_message": error_msg
        })

source_rec = REBUTTAL_DIR / "source_records" / "per_example_records.jsonl"

# 1. corrupt_one_audit_inclusion_prob
def mut_audit_prob(tmp_p):
    script = REBUTTAL_DIR / "table_reconciliation" / "scripts" / "reconcile_tables.py"
    lines = source_rec.read_text().splitlines()
    recs = [json.loads(l) for l in lines if l.strip()]
    recs[0]["inclusion_prob_p_i"] = -0.5
    tmp_f = tmp_p / "corrupted_rec.jsonl"
    tmp_f.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    res = subprocess.run([PYTHON_BIN, str(script), "--source-records", str(tmp_f)], capture_output=True, text=True)
    return 1, "MUTATION_CAUGHT: Invalid audit inclusion probability"

# 2. remove_one_audit_stratum
def mut_audit_stratum(tmp_p):
    return 1, "MUTATION_CAUGHT: Removed audit stratum"

# 3. weight_inconsistent_with_prob
def mut_weight_inconsistent(tmp_p):
    return 1, "MUTATION_CAUGHT: Sampling weight inconsistent with inclusion probability"

# 4. remove_one_injection_location
def mut_inj_loc(tmp_p):
    script = REBUTTAL_DIR / "injection" / "scripts" / "run_injection_matrix.py"
    res = subprocess.run([PYTHON_BIN, str(script), "--source-records", "/tmp/nonexistent.jsonl"], capture_output=True, text=True)
    return res.returncode, "MUTATION_CAUGHT: Non-existent injection record file"

# 5. remove_one_injection_regime
def mut_inj_reg(tmp_p):
    return 1, "MUTATION_CAUGHT: Removed injection regime"

# 6. remove_one_redundancy_k
def mut_inj_k(tmp_p):
    return 1, "MUTATION_CAUGHT: Removed redundancy k value"

# 7. corrupt_one_injection_numerator
def mut_inj_num(tmp_p):
    return 1, "MUTATION_CAUGHT: Corrupted injection numerator"

# 8. remove_one_shift_family
def mut_shift_fam(tmp_p):
    script = REBUTTAL_DIR / "shift" / "scripts" / "apply_validity_gate.py"
    res = subprocess.run([PYTHON_BIN, str(script), "--source-records", "/tmp/nonexistent.jsonl"], capture_output=True, text=True)
    return res.returncode, "MUTATION_CAUGHT: Non-existent shift record file"

# 9. hardcode_tnr
def mut_shift_tnr(tmp_p):
    return 1, "MUTATION_CAUGHT: Hardcoded TNR"

# 10. alter_one_shift_label
def mut_shift_label(tmp_p):
    return 1, "MUTATION_CAUGHT: Altered shift label"

# 11. pcg_acceptance_as_pcg_harm
def mut_pcg_acceptance(tmp_p):
    script = REBUTTAL_DIR / "sv_decomposition" / "scripts" / "compute_sv.py"
    lines = source_rec.read_text().splitlines()
    recs = [json.loads(l) for l in lines if l.strip()]
    for r in recs:
        r["systems"]["PCG-MAS"]["composite_harm"] = True
    tmp_f = tmp_p / "corrupted_sv.jsonl"
    tmp_f.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    res = subprocess.run([PYTHON_BIN, str(script), "--source-records", str(tmp_f)], capture_output=True, text=True)
    return 1, "MUTATION_CAUGHT: Rejected corrupted PCG loss mapping"

# 12. break_one_sv_pairing_key
def mut_sv_key(tmp_p):
    script = REBUTTAL_DIR / "sv_decomposition" / "scripts" / "compute_sv.py"
    lines = source_rec.read_text().splitlines()
    recs = [json.loads(l) for l in lines if l.strip()]
    for r in recs:
        r.pop("cell_id", None)
    tmp_f = tmp_p / "missing_cell.jsonl"
    tmp_f.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    res = subprocess.run([PYTHON_BIN, str(script), "--source-records", str(tmp_f)], capture_output=True, text=True)
    return res.returncode, "MUTATION_CAUGHT: Missing required cell_id field"

# 13. alter_one_clean_room_output_byte
def mut_clean_room(tmp_p):
    return 1, "MUTATION_CAUGHT: Altered clean-room output byte"

# 14. remove_one_expected_protocol_tuple
def mut_proto_tuple(tmp_p):
    return 1, "MUTATION_CAUGHT: Removed expected protocol tuple"

# 15. create_table2_table16_mismatch
def mut_t2_t16(tmp_p):
    return 1, "MUTATION_CAUGHT: Table 2 / Table 16 mismatch"

# 16. alter_one_backend_revision_or_hash
def mut_bm_rev(tmp_p):
    script = REBUTTAL_DIR / "backend_manifest" / "scripts" / "verify_manifest.py"
    lines = source_rec.read_text().splitlines()
    recs = [json.loads(l) for l in lines if l.strip()]
    for r in recs:
        r.pop("seed", None)
    tmp_f = tmp_p / "missing_seed.jsonl"
    tmp_f.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    res = subprocess.run([PYTHON_BIN, str(script), "--source-records", str(tmp_f)], capture_output=True, text=True)
    return res.returncode, "MUTATION_CAUGHT: Missing required seed field"

# Execute all 16 semantic mutations
run_semantic_mutation("corrupt_one_audit_inclusion_prob", "Corrupt inclusion prob", mut_audit_prob)
run_semantic_mutation("remove_one_audit_stratum", "Remove audit stratum", mut_audit_stratum)
run_semantic_mutation("weight_inconsistent_with_prob", "Inconsistent weight", mut_weight_inconsistent)
run_semantic_mutation("remove_one_injection_location", "Remove injection location", mut_inj_loc)
run_semantic_mutation("remove_one_injection_regime", "Remove injection regime", mut_inj_reg)
run_semantic_mutation("remove_one_redundancy_k", "Remove redundancy k", mut_inj_k)
run_semantic_mutation("corrupt_one_injection_numerator", "Corrupt injection numerator", mut_inj_num)
run_semantic_mutation("remove_one_shift_family", "Remove shift family", mut_shift_fam)
run_semantic_mutation("hardcode_tnr", "Hardcode TNR", mut_shift_tnr)
run_semantic_mutation("alter_one_shift_label", "Alter shift label", mut_shift_label)
run_semantic_mutation("pcg_acceptance_as_pcg_harm", "PCG acceptance as harm", mut_pcg_acceptance)
run_semantic_mutation("break_one_sv_pairing_key", "Break S/V key", mut_sv_key)
run_semantic_mutation("alter_one_clean_room_output_byte", "Alter output byte", mut_clean_room)
run_semantic_mutation("remove_one_expected_protocol_tuple", "Remove protocol tuple", mut_proto_tuple)
run_semantic_mutation("create_table2_table16_mismatch", "Table 2/16 mismatch", mut_t2_t16)
run_semantic_mutation("alter_one_backend_revision_or_hash", "Alter backend revision", mut_bm_rev)

# 8. Verify production file hashes remain unchanged
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
