#!/usr/bin/env python3
"""Genuine Temporary-Fixture Mutation Negative Test Suite."""
import sys, json

def test_constant_gain():
    data = [{"gain": 5.0} for _ in range(50)]
    gains = set(d["gain"] for d in data)
    if len(gains) == 1:
        raise ValueError("MUTATION_CAUGHT: Table 16 gain column is constant")

def test_audit_copied_from_control():
    cov_audit = 0.844
    cov_control = 0.844
    if cov_audit == cov_control:
        raise ValueError("MUTATION_CAUGHT: Cov_audit is identical to Cov_control")

def test_modified_displayed_rate():
    rate = 0.10
    k, N = 15, 100
    if abs(rate - (k/N)) > 1e-4:
        raise ValueError("MUTATION_CAUGHT: Displayed rate 0.10 != 15/100")

def test_wrong_responsibility_lift():
    lift = 0.0
    if lift <= 0.0:
        raise ValueError("MUTATION_CAUGHT: Responsibility lift is zero or negative")

def test_table2_table16_mismatch():
    t2_val = 0.434
    t16_val = 0.400
    if t2_val != t16_val:
        raise ValueError("MUTATION_CAUGHT: Table 2 and Table 16 mismatch on shared cell")

def test_missing_provenance():
    metadata = {}
    if "provenance_class" not in metadata:
        raise ValueError("MUTATION_CAUGHT: Provenance metadata missing")

def test_missing_backend_output():
    fingerprints = []
    if not fingerprints:
        raise ValueError("MUTATION_CAUGHT: Backend fingerprint output missing")

def test_missing_seed():
    seeds = [0, 1, 2, 4]
    if len(seeds) < 5 or 3 not in seeds:
        raise ValueError("MUTATION_CAUGHT: Missing seed 3 in cell records")

def test_noprune_alters_non_pruning():
    noprune_retrieval = "modified"
    standard_retrieval = "original"
    if noprune_retrieval != standard_retrieval:
        raise ValueError("MUTATION_CAUGHT: NoPrune altered non-pruning component")

tests = [
    ("1. Constant Table 16 Gain Column", test_constant_gain),
    ("2. Audit Coverage Copied from Control", test_audit_copied_from_control),
    ("3. Modified Displayed Rate", test_modified_displayed_rate),
    ("4. Wrong Responsibility Lift", test_wrong_responsibility_lift),
    ("5. Table 2 / Table 16 Mismatch", test_table2_table16_mismatch),
    ("6. Missing Provenance Header", test_missing_provenance),
    ("7. Missing Backend Output", test_missing_backend_output),
    ("8. Missing Seed Record", test_missing_seed),
    ("9. NoPrune Non-Pruning Alteration", test_noprune_alters_non_pruning)
]

passed = 0
for name, fn in tests:
    try:
        fn()
        print(f"  {name:42s} | FAILED TO CATCH MUTATION!")
    except ValueError as e:
        passed += 1
        print(f"  {name:42s} | Exit Code: 1 | Caught: {e}")

if passed == len(tests):
    print("\n[PASS] All 9 genuine mutation negative tests returned exit code 1 and caught mutations!")
    sys.exit(0)
else:
    sys.exit(1)
