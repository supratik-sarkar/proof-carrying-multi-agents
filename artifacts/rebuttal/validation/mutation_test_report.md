# Semantic Mutation Test Report — Submission 9327

## Summary: PASS (16 / 16 Mutations Caught)

| Mutation ID | Description | Status | Expected Code | Observed Exit Code | Error Message |
|---|---|---|---|---|---|
| `corrupt_one_audit_inclusion_prob` | Corrupt selection probability | **PASS** | `INVALID_SELECTION_PROBABILITY` | 1 | MUTATION_CAUGHT: Exit code 1 with expected error 'INVALID_SELECTION_PROBABILITY' |
| `remove_one_audit_stratum` | Remove required stratum | **PASS** | `MISSING_REQUIRED_STRATUM` | 1 | MUTATION_CAUGHT: Exit code 1 with expected error 'MISSING_REQUIRED_STRATUM' |
| `weight_inconsistent_with_prob` | Inconsistent weight | **PASS** | `INCONSISTENT_SAMPLING_WEIGHT` | 1 | MUTATION_CAUGHT: Exit code 1 with expected error 'INCONSISTENT_SAMPLING_WEIGHT' |
| `remove_one_injection_location` | Remove attack location | **PASS** | `MISSING_ATTACK_LOCATION` | 1 | MUTATION_CAUGHT: Exit code 1 with expected error 'MISSING_ATTACK_LOCATION' |
| `remove_one_injection_regime` | Remove verifier regime | **PASS** | `MISSING_VERIFIER_REGIME` | 1 | MUTATION_CAUGHT: Exit code 1 with expected error 'MISSING_VERIFIER_REGIME' |
| `remove_one_redundancy_k` | Remove redundancy k | **PASS** | `MISSING_REDUNDANCY_LEVEL` | 1 | MUTATION_CAUGHT: Exit code 1 with expected error 'MISSING_REDUNDANCY_LEVEL' |
| `corrupt_one_injection_numerator` | Corrupt injection aggregate | **PASS** | `INJECTION_AGGREGATE_MISMATCH` | 1 | MUTATION_CAUGHT: Exit code 1 with expected error 'INJECTION_AGGREGATE_MISMATCH' |
| `remove_one_shift_family` | Remove shift family | **PASS** | `MISSING_SHIFT_FAMILY` | 1 | MUTATION_CAUGHT: Exit code 1 with expected error 'MISSING_SHIFT_FAMILY' |
| `hardcode_tnr` | TNR recomputation mismatch | **PASS** | `TNR_RECOMPUTATION_MISMATCH` | 1 | MUTATION_CAUGHT: Exit code 1 with expected error 'TNR_RECOMPUTATION_MISMATCH' |
| `alter_one_shift_label` | Shift aggregate mismatch | **PASS** | `SHIFT_AGGREGATE_MISMATCH` | 1 | MUTATION_CAUGHT: Exit code 1 with expected error 'SHIFT_AGGREGATE_MISMATCH' |
| `pcg_acceptance_as_pcg_harm` | Invalid PCG harm definition | **PASS** | `INVALID_PCG_HARM_DEFINITION` | 1 | MUTATION_CAUGHT: Exit code 1 with expected error 'INVALID_PCG_HARM_DEFINITION' |
| `break_one_sv_pairing_key` | Unpaired example ID | **PASS** | `UNPAIRED_EXAMPLE_ID` | 1 | MUTATION_CAUGHT: Exit code 1 with expected error 'UNPAIRED_EXAMPLE_ID' |
| `alter_one_clean_room_output_byte` | Clean-room hash mismatch | **PASS** | `CLEAN_ROOM_HASH_MISMATCH` | 1 | MUTATION_CAUGHT: Exit code 1 with expected error 'CLEAN_ROOM_HASH_MISMATCH' |
| `remove_one_expected_protocol_tuple` | Protocol expectation mismatch | **PASS** | `PROTOCOL_EXPECTATION_COUNT_MISMATCH` | 1 | MUTATION_CAUGHT: Exit code 1 with expected error 'PROTOCOL_EXPECTATION_COUNT_MISMATCH' |
| `create_table2_table16_mismatch` | Cross table mismatch | **PASS** | `CROSS_TABLE_MISMATCH` | 1 | MUTATION_CAUGHT: Exit code 1 with expected error 'CROSS_TABLE_MISMATCH' |
| `alter_one_backend_revision_or_hash` | Invalid backend revision/hash | **PASS** | `INVALID_BACKEND_REVISION_OR_HASH` | 1 | MUTATION_CAUGHT: Exit code 1 with expected error 'INVALID_BACKEND_REVISION_OR_HASH' |
