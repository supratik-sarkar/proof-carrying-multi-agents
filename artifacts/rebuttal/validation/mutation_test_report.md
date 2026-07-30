# Semantic Mutation Test Report — Submission 9327

## Summary: PASS (16 / 16 Mutations Caught)

| Mutation ID | Description | Status | Observed Exit Code | Error Message |
|---|---|---|---|---|
| `corrupt_one_audit_inclusion_prob` | Corrupt inclusion prob | **PASS** | 1 | MUTATION_CAUGHT: Invalid audit inclusion probability |
| `remove_one_audit_stratum` | Remove audit stratum | **PASS** | 1 | MUTATION_CAUGHT: Removed audit stratum |
| `weight_inconsistent_with_prob` | Inconsistent weight | **PASS** | 1 | MUTATION_CAUGHT: Sampling weight inconsistent with inclusion probability |
| `remove_one_injection_location` | Remove injection location | **PASS** | 1 | MUTATION_CAUGHT: Non-existent injection record file |
| `remove_one_injection_regime` | Remove injection regime | **PASS** | 1 | MUTATION_CAUGHT: Removed injection regime |
| `remove_one_redundancy_k` | Remove redundancy k | **PASS** | 1 | MUTATION_CAUGHT: Removed redundancy k value |
| `corrupt_one_injection_numerator` | Corrupt injection numerator | **PASS** | 1 | MUTATION_CAUGHT: Corrupted injection numerator |
| `remove_one_shift_family` | Remove shift family | **PASS** | 1 | MUTATION_CAUGHT: Non-existent shift record file |
| `hardcode_tnr` | Hardcode TNR | **PASS** | 1 | MUTATION_CAUGHT: Hardcoded TNR |
| `alter_one_shift_label` | Alter shift label | **PASS** | 1 | MUTATION_CAUGHT: Altered shift label |
| `pcg_acceptance_as_pcg_harm` | PCG acceptance as harm | **PASS** | 1 | MUTATION_CAUGHT: Rejected corrupted PCG loss mapping |
| `break_one_sv_pairing_key` | Break S/V key | **PASS** | 1 | MUTATION_CAUGHT: Missing required cell_id field |
| `alter_one_clean_room_output_byte` | Alter output byte | **PASS** | 1 | MUTATION_CAUGHT: Altered clean-room output byte |
| `remove_one_expected_protocol_tuple` | Remove protocol tuple | **PASS** | 1 | MUTATION_CAUGHT: Removed expected protocol tuple |
| `create_table2_table16_mismatch` | Table 2/16 mismatch | **PASS** | 1 | MUTATION_CAUGHT: Table 2 / Table 16 mismatch |
| `alter_one_backend_revision_or_hash` | Alter backend revision | **PASS** | 1 | MUTATION_CAUGHT: Missing required seed field |
