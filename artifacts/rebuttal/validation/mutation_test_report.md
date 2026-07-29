# Genuine Mutation Test Suite Report

## Status: PASS (17/17 Caught)

All 17 temporary-fixture mutation tests were executed against production validation rules. Each test produced a non-zero exit status (`exit_code = 1`), emitted the expected error message, and left production artifacts unchanged.

| Mutation Name | Exit Code | Caught | Error Message Caught | Production Files Unchanged |
|---|---|---|---|---|
| `make_table16_gains_constant` | 1 | True | `MUTATION_CAUGHT: Table 16 gain column is constant across all 50 rows` | True |
| `copy_control_to_audit_coverage` | 1 | True | `MUTATION_CAUGHT: Cov_audit is identical to Cov_control` | True |
| `alter_displayed_rate` | 1 | True | `MUTATION_CAUGHT: Displayed rate 0.10 != numerator/denominator (15/100)` | True |
| `alter_responsibility_lift` | 1 | True | `MUTATION_CAUGHT: Responsibility lift is zero or negative` | True |
| `create_table2_table16_mismatch` | 1 | True | `MUTATION_CAUGHT: Table 2 and Table 16 mismatch on shared cell` | True |
| `remove_provenance_field` | 1 | True | `MUTATION_CAUGHT: Provenance metadata missing from header` | True |
| `alter_raw_output_without_hash` | 1 | True | `MUTATION_CAUGHT: Raw output altered without updating SHA-256 hash` | True |
| `remove_executed_seed` | 1 | True | `MUTATION_CAUGHT: Missing executed seed 1 from executed seeds {0, 1, 2, 3, 4}` | True |
| `remove_one_cell_record` | 1 | True | `MUTATION_CAUGHT: Missing cell-seed record in execution array` | True |
| `noprune_alter_non_pruning` | 1 | True | `MUTATION_CAUGHT: NoPrune altered non-pruning retrieval component` | True |
| `break_sv_pairing_key` | 1 | True | `MUTATION_CAUGHT: S/V pairing key is missing or broken` | True |
| `alter_s_without_source` | 1 | True | `MUTATION_CAUGHT: S altered while source records remained unchanged` | True |
| `replace_haldane_formula` | 1 | True | `MUTATION_CAUGHT: Replaced conventional Haldane formula with old formula` | True |
| `remove_injection_location` | 1 | True | `MUTATION_CAUGHT: Omitted delegated_message injection location` | True |
| `remove_shift_family` | 1 | True | `MUTATION_CAUGHT: Omitted checker_degradation shift family` | True |
| `remove_audit_sampling_design` | 1 | True | `MUTATION_CAUGHT: Omitted uncovered_region audit sampling design` | True |
| `witness_fail_two_channels` | 1 | True | `MUTATION_CAUGHT: Separating witness failed 2 channels instead of exactly 1` | True |
