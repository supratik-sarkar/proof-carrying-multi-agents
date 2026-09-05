# Python File AST Audit Report

## Summary: BASIC_STATIC_AUDIT_PASS (0 Failures out of 54 Files)

Note: Audit performs static hygiene checks (AST parsing, shebang, absolute paths, argument consumption, non-tautological assertions).

| File Path | Classification | Reads Files | Absolute Paths | Status | Failure Reason |
|---|---|---|---|---|---|
| `artifacts/evidence/audit_sampling/scripts/reproduce_all.py` | `REAL_REPRODUCE_ALL` | True | False | **PASS** |  |
| `artifacts/evidence/audit_sampling/scripts/run_sampling_designs.py` | `REAL_IMPLEMENTATION` | True | False | **PASS** |  |
| `artifacts/evidence/audit_sampling/tests/test_audit_sampling.py` | `REAL_TEST` | False | False | **PASS** |  |
| `artifacts/evidence/backend_manifest/scripts/reproduce_all.py` | `REAL_REPRODUCE_ALL` | True | False | **PASS** |  |
| `artifacts/evidence/backend_manifest/scripts/verify_manifest.py` | `VALIDATOR` | True | False | **PASS** |  |
| `artifacts/evidence/backend_manifest/tests/test_backend_manifest.py` | `REAL_TEST` | False | False | **PASS** |  |
| `artifacts/evidence/citation_only/citation_only_baseline.py` | `REAL_IMPLEMENTATION` | False | False | **PASS** |  |
| `artifacts/evidence/citation_only/scripts/match_coverage.py` | `REAL_IMPLEMENTATION` | True | False | **PASS** |  |
| `artifacts/evidence/citation_only/scripts/reproduce_all.py` | `REAL_REPRODUCE_ALL` | True | False | **PASS** |  |
| `artifacts/evidence/citation_only/tests/test_citation_only.py` | `REAL_TEST` | False | False | **PASS** |  |
| `artifacts/evidence/figures/verify_figure_extraction.py` | `VALIDATOR` | True | False | **PASS** |  |
| `artifacts/evidence/injection/scripts/reproduce_all.py` | `REAL_REPRODUCE_ALL` | True | False | **PASS** |  |
| `artifacts/evidence/injection/scripts/run_injection_matrix.py` | `REAL_IMPLEMENTATION` | True | False | **PASS** |  |
| `artifacts/evidence/injection/tests/test_injection.py` | `REAL_TEST` | False | False | **PASS** |  |
| `artifacts/evidence/separating_witnesses/scripts/reproduce_all.py` | `REAL_REPRODUCE_ALL` | True | False | **PASS** |  |
| `artifacts/evidence/separating_witnesses/scripts/run_witness_suite.py` | `REAL_IMPLEMENTATION` | True | False | **PASS** |  |
| `artifacts/evidence/separating_witnesses/tests/test_separating_witnesses.py` | `REAL_TEST` | False | False | **PASS** |  |
| `artifacts/evidence/separating_witnesses/witness_generators.py` | `REAL_IMPLEMENTATION` | False | False | **PASS** |  |
| `artifacts/evidence/shift/scripts/apply_validity_gate.py` | `REAL_IMPLEMENTATION` | True | False | **PASS** |  |
| `artifacts/evidence/shift/scripts/reproduce_all.py` | `REAL_REPRODUCE_ALL` | True | False | **PASS** |  |
| `artifacts/evidence/shift/tests/test_shift.py` | `REAL_TEST` | False | False | **PASS** |  |
| `artifacts/evidence/sv_decomposition/run_sv_decomposition.py` | `REAL_IMPLEMENTATION` | False | False | **PASS** |  |
| `artifacts/evidence/sv_decomposition/scripts/compute_sv.py` | `REAL_IMPLEMENTATION` | True | False | **PASS** |  |
| `artifacts/evidence/sv_decomposition/scripts/paired_bootstrap.py` | `REAL_IMPLEMENTATION` | True | False | **PASS** |  |
| `artifacts/evidence/sv_decomposition/scripts/reproduce_all.py` | `REAL_REPRODUCE_ALL` | True | False | **PASS** |  |
| `artifacts/evidence/sv_decomposition/tests/test_sv_decomposition.py` | `REAL_TEST` | False | False | **PASS** |  |
| `artifacts/evidence/table_reconciliation/scripts/canonical_metrics.py` | `REAL_IMPLEMENTATION` | True | False | **PASS** |  |
| `artifacts/evidence/table_reconciliation/scripts/reconcile_tables.py` | `REAL_IMPLEMENTATION` | True | False | **PASS** |  |
| `artifacts/evidence/table_reconciliation/scripts/reproduce_all.py` | `REAL_REPRODUCE_ALL` | True | False | **PASS** |  |
| `artifacts/evidence/table_reconciliation/tests/test_table_reconciliation.py` | `REAL_TEST` | False | False | **PASS** |  |
| `artifacts/evidence/validation/validate_artifact.py` | `REAL_IMPLEMENTATION` | True | False | **PASS** |  |
| `scripts/validation/audit_artifact_python.py` | `VALIDATOR` | True | False | **PASS** |  |
| `scripts/validation/execute_all_gates.py` | `VALIDATOR` | True | False | **PASS** |  |
| `scripts/validation/execute_full_workflow.py` | `VALIDATOR` | True | False | **PASS** |  |
| `scripts/validation/finalize_artifacts.py` | `REAL_IMPLEMENTATION` | False | False | **PASS** |  |
| `scripts/validation/generate_execution_matrix.py` | `REAL_IMPLEMENTATION` | True | False | **PASS** |  |
| `scripts/validation/plan_56cell_run.py` | `REAL_IMPLEMENTATION` | True | False | **PASS** |  |
| `scripts/validation/run_56cell_server.py` | `REAL_IMPLEMENTATION` | False | False | **PASS** |  |
| `scripts/validation/run_mutation_tests.py` | `REAL_IMPLEMENTATION` | False | False | **PASS** |  |
| `scripts/validation/run_phase1_audit.py` | `REAL_IMPLEMENTATION` | True | False | **PASS** |  |
| `scripts/validation/sanitize_figures.py` | `REAL_IMPLEMENTATION` | True | False | **PASS** |  |
| `scripts/validation/scan_ai_watermarks.py` | `REAL_IMPLEMENTATION` | True | False | **PASS** |  |
| `scripts/validation/sync_to_git_repo.py` | `REAL_IMPLEMENTATION` | False | False | **PASS** |  |
| `scripts/validation/tables/canonical_metrics.py` | `REAL_IMPLEMENTATION` | False | False | **PASS** |  |
| `scripts/validation/tables/generate_all.py` | `REAL_IMPLEMENTATION` | False | False | **PASS** |  |
| `scripts/validation/tables/generate_records.py` | `REAL_IMPLEMENTATION` | True | False | **PASS** |  |
| `scripts/validation/tables/render_manuscript_tables.py` | `REAL_IMPLEMENTATION` | True | False | **PASS** |  |
| `scripts/validation/tables/render_benchmark_tables.py` | `REAL_IMPLEMENTATION` | False | False | **PASS** |  |
| `scripts/validation/tables/validate_all_tables.py` | `REAL_IMPLEMENTATION` | True | False | **PASS** |  |
| `scripts/validation/validate_executed_protocol.py` | `REAL_IMPLEMENTATION` | True | False | **PASS** |  |
| `scripts/validation/validate_submitted_protocol.py` | `REAL_IMPLEMENTATION` | True | False | **PASS** |  |
| `scripts/validation/verify_clean_room.py` | `VALIDATOR` | True | False | **PASS** |  |
| `scripts/validation/verify_protocol_completion.py` | `VALIDATOR` | True | False | **PASS** |  |
| `scripts/validation/verify_source_record_integrity.py` | `VALIDATOR` | True | False | **PASS** |  |
