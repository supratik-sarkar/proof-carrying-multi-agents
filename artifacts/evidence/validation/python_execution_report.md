# Artifact Python Execution Smoke Test Report

## Summary: FAIL (4 Failures out of 23 Scripts Executed)

> [!NOTE]
> This is a CLI smoke test verifying command-line execution and exit code responses on valid and invalid CLI parameters.

| Script Path | Valid Exit Code | Created Files | Invalid Exit Code | Invalid Error Output | Smoke Test Status |
|---|---|---|---|---|---|
| `artifacts/evidence/audit_sampling/scripts/reproduce_all.py` | 0 | `audit_sampling_summary.json, reproduction_manifest.json, audit_sampling_summary.csv` | 2 | `reproduce_all.py: error: the following arguments are required: --output-dir` | **PASS** |
| `artifacts/evidence/audit_sampling/scripts/run_sampling_designs.py` | 0 | `result.json` | 1 | `FileNotFoundError: Source records not found: /tmp/nonexistent_file_xyz.jsonl` | **PASS** |
| `artifacts/evidence/backend_manifest/scripts/reproduce_all.py` | 0 | `backend_manifest_summary.csv, reproduction_manifest.json, backend_manifest_summary.json` | 2 | `reproduce_all.py: error: the following arguments are required: --output-dir` | **PASS** |
| `artifacts/evidence/backend_manifest/scripts/verify_manifest.py` | 0 | `result.json` | 1 | `FileNotFoundError: Backend manifest not found: /tmp/nonexistent_file_xyz.jsonl` | **PASS** |
| `artifacts/evidence/citation_only/citation_only_baseline.py` | 0 | `none` | 0 | `Exit code 1` | **FAIL** |
| `artifacts/evidence/citation_only/scripts/match_coverage.py` | 0 | `result.json` | 1 | `FileNotFoundError: Source records not found: /tmp/nonexistent_file_xyz.jsonl` | **PASS** |
| `artifacts/evidence/citation_only/scripts/reproduce_all.py` | 0 | `citation_only_comparison.json, reproduction_manifest.json, citation_only_comparison.csv` | 2 | `reproduce_all.py: error: the following arguments are required: --output-dir` | **PASS** |
| `artifacts/evidence/figures/verify_figure_extraction.py` | 0 | `none` | 0 | `figures inspected: 7 | failing: 0` | **FAIL** |
| `artifacts/evidence/injection/scripts/reproduce_all.py` | 0 | `injection_matrix.csv, reproduction_manifest.json, injection_matrix.json` | 2 | `reproduce_all.py: error: the following arguments are required: --output-dir` | **PASS** |
| `artifacts/evidence/injection/scripts/run_injection_matrix.py` | 0 | `result.json` | 1 | `FileNotFoundError: Source records not found: /tmp/nonexistent_file_xyz.jsonl` | **PASS** |
| `artifacts/evidence/separating_witnesses/scripts/reproduce_all.py` | 0 | `reproduction_manifest.json, separating_witnesses.csv, separating_witnesses.json` | 2 | `reproduce_all.py: error: the following arguments are required: --output-dir` | **PASS** |
| `artifacts/evidence/separating_witnesses/scripts/run_witness_suite.py` | 0 | `result.json` | 1 | `FileNotFoundError: Source records not found: /tmp/nonexistent_file_xyz.jsonl` | **PASS** |
| `artifacts/evidence/separating_witnesses/witness_generators.py` | 0 | `none` | 0 | `Exit code 1` | **FAIL** |
| `artifacts/evidence/shift/scripts/apply_validity_gate.py` | 0 | `result.json` | 1 | `FileNotFoundError: Source records not found: /tmp/nonexistent_file_xyz.jsonl` | **PASS** |
| `artifacts/evidence/shift/scripts/reproduce_all.py` | 0 | `shift_validity_summary.json, reproduction_manifest.json, shift_validity_summary.csv` | 2 | `reproduce_all.py: error: the following arguments are required: --output-dir` | **PASS** |
| `artifacts/evidence/sv_decomposition/run_sv_decomposition.py` | 0 | `none` | 0 | `Exit code 1` | **FAIL** |
| `artifacts/evidence/sv_decomposition/scripts/compute_sv.py` | 0 | `result.json` | 1 | `FileNotFoundError: Source records not found: /tmp/nonexistent_file_xyz.jsonl` | **PASS** |
| `artifacts/evidence/sv_decomposition/scripts/paired_bootstrap.py` | 0 | `result.json` | 1 | `FileNotFoundError: Source records not found: /tmp/nonexistent_file_xyz.jsonl` | **PASS** |
| `artifacts/evidence/sv_decomposition/scripts/reproduce_all.py` | 0 | `sv_decomposition.json, reproduction_manifest.json, sv_bootstrap_ci.json, sv_decomposition.csv` | 2 | `reproduce_all.py: error: the following arguments are required: --output-dir` | **PASS** |
| `artifacts/evidence/table_reconciliation/scripts/canonical_metrics.py` | 0 | `result.json` | 1 | `FileNotFoundError: Source records not found: /tmp/nonexistent_file_xyz.jsonl` | **PASS** |
| `artifacts/evidence/table_reconciliation/scripts/reconcile_tables.py` | 0 | `result.json` | 1 | `FileNotFoundError: Source records not found: /tmp/nonexistent_file_xyz.jsonl` | **PASS** |
| `artifacts/evidence/table_reconciliation/scripts/reproduce_all.py` | 0 | `table_reconciliation_summary.json, table_reconciliation_summary.csv, reproduction_manifest.json, table_reconciliation_summary.html` | 2 | `reproduce_all.py: error: the following arguments are required: --output-dir` | **PASS** |
| `artifacts/evidence/validation/validate_artifact.py` | 0 | `result.json` | 1 | `FileNotFoundError: Source records file not found: /tmp/nonexistent_file_xyz.json` | **PASS** |
