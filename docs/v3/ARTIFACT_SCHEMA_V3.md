# ARTIFACT_SCHEMA_V3.md

## Artifact directory contract

Every workstream emits `artifacts/v3_0/<aNN>/`:

| File | Required | Purpose |
|---|---|---|
| `README.md` | yes | what it is + exact reproduction command |
| `spec.json` | yes | frozen pre-registration (hashed) |
| `environment.json` | yes | interpreter, platform, device, capture time |
| `metrics.json` | yes | canonical metric output |
| `checks.json` | yes | per-gate pass/fail |
| `RESULT.md` | yes | findings **and** failures/limitations |
| `SHA256SUMS` | yes | hash of every other file |
| `records.jsonl` | where applicable | raw per-example records |
| `backend_manifest.json`, `pricing_manifest.json` | where model calls occur | identity and billing |

Central tree:

```
artifacts/v3_0/
  a01..a18/            per-workstream artifact directories
  tables/csv/          32 generated CSV
  tables/latex/        32 generated .tex (each carries a GENERATED FILE header)
  figures/data/        source data + generation config per figure
  figures/png/         >= 300 dpi
  figures/pdf/         vector, embedded fonts, extractable text
  checks/              figure_gate.json, verification.json
  manifests/ reports/ demo/
```

## Canonical per-example record — 87 fields

`schemas/per_example_record.v3.schema.json`, `schema_version = 3.0.0`, `additionalProperties: false`.

| Group | Fields |
|---|---|
| identity | `record_id, run_id, experiment_id, system, provenance_class, metric_version, schema_version, config_hash, spec_hash, cell_id, dataset, split, example_id, seed` |
| backend / checker | `model_id, model_revision, tokenizer_id, backend_type, provider_route, dtype, quantization, decoding_config_hash, prompt_hash, backend_fingerprint, checker_fingerprint` |
| claim & evidence | `claim_id, evidence_ids, evidence_hashes` |
| conjuncts | `v_h, v_pi, v_gamma, v_entail, check, certificate_hash` |
| audit channels | `int_fail, replay_fail, drift_fail, check_fail, cov_gap, n_channels_fired` |
| residual labels | `contract_bad, eps_tax_label, eps_src_label` |
| outcome | `attempted, answered, accepted, controller_action, h_support, h_exec, h_joint, utility, loss_nocert, loss_pcg` |
| coverage | `cov_cert, cov_audit` — two **distinct** senses, never one column |
| dependence | `k_redundancy, branch_ids, branch_failures` |
| responsibility | `resp_scores, resp_top1, resp_top3, resp_margin, unresolved, replay_budget_M, failure_origin_known` |
| policy | `policy_bundle_hash, policy_decision` |
| cost | `latency_ms, phase_timings_ms, tokens_in, tokens_out, model_calls, retrieval_calls, tool_calls, checker_calls, replay_calls, billed_cost_usd, cache_state` |
| sampling / regime | `stratum_id, sampling_weight, corruption, shift_regime, injection_regime` |
| environment | `host_fingerprint, device, source_record_hash, code_fingerprint` |

### Validation rules enforced in code

* `check = True` requires all four conjuncts `True` — unknown is failure;
* `n_channels_fired` must equal the number of channels actually `True`;
* a refused example must not carry `loss_pcg`;
* `provenance_class`, `system` and `controller_action` are closed enumerations;
* `failure_origin_known` distinguishes controlled injected faults from natural failures, so R3 never claims ground truth it does not have.

## Provenance classes

`DIRECT` executed records · `DERIVED` disclosed function of DIRECT · `MODELLED` analytic with declared hand-chosen coefficients · `PROTOCOL` design/specification · `STATIC` hand-authored (notation table, workflow schematic) · `TEST_FIXTURE` / `MOCK` / `REPLAY` / `UNKNOWN`.

**A `TEST_FIXTURE` record may never be promoted to `DIRECT`.** Every fixture in `tests/fixtures/v3/` is stamped `TEST_FIXTURE`, and every figure generated from them carries an on-canvas `SYNTHETIC DEMO` stamp.
