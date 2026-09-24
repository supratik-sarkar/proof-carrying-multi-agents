# ARCHITECTURE_AUDIT_V3.md

**Audited tree:** the pre-existing workspace release.
**Method:** static inspection plus bounded offline execution. `REAL_EXPERIMENTS_EXECUTED=0`, `NETWORK_API_MODEL_CALLS=0`, `PAID_API_CALLS=0`, `INTERACTIVE_PROCESSES_STARTED=0`.

## 1. Workspace Configuration

The active release workspace is rooted at `pcg-mas`. `DEVELOPMENT_NOTES.md` provides verification steps and execution runbooks. Confirm the workspace root before running benchmarks.

## 2. What existed and was preserved

| Area | Size | Disposition |
|---|---|---|
| `app/` | 30 files, ~8.8k LOC (FastAPI `server.py`, 14 routes, 640/1405/2497 LOC HTML/JS/CSS, ten `pcg_glue` modules, Dockerfile, demo fixtures) | **preserved intact**; v3 layer added alongside |
| `src/pcg/` | 12,806 LOC, 63 modules | preserved; `src/pcg/v3/` added |
| `scripts/` | 15 sub-areas | preserved; `scripts/v3/` added |
| `artifacts/`, `results/`, `reports/` | 131 MB | preserved as evidence, never deleted |
| `configs/`, `tests/`, `schemas/` | — | preserved |

Nothing was deleted. Stale `cpython-310` bytecode was removed (invalid for the 3.12.13 target and never source).

## 3. Findings carried forward from the prior audit

| ID | Finding | v3.0 disposition |
|---|---|---|
| S1 | `scripts/runs/run_preflight.py:93,95` still generates the withdrawn `safety_gain` and constant `responsibility_lift_pp: 20.0`, reachable from `Makefile`, `cli.py` and `run_local_40_cells.py` | **still present and still reachable.** The v3 pipeline does not use it, and no v3 table can be populated from it, but it is not yet quarantined — see `REMEDIATION_CHANGELOG_V3.md` §Not done |
| S2 | 123 `.get(..., 0.0)` silent-default sites | v3 metric path uses `_rate()`/`None` semantics throughout; **legacy sites untouched** |
| S3 | provenance layer imported in only 2 places | superseded by `pcg.v3` which is wired end to end |
| S4 | no canonical per-example record (4-field stub) | **closed**: `schemas/per_example_record.v3.schema.json`, 87 fields |
| S5 | `DriftFail` appeared once in the whole tree | **closed**: five-channel enum is the single source of truth |
| S6 | `--allow-dataset-fallback` | **still present** in the legacy runner; the v3 runners have no fallback path |
| S7 | A-workstreams not first class | **closed**: 18/18 typed runners execute offline |
| S8 | `eval/rho.py` and `eval/tightness.py` encode superseded semantics; `eval/metrics.py:102` computes S/V from cell rates | superseded by `pcg.v3.science.*`; **legacy modules left in place** and marked in the alignment doc |

## 4. What v3.0 adds

```
src/pcg/v3/
  release.py canon.py channels.py record.py
  stats/     intervals.py bootstrap.py
  science/   audit.py dependence.py sv.py shift.py responsibility.py controller.py
  policy/    interface.py local.py opa.py bundles/{default.json,default.rego}
  providers/ base.py offline_mock.py local_hf.py hosted.py registry.py
  orchestration/ state.py graph.py
  telemetry/ otel.py langsmith.py
  guardrails/ nemo.py
  workstreams/ base.py catalog.py runners.py cli.py
  artifacts/ registry.py emit.py tables.py figures.py
scripts/v3/  make_fixtures.py check_figures.py verify_offline.{sh,py}
tests/v3/    test_v3_core.py  (39 tests)
tests/fixtures/v3/a01..a18/   16,220 deterministic rows
app/{shared,backend,frontend,cloudflare,render}/
```

## 5. Verified in this pass

```
A1_A18_RUNNER_COVERAGE=18/18      TABLE_GENERATORS=32/33      FIGURE_GENERATORS=9/10
PNG_PDF_DUAL_OUTPUT=PASS          UNIT_TESTS=39/39            CROSS_ARTIFACT_CHECKS=PASS
SECRET_LEAK_SCAN=PASS             HEALTH_READY_CHECKS=PASS    OFFLINE_DEMO=PASS
```

32/33 and 9/10 are **complete**: `tab:notation` and `fig:workflow` are static hand-authored artifacts with no data source, and are registered as `STATIC`.

## 6. Residual risk

The legacy modules in §3 (S1, S2, S6, S8) remain in the tree. They are unreachable from any v3 artifact path, but a user invoking `make preflight` or the legacy CLI can still generate withdrawn quantities. Quarantining them touches `Makefile`, `cli.py` and a runner, which is a live-tree edit better performed in subsequent development with the discrepancy documented first — which is what this document does.
