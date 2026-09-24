# REMEDIATION_CHANGELOG_V3.md

`WORKSPACE_MODIFIED=YES` (in the delivered ZIP; your live tree is untouched until you sync).

## Added

**Scientific core** — `src/pcg/v3/`: `release`, `canon`, `channels`, `record`; `stats/{intervals,bootstrap}`; `science/{audit,dependence,sv,shift,responsibility,controller}`; `policy/{interface,local,opa,bundles}`; `providers/{base,offline_mock,local_hf,hosted,registry}`; `orchestration/{state,graph}`; `telemetry/{otel,langsmith}`; `guardrails/nemo`; `workstreams/{base,catalog,runners,cli}`; `artifacts/{registry,emit,tables,figures}`.

**Schemas** — `schemas/per_example_record.v3.schema.json` (87 fields).

**Registry** — `manuscript_artifact_registry.json` (33 tables, 10 figures, subtables attached, verified against the manuscript).

**Scripts** — `scripts/v3/{make_fixtures,check_figures,verify_offline}.py`, `verify_offline.sh`.

**Fixtures** — `tests/fixtures/v3/a01..a18/`, 16,220 deterministic rows, all `TEST_FIXTURE`.

**Tests** — `tests/v3/test_v3_core.py`, 39 tests.

**App** — `app/{shared,backend,frontend,cloudflare,render}/`, `app/DEPLOYMENT.md`, `app/APP_V3_ARCHITECTURE.md`.

**Docs** — this directory, plus `DEVELOPMENT_NOTES.md` at the root.

**Generated artifacts** — `artifacts/v3_0/`: 18 workstream directories, 32 CSV + 32 LaTeX tables, 9 figures as PNG + vector PDF with source data, `checks/{figure_gate,verification}.json`.

## Removed

Stale `__pycache__` / `*.pyc` compiled by **CPython 3.10** — invalid for the 3.12.13 target, never source.

## Unmodified

Everything else, including all of `app/server.py`, `app/static/`, `app/pcg_glue/`, `app/demo_data/`, `src/pcg/*` (legacy), `scripts/*` (legacy), `configs/`, and the whole of `artifacts/`, `results/`, `reports/` as evidence. **The scientific `.tex` and `.md` were not edited.**

## Deliberately NOT done — needs a live-tree decision

| Item | Why deferred |
|---|---|
| Quarantine `scripts/runs/run_preflight.py` (still emits withdrawn `safety_gain` / `responsibility_lift_pp`) | reachable from `Makefile:19`, `src/pcg/cli.py:57`, `run_local_40_cells.py`; removing it changes public CLI behaviour |
| Remove `--allow-dataset-fallback` | same runner |
| Triage 123 legacy `.get(...,0.0)` sites | each needs a counter-vs-measurement judgement |
| Retire `eval/rho.py`, `eval/tightness.py`, `eval/metrics.py:compute_sv_decomposition` | superseded by `pcg.v3.science.*` but still imported by legacy scripts |
| Migrate legacy `app/static` + `pcg_glue` onto the generated contract | changes the existing public demo |

All five are documented in `SCIENTIFIC_SPEC_ALIGNMENT_V3.md` §2 with the conforming v3 implementation named.

## Two frozen decisions requiring your confirmation

Previously open; left open they would have made two checks vacuous, so they are frozen in `catalog.py` and enter every spec hash:

1. **A08** — `n_min=200, k_min=5, q0=2, bar_rho=1.35, Δ=0.15, q_cm=0.05, ε_path=0.18`
2. **A15** — `α_ent=0.05`, "max retained coverage subject to `UCB(FPR) ≤ α_ent`"

## Verification actually performed

```
ARCHITECTURE_HARDENING=PASS   SCIENTIFIC_SPEC_ALIGNMENT=PASS
A1_A18_RUNNER_COVERAGE=18/18  TABLE_GENERATORS=32/33  FIGURE_GENERATORS=9/10
PNG_PDF_DUAL_OUTPUT=PASS      UNIT_TESTS=39/39        PROPERTY_TESTS=39/39
CROSS_ARTIFACT_CHECKS=PASS    OFFLINE_SMOKE_TESTS=PASS
HEALTH_READY_CHECKS=PASS      SECRET_LEAK_SCAN=PASS   OFFLINE_DEMO=PASS
CLOUDFLARE_BUILD=PASS         RENDER_BACKEND_BUILD=PASS
REAL_EXPERIMENTS_EXECUTED=0   NETWORK_API_MODEL_CALLS=0
PAID_API_CALLS=0              INTERACTIVE_PROCESSES_STARTED=0
```

32/33 and 9/10 are complete: `tab:notation` and `fig:workflow` are `STATIC` hand-authored artifacts with no data source.

## Defects found in my own work during this pass, and fixed

1. `tau_star_exact` root-found a bimodal stationarity condition and converged to the boundary — replaced with direct minimisation; the exact minimiser now provably beats the closed-form approximation.
2. The default controller cost model made `Verify`/`Escalate` unreachable, which would have made A17 vacuous — recalibrated so all four actions are optimal on some risk interval (thresholds 0.25/0.5/0.75).
3. Two fixture generators set a channel flag without updating `n_channels_fired`; the record validator caught both.
