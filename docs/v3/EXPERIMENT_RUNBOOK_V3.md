# EXPERIMENT_RUNBOOK_V3.md

Four phases, strictly separated. **Preparation and verification make no model or network call.**

## Phase 0 — preparation (offline, either machine)

```bash
cd /path/to/pcg-mas
python3.12 -m venv .venv-pcg-mas && source .venv-pcg-mas/bin/activate
python -V                                    # expect Python 3.12.13
python -m pip install -U pip setuptools wheel
python -m pip install -r env/constraints-offline.txt
python -m pip install -e .

python app/shared/generate_contract.py
bash scripts/v3/verify_offline.sh            # must end ARCHITECTURE_HARDENING=PASS
```

### Gate 0 — freeze before any model call

Freeze and hash: dataset splits and example IDs; model IDs and exact revisions; tokenizer revisions; provider routes; dtype/quantisation; prompts and templates; decoding settings; tool schemas and policy bundles (`bundle_sha256`); retriever/index/corpus snapshots; corruption and attack generators; seed list (≥4); metric implementation version; calibration thresholds; **A08 evidence floor and gate band**; **A15 `α_ent` and threshold rule**; audit strata and floors; A14 taxonomy mapping rules; controller sensitivity ranges; pricing snapshot; figure/table generators.

`verify_spec()` refuses a run whose `spec.json` hash has drifted. No evaluation threshold may be chosen after seeing the final split.

## Phase 1 — which workstreams need real model calls

| Needs model calls | Offline (record analysis only) |
|---|---|
| A01, A05, A08, A09, A10, A15, A16 | A02, A03, A04, A06, A07, A11, A12, A13, A14, A17, A18 |

Eleven of eighteen run entirely on the laptop. Run those first: they are cheap, and A02/A03 decide whether the empirical story survives at all.

## Phase 2 — gates

**Gate 1 — A01 → A02 → A03.** What actually ran; what the records say; whether the gain is verification or selectivity. **Stop and reframe if any fails.** Never tune a later experiment to recreate an earlier headline.
**Gate 2 — A15-A checker characterisation on calibration data only.** If the checker cannot reach the pre-registered operating point, change it and re-freeze **before** final evaluation.
**Gate 3 — A16 + A10.** Equal-budget frontier and direct cost. If PCG loses at equal cost, the claim changes.
**Gate 4 — A04, A05, A06, A07, A14.** Recomputation invariance, native-scope baselines, witnesses, sampling contract, open-set taxonomy.
**Gate 5 — A08, A09, A11, A12, A13, A15-B/C.** Adversarial, shift, slack, dependence, TCB, checker degradation.
**Gate 6 — A17.** Controller sensitivity; demote R4 rather than tuning it post hoc.
**Optional — A18.** Only if privacy is restored as a substantive empirical claim.

## Phase 3 — expected artifacts

Each run writes `artifacts/v3_0/<aNN>/` containing `README.md`, `spec.json`, `environment.json`, `metrics.json`, `RESULT.md`, `checks.json`, `SHA256SUMS`, plus `records.jsonl` and backend/pricing manifests where applicable. Tables land in `artifacts/v3_0/tables/{csv,latex}/`, figures in `artifacts/v3_0/figures/{data,png,pdf}/`.

## Phase 4 — verification commands

```bash
bash scripts/v3/verify_offline.sh                       # full status block
python scripts/v3/check_figures.py                      # vector + fonts + text + 300dpi
PYTHONPATH=src python -m pcg.v3.workstreams.cli run all  # all 18 runners
PYTHONPATH=src python -m pcg.v3.workstreams.cli list     # tiers and model-call needs
```

## Resume and recovery

Runners are idempotent and keyed by `record_id`; re-running a partial cell resumes rather than duplicating. A run whose frozen spec hash no longer matches **fails** instead of continuing. After a Colab restart, re-mount Drive, re-activate the venv and re-issue the same command. A cell whose seeds do not share one backend identity is split or rerun, never pooled.
