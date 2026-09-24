# DEVELOPMENT_NOTES.md — synchronization and integration notes

This document records synchronization and integration notes for the PCG-MAS release workspace.

---

## 1. Workspace Backup

```bash
cd /path/to/workspace
cp -a pcg-mas "pcg-mas.bak.$(date +%Y%m%d-%H%M%S)"   # if it already exists
```

## 2. Additive Synchronization

```bash
cd /path/to/workspace
DEST=./pcg-mas

rsync -av --exclude '.venv-pcg-mas/' --exclude '__pycache__/' --exclude '*.pyc' \
      /path/to/source/ "$DEST"/

find "$DEST" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null
find "$DEST" -name '*.pyc' -delete 2>/dev/null
```

`rsync` without `--delete` ensures local files are preserved.

## 3. Environment Setup

```bash
cd /path/to/pcg-mas
python3.12 -m venv .venv-pcg-mas
source .venv-pcg-mas/bin/activate
python -V                                   # expect Python 3.12
python -m pip install -U pip setuptools wheel
python -m pip install -r env/constraints-offline.txt
python -m pip install -e .
```

## 4. Verification — Acceptance Test Suite

```bash
python app/shared/generate_contract.py
bash scripts/v3/verify_offline.sh
```

Expected acceptance metrics:

```
PCG_MAS_RELEASE=v3.0
ARCHITECTURE_HARDENING=PASS
SCIENTIFIC_SPEC_ALIGNMENT=PASS
A1_A18_RUNNER_COVERAGE=18/18
TABLE_GENERATORS=32/33
FIGURE_GENERATORS=9/10
PNG_PDF_DUAL_OUTPUT=PASS
UNIT_TESTS=39/39
CROSS_ARTIFACT_CHECKS=PASS
SECRET_LEAK_SCAN=PASS
REAL_EXPERIMENTS_EXECUTED=0
NETWORK_API_MODEL_CALLS=0
```

`32/33` and `9/10` are complete: `tab:notation` and `fig:workflow` are static hand-authored artifacts.

To include manuscript cross-checks:

```bash
PCG_TEX=/path/to/manuscript.tex bash scripts/v3/verify_offline.sh
```

## 5. Regenerating Artifacts on Demand

```bash
PYTHONPATH=src python -m pcg.v3.workstreams.cli list
PYTHONPATH=src python -m pcg.v3.workstreams.cli run all
python scripts/v3/check_figures.py
```

## 6. Documentation References

1. `docs/v3/ARCHITECTURE_AUDIT_V3.md` — system architecture and preserved components
2. `docs/v3/SCIENTIFIC_SPEC_ALIGNMENT_V3.md` — definition-by-definition conformance
3. `docs/v3/REMEDIATION_CHANGELOG_V3.md` — historical changelog and design boundaries
4. `docs/v3/EXPERIMENT_RUNBOOK_V3.md` → `MAC_M4_RUNBOOK.md` / `COLAB_A100_H100_RUNBOOK.md`

## 7. Five Pending Implementation Items

1. Quarantine `scripts/runs/run_preflight.py` (still emits legacy metrics reachable from `Makefile`, `cli.py`, and runner);
2. Remove `--allow-dataset-fallback` from `run_local_40_cells.py`;
3. Triage the 123 legacy `.get(..., 0.0)` sites into counters vs measurements;
4. Retire `eval/rho.py`, `eval/tightness.py` and `eval/metrics.py:compute_sv_decomposition` in favour of `pcg.v3.science.*`;
5. Migrate the legacy `app/static` + `pcg_glue` demo onto the generated contract.

## 8. Frozen Parameter Specifications

Frozen in `src/pcg/v3/workstreams/catalog.py` entering the `spec.json` hash:

* **A08** `n_min=200, k_min=5, q0=2, bar_rho=1.35, Δ=0.15, q_cm=0.05, ε_path=0.18`
* **A15** `α_ent=0.05`, threshold rule "max retained coverage subject to `UCB(FPR) ≤ α_ent`"

## 9. Pipeline Flow

```
v3.0 architecture
  → deterministic verification
  → lightweight execution (11 offline workstreams; Gate 1 and Gate 2)
  → remote compute execution (A05, A08, A09, A15-B/C, A16, large cells)
  → hosted model route where required
  → A02 / A10 / A15 / A16 artifact generation
  → automatic Tables 1-33, Figures 1-10 (PNG + PDF)
  → manuscript result population
  → live-demo artifact refresh
```
