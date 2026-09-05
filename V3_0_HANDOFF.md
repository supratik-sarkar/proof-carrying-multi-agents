# V3_0_HANDOFF.md — synchronising this ZIP into the non-git workspace

**Target (active):** `~/Desktop/pcg-mas-2026`
**Reference only, do not modify:** `~/Desktop/pcg-neurips2026`
**Out of scope, do not touch:** `~/Desktop/My_Git/proof-carrying-multi-agents`

This task was **non-git**. No `git init`, commit, branch, checkout, push or remote change was performed, and none should be performed during the sync.

---

## 0. One discrepancy to resolve first

The instructions name the active workspace `pcg-mas-2026`; the archive I was given was rooted at `pcg-iclr2027`. **I did not guess.** This ZIP is rooted at `pcg-mas-2026` to match the instruction. If your live tree is still named `pcg-iclr2027`, either rename it first or adjust `DEST` below. Do not merge into `pcg-neurips2026`.

## 1. Back up before anything

```bash
cd ~/Desktop
cp -a pcg-mas-2026 "pcg-mas-2026.bak.$(date +%Y%m%d-%H%M%S)"   # if it already exists
```

## 2. Additive sync (recommended)

Everything in this ZIP is additive except removed stale bytecode. Nothing in your tree is deleted.

```bash
cd ~/Desktop
unzip -q pcg-mas-v3.0-nongit.zip -d /tmp/pcgv3
DEST=~/Desktop/pcg-mas-2026

rsync -av --exclude '.venv-pcg-mas/' --exclude '__pycache__/' --exclude '*.pyc' \
      /tmp/pcgv3/pcg-mas-2026/ "$DEST"/

find "$DEST" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null
find "$DEST" -name '*.pyc' -delete 2>/dev/null
```

`rsync` without `--delete` is deliberate: your local files that are not in the ZIP survive.

## 3. Environment

```bash
cd ~/Desktop/pcg-mas-2026
python3.12 -m venv .venv-pcg-mas
source .venv-pcg-mas/bin/activate
python -V                                   # expect Python 3.12.13
python -m pip install -U pip setuptools wheel
python -m pip install -r env/constraints-offline.txt
python -m pip install -e .
```

## 4. Verify — this is the acceptance test

```bash
python app/shared/generate_contract.py
bash scripts/v3/verify_offline.sh
```

Expected (reproduced exactly in the build environment):

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

`32/33` and `9/10` are complete: `tab:notation` and `fig:workflow` are `STATIC` hand-authored artifacts.

To include the manuscript cross-check:

```bash
PCG_TEX=/path/to/pcg_mas_iclr2027_v3-0.tex bash scripts/v3/verify_offline.sh
```

## 5. Regenerate artifacts on demand

```bash
PYTHONPATH=src python -m pcg.v3.workstreams.cli list
PYTHONPATH=src python -m pcg.v3.workstreams.cli run all
python scripts/v3/check_figures.py
```

## 6. Read next

1. `docs/v3/ARCHITECTURE_AUDIT_V3.md` — what existed, what was preserved, what remains
2. `docs/v3/SCIENTIFIC_SPEC_ALIGNMENT_V3.md` — definition-by-definition conformance and the eight discrepancies
3. `docs/v3/REMEDIATION_CHANGELOG_V3.md` — including what was deliberately **not** done
4. `docs/v3/EXPERIMENT_RUNBOOK_V3.md` → `MAC_M4_RUNBOOK.md` / `COLAB_A100_H100_RUNBOOK.md`

## 7. Five live-tree edits left for Antigravity

Deferred because each changes public CLI or demo behaviour, and the constraint was to document a conflict rather than silently decide it:

1. quarantine `scripts/runs/run_preflight.py` (still emits the withdrawn `safety_gain` and `responsibility_lift_pp`, and is reachable from `Makefile:19`, `src/pcg/cli.py:57`, `scripts/runs/run_local_40_cells.py`);
2. remove `--allow-dataset-fallback` from `run_local_40_cells.py`;
3. triage the 123 legacy `.get(..., 0.0)` sites into counters vs measurements;
4. retire `eval/rho.py`, `eval/tightness.py` and `eval/metrics.py:compute_sv_decomposition` in favour of `pcg.v3.science.*`;
5. migrate the legacy `app/static` + `pcg_glue` demo onto the generated contract.

## 8. Two frozen parameters needing your confirmation

Left open, they would have made two release checks vacuous, so they are frozen in `src/pcg/v3/workstreams/catalog.py` and enter every `spec.json` hash:

* **A08** `n_min=200, k_min=5, q0=2, bar_rho=1.35, Δ=0.15, q_cm=0.05, ε_path=0.18`
* **A15** `α_ent=0.05`, threshold rule "max retained coverage subject to `UCB(FPR) ≤ α_ent`"

Changing either after seeing results is a spec violation and `verify_spec()` will refuse the run.

## 9. Next phase

```
Claude v3.0 architecture (this ZIP)
  → Antigravity deterministic sync + verification
  → lightweight Mac-safe execution      (11 offline workstreams; Gate 1 and Gate 2)
  → Google Colab Pro A100/H100          (A05, A08, A09, A15-B/C, A16, large cells)
  → hosted DeepSeek-V3 route where required
  → A02 / A10 / A15 / A16 artifact generation
  → automatic Tables 1-33, Figures 1-10 (PNG + PDF)
  → manuscript result population
  → live-demo artifact refresh
```

None of those experiments was run here.
