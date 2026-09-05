# PCG-MAS — Proof-Carrying Generation for Multi-Agent LLM Systems

Certificate-carrying acceptance for multi-agent LLM runs. A claim is accepted only when it
carries a checkable certificate `Z = (c, S, Π, Γ, p, meta)` and the acceptance predicate
`Check(Z; G_t) = V_H · V_Π · V_Γ · V_⊢` holds: evidence-commitment integrity, replay
consistency, execution-contract compliance, and entailment. The same certificate then supports
audit decomposition, redundancy, mask-and-replay responsibility, and risk-aware control over
`answer / verify / escalate / refuse`.

---

## ⚠️ Scientific status — read before using any number from this repository

This repository is **engineering-complete and empirically unverified.**

The 56-cell record set (`artifacts/evidence/source_records/per_example_records.jsonl`,
13,440 records) is **structurally perfect and of unknown origin**:

| Property | Status |
|---|---|
| Structural completeness | **PASS** |
| Internal consistency | **PASS** |
| Native execution provenance | **NOT_AVAILABLE** |
| Empirical authenticity | **NOT_ESTABLISHED** |
| Origin | **UNKNOWN_WITH_STRONG_PRE_EXECUTION_DOCUMENTARY_INDICATORS** |
| Safe for derived offline reproduction | **CONDITIONAL** |
| Safe for empirical manuscript claims | **NO** |

Every table and figure this repository produces is therefore labelled
`REGENERATED_FROM_UNKNOWN_PROVENANCE_56_CELL` and `NOT_SAFE_FOR_EMPIRICAL_MANUSCRIPT_USE`.
They exist to validate the engineering and the cosmetic contract — **not to support a claim.**

A bounded origin investigation (Phase 4B) found no generator, reproduced no RNG mechanism, and found
no native run evidence. It did find contemporaneous project documentation — `SERVER_RUN_HANDOFF.md`,
written in the same minute as the records — instructing that the matrix be executed and the records
**replaced with DIRECT records from actual runs**. The records are therefore treated as
pre-execution material of unknown origin; they are *not* asserted to be synthetic. Full reasoning and evidence: `reports/ARTIFACT_56_CELL_LINEAGE.md`.

To make this repository empirically usable, execute the 56-cell matrix on real backends and
replace the record file with records carrying native evidence. The contract for that run is
`artifacts/evidence/source_records/SERVER_RUN_HANDOFF.md`.

---

## Install

Python 3.10–3.12, Linux or macOS (Windows via WSL2). The base install is deliberately small: every
heavy dependency in `src/pcg` is imported lazily inside a function, so nothing about importing the
package, running the tests, or reproducing the stored artifacts requires the model stack.

> **Note on network.** These commands download packages from PyPI. Only the *reproduction* of the
> stored 56-cell artifacts is offline (see below); package installation itself is not, and has not
> been tested against a purely local wheel cache.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

### 1. Core / basic

```bash
python -m pip install -e .
```

Installs **numpy** and **scipy** only. Sufficient to import the whole certificate stack —
`pcg.certificate`, `pcg.checker`, `pcg.commitments`, `pcg.graph`, `pcg.responsibility`, `pcg.risk`,
`pcg.privacy`, `pcg.independence`, `pcg.eval.{audit,metrics,rho,stats}`.

Optional-backend modules (`pcg.cli`, `pcg.eval.meter`, `pcg.orchestrator.langgraph_flow`,
`pcg.backends.*`, `pcg.retrieval`) still **import** cleanly under this install; they raise a
`ModuleNotFoundError` naming the missing package only when a function that needs it is called.
The one exception is `pcg.eval.plots_v2`, which needs matplotlib at import time — install `[repro]`.

### 2. Offline reproduction

```bash
python -m pip install -e ".[repro]"
python scripts/repro/reproduce_all.py
```

Adds **matplotlib**. This is all that is needed to validate the 56-cell records, regenerate the
tables and regenerate the figures.

### 3. Development and tests

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

Adds **pytest, pytest-cov, matplotlib, ruff, mypy**. The suite imports only numpy, scipy, pytest
and `pcg` itself.

### 4. Optional — model execution (heavy)

```bash
python -m pip install -e ".[models]"
```

torch, transformers, accelerate, sentence-transformers, datasets, faiss-cpu, langgraph and the rest.
Needed **only to run experiments against real backends**. Not needed for anything documented above.

`torchvision` is deliberately **not** a dependency: no code in `src/`, `scripts/`, `tests/`, `app/`
or `baselines/` imports it. It appears only as an explicit `pip install` line inside two notebooks.
The former `torchvision==0.20.1` pin against `torch>=2.4,<2.6` was a frequent resolver failure on
Apple Silicon and has been removed.

### 5. Optional — privacy, notebooks, frontier APIs, web

```bash
python -m pip install -e ".[privacy]"     # opacus
python -m pip install -e ".[notebooks]"   # ipykernel, jupyterlab, seaborn, colorcet
python -m pip install -e ".[frontier]"    # openai, anthropic  (requires your own keys)
python -m pip install -e ".[web]"         # duckduckgo-search, httpx, beautifulsoup4
python -m pip install -e ".[all]"         # everything
```

## What offline reproduction does and does not do

`python scripts/repro/reproduce_all.py` operates solely on the stored 56-cell artifacts:

- **NO live API calls**
- **NO model inference**
- **NO network calls**
- **NO synthetic fallback** — a missing input fails the step closed rather than substituting a value

(Installing the packages above does use the network; the reproduction step does not.)

## Reproduce everything (one command, offline)

```bash
python scripts/repro/reproduce_all.py
```

### Running the full test suite

Six of eleven test modules run under stdlib `unittest` (42 cases). The remaining **five** need
`pytest` — three import it directly, and two use bare pytest-style functions that `unittest` cannot
collect. They are **BLOCKED offline and are not claimed to pass**:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

Validates inputs first and stops at the first failure. It performs no network access, no model
inference, and never substitutes missing data. Writes `reports/REPRODUCIBILITY_MANIFEST.json`.

## Run the steps individually

```bash
python scripts/validate/validate_56cell.py      # structural validation of the record set
python scripts/validate/provenance_gates.py     # 7 fail-closed contamination/provenance gates
python scripts/repro/regenerate_tables.py       # deterministic tables  -> results/tables/generated/
python scripts/repro/regenerate_figures.py      # figures + block report -> results/figures/
python -m unittest discover -s tests -t .       # 6 modules, 42 cases (stdlib)
```

## Data classes

| Class | Meaning |
|---|---|
| `PROTOCOL` | Configuration and plan; carries no measurement |
| `DIRECT` | Requires native request, prompt, response, model revision, backend, timing, token and hash evidence. **Nothing in this repository currently qualifies.** |
| `DERIVED_FROM_DIRECT` | Complete lineage to verified `DIRECT` records. **Currently unpopulated.** |
| `DERIVED_FROM_UNKNOWN_PROVENANCE_56_CELL` | Computed from the 56-cell records. Everything in `results/` today. |
| `MODELLED` | Analytic, from disclosed formulas with declared parameters. Must be labelled in records, filenames, tables, figures and captions. |
| `UNKNOWN` | May not enter empirical outputs. |

## Layout

```
src/pcg/                 core implementation — certificate, checker, commitments,
                         graph, replay, responsibility, risk, privacy, agents,
                         orchestrator, backends, datasets, eval
scripts/repro/           canonical_metrics.py (the single metric implementation),
                         regenerate_tables.py, regenerate_figures.py, reproduce_all.py
scripts/validate/        validate_56cell.py, provenance_gates.py
scripts/figures/         cosmetic_contract.py (frozen), legacy generators
artifacts/evidence/      validation artifact suite; source_records/ holds the authoritative
                         56-cell file, preserved byte-for-byte
results/tables/generated/ deterministic tables
results/figures/         freshly regenerated figures
reports/                 audit, lineage, manifests, validation, gate output
quarantine/              contaminated inputs, retained as evidence, never read by any pipeline
tests/                   test suite
```

## The single metric implementation

`scripts/repro/canonical_metrics.py` is the only place a reported number is computed. Two
divergent emitters was a defect in the previous pipeline; there is deliberately no second
implementation. Empty denominators return `None`, never `0.0`.

## Cosmetic contract

`scripts/figures/cosmetic_contract.py` freezes the palette, typography, figure sizes, DPI, grid,
spine and save parameters lifted verbatim from the original plotting code. Data loading, paths and
provenance handling may change; these constants may not. Recorded in
`reports/PLOT_COSMETIC_FINGERPRINT.json`.

## Figures: what regenerates and what is blocked

**10 repository-inferred manuscript figures.** Inferred from the LaTeX fragments and the
recovered table/figure map — `FULL_MANUSCRIPT_FIGURE_UNIVERSE_VALIDATION = BLOCKED_MAIN_TEX_ABSENT`,
because the manuscript root is not in this repository and the universe cannot be confirmed against it.

```
MANUSCRIPT_FIGURES_REGENERATED                       1
MANUSCRIPT_FIGURES_WITH_COSMETIC_PARITY              1
MANUSCRIPT_FIGURES_BLOCKED_MISSING_ENGINEERING       2
MANUSCRIPT_FIGURES_BLOCKED_MISSING_SCIENTIFIC_INPUT  7
SUPPLEMENTARY_FIGURES                                6   (not manuscript assets)
```

Outputs are segregated: `results/figures/manuscript/` and `results/figures/supplementary/`, with a
machine-readable guard at `results/figures/FIGURE_CLASSIFICATION.json`. Supplementary figures must
never be wired into the manuscript. Detail: `reports/FINAL_FIGURE_INVENTORY.csv`,
`reports/FINAL_COSMETIC_PARITY.md`.

## Known limitations

1. Native execution provenance is unavailable for all 56-cell outputs (above).
2. `main.tex` is not in this repository; only `latex/experiments.tex` and
   `latex/appendix_exp_details.tex` fragments. Manuscript figure wiring cannot be validated here.
3. The manuscript references `*_v5` figure assets; the repository contains only `v4`.
4. Vendored baselines (ShieldAgent, VeriMAP, AgentRR) are removed from the tree — 394 MB across
   two duplicate copies — and replaced by a pinned-SHA manifest in `configs/`.
5. PRISM/ATLAS, PCN-Rec and CLBC are related work in adjacent domains (model-driven engineering,
   recommendation, covert-channel bounds) and are **not** runnable baselines on this task family.
6. `quarantine/paper_metrics.jsonl` contains the unsupported vertical-slice record
   (`execution_wall_time_sec = 0.0061`); it is retained as evidence and read by nothing.

## Commands verified during this repair

Every command above was executed in a clean checkout. Results: `reports/VALIDATION_REPORT.md`.

## Final readiness

```
ENGINEERING_REPOSITORY_STATUS   = CLEAN_AND_REPRODUCIBLE_OFFLINE
MANUSCRIPT_REPRODUCTION_STATUS  = INCOMPLETE
NATIVE_EXECUTION_PROVENANCE     = NOT_AVAILABLE
EMPIRICAL_AUTHENTICITY          = NOT_ESTABLISHED
SAFE_FOR_EMPIRICAL_MANUSCRIPT_CLAIMS = NO
```

This repository is **clean and reproducible offline**. It is **not manuscript-ready**: nine of ten
inferred manuscript figures cannot be produced, the manuscript root is absent, and the record set
carries no native execution provenance.
