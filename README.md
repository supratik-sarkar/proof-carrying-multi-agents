# Proof-Carrying Generation — Externally Verifiable Multi-Agent Systems (Colab)

Reproducible Colab pipeline implementing Algorithm PCG-MAS (Proof-Carrying Generation for
Multi-Agent Systems) with grounding certificates, replayable pipelines, independent redundancy
bounds, interventional responsibility, and cost-aware risk control.

* 4 datasets × 2 backbone LLM configs (HotpotQA, 2WikiMultihopQA, ToolBench, WebLINX)
* 4 experimental pillars (R1–R4): certificates & checkability, independent redundancy, interventional responsibility, risk control + privacy
* Ablations over k (redundancy), δ/κ (overlap tolerance), ε (DP budget), and λ (risk weight)
* Fully runnable on Colab A100; private HF tokens required for gated backbone checkpoints.

---

## Project map & theory cheat-sheet

*Generated: 2025-09-23 11:17*

This section gives a quick map from code to the theoretical objects used in the paper (inline LaTeX uses $...$). It also includes a depth-limited tree of the repo.

### Repository overview
```
proof-carrying-multi-agents/
  ├── configs/
  │   └── default.yaml
  │   └── hotpotqa.yaml
  │   └── 2wiki.yaml
  │   └── toolbench.yaml
  │   └── weblinx.yaml
  ├── notebooks/
  │   └── pcg_mas_experiments.ipynb
  ├── results/
  │   └── r1_tamper_roc.png
  │   └── r1_reliability_diagram.png
  │   └── r1_risk_control_surface.png
  │   └── r1_false_accept_vs_coverage.png
  │   └── r2_error_vs_k.png
  │   └── r2_overlap_vs_k.png
  │   └── r2_error_3d.png
  │   └── r3_responsibility_heatmap.png
  │   └── r3_bottleneck_accuracy.png
  │   └── r4_pareto_cloud.png
  │   └── r4_pareto_fronts.png
  │   └── r4_privacy_vs_utility.png
  │   └── r4_leakage_proxy.png
  ├── src/
  │   └── agents/
  │       └── prover.py
  │       └── verifier.py
  │       └── attacker.py
  │       └── debugger.py
  │   └── certificates/
  │       └── grounding.py
  │       └── checker.py
  │       └── minimizer.py
  │   └── redundancy/
  │       └── independence.py
  │       └── paths.py
  │       └── bounds.py
  │   └── responsibility/
  │       └── interventions.py
  │       └── masking.py
  │   └── risk_control/
  │       └── decision.py
  │       └── privacy.py
  │   └── io/
  │       └── datamodules.py
  │       └── adapters.py
  │   └── utils/
  │       └── logging.py
  │       └── seed.py
  ├── .gitignore
  ├── CITATION.cff
  ├── LICENSE
  ├── README.md
  ├── requirements.txt
```

### Theory glossary (files → quantities)

| File | Quantity / Symbol | Short description |
| --- | --- | --- |
| `src/certificates/grounding.py` | $\mathcal{Z} = (c, S, \Pi, p, \text{meta})$ | Grounding certificate binding a claim to its committed evidence set and replayable pipeline. |
| `src/certificates/checker.py` | $\text{Check}(\mathcal{Z};\, G_t) \in \{0,1\}$ | Deterministic checkable groundedness: hash integrity, pipeline replay, and entailment check. |
| `src/certificates/minimizer.py` | $S^* \subseteq S_0$ | Certificate minimization via greedy backward elimination preserving acceptance under Eq. (4). |
| `src/redundancy/independence.py` | $(\delta, \kappa)\text{-independent paths}$ | Provenance-disjoint, low-overlap evidence paths enforcing Definition 3.5. |
| `src/redundancy/bounds.py` | $\alpha(k) \leq \rho^{k-1}\varepsilon^k$ | Conservative false-accept envelope under $k$-redundant consensus (Theorem 4.1, Corollary 4.1). |
| `src/responsibility/interventions.py` | $\text{Resp}(e;\,\mathcal{Z}, G_t)$ | Interventional responsibility via mask-and-replay sensitivity (Definition 3.6, Eq. (6)). |
| `src/responsibility/masking.py` | $G_t^{\setminus e}$ | Replay graph with component $e$ masked; supports do-style causal interventions. |
| `src/risk_control/decision.py` | $C(b, a) = C_{\text{lat}} + C_{\text{tok}} + C_{\text{tool}} + \lambda\,\mathbb{E}[L_{\text{harm}}]$ | Per-step cost model over \{answer, verify, escalate, refuse\} actions (Eq. (13)). |
| `src/risk_control/privacy.py` | $\tilde{\psi} = \psi + \xi,\; \xi \sim \mathcal{N}(0, \eta^2 I)$ | Differential privacy noise applied to inter-agent shared certificate statistics. |

### Key modules (implementation map)

| File | API | What it does |
| --- | --- | --- |
| `src/agents/prover.py` | `Prover` | Executes tasks, emits grounding certificates with hash-linked evidence and retrieval metadata. |
| `src/agents/verifier.py` | `Verifier` | Replay checks (hash, schema, tool/trace), risk scoring, and isotonic/Platt calibration. |
| `src/agents/attacker.py` | `Attacker` | Controlled evidence/log tampering harness for adversarial tamper-detection evaluation. |
| `src/agents/debugger.py` | `Debugger` | do-interventions (R3) and risk-control policy execution (R4) with JSONL logging. |
| `src/certificates/checker.py` | `Check(Z, G)` | Deterministic certificate check: hash integrity → pipeline replay → entailment. |
| `src/redundancy/paths.py` | `IndependentPaths` | Constructs $k$ $(\delta,\kappa)$-independent evidence branches (dense + BM25 + rewrite variants). |
| `src/io/datamodules.py` | `load_dataset(...)` | Unified loader for HotpotQA / 2WikiMultihopQA / ToolBench / WebLINX with normalization. |
| `src/utils/seed.py` | `seed_everything` | Reproducibility helpers for deterministic replay and seeded sampling. |
| `notebooks/pcg_mas_experiments.ipynb` | — | End-to-end Colab notebook covering R1–R4: certificates, redundancy, responsibility, and risk control with all result plots and tables. |

---

## File-type distribution

*Generated: 2025-09-23 11:17*

| Extension | Count | Share |
| --- | --- | --- |
| `.py` | 22 | 62.9% |
| `.yaml` | 5 | 14.3% |
| `.ipynb` | 1 | 2.9% |
| `.md` | 2 | 5.7% |
| `.txt` | 2 | 5.7% |
| `.cff` | 1 | 2.9% |
| `(no ext)` | 2 | 5.7% |

**Total files scanned:** 35

## Experimental Results (R1–R4)

The plots below correspond to the four experimental pillars of the paper. All figures are
generated by `notebooks/pcg_mas_experiments.ipynb` and saved under `results/`.

### R1 — Certificates & Checkability

Calibration aligns predicted risk with empirical error; selective answering reduces false
accepts; evidence-linked certificates improve faithfulness; replayable checks detect tampering
without access to model internals.

[![R1: Tamper Detection ROC](https://github.com/supratik-sarkar/proof-carrying-multi-agents/raw/main/results/r1_tamper_roc.png)](https://github.com/supratik-sarkar/proof-carrying-multi-agents/blob/main/results/r1_tamper_roc.png) [![R1: Reliability Diagram](https://github.com/supratik-sarkar/proof-carrying-multi-agents/raw/main/results/r1_reliability_diagram.png)](https://github.com/supratik-sarkar/proof-carrying-multi-agents/blob/main/results/r1_reliability_diagram.png)

[![R1: Risk Control Surface (FAR over coverage × threshold)](https://github.com/supratik-sarkar/proof-carrying-multi-agents/raw/main/results/r1_risk_control_surface.png)](https://github.com/supratik-sarkar/proof-carrying-multi-agents/blob/main/results/r1_risk_control_surface.png) [![R1: False-Accept Rate vs Coverage](https://github.com/supratik-sarkar/proof-carrying-multi-agents/raw/main/results/r1_false_accept_vs_coverage.png)](https://github.com/supratik-sarkar/proof-carrying-multi-agents/blob/main/results/r1_false_accept_vs_coverage.png)

### R2 — Independent Redundancy

Diverse branches (BM25 vs dense, rewrites, leave-top-1-out) outperform correlated redundancy
at matched cost; error decays with $k$ per the conservative envelope $\alpha(k) \leq \rho^{k-1}\varepsilon^k$;
law fits are stable across datasets and backbones.

[![R2: Error vs Redundancy k](https://github.com/supratik-sarkar/proof-carrying-multi-agents/raw/main/results/r2_error_vs_k.png)](https://github.com/supratik-sarkar/proof-carrying-multi-agents/blob/main/results/r2_error_vs_k.png) [![R2: Evidence Overlap vs k](https://github.com/supratik-sarkar/proof-carrying-multi-agents/raw/main/results/r2_overlap_vs_k.png)](https://github.com/supratik-sarkar/proof-carrying-multi-agents/blob/main/results/r2_overlap_vs_k.png) [![R2: 3D Error vs (k, evidence-overlap)](https://github.com/supratik-sarkar/proof-carrying-multi-agents/raw/main/results/r2_error_3d.png)](https://github.com/supratik-sarkar/proof-carrying-multi-agents/blob/main/results/r2_error_3d.png)

### R3 — Interventional Responsibility

do-interventions localize failures to the correct module (retrieval vs tool vs reasoning vs
coordination); controlled broken-component tests validate identification accuracy across
datasets and backbones.

[![R3: Responsibility Heatmap](https://github.com/supratik-sarkar/proof-carrying-multi-agents/raw/main/results/r3_responsibility_heatmap.png)](https://github.com/supratik-sarkar/proof-carrying-multi-agents/blob/main/results/r3_responsibility_heatmap.png) [![R3: Bottleneck Identification Accuracy (Top-1)](https://github.com/supratik-sarkar/proof-carrying-multi-agents/raw/main/results/r3_bottleneck_accuracy.png)](https://github.com/supratik-sarkar/proof-carrying-multi-agents/blob/main/results/r3_bottleneck_accuracy.png)

### R4 — Risk Control + Privacy Budgets

Escalation/refusal introduces non-linear Pareto frontiers; DP shifts attainable trade-offs
while preserving qualitative shapes across backbones; leakage proxy decreases with tighter $\varepsilon$.

[![R4: Pareto Cloud (harm, utility) across ε, policy, ablations](https://github.com/supratik-sarkar/proof-carrying-multi-agents/raw/main/results/r4_pareto_cloud.png)](https://github.com/supratik-sarkar/proof-carrying-multi-agents/blob/main/results/r4_pareto_cloud.png) [![R4: Pareto Fronts (non-dominated points)](https://github.com/supratik-sarkar/proof-carrying-multi-agents/raw/main/results/r4_pareto_fronts.png)](https://github.com/supratik-sarkar/proof-carrying-multi-agents/blob/main/results/r4_pareto_fronts.png)

[![R4: Privacy vs Utility (ε sweep)](https://github.com/supratik-sarkar/proof-carrying-multi-agents/raw/main/results/r4_privacy_vs_utility.png)](https://github.com/supratik-sarkar/proof-carrying-multi-agents/blob/main/results/r4_privacy_vs_utility.png) [![R4: Leakage Proxy vs ε (lower is better)](https://github.com/supratik-sarkar/proof-carrying-multi-agents/raw/main/results/r4_leakage_proxy.png)](https://github.com/supratik-sarkar/proof-carrying-multi-agents/blob/main/results/r4_leakage_proxy.png)

## Submission snapshot

The exact code and assets used at submission time are archived in the GitHub Release
**"Submission (Sep-2025)"**. The default branch may contain replication
notebooks and small fixes for future runs.

> To audit the submission state, download the release tarball or see `RELEASE_NOTES_2025-09.md`.

## Why numbers may differ

Small metric deltas vs. the paper tables are expected due to:

* **Stochastic inference** (sampling, CUDA nondeterminism) even with fixed seeds and declared replay randomness.
* **Library/driver drift** across machines — A100 vs Colab T4/L4 (PyTorch / CUDA / cuDNN / sentence-transformers / BM25 kernels).
* **Model snapshot drift** — HF model revisions or API-served models updated upstream (particularly for Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct gated checkpoints).
* **Dataset loaders** — shuffle order, minor preprocessing differences, or split availability via HuggingFace Hub.

**Mitigations:** version pinning in `requirements.txt`, fixed seeds via `src/utils/seed.py`, frozen model revisions recorded in `configs/`, and a drift audit notebook:
```
jupyter nbconvert --to notebook --execute notebooks/pcg_mas_experiments.ipynb
```

For full deterministic replay of a specific certificate:
```
python -m src.certificates.checker --config configs/hotpotqa.yaml --seed 42
```

Residual ±(1–3)pp variations are typical for this stack.

## Diagnostics

Compare current runs against paper targets (Tables 1–5) using the end-to-end notebook:
```
jupyter nbconvert --to notebook --execute notebooks/pcg_mas_experiments.ipynb
```

The notebook prints averaged tables and full per-dataset/backbone breakdowns for R1–R4.
To run a specific experimental pillar only, set the `run_r1` / `run_r2` / `run_r3` / `run_r4`
flags in `configs/default.yaml` before executing.

## Compute environments

* **Primary (paper):** Databricks (NVIDIA A100) with private HF checkpoints accessed via `HF_TOKEN`. Llama-3.1-8B-Instruct requires HuggingFace model-access acceptance; Qwen2.5-7B-Instruct typically does not.
* **Replication:** For reproducibility, we provide a Colab/GCP notebook (`notebooks/pcg_mas_experiments.ipynb`) with 4-bit quantization fallback (`bitsandbytes`) for A100-constrained environments.  
  Minor numeric differences across plots/tables may occur across hardware and library versions.
  We include environment pins in `requirements.txt`, fixed seeds, and per-config frozen model revision IDs.
