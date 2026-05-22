# PCG-MAS: Proof-Carrying Generation for Multi-Agent Systems

> **PCG-MAS is a proof-carrying runtime for agentic LLM systems.** It converts multi-agent traces into replayable certificates, then accepts a claim only when evidence integrity, replay consistency, execution compliance, and entailment checks pass.

<p align="center">
  <img src="workflow/workflow_v3.png" alt="PCG-MAS workflow" width="980"/>
</p>

<p align="center"><sub>PCG-MAS turns an agentic run into a replayable, externally auditable certificate for every accepted claim.</sub></p>

## Why this repository exists

Modern agentic LLM systems retrieve evidence, call tools, use memory, delegate subtasks, and generate claims. The hard part is not only generating a useful answer; the hard part is making the accepted answer **auditable after the fact**.

PCG-MAS makes accepted claims carry structured support:

```text
prompt / task
  └─► multi-agent generation
        ├─ retriever: finds support
        ├─ prover: drafts claims and support paths
        ├─ verifier: checks intermediate claims
        ├─ debugger: diagnoses weak support
        └─ attacker: stress-tests evidence and execution
              └─► proof-carrying certificate
                    ├─ evidence integrity
                    ├─ replay consistency
                    ├─ execution compliance
                    └─ entailment / claim-support check
                          └─► controller: answer | verify | escalate | refuse
```

The result is a software artifact, not a loose chain-of-thought transcript. Each accepted claim is tied to committed evidence, a replayable support pipeline, an execution contract, and deterministic verification metadata.

## Core verification channels

| Channel | What is checked | Typical failure caught |
|:---|:---|:---|
| Evidence integrity | Hash commitments and canonical support objects | Tampered evidence, stale spans, changed table cells, broken provenance |
| Replay consistency | Reconstructing support from declared retrieval/tool steps | Environment drift, parser drift, retrieval drift, tool-output mismatch |
| Execution compliance | Tool, memory, schema, policy, and delegation contracts | Unsafe tool calls, undeclared memory access, invalid schemas, policy violations |
| Entailment / checking | Whether replayed support justifies the emitted claim or action | Unsupported claims, contradiction, partial-evidence hallucination |

## Repository layout

```text
src/pcg/                  core certificate, checker, risk, replay, privacy, and responsibility logic
scripts/runs/             PCG-MAS benchmark and interactive runners
scripts/figures/          paper and README figure generation
scripts/tables/           metric collection and LaTeX/CSV table generation
scripts/baselines/        independent adapter folders for comparison methods
app/                      Streamlit demo for trace and certificate inspection
artifacts/v4_preview/     public preview figures for README display
workflow/                 README workflow artwork
configs/                  benchmark matrix and experiment configs
notebooks/                optional Colab / Databricks execution helpers
```

## Quick start

```bash
python3.12 -m venv multi-agents
source multi-agents/bin/activate
python -m pip install --upgrade pip
pip install -e .
pip install -r requirements.txt
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
```

Run the interactive PCG-MAS workflow:

```bash
bash scripts/runs/run_pcgmas_interactive.sh
```

Typical small run:

```text
cells: fever:phi-3.5-mini,hotpotqa:qwen2.5-7b
experiments: r1-r5
seeds: 0
examples: 3
backend: hf_local, hf_inference, openai, or mock depending on runtime configuration
```

Run figure and table generation after metrics are available:

```bash
python scripts/figures/make_paper_figures.py \
  --rows results/tables/csv/paper_metrics.jsonl \
  --outdir results/figures \
  --allow-partial

python scripts/tables/make_paper_tables.py \
  --rows results/tables/csv/paper_metrics.jsonl \
  --outdir results/tables/tex \
  --allow-partial
```

## Preview results

<p align="center">
  <img src="artifacts/v4_preview/figures/intro_hero_v4.png" alt="PCG-MAS headline result summary" width="980"/>
</p>

<p align="center"><sub>Headline PCG-MAS safety, utility, audit coverage, responsibility, and cost summary.</sub></p>

A more detailed appendix-style comparison including additional adapter methods is available here: [open the expanded comparison figure](artifacts/v4_preview/figures/appendix_hero_v4.png).

<p align="center">
  <img src="artifacts/v4_preview/figures/r5_overhead_v4.png" alt="PCG-MAS token and runtime overhead" width="980"/>
</p>

<p align="center"><sub>Detailed token and runtime overhead view for certificate construction, checking, replay, and reporting.</sub></p>

## Main experiment families

| Family | Purpose | Primary artifact |
|:---|:---|:---|
| R1 | Certificate checkability and audit-channel decomposition | `r1_audit_decomposition_v4` |
| R2 | Redundancy under adversarial stress | `r2_redundancy_surface_v4` |
| R3 | Interventional responsibility and diagnosis | `r3_responsibility_v4` |
| R4 | Risk-control frontier | `r4_control_frontier_v4` |
| R5 | Runtime, token, and scaling overhead | `r5_overhead_v4` |

Optional public preview figures are stored under:

```text
artifacts/v4_preview/figures/
```

Generated run outputs are written under:

```text
results/figures/
results/tables/csv/
results/tables/tex/
```

The public Git tree keeps `results/tables/tex/` as an empty tracked directory via `.gitkeep`; generated `.tex` files are intentionally excluded from the release tree.

## Baseline adapter policy

Comparison adapters are intentionally isolated by folder:

```text
scripts/baselines/shieldagent/
scripts/baselines/agentrr/
scripts/baselines/verimap/
scripts/baselines/prism/
scripts/baselines/pcn_rec/
scripts/baselines/clbc/
```

Each adapter owns its setup, run, overlay, merge, and verification scripts. Adapter overlays update only method-specific fields in `results/tables/csv/paper_metrics.jsonl`; unrelated method fields are preserved through sidecar overlay rows. This keeps each method independently runnable while allowing `appendix_hero_v4` to summarize all available methods.

## Reproducibility modes

| Mode | Purpose | Private key required |
|:---|:---|:---|
| Artifact rebuild | Recreate figures/tables from existing metric files | No |
| Preflight / mock | Validate dataset loading, certificate construction, replay, and output paths | No |
| Local model run | Execute local Hugging Face / Transformers backends | Maybe, model-dependent |
| Hosted model run | Execute hosted inference or API-backed adapters | Yes, runtime only |

Runtime credentials are read from environment variables or secure prompts. Tokens, local `.env` files, virtual environments, and model caches must not be committed.

## Dynamic repository summary

_Generated on: **2026-05-23 04:26:21**_  
_Total release-style files counted: **196**_

<table>
<tr>
<td valign="top" width="50%">

### Top-level distribution

| Path | Files | Share |
|:---|---:|---:|
| `scripts` | 73 | 37.2% |
| `src` | 48 | 24.5% |
| `artifacts` | 29 | 14.8% |
| `app` | 18 | 9.2% |
| `configs` | 15 | 7.7% |
| `.` | 6 | 3.1% |
| `.github` | 2 | 1.0% |
| `notebooks` | 2 | 1.0% |
| `docs` | 1 | 0.5% |
| `results` | 1 | 0.5% |
| `workflow` | 1 | 0.5% |

</td>
<td valign="top" width="50%">

### File-type distribution

| Extension | Files | Share |
|:---|---:|---:|
| `.py` | 119 | 60.7% |
| `.yaml` | 15 | 7.7% |
| `.pdf` | 14 | 7.1% |
| `.png` | 13 | 6.6% |
| `.sh` | 12 | 6.1% |
| `.txt` | 8 | 4.1% |
| `(no ext)` | 4 | 2.0% |
| `.json` | 3 | 1.5% |
| `.ipynb` | 2 | 1.0% |
| `.md` | 2 | 1.0% |
| `.yml` | 2 | 1.0% |
| `.example` | 1 | 0.5% |
| `.toml` | 1 | 0.5% |

</td>
</tr>
</table>

## Release-style tree

```text
.
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── deploy_space.yml
├── app/
│   ├── components/
│   │   ├── __init__.py
│   │   ├── agent_trace.py
│   │   ├── byok_modal.py
│   │   ├── certificate_card.py
│   │   ├── llm_client.py
│   │   └── theme.py
│   ├── demo_data/
│   │   └── results_fixtures.json
│   ├── pages/
│   │   ├── 1_Live_Run.py
│   │   ├── 2_Certificate_Inspector.py
│   │   ├── 3_Side_by_Side.py
│   │   ├── 4_Results_Browser.py
│   │   ├── 5_Auditor_Demo.py
│   │   ├── __init__.py
│   │   └── _live_run_helpers.py
│   ├── app.py
│   ├── Dockerfile
│   ├── README.md
│   └── requirements.txt
├── artifacts/
│   ├── v4_preview/
│   │   ├── figures/
│   │   │   ├── ablations.pdf
│   │   │   ├── ablations.png
│   │   │   ├── appendix_hero_v4.pdf
│   │   │   ├── appendix_hero_v4.png
│   │   │   ├── harm_clean_adv_split.pdf
│   │   │   ├── intro_hero_v4.pdf
│   │   │   ├── intro_hero_v4.png
│   │   │   ├── pcg-mas_r1_to_r4.pdf
│   │   │   ├── r1_audit_decomposition_v4.pdf
│   │   │   ├── r1_audit_decomposition_v4.png
│   │   │   ├── r1_five_channel_audit.pdf
│   │   │   ├── r1_five_channel_audit.png
│   │   │   ├── r2_redundancy_surface_v4.pdf
│   │   │   ├── r2_redundancy_surface_v4.png
│   │   │   ├── r3_open_mixed.pdf
│   │   │   ├── r3_open_mixed.png
│   │   │   ├── r3_responsibility_v4.pdf
│   │   │   ├── r3_responsibility_v4.png
│   │   │   ├── r4_control_frontier_v4.pdf
│   │   │   ├── r4_control_frontier_v4.png
│   │   │   ├── r4_privacy_frontier.pdf
│   │   │   ├── r4_privacy_frontier.png
│   │   │   ├── r5_overhead_v4.pdf
│   │   │   ├── r5_overhead_v4.png
│   │   │   ├── r5_scaling.pdf
│   │   │   └── r5_scaling.png
│   │   └── manifest_hash.txt
│   ├── coverage_plan.json
│   └── dataset_schema_tatqa_weblinx.txt
├── configs/
│   ├── frontier_merge.yaml
│   ├── local_40_cells.yaml
│   ├── preflight_2_cells.yaml
│   ├── preflight_40_cells.yaml
│   ├── r1_fever.yaml
│   ├── r1_hotpotqa.yaml
│   ├── r1_pubmedqa.yaml
│   ├── r1_tatqa.yaml
│   ├── r1_weblinx.yaml
│   ├── r2_redundancy.yaml
│   ├── r3_responsibility.yaml
│   ├── r4_risk.yaml
│   ├── r5_overhead.yaml
│   ├── r6_cross_domain.yaml
│   └── v4_matrix.yaml
├── docs/
│   └── manifest.json
├── notebooks/
│   ├── pcg_v4_colab_16cells.ipynb
│   └── run_large_llms.ipynb
├── results/
│   └── tables/
│       └── tex/
│           └── .gitkeep
├── scripts/
│   ├── analysis/
│   │   ├── audit_envelope.py
│   │   └── pick_top_k.py
│   ├── baselines/
│   │   ├── agentrr/
│   │   │   ├── merge_agentrr_into_pcgmas.sh
│   │   │   ├── overlay_agentrr_into_pcgmas.py
│   │   │   ├── run_agentrr_interactive.sh
│   │   │   ├── run_agentrr_r1_r5.py
│   │   │   ├── setup_agentrr.sh
│   │   │   └── verify_agentrr_adapter.py
│   │   ├── shieldagent/
│   │   │   ├── export_shieldagent_wide_metrics.py
│   │   │   ├── merge_shieldagent_into_pcgmas.sh
│   │   │   ├── merge_shieldagent_r1_r5.py
│   │   │   ├── overlay_shieldagent_into_pcgmas.py
│   │   │   ├── requirements.shield-agent.entrypoints.txt
│   │   │   ├── requirements.shield-agent.macos.txt
│   │   │   ├── requirements.shield-agent.runtime.txt
│   │   │   ├── run_shieldagent_interactive.sh
│   │   │   ├── run_shieldagent_r1_r5.py
│   │   │   ├── run_shieldagent_r1_r5_comparative.py
│   │   │   └── setup_shieldagent.sh
│   │   └── verimap/
│   │       ├── merge_verimap_into_pcgmas.sh
│   │       ├── overlay_verimap_into_pcgmas.py
│   │       ├── requirements.verimap.runtime.txt
│   │       ├── run_verimap_interactive.sh
│   │       ├── run_verimap_r1_r5.py
│   │       ├── setup_verimap.sh
│   │       └── verify_verimap_adapter.py
│   ├── common/
│   │   ├── __init__.py
│   │   ├── benchmark_specs.py
│   │   ├── experiment_io.py
│   │   ├── paper_metric_validation.py
│   │   ├── paper_metrics.py
│   │   ├── paths.py
│   │   ├── run_manifest.py
│   │   └── schema.py
│   ├── experiments/
│   │   ├── __init__.py
│   │   ├── run_ablations.py
│   │   ├── run_r1_checkability.py
│   │   ├── run_r2_redundancy.py
│   │   ├── run_r3_responsibility.py
│   │   ├── run_r4_risk_privacy.py
│   │   └── run_r5_overhead.py
│   ├── figures/
│   │   ├── __init__.py
│   │   ├── build_all_figures.py
│   │   ├── legacy_r1_r5_plots.py
│   │   ├── make_paper_figures.py
│   │   ├── make_r3_open_mixed.py
│   │   ├── make_r4_privacy_frontier.py
│   │   └── make_r5_scaling.py
│   ├── maintain/
│   │   ├── __init__.py
│   │   ├── audit_forbidden_terms.py
│   │   ├── audit_repo_layout.py
│   │   ├── audit_secrets.py
│   │   └── build_backends_manifest.py
│   ├── notebooks/
│   │   ├── __init__.py
│   │   └── merge_frontier_runs.py
│   ├── runs/
│   │   ├── __init__.py
│   │   ├── run_local_40_cells.py
│   │   ├── run_matrix.py
│   │   ├── run_pcgmas_benchmark_suite.py
│   │   ├── run_pcgmas_interactive.sh
│   │   ├── run_preflight.py
│   │   └── run_preflight_40_cells.py
│   ├── tables/
│   │   ├── __init__.py
│   │   ├── build_all_tables.py
│   │   ├── collect_paper_metrics.py
│   │   ├── make_paper_tables.py
│   │   ├── repair_paper_metrics_metadata.py
│   │   └── validate_paper_metrics.py
│   ├── __init__.py
│   ├── build_paper_artifacts.py
│   ├── build_readme.py
│   ├── deploy_to_anonymous_space.sh
│   └── run_local_llms.sh
├── src/
│   └── pcg/
│       ├── agents/
│       │   ├── __init__.py
│       │   ├── attacker.py
│       │   ├── debugger.py
│       │   ├── prover.py
│       │   └── verifier.py
│       ├── backends/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── hf_inference.py
│       │   ├── hf_local.py
│       │   └── mock.py
│       ├── datasets/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── fever.py
│       │   ├── hotpotqa.py
│       │   ├── pubmedqa.py
│       │   ├── synthetic.py
│       │   ├── tatqa.py
│       │   ├── toolbench.py
│       │   ├── twowiki.py
│       │   └── weblinx.py
│       ├── eval/
│       │   ├── __init__.py
│       │   ├── audit.py
│       │   ├── bootstrap.py
│       │   ├── coverage.py
│       │   ├── intro_hero_v4.py
│       │   ├── latency.py
│       │   ├── meter.py
│       │   ├── metrics.py
│       │   ├── plots_v2.py
│       │   ├── rho.py
│       │   ├── stats.py
│       │   └── tightness.py
│       ├── orchestrator/
│       │   ├── __init__.py
│       │   ├── langgraph_flow.py
│       │   └── replay_handlers.py
│       ├── utils/
│       │   ├── __init__.py
│       │   └── hf_auth.py
│       ├── __init__.py
│       ├── certificate.py
│       ├── checker.py
│       ├── cli.py
│       ├── commitments.py
│       ├── graph.py
│       ├── independence.py
│       ├── privacy.py
│       ├── responsibility.py
│       ├── retrieval.py
│       └── risk.py
├── workflow/
│   └── workflow_v3.png
├── .env.example
├── .gitignore
├── Makefile
├── pyproject.toml
├── README.md
└── requirements.txt
```

## Local-only files intentionally excluded from Git

```text
.env.local and other private environment files
multi-agents/ and .venvs/
artifacts/v4_preview/tables/
user_v1.txt ... user_v4.txt
README_creator*.txt
workflow/pcg-mas-workflow.png
workflow/workflow_v2.png
latex/
tests/
results/figures_staged_selected/
results/tables/tex_staged_selected/
generated files under results/tables/tex/ except .gitkeep
```

## Maintainer checks before publishing

```bash
python scripts/maintain/audit_secrets.py
python scripts/maintain/audit_forbidden_terms.py
python scripts/maintain/audit_repo_layout.py

python -m py_compile scripts/figures/make_paper_figures.py
python -m py_compile scripts/tables/make_paper_tables.py
python -m py_compile scripts/baselines/agentrr/*.py
python -m py_compile scripts/baselines/verimap/*.py
python -m py_compile scripts/baselines/shieldagent/*.py

for f in scripts/baselines/agentrr/*.sh scripts/baselines/verimap/*.sh scripts/baselines/shieldagent/*.sh scripts/runs/*.sh
do
  bash -n "$f"
done
```

## License and citation

License and citation metadata can be added once the public release target is finalized.
