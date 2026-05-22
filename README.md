# PCG-MAS: Proof-Carrying Generation for Multi-Agent Systems

    > **PCG-MAS is a proof-carrying runtime for agentic LLM systems.** It converts multi-agent traces into replayable certificates, then accepts a claim only when evidence integrity, replay consistency, execution compliance, and entailment checks pass.

    <p align="center"><strong>Generation on May-5-2025 09:54</strong></p>

    <p align="center">
  <img src="workflow/workflow_v3.png" alt="PCG-MAS workflow" width="980"/>
<br/><sub>PCG-MAS turns an agentic run into committed evidence, replayable support, executable checks, and calibrated control decisions.</sub>
</p>

    ## Why PCG-MAS exists

    Modern agentic systems retrieve evidence, call tools, use memory, delegate subtasks, and compose outputs across multiple execution paths. A fluent answer is not enough for high-stakes deployment. PCG-MAS turns each accepted claim into a software artifact that can be checked after generation.

    The core object is a certificate attached to an accepted claim:

    ```text
    claim + evidence + pipeline + execution contract + entailment check
      -> externally replayable acceptance predicate
      -> answer | verify | escalate | refuse
    ```

    PCG-MAS is designed for model-risk, audit, and production settings where accepted outputs must remain inspectable after the original agent run has finished.

    ## System flow

    ```text
    Input task
      -> multi-agent generation
          -> retrieval / tools / memory / delegation
          -> claim proposal
              -> certificate construction
                  -> evidence integrity checks
                  -> replay consistency checks
                  -> execution-contract checks
                  -> deterministic entailment checks
                      -> calibrated controller
                          -> answer | verify | escalate | refuse
    ```

    ## Repository layout

    ```text
    app/                         Streamlit demo and certificate inspector
    artifacts/v4_preview/figures/ Release-preview figures used by this README
    configs/                     Benchmark and matrix run configs
    scripts/runs/                PCG-MAS benchmark and interactive runners
    scripts/baselines/           Independent SOTA adapter runners and merge scripts
    scripts/figures/             Figure builders
    scripts/tables/              Metric collection and LaTeX table builders
    src/pcg/                     Core certificate, checker, risk, retrieval, and orchestration code
    workflow/                    README workflow artwork
    results/tables/tex/          Empty tracked table-output directory
    ```

    ## Fresh setup

    The README intentionally keeps setup compact. The full interactive runner handles cell selection, seeds, backend choice, and cleanup policy.

    ```bash
    git clone https://github.com/anonymous-submission/proof-carrying-multi-agents.git
    cd proof-carrying-multi-agents

    python3.12 -m venv multi-agents
    source multi-agents/bin/activate

    python -m pip install --upgrade pip setuptools wheel
    python -m pip install -r requirements.txt

    export PYTHONPATH="$PWD:${PYTHONPATH:-}"
    bash scripts/runs/run_pcgmas_interactive.sh
    ```

    The interactive runner asks for benchmark cells, experiment groups, seeds, example count, backend mode, and whether stale generated artifacts should be cleaned before the run.

    ## PCG-MAS workflow run

    ```bash
    cd proof-carrying-multi-agents
    source multi-agents/bin/activate
    export PYTHONPATH="$PWD:${PYTHONPATH:-}"

    bash scripts/runs/run_pcgmas_interactive.sh
    ```

    Typical cell strings:

    ```text
    fever:phi-3.5-mini,hotpotqa:qwen2.5-7b
    pubmedqa:llama-3.1-8b,tatqa:gemma-2-9b-it
    all
    ```

    Experiment choices:

    ```text
    r1, r2, r3, r4, r5, r1-r5, all
    ```

    ## Baseline and SOTA adapter runs

    PCG-MAS produces exported baseline inputs under `results/tables/csv/baseline_inputs/`. Each adapter below runs independently against those exported records and then overlays its own metrics into the PCG-MAS figure/table pipeline.

    ### ShieldAgent / AutoPolicy adapter

    External references: [ShieldAgent project](https://shieldagent-aiguard.github.io/) · [AutoPolicy implementation](https://github.com/BillChan226/ShieldAgent)

    ```bash
    # 1. Run PCG-MAS first so baseline input files exist.
    bash scripts/runs/run_pcgmas_interactive.sh

    # 2. Clone the policy-extraction source outside this repository.
    mkdir -p /path/to/external/repos
    git clone https://github.com/BillChan226/ShieldAgent.git /path/to/external/repos/ShieldAgent_AutoPolicy

    # 3. Set up the isolated ShieldAgent runtime.
    bash scripts/baselines/shieldagent/setup_shieldagent.sh

    # 4. Run ShieldAgent on the same cells.
    bash scripts/baselines/shieldagent/run_shieldagent_interactive.sh

    # 5. Merge ShieldAgent metrics into the PCG-MAS artifacts.
    bash scripts/baselines/shieldagent/merge_shieldagent_into_pcgmas.sh
    ```

    ### AgentRR adapter

    External reference: [MobiAgent / AgentRR](https://github.com/IPADS-SAI/MobiAgent)

    ```bash
    # 1. Run PCG-MAS first so baseline input files exist.
    bash scripts/runs/run_pcgmas_interactive.sh

    # 2. Clone the source reference outside this repository.
    mkdir -p /path/to/external/repos
    git clone https://github.com/IPADS-SAI/MobiAgent.git /path/to/external/repos/MobiAgent

    # 3. Set up the isolated AgentRR runtime.
    bash scripts/baselines/agentrr/setup_agentrr.sh

    # 4. Run AgentRR-Adapter on the same cells.
    bash scripts/baselines/agentrr/run_agentrr_interactive.sh

    # 5. Merge and verify AgentRR overlay fields.
    bash scripts/baselines/agentrr/merge_agentrr_into_pcgmas.sh
    python scripts/baselines/agentrr/verify_agentrr_adapter.py
    ```

    ### VeriMAP adapter

    External reference: [VeriMAP](https://github.com/megagonlabs/veriMAP)

    ```bash
    # 1. Run PCG-MAS first so baseline input files exist.
    bash scripts/runs/run_pcgmas_interactive.sh

    # 2. Clone the source reference outside this repository.
    mkdir -p /path/to/external/repos
    git clone https://github.com/megagonlabs/veriMAP.git /path/to/external/repos/VeriMAP

    # 3. Set up the isolated VeriMAP runtime.
    bash scripts/baselines/verimap/setup_verimap.sh

    # 4. Run VeriMAP-Adapter on the same cells.
    bash scripts/baselines/verimap/run_verimap_interactive.sh

    # 5. Merge and verify VeriMAP overlay fields.
    bash scripts/baselines/verimap/merge_verimap_into_pcgmas.sh
    python scripts/baselines/verimap/verify_verimap_adapter.py
    ```

    ### Appendix-only adapters under active extension

    The repository keeps independent folders for additional appendix-only adapters:

    ```text
    scripts/baselines/prism/
    scripts/baselines/pcn_rec/
    scripts/baselines/clbc/
    ```

    Each adapter should follow the same contract: run against PCG-MAS exported baseline records, write method-specific overlay fields, update `appendix_hero_v4`, and avoid deleting unrelated figures or prior SOTA overlays.

    ## Results preview

    <p align="center">
  <img src="artifacts/v4_preview/figures/intro_hero_v4.png" alt="PCG-MAS headline result summary" width="980"/>
<br/><sub>Headline safety, utility, audit coverage, responsibility, and cost summary for the primary PCG-MAS view.</sub>
</p>

    A wider appendix view with all currently integrated SOTA overlays is available here: [`artifacts/v4_preview/figures/appendix_hero_v4.png`](artifacts/v4_preview/figures/appendix_hero_v4.png).

    <p align="center">
  <img src="artifacts/v4_preview/figures/r5_overhead_v4.png" alt="PCG-MAS token and runtime overhead" width="980"/>
<br/><sub>Detailed overhead view for certificate construction, checking, replay, and reporting.</sub>
</p>

    ## Agentic deployment architecture

    PCG-MAS is built around an adapter-friendly agentic runtime rather than a single monolithic model call. The architecture supports local Hugging Face backends, hosted inference APIs, OpenAI-compatible and Anthropic-compatible endpoints, isolated SOTA runtime environments, and replayable result overlays. This makes the repository suitable for both local engineering and cloud-scale experiments.

    Large-model deployment can be organized across several tiers:

    - **Local engineering tier:** MacBook Pro M4 Pro for development, small-cell verification, mock/preflight runs, figure generation, and table generation.
    - **Single-GPU research tier:** Google Colab A100 sessions with Drive-backed model/result cache, including large persistent storage quotas where available.
    - **Enterprise GPU tier:** Databricks H100 environments for larger matrix runs, high-throughput local inference, and heavier replay/check workloads.
    - **Model-serving tier:** Hugging Face APIs, OpenAI-compatible APIs, Anthropic-compatible APIs, and local `transformers` backends.
    - **Quantized deployment tier:** 4-bit or higher quantization for Llama-70B/80B-class models, DeepSeek-V3-671B-class mixture-of-experts deployment, and other large-agent backends where full-precision local inference is impractical.

    PCG-MAS treats these execution backends as interchangeable providers behind a certificate and replay layer. The accepted artifact remains the same: a claim with committed evidence, replay metadata, execution-contract checks, and deterministic verification fields.

    ## Theory glossary: files → quantities

    | File | Quantity / Symbol | Short description |
    |---|---|---|
    | `src/pcg/certificate.py` | `Z`, certificate object | Claim-level proof-carrying record containing evidence, metadata, replay support, and checker-facing fields. |
    | `src/pcg/commitments.py` | `H(x(v)) = h(v)` | Evidence integrity commitment used to bind retrieved records to stable hashes. |
    | `src/pcg/checker.py` | `Check(Z; G_t)` | Externally checkable acceptance predicate over the certificate and task graph. |
    | `src/pcg/orchestrator/replay_handlers.py` | `Π`, replay pipeline | Versioned replay support for deterministic or logged regeneration of support paths. |
    | `src/pcg/risk.py` | `R`, risk score | Calibrated risk used by the answer / verify / escalate / refuse controller. |
    | `src/pcg/responsibility.py` | `Resp@1`, responsibility score | Interventional diagnosis of which evidence/tool path is most responsible for failure. |
    | `src/pcg/independence.py` | `ρ`, residual dependence | Dependence correction for redundant support paths and correlated failures. |
    | `src/pcg/privacy.py` | `ε`, private-risk budget | Optional privacy/noise mechanism for risk summaries and audit reporting. |
    | `src/pcg/eval/rho.py` | `ρ^UCB` | Upper-confidence estimate for dependence-aware redundancy bounds. |
    | `src/pcg/eval/coverage.py` | coverage | Fraction of claims that remain answerable under the controller policy. |
    | `scripts/runs/run_matrix.py` | run cell | Dataset/model/seed/experiment execution unit. |
    | `scripts/tables/collect_paper_metrics.py` | metric row | Aggregated row consumed by table and figure builders. |
    | `scripts/figures/make_paper_figures.py` | R1-R5 figures | Figure construction over fully collected PCG-MAS metrics and overlay fields. |

    ## Dynamic repository profile

    Release-style file count: **196**

    ### File-type distribution

    | Type | Files | Share |
    |---|---:|---:|
    | `.py` | 119 | 60.7% |
| `.yaml` | 15 | 7.7% |
| `.pdf` | 14 | 7.1% |
| `.png` | 13 | 6.6% |
| `.sh` | 12 | 6.1% |
| `.txt` | 8 | 4.1% |
| `[no extension]` | 4 | 2.0% |
| `.json` | 3 | 1.5% |
| `.yml` | 2 | 1.0% |
| `.md` | 2 | 1.0% |
| `.ipynb` | 2 | 1.0% |
| `.example` | 1 | 0.5% |

    ### Top-level distribution

    | Path | Files | Share |
    |---|---:|---:|
    | `scripts/` | 73 | 37.2% |
| `src/` | 48 | 24.5% |
| `artifacts/` | 29 | 14.8% |
| `app/` | 18 | 9.2% |
| `configs/` | 15 | 7.7% |
| `.github/` | 2 | 1.0% |
| `notebooks/` | 2 | 1.0% |
| `.env.example/` | 1 | 0.5% |
| `.gitignore/` | 1 | 0.5% |
| `Makefile/` | 1 | 0.5% |
| `README.md/` | 1 | 0.5% |
| `docs/` | 1 | 0.5% |

    ### Compact release tree

    ```text
    .env.example
.github/workflows/ci.yml
.github/workflows/deploy_space.yml
.gitignore
Makefile
README.md
app/Dockerfile
app/README.md
app/app.py
app/components/__init__.py
app/components/agent_trace.py
app/components/byok_modal.py
app/components/certificate_card.py
app/components/llm_client.py
app/components/theme.py
app/demo_data/results_fixtures.json
app/pages/1_Live_Run.py
app/pages/2_Certificate_Inspector.py
app/pages/3_Side_by_Side.py
app/pages/4_Results_Browser.py
app/pages/5_Auditor_Demo.py
app/pages/__init__.py
app/pages/_live_run_helpers.py
app/requirements.txt
artifacts/coverage_plan.json
artifacts/dataset_schema_tatqa_weblinx.txt
artifacts/v4_preview/manifest_hash.txt
configs/frontier_merge.yaml
configs/local_40_cells.yaml
configs/preflight_2_cells.yaml
configs/preflight_40_cells.yaml
configs/r1_fever.yaml
configs/r1_hotpotqa.yaml
configs/r1_pubmedqa.yaml
configs/r1_tatqa.yaml
configs/r1_weblinx.yaml
configs/r2_redundancy.yaml
configs/r3_responsibility.yaml
configs/r4_risk.yaml
configs/r5_overhead.yaml
configs/r6_cross_domain.yaml
configs/v4_matrix.yaml
docs/manifest.json
notebooks/pcg_v4_colab_16cells.ipynb
notebooks/run_large_llms.ipynb
pyproject.toml
requirements.txt
scripts/__init__.py
scripts/analysis/audit_envelope.py
scripts/analysis/pick_top_k.py
scripts/build_paper_artifacts.py
scripts/build_readme.py
scripts/common/__init__.py
scripts/common/benchmark_specs.py
scripts/common/experiment_io.py
scripts/common/paper_metric_validation.py
scripts/common/paper_metrics.py
scripts/common/paths.py
scripts/common/run_manifest.py
scripts/common/schema.py
scripts/deploy_to_anonymous_space.sh
scripts/experiments/__init__.py
scripts/experiments/run_ablations.py
scripts/experiments/run_r1_checkability.py
scripts/experiments/run_r2_redundancy.py
scripts/experiments/run_r3_responsibility.py
scripts/experiments/run_r4_risk_privacy.py
scripts/experiments/run_r5_overhead.py
scripts/figures/__init__.py
scripts/figures/build_all_figures.py
scripts/figures/legacy_r1_r5_plots.py
scripts/figures/make_paper_figures.py
scripts/figures/make_r3_open_mixed.py
scripts/figures/make_r4_privacy_frontier.py
scripts/figures/make_r5_scaling.py
scripts/maintain/__init__.py
scripts/maintain/audit_forbidden_terms.py
scripts/maintain/audit_repo_layout.py
scripts/maintain/audit_secrets.py
scripts/maintain/build_backends_manifest.py
scripts/notebooks/__init__.py
scripts/notebooks/merge_frontier_runs.py
scripts/run_local_llms.sh
scripts/runs/__init__.py
scripts/runs/run_local_40_cells.py
scripts/runs/run_matrix.py
scripts/runs/run_pcgmas_benchmark_suite.py
scripts/runs/run_pcgmas_interactive.sh
scripts/runs/run_preflight.py
scripts/runs/run_preflight_40_cells.py
scripts/tables/__init__.py
scripts/tables/build_all_tables.py
scripts/tables/collect_paper_metrics.py
scripts/tables/make_paper_tables.py
scripts/tables/repair_paper_metrics_metadata.py
scripts/tables/validate_paper_metrics.py
src/pcg/__init__.py
src/pcg/certificate.py
src/pcg/checker.py
src/pcg/cli.py
src/pcg/commitments.py
src/pcg/graph.py
src/pcg/independence.py
src/pcg/privacy.py
src/pcg/responsibility.py
src/pcg/retrieval.py
src/pcg/risk.py
workflow/workflow_v3.png
    ```

    ## Compute environments

    - Databricks (H100) Enterprise subscription
    - MacBook Pro M4 Pro
    - Google Colab A100 with Drive-backed cache for large model and result persistence

    ## License

    This repository is intended for reproducible research and systems benchmarking. See the repository license file when present.
