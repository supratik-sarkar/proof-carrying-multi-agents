# GATE0_API_FIRST_FREEZE.md — PCG-MAS v3.0 Gate-0.1 Scientific Freeze

**Freeze Date:** 2026-09-05
**Status:** **FROZEN (Pre-Registration Locked — Gate-0.1 Pass)**
**Target Platform:** macOS Darwin `arm64` (Apple Silicon M-Series Host)
**Target Workspace:** `.` (PCG-MAS v3.0 Repository Root)
**Environment:** `.venv-pcg-mas` (`Python 3.12.13`)
**Author:** Antigravity (Independent Release & Verification Operator)

---

## 1. Frozen Primary Benchmark Architecture

The primary scientific evaluation of PCG-MAS v3.0 is frozen as an **API-First 7-Model Benchmark**:

```text
7 Models × 8 Datasets = 56 Primary Benchmark Cells
Primary Experiment Seeds: [0, 1, 2, 3] (Holdout Seed: 4)
Final Benchmark Target N: PENDING_SAMPLE_SIZE_FREEZE (Governed by GATE0_SAMPLE_SIZE_PROTOCOL.json)
```

No local GPU cluster or Google Colab inference is required for the primary benchmark path. The MacBook Pro M4 Pro acts as the local orchestration, statistical estimation, policy checking, and artifact generation host.

### 1.1 The Seven Frozen Providers & Models

| # | Provider | Frozen Canonical Model ID | Secret Env Key | Endpoint Mode | Sampling Seed Parameter |
|---|---|---|---|---|---|
| 1 | **OpenAI** | `gpt-5.6-sol` | `OPENAI_API_KEY` | `REST_CHAT_COMPLETIONS` | `seed` (best-effort) |
| 2 | **Anthropic** | `claude-opus-5` | `ANTHROPIC_API_KEY` | `REST_MESSAGES` | `null` (not supported) |
| 3 | **DeepSeek** | `deepseek-v4-pro` | `DEEPSEEK_API_KEY` | `REST_CHAT_COMPLETIONS` | `null` (not supported) |
| 4 | **Google Gemini** | `gemini-3.1-pro-preview` | `GEMINI_API_KEY` | `REST_GENERATE_CONTENT` | `seed` (best-effort) |
| 5 | **xAI** | `grok-4.6` | `XAI_API_KEY` | `REST_CHAT_COMPLETIONS` | `seed` (best-effort) |
| 6 | **Mistral** | `mistral-medium-3-5` | `MISTRAL_API_KEY` | `REST_CHAT_COMPLETIONS` | `random_seed` (best-effort) |
| 7 | **Cohere** | `command-a-plus-05-2026` | `COHERE_API_KEY` | `REST_CHAT` | `seed` (best-effort) |

Optional Tracing:
- `LANGSMITH_API_KEY`, `LANGSMITH_TRACING=false`, `LANGSMITH_PROJECT=pcg-mas-v3-0-private`

---

## 2. Frozen Scientific Specifications & Companion Manifests

All companion specifications have been generated with cryptographic SHA-256 integrity:

| Artifact | File | SHA-256 Digest |
|---|---|---|
| **Backend Manifest** | `GATE0_BACKEND_MANIFEST.json` | `0b62357113afdcbc78797629432656e55eeb4f87848c599e5840f32ae1f5b3a8` |
| **56-Cell Matrix** | `GATE0_56_CELL_MATRIX.csv` | `99b17100c39f722961fcf5bba34f3ec710e932448c279289b5c3a361cca11e15` |
| **Sample Size Protocol** | `GATE0_SAMPLE_SIZE_PROTOCOL.json` | `8288c5ad4671f70076f03d654959e5af99b0348146a77e72877e28642c51810b` |
| **Provider Semantics** | `GATE0_PROVIDER_SEMANTICS.json` | `1874e1754eae6d97a4297dab3ee6d2b5aa5484826b167b126aab70103fc32fee` |
| **Seeds & Bootstrap** | `GATE0_SEEDS.json` | `9ffba6b372773ea4f5518da734d73994e93dbe35aeb256b53ea5f19679890afa` |
| **Checker Protocol (A15)** | `GATE0_CHECKER_PROTOCOL.json` | `27f6505f433c6051144eb71df74f2cc029bba61531f78006341ee8d842cfe509` |
| **Dependence Spec (A08)** | `GATE0_A8_DEPENDENCE_SPEC.json` | `d7bfab33e208013775657d537e38b12a74e54fcf78d2d2ee24e67877d581a316` |
| **Controller Spec (A17)** | `GATE0_A17_CONTROLLER_SPEC.json` | `b37e6119bfb471c4f5d738d49a5652fac8fd4cd5099df8b82037b858c83148c5` |
| **Resource Accounting** | `GATE0_RESOURCE_ACCOUNTING.json` | `97832104fd5152e2a9c6754378c44914998bee8071f45f7fa256cc7335ebe2ec` |
| **Pricing Manifest Schema**| `GATE0_PRICING_SCHEMA.json` | `7c13a19ef3072e7f9af4fec7940e04eaaeb8497bcacbe1cfcbc98de85cce2b20` |
| **Manuscript Alignment** | `GATE0_MANUSCRIPT_ALIGNMENT.md` | `ba0827221053e91b1d6147ae11123853c9af3640c8f7e247aaccae8cc93865ae` |
| **Anonymity Audit** | `ANONYMITY_AUDIT_V3.md` | `75cab56bf1a91639094b75817c75fff44476f1ea465e7db19bd8e1b8484e7e9e` |

---

## 3. Core Protocol Rules & Invariants

1. **Pre-Registered Sample Size Protocol:** Removes arbitrary fixed $N=30$. Non-overlapping calibration pilot split used solely for variance/cost estimation. Final sample size determined by formal power/precision protocol ($1-\beta=0.80, \alpha=0.05, w=0.03, \delta=0.05$).
2. **Seed Architecture & Provider Semantics:** Clear separation between repository `experiment_seed` (`[0, 1, 2, 3]`) and provider-specific `provider_sampling_seed`. Lowest variance controlled mode per provider without unsupported parameter forcing.
3. **Publication-Grade Provenance:** Every future API invocation captures all 23 provenance fields specified in `GATE0_BACKEND_MANIFEST.json`.
4. **Statistical Inference:** Paired crossed seed $\times$ example bootstrap ($B=2000$) with $S + V = \Delta$ preserved exactly.
5. **Checker Firewall (A15):** Calibration parameters frozen on the calibration partition prior to test-set ingestion. Target constraint $\alpha_{\text{ent}} = 0.05$.
6. **A8 Dependence Evidence Sufficiency:** Positive marginal LCBs + frozen $\rho$ precision requirement (`max_ci_width <= 0.50`, `max_rel_width <= 0.40`) + implementation evidence floors ($n_{\text{min}}=200, k_{\text{min}}=5, q_0=2$). Exact-$k$ $U_{\text{joint}}(k, \delta)$ fallback without extrapolation.
7. **A17 Controller Separation:** Strict theoretical quarantine between A17-A (fixed-model calibration error bound $R_{\text{max}}^{\text{cal}} \le 2 L_{\text{ctrl}} \varepsilon_{\text{cal}}$) and A17-B (descriptive parametric sensitivity against per-$\theta$ oracle).
8. **Double-Blind Anonymity:** 100% verified across all reviewer-facing components.
