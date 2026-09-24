# GATE0_MANUSCRIPT_ALIGNMENT.md — PCG-MAS v3.0 Gate-0.1 Manuscript Alignment Report

**Date:** 2026-09-05
**Pass:** Gate-0.1 Scientific Correction Pass
**Author:** Automated Release & Verification Suite
**Target:** `.` (PCG-MAS v3.0 Repository Root)

---

## 1. Scientific Alignment & Frozen Model Roster

The PCG-MAS v3.0 experimental design is formally reconciled with the Gate-0.1 API-first seven-model primary benchmark:

| Provider | Canonical Frozen Model ID | Primary Secret Mapping | Endpoint Mode | Sampling Seed Support |
|---|---|---|---|---|
| **OpenAI** | `gpt-5.6-sol` | `OPENAI_API_KEY` | `REST_CHAT_COMPLETIONS` | `seed` supported (best-effort) |
| **Anthropic** | `claude-opus-5` | `ANTHROPIC_API_KEY` | `REST_MESSAGES` | `null` (not supported) |
| **DeepSeek** | `deepseek-v4-pro` | `DEEPSEEK_API_KEY` | `REST_CHAT_COMPLETIONS` | `null` (not supported) |
| **Google Gemini** | `gemini-3.1-pro-preview` | `GEMINI_API_KEY` | `REST_GENERATE_CONTENT` | `seed` supported (best-effort) |
| **xAI** | `grok-4.6` | `XAI_API_KEY` | `REST_CHAT_COMPLETIONS` | `seed` supported (best-effort) |
| **Mistral** | `mistral-medium-3-5` | `MISTRAL_API_KEY` | `REST_CHAT_COMPLETIONS` | `random_seed` supported (best-effort) |
| **Cohere** | `command-a-plus-05-2026` | `COHERE_API_KEY` | `REST_CHAT` | `seed` supported (best-effort) |

---

## 2. 56-Cell Primary Matrix & Pre-Registered Sample Size Protocol

The primary evaluation matrix is defined as:
$$\text{Primary Benchmark} = 7 \text{ API Models} \times 8 \text{ Datasets} = 56 \text{ Cells}$$
Evaluated across primary experiment seeds: `[0, 1, 2, 3]` (plus holdout seed `4`).

**Datasets (8):**
1. `hotpotqa` (HotpotQA)
2. `twowiki` (2WikiMultiHopQA)
3. `tatqa` (TAT-QA)
4. `toolbench` (ToolBench)
5. `fever` (FEVER)
6. `pubmedqa` (PubMedQA)
7. `weblinx` (WebLinx)
8. `adversarial_integrity` (Adversarial Challenge Set)

**Sample Size Protocol (`GATE0_SAMPLE_SIZE_PROTOCOL.json`):**
- Unjustified fixed $N=30$ target removed.
- Matrix cell targets set to `PENDING_SAMPLE_SIZE_FREEZE`.
- Formal distinction between non-overlapping calibration pilot (variance/cost estimation only, zero contamination into final statistics) and final evaluation sample.
- Pre-registered sample size formula with power ($1-\beta=0.80$), significance ($\alpha=0.05$), precision half-width ($w=0.03$), MDE ($\delta=0.05$), finite-population correction, and fail-closed missing data handling.

---

## 3. Manuscript Tables (33) & Figures (10) Integrity

All 33 tables and 10 figures in `manuscript_artifact_registry.json` map cleanly to the frozen matrix:
- **Table Registry Coverage:** 33/33 registered (32 dynamic generators + 1 static notation table `tab:notation`).
- **Figure Registry Coverage:** 10/10 registered (9 dynamic dual PNG+PDF generators + 1 static workflow schematic `fig:workflow`).
- **Empirical Decoupling:** All empirical result placeholders remain strictly `[TBD]`. Zero fake or unverified empirical numbers are populated.

---

## 4. Invariant & Workstream Preservation

1. **Four Conjuncts & Five Audit Channels:**
   - $\text{Check} = V_H \cdot V_\Pi \cdot V_\Gamma \cdot V_\vdash$
   - Audit channels: $\text{IntFail}, \text{ReplayFail}, \text{DriftFail}, \text{CheckFail}, \text{CovGap}$
2. **Harm Decomposition:**
   - $S + V = \Delta$ holds as an exact numerical and theoretical invariant.
3. **Seed Semantics:**
   - `experiment_seed` (`[0, 1, 2, 3]`) governs repository stochasticity and paired bootstrap.
   - `provider_sampling_seed` is provider-specific with explicit determinism caveats.
4. **A8 Dependence Sufficiency:**
   - Rule: Positive marginal LCBs + frozen $\rho$ precision requirement (`max_ci_width <= 0.50`, `max_rel_width <= 0.40`) + implementation evidence floors ($n_{\text{min}}=200, k_{\text{min}}=5, q_0=2$).
   - Exact-$k$ $U_{\text{joint}}(k, \delta)$ fallback without extrapolation.
5. **A17 Controller Separation:**
   - A17-A: Perturbs only calibration risk $r_{\text{cal}}$ under fixed objective; tests $R_{\text{max}}^{\text{cal}} \le 2 L_{\text{ctrl}} \varepsilon_{\text{cal}}$.
   - A17-B: Parametric sweep across $\lambda, L_{\text{max}}$, costs, harm models; descriptive sensitivity against per-$\theta$ oracle, strictly quarantined from A17-A theorem comparison.
