# SCIENTIFIC_SPEC_ALIGNMENT_V3.md

Authority: `pcg_mas_iclr2027_v3-0.tex` and `pcg_mas_iclr2027_v3-0.md` jointly. Where the implementation disagreed, the implementation conformed; every such case is recorded here.

## 1. Definitions implemented exactly

| Object | Definition honoured | Module | Verified by |
|---|---|---|---|
| Four conjuncts | `Check = V_H · V_Π · V_Γ · V_⊢`, **unknown counts as failure** | `channels.check` | `test_unknown_conjunct_is_failure` |
| Five audit channels | `IntFail, ReplayFail, DriftFail, CheckFail, CovGap` — a **different layer** from the conjuncts | `channels.CHANNELS` | `test_five_channels_and_four_conjuncts` |
| Conjunct→channel map | `V_Π` splits by *when* divergence is seen (pinned vs fresh); `CheckFail` is cross-cutting | `channels.CONJUNCT_TO_CHANNELS` | contract export |
| `λ_[k]` | `Pr(∩E_i)/∏p_i = (dP/dQ)(1,…,1)`; **`None` when any marginal is 0** | `science.dependence.lambda_all_fail` | `test_lambda_undefined_when_marginal_zero` |
| `ρ_[k]` | `max{1, λ^{1/(k−1)}}` — the clamp is part of the definition | `rho_from_lambda` | `test_rho_clamped_at_one` |
| `ρ̂_UCB` | lattice max over `\|I\|≥q₀`; `p̂ᵢ⁻ = 0 ⇒ ∞` (fail closed) | `rho_ucb` | `test_gate_is_three_state...` |
| Gate | `OPEN / CLOSED / INSUFFICIENT_EVIDENCE`, with a frozen evidence floor | `GateState`, `EvidenceFloor` | same |
| `U_joint(k,δ)` | binomial UCB supporting **that exact k**, no extrapolation | `u_joint` | `test_u_joint_defined_without_rho` |
| Common-mode floor | `Pr(∩E_i) ≥ q_cm`; `k* = ⌈log q_cm / log ε_path⌉` | `common_mode_floor` | `test_common_mode_floor_formula` |
| `B_cov(δ)` | `Σ_h π_h · min{1, Σ_j U_{j,h}}` — **`π_unc` charged once**, inner sum clipped | `science.audit.stratified_envelope` | `test_pi_unc_charged_once_not_per_channel`, `test_inner_sum_is_clipped_at_one` |
| `ε_tax^cov` | covered-strata scoped; challenge-set **alarm**, never a deployment bound | `eps_tax_challenge` | `test_eps_tax_is_labelled_as_alarm` |
| `ε_src` | named, outside the certificate, never bounded by it | record field + docs | — |
| `Λ_∪` | `Σ_j Pr(Fail_j) − Pr(∪_j Fail_j) = E[(N_F−1)_+]` | `union_slack` | `test_union_slack_equals_expected_excess_multiplicity` |
| `Δ = S + V` | exact on per-example losses, asserted to `1e-12` on **every** resample | `science.sv`, `stats.bootstrap` | `test_sv_identity_exact_random` |
| Shift alarm | `D_alarm = max{0, 2a_LCB − 1}`; **never** substituted for `D̄_t` | `science.shift` | `test_alarm_is_lower_bound_and_never_a_bound_substitute` |
| Responsibility | replay-interventional total effect; `Unresolved` at margin `< 2√(2log(2\|U\|/δ)/M)` | `science.responsibility` | `test_unresolved_when_margin_small` |
| `τ*` | numerical minimiser; the closed form is exposed separately as an approximation | `tau_star_exact` / `tau_star_approx` | `test_tau_star_exact_beats_approximation` |
| A17-A / A17-B | A17-A tests `2·L_ctrl·ε_cal`; A17-B is **model-relative to a per-θ oracle** only | `science.controller` | `test_regret_respects_analytic_bound`, `test_sensitivity_uses_per_theta_oracle` |

## 2. Discrepancies found and how they were resolved

| # | v3 requirement | Legacy implementation | Resolution |
|---|---|---|---|
| D1 | S/V exact on per-example losses | `eval/metrics.py:102` computes `S = harm_nocert·(1−α)`, `V = α·(harm_nocert − harm_pcg)` from **cell rates** | superseded by `pcg.v3.science.sv`; the rate form cannot satisfy the `1e-12` assertion and is not used by any v3 artifact. Legacy module left in place, documented here |
| D2 | five channels | `DriftFail` appeared once tree-wide; `checker.py` docstring says "Four-channel" | v3 enum is authoritative; legacy docstring untouched (scientific TeX and legacy code are not silently edited) |
| D3 | λ vs ρ separated, fail-closed | `eval/rho.py` estimates one scalar `rho` under a homogeneous-marginal assumption, no clamp, no lattice, no zero-denominator rule | superseded by `science.dependence` |
| D4 | union slack multiplicity weighted | `eval/tightness.py` reports one conflated `RHS − LHS` over **four** channels | superseded by `science.audit.union_slack` |
| D5 | no undefined denominator becomes 0 | 123 `.get(...,0.0)` sites in legacy code | v3 metric path returns `None`; verified by `test_missing_probes_yield_undefined_not_zero` |
| D6 | withdrawn quantities ungenerable | `run_preflight.py` still emits `safety_gain` / `responsibility_lift_pp` | **not resolved in this pass** — requires live-tree edits to `Makefile`/`cli.py`; documented in the changelog |
| D7 | evidence floor frozen pre-evaluation | absent | `EvidenceFloor(n_min=200, k_min=5, q0=2)` frozen in `catalog.py` A08 params |
| D8 | `α_ent` frozen pre-evaluation | absent | `alpha_ent = 0.05` frozen in A15 params with the threshold rule recorded |

## 3. Two decisions that were frozen, and need your confirmation

Both were previously flagged as open and could not stay open without making a check vacuous:

1. **A08 evidence floor and gate band** — `n_min=200, k_min=5, q0=2`, `bar_rho=1.35`, `Δ=0.15`, `q_cm=0.05`, `ε_path=0.18`.
2. **A15 checker constraint** — `α_ent=0.05`, threshold rule "max retained coverage subject to `UCB(FPR) ≤ α_ent`".

They are frozen in `src/pcg/v3/workstreams/catalog.py` and enter every `spec.json` hash. Changing one after seeing results is a spec violation that `verify_spec()` refuses.

## 4. Guarantee-boundary discipline in code

* the app never re-defines a scientific term — it reads `app/shared/contract.json`, generated from the core;
* `HostedProvider.generate` raises rather than calling out, so `NETWORK_API_MODEL_CALLS` cannot drift from 0 by accident;
* audit channels fire **only on accepted-and-bad runs**, so a correct rejection fires none — the demo reflects this rather than inventing a signal.
