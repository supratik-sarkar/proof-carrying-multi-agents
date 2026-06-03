"""
Demo-runtime implementation of the PCG-MAS paper algorithm.

This module implements the same public semantics as the paper implementation in:
    src/pcg/risk.py

It is intentionally written against the lightweight demo schema in
app/pcg_glue/schemas.py so the Hugging Face Space remains self-contained
and does not require the full R1-R5 experiment harness objects such as
GroundingCertificate, Checker, or AgenticRuntimeGraph.

Lifted algorithms (paper equations preserved):
    - C(b, a) cost model               Eq. 22, App. C.4 Eq. 93
    - posterior false-accept risk r    Eq. 24
    - ThresholdPolicy.choose           Thm 3(ii), App. C.4

Action mapping (paper -> schema):
    Action.ANSWER   -> RiskAction.ANSWER
    Action.VERIFY   -> RiskAction.VERIFY
    Action.ESCALATE -> RiskAction.ESCALATE
    Action.REFUSE   -> RiskAction.REFUSE

Calibration (Calibrator / LearnedPolicy / ECE machinery in the paper) is
intentionally omitted from the demo: it requires labelled rollouts which
the live demo does not have. The paper continues to own that.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from pcg_glue.schemas import (
    ChannelName, ChannelState,
    ClaimCertificate,
    RiskAction, RiskDecision,
)


# =============================================================================
# Cost model (Eq. 22 / App. C.4 Eq. 93)
# =============================================================================

@dataclass
class CostModel:
    """C(b, a) = C_lat(a) + C_tok(a) + C_tool(a) + lambda * E[L_harm | a]

    Affine-in-r harm parameterization (App. C.4, Eq. 93):

        E[L_harm | a] =
            H_FA * eta_a * r          if a in {Answer, Verify, Escalate}
            H_Ref                     if a == Refuse

    Demo defaults are chosen so that the risk controller's behavior is
    intuitively interpretable:
        - tiny non-harm costs (Answer ~0, Verify > Answer, Escalate >> Verify)
        - lambda = 1, H_FA = 1, H_Ref = 0.05
        - eta_Answer = 1.0, eta_Verify = 0.4, eta_Escalate = 0.1, eta_Refuse = 0
    """
    # Non-harm cost components (latency + tokens + tool calls).
    # Sums chosen so the 4 action curves cross at clean r thresholds:
    #   nonharm(Answer)   = 0.01
    #   nonharm(Verify)   = 0.13   (0.06 + 0.05 + 0.02)
    #   nonharm(Escalate) = 0.295  (0.150 + 0.055 + 0.090)
    #   nonharm(Refuse)   = 0.00
    c_lat:  dict[RiskAction, float] = field(default_factory=lambda: {
        RiskAction.ANSWER:   0.000,
        RiskAction.VERIFY:   0.060,
        RiskAction.ESCALATE: 0.150,
        RiskAction.REFUSE:   0.000,
    })
    c_tok:  dict[RiskAction, float] = field(default_factory=lambda: {
        RiskAction.ANSWER:   0.010,
        RiskAction.VERIFY:   0.050,
        RiskAction.ESCALATE: 0.055,
        RiskAction.REFUSE:   0.000,
    })
    c_tool: dict[RiskAction, float] = field(default_factory=lambda: {
        RiskAction.ANSWER:   0.000,
        RiskAction.VERIFY:   0.020,
        RiskAction.ESCALATE: 0.090,
        RiskAction.REFUSE:   0.000,
    })

    # Harm parameters — chosen so the controller's crossovers fall at:
    #   r ≈ 0.20  Answer  → Verify
    #   r ≈ 0.55  Verify  → Escalate
    #   r ≈ 0.85  Escalate → Refuse
    lam:    float = 1.0
    h_fa:   float = 1.0
    h_ref:  float = 0.38

    # Residual risk multipliers (Answer carries full r, Refuse carries none).
    # The decay (1.0 → 0.4 → 0.1 → 0.0) reflects: a human-reviewed Escalation
    # has 10× less residual false-accept risk than a direct Answer.
    eta: dict[RiskAction, float] = field(default_factory=lambda: {
        RiskAction.ANSWER:   1.0,
        RiskAction.VERIFY:   0.40,
        RiskAction.ESCALATE: 0.10,
        RiskAction.REFUSE:   0.00,
    })

    def nonharm(self, a: RiskAction) -> float:
        return (self.c_lat.get(a, 0.0)
                + self.c_tok.get(a, 0.0)
                + self.c_tool.get(a, 0.0))

    def cost(self, a: RiskAction, r: float) -> float:
        """C(b, a) evaluated with posterior false-accept risk r."""
        if a == RiskAction.REFUSE:
            return self.nonharm(a) + self.lam * self.h_ref
        eta_a = self.eta.get(a, 1.0)
        return self.nonharm(a) + self.lam * self.h_fa * eta_a * r


# =============================================================================
# Posterior false-accept risk (Eq. 24)
# =============================================================================

def posterior_risk_from_certificates(
    claim_certs: list[ClaimCertificate],
    rho: float = 1.0,
) -> float:
    """Compute r(b, Z) from the paper's Eq. 24, instantiated for the demo:

        r ~ rho^{k-1} * prod_i (1 - confidence_i) * I[claim_i accepted]

    Each ClaimCertificate contributes one factor:
      - If accepted: factor = max(0, 1 - claim.confidence)
      - If rejected: factor = 1.0  (the un-verified claim is maximally risky)

    `rho` is the inter-claim dependence inflation (Assumption 3 / Eq. 50);
    rho=1 means independent claims (the cleanest case).
    """
    if not claim_certs:
        return 1.0
    prod = 1.0
    for cc in claim_certs:
        if cc.accepted:
            prod *= max(0.0, 1.0 - float(cc.claim.confidence))
        else:
            prod *= 1.0
    k = len(claim_certs)
    return min(1.0, float(rho) ** max(0, k - 1) * prod)


# =============================================================================
# Threshold policy (Thm 3 ii / App. C.4)
# =============================================================================

@dataclass
class ThresholdPolicy:
    """Piecewise-threshold policy over {Answer, Verify, Escalate, Refuse}.

    Given a CostModel, computes the lower-envelope of the four affine-in-r
    lines and selects the argmin for each r in [0, 1].
    """
    cost_model: CostModel
    actions: tuple[RiskAction, ...] = (
        RiskAction.ANSWER, RiskAction.VERIFY,
        RiskAction.ESCALATE, RiskAction.REFUSE,
    )

    def action_cost_line(self, a: RiskAction) -> tuple[float, float]:
        """Return (intercept, slope) such that C(a, r) = intercept + slope * r."""
        if a == RiskAction.REFUSE:
            return (self.cost_model.nonharm(a) + self.cost_model.lam * self.cost_model.h_ref, 0.0)
        slope = self.cost_model.lam * self.cost_model.h_fa * self.cost_model.eta.get(a, 1.0)
        return (self.cost_model.nonharm(a), slope)

    def choose(self, r: float) -> RiskAction:
        """arg min_a C(a, r). Tie-broken by action preference order."""
        best_a = self.actions[0]
        best_c = math.inf
        for a in self.actions:
            intercept, slope = self.action_cost_line(a)
            c = intercept + slope * r
            if c < best_c - 1e-12:
                best_c = c
                best_a = a
        return best_a

    def thresholds(self) -> list[tuple[float, RiskAction, RiskAction]]:
        """Sorted list of crossing points (tau, a_below, a_above) on r in [0,1]."""
        crossings: list[tuple[float, RiskAction, RiskAction]] = []
        acts = list(self.actions)
        for i, a in enumerate(acts):
            for b in acts[i + 1:]:
                ia, sa = self.action_cost_line(a)
                ib, sb = self.action_cost_line(b)
                if abs(sa - sb) < 1e-12:
                    continue
                tau = (ib - ia) / (sa - sb)
                if 0.0 <= tau <= 1.0:
                    below, above = (a, b) if sa < sb else (b, a)
                    crossings.append((tau, below, above))
        return sorted(crossings, key=lambda x: x[0])


# =============================================================================
# Top-level decide() — produces a RiskDecision for the certificate
# =============================================================================

def decide(
    claim_certs: list[ClaimCertificate],
    *,
    cost_model: Optional[CostModel] = None,
    rho: float = 1.0,
) -> RiskDecision:
    """Run the full Theorem 3 (ii) controller and produce a RiskDecision.

    Steps:
        1. r = posterior_risk_from_certificates(claim_certs, rho)
        2. C(a, r) for each action via CostModel
        3. action* = ThresholdPolicy.choose(r)
        4. Dominant failure channel = mode of failed channels across rejected claims
        5. Reason codes summarize why the controller chose what it did
    """
    cm = cost_model or CostModel()
    policy = ThresholdPolicy(cm)

    r = posterior_risk_from_certificates(claim_certs, rho=rho)
    expected_cost: dict[RiskAction, float] = {
        a: round(cm.cost(a, r), 6) for a in RiskAction
    }
    residual_risk: dict[RiskAction, float] = {
        RiskAction.ANSWER:   r,
        RiskAction.VERIFY:   r * cm.eta.get(RiskAction.VERIFY, 0.4),
        RiskAction.ESCALATE: r * cm.eta.get(RiskAction.ESCALATE, 0.1),
        RiskAction.REFUSE:   0.0,
    }
    chosen = policy.choose(r)

    # Identify dominant failure channel across rejected claims
    dominant: Optional[ChannelName] = None
    if any(not cc.accepted for cc in claim_certs):
        fail_counts: dict[ChannelName, int] = {}
        for cc in claim_certs:
            if not cc.accepted:
                for chname, verdict in cc.channels.items():
                    if verdict.state == ChannelState.FAIL:
                        fail_counts[chname] = fail_counts.get(chname, 0) + 1
        if fail_counts:
            dominant = max(fail_counts, key=fail_counts.get)

    # Reason codes
    reasons: list[str] = []
    if not claim_certs:
        reasons.append("no claims extracted")
    else:
        n_acc = sum(1 for cc in claim_certs if cc.accepted)
        n_tot = len(claim_certs)
        reasons.append(f"{n_acc}/{n_tot} claims accepted")
        reasons.append(f"posterior_risk={r:.3f}")
        if dominant is not None:
            reasons.append(f"dominant_failure_channel={dominant.value}")
        crossings = policy.thresholds()
        if crossings:
            reasons.append(
                "thresholds=" + ", ".join(
                    f"r={tau:.2f}:{below.value}->{above.value}"
                    for tau, below, above in crossings
                )
            )

    summary = _build_summary(chosen, r, claim_certs, dominant)

    return RiskDecision(
        action=chosen,
        posterior_risk=r,
        expected_cost=expected_cost,
        residual_risk=residual_risk,
        dominant_failure_channel=dominant,
        reason_codes=reasons,
        summary=summary,
    )


def _build_summary(
    chosen: RiskAction,
    r: float,
    claim_certs: list[ClaimCertificate],
    dominant: Optional[ChannelName],
) -> str:
    if not claim_certs:
        return "No claims to evaluate; controller chose Refuse by default."

    n_acc = sum(1 for cc in claim_certs if cc.accepted)
    n_tot = len(claim_certs)
    base = f"r={r:.2f}, {n_acc}/{n_tot} claims accepted."

    if chosen == RiskAction.ANSWER:
        return base + " Posterior risk is low enough to answer directly."
    if chosen == RiskAction.VERIFY:
        return base + " Posterior risk warrants a second-pass verification before answering."
    if chosen == RiskAction.ESCALATE:
        return base + " Posterior risk is high; escalate to a stronger verifier / human review."
    if chosen == RiskAction.REFUSE:
        if dominant is not None:
            return base + f" Refusing: {dominant.value} channel failed."
        return base + " Refusing: residual risk exceeds all action thresholds."
    return base


__all__ = [
    "CostModel",
    "ThresholdPolicy",
    "decide",
    "posterior_risk_from_certificates",
]
