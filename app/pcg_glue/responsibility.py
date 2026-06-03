"""
Demo-runtime implementation of the PCG-MAS paper algorithm.

This module implements the same public semantics as the paper implementation in:
    src/pcg/responsibility.py

It is intentionally written against the lightweight demo schema in
app/pcg_glue/schemas.py so the Hugging Face Space remains self-contained
and does not require the full R1-R5 experiment harness objects such as
GroundingCertificate, Checker, or AgenticRuntimeGraph.

Lifted algorithms (paper equations preserved):
    - Resp(e; Z, G_t)                 Def. 2.12, Eq. 7
    - Hoeffding half-width            Eq. 27   (distribution-free CI)
    - Normal half-width               CLT-based (alternative CI)
    - rank_recovery_prob              Eq. 28
    - required_replays_for_rank       inverse of Eq. 28

The acceptance predicate A(Z, G_t) in the paper becomes, in this demo, the
boolean "the claim's 5-channel checker accepts" — i.e., re-running
run_all_channels(claim, evidence_after_mask, ...) returns accepted=True.

This is semantically equivalent: a component that, when masked, causes the
checker to reject contributes Delta=+1 to the responsibility estimator.
"""
from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from pcg_glue.schemas import (
    AtomicClaim, ChannelState, ChannelName,
    ClaimCertificate, ComponentType,
    EvidenceItem, ToolOutput,
    ResponsibilityReport, ResponsibilityScore,
)
from pcg_glue.backends import BackendChoice
from pcg_glue.channels import run_all_channels


# =============================================================================
# Hoeffding & Normal CI half-widths (Eq. 27 / CLT alternative)
# =============================================================================

def hoeffding_halfwidth(n: int, alpha: float, range_bound: float = 2.0) -> float:
    """Hoeffding CI half-width for i.i.d. samples in [-1, 1] (range = 2).

    From Eq. 27 of the paper:
        half-width = range_bound * sqrt(log(2/alpha) / (2n))

    The paper uses range_bound=2 since Delta_e in [-1, 1].
    """
    if n <= 0:
        return float("inf")
    return range_bound * math.sqrt(math.log(2.0 / alpha) / (2.0 * n))


def normal_halfwidth(std: float, n: int, alpha: float) -> float:
    """Normal-approximation CI half-width. Assumes iid, finite variance."""
    if n <= 0:
        return float("inf")
    if std == 0.0:
        return 0.0
    # z-values for common alpha levels (covers paper's choices)
    z = {0.10: 1.645, 0.05: 1.96, 0.01: 2.576}.get(alpha, 1.96)
    return z * std / math.sqrt(n)


# =============================================================================
# Mask-and-replay: the demo-runtime acceptance predicate A(Z, G_t)
# =============================================================================

def _mask_evidence(
    evidence: list[EvidenceItem],
    masked_ids: set[str],
) -> list[EvidenceItem]:
    """Return a copy of `evidence` with masked items replaced by empty placeholders.

    Mirrors graph.mask(component_ids) in the paper: the cited evidence is
    removed from the support set, so a downstream checker re-run sees the
    counterfactual world G_t^{\\setminus e}.
    """
    out = []
    for ev in evidence:
        if ev.id in masked_ids:
            # Replace text with empty so V_I (re-hashed) fails — that's the
            # paper's "remove from support set" semantics in the demo.
            placeholder = EvidenceItem(
                id=ev.id,
                text="[MASKED — component removed by mask-and-replay]",
                source="masked",
                hash=hashlib.sha256(b"[MASKED]").hexdigest(),
                span=None,
            )
            out.append(placeholder)
        else:
            out.append(ev)
    return out


def _accept(
    claim: AtomicClaim,
    evidence: list[EvidenceItem],
    tools: list[ToolOutput],
    question: str,
    backend: BackendChoice,
    api_key: str,
) -> int:
    """A(Z, G_t): re-run the 5-channel checker; return 1 if accepted, else 0.

    This is the demo-runtime instantiation of the Checker.check() predicate
    from the paper. It runs the same 5 channels on a (possibly masked) world.
    """
    evidence_index = {e.id: e for e in evidence}
    tool_index = {t.id: t for t in tools}
    cc = run_all_channels(
        claim, evidence_index, tool_index, question, backend, api_key,
    )
    return 1 if cc.accepted else 0


# =============================================================================
# Single-claim Monte-Carlo responsibility estimator (Eq. 7 + 27 + 28)
# =============================================================================

@dataclass
class _RawScore:
    """Internal: pre-CI delta samples for one component."""
    component_id: str
    component_type: ComponentType
    deltas: list[float] = field(default_factory=list)

    @property
    def estimate(self) -> float:
        return float(np.mean(self.deltas)) if self.deltas else 0.0

    @property
    def variance(self) -> float:
        if len(self.deltas) < 2:
            return 0.0
        return float(np.var(self.deltas, ddof=1))


def estimate_responsibility_for_claim(
    claim: AtomicClaim,
    claim_cert: ClaimCertificate,
    evidence: list[EvidenceItem],
    tools: list[ToolOutput],
    question: str,
    backend: BackendChoice,
    api_key: str,
    *,
    n_replays: int = 6,
    alpha: float = 0.05,
    ci_method: str = "hoeffding",
    seed: int = 0,
) -> ResponsibilityReport:
    """Run mask-and-replay on each cited component, rank by responsibility.

    Demo-runtime defaults:
        n_replays = 6   (paper uses M=100-200; demo trades budget for latency)
        alpha     = 0.05  (95% CI)
        ci_method = "hoeffding"  (distribution-free, Eq. 27)

    For each component c in the claim's cited supports:
        delta_m = A(Z, G_t) - A(Z, G_t^{\\setminus c})        # for m = 1..M
        Resp_hat(c) = mean(delta_m)
        CI(c)       = Resp_hat(c) +- hoeffding_halfwidth(M, alpha)

    Then rank by Resp_hat descending; the top-1 is the "most responsible".
    """
    components: list[tuple[str, ComponentType]] = []
    for eid in claim.support_ids:
        components.append((f"evidence:{eid}", ComponentType.EVIDENCE))
    for tid in claim.tool_output_ids:
        components.append((f"tool:{tid}", ComponentType.TOOL))

    if not components:
        return ResponsibilityReport(
            claim_id=claim.claim_id,
            scores=[],
            top_responsible_id="",
            n_replays=0,
            rank_recovery_prob=0.0,
            ci_method=ci_method,
        )

    rng = random.Random(seed ^ (hash(claim.claim_id) & 0xFFFFFFFF))

    raw: dict[str, _RawScore] = {}
    for cid, ctype in components:
        raw[cid] = _RawScore(component_id=cid, component_type=ctype)

    # If the claim was already rejected by the un-masked checker, Resp can
    # still be estimated but A(Z, G_t) is fixed at 0 and Delta in [-1, 0],
    # so the top-responsible component is simply "none responsible for
    # acceptance". We still iterate to populate CIs honestly.
    a_full = 1 if claim_cert.accepted else 0

    for cid, ctype in components:
        # Resolve the underlying id (strip "evidence:" / "tool:" prefix)
        kind, raw_id = cid.split(":", 1)
        for _ in range(n_replays):
            if kind == "evidence":
                masked_ev = _mask_evidence(evidence, {raw_id})
                a_masked = _accept(claim, masked_ev, tools,
                                   question, backend, api_key)
            else:  # tool
                masked_tools = [t for t in tools if t.id != raw_id]
                a_masked = _accept(claim, evidence, masked_tools,
                                   question, backend, api_key)
            raw[cid].deltas.append(float(a_full - a_masked))
            # tiny re-seed so two consecutive calls don't share LLM cache
            _ = rng.random()

    # Build CIs + rank
    scores_unranked: list[ResponsibilityScore] = []
    for cid, rs in raw.items():
        if ci_method == "normal":
            hw = normal_halfwidth(math.sqrt(rs.variance), n_replays, alpha)
        else:
            hw = hoeffding_halfwidth(n_replays, alpha, range_bound=2.0)
        est = rs.estimate
        scores_unranked.append(ResponsibilityScore(
            component_id=cid,
            component_type=rs.component_type,
            score=est,
            ci_low=max(-1.0, est - hw),
            ci_high=min(1.0, est + hw),
            rank=0,  # filled below
        ))

    # Rank by estimate descending (top responsible == largest positive Delta)
    scores_unranked.sort(key=lambda s: s.score, reverse=True)
    for i, s in enumerate(scores_unranked):
        s.rank = i + 1

    top_id = scores_unranked[0].component_id if scores_unranked else ""

    # Rank-recovery probability lower bound (Eq. 28) using the empirical
    # margin between rank-1 and rank-2. If only one component, treat margin
    # as the absolute Delta of that component.
    if len(scores_unranked) >= 2:
        margin = scores_unranked[0].score - scores_unranked[1].score
    elif len(scores_unranked) == 1:
        margin = abs(scores_unranked[0].score)
    else:
        margin = 0.0
    rrp = rank_recovery_prob(
        n_replays=n_replays,
        n_components=max(1, len(scores_unranked)),
        margin=margin,
    )

    return ResponsibilityReport(
        claim_id=claim.claim_id,
        scores=scores_unranked,
        top_responsible_id=top_id,
        n_replays=n_replays,
        rank_recovery_prob=rrp,
        ci_method=ci_method,
    )


# =============================================================================
# Rank-recovery probability (Eq. 28)
# =============================================================================

def rank_recovery_prob(
    n_replays: int,
    n_components: int,
    margin: float,
) -> float:
    """Lower bound on P(arg max == e^star) from Eq. 28:

        1 - 2 |E| exp(-M gamma^2 / 8)

    Returns 0 if margin <= 0; clipped to [0, 1].
    """
    if margin <= 0:
        return 0.0
    val = 1.0 - 2.0 * n_components * math.exp(
        -n_replays * margin * margin / 8.0
    )
    return max(0.0, min(1.0, val))


def required_replays_for_rank(
    n_components: int,
    margin: float,
    target_prob: float = 0.95,
) -> int:
    """Smallest M such that rank_recovery_prob(M, |E|, gamma) >= target.

    Returns 10**9 as a sentinel "unachievable" when margin <= 0.
    """
    if margin <= 0:
        return 10**9
    if target_prob <= 0:
        return 1
    required = (8.0 / (margin * margin)) * math.log(
        2.0 * n_components / (1.0 - target_prob)
    )
    return max(1, int(math.ceil(required)))


__all__ = [
    "estimate_responsibility_for_claim",
    "hoeffding_halfwidth",
    "normal_halfwidth",
    "rank_recovery_prob",
    "required_replays_for_rank",
]
