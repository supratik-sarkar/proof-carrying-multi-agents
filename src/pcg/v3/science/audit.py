"""Contract-audit envelope, stratified sampling contract, and union slack.

v3.0 composition (the uncovered mass is charged ONCE, at the union level):

    B_cov(delta) = sum_{h in H_cov} pi_h * min{1, sum_{j in J} U_{j,h}(delta_{j,h})}
    Pr(ContractBad) <= B_cov(delta) + pi_unc + eps_tax^cov
    Pr(Bad)         <= Pr(ContractBad) + eps_src

The pooled form Pr(ContractBad) <= sum_j U_j(delta) is the special case of a
single stratum carrying all deployment mass with pi_unc = 0.

Exact union-overlap slack is multiplicity weighted:

    Lambda_union = sum_j Pr(Fail_j) - Pr(union_j Fail_j) = E[(N_F - 1)_+]
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

from ..channels import CHANNELS, CHANNEL_FIELD, Channel
from ..stats.intervals import hoeffding_halfwidth


@dataclass
class ChannelEnvelope:
    channel: str
    n_probes: int
    failures: int
    beta_hat: Optional[float]
    half_width: float
    upper: Optional[float]

    def to_dict(self) -> dict:
        return vars(self)


def channel_envelope(failures: int, n_probes: int, delta: float,
                     n_tests: int, channel: str = "") -> ChannelEnvelope:
    """U_j(delta) = beta_hat + sqrt(log(n_tests/delta)/(2n)). Undefined n -> None."""
    if n_probes <= 0:
        return ChannelEnvelope(channel, 0, failures, None, float("inf"), None)
    beta = failures / n_probes
    hw = hoeffding_halfwidth(n_probes, delta, n_tests)
    return ChannelEnvelope(channel, n_probes, failures, beta, hw, min(1.0, beta + hw))


@dataclass
class StratumProbe:
    stratum_id: str
    pi_h: float
    probes: Dict[str, int] = field(default_factory=dict)     # channel -> n
    failures: Dict[str, int] = field(default_factory=dict)   # channel -> k


@dataclass
class AuditEnvelope:
    b_cov: Optional[float]
    pi_unc: float
    eps_tax_cov: Optional[float]
    contract_bad_bound: Optional[float]
    eps_src: Optional[float]
    bad_accept_bound: Optional[float]
    per_stratum: List[dict] = field(default_factory=list)
    delta: float = 0.05
    clipped_strata: int = 0

    def to_dict(self) -> dict:
        return vars(self)


def stratified_envelope(strata: Sequence[StratumProbe], delta: float,
                        eps_tax_cov: Optional[float] = None,
                        eps_src: Optional[float] = None) -> AuditEnvelope:
    """Compose B_cov with a simultaneous budget split across strata x channels.

    pi_unc = 1 - sum of covered masses, charged ONCE. Inner sum is clipped at 1
    because on any stratum the probability that *some* channel fires is <= 1.
    """
    covered = [s for s in strata if s.pi_h > 0]
    n_tests = max(1, len(covered) * len(CHANNELS))
    pi_cov = sum(s.pi_h for s in covered)
    pi_unc = max(0.0, 1.0 - pi_cov)

    total = 0.0
    rows: List[dict] = []
    clipped = 0
    ok = True
    for s in covered:
        inner = 0.0
        detail = {}
        for c in CHANNELS:
            key = CHANNEL_FIELD[c]
            n = s.probes.get(key, 0)
            k = s.failures.get(key, 0)
            env = channel_envelope(k, n, delta, n_tests, c.value)
            if env.upper is None:
                ok = False
                detail[key] = None
                continue
            inner += env.upper
            detail[key] = env.upper
        clipped_here = inner > 1.0
        if clipped_here:
            clipped += 1
        contrib = s.pi_h * min(1.0, inner)
        total += contrib
        rows.append({"stratum_id": s.stratum_id, "pi_h": s.pi_h,
                     "inner_sum": inner, "clipped": clipped_here,
                     "contribution": contrib, "channels": detail})

    b_cov = total if ok else None
    cb = None
    if b_cov is not None and eps_tax_cov is not None:
        cb = min(1.0, b_cov + pi_unc + eps_tax_cov)
    ba = None
    if cb is not None and eps_src is not None:
        ba = min(1.0, cb + eps_src)
    return AuditEnvelope(b_cov, pi_unc, eps_tax_cov, cb, eps_src, ba, rows, delta, clipped)


def pooled_envelope(failures: Dict[str, int], probes: Dict[str, int],
                    delta: float) -> Optional[float]:
    """Pooled special case: sum_j U_j(delta) with |J| tests."""
    tot = 0.0
    for c in CHANNELS:
        key = CHANNEL_FIELD[c]
        env = channel_envelope(failures.get(key, 0), probes.get(key, 0), delta, len(CHANNELS), c.value)
        if env.upper is None:
            return None
        tot += env.upper
    return tot


# ------------------------------------------------------------------ union slack
def union_slack(records: Iterable[dict]) -> dict:
    """Lambda_union = E[(N_F - 1)_+] plus the components A11 must report separately."""
    n = 0
    sum_excess = 0
    any_fire = 0
    per_channel = {CHANNEL_FIELD[c]: 0 for c in CHANNELS}
    multi = 0
    for r in records:
        n += 1
        fired = [CHANNEL_FIELD[c] for c in CHANNELS if bool(r.get(CHANNEL_FIELD[c], False))]
        nf = len(fired)
        for f in fired:
            per_channel[f] += 1
        if nf >= 1:
            any_fire += 1
        if nf >= 2:
            multi += 1
        sum_excess += max(0, nf - 1)
    if n == 0:
        return {"n": 0, "lambda_union": None, "sum_marginals": None,
                "pr_union": None, "multi_channel_rate": None, "per_channel": per_channel}
    sum_marg = sum(per_channel.values()) / n
    pr_union = any_fire / n
    return {
        "n": n,
        "lambda_union": sum_excess / n,          # == E[(N_F-1)_+]
        "sum_marginals": sum_marg,
        "pr_union": pr_union,
        "identity_residual": abs((sum_marg - pr_union) - (sum_excess / n)),
        "multi_channel_rate": multi / n,
        "per_channel": per_channel,
    }


def eps_tax_challenge(records: Iterable[dict]) -> dict:
    """eps_tax^chal: ContractBad events on the challenge suite that NO channel caught.

    This is a challenge-set incompleteness DIAGNOSTIC, not a deployment estimate:
    a non-zero value constructs a counterexample to universal coverage; a zero
    value does not prove deployment completeness.
    """
    n = miss = bad = 0
    for r in records:
        n += 1
        if not bool(r.get("contract_bad", False)):
            continue
        bad += 1
        if not any(bool(r.get(CHANNEL_FIELD[c], False)) for c in CHANNELS):
            miss += 1
    return {"n": n, "contract_bad": bad, "unclassified": miss,
            "eps_tax_chal": (miss / n) if n else None,
            "rate_within_bad": (miss / bad) if bad else None,
            "interpretation": "challenge-set alarm; not a deployment upper bound"}
