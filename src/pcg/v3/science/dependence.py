"""Branch dependence: lambda_I, rho_I, the three-state gate, and U_joint.

v3.0 definitions (do not simplify):

    lambda_[k] = Pr(cap_i E_i) / prod_i p_i = (dP/dQ)(1,...,1) <= exp(D_inf(P||Q))
    rho_[k]    = max{1, lambda_[k]^(1/(k-1))}

    rho_UCB    = max_{|I|>=q0} max{1, (p_I^+ / prod_{i in I} p_i^-)^(1/(|I|-1))}
                 with p_i^- == 0  =>  rho_UCB = +inf     (fail closed)

Because Clopper-Pearson's lower limit is EXACTLY zero when zero failures are
observed, a two-state gate is degenerate in the low-failure regime the paper
targets. The operative status is therefore three-state:

    OPEN | CLOSED | INSUFFICIENT_EVIDENCE

Redundancy extrapolation is credited only for OPEN. When rho is not estimable,
U_joint(k, delta) -- a direct binomial upper bound on the observed all-branch
co-failure rate -- may support that exact k-configuration but never licenses
extrapolation to another k.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..stats.intervals import clopper_pearson_lower, clopper_pearson_upper


class GateState(str, Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True)
class ImplementationEvidenceFloors:
    """Implementation guards frozen before evaluation.

    Separate from scientific sufficiency definition.
    """
    n_min: int = 200          # trials required per branch
    k_min: int = 5            # observed failures required per branch
    q0: int = 2               # smallest subset size entering the lattice maximum

    def to_dict(self) -> dict:
        return {"n_min": self.n_min, "k_min": self.k_min, "q0": self.q0}


# Backward-compatible alias
EvidenceFloor = ImplementationEvidenceFloors


@dataclass(frozen=True)
class PrecisionRequirement:
    """Scientific statistical precision condition on rho_hat / rho_UCB."""
    max_ci_width: float = 0.50     # maximum allowed width (rho_ucb - rho_k)
    max_rel_width: float = 0.40    # maximum allowed relative width (rho_ucb - rho_k) / rho_k
    enforce_precision: bool = False # enabled for pre-registered Gate-0.1 validation

    def to_dict(self) -> dict:
        return {"max_ci_width": self.max_ci_width, "max_rel_width": self.max_rel_width,
                "enforce_precision": self.enforce_precision}


@dataclass
class DependenceResult:
    k: int
    n_trials: int
    lambda_k: Optional[float]
    rho_k: Optional[float]
    rho_ucb: Optional[float]
    state: GateState
    u_joint: Optional[float]
    subsets_evaluated: int
    subsets_insufficient: int
    bar_rho: Optional[float] = None
    delta_tol: Optional[float] = None
    precision_passed: Optional[bool] = None
    detail: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {k: v for k, v in vars(self).items() if k != "detail"}
        d["state"] = self.state.value
        d["detail"] = self.detail
        return d


def lambda_all_fail(joint_count: int, marginal_counts: Sequence[int], n: int) -> Optional[float]:
    """Plug-in lambda_[k] = Pr(cap E_i) / prod p_i. None when undefined."""
    if n <= 0:
        return None
    if any(c == 0 for c in marginal_counts):
        return None                      # denominator zero -> undefined, never 0.0
    num = joint_count / n
    den = 1.0
    for c in marginal_counts:
        den *= c / n
    if den <= 0.0:
        return None
    return num / den


def rho_from_lambda(lam: Optional[float], k: int) -> Optional[float]:
    """rho_[k] = max{1, lambda^(1/(k-1))}. Clamp is part of the definition."""
    if lam is None or k < 2:
        return None
    if lam <= 0.0:
        return 1.0
    return max(1.0, lam ** (1.0 / (k - 1)))


def renyi_infinity_bound(joint: Dict[Tuple[int, ...], int], n: int) -> Optional[float]:
    """exp(D_inf(P||Q)) from a fully populated 2^k contingency table (A12, secondary)."""
    if n <= 0 or not joint:
        return None
    k = len(next(iter(joint)))
    marg = [sum(c for v, c in joint.items() if v[i] == 1) / n for i in range(k)]
    if any(m <= 0.0 or m >= 1.0 for m in marg):
        return None
    worst = 0.0
    for vec, cnt in joint.items():
        p = cnt / n
        if p <= 0.0:
            continue
        q = 1.0
        for i, b in enumerate(vec):
            q *= marg[i] if b == 1 else (1.0 - marg[i])
        if q <= 0.0:
            return None
        worst = max(worst, p / q)
    return worst


def u_joint(k_obs: int, n: int, delta: float = 0.05) -> Optional[float]:
    """Direct binomial UCB on the all-branch co-failure rate for THIS k only."""
    if n <= 0:
        return None
    return clopper_pearson_upper(k_obs, n, delta)


def rho_ucb(branch_failures: Sequence[Sequence[bool]], delta: float = 0.05,
            floor: ImplementationEvidenceFloors = ImplementationEvidenceFloors(),
            precision: PrecisionRequirement = PrecisionRequirement(),
            bar_rho: Optional[float] = None,
            delta_tol: float = 0.0) -> DependenceResult:
    """Compute rho_UCB over the upper set of the subset lattice, with a three-state gate.

    branch_failures: n rows of k booleans (True == that branch accepted a false claim).
    """
    n = len(branch_failures)
    if n == 0:
        return DependenceResult(0, 0, None, None, None, GateState.INSUFFICIENT_EVIDENCE,
                                None, 0, 0, bar_rho, delta_tol, None)
    k = len(branch_failures[0])
    if any(len(r) != k for r in branch_failures):
        raise ValueError("ragged branch_failures matrix")

    marg_counts = [sum(1 for r in branch_failures if r[i]) for i in range(k)]
    joint_all = sum(1 for r in branch_failures if all(r))

    best: Optional[float] = None
    evaluated = insufficient = 0
    saw_zero_lower = False

    for size in range(max(2, floor.q0), k + 1):
        for I in itertools.combinations(range(k), size):
            evaluated += 1
            # implementation evidence floor: frozen before evaluation
            if n < floor.n_min or any(marg_counts[i] < floor.k_min for i in I):
                insufficient += 1
                continue
            k_I = sum(1 for r in branch_failures if all(r[i] for i in I))
            p_I_up = clopper_pearson_upper(k_I, n, delta)
            den = 1.0
            zero = False
            for i in I:
                lo = clopper_pearson_lower(marg_counts[i], n, delta)
                if lo <= 0.0:
                    zero = True
                    break
                den *= lo
            if zero:
                saw_zero_lower = True
                best = math.inf           # fail closed
                continue
            ratio = p_I_up / den
            val = max(1.0, ratio ** (1.0 / (size - 1)))
            best = val if best is None else max(best, val)

    lam = lambda_all_fail(joint_all, marg_counts, n)
    rho = rho_from_lambda(lam, k)
    uj = u_joint(joint_all, n, delta)

    precision_passed: Optional[bool] = None
    if best is not None and not math.isinf(best) and rho is not None:
        ci_w = best - rho
        rel_w = (best - rho) / rho if rho > 0 else math.inf
        precision_passed = (ci_w <= precision.max_ci_width and rel_w <= precision.max_rel_width)

    if best is None:
        state = GateState.INSUFFICIENT_EVIDENCE
    elif precision.enforce_precision and precision_passed is False:
        state = GateState.INSUFFICIENT_EVIDENCE
    elif math.isinf(best):
        state = GateState.CLOSED
    elif bar_rho is None:
        state = GateState.OPEN
    else:
        state = GateState.OPEN if best <= bar_rho + delta_tol else GateState.CLOSED

    return DependenceResult(
        k=k, n_trials=n, lambda_k=lam, rho_k=rho,
        rho_ucb=(None if best is None else best),
        state=state, u_joint=uj,
        subsets_evaluated=evaluated, subsets_insufficient=insufficient,
        bar_rho=bar_rho, delta_tol=delta_tol,
        precision_passed=precision_passed,
        detail={"joint_all_fail": joint_all, "marginal_counts": marg_counts,
                "saw_zero_lower_bound": saw_zero_lower},
    )


def common_mode_floor(q_cm: float, eps_path: float) -> Dict[str, float]:
    """Pr(cap E_i) >= q_cm for every k; k* = ceil(log q_cm / log eps_path).

    Beyond k*, the independent-coupling target lies below the floor. Actual
    false acceptance may still decrease toward the floor -- monotone decrease in
    k is therefore NOT a refutation; only a value below q_cm would be.
    """
    if not (0.0 < q_cm < 1.0) or not (0.0 < eps_path < 1.0):
        raise ValueError("q_cm and eps_path must lie in (0,1)")
    k_star = math.ceil(math.log(q_cm) / math.log(eps_path))
    return {"q_cm": q_cm, "eps_path": eps_path, "k_star": float(k_star),
            "floor": q_cm}
