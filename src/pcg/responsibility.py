"""
Interventional responsibility (Definition 2.12, Eq. 7) and Theorem 3(i).

    Resp(e; Z, G_t) = E_xi [A(Z, G_t) - A(Z, G_t^{\\setminus e})]

Given the Checker-based acceptance predicate A (from pcg.checker), this
module provides:

    - `intervene(...)`          : a single mask-and-replay round
    - `ResponsibilityEstimator` : Monte-Carlo estimator with configurable
                                  replay budget M and confidence level
    - `hoeffding_ci(...)`       : distribution-free CI (Thm 3(i), Eq. 27)
    - `normal_ci(...)`          : narrower CI under finite-variance CLT
    - `rank_recovery_bound(...)` : Eq. (28) of the paper for the margin gamma
    - `shapley_responsibility(...)` : optional Shapley-value alternative that
                                       addresses the identifiability concern
                                       raised in the theory review

The Shapley implementation is exact on small component sets and approximate
(stratified Monte Carlo) on larger ones. It is provided as an *alternative*
to the main estimator, not a replacement: the main Top-1 identification
claim in the paper uses the cheaper single-mask estimator.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from itertools import permutations
from typing import Callable

import numpy as np

from pcg.certificate import GroundingCertificate
from pcg.checker import Checker, CheckResult
from pcg.graph import AgenticRuntimeGraph


# -----------------------------------------------------------------------------
# Single-component intervention
# -----------------------------------------------------------------------------


def intervene(
    checker: Checker,
    cert: GroundingCertificate,
    graph: AgenticRuntimeGraph,
    component_ids: frozenset[str],
) -> int:
    """Return 1 if Check(Z; G_t^{\\setminus E}) == 1, else 0.

    `component_ids` may be a single-element set (standard responsibility) or
    a coalition (used by Shapley-value estimation).
    """
    masked = graph.mask(component_ids)
    result = checker.check(cert, masked)
    return 1 if result.passed else 0


# -----------------------------------------------------------------------------
# Monte-Carlo responsibility estimator (Eq. 22 of the paper)
# -----------------------------------------------------------------------------


@dataclass
class ResponsibilityResult:
    component_id: str
    estimate: float                    # Resp_hat
    lower_hoeffding: float             # Eq. 27 (distribution-free, default)
    upper_hoeffding: float
    lower_normal: float                # CLT-based (reported as alternative)
    upper_normal: float
    n_replays: int
    variance: float

    def to_dict(self) -> dict[str, float | str | int]:
        return {
            "component_id": self.component_id,
            "estimate": self.estimate,
            "lower_hoeffding": self.lower_hoeffding,
            "upper_hoeffding": self.upper_hoeffding,
            "lower_normal": self.lower_normal,
            "upper_normal": self.upper_normal,
            "n_replays": self.n_replays,
            "variance": self.variance,
        }


def hoeffding_halfwidth(n: int, alpha: float, range_bound: float = 2.0) -> float:
    """Hoeffding CI half-width for i.i.d. samples in [-1, 1] (range = 2).

    From Eq. (27) of the paper / Hoeffding's inequality:
        half-width = range_bound * sqrt(log(2/alpha) / (2n))

    The paper uses the special case range_bound = 2 (since Delta_e in [-1, 1]).
    """
    if n <= 0:
        return float("inf")
    return range_bound * math.sqrt(math.log(2.0 / alpha) / (2.0 * n))


def normal_halfwidth(std: float, n: int, alpha: float) -> float:
    """Normal-approximation CI half-width. Assumes iid, finite variance."""
    if n <= 0 or std == 0.0:
        return 0.0 if n > 0 else float("inf")
    # Use inverse CDF via scipy if available; fall back to a polynomial approx.
    try:
        from scipy.stats import norm
        z = float(norm.ppf(1 - alpha / 2))
    except ImportError:
        # Beasley-Springer-Moro would be overkill; use a cheap approx.
        # z_{0.975} ~= 1.96, z_{0.95} ~= 1.645, z_{0.99} ~= 2.576
        approx = {0.10: 1.645, 0.05: 1.96, 0.01: 2.576}
        z = approx.get(alpha, 1.96)
    return z * std / math.sqrt(n)


@dataclass
class ResponsibilityEstimator:
    """Monte-Carlo estimator for Resp(e; Z, G_t).

    Parameters:
        checker: the Checker to use for A(Z, G)
        n_replays: M, the number of replay rounds per component
        alpha: 1 - confidence level for the CI (default 0.05 => 95%)
        paired: if True, use common random numbers (same seed for G_t and
                G_t^{\\setminus e} within each replay); variance reduction
                trick from Appendix C.8.

    Usage:
        est = ResponsibilityEstimator(checker, n_replays=200)
        results = est.estimate_many(cert, graph, component_ids=[...])
    """

    checker: Checker
    n_replays: int = 100
    alpha: float = 0.05
    paired: bool = True
    seed: int = 0

    def estimate_one(
        self,
        cert: GroundingCertificate,
        graph: AgenticRuntimeGraph,
        component_id: str,
        reseed_fn: Callable[[int], None] | None = None,
    ) -> ResponsibilityResult:
        """Estimate Resp_hat(e) for a single component id.

        `reseed_fn(seed)` optionally sets the checker/replayer seed so that
        paired replays see the same randomness in both G_t and G_t^{\\setminus e}.
        """
        rng = random.Random(self.seed ^ hash(component_id) & 0xFFFFFFFF)
        deltas: list[float] = []

        for m in range(self.n_replays):
            seed_m = rng.randint(0, 2**31 - 1)
            if self.paired and reseed_fn is not None:
                reseed_fn(seed_m)
            a_full = int(self.checker.check(cert, graph).passed)
            if self.paired and reseed_fn is not None:
                reseed_fn(seed_m)     # same seed for masked replay
            a_masked = intervene(self.checker, cert, graph, frozenset({component_id}))
            deltas.append(a_full - a_masked)

        arr = np.asarray(deltas, dtype=np.float64)
        est = float(arr.mean())
        var = float(arr.var(ddof=1)) if len(arr) > 1 else 0.0
        hw_h = hoeffding_halfwidth(self.n_replays, self.alpha, range_bound=2.0)
        hw_n = normal_halfwidth(math.sqrt(var), self.n_replays, self.alpha)

        return ResponsibilityResult(
            component_id=component_id,
            estimate=est,
            lower_hoeffding=est - hw_h,
            upper_hoeffding=est + hw_h,
            lower_normal=est - hw_n,
            upper_normal=est + hw_n,
            n_replays=self.n_replays,
            variance=var,
        )

    def estimate_many(
        self,
        cert: GroundingCertificate,
        graph: AgenticRuntimeGraph,
        component_ids: list[str],
        reseed_fn: Callable[[int], None] | None = None,
    ) -> list[ResponsibilityResult]:
        return [
            self.estimate_one(cert, graph, cid, reseed_fn=reseed_fn)
            for cid in component_ids
        ]


def rank_recovery_prob(
    n_replays: int,
    n_components: int,
    margin: float,
    alpha_family: float = 0.05,
) -> float:
    """Lower bound on P(arg max == e^star), Eq. (28) of the paper.

    Returns a *lower bound* via
        1 - 2 |E| exp(-M gamma^2 / 8)

    so a value close to 1 is a strong guarantee, a value near 0 means you
    need more replays or a larger margin.
    """
    if margin <= 0:
        return 0.0
    return max(
        0.0,
        1.0 - 2.0 * n_components * math.exp(-n_replays * margin * margin / 8.0),
    )


# -----------------------------------------------------------------------------
# Required replay budget to achieve a target rank-recovery probability
# -----------------------------------------------------------------------------


def required_replays_for_rank(
    n_components: int,
    margin: float,
    target_prob: float = 0.95,
) -> int:
    """Smallest M such that 1 - 2|E| exp(-M gamma^2 / 8) >= target_prob."""
    if margin <= 0:
        return int(10**9)
    if target_prob <= 0:
        return 1
    required = (8.0 / (margin * margin)) * math.log(2.0 * n_components / (1.0 - target_prob))
    return max(1, int(math.ceil(required)))


# -----------------------------------------------------------------------------
# Shapley-value alternative (addresses the "single-mask ignores interactions"
# concern flagged in the theory review)
# -----------------------------------------------------------------------------


@dataclass
class ShapleyResult:
    component_id: str
    value: float
    std_err: float                     # 0 for exact, >0 for MC
    n_samples: int


def shapley_responsibility(
    checker: Checker,
    cert: GroundingCertificate,
    graph: AgenticRuntimeGraph,
    components: list[str],
    mode: str = "auto",
    n_permutations: int = 200,
    seed: int = 0,
) -> list[ShapleyResult]:
    """Shapley value of each component with respect to the value function
    v(S) = 1 - A(Z, G^{\\setminus S}) (the *lost* acceptance when masking S).

    For components not in the certificate's run the Shapley value is 0
    (masking an irrelevant component cannot affect acceptance under A).

    mode:
        "exact": enumerate all permutations. O(|E|!) — use only when |E| <= 8.
        "monte_carlo": sample `n_permutations` permutations uniformly. Std err
            reported from sample variance.
        "auto": exact when |E| <= 8, else monte_carlo.
    """
    if mode == "auto":
        mode = "exact" if len(components) <= 8 else "monte_carlo"

    E = list(components)
    n = len(E)

    def value(S: frozenset[str]) -> float:
        """v(S) = 1 - A(Z, G^{\\setminus S})."""
        return 1.0 - float(intervene(checker, cert, graph, S))

    if mode == "exact":
        contributions = {c: 0.0 for c in E}
        count = 0
        for perm in permutations(E):
            prev: frozenset[str] = frozenset()
            v_prev = value(prev)
            for c in perm:
                new = prev | {c}
                v_new = value(new)
                contributions[c] += v_new - v_prev
                prev = new
                v_prev = v_new
            count += 1
        return [
            ShapleyResult(component_id=c, value=contributions[c] / count,
                          std_err=0.0, n_samples=count)
            for c in E
        ]

    # Monte-Carlo
    rng = random.Random(seed)
    per_comp: dict[str, list[float]] = {c: [] for c in E}
    for _ in range(n_permutations):
        perm = E[:]
        rng.shuffle(perm)
        prev: frozenset[str] = frozenset()
        v_prev = value(prev)
        for c in perm:
            new = prev | {c}
            v_new = value(new)
            per_comp[c].append(v_new - v_prev)
            prev = new
            v_prev = v_new
    results: list[ShapleyResult] = []
    for c in E:
        samples = np.asarray(per_comp[c], dtype=np.float64)
        mean = float(samples.mean()) if len(samples) else 0.0
        se = float(samples.std(ddof=1) / math.sqrt(len(samples))) if len(samples) > 1 else 0.0
        results.append(ShapleyResult(component_id=c, value=mean,
                                     std_err=se, n_samples=len(samples)))
    return results
