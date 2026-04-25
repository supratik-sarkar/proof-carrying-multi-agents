"""
Risk-aware control (Theorem 3 part (ii)).

Provides:
    - `Calibrator`: isotonic and Platt calibrators with held-out ECE estimation
    - `ThresholdPolicy`: piecewise-threshold policy over {answer, verify,
                         escalate, refuse} implementing the affine-cost-in-r
                         structure of Appendix A.5 / C.4
    - `LearnedPolicy`: contextual policy trained on logged rollouts (used as a
                       baseline comparator in R4)
    - `posterior_risk`: the r(b, Z) estimator from Eq. (24)
    - `expected_cost`: the C(b, a) model from Eq. (22)

IMPORTANT THEORY FIX: The paper's Eq. (30) presents the refusal threshold as
    tau = Delta_C_bar / (lambda * L_max)
which disagrees with the full derivation in Appendix B.3 (Eq. 65):
    tau = (Delta_C + lambda * H_Ref) / (lambda * H_FA)

The implementation below uses the full derivation. The main-text formula
becomes the special case H_Ref=0, H_FA=L_max. We expose both parameterizations
so the paper and code can be trivially reconciled during the camera-ready.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Sequence

import numpy as np


# -----------------------------------------------------------------------------
# Action enum
# -----------------------------------------------------------------------------


class Action(str, Enum):
    ANSWER = "answer"
    VERIFY = "verify"
    ESCALATE = "escalate"
    REFUSE = "refuse"


# -----------------------------------------------------------------------------
# Cost model  (Eq. 22)
# -----------------------------------------------------------------------------


@dataclass
class CostModel:
    """C(b, a) = C_lat(a) + C_tok(a) + C_tool(a) + lambda * E_theta[L_harm(theta, a)].

    Uses the affine-in-r harm parameterization from Appendix C.4, Eq. (93):

        E[L_harm | a] =
            H_FA * eta_a * r          if a in {Answer, Verify, Escalate}
            H_Ref                     if a == Refuse

    with post-action residual-risk multipliers eta satisfying
        eta_Answer = 1 >= eta_Verify >= eta_Escalate > 0.
    """

    # Non-harm cost components per action (latency + tokens + tool calls)
    c_lat: dict[Action, float] = field(default_factory=dict)
    c_tok: dict[Action, float] = field(default_factory=dict)
    c_tool: dict[Action, float] = field(default_factory=dict)
    # Harm params
    lam: float = 1.0                      # lambda, risk weight
    h_fa: float = 1.0                     # H_FA, harm scale for a false accept
    h_ref: float = 0.0                    # H_Ref, opportunity cost of refusal
    # Residual risk multipliers
    eta: dict[Action, float] = field(default_factory=lambda: {
        Action.ANSWER: 1.0,
        Action.VERIFY: 0.5,
        Action.ESCALATE: 0.1,
        Action.REFUSE: 0.0,   # refusal does not carry residual false-accept risk
    })

    def nonharm(self, a: Action) -> float:
        return (
            self.c_lat.get(a, 0.0)
            + self.c_tok.get(a, 0.0)
            + self.c_tool.get(a, 0.0)
        )

    def cost(self, a: Action, r: float) -> float:
        """C(b, a) evaluated with posterior false-accept risk r."""
        if a == Action.REFUSE:
            return self.nonharm(a) + self.lam * self.h_ref
        eta_a = self.eta.get(a, 1.0)
        return self.nonharm(a) + self.lam * self.h_fa * eta_a * r


# -----------------------------------------------------------------------------
# Posterior false-accept risk (Eq. 24)
# -----------------------------------------------------------------------------


def posterior_risk(
    confidences: Sequence[float],
    pass_flags: Sequence[bool],
    rho: float = 1.0,
) -> float:
    """A simple instantiation of r(b, Z): product of per-branch failure
    probabilities, inflated by rho^{k-1} following Assumption 3 / Eq. (50):

        r ~ rho^{k-1} * prod_i (1 - confidence_i) * I[branch_i passed]

    Branches that did not pass contribute 1.0 (maximally risky) to the product.
    The confidence is assumed already calibrated (see Calibrator). This is a
    strictly monotone function of certificate strength, so Assumption 3.5 holds.
    """
    if len(confidences) != len(pass_flags):
        raise ValueError("confidences and pass_flags must have equal length")
    if not confidences:
        return 1.0
    prod = 1.0
    for p, ok in zip(confidences, pass_flags):
        if not ok:
            prod *= 1.0
        else:
            prod *= max(0.0, 1.0 - p)
    k = len(confidences)
    return min(1.0, float(rho) ** max(0, k - 1) * prod)


# -----------------------------------------------------------------------------
# Calibration (Assumption 3.4 with both expectation and pointwise semantics)
# -----------------------------------------------------------------------------


@dataclass
class CalibrationReport:
    ece: float                            # binned ECE (Eq. 49)
    pointwise_ece_ucb: float              # upper bound on pointwise error (new)
    n_bins: int
    n: int


def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Binned ECE from Eq. (49) of the paper."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for b in range(n_bins):
        mask = (probs >= bins[b]) & (probs < bins[b + 1]) if b < n_bins - 1 \
            else (probs >= bins[b]) & (probs <= bins[b + 1])
        if not mask.any():
            continue
        bin_conf = float(probs[mask].mean())
        bin_acc = float(labels[mask].mean())
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return float(ece)


def pointwise_calibration_ucb(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
    alpha: float = 0.05,
) -> float:
    """Upper confidence bound on pointwise calibration error.

    For each bin, we compute the Wilson-score upper bound on |accuracy -
    mean(conf)| at level 1 - alpha, then report the max across bins. This is
    the quantity that actually matters for Theorem 3(ii)'s O(eps_cal) regret
    bound, where the paper (as written) only bounds the bin-averaged ECE.
    """
    z = 1.96 if alpha == 0.05 else math.sqrt(2.0) * _erfinv(1.0 - alpha)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    max_err = 0.0
    for b in range(n_bins):
        mask = (probs >= bins[b]) & (probs < bins[b + 1]) if b < n_bins - 1 \
            else (probs >= bins[b]) & (probs <= bins[b + 1])
        if not mask.any():
            continue
        n_b = int(mask.sum())
        p_hat = float(labels[mask].mean())
        # Wilson interval half-width for p_hat
        denom = 1.0 + z * z / n_b
        center = (p_hat + z * z / (2 * n_b)) / denom
        half = z * math.sqrt(p_hat * (1 - p_hat) / n_b + z * z / (4 * n_b * n_b)) / denom
        bin_conf = float(probs[mask].mean())
        # UCB on |conf - true_acc|
        err = max(abs(bin_conf - (center - half)), abs(bin_conf - (center + half)))
        max_err = max(max_err, err)
    return max_err


def _erfinv(x: float) -> float:
    """Inverse error function for the rare case when scipy is not installed."""
    # Winitzki approximation; good to ~4e-3
    a = 0.147
    ln = math.log(1 - x * x)
    t = 2.0 / (math.pi * a) + ln / 2.0
    return math.copysign(math.sqrt(math.sqrt(t * t - ln / a) - t), x)


class Calibrator:
    """Wraps an isotonic regressor (sklearn) for post-hoc calibration.

    Falls back to Platt scaling if sklearn is unavailable.
    """

    def __init__(self, method: str = "isotonic") -> None:
        self.method = method
        self._model = None

    def fit(self, raw_probs: np.ndarray, labels: np.ndarray) -> "Calibrator":
        if self.method == "isotonic":
            try:
                from sklearn.isotonic import IsotonicRegression
                self._model = IsotonicRegression(out_of_bounds="clip")
                self._model.fit(raw_probs, labels)
                return self
            except ImportError:
                pass
        # Platt scaling via logistic regression, or manual fit if sklearn absent.
        try:
            from sklearn.linear_model import LogisticRegression
            lr = LogisticRegression()
            lr.fit(raw_probs.reshape(-1, 1), labels)
            self._model = lr
            self.method = "platt"
        except ImportError:   # pragma: no cover
            # Manual Platt scaling via Newton's method would go here; we raise
            # instead since sklearn is a listed dependency.
            raise
        return self

    def transform(self, raw_probs: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Calibrator.fit(...) must be called first")
        if self.method == "isotonic":
            return np.asarray(self._model.transform(raw_probs))
        # Platt
        proba = self._model.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
        return np.asarray(proba)

    def report(
        self,
        raw_probs: np.ndarray,
        labels: np.ndarray,
        n_bins: int = 10,
    ) -> CalibrationReport:
        calibrated = self.transform(raw_probs)
        return CalibrationReport(
            ece=expected_calibration_error(calibrated, labels, n_bins),
            pointwise_ece_ucb=pointwise_calibration_ucb(calibrated, labels, n_bins),
            n_bins=n_bins,
            n=len(raw_probs),
        )


# -----------------------------------------------------------------------------
# Piecewise-threshold control (Theorem 3(ii), Appendix C.4)
# -----------------------------------------------------------------------------


@dataclass
class ThresholdPolicy:
    """Piecewise-threshold policy over {Answer, Verify, Escalate, Refuse}.

    Given a `CostModel`, we compute the lower-envelope of the four affine-in-r
    lines and select the argmin for each r in [0, 1]. The boundaries between
    regimes are the `taus` returned by `thresholds()`.

    The binary Answer-vs-Refuse subproblem has the closed-form threshold:
        tau_AR = (Delta_C + lambda * H_Ref) / (lambda * H_FA)
    matching Eq. (65) of the paper (NOT the shorter main-text Eq. 30, which
    is a special case).
    """

    cost_model: CostModel
    actions: tuple[Action, ...] = (Action.ANSWER, Action.VERIFY, Action.ESCALATE, Action.REFUSE)

    def action_cost_line(self, a: Action) -> tuple[float, float]:
        """Return (intercept, slope) for C(a, r) = intercept + slope * r."""
        if a == Action.REFUSE:
            return (self.cost_model.nonharm(a) + self.cost_model.lam * self.cost_model.h_ref, 0.0)
        slope = self.cost_model.lam * self.cost_model.h_fa * self.cost_model.eta.get(a, 1.0)
        return (self.cost_model.nonharm(a), slope)

    def choose(self, r: float) -> Action:
        """arg min_a C(a, r). Ties broken by action preference order."""
        best_a = self.actions[0]
        best_c = math.inf
        for a in self.actions:
            intercept, slope = self.action_cost_line(a)
            c = intercept + slope * r
            if c < best_c - 1e-12:
                best_c = c
                best_a = a
        return best_a

    def thresholds(self) -> list[tuple[float, Action, Action]]:
        """Return a sorted list of crossing points (tau, a_below, a_above).

        Useful for plotting and for debugging Theorem 3(ii) predictions.
        """
        crossings: list[tuple[float, Action, Action]] = []
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


@dataclass
class LearnedPolicy:
    """A logistic-contextual policy over actions, for use as the R4 baseline
    comparator.

    The features are expected to include calibrated risk r, redundancy level
    k, branch disagreement, and verifier margin. Fit on logged rollouts with
    off-policy evaluation (inverse propensity score), but here we implement
    the forward-fit variant since our experiments use uniformly random
    baseline policies for the log collection.
    """

    feature_fn: Callable[[dict], np.ndarray]
    _model: object | None = None

    def fit(self, contexts: list[dict], actions: list[Action], rewards: np.ndarray) -> "LearnedPolicy":
        # Supervised surrogate: predict the action that minimized cost in each log entry.
        from sklearn.linear_model import LogisticRegression
        X = np.vstack([self.feature_fn(c) for c in contexts])
        y = np.asarray([list(Action).index(a) for a in actions])
        sample_weight = np.exp(-rewards)   # cost -> weight
        self._model = LogisticRegression(max_iter=1000).fit(X, y, sample_weight=sample_weight)
        return self

    def choose(self, context: dict) -> Action:
        if self._model is None:
            raise RuntimeError("LearnedPolicy.fit(...) must be called first")
        X = self.feature_fn(context).reshape(1, -1)
        idx = int(self._model.predict(X)[0])     # type: ignore[attr-defined]
        return list(Action)[idx]
