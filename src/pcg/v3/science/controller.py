"""Selective controller: A17-A (fixed-model regret bound) vs A17-B (misspecification).

Actions: Answer / Verify / Escalate / Refuse. One-step scores are affine in the
risk r, so the induced policy is piecewise-threshold. Substituting r_hat for r
costs at most 2*L_ctrl*eps_cal one-step regret UNDER THE FIXED DECLARED MODEL.

A17-B perturbs the model itself. There, regret is only defined relative to the
per-theta oracle (optimal for that declared candidate objective), never against
a world-truth optimum.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

ACTIONS = ("Answer", "Verify", "Escalate", "Refuse")


@dataclass(frozen=True)
class CostModel:
    """theta: the declared one-step cost/risk model."""
    lam: float = 1.0                       # risk weight
    L_max: float = 1.0                     # harm scale
    # Defaults are FROZEN IN spec.json for any real run. They are calibrated so that
    # all four actions are optimal on some risk interval; a model in which Verify or
    # Escalate is never optimal makes the controller experiment vacuous.
    c_lat: Dict[str, float] = field(default_factory=lambda: {"Answer": 0.000, "Verify": 0.075, "Escalate": 0.150, "Refuse": 0.225})
    c_tok: Dict[str, float] = field(default_factory=lambda: {"Answer": 0.000, "Verify": 0.035, "Escalate": 0.085, "Refuse": 0.120})
    c_tool: Dict[str, float] = field(default_factory=lambda: {"Answer": 0.000, "Verify": 0.015, "Escalate": 0.040, "Refuse": 0.080})
    eta: Dict[str, float] = field(default_factory=lambda: {"Answer": 1.00, "Verify": 0.50, "Escalate": 0.20, "Refuse": 0.00})

    def base_cost(self, a: str) -> float:
        return self.c_lat[a] + self.c_tok[a] + self.c_tool[a]

    def Q(self, a: str, r: float) -> float:
        """Q(a; r) = Cbar(a) + lam * L_max * eta_a * r   (affine in r)."""
        return self.base_cost(a) + self.lam * self.L_max * self.eta[a] * r

    def policy(self, r: float) -> str:
        return min(ACTIONS, key=lambda a: (self.Q(a, r), ACTIONS.index(a)))

    def L_ctrl(self) -> float:
        """Lipschitz constant of the action score in r."""
        return self.lam * self.L_max * max(abs(v) for v in self.eta.values())

    def to_dict(self) -> dict:
        return {"lam": self.lam, "L_max": self.L_max, "c_lat": self.c_lat,
                "c_tok": self.c_tok, "c_tool": self.c_tool, "eta": self.eta}


def thresholds(model: CostModel, lo: float = 0.0, hi: float = 1.0, steps: int = 20001) -> List[Tuple[float, str]]:
    """Piecewise-threshold structure of the induced policy over r."""
    out: List[Tuple[float, str]] = []
    prev = None
    for i in range(steps):
        r = lo + (hi - lo) * i / (steps - 1)
        a = model.policy(r)
        if a != prev:
            out.append((r, a))
            prev = a
    return out


def regret_fixed_model(model: CostModel, r_true: Sequence[float], r_hat: Sequence[float]) -> Dict[str, float]:
    """A17-A: empirical one-step regret against the analytic 2*L_ctrl*eps_cal bound."""
    if len(r_true) != len(r_hat):
        raise ValueError("length mismatch")
    eps_cal = max(abs(a - b) for a, b in zip(r_true, r_hat)) if r_true else 0.0
    worst = 0.0
    tot = 0.0
    switches = 0
    for rt, rh in zip(r_true, r_hat):
        a_hat = model.policy(rh)
        a_star = model.policy(rt)
        reg = model.Q(a_hat, rt) - model.Q(a_star, rt)
        worst = max(worst, reg)
        tot += reg
        switches += int(a_hat != a_star)
    n = max(1, len(r_true))
    bound = 2.0 * model.L_ctrl() * eps_cal
    return {"eps_cal": eps_cal, "regret_max": worst, "regret_mean": tot / n,
            "switch_rate": switches / n, "analytic_bound": bound,
            "bound_respected": worst <= bound + 1e-12}


def sensitivity(models: Dict[str, CostModel], nominal: str,
                r_hat: Sequence[float]) -> Dict[str, object]:
    """A17-B: model-relative regret against the PER-THETA oracle.

    R_max = max_theta [ J_theta(pi_theta*) - J_theta(pi_nominal) ], reported
    alongside realized outcomes and policy-switch rate. The per-theta oracle is
    optimal only for that declared candidate objective, not a world-truth oracle.
    """
    if nominal not in models:
        raise ValueError("nominal model must be in models")
    pol_nom = [models[nominal].policy(r) for r in r_hat]
    rows = []
    r_max = 0.0
    for name, m in models.items():
        j_nom = sum(m.Q(a, r) for a, r in zip(pol_nom, r_hat))
        j_orc = sum(min(m.Q(a, r) for a in ACTIONS) for r in r_hat)
        n = max(1, len(r_hat))
        reg = (j_nom - j_orc) / n
        r_max = max(r_max, reg)
        pol_theta = [m.policy(r) for r in r_hat]
        switch = sum(1 for a, b in zip(pol_nom, pol_theta) if a != b) / n
        dist = {a: pol_nom.count(a) / n for a in ACTIONS}
        rows.append({"theta": name, "model_relative_regret": reg,
                     "switch_rate_vs_nominal": switch,
                     "J_nominal": j_nom / n, "J_oracle": j_orc / n,
                     "nominal_action_distribution": dist})
    return {"nominal": nominal, "R_max": r_max,
            "R_avg": sum(r["model_relative_regret"] for r in rows) / max(1, len(rows)),
            "per_theta": rows,
            "semantics": "regret is relative to the per-theta oracle for that declared objective"}
