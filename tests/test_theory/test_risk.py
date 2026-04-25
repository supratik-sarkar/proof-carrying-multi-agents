"""
Property tests for the risk + responsibility layers.

Invariants:
    1. ThresholdPolicy.choose is monotone: as r increases, action shifts
       toward Refuse (in cost-induced order).
    2. The binary refuse threshold matches the closed form
           tau = (Delta_C + lambda * H_Ref) / (lambda * H_FA)
    3. Hoeffding CI half-widths shrink as 1/sqrt(M).
    4. rank_recovery_prob is monotone in M and in margin.
"""
from __future__ import annotations

import math

import pytest

from pcg.responsibility import (
    hoeffding_halfwidth,
    rank_recovery_prob,
    required_replays_for_rank,
)
from pcg.risk import (
    Action,
    CostModel,
    ThresholdPolicy,
)


# ---------------------------------------------------------------------------
# Threshold policy properties
# ---------------------------------------------------------------------------


def test_threshold_policy_choose_monotone():
    cm = CostModel(
        c_lat={Action.ANSWER: 1.0, Action.VERIFY: 5.0,
               Action.ESCALATE: 50.0, Action.REFUSE: 0.0},
        c_tok={a: 0.0 for a in Action},
        c_tool={a: 0.0 for a in Action},
        lam=10.0, h_fa=10.0, h_ref=0.5,
    )
    policy = ThresholdPolicy(cost_model=cm)
    chosen_seq = [policy.choose(r) for r in [0.0, 0.05, 0.1, 0.3, 0.6, 0.9, 1.0]]
    # The chosen action should never go BACKWARDS toward Answer as r grows.
    rank = {Action.ANSWER: 0, Action.VERIFY: 1, Action.ESCALATE: 2, Action.REFUSE: 3}
    ranks = [rank[a] for a in chosen_seq]
    assert ranks == sorted(ranks), f"non-monotone: {ranks}"


def test_binary_threshold_closed_form():
    """Binary Answer-vs-Refuse subproblem with verify/escalate disabled."""
    # Construct a cost model where only Answer and Refuse are "cheap enough"
    # to ever be selected. Set Verify and Escalate to absurd nonharm costs.
    cm = CostModel(
        c_lat={Action.ANSWER: 1.0, Action.VERIFY: 1e6,
               Action.ESCALATE: 1e6, Action.REFUSE: 0.0},
        c_tok={a: 0.0 for a in Action},
        c_tool={a: 0.0 for a in Action},
        lam=1.0, h_fa=10.0, h_ref=0.0,
    )
    policy = ThresholdPolicy(cost_model=cm)
    # Closed-form tau:
    #   Cost(Answer, r) = nonharm(Answer) + lambda * h_fa * eta_Answer * r
    #                   = 1.0 + 10 * r
    #   Cost(Refuse, r) = nonharm(Refuse) + lambda * h_ref
    #                   = 0.0 + 0
    # Crossover: 1.0 + 10r = 0  =>  r = -0.1.   Negative => Answer always wins.
    # So in this configuration Refuse only wins if h_ref < nonharm(Answer)/lambda.
    # Let's flip: make nonharm(Answer) = 0 and h_ref = 0.05.
    cm2 = CostModel(
        c_lat={Action.ANSWER: 0.0, Action.VERIFY: 1e6,
               Action.ESCALATE: 1e6, Action.REFUSE: 0.0},
        c_tok={a: 0.0 for a in Action},
        c_tool={a: 0.0 for a in Action},
        lam=1.0, h_fa=10.0, h_ref=0.05,
    )
    policy2 = ThresholdPolicy(cost_model=cm2)
    expected_tau = (0.0 - 0.0 + 1.0 * 0.05) / (1.0 * 10.0 * 1.0)   # = 0.005
    # Test bracketing
    assert policy2.choose(expected_tau - 0.001) == Action.ANSWER
    assert policy2.choose(expected_tau + 0.001) == Action.REFUSE


# ---------------------------------------------------------------------------
# Hoeffding properties
# ---------------------------------------------------------------------------


def test_hoeffding_halfwidth_shrinks_with_n():
    hw_100 = hoeffding_halfwidth(100, alpha=0.05, range_bound=2.0)
    hw_400 = hoeffding_halfwidth(400, alpha=0.05, range_bound=2.0)
    # 4x samples -> 2x narrower
    assert hw_100 / hw_400 == pytest.approx(2.0, rel=1e-9)


def test_hoeffding_halfwidth_increases_with_confidence():
    hw_05 = hoeffding_halfwidth(100, alpha=0.05, range_bound=2.0)
    hw_01 = hoeffding_halfwidth(100, alpha=0.01, range_bound=2.0)
    # Higher confidence (lower alpha) => wider CI
    assert hw_01 > hw_05


def test_rank_recovery_prob_monotone_in_M():
    p1 = rank_recovery_prob(n_replays=50, n_components=10, margin=0.3)
    p2 = rank_recovery_prob(n_replays=200, n_components=10, margin=0.3)
    assert p2 >= p1


def test_rank_recovery_prob_monotone_in_margin():
    p1 = rank_recovery_prob(n_replays=100, n_components=10, margin=0.1)
    p2 = rank_recovery_prob(n_replays=100, n_components=10, margin=0.5)
    assert p2 > p1


def test_required_replays_inverse_square_in_margin():
    """At fixed target prob, M scales like 1/margin^2."""
    n = 10
    target = 0.95
    m_05 = required_replays_for_rank(n_components=n, margin=0.5, target_prob=target)
    m_025 = required_replays_for_rank(n_components=n, margin=0.25, target_prob=target)
    # Halving margin should ~quadruple required samples
    ratio = m_025 / max(m_05, 1)
    assert 3.5 <= ratio <= 4.5, f"Expected ~4x scaling, got {ratio}"
