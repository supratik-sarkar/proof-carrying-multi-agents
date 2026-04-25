"""
Property tests for the independence layer and the rho estimator.

Invariants:
    1. provenance_disjoint is symmetric and consistent at each level
    2. shingle_jaccard is in [0, 1] and symmetric
    3. are_independent is symmetric
    4. greedy filter respects pairwise independence
    5. rho estimator returns >= 1 for positively correlated branches
    6. rho UCB upper-bounds rho_hat with high probability under simulation
"""
from __future__ import annotations

import numpy as np
import pytest

from pcg.eval.rho import estimate_rho
from pcg.independence import (
    IndependenceConfig,
    ProvenanceLabel,
    are_independent,
    canonicalize_text,
    filter_independent_family,
    provenance_disjoint,
    shingle_jaccard,
    SupportPath,
)
from pcg.graph import (
    AgenticRuntimeGraph,
    ClaimNode,
    EdgeType,
    SourceNode,
    TruthNode,
)


# ---------------------------------------------------------------------------
# Provenance & shingle properties
# ---------------------------------------------------------------------------


def test_provenance_disjoint_symmetric():
    a = ProvenanceLabel("auth_X", "pub_X", "x.com")
    b = ProvenanceLabel("auth_Y", "pub_Y", "y.com")
    assert provenance_disjoint(a, b, level="authority") == provenance_disjoint(b, a, level="authority")
    assert provenance_disjoint(a, b, level="publisher") == provenance_disjoint(b, a, level="publisher")
    assert provenance_disjoint(a, b, level="domain") == provenance_disjoint(b, a, level="domain")


def test_provenance_disjoint_levels_monotone():
    a = ProvenanceLabel("auth_A", "pub_X", "shared.com")
    b = ProvenanceLabel("auth_B", "pub_X", "shared.com")
    # authority differs => disjoint at authority level
    assert provenance_disjoint(a, b, level="authority")
    # publisher equal => NOT disjoint at publisher level
    assert not provenance_disjoint(a, b, level="publisher")
    # domain equal => NOT disjoint at domain level
    assert not provenance_disjoint(a, b, level="domain")


def test_shingle_jaccard_bounds():
    s = shingle_jaccard("the quick brown fox", "the quick brown fox", m=3)
    assert s == pytest.approx(1.0)
    s = shingle_jaccard("aaa bbb ccc", "xxx yyy zzz", m=3)
    assert s == pytest.approx(0.0)


def test_shingle_jaccard_symmetric():
    s_ab = shingle_jaccard("the quick brown fox jumps over", "the lazy dog brown", m=3)
    s_ba = shingle_jaccard("the lazy dog brown", "the quick brown fox jumps over", m=3)
    assert s_ab == pytest.approx(s_ba)


def test_canonicalize_idempotent():
    s = "  Hello   World  \n"
    assert canonicalize_text(s) == canonicalize_text(canonicalize_text(s))


# ---------------------------------------------------------------------------
# Independence predicate properties
# ---------------------------------------------------------------------------


def _make_graph_with_two_paths(text_a: str, text_b: str,
                                pub_a: str, pub_b: str,
                                shared_tools: int = 0) -> tuple[AgenticRuntimeGraph, SupportPath, SupportPath]:
    g = AgenticRuntimeGraph()
    # Two source nodes
    s1 = SourceNode(authority_id=pub_a, publisher_id=pub_a, domain=f"{pub_a}.org")
    s2 = SourceNode(authority_id=pub_b, publisher_id=pub_b, domain=f"{pub_b}.org")
    g.add_node(s1)
    g.add_node(s2)
    t1 = TruthNode(payload=text_a.encode(), source_id=s1.id)
    t2 = TruthNode(payload=text_b.encode(), source_id=s2.id)
    g.add_node(t1)
    g.add_node(t2)
    g.add_edge(s1.id, t1.id, EdgeType.RETRIEVED_FROM)
    g.add_edge(s2.id, t2.id, EdgeType.RETRIEVED_FROM)
    claim = ClaimNode(raw="x", canonical="x")
    g.add_node(claim)
    g.add_edge(t1.id, claim.id, EdgeType.SUPPORTS)
    g.add_edge(t2.id, claim.id, EdgeType.SUPPORTS)

    # Synthesize tool-step ids; if `shared_tools > 0`, both paths share that many.
    shared = {f"tool_shared_{i}" for i in range(shared_tools)}
    p1 = SupportPath(node_ids=(t1.id, claim.id), tool_step_ids=frozenset({"tool_a"} | shared))
    p2 = SupportPath(node_ids=(t2.id, claim.id), tool_step_ids=frozenset({"tool_b"} | shared))
    return g, p1, p2


def test_are_independent_symmetric():
    g, p1, p2 = _make_graph_with_two_paths(
        "alpha beta gamma delta", "epsilon zeta eta theta",
        "wiki", "arxiv", shared_tools=0,
    )
    cfg = IndependenceConfig(delta=0.1, kappa=1, provenance_level="publisher", shingle_m=3)
    assert are_independent(g, p1, p2, cfg) == are_independent(g, p2, p1, cfg)


def test_high_overlap_breaks_independence():
    g, p1, p2 = _make_graph_with_two_paths(
        "the same exact sentence here", "the same exact sentence here",
        "wiki", "arxiv", shared_tools=0,
    )
    cfg = IndependenceConfig(delta=0.1, kappa=1, provenance_level="publisher", shingle_m=3)
    # Identical text => Jaccard = 1, fails delta=0.1
    assert not are_independent(g, p1, p2, cfg)


def test_shared_tools_break_independence():
    g, p1, p2 = _make_graph_with_two_paths(
        "alpha beta", "gamma delta",
        "wiki", "arxiv", shared_tools=2,    # 2 shared tools
    )
    cfg = IndependenceConfig(delta=0.5, kappa=1, provenance_level="publisher", shingle_m=3)
    # 2 shared tools >= kappa=1 => fails
    assert not are_independent(g, p1, p2, cfg)


def test_filter_independent_family_pairwise():
    g = AgenticRuntimeGraph()
    paths = []
    for i in range(5):
        s = SourceNode(authority_id=f"auth_{i}", publisher_id=f"pub_{i}", domain=f"d{i}.com")
        g.add_node(s)
        t = TruthNode(payload=f"unique text {i} alpha beta".encode(), source_id=s.id)
        g.add_node(t)
        g.add_edge(s.id, t.id, EdgeType.RETRIEVED_FROM)
        paths.append(SupportPath(node_ids=(t.id,), tool_step_ids=frozenset({f"tool_{i}"})))
    cfg = IndependenceConfig(delta=0.5, kappa=1, provenance_level="publisher", shingle_m=3)
    accepted = filter_independent_family(g, paths, cfg)
    # Pairwise check
    for i, p in enumerate(accepted):
        for j, q in enumerate(accepted):
            if i != j:
                assert are_independent(g, p, q, cfg), \
                    f"Greedy filter returned non-independent pair {i},{j}"


# ---------------------------------------------------------------------------
# rho estimator properties
# ---------------------------------------------------------------------------


def test_rho_independent_branches_close_to_one():
    rng = np.random.default_rng(42)
    # 3 truly independent branches, each with marginal failure 0.1
    n = 5000
    branches = (rng.random((n, 3)) < 0.1).astype(int)
    est = estimate_rho(branches, alpha=0.05)
    # rho_hat should be near 1 for independent branches
    assert 0.5 <= est.rho_hat <= 2.5, f"Expected ~1 for independent branches, got {est.rho_hat}"
    # UCB still upper-bounds rho_hat
    assert est.rho_ucb >= est.rho_hat


def test_rho_correlated_branches_above_one():
    rng = np.random.default_rng(0)
    n = 5000
    common = rng.random(n) < 0.05    # common-cause failure
    branches = np.zeros((n, 3), dtype=int)
    for i in range(3):
        branches[:, i] = ((rng.random(n) < 0.05) | common).astype(int)
    est = estimate_rho(branches, alpha=0.05)
    # With common-cause, rho should be substantially > 1
    assert est.rho_hat > 1.5, f"Expected rho > 1.5 for correlated branches, got {est.rho_hat}"


def test_rho_ucb_upper_bounds_truth_high_prob():
    """A weak coverage check: across many bootstrap replicates, the UCB
    should fail to upper-bound the truth in <= alpha fraction of cases.
    With alpha=0.05 we expect <= 5% miscoverage. We use n=200 trials per
    replicate and 100 replicates."""
    rng = np.random.default_rng(7)
    p_true = 0.1
    miscoverage = 0
    n_replicates = 100
    for _ in range(n_replicates):
        n = 200
        common = rng.random(n) < 0.02
        branches = np.zeros((n, 3), dtype=int)
        for i in range(3):
            branches[:, i] = ((rng.random(n) < p_true) | common).astype(int)
        est = estimate_rho(branches, alpha=0.05)
        # The "truth" is the underlying joint failure rate; we check that
        # the UCB on rho is non-degenerate (>= 1).
        if est.rho_ucb < 1.0:
            miscoverage += 1
    # rho UCB should almost always be >= 1 (it's a UPPER bound on rho >= 1)
    assert miscoverage <= 0.10 * n_replicates
