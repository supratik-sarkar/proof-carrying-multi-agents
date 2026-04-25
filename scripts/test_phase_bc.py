"""
scripts/test_phase_bc.py — smoke-test Phase B (data) + Phase C (backends).

Run after copying Phase B/C files in:
    python scripts/test_phase_bc.py

What it checks:
    1. synthetic dataset loader yields valid QAExamples
    2. BM25Index builds and searches deterministically
    3. MockBackend generates without any model download
    4. End-to-end: synthetic example -> BM25 retrieval -> MockBackend gen
"""
from __future__ import annotations

import sys


def main() -> int:
    print("=" * 60)
    print("Phase B + C smoke test")
    print("=" * 60)

    # ---- 1. Synthetic dataset ----
    print("\n[1] Synthetic dataset")
    from pcg.datasets import load_dataset_by_name
    examples = list(load_dataset_by_name("synthetic", n_examples=3))
    print(f"    Loaded {len(examples)} examples.")
    for ex in examples:
        gold_count = sum(1 for e in ex.evidence if e.is_gold)
        print(f"    {ex.id}: '{ex.question[:50]}...' "
              f"({len(ex.evidence)} evidence, {gold_count} gold)")
    assert len(examples) == 3, "Expected 3 examples"
    assert all(ex.evidence for ex in examples), "Examples must have evidence"

    # ---- 2. BM25 retrieval ----
    print("\n[2] BM25 retrieval")
    from pcg.retrieval import BM25Index
    ex = examples[0]
    idx = BM25Index.build(ex.evidence)
    hits = idx.search(ex.question, top_k=3)
    print(f"    Query: '{ex.question}'")
    for item, score in hits:
        marker = "GOLD" if item.is_gold else "    "
        print(f"    [{marker}] score={score:.3f} | {item.title}: {item.text[:60]}...")
    assert len(hits) == 3
    # Determinism check
    hits2 = idx.search(ex.question, top_k=3)
    assert [h[0].id for h in hits] == [h[0].id for h in hits2], "BM25 not deterministic"
    print("    ... BM25 deterministic.")

    # ---- 3. Mock backend ----
    print("\n[3] MockBackend")
    from pcg.backends import MockBackend
    backend = MockBackend()
    prompt = (
        f"Question: {ex.question}\n"
        f"Context: {' '.join(e.text for e in ex.evidence if e.is_gold)}\n"
        f"Answer:"
    )
    out = backend.generate(prompt, max_tokens=64, seed=0)
    print(f"    Generated: '{out.text[:80]}...'")
    print(f"    tokens_in={out.tokens_in}, tokens_out={out.tokens_out}, "
          f"latency_ms={out.latency_ms:.2f}")
    out2 = backend.generate(prompt, max_tokens=64, seed=0)
    assert out.text == out2.text, "MockBackend not deterministic"
    print("    ... MockBackend deterministic.")

    # ---- 4. End-to-end ----
    print("\n[4] End-to-end: retrieval + generation")
    for ex in examples:
        idx = BM25Index.build(ex.evidence)
        hits = idx.search(ex.question, top_k=2)
        ctx = " ".join(item.text for item, _ in hits)
        prompt = f"Question: {ex.question}\nContext: {ctx}\nAnswer:"
        out = backend.generate(prompt, max_tokens=64)
        gold_set = set(ex.gold_answers)
        contains_gold = any(g.lower() in out.text.lower() for g in gold_set)
        marker = "✓" if contains_gold else "✗"
        print(f"    [{marker}] {ex.id}: pred='{out.text[:40]}...' gold={ex.gold_answers}")

    # ---- 5. Audit decomposition smoke test ----
    print("\n[5] Audit decomposition (Theorem 1) smoke test")
    from pcg.checker import CheckResult
    from pcg.eval import estimate_audit_decomposition
    fake_results = [
        CheckResult(passed=True, integrity_ok=True, replay_ok=True,
                    entailment_ok=True, execution_ok=True),
        CheckResult(passed=False, integrity_ok=False, replay_ok=True,
                    entailment_ok=True, execution_ok=True),
        CheckResult(passed=False, integrity_ok=True, replay_ok=False,
                    entailment_ok=True, execution_ok=True),
    ] * 5
    fake_truth = [True, False, False] * 5
    decomp = estimate_audit_decomposition(fake_results, fake_truth, alpha=0.05)
    print(f"    LHS (accept & wrong): {decomp.lhs_accept_and_wrong:.3f} "
          f"CI={decomp.ci_lhs}")
    print(f"    RHS union: {decomp.rhs_union:.3f}")
    print(f"    -> bound holds: {decomp.lhs_accept_and_wrong <= decomp.rhs_union}")

    # ---- 6. rho estimator smoke test ----
    print("\n[6] rho estimator (Theorem 2) smoke test")
    import numpy as np
    from pcg.eval import estimate_rho
    rng = np.random.default_rng(0)
    # Simulate k=3 branches with marginal failure rate 0.1, mild dependence
    n_trials = 1000
    common = rng.random(n_trials) < 0.02   # rare common-cause failure
    branches = np.zeros((n_trials, 3), dtype=int)
    for i in range(3):
        branches[:, i] = ((rng.random(n_trials) < 0.10) | common).astype(int)
    rho = estimate_rho(branches)
    print(f"    rho_hat = {rho.rho_hat:.2f}  (expect > 1 due to common-cause)")
    print(f"    rho_UCB = {rho.rho_ucb:.2f}  (95% upper bound)")
    print(f"    p_marg  = {rho.p_marg:.3f}  (expect ~0.12)")

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
