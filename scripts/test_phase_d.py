"""
scripts/test_phase_d.py — smoke-test Phase D (orchestrator + agents).

Exercises the full pipeline:
    Prover -> Verifier -> [Debugger] for clean cases
    Prover -> Attacker -> Verifier (-> Debugger) for adversarial cases

Uses the synthetic dataset and MockBackend so it runs in <1 second with
zero downloads.
"""
from __future__ import annotations

import sys


def main() -> int:
    print("=" * 60)
    print("Phase D smoke test (orchestrator + agents)")
    print("=" * 60)

    from pcg.backends import MockBackend
    from pcg.checker import Checker, ExactMatchEntailment, build_default_replayer
    from pcg.datasets import load_dataset_by_name
    from pcg.orchestrator.langgraph_flow import OrchestratorConfig, run_one_example

    backend = MockBackend()
    checker = Checker(
        entailment=ExactMatchEntailment(case_insensitive=True),
        replayer=build_default_replayer(),
    )

    examples = list(load_dataset_by_name("synthetic", n_examples=5))
    print(f"\n[1] Loaded {len(examples)} synthetic examples")

    # --- (a) Clean run: no attacker, no debugger ---
    print("\n[2] Clean orchestrator run (Prover -> Verifier)")
    cfg_clean = OrchestratorConfig(enable_attacker=False, enable_debugger=False)
    pass_count = 0
    for ex in examples:
        state = run_one_example(ex, backend=backend, checker=checker, cfg=cfg_clean)
        cr = state.check_result
        ok = cr is not None and cr.passed
        if ok:
            pass_count += 1
        marker = "✓" if ok else "✗"
        reasons = "; ".join((cr.reasons if cr else ["no_check"])[:2])
        print(f"    [{marker}] {ex.id}: passed={ok} reasons={reasons}")
    print(f"    {pass_count}/{len(examples)} passed (expected ~3-4 with mock LLM)")

    # --- (b) Adversarial run: evidence_swap attack ---
    print("\n[3] Adversarial run (Prover -> Attacker:evidence_swap -> Verifier)")
    cfg_adv = OrchestratorConfig(enable_attacker=True, enable_debugger=False,
                                  attack_kind="evidence_swap", max_retries=0)
    detected = 0
    for ex in examples:
        state = run_one_example(ex, backend=backend, checker=checker, cfg=cfg_adv)
        cr = state.check_result
        if cr is not None and not cr.passed and not cr.integrity_ok:
            detected += 1
        marker = "✓" if (cr and not cr.integrity_ok) else "✗"
        print(f"    [{marker}] {ex.id}: integrity_ok={cr.integrity_ok if cr else None}")
    print(f"    Attacker detected: {detected}/{len(examples)} "
          f"({'PASS' if detected == len(examples) else 'FAIL — soundness bug?'})")
    assert detected == len(examples), "All evidence_swap attacks must be detected by Check_clm"

    # --- (c) Debugger run with risk policy ---
    print("\n[4] Debugger run (Prover -> Verifier -> Debugger)")
    cfg_dbg = OrchestratorConfig(enable_attacker=False, enable_debugger=True,
                                  risk_lambda=1.0, h_fa=10.0, h_ref=0.05)
    actions = []
    for ex in examples:
        state = run_one_example(ex, backend=backend, checker=checker, cfg=cfg_dbg)
        a = state.chosen_action
        actions.append(a)
        n_resp = len(state.responsibility)
        risk = state.meta.get("posterior_risk", -1)
        print(f"    {ex.id}: action={a:8s}  risk={risk:.4f}  n_resp={n_resp}")
    print(f"    Actions chosen: {dict((a, actions.count(a)) for a in set(actions))}")

    # --- (d) Meter overhead R5 sanity check ---
    print("\n[5] Meter overhead (R5)")
    state = run_one_example(examples[0], backend=backend, checker=checker, cfg=cfg_dbg)
    rep = state.meter.report()
    print(f"    Wall: {rep.wall_ms:.1f} ms  |  Total tokens: {rep.total_tokens()}")
    for phase_name, p in sorted(rep.phases.items()):
        print(f"      {phase_name:30s}  {p.total_ms:7.2f} ms  tok_in={p.total_tokens_in:4d}  tok_out={p.total_tokens_out:4d}")

    # --- (e) Replay handlers registry ---
    print("\n[6] Replay handlers registry")
    from pcg.orchestrator import build_replayer_with_handlers
    rep_full = build_replayer_with_handlers()
    print(f"    Registered ops: {sorted(rep_full.handlers)}")
    expected = {"identity", "concat", "bm25_retrieve_replay", "span_extract", "nli_filter", "schema_validate"}
    assert expected.issubset(set(rep_full.handlers)), f"Missing ops: {expected - set(rep_full.handlers)}"

    print("\n" + "=" * 60)
    print("PHASE D ALL CHECKS PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
