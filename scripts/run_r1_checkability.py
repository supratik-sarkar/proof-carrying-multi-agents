"""
R1: Audit decomposition / checkability.

Theorem 1 says
    Pr(accept & wrong) <= Pr(IntFail) + Pr(ReplayFail) + Pr(CheckFail) + Pr(CovGap).

This script:
    1. Runs the Prover on N HotpotQA examples (clean and adversarial mix).
    2. Verifier checks each certificate, recording per-channel flags.
    3. Compares predictions to gold answers to label CovGap.
    4. Estimates each channel's empirical probability with Wilson CIs.
    5. Writes results/<run_id>/r1.json — picked up by make_figures and make_tables.

Usage:
    python scripts/run_r1_checkability.py --config configs/r1_hotpotqa.yaml --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from _common import (
    build_backend,
    cfg_get,
    load_config,
    log_info,
    log_section,
    make_output_dir,
    make_run_id,
    write_json,
)


def main(
    config: str = "configs/r1_hotpotqa.yaml",
    seeds: list[int] | None = None,
    n_examples: int | None = None,
    backend: str | None = None,
) -> int:
    seeds = seeds or [0, 1, 2, 3, 4]
    cfg = load_config(config)
    if n_examples is not None:
        cfg.setdefault("dataset", {})["n_examples"] = n_examples

    run_id = make_run_id(config)
    out_dir = make_output_dir(cfg, run_id)
    log_section(f"R1 run: {run_id}")
    log_info(f"Output: {out_dir}")

    write_json(out_dir / "config_snapshot.json", cfg)

    # Imports are lazy here so the smoke path doesn't pull in heavy deps
    from pcg.checker import (
        Checker,
        ExactMatchEntailment,
        build_default_replayer,
    )
    from pcg.datasets import load_dataset_by_name
    from pcg.eval import bootstrap_ci, estimate_audit_decomposition
    from pcg.eval.metrics import f1_score
    from pcg.orchestrator.langgraph_flow import OrchestratorConfig, run_one_example
    from pcg.orchestrator import build_replayer_with_handlers

    backend_obj = build_backend(cfg, override=backend)
    log_info(f"Backend: {backend_obj.name}")
    checker = Checker(
        entailment=ExactMatchEntailment(case_insensitive=True),
        replayer=build_replayer_with_handlers(),
    )

    n = cfg_get(cfg, "dataset.n_examples", 200)
    adv_enabled = cfg_get(cfg, "adversarial.enabled", False)
    adv_frac = cfg_get(cfg, "adversarial.fraction", 0.30)
    attack_kinds = cfg_get(cfg, "adversarial.attack_kinds",
                            ["evidence_swap", "schema_break", "policy_violation"])

    seed_results: list[dict] = []

    for seed in seeds:
        log_section(f"seed={seed}")
        examples = list(load_dataset_by_name(
            cfg_get(cfg, "dataset.name", "hotpotqa"),
            split=cfg_get(cfg, "dataset.split", "validation"),
            n_examples=n, seed=seed,
        ))
        log_info(f"  loaded {len(examples)} examples")

        check_results = []
        gt_correct = []
        per_example_records = []

        for i, ex in enumerate(examples):
            # Decide attack
            adv_idx = (seed * 7919 + i) % 1000   # deterministic per (seed, i)
            attack = (adv_idx / 1000.0) < adv_frac if adv_enabled else False
            kind = attack_kinds[adv_idx % len(attack_kinds)] if attack else "none"

            cfg_run = OrchestratorConfig(
                enable_attacker=attack,
                enable_debugger=False,
                attack_kind=kind if attack else "evidence_swap",
                max_retries=0,
            )
            state = run_one_example(
                ex, backend=backend_obj, checker=checker, cfg=cfg_run,
            )
            cr = state.check_result
            if cr is None:
                continue
            check_results.append(cr)

            # CovGap baseline: did the (claim text) match a gold answer?
            raw = state.meta.get("raw_answer", "")
            f1 = f1_score(raw, list(ex.gold_answers))
            gt_correct.append(f1 >= 0.5 and not attack)   # attacks => gt_correct=False

            per_example_records.append({
                "id": ex.id, "passed": cr.passed,
                "integrity_ok": cr.integrity_ok, "replay_ok": cr.replay_ok,
                "entailment_ok": cr.entailment_ok, "execution_ok": cr.execution_ok,
                "attack": kind, "f1_to_gold": f1,
                "raw_answer": raw[:200],
            })

            if (i + 1) % 25 == 0:
                log_info(f"    {i+1}/{len(examples)}")

        decomp = estimate_audit_decomposition(check_results, gt_correct, alpha=0.05)
        log_info(f"  LHS = {decomp.lhs_accept_and_wrong:.4f}  RHS-union = {decomp.rhs_union:.4f}")

        seed_results.append({
            "seed": seed, "n": len(check_results),
            "decomposition": decomp.to_dict(),
            "per_example": per_example_records,
        })
        write_json(out_dir / f"seed_{seed}.json", seed_results[-1])

    # Aggregate across seeds: bootstrap CI on each channel
    aggregated = {}
    for key in ("p_int_fail", "p_replay_fail", "p_check_fail", "p_cov_gap",
                 "lhs_accept_and_wrong", "rhs_union"):
        values = [s["decomposition"][key] for s in seed_results]
        ci = bootstrap_ci(values, n_boot=cfg_get(cfg, "stats.bootstrap_n", 2000),
                          alpha=cfg_get(cfg, "stats.alpha", 0.05))
        aggregated[key] = {"mean": ci.estimate, "ci": [ci.lower, ci.upper]}

    final = {
        "run_id": run_id, "seeds": seeds,
        "per_seed": seed_results, "aggregated": aggregated,
    }
    write_json(out_dir / "r1.json", final)
    log_section("R1 complete")
    log_info(f"Result: {out_dir / 'r1.json'}")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/r1_hotpotqa.yaml")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--n-examples", type=int, default=None)
    p.add_argument("--backend", default=None,
                    choices=[None, "mock", "hf_local", "hf_inference"])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sys.exit(main(config=args.config, seeds=args.seeds,
                   n_examples=args.n_examples, backend=args.backend))
