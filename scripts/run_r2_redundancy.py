"""
R2: Redundancy law — empirical false-accept rate vs k.

Theorem 2 (homogeneous): Pr(A_t^{(k,k)} & false) <= rho^{k-1} * eps^k.

This script:
    1. For each example, runs k Provers with DIFFERENT retrievers (BM25,
       dense, hybrid, paraphrased-BM25) so the branches are operationally
       (delta, kappa)-independent.
    2. Records per-branch failure indicator E_i (= certificate accepts but
       the answer is wrong).
    3. Estimates the marginal failure rate eps and the dependence factor rho
       (with UCB).
    4. Compares empirical Pr(A_t^{(k,k)} & false) to the theoretical envelope
       at each k.

Output: results/<run_id>/r2.json containing arrays for the redundancy plot.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

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
    config: str = "configs/r2_redundancy.yaml",
    seeds: list[int] | None = None,
    k_values: list[int] | None = None,
    n_examples: int | None = None,
    backend: str | None = None,
) -> int:
    seeds = seeds or [0, 1, 2, 3, 4]
    cfg = load_config(config)
    if n_examples is not None:
        cfg.setdefault("dataset", {})["n_examples"] = n_examples
    if k_values is not None:
        cfg["k_values"] = k_values
    k_values = cfg_get(cfg, "k_values", [1, 2, 4, 8])

    run_id = make_run_id(config)
    out_dir = make_output_dir(cfg, run_id)
    log_section(f"R2 run: {run_id}")
    log_info(f"Output: {out_dir}; k_values={k_values}")
    write_json(out_dir / "config_snapshot.json", cfg)

    from pcg.agents.prover import ProverConfig, build_default_prover
    from pcg.checker import Checker, ExactMatchEntailment
    from pcg.datasets import load_dataset_by_name
    from pcg.eval import bootstrap_ci, estimate_rho
    from pcg.eval.metrics import f1_score
    from pcg.eval.rho import predicted_false_accept_rate
    from pcg.eval.stats import wilson_interval
    from pcg.orchestrator.langgraph_flow import (
        OrchestratorConfig,
        PCGState,
        run_one_example,
    )
    from pcg.orchestrator import build_replayer_with_handlers

    backend_obj = build_backend(cfg, override=backend)
    checker = Checker(
        entailment=ExactMatchEntailment(case_insensitive=True),
        replayer=build_replayer_with_handlers(),
    )

    retrievers = cfg_get(
        cfg, "prover.retrievers_for_branches",
        ["bm25", "dense", "hybrid", "bm25_with_query_paraphrase"],
    )
    n = cfg_get(cfg, "dataset.n_examples", 150)

    seed_results = []
    for seed in seeds:
        log_section(f"seed={seed}")
        examples = list(load_dataset_by_name(
            cfg_get(cfg, "dataset.name", "hotpotqa"),
            split=cfg_get(cfg, "dataset.split", "validation"),
            n_examples=n, seed=seed,
        ))
        max_k = max(k_values)
        log_info(f"  examples={len(examples)}  max_k={max_k}")

        # branch_fail_matrix[i, j] = 1 iff branch j on example i had a "failed"
        # certificate event (E_i in the paper). We define E_i := certificate
        # accepted AND answer is wrong (i.e., would be a false-accept event
        # that the redundancy rule must catch).
        n_examples_actual = 0
        rows: list[list[int]] = []
        per_example: list[dict] = []

        for i, ex in enumerate(examples):
            # Run max_k branches with different retrievers
            row = []
            branch_outcomes = []
            for j in range(max_k):
                retriever = retrievers[j % len(retrievers)]
                pcfg = ProverConfig(
                    top_k=cfg_get(cfg, "prover.top_k", 4),
                    max_answer_tokens=cfg_get(cfg, "prover.max_answer_tokens", 64),
                    temperature=0.0,
                    seed=seed * 100 + j,
                    retriever=retriever,
                )
                prover_fn = build_default_prover(backend=backend_obj, config=pcfg)
                state = PCGState(example=ex)
                state = prover_fn(state)
                cr = checker.check(state.certificate, state.graph) if state.certificate else None
                if cr is None:
                    row.append(1)   # treat missing as failure
                    branch_outcomes.append({"retriever": retriever, "accepted": False, "f1": 0.0})
                    continue
                raw = state.meta.get("raw_answer", "")
                f1 = f1_score(raw, list(ex.gold_answers))
                # E_j = 1 iff accepted & wrong
                event_E = 1 if (cr.passed and f1 < 0.5) else 0
                row.append(event_E)
                branch_outcomes.append({"retriever": retriever, "accepted": cr.passed, "f1": f1})
            rows.append(row)
            n_examples_actual += 1
            per_example.append({"id": ex.id, "branches": branch_outcomes})
            if (i + 1) % 25 == 0:
                log_info(f"    {i+1}/{len(examples)}")

        bf_matrix = np.asarray(rows, dtype=int)

        per_k = []
        for k in k_values:
            sub = bf_matrix[:, :k]
            rho_est = estimate_rho(sub, alpha=cfg_get(cfg, "stats.alpha", 0.05))
            # Empirical Pr(A^{(k,k)} & false) = Pr(all k branches passed AND wrong)
            joint_fail = (sub.sum(axis=1) == k).astype(int)
            n_trials = len(joint_fail)
            n_joint = int(joint_fail.sum())
            p_hat, lo, hi = wilson_interval(n_joint, n_trials, alpha=0.05)
            # Theory predictions
            theory_plug_in = predicted_false_accept_rate(
                rho_ucb_value=rho_est.rho_hat, p_marg_upper=rho_est.p_marg, k=k,
            )
            theory_ucb = predicted_false_accept_rate(
                rho_ucb_value=rho_est.rho_ucb, p_marg_upper=rho_est.p_marg, k=k,
            )
            per_k.append({
                "k": k,
                "rho": rho_est.to_dict(),
                "empirical": {"p_hat": p_hat, "lo": lo, "hi": hi,
                              "n_joint": n_joint, "n_trials": n_trials},
                "theory_plug_in": theory_plug_in,
                "theory_ucb": theory_ucb,
            })

        seed_results.append({
            "seed": seed, "n": n_examples_actual,
            "per_k": per_k,
            "branch_fail_matrix": bf_matrix.tolist(),
            "per_example": per_example,
        })
        write_json(out_dir / f"seed_{seed}.json", seed_results[-1])

    # Aggregate across seeds
    agg_per_k = []
    for ki, k in enumerate(k_values):
        emp_vals = [s["per_k"][ki]["empirical"]["p_hat"] for s in seed_results]
        rho_vals = [s["per_k"][ki]["rho"]["rho_hat"] for s in seed_results]
        rho_ucb_vals = [s["per_k"][ki]["rho"]["rho_ucb"] for s in seed_results]
        emp_ci = bootstrap_ci(emp_vals, n_boot=cfg_get(cfg, "stats.bootstrap_n", 2000))
        agg_per_k.append({
            "k": k,
            "empirical_mean": emp_ci.estimate,
            "empirical_ci": [emp_ci.lower, emp_ci.upper],
            "rho_hat_mean": float(np.mean(rho_vals)),
            "rho_ucb_max": float(np.max(rho_ucb_vals)),
            "theory_plug_in_mean": float(np.mean([s["per_k"][ki]["theory_plug_in"] for s in seed_results])),
            "theory_ucb_max": float(np.max([s["per_k"][ki]["theory_ucb"] for s in seed_results])),
        })
    final = {
        "run_id": run_id, "seeds": seeds, "k_values": k_values,
        "per_seed": seed_results, "aggregated_per_k": agg_per_k,
    }
    write_json(out_dir / "r2.json", final)
    log_section("R2 complete")
    log_info(f"Result: {out_dir / 'r2.json'}")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/r2_redundancy.yaml")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--k-values", type=int, nargs="+", default=None)
    p.add_argument("--n-examples", type=int, default=None)
    p.add_argument("--backend", default=None)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    sys.exit(main(config=a.config, seeds=a.seeds, k_values=a.k_values,
                   n_examples=a.n_examples, backend=a.backend))
