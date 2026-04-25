"""
R3: Responsibility / diagnosis (Theorem 3 part i).

For each example we estimate Resp_hat for every certificate component and
report:
    - The top-1 component identified as root-cause
    - The Hoeffding CI for each Resp_hat
    - The rank-recovery probability lower bound from Eq. (28)
    - Aggregate accuracy of root-cause identification under controlled
      corruption regimes (we know which component to corrupt -> we know
      the correct top-1 answer -> we score whether the estimator finds it)

Corruption regimes (configurable in YAML):
    - clean   : no induced failures; tests false-positive rate
    - light   : 10% of evidence randomly tampered
    - heavy   : 30% of evidence randomly tampered
"""
from __future__ import annotations

import argparse
import random
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
    config: str = "configs/r3_responsibility.yaml",
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
    log_section(f"R3 run: {run_id}")
    log_info(f"Output: {out_dir}")
    write_json(out_dir / "config_snapshot.json", cfg)

    from pcg.checker import Checker, ExactMatchEntailment
    from pcg.datasets import load_dataset_by_name
    from pcg.eval import bootstrap_ci
    from pcg.orchestrator.langgraph_flow import (
        OrchestratorConfig,
        PCGState,
        run_one_example,
    )
    from pcg.orchestrator import build_replayer_with_handlers
    from pcg.responsibility import ResponsibilityEstimator, rank_recovery_prob
    from pcg.agents.prover import ProverConfig, build_default_prover
    from pcg.graph import TruthNode

    backend_obj = build_backend(cfg, override=backend)
    checker = Checker(
        entailment=ExactMatchEntailment(case_insensitive=True),
        replayer=build_replayer_with_handlers(),
    )

    n = cfg_get(cfg, "dataset.n_examples", 100)
    n_replays = cfg_get(cfg, "debugger.n_replays", 100)
    alpha = cfg_get(cfg, "debugger.alpha", 0.05)
    regimes = cfg_get(cfg, "corruption_regimes", [
        {"name": "clean", "p_swap": 0.0},
        {"name": "light", "p_swap": 0.10},
        {"name": "heavy", "p_swap": 0.30},
    ])

    seed_results = []
    for seed in seeds:
        log_section(f"seed={seed}")
        examples = list(load_dataset_by_name(
            cfg_get(cfg, "dataset.name", "hotpotqa"),
            split=cfg_get(cfg, "dataset.split", "validation"),
            n_examples=n, seed=seed,
        ))

        per_regime = []
        for regime in regimes:
            log_info(f"  regime={regime['name']} (p_swap={regime['p_swap']})")
            rng = random.Random(seed * 1000 + hash(regime["name"]) % 1000)

            # Track top-1 identification accuracy (when there's a known target).
            n_correct_top1 = 0
            n_with_target = 0
            margins = []
            rrp_values = []
            mean_resps = []

            for i, ex in enumerate(examples):
                # Build a baseline certificate (no corruption injection at Prover time)
                pcfg = ProverConfig(
                    top_k=cfg_get(cfg, "prover.top_k", 4),
                    temperature=0.0, seed=seed * 100 + i,
                    retriever=cfg_get(cfg, "prover.retriever", "bm25"),
                )
                prover_fn = build_default_prover(backend=backend_obj, config=pcfg)
                state = PCGState(example=ex)
                state = prover_fn(state)
                if state.certificate is None:
                    continue

                # Inject corruption: with probability p_swap, tamper exactly one
                # evidence node. We RECORD which one was tampered (target_id).
                target_id: str | None = None
                if regime["p_swap"] > 0 and rng.random() < regime["p_swap"]:
                    truth_ids = list(state.certificate.claim_cert.evidence_ids)
                    if truth_ids:
                        target_id = rng.choice(truth_ids)
                        node = state.graph.nodes[target_id]
                        if isinstance(node, TruthNode):
                            node.payload = b"CORRUPTED " + rng.randbytes(16)

                # Now estimate Resp for every certificate component.
                comp_ids = list(state.certificate.claim_cert.evidence_ids) \
                    + list(state.certificate.exec_cert.tool_call_ids)
                est = ResponsibilityEstimator(
                    checker=checker, n_replays=n_replays,
                    alpha=alpha, paired=cfg_get(cfg, "debugger.use_paired_replay", True),
                    seed=seed * 1000 + i,
                )
                results = est.estimate_many(state.certificate, state.graph, comp_ids)
                if not results:
                    continue
                results.sort(key=lambda r: r.estimate, reverse=True)
                top = results[0]
                margin = top.estimate - (results[1].estimate if len(results) > 1 else 0.0)
                rrp = rank_recovery_prob(n_replays=n_replays,
                                           n_components=len(results), margin=margin,
                                           alpha_family=alpha)
                margins.append(margin)
                rrp_values.append(rrp)
                mean_resps.append(float(np.mean([r.estimate for r in results])))

                if target_id is not None:
                    n_with_target += 1
                    if top.component_id == target_id:
                        n_correct_top1 += 1

                if (i + 1) % 25 == 0:
                    log_info(f"    {i+1}/{len(examples)}")

            top1_acc = (n_correct_top1 / n_with_target) if n_with_target > 0 else None
            per_regime.append({
                "regime": regime["name"],
                "p_swap": regime["p_swap"],
                "n_with_target": n_with_target,
                "top1_accuracy": top1_acc,
                "mean_margin": float(np.mean(margins)) if margins else 0.0,
                "median_rank_recovery_prob": float(np.median(rrp_values)) if rrp_values else 0.0,
                "mean_resp": float(np.mean(mean_resps)) if mean_resps else 0.0,
            })

        seed_results.append({"seed": seed, "per_regime": per_regime})
        write_json(out_dir / f"seed_{seed}.json", seed_results[-1])

    # Aggregate
    regime_names = [r["name"] for r in regimes]
    agg = {}
    for rn in regime_names:
        acc_vals = [r["top1_accuracy"] for s in seed_results
                     for r in s["per_regime"] if r["regime"] == rn and r["top1_accuracy"] is not None]
        if acc_vals:
            ci = bootstrap_ci(acc_vals, n_boot=cfg_get(cfg, "stats.bootstrap_n", 2000))
            agg[rn] = {"top1_accuracy_mean": ci.estimate, "ci": [ci.lower, ci.upper]}
        else:
            agg[rn] = {"top1_accuracy_mean": None}

    final = {"run_id": run_id, "seeds": seeds, "per_seed": seed_results, "aggregated": agg}
    write_json(out_dir / "r3.json", final)
    log_section("R3 complete")
    log_info(f"Result: {out_dir / 'r3.json'}")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/r3_responsibility.yaml")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--n-examples", type=int, default=None)
    p.add_argument("--backend", default=None)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    sys.exit(main(config=a.config, seeds=a.seeds,
                   n_examples=a.n_examples, backend=a.backend))
