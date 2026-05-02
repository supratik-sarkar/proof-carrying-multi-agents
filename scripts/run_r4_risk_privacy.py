"""
R4: Risk-aware control + privacy.

Compares three policies on the same logged rollouts:
    - always_answer       : ignores risk; answers everything
    - threshold_pcg       : Theorem 3 part ii; piecewise threshold over r
    - learned             : logistic-contextual policy (baseline)

For each policy and each privacy level eps (in cfg.privacy.eps_values), we
report (cost, harm) so the make_figures step can plot the Pareto frontier.

Implementation note: the LLM Prover is run ONCE per example (the expensive
part). All policies share the same logged certificate, only the action choice
differs. This means R4 wall time ~ R1 wall time + a few seconds.
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


def _parse_eps(s: str | float) -> float:
    if isinstance(s, (int, float)):
        return float(s)
    if str(s).lower() == "inf":
        return float("inf")
    return float(s)


def main(
    config: str = "configs/r4_risk.yaml",
    seeds: list[int] | None = None,
    eps_values: list[str] | None = None,
    n_examples: int | None = None,
    backend: str | None = None,
) -> int:
    seeds = seeds or [0, 1, 2, 3, 4]
    cfg = load_config(config)
    if n_examples is not None:
        cfg.setdefault("dataset", {})["n_examples"] = n_examples
    if eps_values is not None:
        cfg.setdefault("privacy", {})["eps_values"] = eps_values

    run_id = make_run_id(config)
    out_dir = make_output_dir(cfg, run_id)
    log_section(f"R4 run: {run_id}")
    write_json(out_dir / "config_snapshot.json", cfg)

    from pcg.checker import Checker, ExactMatchEntailment
    from pcg.datasets import load_dataset_by_name
    from pcg.eval import bootstrap_ci
    from pcg.eval.metrics import f1_score
    from pcg.orchestrator.langgraph_flow import (
        OrchestratorConfig,
        PCGState,
        run_one_example,
    )
    from pcg.orchestrator import build_replayer_with_handlers
    from pcg.privacy import gaussian_mechanism
    from pcg.risk import Action, Calibrator, CostModel, ThresholdPolicy, posterior_risk
    from pcg.agents.prover import ProverConfig, build_default_prover

    backend_obj = build_backend(cfg, override=backend)
    checker = Checker(
        entailment=ExactMatchEntailment(case_insensitive=True),
        replayer=build_replayer_with_handlers(),
    )

    cm_cfg = cfg_get(cfg, "cost_model", {})
    cm = CostModel(
        c_lat={Action(k): v for k, v in cm_cfg.get("c_lat", {}).items()},
        c_tok={Action(k): v for k, v in cm_cfg.get("c_tok", {}).items()},
        c_tool={Action(k): v for k, v in cm_cfg.get("c_tool", {}).items()},
        lam=float(cm_cfg.get("lambda_risk", 1.0)),
        h_fa=float(cm_cfg.get("h_fa", 1.0)),
        h_ref=float(cm_cfg.get("h_ref", 0.0)),
        eta={Action(k): v for k, v in cm_cfg.get("eta", {}).items()},
    )
    eps_values_parsed = [_parse_eps(e) for e in cfg_get(cfg, "privacy.eps_values", ["inf", 8, 3, 1])]
    n = cfg_get(cfg, "dataset.n_examples", 200)

    seed_results = []
    for seed in seeds:
        log_section(f"seed={seed}")
        examples = list(load_dataset_by_name(
            cfg_get(cfg, "dataset.name", "hotpotqa"),
            split=cfg_get(cfg, "dataset.split", "validation"),
            n_examples=n, seed=seed,
        ))

        # Run the prover once per example, store certificates + raw confidences
        rows: list[dict] = []
        for i, ex in enumerate(examples):
            pcfg = ProverConfig(
                top_k=cfg_get(cfg, "prover.top_k", 4),
                temperature=0.0, seed=seed * 100 + i,
            )
            prover_fn = build_default_prover(backend=backend_obj, config=pcfg)
            state = PCGState(example=ex)
            state = prover_fn(state)
            if state.certificate is None:
                continue
            cr = checker.check(state.certificate, state.graph)
            raw = state.meta.get("raw_answer", "")
            f1 = f1_score(raw, list(ex.gold_answers))
            wrong = (f1 < 0.5)
            rows.append({
                "id": ex.id,
                "passed": cr.passed,
                "raw_conf": state.certificate.confidence,
                "f1": f1, "wrong": wrong,
                "raw_answer": raw,
            })
            if (i + 1) % 25 == 0:
                log_info(f"    {i+1}/{len(examples)}")

        # ---- Calibrate confidence on a held-out half ----
        confs = np.asarray([r["raw_conf"] for r in rows])
        labels = np.asarray([1 if (r["passed"] and not r["wrong"]) else 0 for r in rows])
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(rows))
        cut = len(rows) // 2
        cal = Calibrator(method=cfg_get(cfg, "calibration.method", "isotonic"))
        cal.fit(confs[idx[:cut]], labels[idx[:cut]])
        cal_confs = cal.transform(confs[idx[cut:]])
        eval_rows = [rows[j] for j in idx[cut:]]

        per_eps: list[dict] = []
        sens = cfg_get(cfg, "privacy.sensitivities", {"redundancy": 1.0,
                                                       "disagreement": 1.0,
                                                       "verifier_margin": 0.5})
        for eps in eps_values_parsed:
            # DP step: apply Gaussian noise to the calibrated confidence with
            # sensitivity = max in the dict (worst case across shared features).
            sigma_sens = max(sens.values())
            if eps != float("inf"):
                noisy = gaussian_mechanism(
                    cal_confs.copy(),
                    sensitivity=sigma_sens,
                    epsilon=eps,
                    delta=cfg_get(cfg, "privacy.delta", 1e-5),
                    rng=np.random.default_rng(seed),
                )
                noisy = np.clip(noisy, 0.0, 1.0)
            else:
                noisy = cal_confs

            # Compute (cost, harm) for each policy
            policy_results = {}
            for pname in ["always_answer", "threshold_pcg", "learned"]:
                if pname == "always_answer":
                    actions = [Action.ANSWER] * len(eval_rows)
                elif pname == "threshold_pcg":
                    pol = ThresholdPolicy(cost_model=cm)
                    actions = []
                    for j, row in enumerate(eval_rows):
                        r = posterior_risk(
                            confidences=[float(noisy[j])],
                            pass_flags=[row["passed"]],
                            rho=1.0,
                        )
                        actions.append(pol.choose(r))
                elif pname == "learned":
                    # Quick logistic surrogate using calibrated conf as the
                    # only feature (in practice this would use more features).
                    from sklearn.linear_model import LogisticRegression
                    train_X = noisy[:cut].reshape(-1, 1) if False else cal_confs[:cut].reshape(-1, 1)
                    y = labels[idx[:cut]]
                    if len(set(y)) < 2:
                        actions = [Action.ANSWER] * len(eval_rows)
                    else:
                        lr = LogisticRegression(max_iter=300).fit(train_X, y)
                        preds = lr.predict_proba(noisy.reshape(-1, 1))[:, 1]
                        # If predicted "wrong" prob > 0.5 -> refuse, else answer
                        actions = [Action.REFUSE if p < 0.5 else Action.ANSWER for p in preds]

                # Score the chosen actions
                cost_total = 0.0
                harm_total = 0.0
                for j, (row, a) in enumerate(zip(eval_rows, actions)):
                    # Posterior risk at this confidence
                    r_post = max(0.0, 1.0 - float(noisy[j]))
                    c = cm.cost(a, r_post)
                    cost_total += c
                    # Harm: only if action is Answer and example is wrong
                    if a == Action.ANSWER and row["wrong"]:
                        harm_total += cm.h_fa * cm.eta.get(a, 1.0)
                    elif a == Action.REFUSE:
                        harm_total += cm.h_ref
                policy_results[pname] = {
                    "cost_per_claim": cost_total / max(1, len(eval_rows)),
                    "harm_per_claim": harm_total / max(1, len(eval_rows)),
                    "n": len(eval_rows),
                }

            per_eps.append({"eps": eps if eps != float("inf") else "inf",
                             "policies": policy_results})

        seed_results.append({"seed": seed, "per_eps": per_eps,
                              "n_eval": len(eval_rows)})
        write_json(out_dir / f"seed_{seed}.json", seed_results[-1])

    # Aggregate
    eps_keys = [e["eps"] for e in seed_results[0]["per_eps"]]
    aggregated = {}
    for eps in eps_keys:
        for pname in ["always_answer", "threshold_pcg", "learned"]:
            costs = []
            harms = []
            for s in seed_results:
                for entry in s["per_eps"]:
                    if entry["eps"] == eps:
                        costs.append(entry["policies"][pname]["cost_per_claim"])
                        harms.append(entry["policies"][pname]["harm_per_claim"])
            ci_c = bootstrap_ci(costs, n_boot=cfg_get(cfg, "stats.bootstrap_n", 2000))
            ci_h = bootstrap_ci(harms, n_boot=cfg_get(cfg, "stats.bootstrap_n", 2000))
            aggregated.setdefault(str(eps), {})[pname] = {
                "cost_mean": ci_c.estimate, "cost_ci": [ci_c.lower, ci_c.upper],
                "harm_mean": ci_h.estimate, "harm_ci": [ci_h.lower, ci_h.upper],
            }

    final = {"run_id": run_id, "seeds": seeds, "eps_values": eps_keys,
              "per_seed": seed_results, "aggregated": aggregated}
    write_json(out_dir / "r4.json", final)
    log_section("R4 complete")
    log_info(f"Result: {out_dir / 'r4.json'}")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/r4_risk.yaml")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--eps-values", nargs="+", default=None)
    p.add_argument("--n-examples", type=int, default=None)
    p.add_argument("--backend", default=None)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    sys.exit(main(config=a.config, seeds=a.seeds, eps_values=a.eps_values,
                   n_examples=a.n_examples, backend=a.backend))
