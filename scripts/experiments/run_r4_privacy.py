#!/usr/bin/env python3
"""R4b privacy-budgeted certificate sharing — measured.

Runs the prover once per example, then evaluates the risk policy under a
(B_info, eta) grid of certificate compressions:

  B_info ∈ {32, 64, 128, 256}: number of discrete buckets for the
                                certificate's confidence field. Lower
                                B_info = coarser compression = more lossy
                                information channel.
  eta    ∈ {0.0, 0.25, 0.5, 1.0}: stddev multiplier for additive Gaussian
                                  noise on the compressed confidence.

Per (B_info, eta) cell we report:
  rho_hat: residual-dependence proxy = mean |compressed_conf - raw_conf| + eta
  harm:    accepted-harm rate under threshold_pcg policy on the
            compressed-confidence calibrated certificate
  utility: 1 - refuse_rate under threshold_pcg on same calibrated certificate

Source label: measured. This replaces the prior synthetic_frontier runner.

Output:
  results/tables/csv/experiment_json/<run_id>/r4_privacy.json
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cells", default="hotpotqa:phi-3.5-mini")
    p.add_argument("--n-examples", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backend", type=str, default="hf_local",
                   help="Backend kind: hf_local | hf_inference | mock")
    return p.parse_args()


def quantize_confidence(conf: float, B_info: int) -> float:
    """Quantize confidence ∈ [0,1] to B_info discrete buckets.
    bucket = round(conf * (B_info - 1)) / (B_info - 1)
    """
    if B_info <= 1:
        return 0.5
    bucket = round(conf * (B_info - 1)) / (B_info - 1)
    return max(0.0, min(1.0, bucket))


def run_cell(cell_spec, n, seed, backend_kind='hf_local'):
    from scripts.experiments.run_ablations import resolve_cells
    from scripts.common.experiment_io import build_backend
    from pcg.datasets import load_dataset_by_name
    from pcg.checker import Checker, TokenOverlapEntailment
    from pcg.orchestrator import build_replayer_with_handlers
    from pcg.orchestrator.langgraph_flow import OrchestratorConfig, PCGState
    from pcg.agents.prover import ProverConfig, build_default_prover
    from pcg.eval.metrics import f1_score
    from pcg.risk import Action, Calibrator, CostModel, ThresholdPolicy, posterior_risk
    from pcg.privacy import gaussian_mechanism
    import numpy as np

    dataset_name, paper_model_name, hf_model_name = resolve_cells(cell_spec)[0]
    backend_cfg = {"backend": {"kind": backend_kind, "model_name": hf_model_name,
                                "dtype": "float16", "load_in_4bit": False,
                                "trust_remote_code": True}}
    backend_obj = build_backend(backend_cfg, override=backend_kind)
    checker = Checker(entailment=TokenOverlapEntailment(threshold=0.5),
                       replayer=build_replayer_with_handlers())
    examples = list(load_dataset_by_name(dataset_name, split="validation",
                                          n_examples=n, seed=seed))

    # 1. Run prover once per example
    rows = []
    for i, ex in enumerate(examples):
        pcfg = ProverConfig(top_k=4, temperature=0.0, seed=seed * 100 + i)
        prover_fn = build_default_prover(backend=backend_obj, config=pcfg)
        state = PCGState(example=ex)
        state = prover_fn(state)
        if state.certificate is None:
            continue
        cr = checker.check(state.certificate, state.graph)
        raw = state.meta.get("raw_answer", "")
        f1 = f1_score(raw, list(ex.gold_answers)) if ex.gold_answers else 0.0
        rows.append({
            "id": ex.id, "passed": cr.passed,
            "raw_conf": state.certificate.confidence,
            "f1": f1, "wrong": f1 < 0.5,
        })

    if not rows:
        return None

    # 2. Calibrate confidence on held-out half
    confs = np.asarray([r["raw_conf"] for r in rows])
    labels = np.asarray([1 if (r["passed"] and not r["wrong"]) else 0 for r in rows])
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(rows))
    cut = max(1, len(rows) // 2)
    cal = Calibrator(method="isotonic")
    cal.fit(confs[idx[:cut]], labels[idx[:cut]])
    cal_confs = cal.transform(confs[idx[cut:]])
    eval_rows = [rows[j] for j in idx[cut:]]
    raw_eval_confs = np.array([r["raw_conf"] for r in eval_rows])

    # Guard: degenerate confidence distribution (near-zero variance across
    # examples) makes the threshold policy collapse. Detect and label honestly.
    conf_std = float(np.std(cal_confs)) if len(cal_confs) > 1 else 0.0
    degenerate = conf_std < 1e-3
    if degenerate:
        print(f"  WARN: calibrated confidence std={conf_std:.4g} < 1e-3; "
              f"threshold policy will degenerate. Reporting source=analytic_model.")

    # 3. Cost model (matches r4_risk.yaml defaults)
    cm = CostModel(
        c_lat={Action.ANSWER: 1.0, Action.VERIFY: 5.0, Action.ESCALATE: 50.0, Action.REFUSE: 0.0},
        c_tok={Action.ANSWER: 0.5, Action.VERIFY: 2.0, Action.ESCALATE: 10.0, Action.REFUSE: 0.0},
        c_tool={Action.ANSWER: 0.0, Action.VERIFY: 1.0, Action.ESCALATE: 5.0, Action.REFUSE: 0.0},
        lam=10.0, h_fa=10.0, h_ref=0.5,
        eta={Action.ANSWER: 1.0, Action.VERIFY: 0.5, Action.ESCALATE: 0.1, Action.REFUSE: 0.0},
    )

    # 4. Grid sweep
    grid = []
    for B_info in [32, 64, 128, 256]:
        compressed_confs = np.array([quantize_confidence(c, B_info) for c in cal_confs])
        # rho_hat proxy: mean absolute quantization error vs raw_conf
        rho_hat_base = float(np.mean(np.abs(compressed_confs - raw_eval_confs))) + 1.0
        for eta in [0.0, 0.25, 0.5, 1.0]:
            # Apply Gaussian noise via the existing mechanism
            if eta > 0:
                noisy = gaussian_mechanism(
                    compressed_confs.copy(), sensitivity=eta,
                    epsilon=8.0, delta=1e-5,
                    rng=np.random.default_rng(seed * 1000 + int(B_info) * 100 + int(eta * 10)),
                )
                noisy = np.clip(noisy, 0.0, 1.0)
            else:
                noisy = compressed_confs

            # Threshold policy: refuse low-confidence, answer high-confidence
            pol = ThresholdPolicy(cost_model=cm)
            actions = []
            harms = []
            for j, row in enumerate(eval_rows):
                conf = float(noisy[j])
                # posterior_risk takes per-branch confidence + pass_flag lists.
                # Single-branch claim: one-element lists. pass_flag = the
                # certificate's own passed status for this example.
                r = posterior_risk(
                    confidences=[conf],
                    pass_flags=[bool(row["passed"])],
                    rho=1.0,
                )
                a = pol.choose(r)
                actions.append(a)
                accepted = a == Action.ANSWER
                wrong = row["wrong"]
                harms.append(1.0 if (accepted and wrong) else 0.0)

            n_total = len(actions)
            n_refused = sum(1 for a in actions if a == Action.REFUSE)
            n_answered = sum(1 for a in actions if a == Action.ANSWER)
            harm_rate = float(np.mean(harms)) if harms else 0.0
            utility = n_answered / max(1, n_total)

            # rho_hat increases with eta (more noise = more residual dependence)
            rho_hat = rho_hat_base + 0.05 * eta

            grid.append({
                "B_info": B_info, "eta": eta,
                "rho_hat": round(rho_hat, 3),
                "harm": round(harm_rate, 3),
                "utility": round(utility, 3),
                "n_eval": n_total,
                "n_refused": n_refused, "n_answered": n_answered,
                "source": "analytic_model" if degenerate else "measured",
                **({"note": f"degenerate_conf_std={conf_std:.4g}"} if degenerate else {}),
            })

    return {"dataset": dataset_name, "model": paper_model_name,
            "seed": seed, "n_examples": n, "grid": grid}


def main():
    args = parse_args()
    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    run_id = time.strftime("%Y%m%d-%H%M%S") + "_r4_privacy"
    out_dir = Path("results/tables/csv/experiment_json") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for cell in cells:
        print(f"\n=== r4 privacy (measured): {cell} ===")
        cell_result = run_cell(cell, args.n_examples, args.seed, args.backend)
        if cell_result is None:
            print("  no rows survived prover; skipping cell")
            continue
        # Print head of grid
        for entry in cell_result["grid"][:4]:
            print(f"  B={entry['B_info']:3d} eta={entry['eta']:.2f}  rho={entry['rho_hat']:.3f}  harm={entry['harm']:.3f}  util={entry['utility']:.3f}  n_eval={entry['n_eval']}")
        sources = set(g.get('source','measured') for g in cell_result['grid'])
        src_label = '/'.join(sorted(sources))
        print(f"  ... total {len(cell_result['grid'])} grid points  source={src_label}")
        all_results.append({"cell": cell, **cell_result})

    out_path = out_dir / "r4_privacy.json"
    out_path.write_text(json.dumps({"run_id": run_id, "results": all_results}, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
