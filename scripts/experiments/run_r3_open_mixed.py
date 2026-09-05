#!/usr/bin/env python3
"""R3b open-set / mixed-channel diagnosis runner.

Extends R3 responsibility evaluation in two directions:

  open-set: corrupt an ActionNode (a component OUTSIDE the closed
            {V_H, V_Pi, V_Gamma, V_entail} channel taxonomy). A
            well-calibrated estimator returns low confidence rather
            than confidently mis-attributing to a closed-set channel.
            Score: unknown_acc = fraction of examples where the
            top-1 |Resp| margin is below an "unknown" threshold.

  mixed-channel: corrupt TWO components in different channels
            (evidence + schema). A well-calibrated estimator should
            rank BOTH targets in its top-2 by |Resp|. Score: multi_label_f1
            computed over (predicted top-2, actual two targets).

Output:
  results/tables/csv/experiment_json/<run_id>/r3_open_mixed.json
"""
from __future__ import annotations
import argparse, json, sys, time, random
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
    p.add_argument(
        "--unknown-margin-threshold", type=float, default=0.5,
        help="Top-1 |Resp| margin below this counts as 'unknown' attribution.",
    )
    return p.parse_args()


def run_one_regime(cell_spec, n, seed, attack_kind, unknown_thresh, backend_kind='hf_local'):
    """Run n examples under a single attack regime; return per-example records."""
    from scripts.experiments.run_ablations import resolve_cells
    from scripts.common.experiment_io import build_backend
    from pcg.datasets import load_dataset_by_name
    from pcg.checker import Checker, TokenOverlapEntailment
    from pcg.orchestrator import build_replayer_with_handlers
    from pcg.orchestrator.langgraph_flow import OrchestratorConfig, run_one_example
    from pcg.agents.prover import ProverConfig, build_default_prover
    from pcg.agents.attacker import build_default_attacker
    from pcg.agents.debugger import build_default_debugger
    from pcg.responsibility import ResponsibilityEstimator

    dataset_name, paper_model_name, hf_model_name = resolve_cells(cell_spec)[0]
    backend_cfg = {"backend": {"kind": backend_kind, "model_name": hf_model_name,
                                "dtype": "float16", "load_in_4bit": False,
                                "trust_remote_code": True}}
    backend_obj = build_backend(backend_cfg, override=backend_kind)
    checker = Checker(entailment=TokenOverlapEntailment(threshold=0.5),
                       replayer=build_replayer_with_handlers())
    examples = list(load_dataset_by_name(dataset_name, split="validation",
                                          n_examples=n, seed=seed))

    records = []
    for i, ex in enumerate(examples):
        cfg = OrchestratorConfig(
            enable_attacker=True, enable_debugger=True, max_retries=0,
            attack_kind=attack_kind, epsilon_adv=0.25,
        )
        try:
            state = run_one_example(ex, backend=backend_obj, checker=checker, cfg=cfg)
        except Exception as exc:
            records.append({
                "example_id": ex.id, "regime": attack_kind,
                "error": f"{type(exc).__name__}:{str(exc)[:80]}",
                "per_component": [],
            })
            continue

        # Pull the responsibility estimates the debugger produced.
        resp_dict = dict(state.responsibility or {})
        per_component = [
            {"component_id": k, "resp": float(v)}
            for k, v in resp_dict.items()
        ]
        per_component.sort(key=lambda x: abs(x["resp"]), reverse=True)

        # Targets: read directly from state.meta["attacks"] (the attacker
        # records every attack with kind + desc; desc has format
        # "<kind>:<id1>,<id2>,..." per attacker.py).
        targets = []
        for atk in state.meta.get("attacks", []):
            desc = atk.get("desc", "")
            if ":" in desc:
                targets = [t.strip() for t in desc.split(":", 1)[1].split(",") if t.strip()]
                break

        top1 = per_component[0] if per_component else None
        top2 = per_component[:2]
        top1_margin = (abs(top1["resp"]) - abs(per_component[1]["resp"])) if len(per_component) >= 2 \
                      else (abs(top1["resp"]) if top1 else 0.0)

        # Open-set metric: unknown_correct if no confident top-1
        unknown_correct = (top1_margin < unknown_thresh) if attack_kind == "action_replay" \
                          else None

        # Mixed-channel metric: count overlap between top-2 and targets
        top2_ids = set(t["component_id"] for t in top2)
        tp = len(top2_ids & set(targets))
        fp = len(top2_ids - set(targets))
        fn = len(set(targets) - top2_ids)

        records.append({
            "example_id": ex.id, "regime": attack_kind,
            "per_component": per_component,
            "top1_component_id": top1["component_id"] if top1 else None,
            "top1_resp": top1["resp"] if top1 else None,
            "top1_margin": top1_margin,
            "targets": targets,
            "unknown_correct": unknown_correct,
            "multilabel_tp": tp, "multilabel_fp": fp, "multilabel_fn": fn,
        })

    return {"dataset": dataset_name, "model": paper_model_name,
            "regime": attack_kind, "records": records}


def summarize(open_block, mixed_block):
    """Compute open_top2, multi_label_f1, unknown_acc, closed_top1 (proxy)."""
    summary = {}

    # Open-set: unknown_acc on action_replay examples
    if open_block:
        recs = open_block["records"]
        n_open = len(recs)
        n_unknown_correct = sum(1 for r in recs if r.get("unknown_correct") is True)
        summary["unknown_acc"] = n_unknown_correct / max(1, n_open)
        summary["n_open_examples"] = n_open
    else:
        summary["unknown_acc"] = None

    # Mixed-channel: multi-label F1 + open_top2 (= top-2 recall against targets)
    if mixed_block:
        recs = mixed_block["records"]
        n_mixed = len(recs)
        tp = sum(r.get("multilabel_tp", 0) for r in recs)
        fp = sum(r.get("multilabel_fp", 0) for r in recs)
        fn = sum(r.get("multilabel_fn", 0) for r in recs)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        summary["multi_label_f1"] = f1
        summary["open_top2"] = sum(
            1 for r in recs if r.get("multilabel_tp", 0) >= 1
        ) / max(1, n_mixed)
        summary["n_mixed_examples"] = n_mixed
    else:
        summary["multi_label_f1"] = None
        summary["open_top2"] = None

    return summary


def main():
    args = parse_args()
    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    run_id = time.strftime("%Y%m%d-%H%M%S") + "_r3_open_mixed"
    out_dir = Path("results/tables/csv/experiment_json") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for cell in cells:
        print(f"\n=== r3 open/mixed: {cell} ===")
        open_block = run_one_regime(cell, args.n_examples, args.seed,
                                     "action_replay", args.unknown_margin_threshold, args.backend)
        print(f"  open-set (action_replay): n={len(open_block['records'])}")

        mixed_block = run_one_regime(cell, args.n_examples, args.seed,
                                      "mixed_channel", args.unknown_margin_threshold, args.backend)
        print(f"  mixed-channel:            n={len(mixed_block['records'])}")

        summary = summarize(open_block, mixed_block)
        print(f"  unknown_acc    = {summary.get('unknown_acc')}")
        print(f"  multi_label_f1 = {summary.get('multi_label_f1')}")
        print(f"  open_top2      = {summary.get('open_top2')}")

        all_results.append({
            "dataset": open_block["dataset"], "model": open_block["model"],
            "seed": args.seed, "n_examples": args.n_examples,
            "open_set": open_block, "mixed_channel": mixed_block,
            "summary": summary, "source": "measured",
        })

    out_path = out_dir / "r3_open_mixed.json"
    out_path.write_text(json.dumps({"run_id": run_id, "results": all_results},
                                     indent=2, default=str))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
