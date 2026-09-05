#!/usr/bin/env python3
"""R1b replay drift / coverage decomposition runner.

Contrasts two modes on the same clean (non-adversarial) examples:
  snapshot mode (p_fresh=0.0): committed evidence is immutable.
  fresh    mode (p_fresh=0.3): per-evidence-node probability of byte drift
                               between commit and check time.

Drift = replay passed in snapshot mode but failed in fresh mode for the same
example. CovGap = examples that produced a correct answer (f1>=0.5) but
were rejected by the checker, normalized over n.

Output:
  results/tables/csv/experiment_json/<run_id>/replay_drift.json
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
    p.add_argument("--p-fresh", type=float, default=0.30)
    return p.parse_args()


def run_mode(cell_spec, n, seed, p_fresh, mode_label, backend_kind='hf_local'):
    """Run n clean examples under the given p_fresh and return per-example records."""
    from scripts.experiments.run_ablations import resolve_cells
    from scripts.common.experiment_io import build_backend
    from pcg.datasets import load_dataset_by_name
    from pcg.checker import Checker, TokenOverlapEntailment
    from pcg.orchestrator import build_replayer_with_handlers
    from pcg.orchestrator.langgraph_flow import OrchestratorConfig, run_one_example
    from pcg.eval.metrics import f1_score

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
            enable_attacker=False, enable_debugger=False, max_retries=0,
            epsilon_adv=0.0, p_fresh=float(p_fresh),
        )
        try:
            state = run_one_example(ex, backend=backend_obj, checker=checker, cfg=cfg)
        except Exception as exc:
            records.append({"example_id": ex.id, "mode": mode_label, "p_fresh": p_fresh,
                             "passed": False, "f1": 0.0,
                             "error": f"{type(exc).__name__}:{str(exc)[:80]}"})
            continue

        cr = state.check_result
        raw = state.meta.get("raw_answer", "")
        f1 = f1_score(raw, list(ex.gold_answers)) if ex.gold_answers else 0.0
        records.append({
            "example_id": ex.id, "mode": mode_label, "p_fresh": p_fresh,
            "passed": bool(cr.passed) if cr else False,
            "integrity_ok": bool(getattr(cr, "integrity_ok", True)) if cr else False,
            "replay_ok": bool(getattr(cr, "replay_ok", True)) if cr else False,
            "f1": float(f1), "raw_answer": raw[:80],
        })
    return dataset_name, paper_model_name, records


def summarize(records, n_examples):
    if not records:
        return {"n_examples": 0}
    n = len(records)
    n_passed = sum(1 for r in records if r.get("passed"))
    n_correct = sum(1 for r in records if r.get("f1", 0.0) >= 0.5)
    n_passed_correct = sum(1 for r in records if r.get("passed") and r.get("f1", 0.0) >= 0.5)
    n_replay_fail = sum(1 for r in records if not r.get("replay_ok", True))
    return {
        "n_examples": n,
        "pass_rate": n_passed / max(1, n),
        "correct_rate": n_correct / max(1, n),
        "replay_fail_rate": n_replay_fail / max(1, n),
        "covgap": (n_correct - n_passed_correct) / max(1, n),
    }


def main():
    args = parse_args()
    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    run_id = time.strftime("%Y%m%d-%H%M%S") + "_replay_drift"
    out_dir = Path("results/tables/csv/experiment_json") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for cell in cells:
        print(f"\n=== replay drift: {cell} ===")
        ds, model, snap_recs = run_mode(cell, args.n_examples, args.seed, 0.0, "snapshot", args.backend)
        snap_summary = summarize(snap_recs, args.n_examples)
        print(f"  snapshot pass={snap_summary['pass_rate']:.3f} "
              f"replay_fail={snap_summary['replay_fail_rate']:.3f} "
              f"covgap={snap_summary['covgap']:.3f}")

        ds, model, fresh_recs = run_mode(cell, args.n_examples, args.seed, args.p_fresh, "fresh", args.backend)
        fresh_summary = summarize(fresh_recs, args.n_examples)
        print(f"  fresh    pass={fresh_summary['pass_rate']:.3f} "
              f"replay_fail={fresh_summary['replay_fail_rate']:.3f} "
              f"covgap={fresh_summary['covgap']:.3f}")

        snap_by_id = {r["example_id"]: r for r in snap_recs}
        n_drift = sum(1 for fr in fresh_recs
                      if snap_by_id.get(fr["example_id"], {}).get("passed", False)
                      and not fr.get("passed", False))
        snap_summary["drift_rate"] = 0.0
        fresh_summary["drift_rate"] = n_drift / max(1, len(fresh_recs))

        all_results.append({
            "dataset": ds, "model": model, "seed": args.seed,
            "n_examples": args.n_examples, "p_fresh": args.p_fresh,
            "snapshot": snap_summary, "fresh": fresh_summary,
            "snap_records": snap_recs, "fresh_records": fresh_recs,
        })

    out_path = out_dir / "replay_drift.json"
    out_path.write_text(json.dumps({"run_id": run_id, "results": all_results},
                                     indent=2, default=str))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
