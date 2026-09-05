#!/usr/bin/env python3
"""R5b scaling slopes runner.

Three sweeps:
  k     (redundancy):  measured from existing Phase 7 R5 outputs (no re-run).
  |S0|  (support):     measured here via top_k ∈ {2,4,8,16,32} on one cell.
  d     (chain depth): analytic_model only.

Note (d-scaling): No native chained prover is used in the current runtime;
d-scaling is reported as an analytic model unless a future multi-pass
refinement loop is enabled.

Output:
  results/tables/csv/experiment_json/<run_id>/r5_scaling.json
"""
from __future__ import annotations
import argparse, json, math, sys, time
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
    p.add_argument("--s0-sweep", type=str, default="2,4,8,16,32",
                   help="Comma-sep top_k values to measure.")
    return p.parse_args()


def fit_slope_loglog(xs, ys):
    pairs = [(math.log(x), math.log(y)) for x, y in zip(xs, ys) if x > 0 and y > 0]
    if len(pairs) < 2:
        return None
    n = len(pairs)
    sx = sum(a for a, _ in pairs); sy = sum(b for _, b in pairs)
    sxx = sum(a*a for a, _ in pairs); sxy = sum(a*b for a, b in pairs)
    denom = n*sxx - sx*sx
    if denom == 0:
        return None
    return (n*sxy - sx*sy) / denom


def load_measured_k_sweep():
    """Read existing R5 overhead outputs and fit slopes vs k."""
    base = Path("results/tables/csv/experiment_json")
    r5_paths = sorted(base.glob("*r5_overhead*/r5.json"))
    if not r5_paths:
        return None
    import re
    data = json.loads(r5_paths[-1].read_text())
    agg = data.get("aggregated", [])
    ks, toks, lats = [], [], []
    for entry in agg:
        cname = str(entry.get("config", "")).lower()
        m = re.search(r"k(\d+)", cname)
        if not m:
            continue
        ks.append(int(m.group(1)))
        toks.append(float(entry.get("tokens_per_claim_mean", 0)))
        lats.append(float(entry.get("wall_ms_per_claim_mean", 0)))
    if len(ks) < 2:
        return None
    return {
        "variable": "k", "sweep": ks, "tokens": toks, "latency_ms": lats,
        "token_slope": fit_slope_loglog(ks, toks),
        "latency_slope": fit_slope_loglog(ks, lats),
        "source": "measured", "source_file": str(r5_paths[-1]),
    }


def measure_s0_sweep(cell_spec, n, seed, s0_values, backend_kind='hf_local'):
    """Measured |S0| sweep: run prover at each top_k, measure tokens + latency."""
    from scripts.experiments.run_ablations import resolve_cells
    from scripts.common.experiment_io import build_backend
    from pcg.datasets import load_dataset_by_name
    from pcg.orchestrator.langgraph_flow import PCGState
    from pcg.agents.prover import ProverConfig, build_default_prover

    dataset_name, paper_model_name, hf_model_name = resolve_cells(cell_spec)[0]
    backend_cfg = {"backend": {"kind": backend_kind, "model_name": hf_model_name,
                                "dtype": "float16", "load_in_4bit": False,
                                "trust_remote_code": True}}
    backend_obj = build_backend(backend_cfg, override=backend_kind)
    examples = list(load_dataset_by_name(dataset_name, split="validation",
                                          n_examples=n, seed=seed))

    # Guard: if examples have <=2 evidence items, |S0| sweep is degenerate
    # (retrieval saturates at the dataset's evidence cap, slope is meaningless).
    # Emit source=analytic_model + explanatory note rather than fake measured.
    max_evidence = max((len(ex.evidence) for ex in examples), default=0)
    if max_evidence <= 2:
        return {
            "variable": "support_size",
            "sweep": s0_values, "tokens": None, "latency_ms": None,
            "token_slope": None, "latency_slope": None,
            "source": "analytic_model",
            "note": (f"dataset {dataset_name!r} provides max_evidence={max_evidence} "
                     f"per example; |S0| sweep is degenerate (retrieval saturates "
                     f"below smallest sweep value). Use a multi-evidence dataset "
                     f"(e.g. hotpotqa, pubmedqa) for measured |S0| scaling."),
            "max_evidence_in_dataset": max_evidence,
        }

    sweep_results = []
    for top_k in s0_values:
        token_totals, latency_totals = [], []
        for i, ex in enumerate(examples):
            pcfg = ProverConfig(top_k=top_k, temperature=0.0, seed=seed * 100 + i)
            prover_fn = build_default_prover(backend=backend_obj, config=pcfg)
            state = PCGState(example=ex)
            t0 = time.time()
            try:
                state = prover_fn(state)
            except Exception as exc:
                print(f"    top_k={top_k} ex={i}: ERROR {type(exc).__name__}: {str(exc)[:60]}")
                continue
            dt_ms = (time.time() - t0) * 1000.0
            phases = state.meter.report().phases
            tok_in = sum(p.total_tokens_in for p in phases.values())
            tok_out = sum(p.total_tokens_out for p in phases.values())
            token_totals.append(tok_in + tok_out)
            latency_totals.append(dt_ms)
        if not token_totals:
            continue
        sweep_results.append({
            "top_k": top_k,
            "tokens_mean": sum(token_totals) / len(token_totals),
            "latency_ms_mean": sum(latency_totals) / len(latency_totals),
            "n_examples": len(token_totals),
        })
        print(f"    |S0|={top_k:2d}  tokens={sweep_results[-1]['tokens_mean']:.0f}  latency={sweep_results[-1]['latency_ms_mean']:.0f}ms  n={sweep_results[-1]['n_examples']}")

    if len(sweep_results) < 2:
        return None
    s0s = [r["top_k"] for r in sweep_results]
    toks = [r["tokens_mean"] for r in sweep_results]
    lats = [r["latency_ms_mean"] for r in sweep_results]
    return {
        "variable": "support_size",
        "sweep": s0s, "tokens": toks, "latency_ms": lats,
        "token_slope": fit_slope_loglog(s0s, toks),
        "latency_slope": fit_slope_loglog(s0s, lats),
        "source": "measured",
        "n_examples_per_point": n,
    }


def main():
    args = parse_args()
    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    s0_values = [int(x) for x in args.s0_sweep.split(",") if x.strip()]

    run_id = time.strftime("%Y%m%d-%H%M%S") + "_r5_scaling"
    out_dir = Path("results/tables/csv/experiment_json") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for cell in cells:
        print(f"\n=== r5 scaling: {cell} ===")

        k = load_measured_k_sweep()
        if k:
            print(f"  k (measured)  tokens slope={k['token_slope']:.3f}  latency slope={k['latency_slope']:.3f}")
        else:
            print(f"  k             MISSING (no r5_overhead outputs found)")

        print(f"  |S0| (measured) sweep over top_k = {s0_values}")
        s0 = measure_s0_sweep(cell, args.n_examples, args.seed, s0_values, args.backend)
        if s0:
            ts = s0.get('token_slope'); ls = s0.get('latency_slope')
            ts_str = f"{ts:.3f}" if ts is not None else "None (analytic_model)"
            ls_str = f"{ls:.3f}" if ls is not None else "None (analytic_model)"
            note = s0.get('note', '')
            print(f"  |S0| ({s0.get('source','measured')})  tokens slope={ts_str}  latency slope={ls_str}")
            if note:
                print(f"       note: {note}")
        else:
            s0 = {"variable": "support_size", "sweep": s0_values,
                   "token_slope": None, "latency_slope": None,
                   "source": "missing", "note": "no_S0_data_collected"}
            print(f"  |S0|           MISSING (sweep failed)")

        d = {
            "variable": "chain_depth", "sweep": [2, 5, 10, 20],
            "token_slope": None, "latency_slope": None,
            "source": "analytic_model",
            "note": ("No native chained prover is used in the current runtime; "
                     "d-scaling is reported as an analytic model unless a future "
                     "multi-pass refinement loop is enabled."),
        }
        print(f"  d              source=analytic_model  slopes=None")

        all_results.append({"cell": cell, "k": k, "s0": s0, "d": d})

    out_path = out_dir / "r5_scaling.json"
    out_path.write_text(json.dumps({"run_id": run_id, "results": all_results},
                                     indent=2, default=str))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
