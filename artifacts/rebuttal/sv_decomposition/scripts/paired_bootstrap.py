#!/usr/bin/env python3
"""Joint paired bootstrap confidence intervals over immutable example IDs."""

import argparse
import json
import sys
import numpy as np
from pathlib import Path

def run_paired_bootstrap(source_records_path, n_bootstraps=1000, seed=42, output_path=None):
    source_p = Path(source_records_path)
    if not source_p.exists():
        raise FileNotFoundError(f"Source records not found: {source_p}")

    records = [json.loads(line) for line in source_p.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError("Empty source records file.")

    rng = np.random.RandomState(seed)
    N = len(records)

    s_boots, v_boots = [], []

    for _ in range(n_bootstraps):
        idxs = rng.choice(N, size=N, replace=True)
        sampled_recs = [records[i] for i in idxs]

        all_l_nc = [1.0 if r.get("systems", {}).get("NoCert", {}).get("composite_harm", False) else 0.0 for r in sampled_recs]
        accepted_pcg = [r for r in sampled_recs if r.get("systems", {}).get("PCG-MAS", {}).get("accepted", False)]

        A_l_nc = [1.0 if r.get("systems", {}).get("NoCert", {}).get("composite_harm", False) else 0.0 for r in accepted_pcg]
        A_l_pcg = [1.0 if (r.get("systems", {}).get("PCG-MAS", {}).get("accepted", False) and r.get("systems", {}).get("PCG-MAS", {}).get("composite_harm", False)) else 0.0 for r in accepted_pcg]

        I_nc_all = np.mean(all_l_nc) if all_l_nc else 0.0
        H_nc_A = np.mean(A_l_nc) if A_l_nc else 0.0
        H_pcg_A = np.mean(A_l_pcg) if A_l_pcg else 0.0

        s_boots.append(float(I_nc_all - H_nc_A))
        v_boots.append(float(H_nc_A - H_pcg_A))

    res = {
        "n_bootstraps": n_bootstraps,
        "seed": seed,
        "S_mean": round(float(np.mean(s_boots)), 4),
        "S_ci_low": round(float(np.percentile(s_boots, 2.5)), 4),
        "S_ci_high": round(float(np.percentile(s_boots, 97.5)), 4),
        "V_mean": round(float(np.mean(v_boots)), 4),
        "V_ci_low": round(float(np.percentile(v_boots, 2.5)), 4),
        "V_ci_high": round(float(np.percentile(v_boots, 97.5)), 4)
    }

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")

    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run joint paired bootstrap resampling.")
    parser.add_argument("--source-records", required=True)
    parser.add_argument("--n-bootstraps", type=int, default=1000)
    parser.add_argument("--output", required=False)
    args = parser.parse_args()

    res = run_paired_bootstrap(args.source_records, args.n_bootstraps, output_path=args.output)
    print(f"Paired bootstrap completed: S_mean={res['S_mean']}, V_mean={res['V_mean']}")
