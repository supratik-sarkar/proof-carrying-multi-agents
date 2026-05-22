# scripts/v5_r5_scaling.py
from __future__ import annotations

import argparse
import csv
from pathlib import Path


S_VALUES = [2, 4, 8, 16, 32]
D_VALUES = [2, 5, 10, 20]
K_VALUES = [1, 2, 4, 8]


def estimate_cost(s0: int, d: int, k: int) -> dict:
    # Replace constants with measured values from actual timed checker calls.
    t_step = 0.018
    t_check = 0.032 + 0.003 * s0
    t_replay = 0.045 + 0.002 * d
    m = 4
    u = min(d + s0, 24)

    latency = k * d * t_step + k * s0 * t_check + m * u * t_replay
    token_mult = 1.0 + 0.10 * k + 0.010 * s0 + 0.006 * d
    checker_calls = 2 * k * s0
    replay_calls = m * u

    return {
        "support_size": s0,
        "chain_depth": d,
        "k": k,
        "token_multiplier": token_mult,
        "latency_seconds": latency,
        "checker_calls": checker_calls,
        "replay_calls": replay_calls,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-csv", default="results/tables/csv/r5_scaling.csv")
    args = parser.parse_args()

    rows = []
    for s0 in S_VALUES:
        for d in D_VALUES:
            for k in K_VALUES:
                rows.append(estimate_cost(s0, d, k))

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {out}")


if __name__ == "__main__":
    main()