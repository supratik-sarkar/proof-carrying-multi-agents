from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def get_wall_ms(cell):
    if isinstance(cell, dict):
        return float(cell.get("wall_ms", 0.0))
    return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, required=True)
    args = parser.parse_args()

    df = load_jsonl(Path(args.run))
    fig_dir = Path("outputs/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    stage_names = ["generation", "certificate", "replay", "verifier"]
    stage_means = []
    for s in stage_names:
        if s in df.columns:
            stage_means.append(df[s].apply(get_wall_ms).mean())
        else:
            stage_means.append(0.0)

    plt.figure(figsize=(8, 5))
    plt.bar(stage_names, stage_means)
    plt.ylabel("Mean wall time (ms)")
    plt.title("Latency decomposition by stage")
    out1 = fig_dir / (Path(args.run).stem + "_latency_breakdown.png")
    plt.tight_layout()
    plt.savefig(out1, dpi=200)
    plt.close()

    if "total_tokens" in df.columns:
        plt.figure(figsize=(8, 5))
        plt.hist(df["total_tokens"], bins=15)
        plt.xlabel("Total tokens")
        plt.ylabel("Count")
        plt.title("Token distribution per example")
        out2 = fig_dir / (Path(args.run).stem + "_token_hist.png")
        plt.tight_layout()
        plt.savefig(out2, dpi=200)
        plt.close()
    else:
        out2 = "(no token histogram created)"

    print(f"Saved figures:\n{out1}\n{out2}")


if __name__ == "__main__":
    main()
