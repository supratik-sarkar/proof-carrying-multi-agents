from __future__ import annotations

import argparse
import json
from pathlib import Path

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

    def stage_mean(stage_name: str):
        if stage_name not in df.columns:
            return 0.0
        return float(df[stage_name].apply(get_wall_ms).mean())

    out = {
        "dataset": df["dataset"].iloc[0] if "dataset" in df.columns else "",
        "backbone": df["backbone"].iloc[0] if "backbone" in df.columns else "",
        "mode": df["mode"].iloc[0] if "mode" in df.columns else "",
        "n": len(df),
        "acceptance_rate": float(df["accepted"].mean()) if "accepted" in df.columns else 0.0,
        "answer_accuracy": float(df["answer_correct"].fillna(False).mean()) if "answer_correct" in df.columns else 0.0,
        "mean_total_tokens": float(df["total_tokens"].mean()) if "total_tokens" in df.columns else 0.0,
        "mean_total_wall_ms": float(df["total_wall_ms"].mean()) if "total_wall_ms" in df.columns else 0.0,
        "generation_ms": stage_mean("generation"),
        "certificate_ms": stage_mean("certificate"),
        "replay_ms": stage_mean("replay"),
        "verifier_ms": stage_mean("verifier"),
    }

    out_df = pd.DataFrame([out])
    out_path = Path("outputs/tables") / (Path(args.run).stem + "_summary.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(out_df.to_string(index=False))
    print(f"Saved summary to: {out_path}")


if __name__ == "__main__":
    main()
