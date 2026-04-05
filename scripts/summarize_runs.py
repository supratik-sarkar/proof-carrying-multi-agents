from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
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


def get_series_or_default(df: pd.DataFrame, col: str, default=0.0):
    if col not in df.columns:
        return pd.Series([default] * len(df))
    return df[col]


def stage_series(df: pd.DataFrame, stage_name: str) -> pd.Series:
    if stage_name not in df.columns:
        return pd.Series([0.0] * len(df))
    return df[stage_name].apply(get_wall_ms)


def q(series: pd.Series, quantile: float) -> float:
    if len(series) == 0:
        return 0.0
    return float(np.quantile(series, quantile))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, required=True)
    args = parser.parse_args()

    run_path = Path(args.run)
    df = load_jsonl(run_path)

    generation_ms = stage_series(df, "generation")
    certificate_ms = stage_series(df, "certificate")
    replay_ms = stage_series(df, "replay")
    verifier_ms = stage_series(df, "verifier")

    total_wall = get_series_or_default(df, "total_wall_ms", 0.0).astype(float)
    total_tokens = get_series_or_default(df, "total_tokens", 0.0).astype(float)
    accepted = get_series_or_default(df, "accepted", False).fillna(False).astype(bool)
    answer_correct = get_series_or_default(df, "answer_correct", False).fillna(False).astype(bool)

    stage_total_mean = (
        generation_ms.mean()
        + certificate_ms.mean()
        + replay_ms.mean()
        + verifier_ms.mean()
    )

    def share(x):
        return float(x.mean() / stage_total_mean) if stage_total_mean > 0 else 0.0

    out = {
        "run_file": run_path.name,
        "dataset": df["dataset"].iloc[0] if "dataset" in df.columns else "",
        "backbone": df["backbone"].iloc[0] if "backbone" in df.columns else "",
        "mode": df["mode"].iloc[0] if "mode" in df.columns else "",
        "n": len(df),

        "acceptance_rate": float(accepted.mean()),
        "answer_accuracy": float(answer_correct.mean()),
        "accepted_accuracy": float(answer_correct[accepted].mean()) if accepted.any() else 0.0,

        "mean_total_tokens": float(total_tokens.mean()),
        "median_total_tokens": float(total_tokens.median()),
        "p95_total_tokens": q(total_tokens, 0.95),

        "mean_total_wall_ms": float(total_wall.mean()),
        "median_total_wall_ms": float(total_wall.median()),
        "p95_total_wall_ms": q(total_wall, 0.95),

        "generation_ms_mean": float(generation_ms.mean()),
        "certificate_ms_mean": float(certificate_ms.mean()),
        "replay_ms_mean": float(replay_ms.mean()),
        "verifier_ms_mean": float(verifier_ms.mean()),

        "generation_share": share(generation_ms),
        "certificate_share": share(certificate_ms),
        "replay_share": share(replay_ms),
        "verifier_share": share(verifier_ms),
    }

    out_df = pd.DataFrame([out])

    out_path = Path("outputs/tables") / f"{run_path.stem}_summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(out_df.to_string(index=False))
    print(f"Saved summary to: {out_path}")


if __name__ == "__main__":
    main()
