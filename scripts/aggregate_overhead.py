from __future__ import annotations

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


def stage_series(df: pd.DataFrame, stage_name: str) -> pd.Series:
    if stage_name not in df.columns:
        return pd.Series([0.0] * len(df))
    return df[stage_name].apply(get_wall_ms)


def safe_series(df: pd.DataFrame, col: str, default=0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series([default] * len(df))
    return df[col]


def q(series: pd.Series, quantile: float) -> float:
    if len(series) == 0:
        return 0.0
    return float(np.quantile(series.astype(float), quantile))


def main():
    run_dir = Path("outputs/runs")
    paths = sorted(run_dir.glob("*.jsonl"))

    rows = []

    for p in paths:
        df = load_jsonl(p)
        if len(df) == 0:
            continue

        generation_ms = stage_series(df, "generation")
        certificate_ms = stage_series(df, "certificate")
        replay_ms = stage_series(df, "replay")
        verifier_ms = stage_series(df, "verifier")

        total_wall = safe_series(df, "total_wall_ms", 0.0).astype(float)
        total_tokens = safe_series(df, "total_tokens", 0.0).astype(float)
        prompt_tokens = safe_series(df, "total_prompt_tokens", 0.0).astype(float)
        completion_tokens = safe_series(df, "total_completion_tokens", 0.0).astype(float)
        accepted = safe_series(df, "accepted", False).fillna(False).astype(bool)
        answer_correct = safe_series(df, "answer_correct", False).fillna(False).astype(bool)

        stage_total_mean = (
            generation_ms.mean()
            + certificate_ms.mean()
            + replay_ms.mean()
            + verifier_ms.mean()
        )

        def share(x):
            return float(x.mean() / stage_total_mean) if stage_total_mean > 0 else 0.0

        rows.append({
            "run_file": p.name,
            "dataset": df["dataset"].iloc[0] if "dataset" in df.columns else "",
            "backbone": df["backbone"].iloc[0] if "backbone" in df.columns else "",
            "mode": df["mode"].iloc[0] if "mode" in df.columns else "",
            "n": len(df),

            "acceptance_rate": float(accepted.mean()),
            "answer_accuracy": float(answer_correct.mean()),
            "accepted_accuracy": float(answer_correct[accepted].mean()) if accepted.any() else 0.0,

            "mean_prompt_tokens": float(prompt_tokens.mean()),
            "mean_completion_tokens": float(completion_tokens.mean()),
            "mean_total_tokens": float(total_tokens.mean()),
            "median_total_tokens": float(total_tokens.median()),
            "p95_total_tokens": q(total_tokens, 0.95),

            "mean_latency_query_ms": float(total_wall.mean()),
            "median_latency_query_ms": float(total_wall.median()),
            "p95_latency_query_ms": q(total_wall, 0.95),
            "mean_latency_accepted_ms": float(total_wall[accepted].mean()) if accepted.any() else 0.0,

            "generation_ms": float(generation_ms.mean()),
            "certificate_ms": float(certificate_ms.mean()),
            "replay_ms": float(replay_ms.mean()),
            "verifier_ms": float(verifier_ms.mean()),

            "generation_share": share(generation_ms),
            "certificate_share": share(certificate_ms),
            "replay_share": share(replay_ms),
            "verifier_share": share(verifier_ms),
        })

    out = pd.DataFrame(rows)

    if len(out) == 0:
        print("No run files found in outputs/runs")
        return

    baseline = out[out["mode"] == "baseline_posthoc_verify"][["dataset", "backbone", "mean_total_tokens"]].copy()
    baseline = baseline.rename(columns={"mean_total_tokens": "baseline_total_tokens"})

    out = out.merge(baseline, on=["dataset", "backbone"], how="left")
    out["token_overhead_ratio_vs_posthoc"] = out["mean_total_tokens"] / out["baseline_total_tokens"]

    out_path = Path("outputs/tables/overhead_main.csv")
    out.to_csv(out_path, index=False)

    print(out.to_string(index=False))
    print(f"Saved overhead table to: {out_path}")


if __name__ == "__main__":
    main()
