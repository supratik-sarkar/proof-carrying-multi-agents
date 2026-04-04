from __future__ import annotations

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


def stage_mean(df: pd.DataFrame, stage_name: str) -> float:
    if stage_name not in df.columns:
        return 0.0
    return float(df[stage_name].apply(get_wall_ms).mean())


def safe_mean(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return 0.0
    return float(df[col].mean())


def main():
    run_dir = Path("outputs/runs")
    paths = sorted(run_dir.glob("*.jsonl"))

    rows = []
    for p in paths:
        df = load_jsonl(p)
        if len(df) == 0:
            continue

        accepted_mean = None
        if "accepted" in df.columns and df["accepted"].any() and "total_wall_ms" in df.columns:
            accepted_mean = float(df.loc[df["accepted"], "total_wall_ms"].mean())

        rows.append(
            {
                "run_file": p.name,
                "dataset": df["dataset"].iloc[0] if "dataset" in df.columns else "",
                "backbone": df["backbone"].iloc[0] if "backbone" in df.columns else "",
                "mode": df["mode"].iloc[0] if "mode" in df.columns else "",
                "n": len(df),
                "acceptance_rate": float(df["accepted"].mean()) if "accepted" in df.columns else 0.0,
                "answer_accuracy": float(df["answer_correct"].fillna(False).mean()) if "answer_correct" in df.columns else 0.0,
                "mean_prompt_tokens": safe_mean(df, "total_prompt_tokens"),
                "mean_completion_tokens": safe_mean(df, "total_completion_tokens"),
                "mean_total_tokens": safe_mean(df, "total_tokens"),
                "mean_latency_query_ms": safe_mean(df, "total_wall_ms"),
                "mean_latency_accepted_ms": accepted_mean,
                "generation_ms": stage_mean(df, "generation"),
                "certificate_ms": stage_mean(df, "certificate"),
                "replay_ms": stage_mean(df, "replay"),
                "verifier_ms": stage_mean(df, "verifier"),
            }
        )

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
