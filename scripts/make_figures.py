from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PRIMARY = "#4C78A8"
ACCENT = "#F58518"
NEUTRAL = "#9AA1A9"
DARK = "#222222"
LIGHT_GRID = "#D9D9D9"


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
    return df[stage_name].apply(get_wall_ms).astype(float)


def safe_series(df: pd.DataFrame, col: str, default=0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series([default] * len(df))
    return df[col]


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.35, color=LIGHT_GRID)
    ax.set_axisbelow(True)


def add_footer(fig, text: str):
    fig.text(
        0.5,
        0.01,
        text,
        ha="center",
        va="bottom",
        fontsize=9,
        wrap=True,
    )


def save_fig(fig, path: Path):
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, required=True)
    args = parser.parse_args()

    run_path = Path(args.run)
    df = load_jsonl(run_path)

    fig_dir = Path("outputs/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    dataset = df["dataset"].iloc[0] if "dataset" in df.columns else "unknown"
    mode = df["mode"].iloc[0] if "mode" in df.columns else "unknown"
    backbone = df["backbone"].iloc[0] if "backbone" in df.columns else "unknown"

    generation_ms = stage_series(df, "generation")
    certificate_ms = stage_series(df, "certificate")
    replay_ms = stage_series(df, "replay")
    verifier_ms = stage_series(df, "verifier")

    total_tokens = safe_series(df, "total_tokens", 0.0).astype(float)
    accepted = safe_series(df, "accepted", False).fillna(False).astype(bool)
    answer_correct = safe_series(df, "answer_correct", False).fillna(False).astype(bool)
    risk = safe_series(df, "risk_cal", 0.0).astype(float)

    stage_names = ["Generation", "Certificate", "Replay", "Verifier"]
    stage_means = np.array([
        float(generation_ms.mean()),
        float(certificate_ms.mean()),
        float(replay_ms.mean()),
        float(verifier_ms.mean()),
    ])

    stage_display = np.maximum(stage_means, 0.03)

    # 1. Latency breakdown: true values labeled, visual floor only for display
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    colors = [PRIMARY, NEUTRAL, NEUTRAL, ACCENT]
    bars = ax.bar(stage_names, stage_display, color=colors, edgecolor=DARK, linewidth=0.8)
    style_axes(ax)
    ax.set_ylabel("Mean latency (ms)")
    ax.set_title(f"Latency Breakdown by Stage\n{dataset} | {mode} | {backbone}", fontsize=14, pad=14)

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=PRIMARY, edgecolor=DARK, label="Main generation stage"),
        Patch(facecolor=NEUTRAL, edgecolor=DARK, label="Certificate / replay lightweight stages"),
        Patch(facecolor=ACCENT, edgecolor=DARK, label="Verifier stage"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=False, fontsize=9)

    for bar, true_val in zip(bars, stage_means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{true_val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=DARK,
        )

    footer = (
        "Objective: decompose end-to-end latency into generation, certificate construction, replay checking, and verifier scoring. "
        "Small stages are shown with a visibility floor for readability, while labels report the true measured values."
    )
    add_footer(fig, footer)
    out1 = fig_dir / f"{run_path.stem}_latency_breakdown.png"
    save_fig(fig, out1)

    # 2. Latency distribution boxplot across stages
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    box_data = [generation_ms, certificate_ms, replay_ms, verifier_ms]
    bp = ax.boxplot(
        box_data,
        patch_artist=True,
        labels=stage_names,
        showfliers=True,
        medianprops=dict(color=DARK, linewidth=1.3),
        boxprops=dict(edgecolor=DARK, linewidth=0.8),
        whiskerprops=dict(color=DARK, linewidth=0.8),
        capprops=dict(color=DARK, linewidth=0.8),
    )
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(PRIMARY if i == 0 else (ACCENT if i == 3 else "#DCE6F2"))
        patch.set_alpha(0.9)

    style_axes(ax)
    ax.set_ylabel("Per-example latency (ms)")
    ax.set_title(f"Latency Distribution Across Stages\n{dataset} | {mode} | {backbone}", fontsize=14, pad=14)

    footer = (
        "Objective: show not only mean latency, but also variability across examples. "
        "This reveals whether runtime is stable or driven by a few outlier cases."
    )
    add_footer(fig, footer)
    out2 = fig_dir / f"{run_path.stem}_latency_boxplot.png"
    save_fig(fig, out2)

    # 3. Token distribution with density-style clarity and summary lines
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    bins = min(12, max(5, len(total_tokens)))
    ax.hist(total_tokens, bins=bins, color=PRIMARY, alpha=0.85, edgecolor=DARK, linewidth=0.8)

    mean_tokens = float(total_tokens.mean()) if len(total_tokens) else 0.0
    median_tokens = float(total_tokens.median()) if len(total_tokens) else 0.0
    p95_tokens = float(np.quantile(total_tokens, 0.95)) if len(total_tokens) else 0.0

    ax.axvline(mean_tokens, color=ACCENT, linestyle="--", linewidth=2.0, label=f"Mean = {mean_tokens:.1f}")
    ax.axvline(median_tokens, color=DARK, linestyle="-.", linewidth=1.8, label=f"Median = {median_tokens:.1f}")
    ax.axvline(p95_tokens, color=NEUTRAL, linestyle=":", linewidth=2.2, label=f"P95 = {p95_tokens:.1f}")

    style_axes(ax)
    ax.set_xlabel("Total tokens per example")
    ax.set_ylabel("Count")
    ax.set_title(f"Token Distribution\n{dataset} | {mode} | {backbone}", fontsize=14, pad=14)
    ax.legend(loc="upper right", frameon=False, fontsize=9)

    footer = (
        "Objective: characterize token-cost distribution. "
        "Mean, median, and P95 indicate typical usage and whether a long-tail of expensive examples exists."
    )
    add_footer(fig, footer)
    out3 = fig_dir / f"{run_path.stem}_token_distribution.png"
    save_fig(fig, out3)

    # 4. Quality overview with minimal, meaningful color usage
    acceptance_rate = float(accepted.mean()) if len(accepted) else 0.0
    overall_accuracy = float(answer_correct.mean()) if len(answer_correct) else 0.0
    accepted_accuracy = float(answer_correct[accepted].mean()) if accepted.any() else 0.0
    verifier_confidence = 1.0 - float(risk.mean()) if len(risk) else 0.0

    metrics = ["Acceptance", "Overall Accuracy", "Accepted Accuracy", "Verifier Confidence"]
    values = [acceptance_rate, overall_accuracy, accepted_accuracy, verifier_confidence]
    metric_colors = [PRIMARY, NEUTRAL, ACCENT, PRIMARY]

    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    bars = ax.bar(metrics, values, color=metric_colors, edgecolor=DARK, linewidth=0.8)
    style_axes(ax)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Rate / score")
    ax.set_title(f"Quality Overview\n{dataset} | {mode} | {backbone}", fontsize=14, pad=14)

    legend_handles = [
        Patch(facecolor=PRIMARY, edgecolor=DARK, label="Primary operational metric"),
        Patch(facecolor=ACCENT, edgecolor=DARK, label="Accepted-answer precision"),
        Patch(facecolor=NEUTRAL, edgecolor=DARK, label="Supporting reference metric"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=False, fontsize=9)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(val + 0.025, 0.98),
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color=DARK,
        )

    footer = (
        "Objective: summarize decision quality in one place. "
        "Acceptance shows selectivity, overall accuracy shows end-to-end correctness, "
        "accepted accuracy measures precision among accepted answers, and verifier confidence summarizes the risk model output."
    )
    add_footer(fig, footer)
    out4 = fig_dir / f"{run_path.stem}_quality_overview.png"
    save_fig(fig, out4)

    print("Saved figures:")
    print(out1)
    print(out2)
    print(out3)
    print(out4)


if __name__ == "__main__":
    main()
