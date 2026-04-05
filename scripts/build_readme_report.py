from __future__ import annotations

from pathlib import Path
import pandas as pd


def pick_latest(pattern: str):
    files = sorted(
        Path("outputs/figures").glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return files[0] if files else None


def md_image(path: Path | None) -> str:
    if path is None:
        return "_No figure found._"
    return f"![]({str(path).replace(' ', '%20')})"


def main():
    overhead_path = Path("outputs/tables/overhead_main.csv")
    overhead_md = "_No overhead table found._"

    if overhead_path.exists():
        df = pd.read_csv(overhead_path)
        preferred_cols = [
            "dataset",
            "mode",
            "acceptance_rate",
            "answer_accuracy",
            "accepted_accuracy",
            "mean_total_tokens",
            "mean_latency_query_ms",
            "generation_ms",
            "certificate_ms",
            "replay_ms",
            "verifier_ms",
            "token_overhead_ratio_vs_posthoc",
        ]
        cols = [c for c in preferred_cols if c in df.columns]
        overhead_md = df[cols].round(4).to_markdown(index=False)

    latency_plot = pick_latest("*_latency_breakdown.png")
    latency_box = pick_latest("*_latency_boxplot.png")
    token_plot = pick_latest("*_token_distribution.png")
    quality_plot = pick_latest("*_quality_overview.png")

    lines = [
        "# Proof-Carrying Multi-Agents — READMEv1",
        "",
        "This is an interim report page generated before the Colab/GPU stage.",
        "",
        "## Included in this snapshot",
        "- runtime telemetry",
        "- certificate construction and replay checks",
        "- healthcare-oriented evaluation adapters",
        "- overhead analysis",
        "- publication-style plots and summary tables",
        "",
        "## Latest Result Highlights",
        "",
        "### 1. Latency Breakdown",
        md_image(latency_plot),
        "",
        "This figure decomposes end-to-end runtime into generation, certificate construction, replay validation, and verifier scoring.",
        "",
        "### 2. Latency Distribution Across Stages",
        md_image(latency_box),
        "",
        "This figure shows spread and outliers in latency across examples for each stage.",
        "",
        "### 3. Token Distribution",
        md_image(token_plot),
        "",
        "This figure summarizes how token usage is distributed across examples and highlights typical versus tail-heavy cost behavior.",
        "",
        "### 4. Quality Overview",
        md_image(quality_plot),
        "",
        "This figure summarizes acceptance rate, overall accuracy, accepted-answer accuracy, and verifier confidence in one place.",
        "",
        "## Aggregate Overhead Table",
        "",
        overhead_md,
        "",
        "## Reproducibility",
        "",
        "Typical commands:",
        "",
        "```bash",
        "PYTHONPATH=. python scripts/run_eval.py --config configs/medmcqa.yaml",
        "PYTHONPATH=. python scripts/run_healthcare.py",
        "PYTHONPATH=. python scripts/run_overhead.py",
        "PYTHONPATH=. python scripts/aggregate_overhead.py",
        "```",
        "",
    ]

    report = "\n".join(lines)
    Path("READMEv1.md").write_text(report, encoding="utf-8")
    print("Wrote READMEv1.md")


if __name__ == "__main__":
    main()
