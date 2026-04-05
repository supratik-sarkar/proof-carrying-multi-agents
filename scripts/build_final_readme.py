from __future__ import annotations

from pathlib import Path
from collections import Counter
import pandas as pd


ROOT = Path(".")
SRC = ROOT / "src"
CONFIGS = ROOT / "configs"
SCRIPTS = ROOT / "scripts"
OUTPUTS = ROOT / "outputs"


def count_file_types(base: Path) -> Counter:
    counter = Counter()
    for p in base.rglob("*"):
        if p.is_file() and ".git" not in p.parts and ".venv" not in p.parts:
            suffix = p.suffix.lower() if p.suffix else "[no_ext]"
            counter[suffix] += 1
    return counter


def tree_lines(base: Path, depth: int = 2, prefix: str = ""):
    if depth < 0 or not base.exists():
        return []
    entries = sorted(base.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    lines = []
    for i, entry in enumerate(entries):
        connector = "└── " if i == len(entries) - 1 else "├── "
        lines.append(prefix + connector + entry.name)
        if entry.is_dir() and depth > 0:
            extension = "    " if i == len(entries) - 1 else "│   "
            lines.extend(tree_lines(entry, depth - 1, prefix + extension))
    return lines


def pick_latest(pattern: str):
    files = sorted(
        OUTPUTS.joinpath("figures").glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return files[0] if files else None


def md_image(path: Path | None) -> str:
    if path is None:
        return "_No figure found._"
    return f"![]({str(path).replace(' ', '%20')})"


def read_readmev1_block() -> str:
    p = ROOT / "READMEv1.md"
    if not p.exists():
        return "_READMEv1.md not found._"
    return p.read_text(encoding="utf-8")


def theory_glossary_table() -> str:
    rows = [
        ["`src/common/telemetry.py`", "`StageStats`, `ExampleTelemetry`", "Telemetry containers for per-stage latency, tokens, and acceptance-related logging."],
        ["`src/common/token_count.py`", "`prompt_tokens`, `completion_tokens`, `total_tokens`", "Tokenizer-based accounting used for overhead and token-cost analysis."],
        ["`src/pcg/certificates.py`", "`claim`, `certificate_hash`, `minimal_support_ids`", "Builds proof-carrying certificates with normalized evidence, hashes, and minimal support subsets."],
        ["`src/pcg/checker.py`", "`Check(Z; G_t)` proxy", "Replay-style checker that validates certificate structure, support IDs, and certificate hashes."],
        ["`src/pcg/verifier.py`", "`risk`, `score`, `pred_idx`", "Maps model output to a candidate answer and computes a verifier confidence / risk signal."],
        ["`src/pcg/decision.py`", "`answer / refuse`", "Risk-aware acceptance rule used to decide whether an answer is accepted."],
        ["`scripts/run_eval.py`", "`accepted`, `answer_correct`", "Main evaluation loop that produces run-level JSONL artifacts."],
        ["`scripts/summarize_runs.py`", "`mean_total_wall_ms`, `accepted_accuracy`", "Per-run summary builder used for README-facing tables."],
        ["`scripts/aggregate_overhead.py`", "`token_overhead_ratio_vs_posthoc`", "Aggregates overhead metrics across modes into a single comparison table."],
        ["`scripts/make_figures.py`", "`latency_breakdown`, `quality_overview`", "Builds README-ready figures from run logs and summary statistics."],
        ["`outputs/runs/*.jsonl`", "`run_id`, stage logs", "Raw run artifacts used as the source of truth for tables and plots."],
        ["`outputs/tables/overhead_main.csv`", "`acceptance_rate`, `answer_accuracy`, `latency`, `tokens`", "Aggregate overhead table across PCG and baseline modes."],
    ]
    df = pd.DataFrame(rows, columns=["File", "Quantity / Symbol", "Short description"])
    return df.to_markdown(index=False)


def main():
    file_counts = count_file_types(ROOT)
    total_files = sum(file_counts.values())

    top_types = sorted(file_counts.items(), key=lambda x: (-x[1], x[0]))[:12]
    file_type_table = pd.DataFrame(top_types, columns=["extension", "count"]).to_markdown(index=False)

    tree = []
    for folder in [CONFIGS, SCRIPTS, SRC, OUTPUTS]:
        if folder.exists():
            tree.append(folder.name + "/")
            tree.extend(tree_lines(folder, depth=2, prefix=""))
            tree.append("")
    tree_block = "\n".join(tree).rstrip()

    overhead_path = OUTPUTS / "tables" / "overhead_main.csv"
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
        if cols:
            overhead_md = df[cols].round(4).to_markdown(index=False)

    latency_plot = pick_latest("*_latency_breakdown.png")
    latency_box = pick_latest("*_latency_boxplot.png")
    token_plot = pick_latest("*_token_distribution.png")
    quality_plot = pick_latest("*_quality_overview.png")

    readme_v1_text = read_readmev1_block()
    theory_md = theory_glossary_table()

    lines = [
        "# Proof-Carrying Generation for Agentic AI (PCG-MAS)",
        "",
        "Reproducible scaffold for proof-carrying multi-agent evaluation with certificate construction, replay checking, telemetry, overhead analysis, and healthcare-oriented stress-testing.",
        "",
        "- Multi-stage telemetry: generation, certificate, replay, verifier",
        "- Overhead analysis with token/latency summaries",
        "- Healthcare-oriented evaluation adapters (MedMCQA / MedQA path)",
        "- Publication-style plots and reproducible local + Colab workflow",
        "",
        "---",
        "",
        "## Project map & methodology cheat-sheet",
        "",
        "This README mirrors the structure of the reference repo style while reflecting the current `proof-carrying-multi-agents` repository and its rebuilt experimental scaffold.",
        "",
        "### Repository overview",
        "",
        f"- **Repository root:** `{ROOT.resolve()}`",
        "- **Primary goal:** evaluate proof-carrying acceptance, replayability, and operational overhead in controlled LLM-driven pipelines.",
        "- **Main outputs:** run JSONL logs, per-run summary CSVs, aggregate overhead tables, and README-ready figures.",
        f"- **Current tracked file count (approx.):** {total_files}",
        "",
        "### Theory glossary (files → quantities)",
        "",
        theory_md,
        "",
        "### Key modules (implementation map)",
        "",
        "- `src/common/` → telemetry, timers, token accounting, JSONL logging",
        "- `src/data/` → dataset adapters",
        "- `src/pcg/` → prover, certificates, checker, verifier, decision logic",
        "- `scripts/` → run, summarize, aggregate, plot, and README generation workflows",
        "- `outputs/` → generated runs, tables, and figures",
        "",
        "### Repository tree (depth-limited)",
        "",
        "```text",
        tree_block,
        "```",
        "",
        "## File-type distribution",
        "",
        f"Total files scanned: **{total_files}**",
        "",
        file_type_table,
        "",
        "## Latest result highlights",
        "",
        "### 1. Latency Breakdown",
        md_image(latency_plot),
        "",
        "This figure decomposes end-to-end runtime into generation, certificate construction, replay validation, and verifier scoring.",
        "",
        "### 2. Latency Distribution Across Stages",
        md_image(latency_box),
        "",
        "This figure shows stage-wise variability and outliers instead of only mean latency.",
        "",
        "### 3. Token Distribution",
        md_image(token_plot),
        "",
        "This figure summarizes token-cost variability across examples and highlights whether cost is stable or tail-heavy.",
        "",
        "### 4. Quality Overview",
        md_image(quality_plot),
        "",
        "This figure summarizes acceptance rate, overall accuracy, accepted-answer accuracy, and verifier confidence.",
        "",
        "## Aggregate overhead table",
        "",
        overhead_md,
        "",
        "## Submission snapshot",
        "",
        "- **Stage:** local rebuild completed; Colab/GPU phase still pending",
        "- **Current model path:** open-source Hugging Face models through local script-first execution",
        "- **Current datasets:** MedMCQA pipeline active; MedQA path scaffolded",
        "- **Artifacts generated:** run logs, per-run tables, aggregate overhead table, and README-ready figures",
        "- **Intended next stage:** Colab-backed reruns with stronger backbones and fuller overhead/healthcare reporting",
        "",
        "## Why numbers may differ",
        "",
        "- Local CPU/MPS runs and later Colab GPU runs can differ in runtime, tokenization edge-cases, and model generation behavior.",
        "- Public dataset loaders and package versions may introduce small changes in row ordering or field behavior.",
        "- The current backend is a rebuilt scaffold and still evolving; certificate/checker/verifier logic may strengthen further before final experimental reporting.",
        "- Figures and tables in this README reflect the currently generated local artifacts, not yet the final Colab-scale experimental pass.",
        "",
        "## Interim READMEv1 content (merged)",
        "",
        "> The following block preserves and merges the previously generated READMEv1 content.",
        "",
        readme_v1_text,
        "",
        "## Diagnostics",
        "",
        "Typical commands:",
        "",
        "```bash",
        "PYTHONPATH=. python scripts/run_eval.py --config configs/medmcqa.yaml",
        "PYTHONPATH=. python scripts/run_healthcare.py",
        "PYTHONPATH=. python scripts/run_overhead.py",
        "PYTHONPATH=. python scripts/aggregate_overhead.py",
        "PYTHONPATH=. python scripts/make_figures.py --run outputs/runs/<run_file>.jsonl",
        "```",
        "",
        "## Compute environments",
        "",
        "- **Local development:** MacBook Pro environment via `.venv`",
        "- **Planned replication / scale-up:** Google Colab GPU runtime",
        "- **Goal:** align local debug runs with later Colab-backed experimental reruns",
        "",
    ]

    report = "\n".join(lines)
    Path("README.md").write_text(report, encoding="utf-8")
    print("Wrote README.md")


if __name__ == "__main__":
    main()
