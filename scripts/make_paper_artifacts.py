"""
scripts/make_paper_artifacts.py

One-command driver that regenerates the entire set of paper artifacts:

    figures/         <- via make_figures.py
    docs/tables/     <- via make_tables.py
    figures/intro_hero.{pdf,png}  <- via make_intro_hero.py
    docs/manifest.json       <- mapping of paper figure/table -> source JSON

This is the script CI runs to verify reproducibility: starting from
results/*/r*.json, every paper artifact is regenerated deterministically.

Usage:
    python scripts/make_paper_artifacts.py
    # or via the Makefile:
    make paper
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from _common import git_sha, log_info, log_section, project_root


def main(
    results_dir: str = "results",
    figures_dir: str = "figures",
    tables_dir: str = "docs/tables",
) -> int:
    log_section("make_paper_artifacts")
    log_info(f"  cwd:        {project_root()}")
    log_info(f"  git sha:    {git_sha()}")
    log_info(f"  results:    {results_dir}")
    log_info(f"  figures:    {figures_dir}")
    log_info(f"  tables:     {tables_dir}")

    from make_figures import main as make_figures_main
    from make_intro_hero import main as make_intro_hero_main
    from make_summary_benchmark import main as make_summary_benchmark_main
    from make_tables import main as make_tables_main

    rc_a = make_figures_main(results_dir=results_dir, out=figures_dir)
    rc_b = make_tables_main(results_dir=results_dir, out=tables_dir)
    rc_c = make_intro_hero_main(results_dir=results_dir, out=figures_dir)
    rc_d = make_summary_benchmark_main(results_dir=results_dir, out=figures_dir)

    # Collect manifests from each step
    fig_manifest_path = Path(figures_dir if Path(figures_dir).is_absolute()
                              else project_root() / figures_dir) / "manifest.json"
    tab_manifest_path = Path(tables_dir if Path(tables_dir).is_absolute()
                              else project_root() / tables_dir) / "manifest.json"
    hero_manifest_path = Path(figures_dir if Path(figures_dir).is_absolute()
                               else project_root() / figures_dir) / "intro_hero_manifest.json"

    figs = []
    if fig_manifest_path.exists():
        with fig_manifest_path.open("r") as fh:
            figs = json.load(fh).get("figures", [])
    tabs = []
    if tab_manifest_path.exists():
        with tab_manifest_path.open("r") as fh:
            tabs = json.load(fh).get("tables", [])
    hero = None
    if hero_manifest_path.exists():
        with hero_manifest_path.open("r") as fh:
            hero = json.load(fh)

    # Map paper labels to artifact files. This is the contract reviewers and
    # the camera-ready compilation rely on.
    paper_map = {
        "figures": {
            "fig:intro-hero":      "figures/intro_hero.pdf",
            "fig:r1-audit":        "figures/r1_audit_decomposition.pdf",
            "fig:r2-redundancy":   "figures/r2_redundancy_law.pdf",
            "fig:r3-resp":         "figures/r3_responsibility.pdf",
            "fig:r4-pareto":       "figures/r4_risk_pareto.pdf",
            "fig:r4-privacy":      "figures/r4_privacy_utility.pdf",
            "fig:r5-overhead":     "figures/r5_overhead.pdf",
            "fig:r5-vs-k":         "figures/r5_overhead_vs_k.pdf",
        },
        "tables": {
            "tab:r1-audit":         "docs/tables/t1_audit_decomposition.tex",
            "tab:r2-redundancy":    "docs/tables/t2_redundancy_law.tex",
            "tab:r3-responsibility":"docs/tables/t3_responsibility.tex",
            "tab:r4-risk-privacy":  "docs/tables/t4_risk_privacy.tex",
            "tab:r5-overhead":      "docs/tables/t5_overhead.tex",
        },
    }

    docs_dir = project_root() / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    full_manifest = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_sha": git_sha(),
        "paper_label_to_path": paper_map,
        "figures_emitted": figs,
        "tables_emitted": tabs,
        "intro_hero": hero,
    }
    out_manifest = docs_dir / "manifest.json"
    with out_manifest.open("w") as fh:
        json.dump(full_manifest, fh, indent=2)
    log_info(f"Wrote {out_manifest}")

    rc = 0 if (rc_a == 0 and rc_b == 0 and rc_c == 0 and rc_d == 0) else 1
    log_section(f"DONE (rc={rc})")
    return rc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results")
    p.add_argument("--figures-dir", default="figures")
    p.add_argument("--tables-dir", default="docs/tables")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    sys.exit(main(results_dir=a.results_dir, figures_dir=a.figures_dir,
                   tables_dir=a.tables_dir))
