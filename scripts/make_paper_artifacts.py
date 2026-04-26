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

    # Phase M: per-experiment R1..R5 plots with diverse coverage cells.
    # These OVERWRITE the legacy r{1..5}_*.pdf produced by make_figures
    # because the new format is what the paper references.
    rc_m = 0
    try:
        from make_per_experiment_plots import main as make_per_exp_main
        rc_m = make_per_exp_main(
            results_dir=results_dir, out_dir=figures_dir,
        )
    except Exception as e:
        log_info(f"  make_per_experiment_plots FAILED: {e}")
        rc_m = 1

    # Phase L: the new 7-LLM intro hero + Phase L+ compact 4-LLM variant.
    # Uses mock entries when no real backend results exist; the run
    # scripts populate real entries by writing to results/heroes/
    rc_e = 0
    try:
        from pcg.eval.intro_hero_v3 import (
            plot_intro_hero_v3, make_mock_entries, save_fig_v2,
        )
        from pcg.eval.intro_hero_v4 import (
            plot_intro_hero_v4, make_mock_entries_v4,
        )
        from pcg.eval.plots_v2 import detect_mock_runs

        figures_path = Path(figures_dir if Path(figures_dir).is_absolute()
                            else project_root() / figures_dir)
        figures_path.mkdir(parents=True, exist_ok=True)

        # Try to load real entries first; fall back to mock.
        results_path = Path(results_dir if Path(results_dir).is_absolute()
                            else project_root() / results_dir)
        entries_full = _load_real_hero_entries(results_path)
        is_mock = (entries_full is None) or detect_mock_runs(
            list(results_path.glob("r1*")) + list(results_path.glob("r5*"))
        )
        if entries_full is None:
            log_info("  intro_hero_v3/v4: no real LLM entries yet, rendering mock")
            entries_full = make_mock_entries()

        srcs = [
            d.name for d in
            list(results_path.glob("r1*")) + list(results_path.glob("r5*"))
        ]

        # v3: full 7-LLM appendix variant
        fig3 = plot_intro_hero_v3(
            entries=entries_full, source_runs=srcs, is_mock=is_mock,
        )
        out_v3 = save_fig_v2(fig3, figures_path / "intro_hero_v3")
        log_info(f"  intro_hero_v3 written: {out_v3}")

        # v4: compact 4-LLM main-text variant (phi-3.5-mini, Gemma-2-9b-it,
        # Llama-3.3-70B, deepseek-v3) with no headline banner so caption
        # space is preserved for hand-written quantitative claims.
        # Filter the same entries to v4's 4-LLM cohort if real, else use
        # the dedicated v4 mock builder.
        from pcg.eval.intro_hero_v4 import V4_LLMS
        entries_v4 = (
            [e for e in entries_full if e.name in V4_LLMS] or
            make_mock_entries_v4()
        )
        fig4 = plot_intro_hero_v4(
            entries=entries_v4, source_runs=srcs, is_mock=is_mock,
        )
        out_v4 = save_fig_v2(fig4, figures_path / "intro_hero_v4")
        log_info(f"  intro_hero_v4 written: {out_v4}")
    except Exception as e:
        log_info(f"  intro_hero_v3/v4 FAILED: {e}")
        rc_e = 1

    # Phase M: top-3 comparative R-plots (richer R4/R5 visualizations,
    # 3 (LLM, dataset) cells per plot, diversity-aware selection).
    rc_f = 0
    try:
        figures_path = Path(figures_dir if Path(figures_dir).is_absolute()
                            else project_root() / figures_dir)
        figures_path.mkdir(parents=True, exist_ok=True)
        results_path = Path(results_dir if Path(results_dir).is_absolute()
                            else project_root() / results_dir)
        from pcg.eval.plots_v2 import detect_mock_runs
        is_mock_top3 = detect_mock_runs(list(results_path.glob("r*")))
        rendered = _render_top3_plots(figures_path, results_path, is_mock_top3)
        log_info(f"  top3 plots rendered: {len(rendered)}")
    except Exception as e:
        log_info(f"  top3 plots FAILED: {e}")
        rc_f = 1

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
            "fig:intro-hero-v3":   "figures/intro_hero_v3.pdf",
            "fig:intro-hero-v4":   "figures/intro_hero_v4.pdf",
            "fig:r1-audit":        "figures/r1_audit_decomposition.pdf",
            "fig:r1-top3":         "figures/r1_top3.pdf",
            "fig:r2-redundancy":   "figures/r2_redundancy_law.pdf",
            "fig:r2-top3":         "figures/r2_top3.pdf",
            "fig:r3-resp":         "figures/r3_responsibility.pdf",
            "fig:r3-top3":         "figures/r3_top3.pdf",
            "fig:r4-pareto":       "figures/r4_risk_pareto.pdf",
            "fig:r4-pareto-top3":  "figures/r4_pareto_top3.pdf",
            "fig:r4-privacy":      "figures/r4_privacy_utility.pdf",
            "fig:r4-privacy-top3": "figures/r4_privacy_top3.pdf",
            "fig:r5-overhead":     "figures/r5_overhead.pdf",
            "fig:r5-overhead-top3": "figures/r5_overhead_top3.pdf",
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

    rc = 0 if (rc_a == 0 and rc_b == 0 and rc_c == 0 and rc_d == 0
                and rc_e == 0 and rc_f == 0 and rc_m == 0) else 1
    log_section(f"DONE (rc={rc})")
    return rc


def _render_top3_plots(figures_path, results_path, is_mock_global):
    """Phase M: render all 6 top-3 comparative plots.

    Each plot picks 3 (LLM, dataset) cells from a 5×8 candidate matrix and
    shows PCG-MAS vs baseline on that specific cell. Diversity-aware
    selection ensures different cells appear across plots."""
    from pcg.eval.plots_top3 import (
        plot_r1_top3, plot_r2_top3, plot_r3_top3,
        plot_r4_pareto_top3, plot_r4_privacy_top3, plot_r5_overhead_top3,
        make_mock_cells, reset_diversity_tracker,
    )
    from pcg.eval.intro_hero_v3 import save_fig_v2

    # Reset the diversity tracker so each driver run starts fresh
    reset_diversity_tracker()

    # Load real cells if available, else use mock. Real cells come from
    # results/cells/<llm>__<dataset>.json files emitted by run_r* scripts
    # when real backends are invoked.
    cells = _load_real_cells(results_path)
    if not cells:
        log_info("  top3 plots: no real cells available, using mock cells")
        cells = make_mock_cells()

    sources = [
        d.name for d in
        list(results_path.glob("r1*"))
        + list(results_path.glob("r2*"))
        + list(results_path.glob("r3*"))
        + list(results_path.glob("r4*"))
        + list(results_path.glob("r5*"))
    ]

    plots_to_render = [
        ("r1_top3",         plot_r1_top3),
        ("r2_top3",         plot_r2_top3),
        ("r3_top3",         plot_r3_top3),
        ("r4_pareto_top3",  plot_r4_pareto_top3),
        ("r4_privacy_top3", plot_r4_privacy_top3),
        ("r5_overhead_top3", plot_r5_overhead_top3),
    ]
    rendered: list[str] = []
    for name, fn in plots_to_render:
        try:
            fig = fn(cells=cells, source_runs=sources, is_mock=is_mock_global)
            paths = save_fig_v2(fig, figures_path / name)
            rendered.append(paths[0])
            log_info(f"  {name}: {paths[0]}")
        except Exception as exc:
            log_info(f"  {name} FAILED: {exc}")
    return rendered


def _load_real_cells(results_path):
    """Walk results/cells/ for real-LLM cell JSONs.

    Each file is `<llm>__<dataset>.json` with the Cell field shape from
    pcg.eval.plots_top3.Cell. Returns a list of Cell instances or [] if
    none are found (the caller falls back to mock data)."""
    from pcg.eval.plots_top3 import Cell
    cells_dir = results_path / "cells"
    if not cells_dir.exists():
        return []
    cells: list[Cell] = []
    for f in sorted(cells_dir.glob("*.json")):
        try:
            d = json.loads(f.read_text())
            cells.append(Cell(
                llm=d["llm"], dataset=d["dataset"],
                r1_lhs=d.get("r1_lhs"),
                r1_rhs=d.get("r1_rhs"),
                r1_lhs_ci=tuple(d["r1_lhs_ci"]) if d.get("r1_lhs_ci") else None,
                r1_rhs_ci=tuple(d["r1_rhs_ci"]) if d.get("r1_rhs_ci") else None,
                r1_baseline_harm=d.get("r1_baseline_harm"),
                r1_pcg_harm=d.get("r1_pcg_harm"),
                r2_ks=d.get("r2_ks"),
                r2_emp=d.get("r2_emp"),
                r2_emp_ci=[tuple(c) for c in d["r2_emp_ci"]] if d.get("r2_emp_ci") else None,
                r2_theory=d.get("r2_theory"),
                r3_regimes=d.get("r3_regimes"),
                r3_top1_acc=d.get("r3_top1_acc"),
                r3_top1_acc_ci=[tuple(c) for c in d["r3_top1_acc_ci"]] if d.get("r3_top1_acc_ci") else None,
                r4_eps_axis=d.get("r4_eps_axis"),
                r4_always_harm=d.get("r4_always_harm"),
                r4_threshold_harm=d.get("r4_threshold_harm"),
                r4_learned_harm=d.get("r4_learned_harm"),
                r4_always_cost=d.get("r4_always_cost"),
                r4_threshold_cost=d.get("r4_threshold_cost"),
                r4_learned_cost=d.get("r4_learned_cost"),
                r4_priv_eps=d.get("r4_priv_eps"),
                r4_priv_acc=d.get("r4_priv_acc"),
                r4_priv_acc_ci=[tuple(c) for c in d["r4_priv_acc_ci"]] if d.get("r4_priv_acc_ci") else None,
                r5_phase_means=d.get("r5_phase_means"),
                r5_phase_samples=d.get("r5_phase_samples"),
                r5_baseline_total=d.get("r5_baseline_total"),
            ))
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            print(f"  skipping {f.name}: {exc}")
            continue
    return cells


def _load_real_hero_entries(results_path: Path):
    """Walk results/ to assemble per-LLM HeroEntry list from real runs.

    Returns None if no real runs found; the caller falls back to mock.
    Looks for results/heroes/llm-<model>.json files emitted by run_r1
    and run_r5 when the LLM is invoked.
    """
    from pcg.eval.intro_hero_v3 import HeroEntry, CANONICAL_LLMS
    heroes_dir = results_path / "heroes"
    if not heroes_dir.exists():
        return None
    entries: list[HeroEntry] = []
    for llm in CANONICAL_LLMS:
        p = heroes_dir / f"llm-{llm}.json"
        if not p.exists():
            continue
        try:
            d = json.loads(p.read_text())
            entries.append(HeroEntry(
                name=llm,
                harm_pcg=float(d["harm_pcg"]),
                harm_pcg_ci=tuple(d["harm_pcg_ci"]),
                harm_base=float(d["harm_base"]),
                harm_base_ci=tuple(d["harm_base_ci"]),
                tightness_pcg=float(d["tightness_pcg"]),
                tightness_pcg_ci=tuple(d["tightness_pcg_ci"]),
                tokens_pcg=float(d["tokens_pcg"]),
                tokens_base=float(d["tokens_base"]),
            ))
        except (KeyError, ValueError, TypeError, json.JSONDecodeError):
            continue
    return entries if entries else None


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
