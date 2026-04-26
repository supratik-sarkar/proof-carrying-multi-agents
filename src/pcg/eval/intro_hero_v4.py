"""
Phase L+: intro_hero_v4 — compact 4-LLM variant for the main paper.

Designed for the introduction section where horizontal space is at a
premium. Four LLMs span the full scale range (3.8B → 671B parameters):
    - phi-3.5-mini    (3.8B)         — small / efficient
    - Gemma-2-9b-it   (9B)           — medium / instruction-tuned
    - Llama-3.3-70B   (70B)          — large open-weight
    - deepseek-v3     (671B MoE)     — frontier

The upper headline banner from v3 is REMOVED here — the analogous
quantitative claim is meant to live in the LaTeX figure caption so
the introduction prose can highlight whichever metric is most
relevant to that section's argument.

All other v3 design choices (bar styles, monospace LLM names, panel
titles, the "(Our) Theorem-1 tightness" wording, the "(ours)" in
legends) are preserved verbatim.
"""
from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from pcg.eval.plots_v2 import (
    BOLD_THEME, PlotTheme,
    _add_provenance_footer, save_fig_v2, setup_matplotlib_for_theme,
)
from pcg.eval.intro_hero_v3 import (
    HeroEntry, _panel_harm, _panel_tightness, _panel_overhead,
    make_mock_entries as _v3_make_mock_entries,
)


# ---------------------------------------------------------------------------
# The compact 4-LLM cohort
# ---------------------------------------------------------------------------

V4_LLMS = (
    "phi-3.5-mini",     # smallest, instruction-tuned
    "Gemma-2-9b-it",    # mid-scale instruction-tuned
    "Llama-3.3-70B",    # large open-weight
    "deepseek-v3",      # frontier MoE
)


def make_mock_entries_v4() -> list[HeroEntry]:
    """Pull just the 4 v4 LLMs out of the full v3 mock set, preserving
    v3's calibrated synthetic numbers so the two figures stay coherent."""
    full = {e.name: e for e in _v3_make_mock_entries()}
    return [full[name] for name in V4_LLMS if name in full]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plot_intro_hero_v4(
    *,
    entries: Sequence[HeroEntry] | None = None,
    theme: PlotTheme = BOLD_THEME,
    source_runs: list[str] | None = None,
    is_mock: bool = False,
) -> plt.Figure:
    """Compact 4-LLM intro hero. No upper banner, smaller figure size.

    Use this in the main paper introduction. Use v3 in the appendix
    or supplementary materials when you have the space for all 7 LLMs.
    """
    setup_matplotlib_for_theme(theme)
    entries = list(entries or [])

    # Smaller height — 4 rows take less vertical space than 7
    fig = plt.figure(
        figsize=(15.0, 5.4), dpi=theme.dpi,
        facecolor=theme.palette["bg_panel"],
    )
    # No banner: panels can extend almost to the top of the figure
    gs = GridSpec(
        1, 3, figure=fig,
        wspace=0.45,
        left=0.10, right=0.97,
        top=0.92, bottom=0.13,
    )
    ax_harm  = fig.add_subplot(gs[0, 0])
    ax_tight = fig.add_subplot(gs[0, 1])
    ax_over  = fig.add_subplot(gs[0, 2])

    _panel_harm(ax_harm, entries, theme)
    _panel_tightness(ax_tight, entries, theme)
    _panel_overhead(ax_over, entries, theme)

    _add_provenance_footer(fig, theme, source_runs=source_runs, is_mock=is_mock)
    return fig


# ---------------------------------------------------------------------------
# CLI shim
# ---------------------------------------------------------------------------


def main_cli(out_path: str = "figures/intro_hero_v4") -> list[str]:
    fig = plot_intro_hero_v4(
        entries=make_mock_entries_v4(),
        source_runs=["mock-smoke"],
        is_mock=True,
    )
    return save_fig_v2(fig, out_path)


if __name__ == "__main__":
    paths = main_cli()
    for p in paths:
        print(p)
