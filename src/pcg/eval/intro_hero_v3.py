"""
Phase L: paper centerpiece — the intro hero.

A single dominant figure that answers, in one glance, the question:
"Why is PCG-MAS uniquely useful?"

Design:
- Three side-by-side panels, one per high-impact metric
- All 7 LLMs (phi-3.5-mini, qwen2.5-7B, deepseek-llm-7b-chat,
  Llama-3.1-8B, Gemma-2-9b-it, Llama-3.3-70B, deepseek-v3) listed
  by NAME on the y-axis of each panel
- For each LLM and each metric, two bars: PCG-MAS (red) vs baseline (slate)
- Confidence intervals via paired bootstrap (when paired data exists)
- Headline summary banner above the panels stating the absolute
  improvements — quantitative markers reviewers can quote directly

Metrics chosen:
1. Harm rate (R1)               — the SAFETY story, lower is better
2. Audit-bound tightness % (R1) — the THEORETICAL story (Thm 1)
3. Token overhead (R5)          — the PRACTICALITY story

The figure is published in the paper introduction and reused in the
README.md hero block.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from pcg.eval.plots_v2 import (
    BOLD_THEME, PlotTheme,
    _add_provenance_footer, _annotate_missing, _panel_heading,
    setup_matplotlib_for_theme, save_fig_v2,
)


# ---------------------------------------------------------------------------
# Canonical LLM list — order matters: smallest → largest, then frontier
# ---------------------------------------------------------------------------

CANONICAL_LLMS = (
    "phi-3.5-mini",         # 3.8B
    "qwen2.5-7B",           # 7B
    "deepseek-llm-7b-chat", # 7B
    "Llama-3.1-8B",         # 8B
    "Gemma-2-9b-it",        # 9B
    "Llama-3.3-70B",        # 70B
    "deepseek-v3",          # 671B (MoE, ~37B active)
)


@dataclass
class HeroEntry:
    """Per-LLM measurements for the intro hero.

    Fields are paired: PCG-MAS value first, baseline value second.
    Each pair carries its own (low, high) CI so the bars get error
    whiskers consistent with the bootstrap module.
    """
    name: str
    # Harm rate: lower is better
    harm_pcg: float
    harm_pcg_ci: tuple[float, float]
    harm_base: float
    harm_base_ci: tuple[float, float]
    # Audit-bound tightness: LHS/RHS as a fraction in [0, 1]
    tightness_pcg: float
    tightness_pcg_ci: tuple[float, float]
    # Tokens per claim: PCG-MAS (incl. verifier+redundancy) vs baseline
    tokens_pcg: float
    tokens_base: float

    @property
    def harm_reduction_factor(self) -> float:
        """e.g. 18.0 ↦ baseline harm is 18x ours"""
        return self.harm_base / max(self.harm_pcg, 1e-6)

    @property
    def overhead_factor(self) -> float:
        """e.g. 1.8 ↦ PCG-MAS uses 1.8× the tokens"""
        return self.tokens_pcg / max(self.tokens_base, 1.0)


# ---------------------------------------------------------------------------
# Panel renderers
# ---------------------------------------------------------------------------


def _set_llm_yaxis(ax, entries: Sequence[HeroEntry], theme: PlotTheme) -> None:
    """Common y-axis treatment: LLM names, mono font, slight emphasis on
    the largest models so reviewers see scale at a glance."""
    n = len(entries)
    ax.set_yticks(range(n))
    ax.set_yticklabels(
        [e.name for e in entries],
        fontsize=theme.label_size,
        fontfamily="monospace",
    )
    ax.invert_yaxis()  # so the first LLM in the list appears at the top
    ax.tick_params(axis="y", length=0)  # no ticks; labels alone


def _panel_harm(
    ax, entries: Sequence[HeroEntry], theme: PlotTheme,
) -> None:
    """Horizontal grouped-bar: harm rate, PCG vs baseline.

    Log scale on x because the gap is order-of-magnitude. Annotates the
    fold-reduction factor at the right edge of each PCG bar."""
    if not entries:
        _annotate_missing(ax, "No backend results yet", theme)
        return

    n = len(entries)
    y = np.arange(n)
    bar_h = 0.36

    harm_pcg = np.array([e.harm_pcg for e in entries])
    harm_base = np.array([e.harm_base for e in entries])

    # CI half-widths
    pcg_err = np.array([
        [e.harm_pcg - e.harm_pcg_ci[0], e.harm_pcg_ci[1] - e.harm_pcg]
        for e in entries
    ]).T
    base_err = np.array([
        [e.harm_base - e.harm_base_ci[0], e.harm_base_ci[1] - e.harm_base]
        for e in entries
    ]).T

    # Baseline bars (pushed up, drawn first so PCG sits on top visually)
    ax.barh(
        y - bar_h / 2, harm_base, bar_h,
        xerr=base_err, error_kw={"capsize": 2.0, "elinewidth": 0.7},
        color=theme.palette["base_strong"],
        edgecolor="white", linewidth=0.6,
        label="No certificate",
    )
    # PCG bars
    ax.barh(
        y + bar_h / 2, harm_pcg, bar_h,
        xerr=pcg_err, error_kw={"capsize": 2.0, "elinewidth": 0.7},
        color=theme.palette["ours"],
        edgecolor="white", linewidth=0.6,
        label="PCG-MAS (ours)",
    )

    # Fold-reduction annotation at right of each PCG bar
    x_max = max(harm_base.max(), harm_pcg.max())
    for i, e in enumerate(entries):
        factor = e.harm_reduction_factor
        ax.text(
            x_max * 1.05, i + bar_h / 2,
            f"{factor:.0f}× lower",
            ha="left", va="center",
            fontsize=theme.annotation_size,
            color=theme.palette["ours_dark"],
            fontweight="bold",
        )

    ax.set_xscale("log")
    ax.set_xlim(left=max(harm_pcg.min() * 0.6, 1e-4), right=x_max * 1.55)
    ax.set_xlabel("Harm rate (log scale, lower is better)",
                  fontsize=theme.label_size)
    _set_llm_yaxis(ax, entries, theme)
    ax.legend(
        loc="lower right", frameon=False,
        fontsize=theme.annotation_size,
    )
    _panel_heading(
        ax, theme,
        title="Safety",
        subtitle="harm rate per accepted claim",
    )


def _panel_tightness(
    ax, entries: Sequence[HeroEntry], theme: PlotTheme,
) -> None:
    """Single bar per LLM: Theorem 1 bound tightness as a percentage.

    Higher is better — a tight bound means the audit decomposition
    captures essentially all the failure modes."""
    if not entries:
        _annotate_missing(ax, "No backend results yet", theme)
        return

    n = len(entries)
    y = np.arange(n)
    bar_h = 0.6

    tight = np.array([e.tightness_pcg * 100 for e in entries])
    err = np.array([
        [(e.tightness_pcg - e.tightness_pcg_ci[0]) * 100,
         (e.tightness_pcg_ci[1] - e.tightness_pcg) * 100]
        for e in entries
    ]).T

    ax.barh(
        y, tight, bar_h,
        xerr=err, error_kw={"capsize": 2.0, "elinewidth": 0.7},
        color=theme.palette["ours"],
        edgecolor="white", linewidth=0.6,
    )

    # Annotate value at a uniform x position past where any error whisker can land
    label_x = max(tight + err[1, :]) + 4
    for i, v in enumerate(tight):
        ax.text(
            label_x, i, f"{v:.0f}%",
            ha="left", va="center",
            fontsize=theme.annotation_size,
            color=theme.palette["ours_dark"],
            fontweight="bold",
        )

    # Reference line at 100% — perfect tightness
    ax.axvline(
        100, color=theme.palette["neutral"],
        linestyle=":", linewidth=1.0, alpha=0.7, zorder=1,
    )
    ax.text(
        100.5, n - 0.4, "perfect\nbound",
        ha="left", va="center",
        fontsize=theme.annotation_size - 1,
        color=theme.palette["ink_light"],
        style="italic",
    )

    ax.set_xlim(0, 125)
    ax.set_xlabel("Bound tightness (%, higher is better)",
                  fontsize=theme.label_size)
    _set_llm_yaxis(ax, entries, theme)
    _panel_heading(
        ax, theme,
        title="Theorem 1 tightness",
        subtitle="LHS / RHS, fraction explained",
    )


def _panel_overhead(
    ax, entries: Sequence[HeroEntry], theme: PlotTheme,
) -> None:
    """Tokens per claim: stacked bars of baseline (smaller) on top of
    overhead (the additional tokens PCG-MAS requires).

    Shows reviewers the practical price of the safety win — "you pay
    1.7× the tokens to get 18× lower harm" is a great trade summary."""
    if not entries:
        _annotate_missing(ax, "No R5 overhead data yet", theme)
        return

    n = len(entries)
    y = np.arange(n)
    bar_h = 0.6

    base = np.array([e.tokens_base for e in entries])
    extra = np.array([max(e.tokens_pcg - e.tokens_base, 0) for e in entries])

    ax.barh(
        y, base, bar_h,
        color=theme.palette["base_weak_l"],
        edgecolor="white", linewidth=0.6,
        label="Baseline tokens",
    )
    ax.barh(
        y, extra, bar_h, left=base,
        color=theme.palette["ours"],
        edgecolor="white", linewidth=0.6,
        label="PCG-MAS overhead",
    )

    # Annotate overhead factor at end of each bar
    x_max = (base + extra).max()
    for i, e in enumerate(entries):
        factor = e.overhead_factor
        ax.text(
            (e.tokens_pcg) + x_max * 0.02, i,
            f"{factor:.1f}×",
            ha="left", va="center",
            fontsize=theme.annotation_size,
            color=theme.palette["ours_dark"],
            fontweight="bold",
        )

    ax.set_xlim(0, x_max * 1.18)
    ax.set_xlabel("Tokens per claim", fontsize=theme.label_size)
    _set_llm_yaxis(ax, entries, theme)
    ax.legend(
        loc="lower right", frameon=False,
        fontsize=theme.annotation_size,
    )
    _panel_heading(
        ax, theme,
        title="Cost overhead",
        subtitle="per-claim token budget",
    )


# ---------------------------------------------------------------------------
# Headline banner — the "money sentence" above the panels
# ---------------------------------------------------------------------------


def _draw_headline_banner(
    fig, entries: Sequence[HeroEntry], theme: PlotTheme,
) -> None:
    """Single bold quantitative claim drawn at the top of the figure.

    Picks the strongest aggregated improvement across the LLM panel and
    states it in language reviewers can quote directly."""
    if not entries:
        return

    # Aggregate: median fold-reduction in harm, median tightness, median overhead
    harm_factors = sorted(e.harm_reduction_factor for e in entries)
    tight_pcts = sorted(e.tightness_pcg * 100 for e in entries)
    overheads = sorted(e.overhead_factor for e in entries)

    def _median(xs):
        m = len(xs) // 2
        return xs[m] if len(xs) % 2 else 0.5 * (xs[m - 1] + xs[m])

    harm_med = _median(harm_factors)
    tight_med = _median(tight_pcts)
    over_med = _median(overheads)

    # Centered headline — separate spans for color
    fig.text(
        0.5, 0.95,
        f"Across 7 LLMs from 3.8B to 671B parameters,",
        ha="center", va="top",
        fontsize=theme.title_size + 4,
        fontweight="bold",
        color=theme.palette["ink"],
    )
    fig.text(
        0.5, 0.90,
        f"PCG-MAS delivers {harm_med:.0f}× lower harm  ·  "
        f"{tight_med:.0f}% audit-bound tightness  ·  "
        f"only {over_med:.1f}× token overhead",
        ha="center", va="top",
        fontsize=theme.title_size + 1,
        color=theme.palette["ours_dark"],
        fontweight="bold",
    )


# ---------------------------------------------------------------------------
# Public API: plot_intro_hero_v3
# ---------------------------------------------------------------------------


def plot_intro_hero_v3(
    *,
    entries: Sequence[HeroEntry] | None = None,
    theme: PlotTheme = BOLD_THEME,
    source_runs: list[str] | None = None,
    is_mock: bool = False,
) -> plt.Figure:
    """Produce the paper-introduction hero figure (Phase L).

    Layout: 3 horizontal panels of equal width, with a 2-line headline
    banner above. All 7 LLMs visible by name on each panel's y-axis.
    """
    setup_matplotlib_for_theme(theme)
    entries = list(entries or [])

    fig = plt.figure(
        figsize=(15.5, 8.4), dpi=theme.dpi,
        facecolor=theme.palette["bg_panel"],
    )
    gs = GridSpec(
        1, 3, figure=fig,
        wspace=0.45,
        left=0.10, right=0.97,
        top=0.74, bottom=0.09,
    )
    ax_harm = fig.add_subplot(gs[0, 0])
    ax_tight = fig.add_subplot(gs[0, 1])
    ax_over = fig.add_subplot(gs[0, 2])

    _panel_harm(ax_harm, entries, theme)
    _panel_tightness(ax_tight, entries, theme)
    _panel_overhead(ax_over, entries, theme)

    _draw_headline_banner(fig, entries, theme)
    _add_provenance_footer(fig, theme, source_runs=source_runs, is_mock=is_mock)
    return fig


# ---------------------------------------------------------------------------
# Mock-data builder for smoke runs
# ---------------------------------------------------------------------------


def make_mock_entries() -> list[HeroEntry]:
    """Build a realistic-looking but synthetic 7-LLM result set.

    The numbers are model-scale-aware: smaller models tend to have
    higher baseline harm (so larger absolute reduction); larger models
    have lower baseline harm but proportionally similar reductions.

    Used by smoke-full so the intro hero has populated bars even before
    real LLM runs complete."""
    rng = np.random.default_rng(0)
    raw = [
        # name,                base_harm, ours_harm, tight, base_tok, our_tok
        ("phi-3.5-mini",       0.34,      0.022,     0.83,  82,   192),
        ("qwen2.5-7B",         0.28,      0.014,     0.86,  108,  224),
        ("deepseek-llm-7b-chat", 0.30,    0.016,     0.82,  112,  238),
        ("Llama-3.1-8B",       0.25,      0.012,     0.85,  118,  248),
        ("Gemma-2-9b-it",      0.23,      0.011,     0.87,  124,  256),
        ("Llama-3.3-70B",      0.16,      0.008,     0.91,  226,  392),
        ("deepseek-v3",        0.13,      0.006,     0.93,  192,  316),
    ]
    out: list[HeroEntry] = []
    for name, bh, oh, t, btok, otok in raw:
        # Realistic-looking CIs: ~10% relative half-width
        bh_ci = (bh * 0.9, bh * 1.1)
        oh_ci = (oh * 0.85, oh * 1.15)
        t_ci = (max(0, t - 0.04), min(1.0, t + 0.04))
        out.append(HeroEntry(
            name=name,
            harm_base=bh, harm_base_ci=bh_ci,
            harm_pcg=oh, harm_pcg_ci=oh_ci,
            tightness_pcg=t, tightness_pcg_ci=t_ci,
            tokens_base=btok, tokens_pcg=otok,
        ))
    return out


# ---------------------------------------------------------------------------
# CLI shim used by make_paper_artifacts
# ---------------------------------------------------------------------------


def main_cli(out_path: str = "figures/intro_hero_v3") -> list[str]:
    """Render the v3 intro hero with mock data when no real R-runs exist.

    The driver script (`make_paper_artifacts.py`) should call this with
    real entries when results are available; for smoke runs the mock
    builder gives a populated figure."""
    fig = plot_intro_hero_v3(
        entries=make_mock_entries(),
        source_runs=["mock-smoke"],
        is_mock=True,
    )
    return save_fig_v2(fig, out_path)


if __name__ == "__main__":
    paths = main_cli()
    for p in paths:
        print(p)
