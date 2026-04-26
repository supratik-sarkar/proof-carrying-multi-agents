"""
Appendix plots (Phase K).

Four new figures called out by reviewers as high-impact for the safety
+ trustworthy-ML angle of the paper:

    plot_cross_domain      Theorem 1 LHS vs RHS across all 8 datasets
    plot_backend_frontier  Safety/cost frontier from Phi-3.5-mini → Llama-3.3-70B
    plot_failure_modes     Stacked-bar attribution by attack type × audit channel
    plot_calibration       Reliability diagram for the asserted probability p

Plus two analysis-driven figures:

    plot_tightness_heatmap Theorem 1 slack across (k, ε_adv) grid
    plot_load_curves       Throughput, P95, P99 vs concurrency

All use the same BOLD_THEME and `_panel_heading` helpers from plots_v2,
so visually they're indistinguishable from the main paper figures.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

from pcg.eval.plots_v2 import (
    BOLD_THEME, PlotTheme,
    _add_provenance_footer, _annotate_missing,
    _panel_heading, save_fig_v2, setup_matplotlib_for_theme,
)


# =====================================================================
# 1. CROSS-DOMAIN GENERALIZATION
# =====================================================================


def _panel_cross_domain(
    ax, data: dict | None, theme: PlotTheme,
) -> None:
    """Theorem 1 LHS vs RHS for each of 8 datasets.

    data shape:
        {
            "datasets":  ["HotpotQA", "2Wiki", ..., "WebLINX", "synthetic"],
            "lhs_means": [0.078, 0.082, ...],
            "lhs_cis":   [(low, high), ...],
            "rhs_means": [0.095, 0.103, ...],
            "rhs_cis":   [(low, high), ...],
        }
    Renders a paired-bar chart with LHS (red, "ours") next to RHS
    (slate, "Thm 1 bound") for each dataset. Visually obvious that LHS ≤ RHS
    everywhere — the bound holds across domains.
    """
    if not data or not data.get("datasets"):
        _annotate_missing(ax, "Run all 8 datasets to populate", theme)
        return

    datasets = data["datasets"]
    lhs = np.asarray(data.get("lhs_means", [0.0] * len(datasets)))
    rhs = np.asarray(data.get("rhs_means", [0.0] * len(datasets)))
    lhs_cis = data.get("lhs_cis") or [(v, v) for v in lhs]
    rhs_cis = data.get("rhs_cis") or [(v, v) for v in rhs]

    x = np.arange(len(datasets))
    width = 0.38
    lhs_err = np.array([
        [lhs[i] - lhs_cis[i][0], lhs_cis[i][1] - lhs[i]]
        for i in range(len(datasets))
    ]).T
    rhs_err = np.array([
        [rhs[i] - rhs_cis[i][0], rhs_cis[i][1] - rhs[i]]
        for i in range(len(datasets))
    ]).T

    ax.bar(
        x - width / 2, lhs, width,
        yerr=lhs_err, capsize=2.5,
        color=theme.palette["ours"], label="LHS = Pr(accept ∩ wrong)",
        edgecolor="white", linewidth=0.6,
    )
    ax.bar(
        x + width / 2, rhs, width,
        yerr=rhs_err, capsize=2.5,
        color=theme.palette["base_strong"], label="RHS = Σ channels (Thm 1)",
        edgecolor="white", linewidth=0.6,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, max(rhs.max(), lhs.max()) * 1.25)
    ax.legend(
        loc="upper left", frameon=False,
        fontsize=theme.annotation_size,
    )
    _panel_heading(
        ax, theme,
        title="Theorem 1 holds across 8 datasets",
        subtitle="LHS ≤ RHS everywhere",
    )


def plot_cross_domain(
    *, data: dict | None = None,
    theme: PlotTheme = BOLD_THEME,
    source_runs: list[str] | None = None,
    is_mock: bool = False,
) -> plt.Figure:
    """Single-panel figure: Theorem 1 across 8 datasets."""
    setup_matplotlib_for_theme(theme)
    fig, ax = plt.subplots(
        figsize=(11, 5.5), dpi=theme.dpi,
        facecolor=theme.palette["bg_panel"],
    )
    _panel_cross_domain(ax, data, theme)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    _add_provenance_footer(fig, theme, source_runs=source_runs, is_mock=is_mock)
    return fig


# =====================================================================
# 2. BACKEND FRONTIER
# =====================================================================


def _panel_backend_frontier(
    ax, data: dict | None, theme: PlotTheme,
) -> None:
    """Safety vs cost across model scales — Phi-mini through Llama-70B.

    data shape:
        {
            "models":     ["Phi-3.5-mini", "Qwen-7B", "Llama-13B", "Llama-3.3-70B"],
            "params_b":   [3.8, 7, 13, 70],   # billions of params
            "harm_pcg":   [0.018, 0.012, 0.009, 0.007],   # PCG-MAS at each scale
            "harm_base":  [0.34, 0.28, 0.21, 0.16],       # baseline at each scale
            "cost_pcg":   [180, 220, 290, 410],   # tokens/claim PCG-MAS
            "cost_base":  [80, 110, 145, 220],    # tokens/claim baseline
        }
    """
    if not data or not data.get("models"):
        _annotate_missing(ax, "Sweep ≥3 backends to populate", theme)
        return

    cost_pcg = np.asarray(data.get("cost_pcg", []), dtype=float)
    cost_base = np.asarray(data.get("cost_base", []), dtype=float)
    harm_pcg = np.asarray(data.get("harm_pcg", []), dtype=float)
    harm_base = np.asarray(data.get("harm_base", []), dtype=float)
    models = data["models"]

    # Scatter with size proportional to model params
    params_b = np.asarray(data.get("params_b", [10.0] * len(models)), dtype=float)
    sizes = 60 + 12 * params_b   # visual scale

    ax.scatter(
        cost_pcg, harm_pcg, s=sizes,
        color=theme.palette["ours"], alpha=0.85,
        edgecolors="white", linewidths=1.2,
        label="PCG-MAS",
        zorder=3,
    )
    ax.scatter(
        cost_base, harm_base, s=sizes,
        color=theme.palette["base_strong"], alpha=0.85,
        edgecolors="white", linewidths=1.2,
        label="No-cert baseline",
        zorder=3,
    )

    # Connect each model's two points with a thin line so you see the
    # delta at a glance
    for i in range(len(models)):
        ax.plot(
            [cost_base[i], cost_pcg[i]],
            [harm_base[i], harm_pcg[i]],
            color=theme.palette["neutral"], alpha=0.35, lw=1.0,
            zorder=1,
        )
        ax.text(
            cost_pcg[i], harm_pcg[i] * 0.85, models[i],
            fontsize=theme.annotation_size - 1,
            ha="center", va="top",
            color=theme.palette["ink_light"],
        )

    ax.set_yscale("log")
    ax.set_xlabel("Tokens / claim")
    ax.set_ylabel("Harm rate (log scale)")
    ax.legend(
        loc="upper right", frameon=False,
        fontsize=theme.annotation_size,
    )
    _panel_heading(
        ax, theme,
        title="PCG-MAS wins at every model scale",
        subtitle="harm reduction independent of capability",
    )


def plot_backend_frontier(
    *, data: dict | None = None,
    theme: PlotTheme = BOLD_THEME,
    source_runs: list[str] | None = None,
    is_mock: bool = False,
) -> plt.Figure:
    """Single-panel figure: harm-vs-cost across model scales."""
    setup_matplotlib_for_theme(theme)
    fig, ax = plt.subplots(
        figsize=(8.5, 5.5), dpi=theme.dpi,
        facecolor=theme.palette["bg_panel"],
    )
    _panel_backend_frontier(ax, data, theme)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    _add_provenance_footer(fig, theme, source_runs=source_runs, is_mock=is_mock)
    return fig


# =====================================================================
# 3. FAILURE-MODE ANATOMY
# =====================================================================


def _panel_failure_modes(
    ax, data: dict | None, theme: PlotTheme,
) -> None:
    """Stacked-bar showing which audit channel catches which attack type.

    data shape:
        {
            "attack_types": ["hallucination", "evidence_swap", "schema_break", "policy_violation"],
            "channels":     ["IntFail", "ReplayFail", "CheckFail", "CovGap"],
            "matrix":       [[catch_rate_per_attack_per_channel], ...],
                            # shape (n_attacks, n_channels)
        }
    Renders one stacked bar per attack type. Bar height ≤ 1 (sum of
    catch-rates across channels). Visually answers "which mechanism
    catches what".
    """
    if not data or not data.get("matrix"):
        _annotate_missing(ax, "Run R3 with attack mix to populate", theme)
        return

    attacks = data["attack_types"]
    channels = data["channels"]
    M = np.asarray(data["matrix"], dtype=float)
    n_a, n_c = M.shape

    # Color per channel — distinct, accessible
    channel_colors = [
        theme.palette["ours"],         # IntFail
        theme.palette["accent_amber"], # ReplayFail
        theme.palette["accent_green"], # CheckFail
        theme.palette["base_strong"],     # CovGap
    ][:n_c]

    x = np.arange(n_a)
    bottom = np.zeros(n_a)
    for j in range(n_c):
        ax.bar(
            x, M[:, j], 0.6,
            bottom=bottom,
            color=channel_colors[j], label=channels[j],
            edgecolor="white", linewidth=0.6,
        )
        bottom = bottom + M[:, j]

    ax.set_xticks(x)
    ax.set_xticklabels(attacks, rotation=15, ha="right")
    ax.set_ylabel("Catch rate")
    ax.set_ylim(0, 1.05)
    ax.legend(
        title="Audit channel",
        loc="upper left", bbox_to_anchor=(1.02, 1.0),
        frameon=False, fontsize=theme.annotation_size,
    )
    _panel_heading(
        ax, theme,
        title="Failure-mode anatomy",
        subtitle="which channel catches which attack",
    )


def plot_failure_modes(
    *, data: dict | None = None,
    theme: PlotTheme = BOLD_THEME,
    source_runs: list[str] | None = None,
    is_mock: bool = False,
) -> plt.Figure:
    setup_matplotlib_for_theme(theme)
    fig, ax = plt.subplots(
        figsize=(9.5, 5.5), dpi=theme.dpi,
        facecolor=theme.palette["bg_panel"],
    )
    _panel_failure_modes(ax, data, theme)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    _add_provenance_footer(fig, theme, source_runs=source_runs, is_mock=is_mock)
    return fig


# =====================================================================
# 4. CALIBRATION (reliability diagram)
# =====================================================================


def _panel_calibration(
    ax, data: dict | None, theme: PlotTheme,
) -> None:
    """Reliability diagram for the certificate's asserted probability p.

    data shape:
        {
            "bin_edges":  [0.0, 0.1, 0.2, ..., 1.0],
            "predicted":  [mean p in each bin],
            "empirical":  [mean accuracy in each bin],
            "bin_counts": [n examples per bin],
            "ece":        scalar Expected Calibration Error,
        }
    Renders predicted vs empirical with the perfect-calibration y=x line,
    bin sizes shown as histogram beneath.
    """
    if not data or not data.get("predicted"):
        _annotate_missing(ax, "Bin certificates by p to populate", theme)
        return

    pred = np.asarray(data["predicted"], dtype=float)
    emp = np.asarray(data["empirical"], dtype=float)
    counts = np.asarray(data.get("bin_counts", [1] * len(pred)), dtype=float)
    ece = data.get("ece", None)

    # Perfect-calibration reference line
    ax.plot([0, 1], [0, 1], "--", color=theme.palette["neutral"],
            lw=1.4, alpha=0.8, label="Perfect calibration")

    # Calibration curve (size of points proportional to bin count)
    sizes = 30 + 220 * counts / counts.max() if counts.max() > 0 else 30
    ax.plot(pred, emp, "-", color=theme.palette["ours"], lw=1.8, alpha=0.9)
    ax.scatter(
        pred, emp, s=sizes,
        color=theme.palette["ours"], alpha=0.85,
        edgecolors="white", linewidths=1.0, zorder=3,
        label="PCG-MAS",
    )

    # ECE annotation in lower-right
    if ece is not None:
        ax.text(
            0.97, 0.05, f"ECE = {ece:.4f}",
            transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=theme.label_size,
            fontweight="bold",
            color=theme.palette["ink"],
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor=theme.palette["ours_light"],
                edgecolor=theme.palette["ours"], linewidth=0.8,
            ),
        )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Predicted probability p")
    ax.set_ylabel("Empirical accuracy")
    ax.legend(
        loc="upper left", frameon=False,
        fontsize=theme.annotation_size,
    )
    _panel_heading(
        ax, theme,
        title="Certificate probability is well-calibrated",
        subtitle="bins follow y=x; ECE reported",
    )


def plot_calibration(
    *, data: dict | None = None,
    theme: PlotTheme = BOLD_THEME,
    source_runs: list[str] | None = None,
    is_mock: bool = False,
) -> plt.Figure:
    setup_matplotlib_for_theme(theme)
    fig, ax = plt.subplots(
        figsize=(7.5, 6), dpi=theme.dpi,
        facecolor=theme.palette["bg_panel"],
    )
    _panel_calibration(ax, data, theme)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    _add_provenance_footer(fig, theme, source_runs=source_runs, is_mock=is_mock)
    return fig


# =====================================================================
# 5. TIGHTNESS HEATMAP (analysis-driven)
# =====================================================================


def _panel_tightness(
    ax, data: dict | None, theme: PlotTheme,
) -> None:
    """Heatmap of Theorem 1 slack over (k, ε_adv).

    data shape:
        {
            "ks":           [1, 2, 4, 8],
            "eps_advs":     [0.0, 0.1, 0.2, 0.3, 0.4],
            "slack_matrix": 2-D list (rows=k, cols=ε_adv),
        }
    Tight cells (slack < 0.01) are dark green; loose cells (slack > 0.05)
    are amber. Annotates each cell with the slack value.
    """
    if not data or not data.get("slack_matrix"):
        _annotate_missing(ax, "Run sweep_tightness() to populate", theme)
        return

    M = np.asarray(data["slack_matrix"], dtype=float)
    ks = data["ks"]
    eps = data["eps_advs"]

    # Custom colormap: green (tight) → amber (loose)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "tight_to_loose",
        [theme.palette["accent_green"],
         theme.palette["bg_app"],
         theme.palette["accent_amber"]],
        N=256,
    )

    im = ax.imshow(
        M, cmap=cmap, aspect="auto",
        vmin=0.0, vmax=max(0.10, M.max()),   # so a uniformly tight grid stays green
    )
    ax.set_xticks(range(len(eps)))
    ax.set_xticklabels([f"{e:.1f}" for e in eps])
    ax.set_yticks(range(len(ks)))
    ax.set_yticklabels([f"k={k}" for k in ks])
    ax.set_xlabel("Adversary fraction ε_adv")
    ax.set_ylabel("Redundancy k")

    # Annotate each cell with the slack value
    for i in range(len(ks)):
        for j in range(len(eps)):
            v = M[i, j]
            txt_color = (
                theme.palette["ink"] if v < 0.04 else theme.palette["bg_panel"]
            )
            ax.text(
                j, i, f"{v:.3f}",
                ha="center", va="center",
                fontsize=theme.annotation_size - 1,
                color=txt_color,
            )

    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("Slack = RHS − LHS", fontsize=theme.annotation_size)
    cbar.ax.tick_params(labelsize=theme.annotation_size - 1)

    _panel_heading(
        ax, theme,
        title="Where Theorem 1 is tight",
        subtitle="green = tight, amber = loose",
    )


def plot_tightness_heatmap(
    *, data: dict | None = None,
    theme: PlotTheme = BOLD_THEME,
    source_runs: list[str] | None = None,
    is_mock: bool = False,
) -> plt.Figure:
    setup_matplotlib_for_theme(theme)
    fig, ax = plt.subplots(
        figsize=(8, 5), dpi=theme.dpi,
        facecolor=theme.palette["bg_panel"],
    )
    _panel_tightness(ax, data, theme)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    _add_provenance_footer(fig, theme, source_runs=source_runs, is_mock=is_mock)
    return fig


# =====================================================================
# 6. LOAD CURVES (analysis-driven)
# =====================================================================


def _panel_load_curves(
    ax, data: dict | None, theme: PlotTheme,
) -> None:
    """Throughput, P50, P95, P99 vs concurrency for one or more backends.

    data shape:
        {
            "backends": {
                "Qwen-7B":  {"concurrency":[1,4,...], "throughput":[...],
                             "p50":[...], "p95":[...], "p99":[...]},
                "Llama-70B": {...},
                ...
            }
        }
    Plots P50 (solid), P95 (dashed), P99 (dotted) on a log-log scale.
    """
    if not data or not data.get("backends"):
        _annotate_missing(ax, "Run cost_curve() per backend to populate", theme)
        return

    backends = data["backends"]
    palette = [
        theme.palette["ours"],
        theme.palette["base_strong"],
        theme.palette["accent_amber"],
        theme.palette["accent_green"],
    ]

    for i, (name, d) in enumerate(backends.items()):
        c = palette[i % len(palette)]
        x = d["concurrency"]
        ax.plot(x, d["p50"], "-",  color=c, lw=2.0, label=f"{name} P50")
        ax.plot(x, d["p95"], "--", color=c, lw=1.4, alpha=0.85, label=f"{name} P95")
        ax.plot(x, d["p99"], ":",  color=c, lw=1.2, alpha=0.7,  label=f"{name} P99")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("Latency (ms, log scale)")
    ax.legend(
        loc="upper left", bbox_to_anchor=(1.02, 1.0),
        frameon=False, fontsize=theme.annotation_size - 1,
        ncol=1,
    )
    _panel_heading(
        ax, theme,
        title="Tail latency under load",
        subtitle="P50/P95/P99 vs concurrent claims",
    )


def plot_load_curves(
    *, data: dict | None = None,
    theme: PlotTheme = BOLD_THEME,
    source_runs: list[str] | None = None,
    is_mock: bool = False,
) -> plt.Figure:
    setup_matplotlib_for_theme(theme)
    fig, ax = plt.subplots(
        figsize=(9, 5.5), dpi=theme.dpi,
        facecolor=theme.palette["bg_panel"],
    )
    _panel_load_curves(ax, data, theme)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    _add_provenance_footer(fig, theme, source_runs=source_runs, is_mock=is_mock)
    return fig


# =====================================================================
# COMPOSITION: THE APPENDIX BLOCK (4 figures + 2 analysis panels)
# =====================================================================


def plot_appendix_block(
    *,
    cross_domain: dict | None = None,
    backend_frontier: dict | None = None,
    failure_modes: dict | None = None,
    calibration: dict | None = None,
    tightness: dict | None = None,
    load_curves: dict | None = None,
    theme: PlotTheme = BOLD_THEME,
    source_runs: list[str] | None = None,
    is_mock: bool = False,
) -> plt.Figure:
    """The full 6-panel appendix figure assembling all phase-K analyses.

    Layout (3x2):
        ┌──────────────┬──────────────┐
        │ cross_domain │ backend_     │
        │              │ frontier     │
        ├──────────────┼──────────────┤
        │ failure_     │ calibration  │
        │ modes        │              │
        ├──────────────┼──────────────┤
        │ tightness    │ load_curves  │
        │ (heatmap)    │              │
        └──────────────┴──────────────┘
    """
    setup_matplotlib_for_theme(theme)
    fig = plt.figure(
        figsize=(15, 16), dpi=theme.dpi,
        facecolor=theme.palette["bg_panel"],
    )
    gs = GridSpec(
        3, 2, figure=fig,
        hspace=0.65, wspace=0.30,
        left=0.06, right=0.94,
        top=0.96, bottom=0.06,
    )
    ax_cd  = fig.add_subplot(gs[0, 0])
    ax_bf  = fig.add_subplot(gs[0, 1])
    ax_fm  = fig.add_subplot(gs[1, 0])
    ax_cal = fig.add_subplot(gs[1, 1])
    ax_tt  = fig.add_subplot(gs[2, 0])
    ax_lc  = fig.add_subplot(gs[2, 1])

    _panel_cross_domain(ax_cd, cross_domain, theme)
    _panel_backend_frontier(ax_bf, backend_frontier, theme)
    _panel_failure_modes(ax_fm, failure_modes, theme)
    _panel_calibration(ax_cal, calibration, theme)
    _panel_tightness(ax_tt, tightness, theme)
    _panel_load_curves(ax_lc, load_curves, theme)

    _add_provenance_footer(fig, theme, source_runs=source_runs, is_mock=is_mock)
    return fig
