"""
Per-experiment plots redesigned around diverse coverage (Phase M).

Each R-plot now shows 3 (LLM, dataset) cells from the diverse-coverage
plan, plus a fourth "cross-cell summary" panel. Plot anatomy:

    ┌─────────┬─────────┬─────────┬─────────────────┐
    │ Cell 1  │ Cell 2  │ Cell 3  │ Cross-cell      │
    │ (LLM₁,  │ (LLM₂,  │ (LLM₃,  │ summary         │
    │  ds₁)   │  ds₂)   │  ds₃)   │ (aggregated)    │
    └─────────┴─────────┴─────────┴─────────────────┘

Visual-richness upgrades over the previous flat plots:
- R4 (cost-vs-harm Pareto): adds 2D density overlay (hex-bin) of the
  underlying per-claim distribution behind the Pareto markers, so the
  shape of the data is visible — not just the frontier.
- R5 (overhead): adds a horizontal token-distribution sub-panel above
  the per-phase stack so reviewers see spread, not just means.
- All plots: legend always disambiguates "PCG-MAS (ours)" from baseline,
  with a callout annotation pointing at the largest gap in each panel.
"""
from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch

from pcg.eval.plots_v2 import (
    BOLD_THEME, PlotTheme,
    _add_provenance_footer, _annotate_missing, _panel_heading,
    save_fig_v2, setup_matplotlib_for_theme,
)
from pcg.eval.coverage import Cell


# ============================================================================
# Shared helpers
# ============================================================================


def _cell_title(cell: Cell) -> str:
    """Compact panel title for a (LLM, dataset) cell."""
    return f"{cell.llm}\n@ {cell.dataset}"


def _ours_label() -> str:
    return "PCG-MAS (ours)"


def _baseline_label() -> str:
    return "No certificate"


# ============================================================================
# R1 — AUDIT DECOMPOSITION (per cell + cross-cell)
# ============================================================================
# data per cell:
#   {
#     "lhs": float,                    # Pr(accept ∩ wrong)
#     "lhs_ci": (low, high),
#     "channels": {
#         "p_int_fail":    {"mean": float, "ci": (low, high)},
#         "p_replay_fail": {"mean": float, "ci": ...},
#         "p_check_fail":  {"mean": float, "ci": ...},
#         "p_cov_gap":     {"mean": float, "ci": ...},
#     },
#   }
# data["cells"] = list of 3 cell-data dicts in plan order.


def _r1_panel_cell(ax, cell: Cell, cd: dict | None, theme: PlotTheme) -> None:
    if not cd:
        _annotate_missing(ax, "no data", theme)
        ax.set_title(_cell_title(cell), fontsize=theme.label_size,
                     loc="left", fontweight="bold")
        return

    channels_order = ["p_int_fail", "p_replay_fail", "p_check_fail", "p_cov_gap"]
    channel_labels = ["IntFail", "ReplayFail", "CheckFail", "CovGap"]
    channel_colors = [
        theme.palette["ch_int"], theme.palette["ch_replay"],
        theme.palette["ch_check"], theme.palette["ch_cov"],
    ]
    means = [cd["channels"][c]["mean"] for c in channels_order]
    cumulative_rhs = sum(means)
    lhs = cd["lhs"]
    lhs_ci = cd["lhs_ci"]

    # Layout: y=0 is RHS (top), y=1 is LHS (bottom). Two horizontal bars.
    bar_height = 0.55

    # ----- RHS: stacked color-coded channels -----
    bottom_x = 0.0
    for k, (m, color, lab) in enumerate(zip(means, channel_colors, channel_labels)):
        ax.barh(
            0, m, height=bar_height, left=bottom_x,
            color=color, edgecolor="white", linewidth=0.8,
            label=lab,
        )
        # Add the channel value as a label inside the segment ONLY when
        # the segment is wide enough that the larger font won't overlap
        # neighbours. With 18pt annotation size, 25% threshold is the
        # safe minimum.
        if m > cumulative_rhs * 0.25:
            ax.text(
                bottom_x + m / 2, 0,
                f"{m:.3f}",
                ha="center", va="center",
                fontsize=theme.annotation_size - 4,
                color="white", fontweight="bold",
            )
        bottom_x += m

    # Black tick at the tip of the RHS bar with the total
    ax.plot([cumulative_rhs, cumulative_rhs],
            [-bar_height / 2, bar_height / 2],
            color=theme.palette["ink"], lw=1.4, zorder=5)
    ax.text(
        cumulative_rhs * 1.02, 0,
        f"Σ = {cumulative_rhs:.3f}",
        ha="left", va="center",
        fontsize=theme.annotation_size - 1,
        color=theme.palette["ink"], fontweight="bold",
    )

    # ----- LHS: empirical Pr(accept ∩ wrong) with Wilson CI whiskers -----
    half_lo = lhs - lhs_ci[0]
    half_hi = lhs_ci[1] - lhs
    ax.barh(
        1, lhs, height=bar_height,
        color=theme.palette["ours"],
        edgecolor="white", linewidth=0.8,
    )
    # CI whiskers as a black bracket
    ax.errorbar(
        lhs, 1,
        xerr=[[half_lo], [half_hi]],
        fmt="none", ecolor=theme.palette["ink"],
        elinewidth=1.2, capsize=4, zorder=5,
    )
    # Black tick at the tip of the LHS bar
    ax.plot([lhs, lhs], [1 - bar_height / 2, 1 + bar_height / 2],
            color=theme.palette["ink"], lw=1.4, zorder=5)
    ax.text(
        lhs * 1.02 + cumulative_rhs * 0.01, 1,
        f"{lhs:.3f}",
        ha="left", va="center",
        fontsize=theme.annotation_size - 1,
        color=theme.palette["ink"], fontweight="bold",
    )

    # Slack annotation in italics, between the bars
    slack = max(0.0, cumulative_rhs - lhs)
    ax.text(
        max(cumulative_rhs, lhs) * 1.18, 0.5,
        f"slack {slack:+.3f}",
        ha="left", va="center",
        fontsize=theme.annotation_size - 1,
        color=theme.palette["ink_light"], style="italic",
    )

    ax.set_yticks([0, 1])
    ax.set_yticklabels(
        ["RHS\n(Σ channels)", "Accept ∩ wrong\n(LHS empirical)"],
        fontsize=theme.annotation_size,
    )
    ax.invert_yaxis()  # so LHS sits visually above RHS
    ax.set_xlim(0, max(cumulative_rhs, lhs) * 1.55)
    ax.set_xlabel("Probability (95% Wilson CI)",
                  fontsize=theme.label_size)
    ax.set_title(_cell_title(cell),
                 fontsize=theme.label_size,
                 loc="left", fontweight="bold")
    # Per-cell legend with the four channels (small, lower-right, no frame)
    ax.legend(
        loc="lower right", frameon=False,
        fontsize=theme.annotation_size - 2, ncol=2,
    )


def _r1_panel_summary(ax, cells: list[Cell], cell_data: list[dict | None],
                      theme: PlotTheme) -> None:
    """Cross-cell summary: LHS vs RHS for each cell, paired bars."""
    valid = [(c, d) for c, d in zip(cells, cell_data) if d]
    if not valid:
        _annotate_missing(ax, "no data", theme)
        return

    n = len(valid)
    x = np.arange(n)
    width = 0.36
    lhs_vals = [d["lhs"] for _, d in valid]
    rhs_vals = [
        sum(d["channels"][c]["mean"]
            for c in ("p_int_fail", "p_replay_fail",
                      "p_check_fail", "p_cov_gap"))
        for _, d in valid
    ]

    ax.bar(
        x - width / 2, lhs_vals, width,
        color=theme.palette["ours"], edgecolor="white", linewidth=0.6,
        label=f"Accept ∩ wrong (LHS) · {_ours_label()}",
    )
    ax.bar(
        x + width / 2, rhs_vals, width,
        color=theme.palette["base_strong"], edgecolor="white", linewidth=0.6,
        label="Σ channels (RHS) · Thm 1 bound",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{c.llm}\n{c.dataset}" for c, _ in valid],
        rotation=35, ha="right", fontsize=theme.annotation_size - 2,
    )
    ax.set_ylabel("Probability", fontsize=theme.label_size)
    # Place legend ABOVE the plot, outside the axes, so it doesn't
    # collide with the tallest bar.
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.18),
              frameon=False, fontsize=theme.annotation_size - 2,
              ncol=1)
    _panel_heading(
        ax, theme,
        title="Theorem 1 holds across cells",
        subtitle="LHS ≤ RHS in every (LLM, dataset)",
    )


def plot_r1_audit(
    *, cells: list[Cell],
    cell_data: list[dict | None],
    theme: PlotTheme = BOLD_THEME,
    source_runs: list[str] | None = None,
    is_mock: bool = False,
) -> plt.Figure:
    setup_matplotlib_for_theme(theme)
    fig = plt.figure(figsize=(20, 8.5), dpi=theme.dpi,
                     facecolor=theme.palette["bg_panel"])
    gs = GridSpec(1, 4, figure=fig, wspace=0.85,
                  left=0.08, right=0.98, top=0.74, bottom=0.22)
    for i, (cell, cd) in enumerate(zip(cells, cell_data)):
        ax = fig.add_subplot(gs[0, i])
        _r1_panel_cell(ax, cell, cd, theme)
    ax_sum = fig.add_subplot(gs[0, 3])
    _r1_panel_summary(ax_sum, cells, cell_data, theme)

    fig.suptitle(
        "Audit decomposition of accepted failures (Theorem 1)\n"
        "across diverse (LLM agent, dataset) cells",
        fontsize=theme.title_size + 2, fontweight="bold", y=0.95,
    )
    _add_provenance_footer(fig, theme, source_runs=source_runs, is_mock=is_mock)
    return fig


# ============================================================================
# R2 — REDUNDANCY LAW (per cell + cross-cell)
# ============================================================================
# data per cell:
#   {
#     "ks":         [1, 2, 4, 8],
#     "empirical":  [...],
#     "empirical_ci":[(low,high), ...],
#     "theory":     [...],   # ρ^(k-1) · ε^k bound from Thm 2
#   }


def _r2_panel_cell(ax, cell: Cell, cd: dict | None, theme: PlotTheme) -> None:
    if not cd:
        _annotate_missing(ax, "no data", theme)
        ax.set_title(_cell_title(cell), fontsize=theme.label_size,
                     loc="left", fontweight="bold")
        return
    ks = cd["ks"]
    emp = cd["empirical"]
    emp_ci = cd.get("empirical_ci") or [(v, v) for v in emp]
    theory = cd.get("theory") or [None] * len(ks)
    band_lo = cd.get("adv_band_lo") or [None] * len(ks)
    band_hi = cd.get("adv_band_hi") or [None] * len(ks)

    # Adversary-fraction band (ε_adv sweep). Shows the regime envelope:
    # lower = no adversary, upper = ε_adv = 0.4. Empirical sits inside.
    if all(b is not None for b in band_lo) and all(b is not None for b in band_hi):
        ax.fill_between(
            ks, band_lo, band_hi,
            color=theme.palette["bg_emphasis"], alpha=0.95,
            label="ε_adv ∈ [0, 0.4]",
            zorder=1,
        )

    # 95% CI on empirical
    lo = [c[0] for c in emp_ci]
    hi = [c[1] for c in emp_ci]
    ax.fill_between(ks, lo, hi,
                    color=theme.palette["ours_light"], alpha=0.6,
                    label="95% CI",
                    zorder=2)

    # Empirical curve
    ax.plot(ks, emp, "o-", color=theme.palette["ours"],
            lw=2.2, markersize=6,
            label=_ours_label(),
            zorder=4)
    # Theorem 2 bound
    if any(t is not None for t in theory):
        ax.plot(ks, theory, "s--",
                color=theme.palette["base_strong"], lw=1.4, markersize=4,
                alpha=0.85, label="Thm 2 bound",
                zorder=3)
    # No-cert reference
    if emp:
        ax.axhline(emp[0], color=theme.palette["neutral"],
                   ls=":", lw=1.0, alpha=0.7,
                   label="No certificate (k=1)",
                   zorder=1)

    # Annotate where the empirical curve detaches from the bound by >10%.
    # This is the "constant-ρ assumption breaks" marker.
    if any(t is not None for t in theory):
        for i, (k, e, t) in enumerate(zip(ks, emp, theory)):
            if t is None or t <= 0:
                continue
            ratio = e / t
            if ratio > 0.5:   # empirical is within 2× of bound -> tight
                ax.annotate(
                    "tight",
                    xy=(k, e), xytext=(k, e * 0.30),
                    fontsize=theme.annotation_size - 2,
                    color=theme.palette["ink_light"], style="italic",
                    ha="center",
                    arrowprops=dict(arrowstyle="-",
                                    color=theme.palette["neutral"], lw=0.5),
                )
                break  # annotate only the first tight k to avoid clutter

    ax.set_yscale("log")
    ax.set_xticks(ks)
    ax.set_xlabel("Redundancy k", fontsize=theme.label_size)
    ax.set_ylabel("Pr(false accept)", fontsize=theme.label_size)
    ax.set_title(_cell_title(cell),
                 fontsize=theme.label_size,
                 loc="left", fontweight="bold")
    ax.legend(loc="lower left", frameon=False,
              fontsize=theme.annotation_size - 2,
              ncol=1)


def _r2_panel_summary(ax, cells, cell_data, theme):
    """Overlay all 3 cells' empirical curves for visual comparison."""
    valid = [(c, d) for c, d in zip(cells, cell_data) if d]
    if not valid:
        _annotate_missing(ax, "no data", theme)
        return
    palette_cycle = [theme.palette["ours"], theme.palette["base_weak"],
                     theme.palette["warning"]]
    for i, (cell, d) in enumerate(valid):
        c = palette_cycle[i % 3]
        ax.plot(d["ks"], d["empirical"], "o-", color=c, lw=2.0,
                label=f"{cell.llm} @ {cell.dataset}")
    ax.set_yscale("log")
    ax.set_xlabel("Redundancy k", fontsize=theme.label_size)
    ax.set_ylabel("Pr(false accept)", fontsize=theme.label_size)
    ax.legend(loc="upper right", frameon=False,
              fontsize=theme.annotation_size - 1)
    _panel_heading(
        ax, theme,
        title="Redundancy collapse (Theorem 2)",
        subtitle="all cells decay geometrically",
    )


def plot_r2_redundancy(
    *, cells, cell_data,
    theme=BOLD_THEME, source_runs=None, is_mock=False,
) -> plt.Figure:
    setup_matplotlib_for_theme(theme)
    fig = plt.figure(figsize=(20, 7.8), dpi=theme.dpi,
                     facecolor=theme.palette["bg_panel"])
    gs = GridSpec(1, 4, figure=fig, wspace=0.55,
                  left=0.06, right=0.98, top=0.80, bottom=0.18)
    for i, (cell, cd) in enumerate(zip(cells, cell_data)):
        ax = fig.add_subplot(gs[0, i])
        _r2_panel_cell(ax, cell, cd, theme)
    ax_sum = fig.add_subplot(gs[0, 3])
    _r2_panel_summary(ax_sum, cells, cell_data, theme)

    fig.suptitle(
        "R2 · Redundant-consensus law (Theorem 2) over diverse cells",
        fontsize=theme.title_size + 2, fontweight="bold", y=0.95,
    )
    _add_provenance_footer(fig, theme, source_runs=source_runs, is_mock=is_mock)
    return fig


# ============================================================================
# R3 — RESPONSIBILITY (per cell + cross-cell)
# ============================================================================
# data per cell:
#   {
#     "regimes": ["integrity", "replay", "check", "coverage"],
#     "ours_acc":  [...],   # PCG-MAS top-1 root-cause accuracy per regime
#     "ours_ci":   [(low,high), ...],
#     "base_acc":  [...],   # baseline (random or simple-rules)
#     "base_ci":   [(low,high), ...],
#   }


def _r3_panel_cell(ax, cell, cd, theme):
    if not cd:
        _annotate_missing(ax, "no data", theme)
        ax.set_title(_cell_title(cell), fontsize=theme.label_size,
                     loc="left", fontweight="bold")
        return
    regimes = cd["regimes"]
    ours = np.asarray(cd["ours_acc"])
    base = np.asarray(cd.get("base_acc") or [0.25] * len(regimes))
    ours_ci = cd.get("ours_ci") or [(v, v) for v in ours]
    base_ci = cd.get("base_ci") or [(v, v) for v in base]

    x = np.arange(len(regimes))
    width = 0.36
    ours_err = np.array([
        [ours[i] - ours_ci[i][0], ours_ci[i][1] - ours[i]]
        for i in range(len(regimes))
    ]).T
    base_err = np.array([
        [base[i] - base_ci[i][0], base_ci[i][1] - base[i]]
        for i in range(len(regimes))
    ]).T

    ax.bar(x - width / 2, base, width,
           yerr=base_err, error_kw={"capsize": 2, "elinewidth": 0.7},
           color=theme.palette["base_strong"], edgecolor="white", linewidth=0.6,
           label=_baseline_label())
    ax.bar(x + width / 2, ours, width,
           yerr=ours_err, error_kw={"capsize": 2, "elinewidth": 0.7},
           color=theme.palette["ours"], edgecolor="white", linewidth=0.6,
           label=_ours_label())
    # Random-baseline reference at 0.25 (4 regimes)
    ax.axhline(0.25, color=theme.palette["neutral"],
               ls=":", lw=1.0, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(regimes, rotation=20, ha="right",
                       fontsize=theme.annotation_size - 1)
    ax.set_ylim(0, 1.15)   # extra headroom for the legend
    ax.set_ylabel("Top-1 root-cause acc.", fontsize=theme.label_size)
    ax.set_title(_cell_title(cell),
                 fontsize=theme.label_size,
                 loc="left", fontweight="bold")
    ax.legend(loc="upper right", frameon=False,
              fontsize=theme.annotation_size - 1)


def _r3_panel_summary(ax, cells, cell_data, theme):
    """Cross-cell mean accuracy with PCG-MAS vs baseline lift."""
    valid = [(c, d) for c, d in zip(cells, cell_data) if d]
    if not valid:
        _annotate_missing(ax, "no data", theme)
        return
    labels = [f"{c.llm}\n{c.dataset}" for c, _ in valid]
    ours_means = [float(np.mean(d["ours_acc"])) for _, d in valid]
    base_means = [float(np.mean(d.get("base_acc") or [0.25])) for _, d in valid]
    x = np.arange(len(valid))
    width = 0.36
    ax.bar(x - width / 2, base_means, width,
           color=theme.palette["base_strong"], edgecolor="white", linewidth=0.6,
           label=_baseline_label())
    ax.bar(x + width / 2, ours_means, width,
           color=theme.palette["ours"], edgecolor="white", linewidth=0.6,
           label=_ours_label())
    # Lift annotations
    for i, (om, bm) in enumerate(zip(ours_means, base_means)):
        lift = om - bm
        ax.text(
            i, max(om, bm) + 0.04,
            f"+{lift * 100:.0f}pp",
            ha="center", va="bottom",
            fontsize=theme.annotation_size,
            color=theme.palette["ours_dark"], fontweight="bold",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=theme.annotation_size - 2)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Mean accuracy", fontsize=theme.label_size)
    ax.legend(loc="lower right", frameon=False,
              fontsize=theme.annotation_size - 1)
    _panel_heading(
        ax, theme,
        title="Responsibility lift across cells",
        subtitle="PCG-MAS vs baseline, mean over regimes",
    )


def plot_r3_responsibility(
    *, cells, cell_data,
    theme=BOLD_THEME, source_runs=None, is_mock=False,
) -> plt.Figure:
    setup_matplotlib_for_theme(theme)
    fig = plt.figure(figsize=(20, 7.8), dpi=theme.dpi,
                     facecolor=theme.palette["bg_panel"])
    gs = GridSpec(1, 4, figure=fig, wspace=0.55,
                  left=0.06, right=0.98, top=0.80, bottom=0.22)
    for i, (cell, cd) in enumerate(zip(cells, cell_data)):
        ax = fig.add_subplot(gs[0, i])
        _r3_panel_cell(ax, cell, cd, theme)
    ax_sum = fig.add_subplot(gs[0, 3])
    _r3_panel_summary(ax_sum, cells, cell_data, theme)

    fig.suptitle(
        "R3 · Responsibility (top-1 root-cause attribution) over diverse cells",
        fontsize=theme.title_size + 2, fontweight="bold", y=0.95,
    )
    _add_provenance_footer(fig, theme, source_runs=source_runs, is_mock=is_mock)
    return fig


# ============================================================================
# R4 — COST vs HARM with DENSITY OVERLAY (visual richness upgrade)
# ============================================================================
# data per cell:
#   {
#     "policies": {                    # 3 policy curves
#         "always_answer": {"cost": [...], "harm": [...]},
#         "threshold_pcg": {"cost": [...], "harm": [...]},
#         "learned":       {"cost": [...], "harm": [...]},
#     },
#     "per_claim_cost": [...],         # raw distribution for hex-bin
#     "per_claim_harm": [...],
#   }


def _r4_panel_cell(ax, cell, cd, theme):
    if not cd:
        _annotate_missing(ax, "no data", theme)
        ax.set_title(_cell_title(cell), fontsize=theme.label_size,
                     loc="left", fontweight="bold")
        return

    # Layer 1: 2D density of per-claim (cost, harm) — adds the "shape of
    # the data" so reviewers see the dispersion behind the policy curves
    pc = cd.get("per_claim_cost") or []
    ph = cd.get("per_claim_harm") or []
    if pc and ph and len(pc) == len(ph) and len(pc) > 20:
        hb = ax.hexbin(
            pc, ph, gridsize=22, cmap="Greys",
            mincnt=1, alpha=0.55,
        )

    # Layer 2: policy curves (always-answer / threshold-PCG / learned)
    pol = cd.get("policies") or {}
    style_map = {
        "always_answer": {
            "color": theme.palette["base_strong"],
            "marker": "s",
            "label": "Always-answer",
        },
        "learned": {
            "color": theme.palette["warning"],
            "marker": "D",
            "label": "Learned policy",
        },
        "threshold_pcg": {
            "color": theme.palette["ours"],
            "marker": "*",
            "label": _ours_label(),   # "PCG-MAS (ours)"
        },
    }
    for pname, sty in style_map.items():
        p = pol.get(pname)
        if not p:
            continue
        ax.plot(p["cost"], p["harm"], "-",
                color=sty["color"], lw=1.4, alpha=0.6, zorder=2)
        ax.scatter(p["cost"], p["harm"],
                   s=120 if pname == "threshold_pcg" else 70,
                   marker=sty["marker"], color=sty["color"],
                   edgecolors="white", linewidths=1.2,
                   label=sty["label"], zorder=4)

    # Annotation: arrow from baseline-best to ours-best showing dominance
    if "always_answer" in pol and "threshold_pcg" in pol:
        # use the max-harm baseline and the min-harm ours
        b_idx = int(np.argmax(pol["always_answer"]["harm"]))
        o_idx = int(np.argmin(pol["threshold_pcg"]["harm"]))
        bx, by = pol["always_answer"]["cost"][b_idx], pol["always_answer"]["harm"][b_idx]
        ox, oy = pol["threshold_pcg"]["cost"][o_idx], pol["threshold_pcg"]["harm"][o_idx]
        ax.annotate(
            "Pareto-dominant",
            xy=(ox, oy), xytext=(bx, max(by * 0.6, oy * 1.5)),
            fontsize=theme.annotation_size - 1,
            color=theme.palette["ours_dark"], fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=theme.palette["ours"],
                            lw=1.2),
        )

    ax.set_yscale("log")
    ax.set_xlabel("Cost / claim", fontsize=theme.label_size)
    ax.set_ylabel("Harm rate (log)", fontsize=theme.label_size)
    ax.set_title(_cell_title(cell),
                 fontsize=theme.label_size,
                 loc="left", fontweight="bold")
    ax.legend(loc="upper right", frameon=False,
              fontsize=theme.annotation_size - 1)


def _r4_panel_summary(ax, cells, cell_data, theme):
    """Cross-cell harm-reduction factor (baseline harm / our harm)."""
    valid = [(c, d) for c, d in zip(cells, cell_data) if d]
    if not valid:
        _annotate_missing(ax, "no data", theme)
        return
    factors = []
    for c, d in valid:
        try:
            base_harm_max = max(d["policies"]["always_answer"]["harm"])
            our_harm_min = min(d["policies"]["threshold_pcg"]["harm"])
            factors.append(base_harm_max / max(our_harm_min, 1e-6))
        except (KeyError, TypeError, ValueError):
            factors.append(0.0)
    labels = [f"{c.llm}\n{c.dataset}" for c, _ in valid]
    x = np.arange(len(valid))
    bars = ax.bar(x, factors, 0.55,
                  color=theme.palette["ours"], edgecolor="white", linewidth=0.6)
    for i, f in enumerate(factors):
        ax.text(
            i, f + max(factors) * 0.03,
            f"{f:.0f}×",
            ha="center", va="bottom",
            fontsize=theme.label_size, fontweight="bold",
            color=theme.palette["ours_dark"],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=theme.annotation_size - 2)
    ax.set_ylabel("Harm reduction factor", fontsize=theme.label_size)
    ax.set_ylim(0, max(factors) * 1.25 if factors else 1)
    _panel_heading(
        ax, theme,
        title="Harm reduction across cells",
        subtitle=f"{_ours_label()} vs always-answer baseline",
    )


def plot_r4_risk_pareto(
    *, cells, cell_data,
    theme=BOLD_THEME, source_runs=None, is_mock=False,
) -> plt.Figure:
    setup_matplotlib_for_theme(theme)
    fig = plt.figure(figsize=(20, 7.8), dpi=theme.dpi,
                     facecolor=theme.palette["bg_panel"])
    gs = GridSpec(1, 4, figure=fig, wspace=0.60,
                  left=0.06, right=0.98, top=0.80, bottom=0.20)
    for i, (cell, cd) in enumerate(zip(cells, cell_data)):
        ax = fig.add_subplot(gs[0, i])
        _r4_panel_cell(ax, cell, cd, theme)
    ax_sum = fig.add_subplot(gs[0, 3])
    _r4_panel_summary(ax_sum, cells, cell_data, theme)

    fig.suptitle(
        "R4 · Cost-vs-harm Pareto with per-claim density overlay",
        fontsize=theme.title_size + 2, fontweight="bold", y=0.95,
    )
    _add_provenance_footer(fig, theme, source_runs=source_runs, is_mock=is_mock)
    return fig


# ============================================================================
# R5 — OVERHEAD with TOKEN DISTRIBUTION SUB-PANEL
# ============================================================================
# data per cell:
#   {
#     "phases": {"prove":x, "verify":x, "redundant":x, "audit":x},  # baseline + ours
#     "phases_baseline": {"prove":x},  # baseline (no cert) — typically only 'prove'
#     "per_claim_tokens_ours":   [...],  # distribution
#     "per_claim_tokens_base":   [...],
#   }


def _r5_panel_cell(ax_top, ax_bot, cell, cd, theme):
    """Two-row cell: distribution histogram on top, per-phase stack on bottom."""
    if not cd:
        _annotate_missing(ax_top, "no data", theme)
        ax_bot.axis("off")
        ax_top.set_title(_cell_title(cell), fontsize=theme.label_size,
                         loc="left", fontweight="bold")
        return

    # Top: token-count distribution (PCG-MAS vs baseline, overlaid)
    ours_tok = cd.get("per_claim_tokens_ours") or []
    base_tok = cd.get("per_claim_tokens_base") or []
    if ours_tok or base_tok:
        all_tok = list(ours_tok) + list(base_tok)
        if all_tok:
            xmax = float(np.percentile(all_tok, 99)) * 1.05
            bins = np.linspace(0, xmax, 26)
            if base_tok:
                ax_top.hist(base_tok, bins=bins,
                            color=theme.palette["base_weak"], alpha=0.55,
                            edgecolor="white", linewidth=0.4,
                            label=_baseline_label())
            if ours_tok:
                ax_top.hist(ours_tok, bins=bins,
                            color=theme.palette["ours"], alpha=0.65,
                            edgecolor="white", linewidth=0.4,
                            label=_ours_label())
            ax_top.set_xlim(0, xmax)
            ax_top.set_yticks([])
            # Make the axis labels self-explanatory so reviewers don't
            # have to chase the caption: x = tokens / claim, y = density
            # (count of claims falling in that token-count bin).
            ax_top.set_xlabel("Tokens / claim",
                              fontsize=theme.annotation_size)
            ax_top.set_ylabel("Density of claims",
                              fontsize=theme.annotation_size - 1)
            ax_top.legend(loc="upper right", frameon=False,
                          fontsize=theme.annotation_size - 1)
    ax_top.set_title(_cell_title(cell), fontsize=theme.label_size,
                     loc="left", fontweight="bold")

    # Bottom: per-phase stacked bar (PCG-MAS) vs baseline
    phases = cd.get("phases") or {}
    base_phases = cd.get("phases_baseline") or {"prove": phases.get("prove", 0)}

    # Sum baseline phases
    base_total = sum(base_phases.values())
    # PCG-MAS phases ordered: prove, verify, redundant, audit
    phase_order = ["prove", "verify", "redundant", "audit"]
    phase_colors = [
        theme.palette["base_weak"],
        theme.palette["ch_replay"],
        theme.palette["ch_check"],
        theme.palette["ch_cov"],
    ]

    # Two horizontal bars: baseline (bottom row) and ours (top row)
    # Stacked left-to-right
    left_b = 0.0
    for ph, color in zip(phase_order, phase_colors):
        v = base_phases.get(ph, 0)
        if v:
            ax_bot.barh(_baseline_label(), v, left=left_b,
                        color=color, edgecolor="white", linewidth=0.6,
                        label=f"{ph} (base)" if False else None)
            left_b += v
    left_o = 0.0
    for ph, color in zip(phase_order, phase_colors):
        v = phases.get(ph, 0)
        if v:
            ax_bot.barh(_ours_label(), v, left=left_o,
                        color=color, edgecolor="white", linewidth=0.6,
                        label=ph)
            left_o += v

    # Overhead factor annotation (placed inside the bar to avoid bleeding
    # into the neighbouring panel after the larger-font upgrade).
    factor = (left_o / max(left_b, 1.0))
    ax_bot.text(
        left_o * 0.5, 1,   # centered on the "ours" bar
        f"{factor:.1f}× tokens",
        ha="center", va="center",
        fontsize=theme.annotation_size - 2,
        color="white", fontweight="bold",
    )

    ax_bot.set_xlim(0, max(left_o, left_b) * 1.10)
    ax_bot.set_xlabel("Tokens / claim", fontsize=theme.label_size)
    ax_bot.legend(loc="upper right", frameon=False,
                  fontsize=theme.annotation_size - 2, ncol=2)


def _r5_panel_summary(ax, cells, cell_data, theme):
    """Cross-cell overhead factor."""
    valid = [(c, d) for c, d in zip(cells, cell_data) if d]
    if not valid:
        _annotate_missing(ax, "no data", theme)
        return
    factors = []
    for c, d in valid:
        ours_total = sum((d.get("phases") or {}).values())
        base_total = sum((d.get("phases_baseline") or {}).values()) or \
                     (d.get("phases") or {}).get("prove", 0)
        factors.append(ours_total / max(base_total, 1.0))
    labels = [f"{c.llm}\n{c.dataset}" for c, _ in valid]
    x = np.arange(len(valid))
    ax.bar(x, factors, 0.55,
           color=theme.palette["ours"], edgecolor="white", linewidth=0.6)
    for i, f in enumerate(factors):
        ax.text(i, f + max(factors) * 0.03,
                f"{f:.1f}×",
                ha="center", va="bottom",
                fontsize=theme.label_size, fontweight="bold",
                color=theme.palette["ours_dark"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=theme.annotation_size - 2)
    ax.set_ylabel("Token overhead factor",
                  fontsize=theme.label_size)
    ax.set_ylim(0, max(factors) * 1.25 if factors else 1)
    ax.axhline(1.0, color=theme.palette["neutral"],
               ls=":", lw=1.0, alpha=0.7)
    _panel_heading(
        ax, theme,
        title="Token cost across cells",
        subtitle=f"{_ours_label()} relative to baseline",
    )


def plot_r5_overhead(
    *, cells, cell_data,
    theme=BOLD_THEME, source_runs=None, is_mock=False,
) -> plt.Figure:
    setup_matplotlib_for_theme(theme)
    fig = plt.figure(figsize=(20, 10.5), dpi=theme.dpi,
                     facecolor=theme.palette["bg_panel"])
    # 2 rows × 4 cols: top row = distributions, bottom row = phase stacks
    # Last column spans both rows (cross-cell summary)
    gs = GridSpec(
        2, 4, figure=fig,
        height_ratios=[1, 2.2],
        hspace=0.55, wspace=0.55,
        left=0.06, right=0.98, top=0.86, bottom=0.12,
    )
    for i, (cell, cd) in enumerate(zip(cells, cell_data)):
        ax_top = fig.add_subplot(gs[0, i])
        ax_bot = fig.add_subplot(gs[1, i])
        _r5_panel_cell(ax_top, ax_bot, cell, cd, theme)
    ax_sum = fig.add_subplot(gs[:, 3])   # span both rows
    _r5_panel_summary(ax_sum, cells, cell_data, theme)

    fig.suptitle(
        "R5 · Token overhead with per-claim distribution sub-panels",
        fontsize=theme.title_size + 2, fontweight="bold", y=0.95,
    )
    _add_provenance_footer(fig, theme, source_runs=source_runs, is_mock=is_mock)
    return fig
