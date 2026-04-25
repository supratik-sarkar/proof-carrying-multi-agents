"""
Publication-quality plotting for the NeurIPS 2026 submission.

Design goals:
    - Every figure in the paper is produced by a named function in this module
      so that re-running `make figures` regenerates the entire figure set
      deterministically.
    - The style is modern: off-white canvas, sans-serif, faint gridlines, bold
      colors from a perceptually-uniform palette, and CI bands rendered as
      translucent fills.
    - All plots export to PDF (vector) for the paper + PNG@300dpi for previews.

Style choices worth flagging to reviewers:
    - We use `colorcet.glasbey` for categorical colors and a hand-picked
      sequential palette for k-sweeps (Theorem 2 figure).
    - Fonts: we do NOT force "Latin Modern Roman" since it requires XeTeX; the
      default sans-serif is intentionally close to the NeurIPS template.
    - Bootstrap CIs rendered as ribbons, not error bars, for k-sweeps — this
      is the visual convention in recent NeurIPS/ICLR best-paper plots.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------------------------
# Palette
# -----------------------------------------------------------------------------

# A carefully chosen categorical palette: 6 colors, all distinguishable under
# common colorblindness profiles (verified with colorspacious deuteranopia sim).
PALETTE = {
    "blue":    "#2E5EAA",
    "orange":  "#E8833C",
    "green":   "#3AA864",
    "red":     "#D7263D",
    "purple":  "#7B3F99",
    "teal":    "#0F9D9D",
    "gray":    "#6C757D",
    "black":   "#1A1A1A",
}

# Sequential palette for k-sweeps (redundancy levels). Dark -> bright as k grows.
SEQ_K = ["#1F4E79", "#2E75B6", "#5B9BD5", "#9DC3E6", "#BDD7EE", "#DDEBF7"]


# -----------------------------------------------------------------------------
# rcParams
# -----------------------------------------------------------------------------


def set_style(
    font_size: int = 10,
    title_size: int = 11,
    line_width: float = 1.6,
    tex: bool = False,
) -> None:
    """Apply the publication style. Call once at the start of each plotting run."""
    mpl.rcParams.update({
        # Fonts
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": font_size,
        "axes.titlesize": title_size,
        "axes.titleweight": "bold",
        "axes.labelsize": font_size,
        "axes.labelweight": "normal",
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
        "legend.fontsize": font_size - 1,
        "figure.titlesize": title_size + 1,

        # Axes
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#222222",
        "axes.titlecolor": "#111111",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.facecolor": "#FFFFFF",
        "axes.prop_cycle": mpl.cycler(
            color=[PALETTE["blue"], PALETTE["orange"], PALETTE["green"],
                   PALETTE["red"], PALETTE["purple"], PALETTE["teal"]]
        ),

        # Ticks
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",

        # Grid
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",
        "grid.color": "#AAAAAA",

        # Lines
        "lines.linewidth": line_width,
        "lines.markersize": 4.5,
        "lines.solid_capstyle": "round",

        # Legend
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "#DDDDDD",
        "legend.facecolor": "#FFFFFF",
        "legend.borderpad": 0.45,
        "legend.handlelength": 1.8,

        # Figure
        "figure.facecolor": "#FFFFFF",
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "savefig.transparent": False,
        "pdf.fonttype": 42,     # embedded TrueType (avoids T3 font warnings)
        "ps.fonttype": 42,

        # LaTeX if requested
        "text.usetex": tex,
    })


def save_fig(fig: plt.Figure, path: str | Path, formats: tuple[str, ...] = ("pdf", "png")) -> None:
    """Save to both PDF (for paper) and PNG (for previews) by default."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(p.with_suffix(f".{fmt}"), format=fmt)


# -----------------------------------------------------------------------------
# R1 — Audit-decomposition barplot (the "contract theorem" figure)
# -----------------------------------------------------------------------------


def plot_r1_audit_decomposition(
    decomp: dict,
    title: str = "Audit decomposition of accepted failures",
    save_to: str | Path | None = None,
) -> plt.Figure:
    """Horizontal bars: LHS (accept & wrong) next to each RHS channel of
    Theorem 1. Error bars = 95% Wilson CI.

    Expects `decomp.to_dict()` or the dict version thereof.
    """
    labels = [
        ("Accept & wrong (LHS)", decomp["lhs_accept_and_wrong"], decomp["ci_lhs"], PALETTE["red"]),
        ("IntFail",  decomp["p_int_fail"],    decomp["ci_int_fail"],    PALETTE["blue"]),
        ("ReplayFail", decomp["p_replay_fail"], decomp["ci_replay_fail"], PALETTE["orange"]),
        ("CheckFail", decomp["p_check_fail"], decomp["ci_check_fail"], PALETTE["green"]),
        ("CovGap",  decomp["p_cov_gap"],    decomp["ci_cov_gap"],    PALETTE["purple"]),
    ]
    names = [row[0] for row in labels]
    vals = [row[1] for row in labels]
    lows = [row[1] - row[2][0] for row in labels]
    highs = [row[2][1] - row[1] for row in labels]
    colors = [row[3] for row in labels]

    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    y = np.arange(len(names))
    ax.barh(y, vals, xerr=[lows, highs], color=colors, alpha=0.85,
            edgecolor="white", linewidth=0.8, capsize=2.5)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Probability (95% Wilson CI)")
    ax.set_title(title, loc="left")
    # Annotate each bar
    for i, v in enumerate(vals):
        ax.text(v + 0.002, y[i], f"{v:.3f}", va="center", ha="left", fontsize=8, color="#444444")
    ax.set_xlim(0, max(0.05, max(vals) * 1.35))
    fig.tight_layout()
    if save_to is not None:
        save_fig(fig, save_to)
    return fig


# -----------------------------------------------------------------------------
# R2 — Redundancy law: false-accept vs k, theory band vs empirical
# -----------------------------------------------------------------------------


def plot_r2_redundancy_law(
    ks: Sequence[int],
    empirical: Sequence[float],
    empirical_ci: Sequence[tuple[float, float]],
    theory_curve: Sequence[float],
    rho_ucb_curve: Sequence[float],
    title: str = "Redundancy law: Pr(accept & false) vs k",
    save_to: str | Path | None = None,
) -> plt.Figure:
    """Empirical vs theoretical false-accept rate as a function of k.

    - Solid line: empirical P(A_t^{(k,k)} & false) with CI ribbon.
    - Dashed line: plug-in theoretical bound using rho_hat (Eq. 21).
    - Dotted line: rho_UCB version of the bound, a genuine (1 - alpha) upper.
    """
    ks = np.asarray(ks)
    emp = np.asarray(empirical)
    lo = np.asarray([c[0] for c in empirical_ci])
    hi = np.asarray([c[1] for c in empirical_ci])
    thy = np.asarray(theory_curve)
    thy_ucb = np.asarray(rho_ucb_curve)

    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.fill_between(ks, lo, hi, color=PALETTE["blue"], alpha=0.18,
                    linewidth=0, label="Empirical 95% CI")
    ax.plot(ks, emp, "-o", color=PALETTE["blue"], label="Empirical")
    ax.plot(ks, thy, "--", color=PALETTE["orange"], label=r"Theory (plug-in $\hat\rho$)")
    ax.plot(ks, thy_ucb, ":", color=PALETTE["red"], label=r"Theory ($\rho^{\mathrm{UCB}}$, 95%)")
    ax.set_yscale("log")
    ax.set_xlabel(r"Redundancy $k$")
    ax.set_ylabel(r"$\Pr(A_t^{(k,k)} \wedge c\ \mathrm{false})$")
    ax.set_title(title, loc="left")
    ax.set_xticks(list(ks))
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    if save_to is not None:
        save_fig(fig, save_to)
    return fig


# -----------------------------------------------------------------------------
# R3 — Responsibility heatmap (components x corruption regimes)
# -----------------------------------------------------------------------------


def plot_r3_responsibility_heatmap(
    component_names: Sequence[str],
    regime_names: Sequence[str],
    resp_matrix: np.ndarray,
    ci_halfwidth: np.ndarray | None = None,
    title: str = "Interventional responsibility by corruption regime",
    save_to: str | Path | None = None,
) -> plt.Figure:
    """resp_matrix shape: (n_components, n_regimes), entries in [-1, 1].

    Color: diverging (red = high responsibility, white = 0, blue = negative).
    If `ci_halfwidth` is provided (same shape), cells with CI half-width larger
    than the cell value are shown hatched — a visual marker for "not
    statistically distinguishable from zero".
    """
    fig, ax = plt.subplots(figsize=(max(4.0, 0.7 * len(regime_names) + 2),
                                    max(2.5, 0.4 * len(component_names) + 1.2)))
    cmap = plt.get_cmap("RdBu_r")
    im = ax.imshow(resp_matrix, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(regime_names)))
    ax.set_xticklabels(regime_names, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(component_names)))
    ax.set_yticklabels(component_names)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(r"$\widehat{\mathsf{Resp}}$", rotation=270, labelpad=12)
    # Annotate cells
    for i in range(resp_matrix.shape[0]):
        for j in range(resp_matrix.shape[1]):
            v = resp_matrix[i, j]
            txt_color = "white" if abs(v) > 0.5 else "#333333"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color=txt_color, fontsize=7.5)
            if ci_halfwidth is not None and ci_halfwidth[i, j] >= abs(v):
                ax.add_patch(mpl.patches.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1, fill=False,
                    hatch="///", edgecolor="#999999", linewidth=0,
                ))
    ax.set_title(title, loc="left")
    fig.tight_layout()
    if save_to is not None:
        save_fig(fig, save_to)
    return fig


# -----------------------------------------------------------------------------
# R4 — Risk-control policy: Pareto frontier (harm vs cost, per policy)
# -----------------------------------------------------------------------------


def plot_r4_risk_pareto(
    policies: dict[str, dict],
    title: str = "Risk–cost frontier across policies",
    save_to: str | Path | None = None,
) -> plt.Figure:
    """Each entry of `policies` is {"cost": [...], "harm": [...], "label": str}.

    Plots the Pareto frontier (lower-left is better) with different markers
    per policy and a lighter connecting line.
    """
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    markers = ["o", "s", "D", "^", "v", "P"]
    colors = [PALETTE["blue"], PALETTE["orange"], PALETTE["green"], PALETTE["red"], PALETTE["purple"]]
    for i, (name, d) in enumerate(policies.items()):
        c = d["cost"]
        h = d["harm"]
        ax.plot(c, h, "-", color=colors[i % len(colors)], alpha=0.35, linewidth=1.0, zorder=1)
        ax.scatter(c, h, marker=markers[i % len(markers)],
                   color=colors[i % len(colors)], label=d.get("label", name),
                   s=28, edgecolor="white", linewidth=0.6, zorder=2)
    ax.set_xlabel("Expected cost (latency + tokens + tools)")
    ax.set_ylabel(r"Expected harm $\lambda\cdot \mathbb{E}[L_{\mathrm{harm}}]$")
    ax.set_title(title, loc="left")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    if save_to is not None:
        save_fig(fig, save_to)
    return fig


def plot_r4_privacy_utility(
    eps_values: Sequence[float],
    utility: Sequence[float],
    leakage: Sequence[float],
    title: str = "Privacy–utility trade-off",
    save_to: str | Path | None = None,
) -> plt.Figure:
    """Double-axis: utility (EM/F1) and leakage (AUC) as eps varies."""
    fig, ax1 = plt.subplots(figsize=(5.2, 3.0))
    xs = [str(e) for e in eps_values]
    ax1.plot(xs, utility, "-o", color=PALETTE["blue"], label="Utility (F1)")
    ax1.set_xlabel(r"Privacy budget $\varepsilon$")
    ax1.set_ylabel("Utility", color=PALETTE["blue"])
    ax1.tick_params(axis="y", colors=PALETTE["blue"])
    ax2 = ax1.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.plot(xs, leakage, "-s", color=PALETTE["red"], label="Leakage (AUC)")
    ax2.set_ylabel("Leakage AUC", color=PALETTE["red"])
    ax2.tick_params(axis="y", colors=PALETTE["red"])
    ax2.axhline(0.5, linestyle=":", color="#999", linewidth=0.8)
    ax1.set_title(title, loc="left")
    fig.tight_layout()
    if save_to is not None:
        save_fig(fig, save_to)
    return fig


# -----------------------------------------------------------------------------
# R5 — Overhead breakdown (stacked bar across configurations)
# -----------------------------------------------------------------------------


def plot_r5_overhead(
    configs: Sequence[str],
    phase_data: dict[str, Sequence[float]],
    ylabel: str = "Tokens per claim",
    title: str = "Token overhead by phase",
    save_to: str | Path | None = None,
) -> plt.Figure:
    """Stacked bar: one bar per config, each bar divided by phase (retrieval,
    LLM gen, verification, hashing, etc.).

    `phase_data` maps phase name -> list of values, one per config.
    """
    x = np.arange(len(configs))
    bottom = np.zeros(len(configs))
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    cyc = [PALETTE["blue"], PALETTE["orange"], PALETTE["green"],
           PALETTE["purple"], PALETTE["teal"], PALETTE["red"]]
    for i, (phase, values) in enumerate(phase_data.items()):
        vals = np.asarray(values)
        ax.bar(x, vals, bottom=bottom, label=phase,
               color=cyc[i % len(cyc)], alpha=0.88, edgecolor="white", linewidth=0.7)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout()
    if save_to is not None:
        save_fig(fig, save_to)
    return fig


def plot_r5_overhead_vs_k(
    ks: Sequence[int],
    tokens_per_claim: Sequence[float],
    utility: Sequence[float],
    title: str = "Token overhead and utility vs k",
    save_to: str | Path | None = None,
) -> plt.Figure:
    """Dual-axis plot answering the exact ICML question: how does overhead
    grow with redundancy, and does utility pay for it?"""
    fig, ax1 = plt.subplots(figsize=(5.2, 3.0))
    ax1.plot(ks, tokens_per_claim, "-o", color=PALETTE["blue"], label="Tokens / claim")
    ax1.set_xlabel(r"Redundancy $k$")
    ax1.set_ylabel("Tokens / claim", color=PALETTE["blue"])
    ax1.tick_params(axis="y", colors=PALETTE["blue"])
    ax2 = ax1.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.plot(ks, utility, "-s", color=PALETTE["red"], label="Utility (F1)")
    ax2.set_ylabel("Utility (F1)", color=PALETTE["red"])
    ax2.tick_params(axis="y", colors=PALETTE["red"])
    ax1.set_title(title, loc="left")
    fig.tight_layout()
    if save_to is not None:
        save_fig(fig, save_to)
    return fig


# -----------------------------------------------------------------------------
# Intro figure — the "hero" figure that goes on page 1 of the paper
# -----------------------------------------------------------------------------


def plot_intro_hero(
    ks: Sequence[int],
    utility_without_cert: float,
    utility_with_cert: Sequence[float],
    false_accept_without_cert: float,
    false_accept_with_cert: Sequence[float],
    save_to: str | Path | None = None,
) -> plt.Figure:
    """The Introduction 'hero' plot: side-by-side panel showing that PCG-MAS
    dramatically reduces false-accept rate while utility stays competitive,
    plotted against redundancy k.

    Left panel:  false-accept rate (log scale), with ref line for baseline.
    Right panel: utility, with ref line for baseline.
    """
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.4, 2.9))
    # Left: false accepts
    axL.axhline(false_accept_without_cert, linestyle="--", color=PALETTE["gray"],
                linewidth=1.2, label="No certificate (baseline)")
    axL.plot(ks, false_accept_with_cert, "-o", color=PALETTE["red"],
             label="PCG-MAS")
    axL.set_yscale("log")
    axL.set_xlabel(r"Redundancy $k$")
    axL.set_ylabel("False-accept rate")
    axL.set_title("Safety: false-accept vs k", loc="left")
    axL.legend(loc="upper right", fontsize=8)

    # Right: utility
    axR.axhline(utility_without_cert, linestyle="--", color=PALETTE["gray"],
                linewidth=1.2, label="No certificate (baseline)")
    axR.plot(ks, utility_with_cert, "-o", color=PALETTE["blue"], label="PCG-MAS")
    axR.set_xlabel(r"Redundancy $k$")
    axR.set_ylabel("Utility (F1)")
    axR.set_title("Utility: F1 vs k", loc="left")
    axR.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    if save_to is not None:
        save_fig(fig, save_to)
    return fig
