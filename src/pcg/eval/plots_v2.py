"""
plots_v2.py — NeurIPS-grade hero figure system.

A NeurIPS hero figure has roughly 5 seconds to make a reviewer care about
the paper. This module delivers a 4-panel layout that tells the entire
impact story at a single glance:

    Top row (full width):  certificate Z anatomy schematic  — what we built
    Bottom row, left:      Safety  — false-accept rate vs k (the headline)
    Bottom row, middle:    Audit   — Theorem 1 4-channel decomposition + bound
    Bottom row, right:     Tradeoff — cost vs harm Pareto frontier

Plus a `plot_summary_benchmark` for the appendix / website that aggregates
results across all backends including DeepSeek into a 2x2 comparison panel.

DESIGN PRINCIPLES (the software-engineering perspective)

  1. Separation of computation and rendering.
     Plot functions accept already-computed statistics, never raw data.
     Lets you re-render in different styles without touching experiments.

  2. Single source of style truth.
     PlotTheme dataclass holds palette/fonts/sizes. Switch from bold to
     print-safe in one line, not eight files.

  3. Composable panels.
     Each panel is `_panel_*(ax, data, theme) -> None`. The hero composes
     four of them via gridspec; the summary benchmark reuses the same
     panels with different data slices.

  4. Defensive defaults.
     Every keyword argument is optional. Missing field → grayed reference.
     Missing CI → no error bar (not a crash). Missing run → friendly
     "Run experiment Rn to populate" placeholder. Never blows up.

  5. Self-annotating output.
     Tiny footer carries git SHA, source run IDs, and AOE timestamp.
     Invisible at thumbnail scale; readable when zoomed. Solves "which
     version of the data made this figure?" reviewer questions silently.

  6. Mock-aware watermarking.
     When source data came from a `mock` backend, the figure auto-tags
     itself with a diagonal "MOCK BACKEND · PREVIEW ONLY" stamp. Removed
     automatically when real-LLM runs replace the mock results. Prevents
     accidentally screenshotting mock numbers into the camera-ready.

  7. Format hygiene.
     PDF (vector) for paper inclusion, PNG (raster) for README embedding.
     Both at 300 DPI; TrueType fonts embedded so Overleaf doesn't substitute.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


# =====================================================================
# THEME SYSTEM
# =====================================================================


@dataclass(frozen=True)
class PlotTheme:
    """Single source of truth for visual style.

    Frozen so panels can't accidentally mutate the theme passed to them
    (one panel's hack shouldn't change the look of others).
    """
    name: str
    palette: dict[str, str]
    font_family: str = "Helvetica"
    font_fallbacks: tuple[str, ...] = (
        "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"
    )
    base_size: int = 9
    title_size: int = 11
    label_size: int = 9
    tick_size: int = 8
    annotation_size: int = 8
    grid_alpha: float = 0.18
    grid_lw: float = 0.6
    dpi: int = 300


# Bold theme — high-contrast with one strong accent (red = "ours").
# Inspired by Tableau-bold + Tufte's preference for one-strong-color emphasis.
_BOLD_PALETTE = {
    # Primary identity
    "ours":          "#E63946",   # vivid red — our method
    "ours_light":    "#FAB6BB",   # CI fills
    "ours_dark":     "#9D1C2A",   # for emphasis text/lines

    # Baselines
    "base_strong":   "#1D3557",   # deep navy — strongest baseline
    "base_strong_l": "#9DB1C7",
    "base_weak":     "#457B9D",   # medium blue
    "base_weak_l":   "#B5CADC",

    # Audit channels (Theorem 1)
    "ch_int":        "#264653",   # dark teal — IntFail (lowest-level)
    "ch_replay":     "#2A9D8F",   # teal — ReplayFail
    "ch_check":      "#E9C46A",   # yellow — CheckFail
    "ch_cov":        "#F4A261",   # orange — CovGap (highest-level)

    # Status / semantic
    "success":       "#06A77D",
    "warning":       "#F77F00",
    "danger":        "#D62828",

    # Backgrounds and ink
    "bg_panel":      "#FFFFFF",
    "bg_subtle":     "#F8F9FA",
    "bg_emphasis":   "#FFF4E6",   # very pale orange — highlighted regions
    "ink":           "#1A1A2E",   # near-black, slightly purple
    "ink_light":     "#5C6370",   # secondary text
    "neutral":       "#9CA3AF",   # disabled / structural elements
}

BOLD_THEME = PlotTheme(
    name="bold",
    palette=_BOLD_PALETTE,
    base_size=10,
    title_size=12,
    label_size=10,
    tick_size=9,
    annotation_size=9,
)


# =====================================================================
# UTILITIES
# =====================================================================


def setup_matplotlib_for_theme(theme: PlotTheme = BOLD_THEME) -> None:
    """Apply theme-wide rcParams. Idempotent — safe to call repeatedly."""
    # Silence matplotlib's noisy font-fallback warnings. Helvetica/Arial
    # are not installed on most Linux containers (Colab, CI), so the user
    # gets a wall of "Font family ... not found" messages even though the
    # fallback to DejaVu Sans is fine. Suppress at module level.
    import logging
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

    plt.rcParams.update({
        "font.family": list(theme.font_fallbacks),
        "font.size": theme.base_size,
        "axes.titlesize": theme.title_size,
        "axes.labelsize": theme.label_size,
        "xtick.labelsize": theme.tick_size,
        "ytick.labelsize": theme.tick_size,
        "legend.fontsize": theme.tick_size,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.edgecolor": theme.palette["ink_light"],
        "axes.labelcolor": theme.palette["ink"],
        "xtick.color": theme.palette["ink"],
        "ytick.color": theme.palette["ink"],
        "axes.titleweight": "bold",
        "axes.titlecolor": theme.palette["ink"],
        # Print quality
        "pdf.fonttype": 42,    # TrueType, embedded
        "ps.fonttype": 42,
        "savefig.dpi": theme.dpi,
        "figure.dpi": 100,
        "savefig.bbox": "tight",
        "savefig.facecolor": theme.palette["bg_panel"],
    })


def _lighten(hex_color: str, amount: float = 0.85) -> str:
    """Mix hex_color toward white. amount=0 → original, amount=1 → white."""
    rgb = np.array(mcolors.to_rgb(hex_color))
    mixed = rgb * (1 - amount) + np.ones(3) * amount
    return mcolors.to_hex(np.clip(mixed, 0.0, 1.0))


def _annotate_missing(ax, message: str, theme: PlotTheme) -> None:
    """Draw a centered placeholder when data is unavailable."""
    ax.text(0.5, 0.5, message,
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=theme.label_size,
            color=theme.palette["ink_light"],
            style="italic")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(theme.palette["neutral"])
        spine.set_linestyle("--")
        spine.set_linewidth(0.7)
    ax.set_facecolor(theme.palette["bg_subtle"])


def _panel_heading(
    ax, theme: PlotTheme, title: str, subtitle: str | None = None,
) -> None:
    """Two-line panel heading anchored above the axes via ax.text (not
    ax.set_title), so the subtitle can never overflow into the next panel.

    Bold title on top, italic subtitle below — both left-aligned at x=0
    in axes coordinates so they line up cleanly across panels.
    """
    ax.text(
        0.0, 1.14, title,
        transform=ax.transAxes,
        fontsize=theme.title_size, fontweight="bold",
        color=theme.palette["ink"],
        ha="left", va="bottom",
    )
    if subtitle:
        ax.text(
            0.0, 1.04, subtitle,
            transform=ax.transAxes,
            fontsize=theme.tick_size,
            color=theme.palette["ink_light"],
            ha="left", va="bottom",
            style="italic",
        )


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _aoe_timestamp() -> str:
    aoe = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=-12)))
    return aoe.strftime("%Y-%m-%d %H:%M AOE")


def _add_provenance_footer(
    fig, theme: PlotTheme,
    source_runs: list[str] | None = None,
    is_mock: bool = False,
) -> None:
    """Bottom-left provenance line (git SHA · timestamp · source runs).
    Plus an optional centered diagonal MOCK watermark."""
    parts = [f"git: {_git_sha()}", _aoe_timestamp()]
    if source_runs:
        names = [(s[:42] + "…" if len(s) > 42 else s) for s in source_runs[:2]]
        parts.append("src: " + " · ".join(names))
    fig.text(
        0.005, 0.005, " · ".join(parts),
        fontsize=6, color=theme.palette["ink_light"],
        alpha=0.55, ha="left", va="bottom",
    )

    if is_mock:
        fig.text(
            0.5, 0.5, "MOCK BACKEND · PREVIEW ONLY",
            fontsize=44, color=theme.palette["warning"],
            alpha=0.07, ha="center", va="center",
            rotation=28, fontweight="bold",
            zorder=0,
        )


def detect_mock_runs(run_dirs: Sequence[Path | str]) -> bool:
    """Return True if ANY of the given run directories used a mock backend.

    Reads `config_snapshot.json` from each run directory and checks
    `backend.kind` (single-backend) or `backends[*].kind` (R5-style multi).
    """
    for d in run_dirs:
        if d is None:
            continue
        d = Path(d)
        cfg_path = d / "config_snapshot.json"
        if not cfg_path.exists():
            continue
        try:
            with cfg_path.open("r") as f:
                cfg = json.load(f)
        except Exception:
            continue
        if (cfg.get("backend", {}) or {}).get("kind") == "mock":
            return True
        for b in cfg.get("backends", []) or []:
            if isinstance(b, dict) and b.get("kind") == "mock":
                return True
    return False


# =====================================================================
# PANELS
# =====================================================================


def _panel_schematic(ax, theme: PlotTheme) -> None:
    """The certificate Z anatomy banner.

    Six rounded boxes — c, S, Π, Γ, p, meta — each with a colored border
    and matching light fill. Connected by a subtle horizontal line at
    mid-height to suggest they're a tuple. Title above, punchline below.
    """
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.set_aspect("auto")

    # Title and equation
    ax.text(
        50, 96, "The Grounding Certificate  $Z$",
        ha="center", va="top",
        fontsize=theme.title_size + 1, fontweight="bold",
        color=theme.palette["ink"],
    )
    ax.text(
        50, 81,
        r"$Z = (\,c,\;S,\;\Pi,\;\Gamma,\;p,\;\mathrm{meta}\,)$"
        "   —   every accepted claim is externally checkable",
        ha="center", va="top",
        fontsize=theme.label_size, style="italic",
        color=theme.palette["ink_light"],
    )

    components = [
        ("c",     "Claim",        "the answer",        "ours"),
        ("S",     "Evidence",     "hash-committed",    "base_strong"),
        ("Π",     "Pipeline",     "replayable",        "base_weak"),
        ("Γ",     "Contract",     "tools · schemas",   "ch_check"),
        ("p",     "Confidence",   "calibrated",        "ch_cov"),
        ("meta",  "Provenance",   "Merkle audit log",  "neutral"),
    ]

    n = len(components)
    margin_x = 3.0
    gap = 1.6
    available = 100 - 2 * margin_x - (n - 1) * gap
    box_w = available / n
    box_h = 30
    box_y = 22

    # Subtle connecting line at mid-height (suggests tuple structure)
    ax.plot(
        [margin_x, 100 - margin_x],
        [box_y + box_h / 2, box_y + box_h / 2],
        color=theme.palette["neutral"], lw=1, alpha=0.30, zorder=1,
    )

    for i, (sym, name, cap, color_key) in enumerate(components):
        x_left = margin_x + i * (box_w + gap)
        x_center = x_left + box_w / 2
        color = theme.palette[color_key]
        light = _lighten(color, 0.82)

        box = FancyBboxPatch(
            (x_left, box_y), box_w, box_h,
            boxstyle="round,pad=0.4,rounding_size=2.4",
            linewidth=1.8,
            edgecolor=color, facecolor=light, zorder=3,
        )
        ax.add_patch(box)

        # Symbol — large, top portion
        ax.text(
            x_center, box_y + box_h * 0.68, sym,
            ha="center", va="center",
            fontsize=theme.title_size + 4, fontweight="bold",
            color=color, zorder=4,
        )
        # Name — smaller, bottom portion
        ax.text(
            x_center, box_y + box_h * 0.25, name,
            ha="center", va="center",
            fontsize=theme.tick_size,
            color=theme.palette["ink"], zorder=4,
        )
        # Caption below the box
        ax.text(
            x_center, box_y - 3, cap,
            ha="center", va="top",
            fontsize=theme.annotation_size - 1,
            color=theme.palette["ink_light"],
            style="italic",
        )

    # Punchline at bottom
    ax.text(
        50, 4,
        "▶  external auditors verify in milliseconds without re-running the model",
        ha="center", va="bottom",
        fontsize=theme.annotation_size,
        color=theme.palette["success"],
        fontweight="bold",
    )


def _panel_safety(ax, data: dict | None, theme: PlotTheme) -> None:
    """Headline panel: false-accept rate vs k on log y-axis."""
    if data is None or not data.get("ks"):
        _annotate_missing(ax, "Run R2 to populate this panel", theme)
        return

    ks = np.array(data["ks"])
    emp = np.array(data["empirical"], dtype=float)
    emp_ci = data.get("empirical_ci")
    theory = data.get("theory_curve")
    baseline = data.get("baseline_no_cert")

    # 1. CI ribbon (drawn first so lines render on top)
    if emp_ci:
        lo = np.array([c[0] for c in emp_ci], dtype=float)
        hi = np.array([c[1] for c in emp_ci], dtype=float)
        lo = np.where(lo > 0, lo, 1e-7)
        hi = np.where(hi > 0, hi, 1e-7)
        ax.fill_between(
            ks, lo, hi,
            color=theme.palette["ours_light"],
            alpha=0.55, zorder=2, label="95% CI",
        )

    # 2. Empirical line (the headline)
    emp_safe = np.where(emp > 0, emp, 1e-7)
    ax.plot(
        ks, emp_safe, "-o",
        color=theme.palette["ours"], lw=2.5, markersize=8,
        markeredgecolor="white", markeredgewidth=1.8,
        label="PCG-MAS (ours)", zorder=5,
    )

    # 3. Theory bound
    if theory is not None and len(theory) == len(ks):
        theory_arr = np.array(theory, dtype=float)
        theory_safe = np.where(theory_arr > 0, theory_arr, 1e-7)
        ax.plot(
            ks, theory_safe, "--",
            color=theme.palette["ours_dark"], lw=1.6,
            alpha=0.75, label="Thm 2 bound", zorder=4,
        )

    # 4. No-cert baseline + reduction annotation
    if baseline is not None and baseline > 0:
        ax.axhline(
            baseline, linestyle=":", color=theme.palette["base_strong"],
            lw=2, label="No certificate", zorder=3,
        )
        if len(emp_safe) > 0:
            target = float(emp_safe[-1])
            if target > 0:
                reduction = baseline / target
                if reduction > 1.5:
                    mid_y = float(np.sqrt(baseline * target))
                    ax.annotate(
                        f"{reduction:.0f}× fewer\nfalse accepts",
                        xy=(ks[-1] - 0.2, target * 1.4),
                        xytext=(ks[max(0, len(ks) // 2 - 1)], mid_y * 0.45),
                        fontsize=theme.annotation_size,
                        color=theme.palette["ours_dark"],
                        fontweight="bold", ha="center",
                        arrowprops=dict(
                            arrowstyle="->",
                            color=theme.palette["ours_dark"],
                            lw=1.2,
                            connectionstyle="arc3,rad=-0.2",
                        ),
                        zorder=6,
                    )

    ax.set_yscale("log")
    ax.set_xlabel("Redundancy  $k$", fontsize=theme.label_size)
    ax.set_ylabel(
        r"$\Pr(\mathrm{accept} \cap \mathrm{wrong})$",
        fontsize=theme.label_size,
    )
    _panel_heading(
        ax, theme,
        title="Safety",
        subtitle=r"false-accept rate collapses with $k$",
    )
    ax.set_xticks(ks)
    ax.legend(
        loc="lower left", frameon=False, fontsize=theme.tick_size,
    )
    ax.grid(True, which="both", alpha=theme.grid_alpha, lw=theme.grid_lw)


def _panel_audit(ax, data: dict | None, theme: PlotTheme) -> None:
    """Theorem 1 four-channel decomposition with bound visualization."""
    if data is None:
        _annotate_missing(ax, "Run R1 to populate this panel", theme)
        return

    channels = data.get("channels")
    lhs = data.get("lhs")
    if not channels or lhs is None:
        _annotate_missing(ax, "Run R1 to populate this panel", theme)
        return

    keys = ["int_fail", "replay_fail", "check_fail", "cov_gap"]
    names = ["IntFail", "ReplayFail", "CheckFail", "CovGap"]
    color_keys = ["ch_int", "ch_replay", "ch_check", "ch_cov"]

    means = [channels.get(k, (0.0, (0.0, 0.0)))[0] for k in keys]
    cis = [channels.get(k, (0.0, (0.0, 0.0)))[1] for k in keys]
    err_lo = [max(0.0, m - c[0]) for m, c in zip(means, cis)]
    err_hi = [max(0.0, c[1] - m) for m, c in zip(means, cis)]

    y_pos = np.arange(len(names))
    bar_colors = [theme.palette[ck] for ck in color_keys]

    ax.barh(
        y_pos, means, xerr=[err_lo, err_hi],
        color=bar_colors, edgecolor="white", lw=1.5,
        error_kw=dict(
            ecolor=theme.palette["ink_light"],
            capsize=3, lw=1, alpha=0.7,
        ),
        zorder=3,
    )

    rhs_sum = sum(means)
    lhs_mean, _lhs_ci = lhs

    # LHS line — bold red, the thing being bounded
    ax.axvline(
        lhs_mean, color=theme.palette["danger"], lw=2.5,
        linestyle="-", zorder=6,
        label=fr"LHS = {lhs_mean:.3f}",
    )
    # RHS sum line — navy, the bound
    ax.axvline(
        rhs_sum, color=theme.palette["base_strong"], lw=2,
        linestyle="--", zorder=5,
        label=fr"RHS = {rhs_sum:.3f}",
    )

    # Highlight the slack (RHS - LHS)
    if rhs_sum > lhs_mean:
        ax.axvspan(
            lhs_mean, rhs_sum,
            color=theme.palette["bg_emphasis"], alpha=0.7, zorder=1,
        )
        # "Bound holds ✓" badge ABOVE the topmost bar (y-axis is inverted,
        # so y=-0.55 sits above bar 0). Keeps it clear of the legend below.
        ax.text(
            (lhs_mean + rhs_sum) / 2, -0.55,
            "Theorem 1 bound holds ✓",
            ha="center", va="center",
            fontsize=theme.annotation_size - 1,
            color=theme.palette["success"],
            fontweight="bold", zorder=10,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor=_lighten(theme.palette["success"], 0.85),
                edgecolor=theme.palette["success"],
                linewidth=0.8,
            ),
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=theme.tick_size)
    ax.set_xlabel("Probability", fontsize=theme.label_size)
    _panel_heading(
        ax, theme,
        title="Audit",
        subtitle=r"Thm 1 union bound is tight",
    )
    ax.legend(
        loc="lower right", frameon=False,
        fontsize=theme.tick_size - 1,
    )
    ax.grid(True, axis="x", alpha=theme.grid_alpha, lw=theme.grid_lw)
    ax.invert_yaxis()


def _panel_tradeoff(ax, data: dict | None, theme: PlotTheme) -> None:
    """Cost vs harm Pareto frontier across policies and privacy budgets."""
    if data is None or not data.get("policies"):
        _annotate_missing(ax, "Run R4 to populate this panel", theme)
        return

    policies = data["policies"]

    # Policy → (label, color, marker, size). Star marker for ours.
    style_map = {
        "always_answer": ("Always answer",     theme.palette["base_weak"],   "s", 100),
        "threshold_pcg": ("PCG-MAS (ours)",    theme.palette["ours"],        "*", 320),
        "learned":       ("Learned policy",    theme.palette["base_strong"], "D", 110),
    }

    plotted_anything = False
    for pname, pdata in policies.items():
        costs = pdata.get("costs", []) or []
        harms = pdata.get("harms", []) or []
        if not costs or not harms:
            continue
        plotted_anything = True
        label, color, marker, size = style_map.get(
            pname, (pname, theme.palette["neutral"], "o", 80)
        )
        # Subtle line through privacy-level sweep
        if len(costs) > 1:
            order = np.argsort(costs)
            cs = np.array(costs)[order]
            hs = np.array(harms)[order]
            ax.plot(cs, hs, "-", color=color, alpha=0.35, lw=1.2, zorder=1)
        ax.scatter(
            costs, harms, s=size, marker=marker,
            color=color, edgecolor="white", linewidth=1.5,
            label=label, zorder=4,
        )

    if not plotted_anything:
        _annotate_missing(ax, "Run R4 to populate this panel", theme)
        return

    # Pareto direction arrow
    ax.annotate(
        "", xy=(0.04, 0.04), xytext=(0.22, 0.22),
        xycoords="axes fraction",
        arrowprops=dict(
            arrowstyle="->",
            color=theme.palette["success"], lw=2.2,
        ),
        zorder=10,
    )
    ax.text(
        0.24, 0.24, "Pareto-optimal",
        transform=ax.transAxes,
        fontsize=theme.annotation_size,
        color=theme.palette["success"],
        fontweight="bold",
    )

    ax.set_xlabel("Cost per claim", fontsize=theme.label_size)
    ax.set_ylabel("Harm per claim", fontsize=theme.label_size)
    _panel_heading(
        ax, theme,
        title="Tradeoff",
        subtitle=r"PCG dominates across $\varepsilon$",
    )
    ax.legend(
        loc="upper right", frameon=False,
        fontsize=theme.tick_size - 1,
    )
    ax.grid(True, alpha=theme.grid_alpha, lw=theme.grid_lw)


# =====================================================================
# PANELS USED ONLY BY THE SUMMARY BENCHMARK
# =====================================================================


def _panel_backend_compare(
    ax, data: dict | None, theme: PlotTheme,
) -> None:
    """Generic per-backend metric bar chart with optional CIs."""
    if not data or not data.get("backends"):
        _annotate_missing(ax, "No backend comparison data", theme)
        return

    backends = data["backends"]
    metric = data.get("metric_name", "Metric")
    values = data.get("values", [])
    cis = data.get("cis")

    color_map = data.get("colors", {}) or {}
    default_color = theme.palette["base_weak"]
    bar_colors = [color_map.get(b, default_color) for b in backends]

    err = None
    if cis and len(cis) == len(values):
        err_lo = [max(0.0, float(v) - float(c[0])) for v, c in zip(values, cis)]
        err_hi = [max(0.0, float(c[1]) - float(v)) for v, c in zip(values, cis)]
        err = [err_lo, err_hi]

    x_pos = np.arange(len(backends))
    ax.bar(
        x_pos, values, yerr=err,
        color=bar_colors, edgecolor="white", lw=1.5,
        error_kw=dict(
            ecolor=theme.palette["ink_light"],
            capsize=3, lw=1, alpha=0.7,
        ),
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        backends, rotation=22, ha="right",
        fontsize=theme.tick_size,
    )
    ax.set_ylabel(metric, fontsize=theme.label_size)
    _panel_heading(
        ax, theme,
        title=data.get("title", "Per-backend comparison"),
        subtitle=None,
    )
    ax.grid(True, axis="y", alpha=theme.grid_alpha, lw=theme.grid_lw)


def _panel_overhead_compare(
    ax, data: dict | None, theme: PlotTheme,
) -> None:
    """Per-backend stacked overhead breakdown (tokens by phase)."""
    if not data or not data.get("backends"):
        _annotate_missing(ax, "Run R5 to populate this panel", theme)
        return

    backends = data["backends"]
    phase_data = data.get("phases", {}) or {}
    if not phase_data:
        _annotate_missing(ax, "No per-phase breakdown available", theme)
        return

    x_pos = np.arange(len(backends))
    bottom = np.zeros(len(backends), dtype=float)

    phase_palette = [
        theme.palette["ch_int"],
        theme.palette["ch_replay"],
        theme.palette["ch_check"],
        theme.palette["ch_cov"],
        theme.palette["base_weak"],
        theme.palette["ours"],
    ]

    for i, (ph_name, vals) in enumerate(phase_data.items()):
        vals_arr = np.asarray(vals, dtype=float)
        if len(vals_arr) != len(backends):
            continue
        ax.bar(
            x_pos, vals_arr, bottom=bottom,
            color=phase_palette[i % len(phase_palette)],
            label=ph_name, edgecolor="white", lw=1,
        )
        bottom = bottom + vals_arr

    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        backends, rotation=22, ha="right",
        fontsize=theme.tick_size,
    )
    ax.set_ylabel("Tokens / claim", fontsize=theme.label_size)
    _panel_heading(
        ax, theme,
        title="Overhead breakdown by phase",
        subtitle=None,
    )
    ax.legend(
        loc="upper left", frameon=False,
        fontsize=theme.tick_size - 1, ncol=2,
    )
    ax.grid(True, axis="y", alpha=theme.grid_alpha, lw=theme.grid_lw)


def _panel_headline_numbers(
    ax, kpis: list | None, theme: PlotTheme,
) -> None:
    """Big-number KPI tiles. The 'money shot' for skim-reading the figure.

    Each KPI is a dict with:
        value: str  — the big number (e.g. "80×", "92%", "1.3×")
        label: str  — short bold caption (e.g. "fewer false accepts")
        sub:   str  — italic subtitle giving context (optional)

    Renders 2-4 stacked KPIs vertically with a thin divider between rows.
    Falls back to a friendly placeholder if no KPIs computable.
    """
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    if not kpis:
        _annotate_missing(ax, "No headline KPIs yet", theme)
        return

    # Cap to 4 — beyond that the panel gets crowded
    kpis = list(kpis)[:4]
    n = len(kpis)
    pad_top, pad_bot = 0.16, 0.06   # leave room for the heading
    avail = 1.0 - pad_top - pad_bot
    cell_h = avail / n

    for i, kpi in enumerate(kpis):
        y_top = 1.0 - pad_top - i * cell_h
        y_center = y_top - cell_h / 2

        value = str(kpi.get("value", ""))
        label = str(kpi.get("label", ""))
        sub = str(kpi.get("sub", ""))

        # Big bold value (left half)
        ax.text(
            0.36, y_center, value,
            transform=ax.transAxes,
            ha="right", va="center",
            fontsize=theme.title_size + 11,
            fontweight="bold",
            color=theme.palette["ours"],
        )
        # Label and sub (right half, stacked)
        ax.text(
            0.42, y_center + 0.018 * (1 / max(n, 1)) + 0.025, label,
            transform=ax.transAxes,
            ha="left", va="center",
            fontsize=theme.label_size,
            fontweight="bold",
            color=theme.palette["ink"],
        )
        if sub:
            ax.text(
                0.42, y_center - 0.025, sub,
                transform=ax.transAxes,
                ha="left", va="center",
                fontsize=theme.annotation_size - 1,
                color=theme.palette["ink_light"],
                style="italic",
            )
        # Divider between rows (not above the first row)
        if i > 0:
            ax.plot(
                [0.06, 0.94], [y_top, y_top],
                transform=ax.transAxes,
                color=theme.palette["neutral"],
                lw=0.6, alpha=0.35,
                zorder=1,
            )

    _panel_heading(
        ax, theme,
        title="Headline numbers",
        subtitle="key results at a glance",
    )


# =====================================================================
# COMPOSITION: HERO FIGURE
# =====================================================================


def plot_intro_hero_v2(
    *,
    safety: dict | None = None,
    audit: dict | None = None,
    tradeoff: dict | None = None,
    theme: PlotTheme = BOLD_THEME,
    source_runs: list[str] | None = None,
    is_mock: bool = False,
) -> plt.Figure:
    """The page-1 hero: schematic banner + Safety / Audit / Tradeoff panels.

    Layout (9.5 × 5.8 inches, designed to span full text width in NeurIPS):

        ┌──────────────────────────────────────────────────────────┐
        │           certificate Z anatomy schematic                │
        ├──────────────┬─────────────────┬─────────────────────────┤
        │   Safety     │     Audit       │      Tradeoff           │
        │   (R2)       │     (R1)        │      (R4)               │
        └──────────────┴─────────────────┴─────────────────────────┘

    Each quantitative panel takes its data dict; missing/None renders as
    a friendly "Run experiment Rn to populate" placeholder.

    Args:
        safety:   {ks, empirical, empirical_ci, theory_curve, baseline_no_cert}
        audit:    {channels: {int_fail, replay_fail, check_fail, cov_gap}, lhs}
                  where each value is (mean, (ci_lo, ci_hi))
        tradeoff: {policies: {<name>: {costs: [...], harms: [...]}}}
        theme:    PlotTheme — default is bold
        source_runs: list of run-id names for the provenance footer
        is_mock:  if True, add MOCK BACKEND watermark

    Returns:
        matplotlib Figure (not yet saved).
    """
    setup_matplotlib_for_theme(theme)

    fig = plt.figure(
        figsize=(9.5, 5.8), dpi=theme.dpi,
        facecolor=theme.palette["bg_panel"],
    )

    gs = GridSpec(
        2, 3, figure=fig,
        height_ratios=[1.0, 1.7],
        hspace=0.70, wspace=0.40,
        left=0.06, right=0.98,
        top=0.96, bottom=0.10,
    )
    ax_schematic = fig.add_subplot(gs[0, :])
    ax_safety    = fig.add_subplot(gs[1, 0])
    ax_audit     = fig.add_subplot(gs[1, 1])
    ax_tradeoff  = fig.add_subplot(gs[1, 2])

    _panel_schematic(ax_schematic, theme)
    _panel_safety(ax_safety, safety, theme)
    _panel_audit(ax_audit, audit, theme)
    _panel_tradeoff(ax_tradeoff, tradeoff, theme)

    _add_provenance_footer(
        fig, theme,
        source_runs=source_runs, is_mock=is_mock,
    )
    return fig


# =====================================================================
# COMPOSITION: SUMMARY BENCHMARK (appendix / website)
# =====================================================================


def plot_summary_benchmark(
    *,
    backend_safety: dict | None = None,
    redundancy: dict | None = None,
    backend_responsibility: dict | None = None,
    tradeoff: dict | None = None,
    backend_overhead: dict | None = None,
    headline_numbers: list | None = None,
    theme: PlotTheme = BOLD_THEME,
    source_runs: list[str] | None = None,
    is_mock: bool = False,
) -> plt.Figure:
    """The 2x3 comprehensive comparison panel covering ALL five experiments.

    Suitable for the appendix or project website where space allows the
    full picture across multiple backends (Qwen, DeepSeek, Phi, Llama, etc.)
    and all five R-experiments.

        ┌──────────────┬──────────────┬──────────────────┐
        │ R1 Safety    │ R2 Redundancy│ R3 Responsibility│
        │ per backend  │ collapse     │ per regime       │
        ├──────────────┼──────────────┼──────────────────┤
        │ R4 Cost-Harm │ R5 Overhead  │ Headline Numbers │
        │ Pareto       │ breakdown    │ (KPI tiles)      │
        └──────────────┴──────────────┴──────────────────┘

    Each panel takes its data dict; missing/None renders as a friendly
    "Run experiment Rn to populate" placeholder.

    Args:
        backend_safety: dict for R1 per-backend bars (see _panel_backend_compare)
        redundancy: dict for R2 redundancy curve (see _panel_safety)
        backend_responsibility: dict for R3 per-regime bars
        tradeoff: dict for R4 Pareto scatter (see _panel_tradeoff)
        backend_overhead: dict for R5 stacked tokens/claim
        headline_numbers: list of {value, label, sub} dicts for KPI tiles
        theme, source_runs, is_mock: same as hero
    """
    setup_matplotlib_for_theme(theme)

    fig = plt.figure(
        figsize=(13, 7.8), dpi=theme.dpi,
        facecolor=theme.palette["bg_panel"],
    )
    gs = GridSpec(
        2, 3, figure=fig,
        hspace=0.70, wspace=0.32,
        left=0.05, right=0.98,
        top=0.94, bottom=0.10,
    )
    ax_r1 = fig.add_subplot(gs[0, 0])
    ax_r2 = fig.add_subplot(gs[0, 1])
    ax_r3 = fig.add_subplot(gs[0, 2])
    ax_r4 = fig.add_subplot(gs[1, 0])
    ax_r5 = fig.add_subplot(gs[1, 1])
    ax_kpi = fig.add_subplot(gs[1, 2])

    _panel_backend_compare(ax_r1, backend_safety, theme)
    _panel_safety(ax_r2, redundancy, theme)
    _panel_backend_compare(ax_r3, backend_responsibility, theme)
    _panel_tradeoff(ax_r4, tradeoff, theme)
    _panel_overhead_compare(ax_r5, backend_overhead, theme)
    _panel_headline_numbers(ax_kpi, headline_numbers, theme)

    _add_provenance_footer(
        fig, theme,
        source_runs=source_runs, is_mock=is_mock,
    )
    return fig


# =====================================================================
# SAVE
# =====================================================================


def save_fig_v2(
    fig: plt.Figure, path: str | Path,
    formats: tuple[str, ...] = ("pdf", "png"),
    dpi: int | None = None,
) -> list[str]:
    """Save fig in multiple formats. Returns list of written paths."""
    written: list[str] = []
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        out = (
            str(p) if str(p).endswith(f".{fmt}")
            else f"{p}.{fmt}"
        )
        save_kwargs = {"bbox_inches": "tight"}
        if dpi is not None:
            save_kwargs["dpi"] = dpi
        fig.savefig(out, **save_kwargs)
        written.append(out)
    return written


__all__ = [
    "PlotTheme", "BOLD_THEME",
    "setup_matplotlib_for_theme", "detect_mock_runs",
    "plot_intro_hero_v2", "plot_summary_benchmark",
    "save_fig_v2",
]
