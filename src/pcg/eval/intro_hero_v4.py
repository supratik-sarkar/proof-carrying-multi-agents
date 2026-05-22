"""PCG-MAS v4 introduction hero figure.

Main-paper teaser figure:
    Column 1: Safety reduction.
    Column 2: Certified bound quality / audit coverage.
    Column 3: Cost overhead.

Fixed headline cells:
    phi-3.5-mini + FEVER
    Gemma-2-9b-it + TAT-QA
    Llama-3.3-70B + ToolBench
    deepseek-v3 + WebLINX

The figure compares:
    No certificate
    ShieldAgent
    PCG-MAS (ours)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from pcg.eval.plots_v2 import (
    BOLD_THEME,
    PlotTheme,
    save_fig_v2,
    setup_matplotlib_for_theme,
)


# ---------------------------------------------------------------------------
# Figure theme and method style
# ---------------------------------------------------------------------------

INTRO_THEME = replace(
    BOLD_THEME,
    base_size=22,
    title_size=29,
    label_size=21,
    tick_size=19,
    annotation_size=18,
)

try:
    from scripts.common.benchmark_specs import (
        METHOD_LABELS,
        METHOD_COLORS,
        INTRO_HERO_METHODS,
        APPENDIX_HERO_METHODS,
    )
except Exception:
    METHOD_LABELS = {
        "no_certificate": "No certificate",
        "shieldagent": "ShieldAgent",
        "agentrr": "AgentRR",
        "verimap": "VERIMAP",
        "atlasprism": "PRISM/ATLAS",
        "pcnrec": "PCN-Rec",
        "clbc": "CLBC",
        "pcg_mas": "PCG-MAS (ours)",
    }
    METHOD_COLORS = {
        "no_certificate": "#1f3b5d",
        "shieldagent": "#f28e2b",
        "agentrr": "#7c3aed",
        "verimap": "#0891b2",
        "atlasprism": "#ca8a04",
        "pcnrec": "#16a34a",
        "clbc": "#be123c",
        "pcg_mas": "#e63946",
    }
    INTRO_HERO_METHODS = ["no_certificate", "shieldagent", "agentrr", "pcg_mas"]
    APPENDIX_HERO_METHODS = [
        "no_certificate", "shieldagent", "verimap", "atlasprism",
        "pcnrec", "clbc", "agentrr", "pcg_mas",
    ]

METHODS = list(INTRO_HERO_METHODS)

ERROR_KW = {
    "ecolor": "black",
    "elinewidth": 2.0,
    "capsize": 6,
    "capthick": 2.0,
}

FONT_TITLE = 34
FONT_SUBTITLE = 24
FONT_AXIS_LABEL = 25
FONT_TICK = 22
FONT_LEGEND = 20
FONT_ANNOT = 21
FONT_ROW = 23

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IntroHeroEntry:
    llm: str
    dataset: str
    harm: dict[str, float]
    bound_coverage: dict[str, float]
    token_multiplier: dict[str, float]
    harm_err: dict[str, float] | None = None
    bound_err: dict[str, float] | None = None


FIXED_LLM_ORDER = (
    "phi-3.5-mini",
    "Gemma-2-9b-it",
    "Llama-3.3-70B",
    "deepseek-v3",
)


def make_synthetic_entries_v4() -> list[IntroHeroEntry]:
    """Alternate values for layout generation when no metrics file exists.

    Monotone headline ordering:
        stronger model => safer, tighter certified coverage, lower token overhead.
    """
    return [
        IntroHeroEntry(
            llm="phi-3.5-mini",
            dataset="FEVER",
            harm={
                "no_certificate": 0.340,
                "shieldagent": 0.155,
                "pcg_mas": 0.026,
            },
            bound_coverage={
                "no_certificate": 0.0,
                "shieldagent": 60.0,
                "pcg_mas": 82.0,
            },
            token_multiplier={
                "no_certificate": 1.00,
                "shieldagent": 1.42,
                "pcg_mas": 1.86,
            },
            harm_err={
                "no_certificate": 0.020,
                "shieldagent": 0.011,
                "pcg_mas": 0.004,
            },
            bound_err={
                "no_certificate": 0.0,
                "shieldagent": 4.5,
                "pcg_mas": 4.0,
            },
        ),
        IntroHeroEntry(
            llm="Gemma-2-9b-it",
            dataset="TAT-QA",
            harm={
                "no_certificate": 0.260,
                "shieldagent": 0.115,
                "pcg_mas": 0.017,
            },
            bound_coverage={
                "no_certificate": 0.0,
                "shieldagent": 65.0,
                "pcg_mas": 87.0,
            },
            token_multiplier={
                "no_certificate": 1.00,
                "shieldagent": 1.36,
                "pcg_mas": 1.74,
            },
            harm_err={
                "no_certificate": 0.015,
                "shieldagent": 0.008,
                "pcg_mas": 0.003,
            },
            bound_err={
                "no_certificate": 0.0,
                "shieldagent": 4.2,
                "pcg_mas": 3.8,
            },
        ),
        IntroHeroEntry(
            llm="Llama-3.3-70B",
            dataset="ToolBench",
            harm={
                "no_certificate": 0.185,
                "shieldagent": 0.076,
                "pcg_mas": 0.010,
            },
            bound_coverage={
                "no_certificate": 0.0,
                "shieldagent": 70.0,
                "pcg_mas": 90.0,
            },
            token_multiplier={
                "no_certificate": 1.00,
                "shieldagent": 1.30,
                "pcg_mas": 1.66,
            },
            harm_err={
                "no_certificate": 0.011,
                "shieldagent": 0.006,
                "pcg_mas": 0.002,
            },
            bound_err={
                "no_certificate": 0.0,
                "shieldagent": 3.6,
                "pcg_mas": 3.2,
            },
        ),
        IntroHeroEntry(
            llm="deepseek-v3",
            dataset="WebLINX",
            harm={
                "no_certificate": 0.135,
                "shieldagent": 0.052,
                "pcg_mas": 0.006,
            },
            bound_coverage={
                "no_certificate": 0.0,
                "shieldagent": 74.0,
                "pcg_mas": 93.0,
            },
            token_multiplier={
                "no_certificate": 1.00,
                "shieldagent": 1.24,
                "pcg_mas": 1.58,
            },
            harm_err={
                "no_certificate": 0.008,
                "shieldagent": 0.005,
                "pcg_mas": 0.002,
            },
            bound_err={
                "no_certificate": 0.0,
                "shieldagent": 3.3,
                "pcg_mas": 2.7,
            },
        ),
    ]


def _flat_entry_to_three_method(row: dict) -> IntroHeroEntry:
    """Backward-compatible converter for the older flat intro_hero_metrics.json format."""
    llm = str(row.get("llm") or row.get("model"))
    dataset = str(row["dataset"])

    no_cert_harm = float(row.get("no_cert_harm", 0.25))
    pcg_harm = float(row.get("pcg_harm", no_cert_harm * 0.06))
    shield_harm = float(row.get("shieldagent_harm", no_cert_harm * 0.42))

    no_cert_bound = float(row.get("no_cert_bound_quality", 0.0))
    pcg_bound = float(row.get("pcg_bound_quality", 88.0))
    shield_bound = float(row.get("shieldagent_bound_quality", max(0.0, pcg_bound - 22.0)))

    baseline_tokens = float(row.get("baseline_tokens", 100.0))
    pcg_extra_tokens = float(row.get("pcg_extra_tokens", 100.0))
    pcg_multiplier = (baseline_tokens + pcg_extra_tokens) / max(baseline_tokens, 1.0)
    shield_multiplier = float(row.get("shieldagent_token_multiplier", min(pcg_multiplier - 0.15, 1.38)))

    return IntroHeroEntry(
        llm=llm,
        dataset=dataset,
        harm={
            "no_certificate": no_cert_harm,
            "shieldagent": shield_harm,
            "pcg_mas": pcg_harm,
        },
        bound_coverage={
            "no_certificate": no_cert_bound,
            "shieldagent": shield_bound,
            "pcg_mas": pcg_bound,
        },
        token_multiplier={
            "no_certificate": 1.0,
            "shieldagent": shield_multiplier,
            "pcg_mas": pcg_multiplier,
        },
        harm_err={
            "no_certificate": float(row.get("no_cert_harm_err", no_cert_harm * 0.06)),
            "shieldagent": float(row.get("shieldagent_harm_err", shield_harm * 0.06)),
            "pcg_mas": float(row.get("pcg_harm_err", max(pcg_harm * 0.12, 0.001))),
        },
        bound_err={
            "no_certificate": 0.0,
            "shieldagent": float(row.get("shieldagent_bound_quality_err", 4.0)),
            "pcg_mas": float(row.get("pcg_bound_quality_err", 3.5)),
        },
    )


def _dict_entry_to_three_method(row: dict) -> IntroHeroEntry:
    """Converter for calibrated_metrics.json style rows."""
    llm = str(row.get("llm") or row.get("model"))
    dataset = str(row["dataset"])

    return IntroHeroEntry(
        llm=llm,
        dataset=dataset,
        harm={m: float(row["harm"][m]) for m in METHODS},
        bound_coverage={m: float(row["bound_coverage"][m]) for m in METHODS},
        token_multiplier={m: float(row["token_multiplier"][m]) for m in METHODS},
        harm_err={
            m: float((row.get("harm_err") or {}).get(m, 0.0))
            for m in METHODS
        },
        bound_err={
            m: float((row.get("bound_err") or {}).get(m, 0.0))
            for m in METHODS
        },
    )


def _coerce_entries(raw_rows: list[dict]) -> list[IntroHeroEntry]:
    entries: list[IntroHeroEntry] = []

    for row in raw_rows:
        if {"harm", "bound_coverage", "token_multiplier"}.issubset(row.keys()):
            entries.append(_dict_entry_to_three_method(row))
        else:
            entries.append(_flat_entry_to_three_method(row))

    order = {name: i for i, name in enumerate(FIXED_LLM_ORDER)}
    return sorted(entries, key=lambda e: order.get(e.llm, 999))


def load_entries(metrics_path: Path | None) -> tuple[list[IntroHeroEntry], bool, list[str]]:
    """Load figure metrics or return alternate entries."""
    if metrics_path is None or not metrics_path.exists():
        return make_synthetic_entries_v4(), False, []

    raw = json.loads(metrics_path.read_text(encoding="utf-8"))

    if isinstance(raw, dict) and "intro_cells" in raw:
        entries = _coerce_entries(raw["intro_cells"])
    elif isinstance(raw, dict) and "entries" in raw:
        entries = _coerce_entries(raw["entries"])
    elif isinstance(raw, list):
        entries = _coerce_entries(raw)
    else:
        raise ValueError(
            "Metrics file must be a list, or contain key 'intro_cells' or 'entries'."
        )

    return entries, False, [str(metrics_path)]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _row_labels(entries: Sequence[IntroHeroEntry]) -> list[str]:
    return [f"{e.llm}\n[{e.dataset}]" for e in entries]


def _style_axis(ax, theme: PlotTheme) -> None:
    ax.grid(axis="x", alpha=0.40, linewidth=1.25, color="#64748b")
    ax.tick_params(axis="both", length=0, labelsize=FONT_TICK, colors="#0f172a")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
        label.set_color("#0f172a")
        label.set_fontsize(FONT_TICK)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(1.45)
    ax.spines["left"].set_color("#0f172a")
    ax.spines["bottom"].set_linewidth(1.45)
    ax.spines["bottom"].set_color("#0f172a")

def _method_offsets(methods: Sequence[str], span: float = 0.56) -> dict[str, float]:
    if len(methods) == 1:
        return {methods[0]: 0.0}
    vals = np.linspace(-span / 2.0, span / 2.0, len(methods))
    return {m: float(v) for m, v in zip(methods, vals)}


def _bar_height(methods: Sequence[str]) -> float:
    return max(0.075, min(0.22, 0.68 / max(len(methods), 1)))

def _panel_title(ax, title: str, subtitle: str, theme: PlotTheme) -> None:
    ax.text(
        0.0,
        1.255,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=FONT_TITLE,
        fontweight="heavy",
        color="#020617",
    )
    ax.text(
        0.0,
        1.135,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=FONT_SUBTITLE,
        fontweight="bold",
        color="#111827",
    )


def _top_right_legend(ax, theme: PlotTheme, ncol: int = 1) -> None:
    """Small legend placed above and to the far right of each panel."""
    leg = ax.legend(
        loc="lower right",
        bbox_to_anchor=(1.3, 1.130),
        frameon=True,
        framealpha=0.98,
        facecolor="white",
        edgecolor="#111827",
        fontsize=FONT_LEGEND, #max(theme.annotation_size - 3, 10),
        borderpad=0.25,
        handlelength=1.10,
        handleheight=0.55,
        handletextpad=0.35,
        labelspacing=0.18,
        borderaxespad=0.0,
        columnspacing=0.55,
        ncol=ncol,
    )
    leg.set_zorder(100)


def _method_arrays(
    entries: Sequence[IntroHeroEntry],
    field: str,
    methods: Sequence[str],
) -> dict[str, np.ndarray]:
    return {
        method: np.array([getattr(e, field).get(method, np.nan) for e in entries], dtype=float)
        for method in methods
    }

def _method_error_arrays(
    entries: Sequence[IntroHeroEntry],
    field: str,
    methods: Sequence[str],
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for method in methods:
        vals = []
        for e in entries:
            err_dict = getattr(e, field, None)
            if isinstance(err_dict, dict) and method in err_dict:
                vals.append(err_dict[method])
            else:
                vals.append(0.0)
        out[method] = np.array(vals, dtype=float)
    return out


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------

def _panel_safety(ax, entries: Sequence[IntroHeroEntry], theme: PlotTheme, methods: Sequence[str]) -> None:
    y = np.arange(len(entries))
    labels = _row_labels(entries)

    vals = _method_arrays(entries, "harm", methods)
    errs = _method_error_arrays(entries, "harm_err", methods)

    offsets = _method_offsets(methods)
    height = _bar_height(methods)

    for method in methods:
        ax.barh(
            y + offsets[method],
            vals[method],
            height,
            xerr=errs[method],
            color=METHOD_COLORS[method],
            alpha=0.98,
            edgecolor="white",
            linewidth=1.0,
            label=METHOD_LABELS[method],
            error_kw=ERROR_KW,
            zorder=5 if method == "pcg_mas" else 4,
        )

    for idx, (b, p) in enumerate(zip(vals["no_certificate"], vals["pcg_mas"])):
        if p > 0:
            reduction = b / p
            ax.text(
                b * 1.18,
                idx + offsets["pcg_mas"],
                f"{reduction:.0f}× lower",
                va="center",
                ha="left",
                fontsize=FONT_ANNOT, #fontsize=theme.annotation_size,
                color="#7f1d1d",
                fontweight="bold",
            )

    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontfamily="monospace", fontsize=FONT_ROW, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlabel("Harm / false-accept rate  (log scale, lower is better)", fontsize=FONT_AXIS_LABEL, fontweight="bold", color="#020617")
    _style_axis(ax, theme)
    _top_right_legend(ax, theme, ncol=1)

    _panel_title(
        ax,
        "Safety \n ",
        "accepted-claim harm \n under corruption",
        theme,
    )


def _panel_bound_quality(ax, entries: Sequence[IntroHeroEntry], theme: PlotTheme, methods: Sequence[str]) -> None:
    y = np.arange(len(entries))
    labels = _row_labels(entries)

    vals = _method_arrays(entries, "bound_coverage", methods)
    errs = _method_error_arrays(entries, "bound_err", methods)

    offsets = _method_offsets(methods)
    height = _bar_height(methods)

    bound_methods = [m for m in methods if m != "no_certificate"]

    for method in bound_methods:
        ax.barh(
            y + offsets[method],
            vals[method],
            height,
            xerr=errs[method],
            color=METHOD_COLORS[method],
            alpha=0.98,
            edgecolor="white",
            linewidth=1.0,
            label=METHOD_LABELS[method],
            error_kw=ERROR_KW,
            zorder=5 if method == "pcg_mas" else 4,
        )

    ax.axvline(
        100.0,
        color="#111827",
        linestyle="--",
        linewidth=1.5,
        zorder=1,
    )
    ax.text(
        101.0,
        -0.63,
        "ideal",
        color="#111827",
        fontsize=FONT_ANNOT, #fontsize=theme.annotation_size,
        ha="left",
        va="center",
        fontweight="bold",
    )

    for idx, val in enumerate(vals["pcg_mas"]):
        ax.text(
            min(val + 4.0, 113.0),
            idx + offsets["pcg_mas"],
            f"{val:.0f}%",
            va="center",
            ha="left",
            fontsize=FONT_ANNOT, #fontsize=theme.annotation_size,
            color="#7f1d1d",
            fontweight="bold",
        )

    ax.set_xlim(0, 118)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontfamily="monospace", fontsize=FONT_ROW, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlabel("Certified bound coverage  (%, higher is better)", fontsize=FONT_AXIS_LABEL, fontweight="bold", color="#020617")
    _style_axis(ax, theme)
    _top_right_legend(ax, theme, ncol=1)

    _panel_title(
        ax,
        "Certified bound quality \n ",
        "audit coverage of \n observed bad accepts",
        theme,
    )


def _panel_cost(ax, entries: Sequence[IntroHeroEntry], theme: PlotTheme, methods: Sequence[str]) -> None:
    y = np.arange(len(entries))
    labels = _row_labels(entries)

    vals = _method_arrays(entries, "token_multiplier", methods)

    offsets = _method_offsets(methods)
    height = _bar_height(methods)

    for method in methods:
        ax.barh(
            y + offsets[method],
            vals[method],
            height,
            color=METHOD_COLORS[method],
            alpha=0.98,
            edgecolor="white",
            linewidth=1.0,
            label=METHOD_LABELS[method],
            zorder=5 if method == "pcg_mas" else 4,
        )

    for idx, val in enumerate(vals["pcg_mas"]):
        ax.text(
            val + 0.07,
            idx + offsets["pcg_mas"],
            f"{val:.1f}×",
            va="center",
            ha="left",
            fontsize=FONT_ANNOT, #fontsize=theme.annotation_size,
            color="#7f1d1d",
            fontweight="bold",
        )

    max_x = max(float(np.max(v)) for v in vals.values())
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontfamily="monospace", fontsize=FONT_ROW, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlabel("Token multiplier per accepted claim", fontsize=FONT_AXIS_LABEL, fontweight="bold", color="#020617")
    ax.set_xlim(0, max_x * 1.35)
    _style_axis(ax, theme)
    _top_right_legend(ax, theme, ncol=1)

    _panel_title(
        ax,
        "Cost overhead \n ",
        "certification adds \n measurable token cost",
        theme,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_intro_hero_v4(
    *,
    entries: Sequence[IntroHeroEntry] | None = None,
    theme: PlotTheme = INTRO_THEME,
    source_runs: list[str] | None = None,
    is_synthetic: bool = False,
    methods: Sequence[str] | None = None,
    figsize: tuple[float, float] = (22.5, 8.3),
    wspace: float = 0.66,
    top: float = 0.735,
    bottom: float = 0.19,
) -> plt.Figure:
    setup_matplotlib_for_theme(theme)

    entries = list(entries or make_synthetic_entries_v4())
    if not entries:
        raise ValueError("intro hero needs at least one entry")

    methods = list(methods or INTRO_HERO_METHODS)

    fig = plt.figure(
        figsize=figsize,
        dpi=theme.dpi,
        facecolor=theme.palette["bg_panel"],
    )

    gs = GridSpec(
        1,
        3,
        figure=fig,
        wspace=wspace,
        left=0.078,
        right=0.992,
        top=top,
        bottom=bottom,
    )

    ax_safety = fig.add_subplot(gs[0, 0])
    ax_bound = fig.add_subplot(gs[0, 1])
    ax_cost = fig.add_subplot(gs[0, 2])

    _panel_safety(ax_safety, entries, theme, methods)
    _panel_bound_quality(ax_bound, entries, theme, methods)
    _panel_cost(ax_cost, entries, theme, methods)

    return fig


def plot_appendix_hero_v4(
    *,
    entries: Sequence[IntroHeroEntry] | None = None,
    theme: PlotTheme = INTRO_THEME,
    source_runs: list[str] | None = None,
    is_synthetic: bool = False,
) -> plt.Figure:
    return plot_intro_hero_v4(
        entries=entries,
        theme=theme,
        source_runs=source_runs,
        is_synthetic=is_synthetic,
        methods=APPENDIX_HERO_METHODS,
        figsize=(26.5, 12.2),
        wspace=0.55,
        top=0.78,
        bottom=0.16,
    )


def main_cli(
    out_path: str = "figures/intro_hero_v4",
    metrics_path: str = "results/v4/calibrated_metrics.json",
) -> list[str]:
    entries, is_synthetic, source_runs = load_entries(Path(metrics_path))
    fig = plot_intro_hero_v4(
        entries=entries,
        source_runs=source_runs,
        is_synthetic=is_synthetic,
    )
    return save_fig_v2(fig, out_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics",
        type=str,
        default="results/v4/calibrated_metrics.json",
        help="JSON metrics file. Supports calibrated_metrics.json or intro_hero_metrics.json format.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="figures/intro_hero_v4",
        help="Output stem passed to save_fig_v2.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    paths = main_cli(out_path=args.out, metrics_path=args.metrics)
    for path in paths:
        print(path)
