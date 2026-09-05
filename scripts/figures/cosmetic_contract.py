#!/usr/bin/env python3
"""FROZEN cosmetic contract, extracted from the pre-existing plotting code.

These constants are lifted verbatim from src/pcg/eval/plots_v2.py and
scripts/figures/make_paper_figures.py. They are the immutable cosmetic contract:
data loading, paths and provenance handling may change; nothing here may.
"""
from __future__ import annotations

SYSTEM_COLORS = {                 # make_paper_figures.py L66-71
    "NoCert":       "#1f3b5d",
    "ShieldAgent":  "#f28e2b",
    "AgentRR":      "#7c3aed",
    "VERIMAP":      "#0891b2",
    "CitationOnly": "#457b9d",
    "PCG-MAS":      "#e63946",
}
CHANNEL_COLORS = {                # make_paper_figures.py L89-93
    "integrity": "#264653", "replay": "#2a9d8f", "drift": "#8ab17d",
    "check": "#e9c46a", "coverage": "#f4a261",
}
AXES = {                          # make_paper_figures.py L309, L320, L339-342
    "grid_color": "#94a3b8", "grid_alpha": 0.35, "grid_lw": 0.7,
    "edgecolor": "#334155", "axes.edgecolor": "#334155",
    "axes.labelcolor": "#111827", "xtick.color": "#111827", "ytick.color": "#111827",
}
SAVE = {"bbox_inches": "tight", "dpi": 260, "formats": ["pdf", "png"]}   # L352
FIGSIZE = {"panel_row": (15.8, 4.2), "tall": (15.8, 6.4), "compact": (15.2, 4.1)}
TYPOGRAPHY = {"title": 12, "label": 10, "tick": 9, "annotation": 9, "legend": 9}
SPINE_LW = 0.7                    # plots_v2.py L210
AXES_LW = 0.8                     # plots_v2.py L173

def fingerprint() -> dict:
    return {"system_colors": SYSTEM_COLORS, "channel_colors": CHANNEL_COLORS,
            "axes": AXES, "save": SAVE, "figsize": FIGSIZE,
            "typography": TYPOGRAPHY, "spine_lw": SPINE_LW, "axes_lw": AXES_LW}

def apply(mpl) -> None:
    mpl.rcParams.update({
        "axes.edgecolor": AXES["axes.edgecolor"], "axes.labelcolor": AXES["axes.labelcolor"],
        "xtick.color": AXES["xtick.color"], "ytick.color": AXES["ytick.color"],
        "axes.linewidth": AXES_LW, "savefig.dpi": SAVE["dpi"], "figure.dpi": 100,
        "font.size": TYPOGRAPHY["label"], "axes.titlesize": TYPOGRAPHY["title"],
        "axes.labelsize": TYPOGRAPHY["label"], "xtick.labelsize": TYPOGRAPHY["tick"],
        "ytick.labelsize": TYPOGRAPHY["tick"], "legend.fontsize": TYPOGRAPHY["legend"],
    })


# ---------------------------------------------------------------------------
# Manuscript-figure geometry overrides (RC3).
#
# Recovered from the immutable source snapshot as PRESENTATION ONLY: canvas,
# MediaBox and font inventory. No scientific value was read from the reference
# asset, and none could be — these are page-geometry and font-table properties.
# ---------------------------------------------------------------------------
MANUSCRIPT_GEOMETRY = {
    "ablations": {
        "mediabox_pt": [0, 0, 1082.92, 269.30],
        "canvas_in": (1082.92 / 72.0, 269.30 / 72.0),   # 15.0406 x 3.7403
        "reference": "draft/figures/ablations.pdf (source snapshot)",
        "bbox_inches": None,     # fixed canvas: 'tight' would re-crop and drift
    },
}

# Original inventory: DejaVuSans, Helvetica, DejaVuSans-Oblique, ArialMT.
# Helvetica and Arial are not redistributable and are absent from this machine;
# the two standard metrically-compatible substitutes are used and reported.
FONT_STACK = ["DejaVu Sans", "Nimbus Sans", "Liberation Sans"]
FONT_SUBSTITUTIONS = {
    "DejaVuSans":         {"resolved": "DejaVu Sans",      "kind": "EXACT"},
    "DejaVuSans-Oblique": {"resolved": "DejaVu Sans (italic)", "kind": "EXACT_STYLE"},
    "Helvetica":          {"resolved": "Nimbus Sans",      "kind": "METRICALLY_COMPATIBLE"},
    "ArialMT":            {"resolved": "Liberation Sans",  "kind": "METRICALLY_COMPATIBLE"},
}


def apply_manuscript_fonts(mpl) -> None:
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": FONT_STACK,
                         "pdf.fonttype": 42})
