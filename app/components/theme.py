"""Theme tokens shared across the demo. Mirrors the paper's BOLD_THEME.

Streamlit doesn't expose CSS variables natively, so we inject a small
stylesheet at app startup. The colors here are exactly the same as
`pcg.eval.plots_v2.BOLD_THEME.palette`, so figures rendered for the
paper and screens rendered in the demo look unmistakably the same."""
from __future__ import annotations

# Mirror of BOLD_THEME.palette in src/pcg/eval/plots_v2.py
PALETTE = {
    "bg_app": "#FFFCF7",        # warm cream
    "bg_panel": "#FFFFFF",      # paper white
    "ink": "#0E1116",           # near-black ink
    "ink_light": "#5C6370",     # muted secondary
    "neutral": "#9AA0A6",       # subtle grey
    "ours": "#E63946",          # signature PCG-MAS red
    "ours_light": "#FCE3E5",    # tinted background for our cells
    "baseline": "#46637A",      # muted blue-grey for baselines
    "accent_green": "#0B7A3B",  # for "verified ✓" affordances
    "accent_amber": "#C77700",  # for "warning"
}

CSS = f"""
<style>
:root {{
    --pcg-bg: {PALETTE['bg_app']};
    --pcg-panel: {PALETTE['bg_panel']};
    --pcg-ink: {PALETTE['ink']};
    --pcg-ink-light: {PALETTE['ink_light']};
    --pcg-ours: {PALETTE['ours']};
    --pcg-ours-light: {PALETTE['ours_light']};
    --pcg-baseline: {PALETTE['baseline']};
    --pcg-green: {PALETTE['accent_green']};
    --pcg-amber: {PALETTE['accent_amber']};
}}
.pcg-card {{
    background: var(--pcg-panel);
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}}
.pcg-card-our {{
    border-left: 4px solid var(--pcg-ours);
}}
.pcg-card-baseline {{
    border-left: 4px solid var(--pcg-baseline);
}}
.pcg-card-verified {{
    border-left: 4px solid var(--pcg-green);
}}
.pcg-card-failed {{
    border-left: 4px solid var(--pcg-amber);
    background: #FFF8E1;
}}
.pcg-pill {{
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.78em;
    font-weight: 600;
    margin-right: 6px;
}}
.pcg-pill-ok {{ background: #E6F4EA; color: var(--pcg-green); }}
.pcg-pill-warn {{ background: #FFF4D6; color: var(--pcg-amber); }}
.pcg-pill-fail {{ background: #FCE3E5; color: var(--pcg-ours); }}
.pcg-kpi-value {{
    font-size: 2.2em;
    font-weight: 800;
    color: var(--pcg-ours);
    line-height: 1.0;
}}
.pcg-kpi-label {{
    font-size: 0.95em;
    font-weight: 600;
    color: var(--pcg-ink);
}}
.pcg-kpi-sub {{
    font-size: 0.82em;
    font-style: italic;
    color: var(--pcg-ink-light);
}}
.pcg-mono {{
    font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, monospace;
    font-size: 0.9em;
}}
.pcg-anonymous-banner {{
    background: linear-gradient(90deg, #FCE3E5 0%, #FFF8E1 100%);
    border: 1px solid #F0BFC2;
    color: var(--pcg-ink);
    padding: 8px 14px;
    border-radius: 8px;
    font-size: 0.85em;
    margin-bottom: 16px;
}}
</style>
"""


def inject_css() -> None:
    """Call once at the top of every page."""
    import streamlit as st
    st.markdown(CSS, unsafe_allow_html=True)
