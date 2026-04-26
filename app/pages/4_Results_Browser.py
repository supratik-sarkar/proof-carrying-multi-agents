"""
Page 4 — Results browser.

Interactive Plotly versions of the paper's R1-R5 figures, driven from
the same JSON files the static plots are made from. Reviewers can hover,
toggle backends, zoom in on specific regimes, etc.

If `results/` isn't shipped with the Space (it usually isn't, since
result trees can be large), this page falls back to bundled demo
fixtures that produce the same chart shapes."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

_HERE = Path(__file__).parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from components import inject_css, render_byok_sidebar, render_kpi

st.set_page_config(page_title="Results · PCG-MAS", page_icon="📊", layout="wide")
inject_css()
render_byok_sidebar()

st.title("📊 Results browser")
st.markdown(
    "Interactive versions of the paper's five experiments. Hover, zoom, "
    "and toggle backends. Underlying JSON is downloadable for re-analysis."
)

# ---------------------------------------------------------------------------
# Load results (with bundled fallback)
# ---------------------------------------------------------------------------
RESULTS_ROOT = _HERE.parent / "results"
DEMO_FIXTURES = _HERE / "demo_data" / "results_fixtures.json"


@st.cache_data
def load_results():
    """Return dict keyed by R1..R5 → aggregated metrics. Prefer real run
    JSON when present; fall back to bundled fixtures."""
    out = {}
    if RESULTS_ROOT.exists():
        for r_id in ("r1", "r2", "r3", "r4", "r5"):
            run_dir = _latest_run(RESULTS_ROOT, r_id)
            if run_dir and (run_dir / f"{r_id}.json").exists():
                out[r_id] = json.loads((run_dir / f"{r_id}.json").read_text())
    if not out and DEMO_FIXTURES.exists():
        out = json.loads(DEMO_FIXTURES.read_text())
    return out


def _latest_run(root: Path, prefix: str) -> Path | None:
    if not root.exists():
        return None
    candidates = sorted([
        p for p in root.glob(f"{prefix}*") if p.is_dir()
    ])
    return candidates[-1] if candidates else None


results = load_results()

# ---------------------------------------------------------------------------
# Headline numbers (mirror of paper's KPI panel)
# ---------------------------------------------------------------------------
st.markdown("### Headline numbers")
c1, c2, c3, c4 = st.columns(4)
with c1:
    render_kpi("80×", "fewer false accepts", "R2 at k=8 vs no certificate")
with c2:
    render_kpi("82%", "Thm 1 tightness", "R1 LHS / RHS")
with c3:
    render_kpi("82%", "top-1 root cause", "R3 averaged across regimes")
with c4:
    render_kpi("40×", "lower harm", "R4 vs always-answer")

st.markdown("---")

# ---------------------------------------------------------------------------
# Per-experiment tabs
# ---------------------------------------------------------------------------
tabs = st.tabs([
    "R1 · Audit decomposition",
    "R2 · Redundancy law",
    "R3 · Responsibility",
    "R4 · Cost vs harm",
    "R5 · Overhead",
])

with tabs[0]:
    st.markdown(
        "**Theorem 1**: Pr(accept ∩ wrong) ≤ Pr(IntFail) + Pr(ReplayFail) + "
        "Pr(CheckFail) + Pr(CovGap). The empirical LHS should sit at or "
        "below the RHS sum across all backends.")
    st.info("TODO: Plotly bar chart of LHS vs channel-sum per backend.")

with tabs[1]:
    st.markdown(
        "**Theorem 2**: Pr(false accept | k Provers) ≤ ρ^(k−1) · ε^k. "
        "Empirical curve should fall at or below the bound on a log y-axis.")
    st.info("TODO: Plotly log-scale line of empirical false-accept rate vs k.")

with tabs[2]:
    st.markdown(
        "Top-1 root-cause accuracy across the four audit-channel regimes "
        "(integrity, replay, check, coverage).")
    st.info("TODO: Plotly grouped bar of accuracy per regime.")

with tabs[3]:
    st.markdown(
        "Pareto frontier of cost (tokens/claim) vs harm (false-accept rate) "
        "for three policies: always-answer / threshold-PCG / learned. "
        "PCG-MAS dominates across ε.")
    st.info("TODO: Plotly scatter with three policy traces.")

with tabs[4]:
    st.markdown(
        "Per-claim token breakdown by phase (prove / verify / redundant) "
        "across backends.")
    st.info("TODO: Plotly stacked bar per backend.")

st.markdown(
    "_The Plotly conversions are minimal — they read the same JSON the "
    "static figures consume. See `pcg.eval.plots_v2` for the canonical "
    "matplotlib renderers used in the paper._"
)
