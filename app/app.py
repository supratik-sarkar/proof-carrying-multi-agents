"""
PCG-MAS demo — Streamlit entry point.

This file is the landing page; the five interactive demos live under
`pages/` and Streamlit auto-discovers them. Run with:

    streamlit run app/app.py

For HF Spaces deployment, this same file gets pushed to the Space root.
"""
from __future__ import annotations

import streamlit as st

# Make `from components import ...` work whether we run from repo root
# or from inside the `app/` directory.
import sys
from pathlib import Path
_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from components import inject_css, render_byok_sidebar, render_kpi  # noqa: E402

# ---------------------------------------------------------------------------
# Page config (must be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PCG-MAS · live demo",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "PCG-MAS — Proof-Carrying Generation for Multi-Agent Systems. "
            "Anonymous demo for double-blind review. No analytics."
        ),
    },
)

inject_css()
render_byok_sidebar()


# ---------------------------------------------------------------------------
# Anonymous banner (for double-blind review)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="pcg-anonymous-banner">
        🕶 <strong>Anonymous demo</strong> — this Space is hosted under an
        anonymous account for double-blind review. No author info, no
        analytics, no cookies beyond Streamlit's session state.
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.title("Proof-Carrying Generation for Multi-Agent Systems")
st.markdown(
    "_Every accepted claim ships a certificate `Z = (c, S, Π, Γ, p, meta)` "
    "that an external auditor can verify deterministically and cheaply._"
)

st.markdown("---")

# ---------------------------------------------------------------------------
# What this demo lets you do
# ---------------------------------------------------------------------------
st.markdown("### What this demo lets you do")

col_left, col_right = st.columns([1.0, 1.0])

with col_left:
    st.markdown("""
**1. 🔥 Live Run** — paste any question, watch the multi-agent pipeline run
in real time, and see the certificate `Z` it produces. Free LLM by default;
add your own API key in the sidebar to use frontier models like DeepSeek-V3,
Llama-3.3-70B, GPT-4, or Claude.

**2. 🔍 Certificate Inspector** — paste or upload any past certificate,
see all six components, and re-run the deterministic verifier to confirm
the claim is still backed by evidence.

**3. ⚖️ Side-by-Side Comparison** — same question, run twice: once
through PCG-MAS, once through a no-certificate baseline. See exactly
what gets caught.
""")

with col_right:
    st.markdown("""
**4. 📊 Results Browser** — interactive Plotly charts of all five paper
experiments (R1 Audit decomposition, R2 Redundancy law, R3 Responsibility,
R4 Cost-harm Pareto, R5 Overhead).

**5. 🛡 Auditor Demo** — you are the external auditor. A stream of
certificates arrives; some are tampered. Find the failures using only
the verifier, no LLM access. Demonstrates the auditability contract
that makes PCG-MAS deployable in regulated settings.
""")

st.markdown("---")

# ---------------------------------------------------------------------------
# Headline numbers (mirrors the paper's KPI panel)
# ---------------------------------------------------------------------------
st.markdown("### Headline numbers from the paper")

c1, c2, c3, c4 = st.columns(4)
with c1:
    render_kpi("80×", "fewer false accepts", "at k=8 vs no certificate (R2)")
with c2:
    render_kpi("82%", "Thm 1 bound tightness", "LHS / RHS audit ratio (R1)")
with c3:
    render_kpi("82%", "top-1 root cause", "averaged across 4 regimes (R3)")
with c4:
    render_kpi("40×", "lower harm", "vs always-answer baseline (R4)")

st.markdown("---")

st.markdown(
    """
**Pick a demo from the sidebar** to start. The default backend is
the free Hugging Face Inference tier — no key, no cost, rate-limited
(plenty for review traffic). Add an API key for any premium provider
in the sidebar to use frontier models for fresh runs.
"""
)

st.markdown(
    """
<div style="margin-top:48px; padding-top:16px; border-top: 1px solid #E5E7EB;
            font-size:0.82em; color: var(--pcg-ink-light);">
Anonymous submission · NeurIPS 2026 · MIT License (in the source repo) ·
<a href="https://anonymous.4open.science" target="_blank">source code</a>
</div>
""",
    unsafe_allow_html=True,
)
