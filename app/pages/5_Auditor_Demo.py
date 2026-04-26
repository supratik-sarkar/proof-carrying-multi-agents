"""
Page 5 — External auditor demo.

Reviewer puts on the auditor hat. A stream of certificates arrives.
Some are clean. Some are tampered. The reviewer's job is to spot the
failures using ONLY the deterministic verifier — no LLM access,
no recomputation of evidence.

This page makes the auditability story concrete: "PCG-MAS lets a
regulator who doesn't trust the model provider still verify every
claim, deterministically, in milliseconds.\""""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import streamlit as st

_HERE = Path(__file__).parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from components import (
    Certificate, inject_css, render_byok_sidebar, render_certificate_card,
)

st.set_page_config(page_title="Auditor · PCG-MAS", page_icon="🛡", layout="wide")
inject_css()
render_byok_sidebar()

st.title("🛡 External auditor demo")
st.markdown(
    "**You are an external auditor.** A stream of certificates arrives "
    "from a model provider you don't fully trust. Your task: classify "
    "each as ACCEPT or REJECT using only the deterministic verifier. "
    "No model invocations, no re-running the agents. Just signature "
    "checks and audit-channel arithmetic."
)

# ---------------------------------------------------------------------------
# Load a stream of certificates (mix of clean + tampered)
# ---------------------------------------------------------------------------
@st.cache_data
def load_stream() -> list[dict]:
    p = _HERE / "demo_data" / "auditor_stream.json"
    if p.exists():
        return json.loads(p.read_text())
    # Synthesize a tiny demo stream
    return _synthetic_stream(n=8)


def _synthetic_stream(n: int = 8) -> list[dict]:
    import hashlib
    out = []
    for i in range(n):
        is_tampered = (i in (2, 5))   # known bad indices for the demo
        claim = f"Demo claim #{i}: a fact about topic {chr(ord('A') + i)}."
        sig = hashlib.sha256(claim.encode()).hexdigest()
        if is_tampered:
            sig = sig[:-4] + "0000"   # tamper the signature
        out.append({
            "c": claim,
            "S": [
                {"agent": "prover_1", "role": "drafter", "sig": sig[:32]},
                {"agent": "verifier", "role": "consensus", "sig": sig},
            ],
            "Pi": [{"op": "demo_step", "detail": "stub"}],
            "Gamma": [{
                "title": f"ref-{i}",
                "text": "demo evidence",
                "source_url": "https://example.org/", "is_gold": True,
            }],
            "p": 0.95 - 0.1 * is_tampered,
            "meta": {
                "audit_channels": (
                    {"p_int_fail": 0.30, "p_replay_fail": 0.05,
                     "p_check_fail": 0.20, "p_cov_gap": 0.05}
                    if is_tampered else
                    {"p_int_fail": 0.02, "p_replay_fail": 0.02,
                     "p_check_fail": 0.03, "p_cov_gap": 0.04}
                ),
                "ground_truth_tampered": is_tampered,  # secret; revealed at end
            },
        })
    return out


stream = load_stream()
n = len(stream)

if "auditor_picks" not in st.session_state:
    st.session_state["auditor_picks"] = {}   # idx -> "accept" | "reject"

st.markdown(f"### {n} certificates in this batch")
st.caption(
    "Click ACCEPT or REJECT on each. The verifier-decision is shown but "
    "the auditor's classification is YOUR call."
)

for i, cdict in enumerate(stream):
    cert = Certificate.from_dict(cdict)
    with st.container():
        col_card, col_actions = st.columns([3, 1])
        with col_card:
            render_certificate_card(
                cert, expanded_by_default=False,
                title_prefix=f"#{i}",
            )
            # Run the deterministic verifier visibly
            t0 = time.time()
            from pages._live_run_helpers import deterministic_verify
            verdict = deterministic_verify(cert)
            dt_us = int((time.time() - t0) * 1_000_000)
            audit_sum = sum(cert.meta.get("audit_channels", {}).values())
            st.caption(
                f"Verifier: {'ACCEPT' if verdict else 'REJECT'} · "
                f"audit sum = {audit_sum:.2f} · checked in {dt_us} µs"
            )
        with col_actions:
            current = st.session_state["auditor_picks"].get(i)
            if st.button(
                "✓ ACCEPT", key=f"acc_{i}",
                type=("primary" if current == "accept" else "secondary"),
                use_container_width=True,
            ):
                st.session_state["auditor_picks"][i] = "accept"
            if st.button(
                "✗ REJECT", key=f"rej_{i}",
                type=("primary" if current == "reject" else "secondary"),
                use_container_width=True,
            ):
                st.session_state["auditor_picks"][i] = "reject"

# ---------------------------------------------------------------------------
# Score
# ---------------------------------------------------------------------------
st.markdown("---")
if st.button("📊 Score my decisions", type="primary"):
    picks = st.session_state["auditor_picks"]
    if len(picks) < n:
        st.warning(f"You've classified {len(picks)}/{n} certificates.")
    correct = 0
    for i, cdict in enumerate(stream):
        is_bad = cdict["meta"].get("ground_truth_tampered", False)
        truth = "reject" if is_bad else "accept"
        if picks.get(i) == truth:
            correct += 1
    st.metric(
        "Audit accuracy",
        f"{correct}/{len(picks)} correct",
        delta=f"{100 * correct / max(1, len(picks)):.0f}%",
    )
    st.caption(
        "Real PCG-MAS audits run in microseconds per certificate, on the "
        "auditor's machine, with no model access required. This is the "
        "deployability story."
    )
