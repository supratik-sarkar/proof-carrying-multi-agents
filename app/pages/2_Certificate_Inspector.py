"""
Page 2 — Certificate Inspector.

Paste or upload any past certificate (JSON), see all six components
laid out, and re-run the deterministic verifier to confirm the claim
is still backed by the cited evidence."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

_HERE = Path(__file__).parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from components import (
    Certificate, inject_css, render_byok_sidebar, render_certificate_card,
)

st.set_page_config(page_title="Certificate Inspector · PCG-MAS",
                   page_icon=":mag:", layout="wide")
inject_css()
render_byok_sidebar()

st.title("Certificate Inspector")
st.markdown(
    "Paste a JSON certificate, upload one, or pick from the bundled "
    "samples. The Inspector renders all six fields of `Z = (c, S, Π, "
    "Γ, p, meta)` and lets you re-run the deterministic verifier."
)

# ---------------------------------------------------------------------------
# Source of certificate
# ---------------------------------------------------------------------------
SAMPLES = {
    "Verified · Marie Curie Nobel": _HERE / "demo_data" / "cert_marie_curie.json",
    "Failed · adversarial evidence": _HERE / "demo_data" / "cert_adversarial.json",
    "Verified · TP53 tumor suppressor": _HERE / "demo_data" / "cert_tp53.json",
}

input_mode = st.radio(
    "Source",
    ["Paste JSON", "Upload file", "Pick a sample"],
    horizontal=True,
)

cert_dict: dict | None = None
if input_mode == "Paste JSON":
    text = st.text_area(
        "Paste certificate JSON",
        height=240,
        placeholder='{"c": "...", "S": [...], "Pi": [...], "Gamma": [...], "p": 0.93, "meta": {}}',
    )
    if text.strip():
        try:
            cert_dict = json.loads(text)
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")

elif input_mode == "Upload file":
    f = st.file_uploader("Upload .json", type=["json"])
    if f is not None:
        try:
            cert_dict = json.load(f)
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON in upload: {e}")

else:
    sample_label = st.selectbox("Sample certificate", list(SAMPLES.keys()))
    sample_path = SAMPLES[sample_label]
    if sample_path.exists():
        cert_dict = json.loads(sample_path.read_text())
    else:
        st.info(
            f"Sample file `{sample_path.name}` not present in this build. "
            "Run a Live Run, download the certificate, and paste/upload it here."
        )

# ---------------------------------------------------------------------------
# Render + verify
# ---------------------------------------------------------------------------
if cert_dict is not None:
    cert = Certificate.from_dict(cert_dict)
    st.markdown("---")
    render_certificate_card(cert, expanded_by_default=True)

    if st.button("Re-run deterministic verifier", type="primary"):
        # Use the same lightweight verifier as page 1
        from pages._live_run_helpers import deterministic_verify  # noqa
        cert.is_verified = deterministic_verify(cert)
        if cert.is_verified:
            st.success("Verifier accepted. Signature matches and audit-channel sum below threshold.")
        else:
            st.error("Verifier REJECTED. Either the signature is wrong or the audit sum exceeds threshold.")
        st.markdown("---")
        render_certificate_card(cert, expanded_by_default=True,
                                title_prefix="Re-verified")
else:
    st.info("Provide a certificate to inspect.")
