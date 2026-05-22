"""
Certificate Z = (c, S, Π, Γ, p, meta) renderer.

This is the central visual artifact of PCG-MAS — the "proof" that ships
with every accepted claim. The renderer shows all six fields as
inspectable cards plus a "verify" button that re-runs the deterministic
checker.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import streamlit as st


@dataclass
class Certificate:
    """In-app representation of a PCG-MAS certificate.

    Mirrors `pcg.certificate.Certificate` but kept independent so the
    demo can render certificates that came from disk, from a paste-box,
    or from a fresh agent run."""
    c: str                              # the claim
    S: list[dict] = field(default_factory=list)   # signatures (per-agent)
    Pi: list[dict] = field(default_factory=list)  # plan (the proof DAG)
    Gamma: list[dict] = field(default_factory=list)  # evidence (cited docs)
    p: float = 0.0                      # asserted probability
    meta: dict[str, Any] = field(default_factory=dict)
    is_verified: bool | None = None     # populated by re-verification

    @classmethod
    def from_dict(cls, d: dict) -> "Certificate":
        return cls(
            c=d.get("c", ""),
            S=list(d.get("S", [])),
            Pi=list(d.get("Pi", [])),
            Gamma=list(d.get("Gamma", [])),
            p=float(d.get("p", 0.0)),
            meta=dict(d.get("meta", {})),
            is_verified=d.get("is_verified"),
        )

    def to_dict(self) -> dict:
        return {
            "c": self.c, "S": self.S, "Pi": self.Pi,
            "Gamma": self.Gamma, "p": self.p, "meta": self.meta,
            "is_verified": self.is_verified,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

def render_certificate_card(
    cert: Certificate,
    *,
    expanded_by_default: bool = False,
    title_prefix: str = "Certificate",
) -> None:
    """Compact card view of a certificate, with the 6 components in expanders."""
    status_pill = ""
    if cert.is_verified is True:
        status_pill = "<span class='pcg-pill pcg-pill-ok'>✓ verified</span>"
    elif cert.is_verified is False:
        status_pill = "<span class='pcg-pill pcg-pill-fail'>✗ failed</span>"
    else:
        status_pill = "<span class='pcg-pill pcg-pill-warn'>not yet verified</span>"

    klass = "pcg-card pcg-card-our"
    if cert.is_verified is True:
        klass = "pcg-card pcg-card-verified"
    elif cert.is_verified is False:
        klass = "pcg-card pcg-card-failed"

    st.markdown(
        f"""
        <div class="{klass}">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <strong>{title_prefix}: {cert.c[:120]}{'…' if len(cert.c) > 120 else ''}</strong>
                <div>{status_pill}</div>
            </div>
            <div class="pcg-mono" style="margin-top:4px; color: var(--pcg-ink-light);">
                p = {cert.p:.4f} · |S| = {len(cert.S)} ·
                |Π| = {len(cert.Pi)} · |Γ| = {len(cert.Gamma)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Claim (c)", expanded=expanded_by_default):
        st.write(cert.c)

    with st.expander(
        f"Signatures (S) — {len(cert.S)} agents",
        expanded=expanded_by_default,
    ):
        if not cert.S:
            st.caption("No signatures")
        else:
            for sig in cert.S:
                _render_signature(sig)

    with st.expander(f"Plan / proof DAG (Π) — {len(cert.Pi)} steps"):
        if not cert.Pi:
            st.caption("No plan steps")
        else:
            for i, step in enumerate(cert.Pi):
                _render_plan_step(i, step)

    with st.expander(
        f"Evidence (Γ) — {len(cert.Gamma)} docs",
        expanded=expanded_by_default,
    ):
        if not cert.Gamma:
            st.caption("No evidence cited")
        else:
            for ev in cert.Gamma:
                _render_evidence(ev)

    with st.expander(f"Asserted probability (p) — {cert.p:.4f}"):
        st.markdown(
            "Probability assigned by the Prover that this claim is true "
            "given the cited evidence. Used as input to the threshold "
            "policy and to the responsibility computation."
        )
        st.progress(min(1.0, max(0.0, cert.p)))

    with st.expander("Metadata (meta)"):
        if not cert.meta:
            st.caption("No metadata")
        else:
            st.json(cert.meta)


def _render_signature(sig: dict) -> None:
    agent = sig.get("agent", "unknown")
    role = sig.get("role", "")
    sig_hex = sig.get("sig", "")
    if isinstance(sig_hex, str) and len(sig_hex) > 64:
        sig_hex = sig_hex[:32] + "…" + sig_hex[-16:]
    st.markdown(
        f"**{agent}** ({role}) — "
        f"<span class='pcg-mono'>{sig_hex}</span>",
        unsafe_allow_html=True,
    )


def _render_plan_step(idx: int, step: dict) -> None:
    label = step.get("op", step.get("step", f"step_{idx}"))
    detail = step.get("detail", "")
    st.markdown(f"`{idx:02d}` **{label}** — {detail}")


def _render_evidence(ev: dict) -> None:
    title = ev.get("title", "(untitled)")
    src = ev.get("source_url", "")
    text = ev.get("text", "")
    is_gold = ev.get("is_gold", False)
    pill = (
        "<span class='pcg-pill pcg-pill-ok'>gold</span>"
        if is_gold else ""
    )
    st.markdown(
        f"<div class='pcg-card'>"
        f"<strong>{title}</strong> {pill}<br/>"
        f"<span class='pcg-mono' style='color:var(--pcg-ink-light);'>{src}</span>"
        f"<div style='margin-top:6px;'>{text[:400]}{'…' if len(text) > 400 else ''}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def render_kpi(value: str, label: str, sub: str = "") -> None:
    """Big-number KPI tile. Same look as the paper's headline-numbers panel."""
    st.markdown(
        f"""
        <div class="pcg-card pcg-card-our">
            <div class="pcg-kpi-value">{value}</div>
            <div class="pcg-kpi-label">{label}</div>
            <div class="pcg-kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
