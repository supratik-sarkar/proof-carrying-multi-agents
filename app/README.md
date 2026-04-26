---
title: PCG-MAS · Live Demo
emoji: 🔐
colorFrom: red
colorTo: yellow
sdk: streamlit
sdk_version: 1.39.0
app_file: app.py
pinned: false
license: mit
short_description: Proof-Carrying Multi-Agent Systems · anonymous demo
---

# PCG-MAS — Live demo

**Anonymous submission for double-blind review.** This Space lets you
interact with the multi-agent system described in the paper, inspect
the certificates `Z = (c, S, Π, Γ, p, meta)` it produces, and audit
them deterministically.

The Space runs five interactive pages:

1. **Live Run** — paste a question, watch the agents stream, see the
   certificate get built.
2. **Certificate Inspector** — paste/upload a `Z` and re-verify it.
3. **Side-by-Side** — same question through PCG-MAS vs a no-cert
   baseline. The difference is the certificate.
4. **Results Browser** — interactive Plotly versions of the paper's
   R1–R5 figures.
5. **Auditor Demo** — a stream of certificates arrives; some tampered.
   Find the failures using only the deterministic verifier.

## Backend

By default, runs use **HF Inference (free tier)** at no cost. To use
frontier models (DeepSeek-V3, Llama-3.3-70B, GPT-4, Claude, etc.),
paste your API key in the sidebar — keys are stored only in your
browser session and are never logged or persisted.

## Anonymity

This Space is hosted under an anonymous account specifically for
double-blind review. There are no analytics, no tracking cookies
beyond Streamlit's own session state, and no identifiable metadata
in the source code or commit history.

## Source

Full anonymized source is available at
[anonymous.4open.science/r/pcg-mas-anonymous](https://anonymous.4open.science).
The Space's own Files tab also exposes the deployment subset.
