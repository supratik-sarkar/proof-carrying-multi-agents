---
title: PCG-MAS · interactive verification demo
emoji: 🛡️
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Proof-Carrying Generation for Multi-Agent Systems
---

# PCG-MAS — interactive verification demo

Static HTML / CSS / JavaScript front-end backed by a FastAPI server that
runs the real four-channel verification pipeline (V_H, V_Π, V_Γ, V_⊢) on
every user input. Every accepted answer carries a tamper-evident
SHA-256-stamped certificate.

## Structure

- static/index.html / styles.css / app.js — Apple-grade front-end
- server.py                                — FastAPI + SSE pipeline streaming
- pcg_glue/                                — verification pipeline modules
- demo_data/                               — curated examples + fixture files
- Dockerfile + requirements.txt            — image built by HF Spaces

## Running locally

    cd app
    pip install -r requirements.txt
    python server.py

Then open http://127.0.0.1:7860 in a browser.

## Anonymity

Anonymous space for peer review. No telemetry, no analytics, no key
persistence — API keys live in the browser session and in process memory
for the duration of a single request only.
