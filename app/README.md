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

This Space is an **interactive companion** to the PCG-MAS paper.
Every answer the system produces is accompanied by a tamper-evident,
SHA-256-stamped certificate that audits the answer along five
independent verification channels.

This README explains how to use the live demo. The full paper, source
code, evaluation scripts, and reproducibility scaffold are linked at the
bottom — they are **not** intended to be rebuilt on a laptop. The Space
runs the real pipeline; using it here is the supported path.

---

## Quick start

1. Open the **Live Run** tab.
2. Paste your own LLM API key into the *API key* field
   (OpenAI, Anthropic, DeepSeek, or a Hugging Face Inference token).
   Keys are held in-process for the lifetime of a single request and
   never persisted. Nothing is logged.
3. Pick an input mode:
   * **Text** — paste your question and any supporting context
   * **URL** — supply a public page (Wikipedia works well)
   * **File** — upload PDF / DOCX / CSV / XLSX / MD / TXT / JSON / HTML
     up to 10 GB
4. Press **Run PCG-MAS**.

The five-channel pipeline streams its progress live as it certifies the
answer. Each accepted answer carries a downloadable certificate.

---

## What you are looking at

The app has eight tabs. Each one reads from the **same** latest
completed run — switch freely; nothing re-fetches.

| Tab | What it shows |
|---|---|
| **Live Run** | Input form, live channel-by-channel pipeline visualization, per-claim verdict cards, final risk decision banner |
| **Certificate Inspector** | The full certificate as a dense per-claim grid plus the raw JSON (copy-to-clipboard) |
| **Stress** | Nine adversarial perturbations of the most recent input — paraphrase, evidence reorder, evidence redaction, etc. — each scored with its own certificate |
| **Responsibility** | Per-claim mask-and-replay analysis: which cited components were responsible for the accepted verdict |
| **Risk Controller** | The 4-action expected-cost table (Answer / Verify / Escalate / Refuse), the dominant failure channel, and the audit envelopes per channel |
| **Results** | Side-by-side: a raw LLM answer next to the PCG-MAS-certified answer, for curated cases and your own inputs |
| **Architecture** | Static reference: the 5-channel mapping, the event-streaming choreography, the risk-action regime table |
| **About** | Scope note + source-code link |

### The five channels

Every claim is audited along all five before acceptance:

| Channel | What it checks |
|---|---|
| **V_I** Integrity | Cryptographic hash of cited evidence matches what the run committed |
| **V_R** Replay | Deterministic replay of the claim from the cited evidence reproduces the answer |
| **V_D** Drift | Replayed text has not semantically drifted from the original answer |
| **V_Ch** Checker | Independent entailment check: does the evidence in fact support the claim |
| **V_Cov** Coverage | Evidence substantively addresses the claim, not adjacent topics |

A green pill means the channel passed for **every** claim in the run.
A grey pill means the channel was skipped (a prior channel failed and
the result was not informative). A red pill means at least one claim
failed that channel.

---

## How to interpret a run

### Risk action

The risk controller picks one of four actions based on posterior risk
`r ∈ [0, 1]`:

| Action | When |
|---|---|
| **Answer** | All channels held; answer is emitted with full certificate |
| **Verify** | Borderline acceptance; another pass is warranted before commit |
| **Escalate** | Channels disagree enough that human or stronger-model review is recommended |
| **Refuse** | The pipeline cannot self-certify the answer; the answer is withheld |

A **Refuse** is not a failure of the demo — it is the demo working. The
raw LLM, lacking these checks, would have answered anyway.

### What PCG-MAS adds over a raw LLM

On the **Results** tab the same query is answered twice — once by the
raw LLM, once by PCG-MAS. On well-grounded queries the two often produce
the same text. The asymmetry is that the PCG-MAS answer carries:

* a tamper-evident certificate ID and SHA-256 hash
* per-claim attribution to specific evidence items
* a five-channel pass strip auditing the answer
* a "why this matters" narrative auto-generated from the certificate

On adversarial queries (try the **Clinical · Hallucinated dosage**
fixture in the Results dropdown) the two answers diverge — PCG-MAS
refuses or escalates where the raw LLM would emit an unverifiable
number.

---

## Suggested first runs

To see the system across its main regimes:

* **Grounded acceptance.** Live Run, Text mode.
  Question: *"Who proved Fermat's Last Theorem, and in what year?"*
  Context: *"Fermat's Last Theorem was proved by Andrew Wiles. His proof
  was announced in 1994 and published after correction in the
  mid-1990s."*
  Expected: all five channels green, risk = Answer.

* **Refusal on missing evidence.** Same question, leave the context
  field empty. Expected: V_Cov fails, risk = Refuse.

* **Multi-claim certification.** Paste a short factual paragraph and
  ask for a summary. Multiple atomic claims appear, each independently
  certified.

* **Adversarial divergence.** Results tab, dropdown:
  **Clinical · Hallucinated dosage (MD)**. The trial summary
  deliberately omits the dosage; the contrast with the raw LLM is the
  point.

* **Stress.** After any Live Run, switch to the Stress tab and run the
  nine adversarial variants of your input.

---

## Constraints and conventions

* One file per upload. Multi-file is not supported in this Space.
* The pipeline is honest about latency: a multi-claim run is typically
  10 – 30 seconds; mask-and-replay responsibility is longer; the
  nine-variant stress test is 60 – 150 seconds end to end. Each panel
  shows a progress indicator while it works.
* API keys are never persisted. They live only in the current request.
* No telemetry, no analytics, no account, no login.

---

## Anonymity

This Space is anonymous for peer review. Owner identifying information
has been stripped. The source-code mirror is hosted on
[anonymous.4open.science](https://anonymous.4open.science/r/p-c-m-a-E866).

---

## Source and citation

Full source, paper, and evaluation harness:
<https://anonymous.4open.science/r/p-c-m-a-E866>

This interactive demo is provided for reproducibility and peer review
only; not for production use in healthcare, finance, legal,
safety-critical, or regulated settings.
