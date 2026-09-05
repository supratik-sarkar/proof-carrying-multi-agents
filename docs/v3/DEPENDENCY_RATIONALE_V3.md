# DEPENDENCY_RATIONALE_V3.md

Preference order: **correctness > reproducibility > auditability > deterministic execution > observability > modularity > scalability > sophistication.** Every dependency must earn its place; none may make the research unreproducible on a Mac or in Colab.

## Mandatory — and deliberately tiny

| Dependency | Why | Consequence if removed |
|---|---|---|
| `numpy` | array maths in metrics | core metrics unimplementable |

That is the whole mandatory set for the scientific core. **SciPy is optional**: Clopper–Pearson is implemented from a Lentz continued-fraction regularized incomplete beta plus bisection, validated against published values (`CP(2,10) = 0.0252, 0.5561`; `I_0.5(2,3) = 0.6875`). SciPy is used as a fast path when present.

## Verification only

`pytest`, `hypothesis` — unit, property and contract tests. The suite also runs under a 40-line stdlib harness so it can execute on a bare interpreter.

## Optional adapters — each flagged, each with a working fallback

| Component | Architectural role | Boundary |
|---|---|---|
| **LangGraph** | typed orchestration/state machine over the explicit node sequence | The identical deterministic sequence runs with LangGraph absent (`core_runs_without_langgraph: true`). No hidden loops, no unbounded retries, every transition explicit, every node emits provenance. **The manuscript semantics, not LangGraph, define PCG-MAS.** |
| **LangSmith** | observability only | `LANGSMITH_ENABLED=false` leaves the core complete. Local provenance is authoritative. Sensitive fields are withheld unless explicitly opted in. No cloud call in this pass. |
| **OpenTelemetry** | traces, spans, durations, errors, correlation IDs across orchestration, providers, checkers, replay, audit, controller and the demo backend | Local exporter with a stdlib fallback. **The same telemetry feeds A10** — there is no separate scientific timing system and UI timing system. |
| **OPA / Rego** | one concrete policy backend for `V_Γ` | Does **not** redefine `V_Γ`. Bundle version and SHA-256 enter provenance. A deterministic local evaluator runs with OPA absent, verified in this pass (`opa` binary is not installed here). |
| **NeMo Guardrails** | optional boundary control and guardrail comparator | May not silently rewrite scientific inputs, may not become the hidden implementation of PCG, may not contaminate matched baselines. Every intervention is logged and provenance-tagged. |
| **FastAPI / uvicorn** | demo backend | A stdlib `http.server` fallback serves the identical routes, so offline smoke tests run anywhere. |
| **matplotlib** | figure generation | Only in the figure path; `Agg` backend, no display. |
| **Pydantic v2** | typed schemas | The record is a dataclass with explicit validation plus JSON-Schema export, so the core does not hard-depend on Pydantic; adopt it in the app layer where request validation is the natural fit. |

## Explicitly rejected

* a workflow engine (Airflow/Prefect) — runners are finite and idempotent; a scheduler adds operational surface without improving reproducibility;
* provider SDKs in the backend image — the API must not be able to download a model;
* any dashboard, notebook server or long-lived worker — violates the non-interactive constraint;
* heavier config frameworks — an immutable spec plus a hash is the entire requirement.
