# APP_AUDIT_V3.md

## Existing app — inspected before any change

| Component | Size | Assessment |
|---|---|---|
| `server.py` | 580 LOC, FastAPI, 14 routes (`/api/run`, `/api/stress_stream`, `/api/sidebyside`, upload, fixtures, backends) | genuinely working; SSE streaming for run animation is a real asset |
| `static/` | 640 HTML / 1405 JS / 2497 CSS | substantial; visually dated but functionally complete |
| `pcg_glue/` | 10 modules, ~3.4k LOC (`channels`, `pipeline`, `responsibility`, `redundancy`, `risk_control`, `audit_envelopes`, `attacks`, `claim_extractor`, `sources`, `schemas`) | **duplicates scientific logic** that now lives in `pcg.v3` — the principal drift risk |
| `demo_data/` | 8 fixtures across PDF/DOCX/CSV/JSON/MD | keep; genuinely useful for the demo |
| `Dockerfile` | non-root user, slim base | sound; retained |

**Verdict: preserved, not replaced.** Deleting ~8.8k LOC of working demo would have been gratuitous.

## Principal risk identified: JS/Python scientific drift

`pcg_glue/channels.py` (514 LOC) and `static/app.js` (1405 LOC) both encode channel names and acceptance semantics. Nothing forced them to agree with `src/pcg/`.

**Remediation:** `app/shared/generate_contract.py` emits `contract.json` from the Python core — conjuncts, five channels, conjunct→channel map, gate states, controller actions, provenance classes, graph nodes/terminals, and the full record schema. The v3 frontend reads `/api/contract` and defines no scientific term of its own. The legacy `pcg_glue` path is untouched and still runs, but it is no longer the source of truth for anything the v3 UI shows.

## What v3 adds

| Path | Purpose |
|---|---|
| `app/shared/` | generated Python→JS contract (anti-drift) |
| `app/backend/main.py` | v3 API: `/health`, `/ready`, `/version`, `/api/contract`, `/api/registry`, `/api/artifacts`, `/api/v3/run`, `/api/v3/policy`. FastAPI when installed, stdlib fallback otherwise |
| `app/frontend/` | eleven-view control plane (Live Run, Certificate, Channels, Replay, Attribution, Dependence, Policy, Trace, Cost, Results, Manuscript) |
| `app/cloudflare/` | `_headers` (CSP, nosniff, frame-deny), `_redirects`, `wrangler.toml`, `build.sh` |
| `app/render/` | `render.yaml`, `start.sh`, backend-only requirements |

## Health / readiness

`/health` returns status and uptime only — no artifact scan, no model, no optional component initialisation. `/ready` additionally checks the contract file exists. This is deliberate: prior Render deployments were sensitive to slow readiness probes, and the fix is to make the probe structurally incapable of being slow.

## BYOK handling

The v3 backend accepts a user key in the request body only. It is not persisted, not logged, not echoed, not placed in telemetry and not placed in a URL; `providers/hosted.py` redacts `Authorization`/`X-API-Key` in any header map. In this release `byok_live` returns an explicit refusal because `NETWORK_API_MODEL_CALLS` must stay 0. No project-owned secret exists in any Cloudflare asset — verified by `SECRET_LEAK_SCAN=PASS`.

## Demo-mode separation

`offline_synthetic` (default, deterministic, eight scenarios covering acceptance, replay failure, drift, checker failure, policy failure, verifier-isolation breach, insufficient-evidence gate, controller refusal) · `byok_live` (user-activated, not wired) · `experiment_results` (reads `artifacts/v3_0/`). Modes are never visually conflated: a page banner plus per-value `synthetic` tags state that nothing shown is a DIRECT result.

## Not done

The legacy `static/` UI and `pcg_glue/` remain as they were. Migrating the legacy routes onto the contract, or retiring `pcg_glue` in favour of `pcg.v3`, is a follow-up that changes public behaviour and should be a deliberate decision rather than a side effect of remediation.
