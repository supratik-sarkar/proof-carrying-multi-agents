# APP_V3_ARCHITECTURE.md

## What was preserved

The pre-existing app is substantial and working: `server.py` (580 LOC, 14 routes),
`static/` (640 HTML / 1405 JS / 2497 CSS), ten `pcg_glue/` modules (~3.4k LOC),
`demo_data/` fixtures and a working `Dockerfile`. **None of it was deleted.** The v3
layer is additive and the legacy demo continues to run unchanged.

## What was added

```
app/
  shared/      generate_contract.py, contract.json   <- Python -> JS scientific contract
  backend/     main.py                                <- v3 API, FastAPI or stdlib fallback
  frontend/    index.html, styles.css, app.js         <- v3 control-plane UI
  cloudflare/  _headers, _redirects, wrangler.toml, build.sh
  render/      render.yaml, start.sh, requirements-backend.txt
  DEPLOYMENT.md, APP_V3_ARCHITECTURE.md
```

## Anti-drift design

The frontend does not define a single scientific term. Conjunct names, the five audit
channels, gate states, controller actions, provenance classes and the record schema are
served by `/api/contract`, generated from `pcg.v3`. If the core changes and the contract
is not regenerated, the UI shows the stale version string rather than silently disagreeing.

## Modes

| Mode | Network | Purpose |
|---|---|---|
| `offline_synthetic` | none | default; deterministic fixtures covering acceptance, replay failure, drift, checker failure, policy failure, verifier-isolation breach, insufficient-evidence gate and controller refusal |
| `byok_live` | user key only | user-activated; not wired in v3.0 because `NETWORK_API_MODEL_CALLS` must stay 0 |
| `experiment_results` | none | reads generated artifacts under `artifacts/v3_0/` |

Modes are never visually conflated: synthetic values carry a `synthetic` tag and the page
banner states that nothing shown is a DIRECT experimental result.

## Health and readiness

`/health` returns status and uptime only — no artifact scan, no model, no optional
component. `/ready` additionally checks the contract file. Heavyweight components are
lazily initialised. This matters because prior Render deployments were sensitive to slow
readiness probes.

## Scientific fidelity notes

* **Audit channels fire only on accepted-and-bad runs.** A correct rejection (checker or
  policy denies) fires no channel; the UI reflects this rather than inventing a signal.
* **Unknown is failure.** A conjunct with unknown value renders `UNKNOWN → treated as FAIL`.
* **Attribution is replay-interventional**, labelled as such, with `Unresolved` shown when
  the top-two margin falls below threshold.
* **Dependence** shows `U_joint(k, δ)` when `ρ` is not estimable, with no extrapolation implied.
