# DEPLOYMENT_AUDIT_V3.md

**No deployment was performed.** Configs and runbooks only.

## What existed

A `Dockerfile` (python:3.11-slim, non-root uid 1000) and `app/requirements.txt` pulling the full demo stack including `openai`, `anthropic`, `huggingface_hub`, `reportlab`, `pypdf`, `python-docx`, `openpyxl`. No `render.yaml`, no Cloudflare config, no documented health/readiness contract, no CORS policy, no security headers, no rollback procedure. The public demo at `https://pcg-mas-demo.pages.dev` was not inspected live (that would be a network call).

## Findings

| # | Finding | Severity | v3 disposition |
|---|---|---|---|
| 1 | No `render.yaml`; deployment shape lived only in operator memory | high | `app/render/render.yaml` with pinned `PYTHON_VERSION=3.12.13`, `healthCheckPath: /health`, `autoDeploy: false` |
| 2 | Backend requirements pulled provider SDKs and document parsers into the API image | high | `app/render/requirements-backend.txt` is FastAPI + uvicorn + numpy only; the API image cannot download a model |
| 3 | No fast health endpoint; readiness would have initialised the pipeline | high | `/health` touches nothing heavyweight; `/ready` checks one file |
| 4 | No CORS allowlist | medium | `PCG_CORS_ORIGINS`, defaulting to the pages.dev origin |
| 5 | No security headers | medium | `_headers` sets CSP (`connect-src` limited to self + `*.onrender.com`), nosniff, frame-deny, no-referrer, restrictive Permissions-Policy |
| 6 | No documented rollback | medium | `app/DEPLOYMENT.md` §Rollback for both Pages and Render |
| 7 | Provider SDKs in the frontend build path could leak a key into a public asset | high | frontend is static, has no key, and `SECRET_LEAK_SCAN=PASS` is a release gate |
| 8 | `PCG_OFFLINE_ONLY` had no analogue | medium | set to `1` in `render.yaml`; every non-mock provider route raises `NetworkCallBlocked` |

## Verified here

```
CLOUDFLARE_BUILD=PASS      RENDER_BACKEND_BUILD=PASS
HEALTH_READY_CHECKS=PASS   SECRET_LEAK_SCAN=PASS
```

These check that the configs exist and are well-formed and that `/health` and `/ready` respond correctly in-process. They do **not** assert anything about the live production state, which was deliberately not contacted.

## Before the next deploy

1. Confirm the Render service name and region against the existing service.
2. Set `PCG_API_BASE` as a Pages build variable to the Render URL.
3. Run `bash scripts/v3/verify_offline.sh` — the secret scan is part of it.
4. Deploy Render first, confirm `/health` and `/version`, then deploy Pages.
5. Keep `autoDeploy: false` so a deploy is always deliberate.
