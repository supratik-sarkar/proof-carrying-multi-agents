# PCG-MAS v3.0 — deployment

Nothing here is deployed by the remediation pass. These are prepared configs and runbooks.

## Architecture

```
Cloudflare Pages  (app/frontend)  ──HTTPS──▶  Render web service (app/backend)
      static, no secrets                          FastAPI, PCG_OFFLINE_ONLY=1
```

The frontend consumes `app/shared/contract.json`, generated from the Python core, so
scientific definitions cannot drift between Python and JS.

## Local development

```bash
cd /path/to/pcg-mas
source .venv-pcg-mas/bin/activate            # Python 3.12
python app/shared/generate_contract.py
PYTHONPATH=src python app/backend/main.py     # http://127.0.0.1:8000
```

With FastAPI absent the same routes are served by a stdlib fallback, so offline smoke
tests run anywhere.

## Environment variables

| Variable | Where | Default | Purpose |
|---|---|---|---|
| `PCG_API_BASE` | Pages build | empty | backend origin baked into `frontend/config.js` |
| `PCG_CORS_ORIGINS` | Render | pages.dev origin | comma-separated allowlist |
| `PCG_OFFLINE_ONLY` | Render | `1` | blocks every provider route except `offline_mock` |
| `PCG_ARTIFACT_ROOT` | Render | `artifacts/v3_0` | results-mode artifact directory |
| `LANGSMITH_ENABLED` | both | `false` | optional tracing; core is complete without it |
| `NEMO_GUARDRAILS_ENABLED` | both | `false` | optional boundary control |

**No provider API key is ever set for the public demo.** BYOK keys arrive in the request
body, live for the request only, and are never persisted, logged, echoed or placed in a URL.

## Cloudflare Pages

Build command `bash app/cloudflare/build.sh`, output directory `app/frontend`.
`_headers` sets CSP/nosniff/frame-deny; `_redirects` provides the SPA fallback.
Set `PCG_API_BASE` to the Render URL as a build-time variable.

## Render

Blueprint `app/render/render.yaml`. Health check is `/health` — fast by construction, no
model or artifact scan. `/ready` additionally confirms the contract file is present.
`/version` reports release, schema versions and adapter status. `autoDeploy` is off so a
deploy is always deliberate.

## Rollback

Cloudflare Pages: redeploy the previous successful build from the dashboard; assets are
immutable and content-addressed, so rollback is instant.
Render: `autoDeploy: false` means the previous image stays live until a manual deploy;
roll back by redeploying the prior commit. Because `/health` never touches heavyweight
components, a failed rollout surfaces immediately rather than through a timeout.

## Pre-deployment checklist

- [ ] `python app/shared/generate_contract.py` regenerated and committed
- [ ] `bash scripts/v3/verify_offline.sh` passes
- [ ] secret scan clean (`SECRET_LEAK_SCAN=PASS`)
- [ ] no provider key in any Pages asset
- [ ] `/health` responds under 100 ms cold
- [ ] synthetic values in the UI are visibly tagged
