"""PCG-MAS v3.0 demo backend.

Design constraints honoured here:
  * /health and /ready are FAST and must NOT initialise heavyweight components
    (prior Render deployments made this important);
  * expensive optional components are lazily initialised;
  * no provider API key is ever persisted, logged or placed in telemetry;
  * scientific definitions are imported from `pcg.v3`, never reimplemented;
  * three clearly separated modes: offline_synthetic | byok_live | experiment_results.

FastAPI is used when installed; otherwise a stdlib http.server app exposes the
same routes so offline smoke tests run anywhere.
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from pcg.v3.release import PCG_MAS_RELEASE, version_payload            # noqa: E402
from pcg.v3.orchestration.graph import run_graph, status as graph_status  # noqa: E402
from pcg.v3.telemetry.otel import reset as otel_reset, spans as otel_spans  # noqa: E402
from pcg.v3.policy.local import LocalPolicyBackend                     # noqa: E402
from pcg.v3.policy.opa import OPAPolicyBackend                         # noqa: E402
from pcg.v3.providers.registry import status as provider_status        # noqa: E402
from pcg.v3.telemetry.langsmith import LangSmithAdapter                # noqa: E402
from pcg.v3.guardrails.nemo import NeMoAdapter                         # noqa: E402
from pcg.v3.artifacts.registry import check_registry, load as load_registry  # noqa: E402

ARTIFACT_ROOT = os.environ.get("PCG_ARTIFACT_ROOT", os.path.join(ROOT, "artifacts", "v3_0"))
CONTRACT = os.path.join(HERE, "..", "shared", "contract.json")
FRONTEND = os.path.join(HERE, "..", "frontend")
MODES = ("offline_synthetic", "byok_live", "experiment_results")
START = time.time()

_CONTRACT_CACHE: Optional[dict] = None


def contract() -> dict:
    global _CONTRACT_CACHE
    if _CONTRACT_CACHE is None:
        with open(CONTRACT) as fh:
            _CONTRACT_CACHE = json.load(fh)
    return _CONTRACT_CACHE


# --------------------------------------------------------------------- routes
def r_health() -> Dict[str, Any]:
    """Fast. No model, no artifact scan, no heavyweight import."""
    return {"status": "ok", "uptime_s": round(time.time() - START, 1)}


def r_ready() -> Dict[str, Any]:
    """Cheap readiness: contract present and core importable. Still no models."""
    ok = os.path.exists(CONTRACT)
    return {"ready": ok, "contract_present": ok, "release": PCG_MAS_RELEASE}


def r_version() -> Dict[str, Any]:
    return {**version_payload(), "modes": list(MODES),
            "graph": graph_status(), "providers": provider_status(),
            "langsmith": LangSmithAdapter().status(), "nemo": NeMoAdapter().status(),
            "opa": OPAPolicyBackend().status()}


def r_contract() -> Dict[str, Any]:
    return contract()


def r_run(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the PCG graph. offline_synthetic is the default and needs no key."""
    mode = payload.get("mode", "offline_synthetic")
    if mode not in MODES:
        return {"error": f"unknown mode {mode!r}", "modes": list(MODES)}
    if mode == "byok_live":
        # The key is used for the request lifetime only: never stored, never logged,
        # never echoed back, never placed in telemetry or a URL.
        if not payload.get("api_key"):
            return {"error": "byok_live requires a user-supplied key in the request body"}
        return {"error": "byok_live is not wired in the v3.0 remediation build "
                         "(NETWORK_API_MODEL_CALLS must remain 0)",
                "mode": mode, "key_persisted": False}
    otel_reset()
    scenario = payload.get("scenario", "accept")
    kw = _scenario(scenario)
    st = run_graph(payload.get("request", "Does Policy X permit Action Y?"),
                   run_id=payload.get("run_id", "demo"), **kw)
    return {"mode": mode, "scenario": scenario, "state": st.to_dict(),
            "spans": otel_spans(),
            "provenance_class": "TEST_FIXTURE",
            "banner": "SYNTHETIC DEMO — deterministic fixture, not a DIRECT experimental result"}


def _scenario(name: str) -> Dict[str, Dict[str, Any]]:
    """Deterministic fixtures covering every certificate/audit state."""
    s: Dict[str, Dict[str, Any]] = {}
    if name == "replay_failure":
        s["replay_check"] = {"replay_ok": False}
    elif name == "drift":
        s["replay_check"] = {"replay_ok": True, "drift": True}
    elif name == "checker_failure":
        s["entailment_check"] = {"verdict": False}
    elif name == "policy_failure":
        s["execution_contract"] = {"request": {"actor": "prover", "action": "call",
                                               "tool": "shell"}}
    elif name == "verifier_shared":
        s["execution_contract"] = {"request": {"actor": "prover", "action": "a",
                                               "verifier_context_shared": True}}
    elif name == "dependence_insufficient":
        s["dependence"] = {"dependence": {"state": "INSUFFICIENT_EVIDENCE",
                                          "u_joint": 0.0198,
                                          "note": "evidence floor not met; no extrapolation"}}
    elif name == "high_risk":
        s["controller"] = {"risk": 0.85}
    return s


def r_registry() -> Dict[str, Any]:
    try:
        reg = load_registry(os.path.join(ROOT, "manuscript_artifact_registry.json"))
        return {"check": check_registry(reg), "tables": reg["tables"], "figures": reg["figures"]}
    except Exception as e:
        return {"error": repr(e)}


def r_artifacts() -> Dict[str, Any]:
    """Experiment-results mode reads generated artifacts; it is never a second store."""
    out: Dict[str, Any] = {"root": ARTIFACT_ROOT, "workstreams": {}}
    if not os.path.isdir(ARTIFACT_ROOT):
        return {**out, "available": False}
    for d in sorted(os.listdir(ARTIFACT_ROOT)):
        m = os.path.join(ARTIFACT_ROOT, d, "metrics.json")
        c = os.path.join(ARTIFACT_ROOT, d, "checks.json")
        if os.path.exists(m):
            try:
                out["workstreams"][d] = {
                    "metrics": json.load(open(m)),
                    "checks": json.load(open(c)) if os.path.exists(c) else None}
            except Exception as e:
                out["workstreams"][d] = {"error": repr(e)}
    out["available"] = bool(out["workstreams"])
    return out


def r_policy(payload: Dict[str, Any]) -> Dict[str, Any]:
    backend = LocalPolicyBackend() if payload.get("backend", "local") == "local" else OPAPolicyBackend()
    return backend.evaluate(payload.get("request", {"actor": "prover", "action": "answer"})).to_dict()


ROUTES = {
    ("GET", "/health"): lambda p: r_health(),
    ("GET", "/ready"): lambda p: r_ready(),
    ("GET", "/version"): lambda p: r_version(),
    ("GET", "/api/contract"): lambda p: r_contract(),
    ("GET", "/api/registry"): lambda p: r_registry(),
    ("GET", "/api/artifacts"): lambda p: r_artifacts(),
    ("POST", "/api/v3/run"): r_run,
    ("POST", "/api/v3/policy"): r_policy,
}

# ------------------------------------------------------------------ FastAPI
def create_app():
    try:
        from fastapi import FastAPI, Request
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import JSONResponse, FileResponse
        from fastapi.staticfiles import StaticFiles
    except Exception:
        return None

    app = FastAPI(title="PCG-MAS v3.0 control plane", version=PCG_MAS_RELEASE)
    origins = [o for o in os.environ.get("PCG_CORS_ORIGINS", "").split(",") if o] or \
              ["https://pcg-mas-demo.pages.dev", "http://localhost:5173", "http://localhost:8000"]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"],
                       allow_headers=["content-type"], allow_credentials=False)

    @app.middleware("http")
    async def security_headers(request, call_next):
        resp = await call_next(request)
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["Referrer-Policy"] = "no-referrer"
        resp.headers["PCG-MAS-Release"] = PCG_MAS_RELEASE
        return resp

    for (method, path), fn in ROUTES.items():
        if method == "GET":
            app.add_api_route(path, (lambda _fn=fn: (lambda: _fn(None)))(), methods=["GET"])
        else:
            async def handler(request, _fn=fn):
                return JSONResponse(_fn(await request.json()))
            app.add_api_route(path, handler, methods=["POST"])

    if os.path.isdir(FRONTEND):
        app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")
    return app


# -------------------------------------------------------- stdlib fallback
def serve_stdlib(host: str = "127.0.0.1", port: int = 8000, once: bool = False):
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from urllib.parse import urlparse

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("PCG-MAS-Release", PCG_MAS_RELEASE)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlparse(self.path).path
            fn = ROUTES.get(("GET", path))
            if fn:
                return self._send(fn(None))
            f = os.path.join(FRONTEND, "index.html" if path == "/" else path.lstrip("/"))
            if os.path.isfile(f):
                data = open(f, "rb").read()
                self.send_response(200)
                ct = ("text/html" if f.endswith(".html") else
                      "text/css" if f.endswith(".css") else
                      "application/javascript" if f.endswith(".js") else "application/octet-stream")
                self.send_header("Content-Type", ct)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            self._send({"error": "not found"}, 404)

        def do_POST(self):
            path = urlparse(self.path).path
            fn = ROUTES.get(("POST", path))
            if not fn:
                return self._send({"error": "not found"}, 404)
            n = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(n) or b"{}")
            self._send(fn(payload))

    srv = HTTPServer((host, port), H)
    if once:
        srv.handle_request()
    else:
        srv.serve_forever()
    return srv


app = create_app()

if __name__ == "__main__":
    if app is not None:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
    else:
        serve_stdlib("0.0.0.0", int(os.environ.get("PORT", 8000)))
