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

try:
    from fastapi import Request
except Exception:
    Request = Any

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
from pcg.v3.api.reconstruct import execution_graph                     # noqa: E402
from pcg.v3.api.stream import StreamSession, sse_frame                 # noqa: E402
from pcg.v3.cas.store import ContentAddressedStore                     # noqa: E402
from pcg.v3.lineage import LineageGraph                                # noqa: E402
from pcg.v3.store import AuthoritativeRecordStore                      # noqa: E402
from pcg.v3.store.derived import build_derived                         # noqa: E402
from pcg.v3.store.reconcile import reconcile                           # noqa: E402

ARTIFACT_ROOT = os.environ.get("PCG_ARTIFACT_ROOT", os.path.join(ROOT, "artifacts", "v3_0"))
CAS_ROOT = os.environ.get("PCG_CAS_ROOT", os.path.join(ARTIFACT_ROOT, "cas"))
RECORD_ROOT = os.environ.get("PCG_RECORD_ROOT", os.path.join(ARTIFACT_ROOT, "records"))
DERIVED_ROOT = os.environ.get("PCG_DERIVED_ROOT", os.path.join(ARTIFACT_ROOT, "derived"))
CONTRACT = os.path.join(HERE, "..", "shared", "contract.json")
def _frontend_root() -> str:
    """Prefer a real Vite build; fall back to the dependency-free prebuilt
    bundle; fall back again to the v1 static page. Never silently serve
    nothing."""
    for cand in (os.path.join(HERE, "..", "frontend-v3", "dist"),
                 os.path.join(HERE, "..", "frontend-v3", "prebuilt"),
                 os.path.join(HERE, "..", "frontend")):
        if os.path.isdir(cand) and os.path.exists(os.path.join(cand, "index.html")):
            return os.path.abspath(cand)
    return os.path.abspath(os.path.join(HERE, "..", "frontend"))


FRONTEND = _frontend_root()
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


# ------------------------------------------- unified reconstruction endpoint
def _run_index() -> Dict[str, Dict[str, Any]]:
    """Runs discoverable from the authoritative store, plus their pinned roots."""
    idx: Dict[str, Dict[str, Any]] = {}
    if not os.path.isdir(RECORD_ROOT):
        return idx
    store = AuthoritativeRecordStore(RECORD_ROOT)
    for rec in store.read_all():
        rid = str(rec.get("run_id"))
        e = idx.setdefault(rid, {"run_id": rid, "n_records": 0,
                                 "certificate_roots": set(), "spec_hashes": set(),
                                 "execution_modes": set(), "experiments": set()})
        e["n_records"] += 1
        for key, bucket in (("certificate_root", "certificate_roots"),
                            ("spec_hash", "spec_hashes"),
                            ("execution_mode", "execution_modes"),
                            ("experiment_id", "experiments")):
            if rec.get(key) is not None:
                e[bucket].add(str(rec[key]))
    for e in idx.values():
        for k in ("certificate_roots", "spec_hashes", "execution_modes", "experiments"):
            e[k] = sorted(e[k])
    return idx


def r_runs() -> Dict[str, Any]:
    idx = _run_index()
    return {"n_runs": len(idx), "runs": sorted(idx.values(), key=lambda r: r["run_id"]),
            "record_root_present": os.path.isdir(RECORD_ROOT)}


def r_execution_graph(run_id: str, include_objects: bool = True) -> Dict[str, Any]:
    """GET /api/v3/runs/{run_id}/execution-graph.

    Availability is three-state: a run with no records, or whose certificate
    root is unresolvable, is reported INDETERMINATE rather than returned as a
    partial graph.
    """
    idx = _run_index()
    if run_id not in idx:
        return {"status": "NOT_FOUND", "run_id": run_id,
                "known_runs": sorted(idx), "graph": None}
    meta = idx[run_id]
    store = AuthoritativeRecordStore(RECORD_ROOT)
    records = store.read_all(run_id=run_id)
    roots = meta["certificate_roots"]
    cert_root = roots[0] if len(roots) == 1 else None
    cas = ContentAddressedStore(CAS_ROOT)

    lineage = LineageGraph()
    lp = os.path.join(ARTIFACT_ROOT, "lineage", f"{run_id}.json")
    lineage_status = "ABSENT"
    if os.path.exists(lp):
        try:
            lineage = LineageGraph.from_dict(json.load(open(lp)))
            lineage_status = "LOADED"
        except Exception as exc:
            lineage_status = f"REJECTED: {exc}"

    recon = None
    if os.path.isdir(DERIVED_ROOT):
        try:
            recon = reconcile(records, DERIVED_ROOT, store.manifest()).to_dict()
        except Exception as exc:
            recon = {"status": "UNAVAILABLE", "reasons": [repr(exc)]}

    modes = meta["execution_modes"]
    graph = execution_graph(
        run_id, cert_root, cas,
        spec_hash=(meta["spec_hashes"][0] if len(meta["spec_hashes"]) == 1 else ""),
        execution_mode=(modes[0] if len(modes) == 1 else None),
        lineage=lineage, records=records,
        reconciliation=recon, include_objects=include_objects)
    graph["lineage_source"] = lineage_status
    if len(roots) != 1:
        graph["closure"]["reason"] = (
            "run does not pin exactly one certificate root "
            f"({len(roots)} distinct roots recorded)")
    return {"status": graph["closure"]["status"], "run_id": run_id, "graph": graph}


def r_derived_build(payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Rebuild the derived views and report reconciliation. Never authoritative."""
    if not os.path.isdir(RECORD_ROOT):
        return {"status": "UNAVAILABLE", "reason": "no authoritative record store"}
    store = AuthoritativeRecordStore(RECORD_ROOT)
    records = store.read_all()
    manifest = store.manifest()
    built = build_derived(records, DERIVED_ROOT, manifest)
    rep = reconcile(records, DERIVED_ROOT, manifest)
    return {"authoritative": {"store_root_hash": manifest["store_root_hash"],
                              "n_records": manifest["n_records"]},
            "derived": built.to_dict(), "reconciliation": rep.to_dict(),
            "derived_is_authoritative": False}


def _stream_events(run_id: str, scenario: str = "accept"):
    """Typed SSE frames for one offline_synthetic run. No provider is contacted."""
    import uuid
    session = StreamSession(run_id, uuid.uuid4().hex)
    yield session.emit("stream.open", mode="offline_synthetic",
                       provenance_class="TEST_FIXTURE")
    otel_reset()
    yield session.emit("run.started", scenario=scenario)
    st = run_graph("Does Policy X permit Action Y?", run_id=run_id, **_scenario(scenario))
    d = st.to_dict()
    for span in otel_spans():
        yield session.emit("node.entered", node=span.get("name"),
                           span_id=span.get("span_id"))
        yield session.emit("node.exited", node=span.get("name"),
                           span_id=span.get("span_id"))
    cert = d.get("certificate") or {}
    yield session.emit("policy.decision", v_gamma=cert.get("v_gamma"))
    yield session.emit("checker.decision", v_entail=cert.get("v_entail"))
    for ch in ("int_fail", "replay_fail", "drift_fail", "check_fail", "cov_gap"):
        if d.get("channels", {}).get(ch):
            yield session.emit("channel.fired", channel=ch)
    yield session.emit("certificate.emitted", check=cert.get("check"),
                       certificate_hash=cert.get("certificate_hash"))
    yield session.emit("run.finished", accepted=bool(cert.get("check")))
    yield session.close()


def r_stream_frames(run_id: str = "demo", scenario: str = "accept") -> Dict[str, Any]:
    """Non-SSE view of the same frames, for tests and for clients without SSE."""
    frames = [e.to_dict() for e in _stream_events(run_id, scenario)]
    return {"run_id": run_id, "scenario": scenario, "n_frames": len(frames),
            "frames": frames}


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
    ("GET", "/api/v3/runs"): lambda p: r_runs(),
    ("POST", "/api/v3/run"): r_run,
    ("POST", "/api/v3/policy"): r_policy,
    ("POST", "/api/v3/derived/build"): r_derived_build,
}

#: Path-parameter routes, matched after the exact table. Regex-anchored so a
#: run_id containing a slash cannot escape the intended shape.
import re as _re                                                        # noqa: E402

PATTERN_ROUTES = [
    ("GET", _re.compile(r"^/api/v3/runs/(?P<run_id>[A-Za-z0-9._-]{1,128})/execution-graph$"),
     lambda m, q: r_execution_graph(m.group("run_id"),
                                    include_objects=q.get("objects", ["1"])[0] != "0")),
]

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

    from fastapi.responses import StreamingResponse

    @app.get("/api/v3/runs/{run_id}/execution-graph")
    def _exec_graph(run_id: str, objects: int = 1):          # noqa: ANN202
        return JSONResponse(r_execution_graph(run_id, include_objects=objects != 0))

    @app.get("/api/v3/run/stream")
    def _stream(run_id: str = "demo", scenario: str = "accept"):   # noqa: ANN202
        def gen():
            for ev in _stream_events(run_id, scenario):
                yield sse_frame(ev)
        return StreamingResponse(gen(), media_type="text/event-stream", headers={
            "Cache-Control": "no-store", "X-Accel-Buffering": "no",
            "Connection": "keep-alive"})

    def _make_post_handler(target_fn):
        async def handler(request: Request):
            return JSONResponse(target_fn(await request.json()))
        return handler

    for (method, path), fn in ROUTES.items():
        if method == "GET":
            app.add_api_route(path, (lambda _fn=fn: (lambda: _fn(None)))(), methods=["GET"])
        else:
            app.add_api_route(path, _make_post_handler(fn), methods=["POST"])

    if os.path.isdir(FRONTEND):
        app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")
    return app


# -------------------------------------------------------- stdlib fallback
def serve_stdlib(host: str = "127.0.0.1", port: int = 8000, once: bool = False):
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from urllib.parse import parse_qs, urlparse

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

        def _send_sse(self, run_id, scenario):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            for ev in _stream_events(run_id, scenario):
                self.wfile.write(sse_frame(ev).encode())
                self.wfile.flush()

        def do_GET(self):
            parsed = urlparse(self.path)
            path = parsed.path
            query = parse_qs(parsed.query)
            fn = ROUTES.get(("GET", path))
            if fn:
                return self._send(fn(None))
            if path == "/api/v3/run/stream":
                return self._send_sse(query.get("run_id", ["demo"])[0],
                                      query.get("scenario", ["accept"])[0])
            for method, rx, handler in PATTERN_ROUTES:
                if method != "GET":
                    continue
                m = rx.match(path)
                if m:
                    return self._send(handler(m, query))
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
