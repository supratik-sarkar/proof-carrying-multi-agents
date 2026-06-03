"""FastAPI server for the PCG-MAS static frontend.

Serves the Apple-grade index.html + styles.css + app.js, and exposes
streaming endpoints (Server-Sent Events) that drive the SVG channel
animation from real pcg_glue pipeline events.

Endpoints:
    GET  /                 → static/index.html
    GET  /static/*         → static assets
    POST /api/upload       → multipart file upload, returns server-side path
    POST /api/run          → SSE stream of ChannelEvent + final FinalCertificate
    POST /api/stress       → JSON: list of {attack, decision, rationale}
    POST /api/sidebyside   → JSON: {raw_answer, pcg_answer}
    GET  /api/health       → liveness probe
    GET  /api/comparisons  → curated comparisons manifest (JSON)
    GET  /api/examples     → curated examples manifest (JSON)
    GET  /api/fixture/{name} → serve a fixture file from demo_data/fixtures/

No persistence. API keys arrive in the request body and stay in process
memory for the duration of the call only.
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Make pcg_glue importable
APP_DIR = Path(__file__).parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from pcg_glue.backends import PRESETS, PROVIDER_HINTS, choice_by_label
from pcg_glue.sources import (
    resolve_text, resolve_url, resolve_file,
    SUPPORTED_EXTENSIONS,
)
from pcg_glue.pipeline import (
    run_pcg_mas,                           # NEW master entrypoint
    run_pipeline, run_raw_baseline,        # legacy (used by stress for the answer text)
    ChannelEvent, FinalCertificate,
)
from pcg_glue.schemas import SSEEventType
from pcg_glue.attacks import all_attacks


# ---------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------

STATIC_DIR     = APP_DIR / "static"
DEMO_DATA_DIR  = APP_DIR / "demo_data"
FIXTURES_DIR   = DEMO_DATA_DIR / "fixtures"
EXAMPLES_PATH  = DEMO_DATA_DIR / "examples.json"
COMPARISONS_PATH = DEMO_DATA_DIR / "comparisons.json"

# Server-managed upload directory — cleaned up periodically
UPLOAD_DIR = Path(tempfile.gettempdir()) / "pcg_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="PCG-MAS demo backend",
    description="Streams the four-channel verification pipeline to the static frontend.",
    docs_url="/api/docs",
)

# Serve /static/* from app/static/
# Static assets served at root so index.html's "styles.css" / "app.js" resolve.
# We use a manual route rather than app.mount("/") because mounting at "/"
# would shadow the FastAPI API routes registered below.

@app.get("/styles.css")
async def _styles_css():
    return FileResponse(str(STATIC_DIR / "styles.css"), media_type="text/css")

@app.get("/app.js")
async def _app_js():
    return FileResponse(str(STATIC_DIR / "app.js"), media_type="application/javascript")

@app.get("/favicon.ico")
async def _favicon():
    f = STATIC_DIR / "favicon.ico"
    if f.exists():
        return FileResponse(str(f))
    return JSONResponse({"detail": "no favicon"}, status_code=404)


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class RunRequest(BaseModel):
    mode: str                       # 'text' | 'url' | 'file'
    question: str
    context: Optional[str] = ""
    url: Optional[str] = ""
    file_path: Optional[str] = ""   # server-side path from /api/upload
    backend_label: str
    api_key: str
    replay_check: bool = True
    top_k: int = 6


# ---------------------------------------------------------------------------
# Root + static
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    """Serve the SPA entry point."""
    index = STATIC_DIR / "index.html"
    if not index.exists():
        raise HTTPException(500, "static/index.html missing")
    return FileResponse(str(index))


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "pcg-mas-demo-1.0"}


# ---------------------------------------------------------------------------
# Manifests
# ---------------------------------------------------------------------------

@app.get("/api/examples")
async def examples():
    if not EXAMPLES_PATH.exists():
        return JSONResponse([])
    return JSONResponse(json.loads(EXAMPLES_PATH.read_text()))


@app.get("/api/comparisons")
async def comparisons():
    if not COMPARISONS_PATH.exists():
        return JSONResponse([])
    return JSONResponse(json.loads(COMPARISONS_PATH.read_text()))


@app.get("/api/fixture/{name}")
async def fixture(name: str):
    """Serve a bundled fixture file (used when a curated comparison is loaded)."""
    if "/" in name or ".." in name:
        raise HTTPException(400, "invalid name")
    fp = FIXTURES_DIR / name
    if not fp.exists():
        raise HTTPException(404, f"fixture not found: {name}")
    return FileResponse(str(fp))


@app.get("/api/backends")
async def backends_list():
    """Return the curated backend presets for the dropdown."""
    return JSONResponse([
        {"label": c.name, "provider": c.provider, "model": c.model_id}
        for c in PRESETS
    ])


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    """Accept a file from the browser, save to server-side temp, return path.

    The returned `file_path` is then passed back in the /api/run request body.
    """
    ext = Path(file.filename or "").suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            400,
            f"Unsupported file type: {ext}. Supported: {', '.join(SUPPORTED_EXTENSIONS)}",
        )

    # Random-ish name to avoid collisions, keep the extension
    safe_name = f"upload_{int(time.time() * 1000)}_{Path(file.filename).stem[:40]}{ext}"
    dest = UPLOAD_DIR / safe_name
    contents = await file.read()
    # 25MB hard cap
    if len(contents) > 25 * 1024 * 1024:
        raise HTTPException(413, "File too large (25MB max).")
    dest.write_bytes(contents)
    return {"file_path": str(dest), "size": len(contents), "name": file.filename}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve(req: RunRequest):
    if req.mode == "text":
        return resolve_text(req.question, req.context or "")
    if req.mode == "url":
        return resolve_url(req.question, req.url or "")
    if req.mode == "file":
        if not req.file_path:
            raise ValueError("No file uploaded.")
        return resolve_file(req.question, req.file_path)
    raise ValueError(f"unknown mode: {req.mode}")


def _sse(event_name: str, payload: dict) -> str:
    """Format one Server-Sent Event line."""
    data = json.dumps(payload, default=str)
    return f"event: {event_name}\ndata: {data}\n\n"


# ---------------------------------------------------------------------------
# /api/run — SSE streaming
# ---------------------------------------------------------------------------

@app.post("/api/run")
async def api_run(req: RunRequest):
    """Stream the full PCG-MAS pipeline as SSE.

    Event types streamed to the browser (in order):
      start            - cert_id, n_evidence, backend
      evidence         - parsed evidence items
      claim            - one per atomic claim extracted
      channel          - pending then final per (claim, channel) pair (5 per claim)
      claim_cert       - completed ClaimCertificate per claim
      responsibility   - mask-and-replay report per accepted claim
      audit_envelope   - one per channel (V_I / V_R / V_D / V_Ch / V_Cov)
      risk             - the 4-action RiskDecision
      certificate      - the top-level FullCertificate
      done             - elapsed_ms
      error            - on any pipeline crash
    """
    backend = choice_by_label(req.backend_label)
    if backend is None:
        raise HTTPException(400, f"Unknown backend: {req.backend_label}")
    if not req.api_key:
        raise HTTPException(401, "API key required")

    try:
        resolved = _resolve(req)
    except Exception as e:
        raise HTTPException(400, f"input error: {type(e).__name__}: {e}")

    async def gen():
        try:
            # run_pcg_mas yields {event, data} dicts already shaped for SSE.
            # We forward each one untouched.
            for ev in run_pcg_mas(
                resolved, backend, req.api_key,
                top_k_evidence=int(req.top_k),
                do_responsibility=True,
                do_envelopes=True,
                do_redundancy=True,
            ):
                yield _sse(ev["event"], ev["data"])
                await asyncio.sleep(0)
        except Exception as e:
            yield _sse("error", {
                "type": type(e).__name__,
                "message": str(e),
            })
        finally:
            # Clean up uploaded file if it was in our UPLOAD_DIR
            if req.mode == "file" and req.file_path:
                try:
                    fp = Path(req.file_path)
                    if fp.parent == UPLOAD_DIR and fp.exists():
                        fp.unlink()
                except Exception:
                    pass

    return StreamingResponse(gen(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })


# ---------------------------------------------------------------------------
# /api/stress — 1 ORIGINAL + 8 attack runs, JSON response
# ---------------------------------------------------------------------------

@app.post("/api/stress_stream")
async def api_stress_stream(req: RunRequest):
    """SSE-streaming stress test: emits one 'progress' event per completed
    variant (1 ORIGINAL + 8 attacks = 9 total). Decision is derived from the
    new pipeline's risk action: Answer -> accept, anything else -> reject."""
    backend = choice_by_label(req.backend_label)
    if backend is None:
        raise HTTPException(400, f"Unknown backend: {req.backend_label}")
    if not req.api_key:
        raise HTTPException(401, "API key required")
    try:
        resolved = _resolve(req)
    except Exception as e:
        raise HTTPException(400, f"input error: {type(e).__name__}: {e}")

    cases = [{"name": "ORIGINAL", "description": "Clean input \u2014 baseline run",
              "resolved": resolved}]
    for att in all_attacks(resolved, seed=0):
        cases.append({"name": att.name, "description": att.description,
                      "resolved": att.resolved})

    total = len(cases)

    async def gen():
        yield _sse("start", {"total": total, "provider": backend.provider,
                              "model": backend.model_id})
        for idx, case in enumerate(cases):
            try:
                cert = None
                risk = None
                # Run new pipeline; capture the final certificate + risk event
                for ev in run_pcg_mas(
                    case["resolved"], backend, req.api_key,
                    top_k_evidence=int(req.top_k),
                    do_responsibility=False,   # stress doesn't need responsibility
                    do_envelopes=False,        # nor envelopes
                    do_redundancy=False,
                ):
                    if ev["event"] == "risk":
                        risk = ev["data"]
                    elif ev["event"] == "certificate":
                        cert = ev["data"]
                        break
                if cert is None:
                    row = {"attack": case["name"], "description": case["description"],
                           "decision": "error", "rationale": "(no certificate)",
                           "risk_action": "Refuse"}
                else:
                    risk_action = (risk or {}).get("action", "Refuse")
                    row = {
                        "attack": case["name"],
                        "description": case["description"],
                        "decision": "accept" if cert.get("accepted") else "reject",
                        "risk_action": risk_action,
                        "rationale": (risk or {}).get("summary", "")[:240],
                        "posterior_risk": (risk or {}).get("posterior_risk"),
                        "dominant_failure": (risk or {}).get("dominant_failure_channel"),
                    }
            except Exception as e:
                row = {"attack": case["name"], "description": case["description"],
                       "decision": "error",
                       "rationale": f"{type(e).__name__}: {str(e)[:200]}",
                       "risk_action": "Refuse"}

            yield _sse("progress", {
                "index": idx + 1,
                "total": total,
                "percent": round(((idx + 1) / total) * 100),
                "row": row,
            })
            import asyncio as _aio
            await _aio.sleep(0)
        yield _sse("done", {"total": total})

        # Cleanup uploaded file if any
        if req.mode == "file" and req.file_path:
            try:
                fp = Path(req.file_path)
                if fp.parent == UPLOAD_DIR and fp.exists():
                    fp.unlink()
            except Exception:
                pass

    return StreamingResponse(gen(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })


@app.post("/api/stress")
async def api_stress(req: RunRequest):
    """Non-streaming variant of stress: same logic, single JSON payload at end."""
    backend = choice_by_label(req.backend_label)
    if backend is None:
        raise HTTPException(400, f"Unknown backend: {req.backend_label}")
    if not req.api_key:
        raise HTTPException(401, "API key required")
    try:
        resolved = _resolve(req)
    except Exception as e:
        raise HTTPException(400, f"input error: {type(e).__name__}: {e}")

    cases = [{"name": "ORIGINAL", "description": "Clean input — baseline run",
              "resolved": resolved}]
    for att in all_attacks(resolved, seed=0):
        cases.append({"name": att.name, "description": att.description,
                      "resolved": att.resolved})

    out = []
    for case in cases:
        try:
            cert = None
            risk = None
            for ev in run_pcg_mas(
                case["resolved"], backend, req.api_key,
                top_k_evidence=int(req.top_k),
                do_responsibility=False, do_envelopes=False, do_redundancy=False,
            ):
                if ev["event"] == "risk":
                    risk = ev["data"]
                elif ev["event"] == "certificate":
                    cert = ev["data"]
                    break
            if cert is None:
                out.append({
                    "attack": case["name"], "description": case["description"],
                    "decision": "error", "rationale": "(no certificate)",
                    "risk_action": "Refuse",
                })
            else:
                risk_action = (risk or {}).get("action", "Refuse")
                out.append({
                    "attack": case["name"],
                    "description": case["description"],
                    "decision": "accept" if cert.get("accepted") else "reject",
                    "risk_action": risk_action,
                    "rationale": (risk or {}).get("summary", "")[:240],
                    "posterior_risk": (risk or {}).get("posterior_risk"),
                    "dominant_failure": (risk or {}).get("dominant_failure_channel"),
                })
        except Exception as e:
            out.append({
                "attack": case["name"], "description": case["description"],
                "decision": "error",
                "rationale": f"{type(e).__name__}: {str(e)[:200]}",
                "risk_action": "Refuse",
            })

    # Clean up upload if any
    if req.mode == "file" and req.file_path:
        try:
            fp = Path(req.file_path)
            if fp.parent == UPLOAD_DIR and fp.exists():
                fp.unlink()
        except Exception:
            pass

    return JSONResponse({"rows": out})


# ---------------------------------------------------------------------------
# /api/sidebyside — raw LLM vs PCG-MAS
# ---------------------------------------------------------------------------

@app.post("/api/sidebyside")
async def api_sidebyside(req: RunRequest):
    """Raw LLM vs PCG-MAS side-by-side. PCG side now returns the full
    risk decision + per-channel envelopes so the Results tab can show
    both answers AND the verification overhead in tokens."""
    backend = choice_by_label(req.backend_label)
    if backend is None:
        raise HTTPException(400, f"Unknown backend: {req.backend_label}")
    if not req.api_key:
        raise HTTPException(401, "API key required")
    try:
        resolved = _resolve(req)
    except Exception as e:
        raise HTTPException(400, f"input error: {type(e).__name__}: {e}")

    out = {"raw": None, "pcg": None}

    # Raw baseline
    try:
        raw_text, raw_meta = run_raw_baseline(resolved, backend, req.api_key)
        out["raw"] = {"answer": raw_text, "meta": raw_meta}
    except Exception as e:
        out["raw"] = {"error": f"{type(e).__name__}: {e}"}

    # PCG-MAS via new pipeline
    try:
        cert = None
        risk = None
        envelopes: list = []
        for ev in run_pcg_mas(
            resolved, backend, req.api_key,
            top_k_evidence=int(req.top_k),
            do_responsibility=False,
            do_envelopes=True,
            do_redundancy=True,
        ):
            if ev["event"] == "risk":
                risk = ev["data"]
            elif ev["event"] == "audit_envelope":
                envelopes.append(ev["data"])
            elif ev["event"] == "certificate":
                cert = ev["data"]
                break
        if cert is None:
            out["pcg"] = {"error": "no certificate produced"}
        else:
            # Build compact per-claim summaries for the Results-tab attribution
            # list. Each entry carries everything the frontend needs to render
            # the channel strip and the "supported by <evidence>" line, without
            # shipping the full FullCertificate over the wire.
            claim_certs = cert.get("claim_certificates") or []
            evidence_index = {e.get("id"): e for e in (cert.get("evidence") or [])}
            attribution = []
            for cc in claim_certs:
                ch = cc.get("channels") or {}
                claim = cc.get("claim") or {}
                supports = []
                for eid in (claim.get("support_ids") or []):
                    ev_item = evidence_index.get(eid) or {}
                    supports.append({
                        "id": eid,
                        "source": ev_item.get("source") or "evidence",
                        "snippet": (ev_item.get("text") or "")[:160],
                    })
                attribution.append({
                    "claim_id":   claim.get("claim_id"),
                    "claim_text": claim.get("claim_text"),
                    "accepted":   cc.get("accepted"),
                    "integrity_hash": (cc.get("integrity_hash") or "")[:24],
                    "channels": {
                        ch_name: {"state": (ch.get(ch_name) or {}).get("state", "idle")}
                        for ch_name in ("V_I", "V_R", "V_D", "V_Ch", "V_Cov")
                    },
                    "supports": supports,
                })

            out["pcg"] = {
                "accepted": cert.get("accepted"),
                "answer": cert.get("answer_final") or cert.get("answer_draft"),
                "rationale": (risk or {}).get("summary", ""),
                "risk_action": (risk or {}).get("action"),
                "posterior_risk": (risk or {}).get("posterior_risk"),
                "dominant_failure": (risk or {}).get("dominant_failure_channel"),
                "cert_id": (cert.get("meta") or {}).get("cert_id"),
                "integrity_hash": cert.get("integrity_hash"),
                "tokens_total": (cert.get("meta") or {}).get("tokens_total", 0),
                "envelopes": envelopes,
                "n_claims": len(claim_certs),
                "n_accepted": sum(1 for c in claim_certs if c.get("accepted")),
                "n_evidence": len(evidence_index),
                "attribution": attribution,
            }
    except Exception as e:
        out["pcg"] = {"error": f"{type(e).__name__}: {e}"}

    # Clean up upload if any
    if req.mode == "file" and req.file_path:
        try:
            fp = Path(req.file_path)
            if fp.parent == UPLOAD_DIR and fp.exists():
                fp.unlink()
        except Exception:
            pass

    return JSONResponse(out)


# ---------------------------------------------------------------------------
# CLI launch (local dev)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "7860")),
        reload=False,
        log_level="info",
    )
