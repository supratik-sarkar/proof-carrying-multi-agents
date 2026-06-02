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
    run_pipeline, run_raw_baseline,
    ChannelEvent, FinalCertificate,
)
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
    Streams the upload chunk-by-chunk so multi-GB files don't exhaust memory.
    """
    ext = Path(file.filename or "").suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            400,
            f"Unsupported file type: {ext}. Supported: {', '.join(SUPPORTED_EXTENSIONS)}",
        )

    # 10 GB hard cap (matches the UI message)
    MAX_BYTES = 10 * 1024 * 1024 * 1024
    CHUNK = 1024 * 1024  # 1 MB read window

    safe_name = f"upload_{int(time.time() * 1000)}_{Path(file.filename).stem[:40]}{ext}"
    dest = UPLOAD_DIR / safe_name

    total = 0
    try:
        with open(dest, "wb") as out:
            while True:
                chunk = await file.read(CHUNK)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_BYTES:
                    out.close()
                    try:
                        dest.unlink()
                    except Exception:
                        pass
                    raise HTTPException(413, "File too large (10 GB max).")
                out.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        try:
            if dest.exists():
                dest.unlink()
        except Exception:
            pass
        raise HTTPException(500, f"upload write failed: {type(e).__name__}: {e}")

    return {"file_path": str(dest), "size": total, "name": file.filename}


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
    """Stream channel events + final certificate over Server-Sent Events.

    The browser consumes this with `fetch().then(r => r.body.getReader())`
    and dispatches each event into the SVG state machine.
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
        # 1. Emit a 'start' frame
        yield _sse("start", {
            "claim": resolved.claim,
            "source_kind": resolved.source_kind,
            "source_label": resolved.source_label,
            "n_evidence": len(resolved.evidence),
        })

        try:
            # The pipeline is a sync generator; we marshal it into async chunks.
            # Each yield from run_pipeline becomes one SSE event.
            for ev in run_pipeline(
                resolved, backend, req.api_key,
                replay_check=req.replay_check,
                top_k_evidence=int(req.top_k),
            ):
                if isinstance(ev, ChannelEvent):
                    yield _sse("channel", {
                        "channel": ev.channel,
                        "state": ev.state,
                        "verdict": ev.verdict,
                        "detail": ev.detail,
                        "elapsed_ms": ev.elapsed_ms,
                    })
                elif isinstance(ev, FinalCertificate):
                    yield _sse("certificate", {
                        "accepted": ev.accepted,
                        "claim": ev.claim,
                        "answer": ev.answer,
                        "channels": ev.channels,
                        "backend": ev.backend,
                        "source": ev.source,
                        "integrity_hash": ev.integrity_hash,
                        "cert_id": ev.cert_id,
                        "timestamp": ev.timestamp,
                    })
                # Yield control so the browser can render
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
    variant (1 ORIGINAL + 8 attacks = 9 total). The browser shows a live
    progress bar + appends rows as they arrive."""
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
                final = None
                for ev in run_pipeline(
                    case["resolved"], backend, req.api_key,
                    replay_check=False, top_k_evidence=int(req.top_k),
                ):
                    if isinstance(ev, FinalCertificate):
                        final = ev
                        break
                if final is None:
                    row = {"attack": case["name"], "description": case["description"],
                           "decision": "error", "rationale": "(no certificate)"}
                else:
                    row = {
                        "attack": case["name"],
                        "description": case["description"],
                        "decision": "accept" if final.accepted else "reject",
                        "rationale": final.channels.get("V_entail", {})
                                                  .get("rationale", "")[:240],
                        "judge_score": (final.channels.get("V_gamma", {}) or {})
                                                     .get("score", None),
                    }
            except Exception as e:
                row = {"attack": case["name"], "description": case["description"],
                       "decision": "error",
                       "rationale": f"{type(e).__name__}: {str(e)[:200]}"}

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
            final = None
            for ev in run_pipeline(
                case["resolved"], backend, req.api_key,
                replay_check=False, top_k_evidence=int(req.top_k),
            ):
                if isinstance(ev, FinalCertificate):
                    final = ev
                    break
            if final is None:
                out.append({
                    "attack": case["name"], "description": case["description"],
                    "decision": "error", "rationale": "(no certificate)",
                })
            else:
                out.append({
                    "attack": case["name"],
                    "description": case["description"],
                    "decision": "accept" if final.accepted else "reject",
                    "rationale": final.channels.get("V_entail", {})
                                              .get("rationale", "")[:240],
                    "judge_score": (final.channels.get("V_gamma", {}) or {})
                                                 .get("score", None),
                })
        except Exception as e:
            out.append({
                "attack": case["name"], "description": case["description"],
                "decision": "error",
                "rationale": f"{type(e).__name__}: {str(e)[:200]}",
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
        out["raw"] = {
            "answer": raw_text,
            "meta": raw_meta,
        }
    except Exception as e:
        out["raw"] = {"error": f"{type(e).__name__}: {e}"}

    # PCG-MAS
    try:
        final = None
        for ev in run_pipeline(
            resolved, backend, req.api_key,
            replay_check=False, top_k_evidence=int(req.top_k),
        ):
            if isinstance(ev, FinalCertificate):
                final = ev
                break
        if final is None:
            out["pcg"] = {"error": "no certificate produced"}
        else:
            out["pcg"] = {
                "accepted": final.accepted,
                "answer": final.answer,
                "rationale": final.channels.get("V_entail", {}).get("rationale", ""),
                "cert_id": final.cert_id,
                "integrity_hash": final.integrity_hash,
                "judge_score": (final.channels.get("V_gamma", {}) or {}).get("score"),
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
