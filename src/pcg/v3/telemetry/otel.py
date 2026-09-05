"""OpenTelemetry-compatible local instrumentation with a stdlib fallback.

The SAME telemetry feeds A10 cost/latency analysis and the demo UI: there is not
a separate scientific timing system and UI timing system.
"""
from __future__ import annotations
import contextlib, json, os, threading, time, uuid
from typing import Any, Dict, List, Optional

try:
    from opentelemetry import trace as _otel_trace  # type: ignore
    _TRACER = _otel_trace.get_tracer("pcg-mas")
    HAVE_OTEL = True
except Exception:
    _TRACER = None
    HAVE_OTEL = False

_LOCAL: List[Dict[str, Any]] = []
_LOCK = threading.Lock()
_CORR = threading.local()


def correlation_id() -> str:
    if not getattr(_CORR, "value", None):
        _CORR.value = uuid.uuid4().hex
    return _CORR.value


def set_correlation_id(v: str) -> None:
    _CORR.value = v


@contextlib.contextmanager
def span(name: str, **attrs):
    t0 = time.perf_counter()
    rec = {"name": name, "correlation_id": correlation_id(),
           "start_unix_ms": round(time.time() * 1000, 1), "attributes": dict(attrs)}
    if HAVE_OTEL and _TRACER is not None:
        with _TRACER.start_as_current_span(name) as s:
            for k, v in attrs.items():
                s.set_attribute(k, v)
            try:
                yield rec
            except Exception as e:
                rec["error"] = repr(e)
                raise
            finally:
                rec["duration_ms"] = round((time.perf_counter() - t0) * 1000, 4)
                _emit(rec)
    else:
        try:
            yield rec
        except Exception as e:
            rec["error"] = repr(e)
            raise
        finally:
            rec["duration_ms"] = round((time.perf_counter() - t0) * 1000, 4)
            _emit(rec)


def _emit(rec: Dict[str, Any]) -> None:
    with _LOCK:
        _LOCAL.append(rec)


def spans() -> List[Dict[str, Any]]:
    with _LOCK:
        return list(_LOCAL)


def reset() -> None:
    with _LOCK:
        _LOCAL.clear()


def export(path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        for r in spans():
            fh.write(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n")
    return path


def status() -> Dict[str, Any]:
    return {"otel_sdk_installed": HAVE_OTEL, "local_exporter": True,
            "spans_captured": len(spans())}
