"""Framework-independent API payload builders.

Every endpoint body is produced by a pure function here, so the reconstruction
and streaming contracts are testable without FastAPI, uvicorn or a socket.
"""
from .reconstruct import ExecutionGraphError, execution_graph
from .stream import (SSE_EVENT_TYPES, StreamEvent, SequenceViolation,
                     StreamSession, sse_frame)

__all__ = ["execution_graph", "ExecutionGraphError", "StreamEvent",
           "StreamSession", "SequenceViolation", "sse_frame",
           "SSE_EVENT_TYPES"]
