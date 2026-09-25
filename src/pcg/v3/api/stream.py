"""Typed Server-Sent Events stream with monotonic sequence numbers.

An untyped stream lets a dropped or reordered frame pass unnoticed; a reviewer
watching a live run then sees a plausible but wrong story. This module makes
that detectable on the client:

* every frame carries ``seq``, strictly increasing by one from 0;
* every frame declares an ``event`` drawn from a closed vocabulary;
* every frame carries the ``run_id`` and the emitting session's ``stream_id``,
  so two interleaved runs cannot be spliced;
* the terminal frame carries a ``digest`` over all preceding frames, so a
  client can verify it received the whole stream, in order.

Nothing here writes to a socket. `StreamSession` produces frames; the server
adapter is responsible for transport.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional

from ..canon import canonical_json, sha256_text

#: Closed event vocabulary. A frame with any other type is rejected at emit
#: time rather than being forwarded for the client to puzzle over.
SSE_EVENT_TYPES = (
    "stream.open",
    "run.started",
    "node.entered",
    "node.exited",
    "provider.request",
    "provider.response",
    "retrieval.result",
    "tool.result",
    "guardrail.intervention",
    "policy.decision",
    "checker.decision",
    "channel.fired",
    "certificate.emitted",
    "record.appended",
    "warning",
    "error",
    "run.finished",
    "stream.closed",
)

TERMINAL_EVENTS = ("stream.closed",)


class SequenceViolation(RuntimeError):
    """A consumed stream was not gap-free, ordered and terminated."""


@dataclass(frozen=True)
class StreamEvent:
    seq: int
    event: str
    run_id: str
    stream_id: str
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"seq": self.seq, "event": self.event, "run_id": self.run_id,
                "stream_id": self.stream_id, "data": self.data}


def sse_frame(ev: StreamEvent) -> str:
    """Render one frame in SSE wire format (id/event/data, blank-line framed)."""
    body = canonical_json(ev.to_dict())
    return f"id: {ev.seq}\nevent: {ev.event}\ndata: {body}\n\n"


class StreamSession:
    """Emits typed frames with a monotonic, gap-free sequence."""

    def __init__(self, run_id: str, stream_id: str):
        if not run_id or not stream_id:
            raise ValueError("run_id and stream_id are required")
        self.run_id = run_id
        self.stream_id = stream_id
        self._seq = 0
        self._h = hashlib.sha256()
        self._closed = False

    @property
    def next_seq(self) -> int:
        return self._seq

    def emit(self, event: str, **data: Any) -> StreamEvent:
        if self._closed:
            raise SequenceViolation("stream already closed")
        if event not in SSE_EVENT_TYPES:
            raise SequenceViolation(f"unknown event type: {event!r}")
        ev = StreamEvent(self._seq, event, self.run_id, self.stream_id, dict(data))
        self._h.update(canonical_json(ev.to_dict()).encode("utf-8"))
        self._seq += 1
        if event in TERMINAL_EVENTS:
            self._closed = True
        return ev

    def digest(self) -> str:
        return self._h.hexdigest()

    def close(self, **data: Any) -> StreamEvent:
        """Terminal frame; carries the running digest of everything before it."""
        payload = dict(data)
        payload["frames_before_close"] = self._seq
        payload["digest"] = self.digest()
        return self.emit("stream.closed", **payload)


def verify_stream(frames: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """Client-side check: ordered, gap-free, single-run, terminated, digest ok."""
    h = hashlib.sha256()
    expect = 0
    run_ids, stream_ids = set(), set()
    terminal: Optional[Mapping[str, Any]] = None
    for f in frames:
        if terminal is not None:
            raise SequenceViolation("frame after terminal frame")
        if f.get("seq") != expect:
            raise SequenceViolation(f"expected seq {expect}, got {f.get('seq')!r}")
        if f.get("event") not in SSE_EVENT_TYPES:
            raise SequenceViolation(f"unknown event type: {f.get('event')!r}")
        run_ids.add(f.get("run_id"))
        stream_ids.add(f.get("stream_id"))
        if len(run_ids) > 1 or len(stream_ids) > 1:
            raise SequenceViolation("frames from more than one run or stream")
        if f.get("event") in TERMINAL_EVENTS:
            terminal = f
        else:
            h.update(canonical_json(dict(f)).encode("utf-8"))
        expect += 1
    if terminal is None:
        raise SequenceViolation("stream not terminated")
    declared = (terminal.get("data") or {}).get("digest")
    if declared != h.hexdigest():
        raise SequenceViolation("stream digest mismatch: frames were altered or lost")
    return {"n_frames": expect, "run_id": next(iter(run_ids), None),
            "stream_id": next(iter(stream_ids), None), "digest": declared,
            "status": "COMPLETE"}


def frames_to_sse(events: Iterable[StreamEvent]) -> Iterator[str]:
    for ev in events:
        yield sse_frame(ev)
