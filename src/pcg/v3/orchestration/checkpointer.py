"""Durable checkpointing (D6).

SQLite is the REQUIRED local/reproducible backend and is stdlib, so scientific
reproduction never needs a server. PostgreSQL is an optional hosted backend
behind the same interface and is never required for reproduction.

Persists checkpoint id, parent id, thread id, node, state hash, writes, metadata
and timestamp, supporting resume, history enumeration, time-travel inspection
and deterministic branch forks for replay intervention.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Protocol

from ..canon import canonical_json, sha256_obj


@dataclass
class Checkpoint:
    checkpoint_id: str
    thread_id: str
    parent_id: Optional[str]
    node: str
    state_hash: str
    state: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    ts_ms: float = 0.0
    kind: str = "step"                    # step | fork | replay

    def to_dict(self) -> dict:
        return vars(self)


class CheckpointBackend(Protocol):
    def has_thread(self, thread_id: str) -> bool: ...
    def put(self, cp: Checkpoint) -> str: ...
    def get(self, checkpoint_id: str) -> Optional[Checkpoint]: ...
    def history(self, thread_id: str) -> List[Checkpoint]: ...
    def latest(self, thread_id: str) -> Optional[Checkpoint]: ...


class SQLiteCheckpointer:
    """Required local backend. Durable, single-file, no server."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS checkpoints (
      checkpoint_id TEXT PRIMARY KEY,
      thread_id     TEXT NOT NULL,
      parent_id     TEXT,
      node          TEXT NOT NULL,
      state_hash    TEXT NOT NULL,
      state_json    TEXT NOT NULL,
      metadata_json TEXT NOT NULL,
      ts_ms         REAL NOT NULL,
      kind          TEXT NOT NULL,
      seq           INTEGER
    );
    CREATE INDEX IF NOT EXISTS ix_thread ON checkpoints(thread_id, seq);
    """

    def __init__(self, path: str = ":memory:"):
        self.path = path
        if path != ":memory:":
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._con = sqlite3.connect(path, check_same_thread=False)
        self._con.execute("PRAGMA journal_mode=WAL")
        self._con.executescript(self.SCHEMA)
        self._con.commit()

    def has_thread(self, thread_id: str) -> bool:
        cur = self._con.execute("SELECT 1 FROM checkpoints WHERE thread_id=? LIMIT 1",
                                (thread_id,))
        return cur.fetchone() is not None

    def put(self, cp: Checkpoint) -> str:
        cp.state_hash = sha256_obj(cp.state)
        if not cp.checkpoint_id:
            cp.checkpoint_id = "ck-" + sha256_obj(
                {"t": cp.thread_id, "p": cp.parent_id, "n": cp.node,
                 "s": cp.state_hash, "k": cp.kind})[:24]
        cp.ts_ms = cp.ts_ms or time.time() * 1000.0
        seq = self._con.execute(
            "SELECT COALESCE(MAX(seq),-1)+1 FROM checkpoints WHERE thread_id=?",
            (cp.thread_id,)).fetchone()[0]
        self._con.execute(
            "INSERT OR REPLACE INTO checkpoints VALUES (?,?,?,?,?,?,?,?,?,?)",
            (cp.checkpoint_id, cp.thread_id, cp.parent_id, cp.node, cp.state_hash,
             canonical_json(cp.state), canonical_json(cp.metadata), cp.ts_ms,
             cp.kind, seq))
        self._con.commit()
        return cp.checkpoint_id

    def _row(self, r) -> Checkpoint:
        return Checkpoint(r[0], r[1], r[2], r[3], r[4], json.loads(r[5]),
                          json.loads(r[6]), r[7], r[8])

    def get(self, checkpoint_id: str) -> Optional[Checkpoint]:
        r = self._con.execute(
            "SELECT checkpoint_id,thread_id,parent_id,node,state_hash,state_json,"
            "metadata_json,ts_ms,kind FROM checkpoints WHERE checkpoint_id=?",
            (checkpoint_id,)).fetchone()
        return self._row(r) if r else None

    def history(self, thread_id: str) -> List[Checkpoint]:
        rows = self._con.execute(
            "SELECT checkpoint_id,thread_id,parent_id,node,state_hash,state_json,"
            "metadata_json,ts_ms,kind FROM checkpoints WHERE thread_id=? ORDER BY seq",
            (thread_id,)).fetchall()
        return [self._row(r) for r in rows]

    def latest(self, thread_id: str) -> Optional[Checkpoint]:
        h = self.history(thread_id)
        return h[-1] if h else None

    # --------------------------------------------------------------- replay
    def fork(self, source_checkpoint_id: str, new_thread_id: str,
             intervention: Dict[str, Any]) -> Checkpoint:
        """Deterministic branch fork for replay intervention."""
        src = self.get(source_checkpoint_id)
        if src is None:
            raise KeyError(f"unknown source checkpoint {source_checkpoint_id}")
        state = dict(src.state)
        state.update(intervention.get("state_overrides", {}))
        cp = Checkpoint(
            checkpoint_id="", thread_id=new_thread_id, parent_id=src.checkpoint_id,
            node=src.node, state_hash="", state=state, kind="replay",
            metadata={"source_checkpoint_id": src.checkpoint_id,
                      "intervention_spec_hash": sha256_obj(intervention),
                      "source_span_id": src.metadata.get("span_id")})
        self.put(cp)
        return cp

    def lineage(self, thread_id: str) -> List[Dict[str, Any]]:
        """Artifact-only lineage rows for `reconstruct_replay`."""
        out = []
        for cp in self.history(thread_id):
            row = {"kind": cp.kind, "checkpoint_id": cp.checkpoint_id,
                   "parent_id": cp.parent_id, "node": cp.node,
                   "state_hash": cp.state_hash}
            if cp.kind == "replay":
                row.update({
                    "source_checkpoint_id": cp.metadata.get("source_checkpoint_id"),
                    "source_span_id": cp.metadata.get("source_span_id") or "",
                    "intervention_spec_hash": cp.metadata.get("intervention_spec_hash"),
                    "result_checkpoint_id": cp.checkpoint_id,
                    "replay_trace_id": cp.metadata.get("replay_trace_id", ""),
                })
            out.append(row)
        return out


class PostgresCheckpointer:
    """Optional hosted backend. NEVER required for scientific reproduction."""

    def __init__(self, dsn: str):
        self.dsn = dsn
        self._con = None

    def _connect(self):
        if self._con is None:
            import psycopg  # type: ignore  # lazy: absent by default
            self._con = psycopg.connect(self.dsn)
        return self._con

    @staticmethod
    def available() -> bool:
        try:
            import psycopg  # noqa: F401
            return True
        except Exception:
            return False
