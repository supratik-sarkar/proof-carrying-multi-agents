"""Authoritative append-only JSONL record store.

Every record is written as a self-describing envelope carrying

  * ``seq``               monotonic, gap-free within a segment;
  * ``record_addr``       PCG-CAS-v1 address over the *committed* fields only,
                          so telemetry can never move a record's identity;
  * ``prev_record_hash``  hash of the preceding envelope (genesis = 64 zeros);
  * ``record_hash``       hash of this envelope, chaining the segment.

The chain makes truncation and mid-file edits detectable: replaying the file
recomputes every link, and a segment manifest pins the final link plus the
record count. A derived store (Parquet/DuckDB) is only ever admissible if it
reconciles against this file.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional

from ..canon import canonical_json, sha256_text
from ..cas.address import ObjType, addr
from ..cas.fields import assert_no_observed, split_fields

GENESIS = "0" * 64
ENVELOPE_VERSION = "PCG-JSONL-v1"


class ChainBroken(RuntimeError):
    """The stored hash chain does not reproduce on replay."""


def _envelope_hash(env: Mapping[str, Any]) -> str:
    body = {k: v for k, v in env.items() if k != "record_hash"}
    return sha256_text(canonical_json(body))


@dataclass(frozen=True)
class SegmentManifest:
    """Pins a JSONL segment: anything less would let a truncation pass."""
    path: str
    envelope_version: str
    n_records: int
    head_hash: str
    tail_hash: str
    file_sha256: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class JsonlSegment:
    """One append-only file. Segments are never rewritten, only appended."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self._tail = GENESIS
        self._n = 0
        self._head = GENESIS
        if os.path.exists(path):
            self._replay_state()

    # ---------------------------------------------------------------- writing
    def append(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        committed, observed = split_fields(payload)
        assert_no_observed(committed)
        a = addr(ObjType.PER_EXAMPLE_RECORD, committed)
        env = {
            "envelope_version": ENVELOPE_VERSION,
            "seq": self._n,
            "record_addr": str(a),
            "prev_record_hash": self._tail,
            "committed": committed,
            "observed": observed,
        }
        env["record_hash"] = _envelope_hash(env)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(canonical_json(env) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        if self._n == 0:
            self._head = env["record_hash"]
        self._tail = env["record_hash"]
        self._n += 1
        return env

    def extend(self, payloads: Iterable[Mapping[str, Any]]) -> int:
        n = 0
        for p in payloads:
            self.append(p)
            n += 1
        return n

    # ---------------------------------------------------------------- reading
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        if not os.path.exists(self.path):
            return
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def verify(self) -> SegmentManifest:
        """Replay the chain. Raises ChainBroken on the first divergence."""
        prev, n, head = GENESIS, 0, GENESIS
        for env in self:
            if env.get("seq") != n:
                raise ChainBroken(f"{self.path}: seq {env.get('seq')!r} at position {n}")
            if env.get("prev_record_hash") != prev:
                raise ChainBroken(f"{self.path}: broken link at seq {n}")
            if _envelope_hash(env) != env.get("record_hash"):
                raise ChainBroken(f"{self.path}: envelope hash mismatch at seq {n}")
            expect = str(addr(ObjType.PER_EXAMPLE_RECORD, env.get("committed", {})))
            if expect != env.get("record_addr"):
                raise ChainBroken(f"{self.path}: record_addr mismatch at seq {n}")
            prev = env["record_hash"]
            if n == 0:
                head = prev
            n += 1
        self._tail, self._n, self._head = prev, n, head
        return SegmentManifest(
            path=os.path.basename(self.path),
            envelope_version=ENVELOPE_VERSION,
            n_records=n,
            head_hash=head,
            tail_hash=prev,
            file_sha256=self._file_sha256(),
        )

    def _replay_state(self) -> None:
        m = self.verify()
        self._tail, self._n, self._head = m.tail_hash, m.n_records, m.head_hash

    def _file_sha256(self) -> str:
        h = hashlib.sha256()
        if os.path.exists(self.path):
            with open(self.path, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
        return h.hexdigest()

    @property
    def n_records(self) -> int:
        return self._n

    @property
    def tail_hash(self) -> str:
        return self._tail


class AuthoritativeRecordStore:
    """A directory of JSONL segments, one per (run_id, experiment_id)."""

    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self._open: Dict[str, JsonlSegment] = {}

    def _segment_path(self, run_id: str, experiment_id: str) -> str:
        return os.path.join(self.root, run_id, f"{experiment_id}.jsonl")

    def segment(self, run_id: str, experiment_id: str) -> JsonlSegment:
        key = f"{run_id}/{experiment_id}"
        if key not in self._open:
            self._open[key] = JsonlSegment(self._segment_path(run_id, experiment_id))
        return self._open[key]

    def append(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        run_id = payload.get("run_id")
        experiment_id = payload.get("experiment_id")
        if not run_id or not experiment_id:
            raise ValueError("record requires run_id and experiment_id")
        return self.segment(str(run_id), str(experiment_id)).append(payload)

    def segments(self) -> List[str]:
        out: List[str] = []
        for dirpath, _dirs, files in os.walk(self.root):
            for f in sorted(files):
                if f.endswith(".jsonl"):
                    out.append(os.path.join(dirpath, f))
        return sorted(out)

    def read_all(self, run_id: Optional[str] = None,
                 experiment_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return committed+observed merged dicts, in authoritative order."""
        out: List[Dict[str, Any]] = []
        for path in self.segments():
            rel = os.path.relpath(path, self.root)
            seg_run = rel.split(os.sep)[0]
            seg_exp = os.path.splitext(os.path.basename(path))[0]
            if run_id is not None and seg_run != run_id:
                continue
            if experiment_id is not None and seg_exp != experiment_id:
                continue
            for env in JsonlSegment(path):
                merged = dict(env.get("committed", {}))
                merged.update(env.get("observed", {}))
                merged["_record_addr"] = env.get("record_addr")
                merged["_record_hash"] = env.get("record_hash")
                out.append(merged)
        return out

    def verify_all(self) -> List[SegmentManifest]:
        return [JsonlSegment(p).verify() for p in self.segments()]

    def manifest(self) -> Dict[str, Any]:
        """Store-level manifest: the object a derived view must hash-link to."""
        segs = [m.to_dict() for m in self.verify_all()]
        return {
            "envelope_version": ENVELOPE_VERSION,
            "n_segments": len(segs),
            "n_records": sum(s["n_records"] for s in segs),
            "segments": segs,
            "store_root_hash": sha256_text(canonical_json(segs)),
        }
