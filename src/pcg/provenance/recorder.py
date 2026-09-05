"""Immutable recorder + execution bundles.

Storage separation (item 8): three tiers, never mixed.

  code            the repository itself — versioned, small
  run artifacts   runs/<run_id>/ — bundles + records; LARGE, git-ignored,
                  referenced by hash from publication aggregates
  aggregates      results/ — small, publication-facing, carries source hashes

A bundle is the immutable per-execution evidence unit. It holds the canonical
input, raw output, evidence references, tool outputs, certificate and checker
output, so any auditor can recompute every hash on the record.
"""
from __future__ import annotations

import json
import os
import platform
import socket
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .classify import classify
from .errors import sanitize
from .hashing import hash_obj, hash_text
from .schema import SCHEMA_ID

BUNDLE_VERSION = "pcg-bundle/1"
GITIGNORE = "# Run artifacts: large, hash-referenced from results/. Never committed.\n*\n"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def env_fingerprint() -> str:
    return hash_obj({"python": sys.version.split()[0],
                     "platform": platform.platform(), "machine": platform.machine()})


class RunRecorder:
    def __init__(self, root: Path, experiment_id: str, run_id: str | None = None,
                 code_fingerprint: str | None = None, git_commit: str | None = None):
        self.run_id = run_id or f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
        self.experiment_id = experiment_id
        self.root = Path(root)
        self.dir = self.root / self.run_id
        self.bundles = self.dir / "bundles"
        self.records_path = self.dir / "records.jsonl"
        self.manifest_path = self.dir / "run_manifest.json"
        self.bundles.mkdir(parents=True, exist_ok=True)
        (self.root / ".gitignore").write_text(GITIGNORE, encoding="utf-8")
        self.code_fingerprint = code_fingerprint
        self.git_commit = git_commit
        self._seen: set[str] = {r["record_id"] for r in self.iter_records()}

    # -- resume --------------------------------------------------------------
    def already_done(self, record_id: str) -> bool:
        return record_id in self._seen

    def iter_records(self) -> Iterator[dict]:
        if not self.records_path.exists():
            return
        with self.records_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue          # a torn final line from a hard kill is skipped

    # -- bundle --------------------------------------------------------------
    def write_bundle(self, record_id: str, *, canonical_input: str, raw_output: str | None,
                     evidence_refs: list[dict] | None = None,
                     tool_outputs: list[dict] | None = None,
                     provider_metadata: dict | None = None,
                     certificate: dict | None = None,
                     checker_output: dict | None = None,
                     execution_trace: list[dict] | None = None) -> tuple[str, str]:
        """Write the immutable evidence bundle. Returns (relative_ref, bundle_hash)."""
        bundle = {
            "bundle_version": BUNDLE_VERSION,
            "record_id": record_id,
            "canonical_input": canonical_input,
            "input_hash": hash_text(canonical_input),
            "raw_output": raw_output,
            "output_hash": hash_text(raw_output) if raw_output is not None else None,
            "evidence_refs": evidence_refs or [],
            "tool_outputs": tool_outputs or [],
            "provider_metadata": provider_metadata or {},   # credential-free by contract
            "certificate": certificate,
            "checker_output": checker_output,
            "execution_trace": execution_trace or [],
        }
        p = self.bundles / f"{record_id}.json"
        text = json.dumps(bundle, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
        self._atomic_write(p, text)
        return str(p.relative_to(self.root)), hash_obj(bundle)

    # -- append --------------------------------------------------------------
    def append(self, rec: dict) -> dict:
        for forbidden in ("provenance_class", "execution_class", "outcome_eligible",
                          "usage_completeness"):
            if rec.get(forbidden) is not None:
                raise ValueError(
                    f"'{forbidden}' is derived from evidence and must not be supplied by a caller."
                )
        rec = dict(rec)
        rec["schema_id"] = SCHEMA_ID
        rec["run_id"] = self.run_id
        rec.setdefault("code_fingerprint", self.code_fingerprint)
        rec.setdefault("git_commit", self.git_commit)
        rec.setdefault("env_fingerprint", env_fingerprint())
        if rec.get("error_message"):
            rec["error_message"] = sanitize(rec["error_message"])

        rec.update(classify(rec))

        rid = rec["record_id"]
        if rid in self._seen:
            return rec                                   # immutable
        with self.records_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._seen.add(rid)
        return rec

    # -- manifest ------------------------------------------------------------
    def write_manifest(self, extra: dict[str, Any] | None = None) -> dict:
        counts: dict[str, int] = {}
        ids: list[str] = []
        for r in self.iter_records():
            counts[r["provenance_class"]] = counts.get(r["provenance_class"], 0) + 1
            ids.append(r["record_id"])
        man = {
            "schema_id": SCHEMA_ID, "run_id": self.run_id, "experiment_id": self.experiment_id,
            "written_at": utc_now(), "host": socket.gethostname(),
            "code_fingerprint": self.code_fingerprint, "git_commit": self.git_commit,
            "env_fingerprint": env_fingerprint(), "record_count": len(ids),
            "provenance_counts": counts,
            "record_set_hash": hash_obj(sorted(ids)),
            "records_path": self.records_path.name, "bundles_dir": self.bundles.name,
        }
        if extra:
            man.update(extra)
        self._atomic_write(self.manifest_path, json.dumps(man, indent=2) + "\n")
        return man

    @staticmethod
    def _atomic_write(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text); f.flush(); os.fsync(f.fileno())
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
