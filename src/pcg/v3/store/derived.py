"""Derived Parquet / DuckDB views.

These are *convenience* views for interactive analysis. They carry a manifest
that hash-links each artifact to the authoritative store manifest and to the
projection digest, so a reviewer can tell -- without trusting the writer --
whether a Parquet file corresponds to the JSONL it claims to summarise.

pyarrow and duckdb are imported lazily. When they are absent the builder reports
``BLOCKED_MISSING_DEPENDENCY`` and writes nothing; it never fabricates a
manifest for an artifact it did not produce.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..canon import canonical_json, sha256_file, sha256_text
from .projection import (PROJECTION_COLUMNS, project_record,
                         projection_digest, row_digest)

DERIVED_MANIFEST = "derived_manifest.json"


class DerivedStatus(str):
    BUILT = "BUILT"
    BLOCKED_MISSING_DEPENDENCY = "BLOCKED_MISSING_DEPENDENCY"
    FAILED = "FAILED"


@dataclass
class DerivedArtifact:
    kind: str                     # "parquet" | "duckdb"
    path: Optional[str]
    sha256: Optional[str]
    status: str
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DerivedBuildResult:
    status: str
    artifacts: List[DerivedArtifact] = field(default_factory=list)
    manifest_path: Optional[str] = None
    manifest: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["artifacts"] = [a.to_dict() for a in self.artifacts]
        return d


def _import(name: str):
    try:
        return __import__(name)
    except Exception:                                    # pragma: no cover
        return None


def build_derived(records: Sequence[Mapping[str, Any]],
                  out_dir: str,
                  store_manifest: Mapping[str, Any]) -> DerivedBuildResult:
    """Write parquet + duckdb views and a hash-linked manifest."""
    os.makedirs(out_dir, exist_ok=True)
    rows = [project_record(r) for r in records]
    proj = projection_digest(records)
    artifacts: List[DerivedArtifact] = []

    # Pure-Python derived view. It has no optional dependency, so the
    # reconciliation gate is exercisable in any environment; parquet and duckdb
    # remain the performance views.
    try:
        np_ = os.path.join(out_dir, "records.ndjson")
        with open(np_, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(canonical_json(r) + "\n")
        artifacts.append(DerivedArtifact("ndjson", os.path.basename(np_),
                                         sha256_file(np_), DerivedStatus.BUILT))
    except Exception as exc:                              # pragma: no cover
        artifacts.append(DerivedArtifact("ndjson", None, None,
                                         DerivedStatus.FAILED, repr(exc)))

    pa = _import("pyarrow")
    if pa is None:
        artifacts.append(DerivedArtifact(
            "parquet", None, None, DerivedStatus.BLOCKED_MISSING_DEPENDENCY,
            "pyarrow not importable in this environment"))
    else:
        try:
            import pyarrow.parquet as pq                  # noqa: WPS433
            table = pa.table({c: [r[c] for r in rows] for c in PROJECTION_COLUMNS})
            p = os.path.join(out_dir, "records.parquet")
            pq.write_table(table, p)
            artifacts.append(DerivedArtifact("parquet", os.path.basename(p),
                                             sha256_file(p), DerivedStatus.BUILT))
        except Exception as exc:                          # pragma: no cover
            artifacts.append(DerivedArtifact("parquet", None, None,
                                             DerivedStatus.FAILED, repr(exc)))

    duckdb = _import("duckdb")
    if duckdb is None:
        artifacts.append(DerivedArtifact(
            "duckdb", None, None, DerivedStatus.BLOCKED_MISSING_DEPENDENCY,
            "duckdb not importable in this environment"))
    else:
        try:
            p = os.path.join(out_dir, "records.duckdb")
            if os.path.exists(p):
                os.remove(p)
            con = duckdb.connect(p)
            # Query columns are VARCHAR for convenience, but reconciliation
            # reads row_json: a SQL type coercion must never be able to make a
            # derived view *look* reconciled when its content changed.
            cols = ", ".join(f'"{c}" VARCHAR' for c in PROJECTION_COLUMNS)
            con.execute(f"CREATE TABLE records (row_digest VARCHAR, row_json VARCHAR, {cols})")
            payload = []
            for src, r in zip(records, rows):
                payload.append([row_digest(src), canonical_json(r)]
                               + [None if r[c] is None else str(r[c])
                                  for c in PROJECTION_COLUMNS])
            n_ph = len(PROJECTION_COLUMNS) + 2
            con.executemany(f"INSERT INTO records VALUES ({','.join('?' * n_ph)})",
                            payload)
            con.close()
            artifacts.append(DerivedArtifact("duckdb", os.path.basename(p),
                                             sha256_file(p), DerivedStatus.BUILT))
        except Exception as exc:                          # pragma: no cover
            artifacts.append(DerivedArtifact("duckdb", None, None,
                                             DerivedStatus.FAILED, repr(exc)))

    built = [a for a in artifacts if a.status == DerivedStatus.BUILT]
    status = (DerivedStatus.BUILT if built
              else DerivedStatus.BLOCKED_MISSING_DEPENDENCY)
    manifest = {
        "derived_manifest_version": "PCG-DERIVED-v1",
        "authoritative_store_root_hash": store_manifest.get("store_root_hash"),
        "authoritative_n_records": store_manifest.get("n_records"),
        "projection_version": proj["projection_version"],
        "projection_digest": proj["digest"],
        "n_rows": proj["n_rows"],
        "artifacts": [a.to_dict() for a in artifacts],
        "status": status,
    }
    manifest["manifest_hash"] = sha256_text(canonical_json(
        {k: v for k, v in manifest.items() if k != "manifest_hash"}))
    mp = os.path.join(out_dir, DERIVED_MANIFEST)
    with open(mp, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    return DerivedBuildResult(status=status, artifacts=artifacts,
                              manifest_path=mp, manifest=manifest)


def read_derived_rows(out_dir: str) -> Optional[List[Dict[str, Any]]]:
    """Read the derived rows back, preferring parquet. None if unreadable."""
    pa = _import("pyarrow")
    p = os.path.join(out_dir, "records.parquet")
    if pa is not None and os.path.exists(p):
        import pyarrow.parquet as pq                       # noqa: WPS433
        return pq.read_table(p).to_pylist()
    duckdb = _import("duckdb")
    d = os.path.join(out_dir, "records.duckdb")
    if duckdb is None and not os.path.exists(d):
        n = os.path.join(out_dir, "records.ndjson")
        if os.path.exists(n):
            with open(n, encoding="utf-8") as fh:
                return [json.loads(ln) for ln in fh if ln.strip()]
        return None
    if duckdb is not None and os.path.exists(d):
        con = duckdb.connect(d)
        rows = con.execute("SELECT row_json FROM records").fetchall()
        con.close()
        return [json.loads(r[0]) for r in rows]
    return None
