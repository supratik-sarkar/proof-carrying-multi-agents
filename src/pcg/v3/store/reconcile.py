"""Reconciliation gate between the authoritative JSONL store and derived views.

The gate is three-state and fails closed:

  RECONCILED   the derived view reproduces the authoritative projection digest,
               its manifest hash-links to the authoritative store manifest, and
               every declared artifact file still hashes to its recorded value;
  DIVERGED     a derived view exists but does not reproduce the authoritative
               content -- reported with the localised record_ids;
  UNAVAILABLE  no derived view could be read (typically a missing optional
               dependency). UNAVAILABLE is *not* a pass: it is the honest
               statement that the check did not run.

Only RECONCILED admits a derived view as a *substitute* for the authoritative
store. Absent that, analysis must read JSONL. ``assert_derived_admissible``
enforces this, and is what the emitter path calls.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..canon import canonical_json, sha256_file, sha256_text
from .derived import DERIVED_MANIFEST, read_derived_rows
from .projection import divergent_rows, projection_digest


class ReconcileStatus(str, Enum):
    RECONCILED = "RECONCILED"
    DIVERGED = "DIVERGED"
    UNAVAILABLE = "UNAVAILABLE"


class DerivedNotAdmissible(RuntimeError):
    """A derived view was requested as authoritative but did not reconcile."""


@dataclass
class ReconciliationReport:
    status: ReconcileStatus
    authoritative_digest: Optional[str] = None
    derived_digest: Optional[str] = None
    authoritative_rows: Optional[int] = None
    derived_rows: Optional[int] = None
    store_root_hash_linked: Optional[bool] = None
    artifact_hashes_ok: Optional[bool] = None
    manifest_hash_ok: Optional[bool] = None
    reasons: List[str] = field(default_factory=list)
    localisation: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status is ReconcileStatus.RECONCILED

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


def reconcile(records: Sequence[Mapping[str, Any]],
              derived_dir: str,
              store_manifest: Mapping[str, Any]) -> ReconciliationReport:
    auth = projection_digest(records)
    reasons: List[str] = []

    mp = os.path.join(derived_dir, DERIVED_MANIFEST)
    if not os.path.exists(mp):
        return ReconciliationReport(ReconcileStatus.UNAVAILABLE,
                                    authoritative_digest=auth["digest"],
                                    authoritative_rows=auth["n_rows"],
                                    reasons=["no derived manifest"])
    with open(mp, encoding="utf-8") as fh:
        manifest = json.load(fh)

    recomputed = sha256_text(canonical_json(
        {k: v for k, v in manifest.items() if k != "manifest_hash"}))
    manifest_hash_ok = recomputed == manifest.get("manifest_hash")
    if not manifest_hash_ok:
        reasons.append("derived manifest self-hash mismatch")

    linked = (manifest.get("authoritative_store_root_hash")
              == store_manifest.get("store_root_hash"))
    if not linked:
        reasons.append("derived manifest does not hash-link to this store")

    artifact_hashes_ok = True
    for a in manifest.get("artifacts", []):
        if a.get("status") != "BUILT" or not a.get("path"):
            continue
        p = os.path.join(derived_dir, a["path"])
        if not os.path.exists(p) or sha256_file(p) != a.get("sha256"):
            artifact_hashes_ok = False
            reasons.append(f"artifact hash mismatch or missing: {a.get('path')}")

    rows = read_derived_rows(derived_dir)
    if rows is None:
        reasons.append("derived rows unreadable (missing optional dependency)")
        return ReconciliationReport(ReconcileStatus.UNAVAILABLE,
                                    authoritative_digest=auth["digest"],
                                    authoritative_rows=auth["n_rows"],
                                    store_root_hash_linked=linked,
                                    artifact_hashes_ok=artifact_hashes_ok,
                                    manifest_hash_ok=manifest_hash_ok,
                                    reasons=reasons)

    der = projection_digest(rows)
    if der["digest"] != auth["digest"]:
        reasons.append("projection digest mismatch")

    status = (ReconcileStatus.RECONCILED
              if not reasons and linked and artifact_hashes_ok and manifest_hash_ok
              else ReconcileStatus.DIVERGED)
    loc = ({} if status is ReconcileStatus.RECONCILED
           else divergent_rows(list(records), rows))
    return ReconciliationReport(status,
                                authoritative_digest=auth["digest"],
                                derived_digest=der["digest"],
                                authoritative_rows=auth["n_rows"],
                                derived_rows=der["n_rows"],
                                store_root_hash_linked=linked,
                                artifact_hashes_ok=artifact_hashes_ok,
                                manifest_hash_ok=manifest_hash_ok,
                                reasons=reasons, localisation=loc)


def assert_derived_admissible(report: ReconciliationReport) -> None:
    """Fail closed. UNAVAILABLE is a refusal, not a pass."""
    if not report.ok:
        raise DerivedNotAdmissible(
            f"derived store not admissible: {report.status.value}; "
            f"reasons={report.reasons}")
