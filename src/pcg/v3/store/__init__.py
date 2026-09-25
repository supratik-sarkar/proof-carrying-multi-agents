"""Record storage.

Layering rule (locked): **JSONL is the authoritative record store.** Parquet and
DuckDB are *derived* views. They are hash-linked back to the authoritative store
and may never be read by the manuscript emitter unless the reconciliation gate
reports RECONCILED.
"""
from .jsonl import (AuthoritativeRecordStore, ChainBroken, JsonlSegment,
                    SegmentManifest)
from .projection import (PROJECTION_COLUMNS, projection_digest, project_record,
                         row_digest)
from .reconcile import (ReconcileStatus, ReconciliationReport,
                        assert_derived_admissible, reconcile)

__all__ = ["AuthoritativeRecordStore", "ChainBroken", "JsonlSegment",
           "SegmentManifest", "PROJECTION_COLUMNS", "project_record",
           "projection_digest", "row_digest", "reconcile",
           "ReconciliationReport", "ReconcileStatus",
           "assert_derived_admissible"]
