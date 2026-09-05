"""DIRECT execution provenance (RC2)."""
from .backends import BACKEND_EXECUTION_CLASS, DIRECT_ELIGIBLE, ExecutionClass, execution_class, is_direct_eligible
from .classify import ADVISORY_FAST_CALL_MS, classify, usage_completeness
from .errors import ATTEMPT_ONLY, OUTCOME_ELIGIBLE, ExecutionStatus, sanitize
from .fingerprint import code_fingerprint, write_fingerprint
from .gates import GATES, run_gates
from .hashing import hash_file, hash_obj, hash_set, hash_text
from .identity import RecordIdentity
from .lineage import (COST_ELIGIBLE_CLASSES, FORBIDDEN_CLASSES, OUTCOME_ELIGIBLE_CLASSES,
                      AggregateResult, aggregate, eligible, verify_recomputable)
from .numeric import UnknownMeasurement, add, is_finite_nonneg, mean, ratio, require
from .recorder import RunRecorder, utc_now
from .tables import ForbiddenProvenance, render_table, write_table
from .schema import (SCHEMA_ID, CheckerOutcomes, DecodingParams, DirectExecutionRecord,
                     ExecutionEvidence, ToolCall, json_schema)

__all__ = [n for n in dir() if not n.startswith("_")]
