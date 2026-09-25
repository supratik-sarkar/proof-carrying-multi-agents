"""Single source of truth for v3.2 release and schema identity.

Version discipline: the *record* schema version is bumped whenever a field is
added, because record identity is a hash over committed fields and a reader must
be able to tell which field set produced an address. The metric version is
independent and is bumped only when a metric definition changes; v3.2 adds
fields but redefines no metric, so METRIC_VERSION is unchanged.
"""
from __future__ import annotations

PCG_MAS_RELEASE = "v3.2"
SCIENTIFIC_SCHEMA_VERSION = "3.0.0"
ARTIFACT_SCHEMA_VERSION = "3.2.0"
RECORD_SCHEMA_VERSION = "3.2.1"
METRIC_VERSION = "v3.0.0"

MANUSCRIPT = "pcg_mas_manuscript_v3-2.tex"
DESIGN_LOCK = "pcg_mas_manuscript_v3-2.md"

N_TABLES = 33
N_FIGURES = 10
WORKSTREAMS = [f"A{i:02d}" for i in range(1, 19)]

#: Content-addressing and execution-substrate versions introduced in v3.2.
CAS_SCHEME = "PCG-CAS-v1"
JSONL_ENVELOPE_VERSION = "PCG-JSONL-v1"
PROJECTION_VERSION = "PCG-PROJ-v1"
LINEAGE_VERSION = "PCG-LINEAGE-v1"
EXECUTION_GRAPH_VERSION = "PCG-EXECGRAPH-v1"

def version_payload() -> dict:
    return {
        "PCG_MAS_RELEASE": PCG_MAS_RELEASE,
        "scientific_schema_version": SCIENTIFIC_SCHEMA_VERSION,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "record_schema_version": RECORD_SCHEMA_VERSION,
        "metric_version": METRIC_VERSION,
        "cas_scheme": CAS_SCHEME,
        "jsonl_envelope_version": JSONL_ENVELOPE_VERSION,
        "projection_version": PROJECTION_VERSION,
        "lineage_version": LINEAGE_VERSION,
        "execution_graph_version": EXECUTION_GRAPH_VERSION,
        "manuscript": MANUSCRIPT,
    }
