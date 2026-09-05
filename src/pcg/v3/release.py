"""Single source of truth for v3.0 release and schema identity."""
from __future__ import annotations

PCG_MAS_RELEASE = "v3.0"
SCIENTIFIC_SCHEMA_VERSION = "3.0.0"
ARTIFACT_SCHEMA_VERSION = "3.0.0"
RECORD_SCHEMA_VERSION = "3.0.0"
METRIC_VERSION = "v3.0.0"

MANUSCRIPT = "pcg_mas_iclr2027_v3-0.tex"
DESIGN_LOCK = "pcg_mas_iclr2027_v3-0.md"

N_TABLES = 33
N_FIGURES = 10
WORKSTREAMS = [f"A{i:02d}" for i in range(1, 19)]

def version_payload() -> dict:
    return {
        "PCG_MAS_RELEASE": PCG_MAS_RELEASE,
        "scientific_schema_version": SCIENTIFIC_SCHEMA_VERSION,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "record_schema_version": RECORD_SCHEMA_VERSION,
        "metric_version": METRIC_VERSION,
    }
