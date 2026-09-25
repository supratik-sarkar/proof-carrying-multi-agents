"""Provenance admissibility for release artifacts (BLOCKING gate).

Provenance is carried on the RECORD. It is never inferred from a directory name.
"""
from __future__ import annotations
from enum import Enum
from typing import Any, Dict, Iterable, List, Set


class ProvenanceClass(str, Enum):
    TEST_FIXTURE = "TEST_FIXTURE"
    MOCK = "MOCK"
    ENGINEERING_SMOKE = "ENGINEERING_SMOKE"
    PILOT = "PILOT"
    DIRECT_FINAL = "DIRECT_FINAL"
    DERIVED_FINAL = "DERIVED_FINAL"
    MODELLED = "MODELLED"
    PROTOCOL = "PROTOCOL"
    STATIC = "STATIC"


#: Admissible in a real release artifact.
RELEASE_ADMISSIBLE: Set[str] = {
    ProvenanceClass.DIRECT_FINAL.value, ProvenanceClass.DERIVED_FINAL.value,
    ProvenanceClass.MODELLED.value, ProvenanceClass.PROTOCOL.value,
    ProvenanceClass.STATIC.value,
}
SUBMISSION_ADMISSIBLE = RELEASE_ADMISSIBLE

#: Never admissible; must be stamped and quarantined.
SYNTHETIC: Set[str] = {
    ProvenanceClass.TEST_FIXTURE.value, ProvenanceClass.MOCK.value,
    ProvenanceClass.ENGINEERING_SMOKE.value,
}
#: Never permitted to supply a NUMBER to the two narrative macros.
MACRO_FORBIDDEN: Set[str] = SYNTHETIC | {ProvenanceClass.PILOT.value}

SYNTHETIC_STAMP = "SYNTHETIC --- NOT AN EXPERIMENTAL RESULT"


class InadmissibleProvenance(RuntimeError):
    pass


def classify_records(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    classes: Set[str] = set()
    n = 0
    for r in records:
        n += 1
        classes.add(str(r.get("provenance_class") or ProvenanceClass.TEST_FIXTURE.value))
    synth = sorted(classes & SYNTHETIC)
    return {"n": n, "classes": sorted(classes), "synthetic_classes": synth,
            "release_admissible": bool(classes) and not (classes - RELEASE_ADMISSIBLE),
            "requires_stamp": bool(synth)}


def assert_release_admissible(summary: Dict[str, Any], allow_synthetic: bool) -> None:
    if summary.get("release_admissible") or summary.get("submission_admissible"):
        return
    if not allow_synthetic:
        raise InadmissibleProvenance(
            f"source records carry non-admissible provenance {summary['classes']}. "
            "A release artifact may not be built from fixture/mock/smoke records. "
            "Pass --allow-synthetic to write export_SYNTHETIC/ instead.")


assert_submission_admissible = assert_release_admissible
