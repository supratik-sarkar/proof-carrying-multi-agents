#!/usr/bin/env python3
"""Generate the versioned Python->JS scientific contract.

The frontend must NOT reimplement V bits, channel names, acceptance logic,
dependence formulas, controller semantics or provenance schemas. It consumes
this generated contract so it cannot drift from the core.
"""
from __future__ import annotations
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

from pcg.v3.channels import CHANNELS, CONJUNCTS, CONJUNCT_TO_CHANNELS, RESIDUALS
from pcg.v3.record import CONTROLLER_ACTIONS, PROVENANCE_CLASSES, SYSTEMS, json_schema
from pcg.v3.release import (ARTIFACT_SCHEMA_VERSION, N_FIGURES, N_TABLES,
                            PCG_MAS_RELEASE, RECORD_SCHEMA_VERSION,
                            SCIENTIFIC_SCHEMA_VERSION, WORKSTREAMS)
from pcg.v3.science.dependence import GateState
from pcg.v3.orchestration.state import NODES, TERMINALS


def build() -> dict:
    return {
        "PCG_MAS_RELEASE": PCG_MAS_RELEASE,
        "scientific_schema_version": SCIENTIFIC_SCHEMA_VERSION,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "record_schema_version": RECORD_SCHEMA_VERSION,
        "conjuncts": [c.value for c in CONJUNCTS],
        "audit_channels": [c.value for c in CHANNELS],
        "conjunct_to_channels": {c.value: [x.value for x in v]
                                 for c, v in CONJUNCT_TO_CHANNELS.items()},
        "residuals_outside_channels": list(RESIDUALS),
        "gate_states": [g.value for g in GateState],
        "controller_actions": list(CONTROLLER_ACTIONS),
        "provenance_classes": list(PROVENANCE_CLASSES),
        "systems": list(SYSTEMS),
        "graph_nodes": NODES,
        "graph_terminals": TERMINALS,
        "workstreams": WORKSTREAMS,
        "n_tables": N_TABLES,
        "n_figures": N_FIGURES,
        "semantics": {
            "acceptance": "checker-relative external recomputability, NOT proof of world truth",
            "unknown_is_failure": True,
            "null_is_not_zero": True,
            "responsibility": "replay-interventional attribution; not causal root cause",
            "eps_tax": "open-world taxonomy residual; challenge-set alarm, not a deployment bound",
            "eps_src": "source/world-truth residual; outside the certificate",
        },
        "record_schema": json_schema(),
    }


if __name__ == "__main__":
    out = os.path.join(HERE, "contract.json")
    json.dump(build(), open(out, "w"), indent=2, sort_keys=True)
    print(f"wrote {out}")
