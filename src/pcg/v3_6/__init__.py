"""PCG-MAS v3.6 DEV execution reconciliation layer.

Bridges the single-stage agentic pipeline in ``src/pcg`` (where the prover owns
generation) to the v2.3.2 multi-stage task graph (where generation is an
upstream dispatcher stage), WITHOUT creating a second provider path.

Generation happens exactly once, in the v2.3.2 dispatcher. Certification
replays the identical prompt through the prover's own code path.
"""
V3_6_LAYER = "PCG_MAS_V3_6_DEV_RECONCILIATION_V1"
