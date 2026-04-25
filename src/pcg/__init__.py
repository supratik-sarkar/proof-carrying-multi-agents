"""
PCG-MAS: Proof-Carrying Generation for Multi-Agent Systems.

Reference implementation for the NeurIPS 2026 submission.

Public API (stable across the artifact):
    - Graph / graph types:     pcg.graph
    - Commitments:              pcg.commitments
    - Certificates:             pcg.certificate
    - Checker:                  pcg.checker
    - (delta, kappa)-indep:     pcg.independence
    - Responsibility:           pcg.responsibility
    - Risk / calibration:       pcg.risk
    - Privacy:                  pcg.privacy
    - Evaluation utilities:     pcg.eval
    - Agents:                   pcg.agents
    - Backends:                 pcg.backends
"""
from __future__ import annotations

__version__ = "0.1.0"

from pcg.certificate import (
    ClaimCertificate,
    ExecutionCertificate,
    ExecutionContract,
    GroundingCertificate,
    ReplayableStep,
)
from pcg.checker import CheckResult, Checker
from pcg.graph import (
    AgenticRuntimeGraph,
    ClaimNode,
    DelegationEdge,
    MemoryNode,
    NodeType,
    PolicyNode,
    SchemaNode,
    ToolCallNode,
    TruthNode,
)

__all__ = [
    "AgenticRuntimeGraph",
    "CheckResult",
    "Checker",
    "ClaimCertificate",
    "ClaimNode",
    "DelegationEdge",
    "ExecutionCertificate",
    "ExecutionContract",
    "GroundingCertificate",
    "MemoryNode",
    "NodeType",
    "PolicyNode",
    "ReplayableStep",
    "SchemaNode",
    "ToolCallNode",
    "TruthNode",
    "__version__",
]
