"""PCG-MAS v3.4R Deployment-Visible Runtime Candidate Data Structure.

Defines RuntimeCandidate with deployment-visible fields only.
Zero evaluator fields, zero evaluator references, zero static taint.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass
class RuntimeCandidate:
    """Deployment-visible candidate representation.

    Contains only fields observable at runtime during deployment.
    Evaluator ground-truth fields are strictly excluded.
    """

    candidate_id: str
    model: str
    dataset: str
    example_id: str
    request_hash: str
    response_hash: str
    prompt: str
    candidate_answer: str
    windows: List[str]
    evidence_hashes: List[str]
    obligations: List[Dict[str, Any]]
    resource_metrics: Dict[str, Any]
    policy_context: Optional[Dict[str, Any]] = None
    tool_snapshots: Optional[List[Dict[str, Any]]] = None
    action_trace: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
