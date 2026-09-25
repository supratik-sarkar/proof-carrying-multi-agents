"""PCG-MAS v3.4R Grouped Five-Fold Outer Cross-Fitting.

Enforces:
- Grouping unit: (dataset, example_id).
- All model executions for one benchmark example reside in the exact same fold.
- OUTER_GROUP_LEAKAGE_COUNT == 0.
- Finite development search space serialized and hashed before held-out scoring.
"""

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Dict, List, Set, Tuple

from pcg.v3_4r.firewall import RuntimeCandidate


def assign_fold(dataset: str, example_id: str, num_folds: int = 5) -> int:
    """Deterministically assigns (dataset, example_id) to fold [0..num_folds-1]."""
    key = f"{dataset.strip()}|||{example_id.strip()}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest, 16) % num_folds


def partition_candidates_by_fold(
    candidates: List[RuntimeCandidate], num_folds: int = 5
) -> Dict[int, List[RuntimeCandidate]]:
    """Partitions a list of RuntimeCandidate objects into grouped folds."""
    folds: Dict[int, List[RuntimeCandidate]] = defaultdict(list)
    for c in candidates:
        f_idx = assign_fold(c.dataset, c.example_id, num_folds=num_folds)
        folds[f_idx].append(c)
    return dict(folds)


def audit_group_leakage(
    folds: Dict[int, List[RuntimeCandidate]],
) -> Tuple[int, Dict[str, Any]]:
    """Audits outer folds for any example_id appearing in multiple folds.

    Returns:
        (leakage_count, details)
    """
    example_to_folds: Dict[Tuple[str, str], Set[int]] = defaultdict(set)
    for f_idx, c_list in folds.items():
        for c in c_list:
            example_to_folds[(c.dataset, c.example_id)].add(f_idx)

    leakage_count = 0
    leaked_examples = []
    for (ds, eid), assigned in example_to_folds.items():
        if len(assigned) > 1:
            leakage_count += 1
            leaked_examples.append(
                {"dataset": ds, "example_id": eid, "folds": list(assigned)}
            )

    return leakage_count, {
        "total_examples": len(example_to_folds),
        "leakage_count": leakage_count,
        "leaked_examples": leaked_examples,
        "clean": leakage_count == 0,
    }


@dataclass
class DevelopmentSearchSpace:
    """Finite development parameter space pre-committed before evaluation."""

    tau_cov_grid: List[float]
    tau_e_grid: List[float]
    tau_c_grid: List[float]

    def serialize_and_hash(self) -> Tuple[str, str]:
        """Serializes search space deterministically and computes SHA-256."""
        payload = {
            "schema": "PCG_MAS_V3_4R_SEARCH_SPACE_PRECOMMIT_V1",
            "tau_cov_grid": sorted(self.tau_cov_grid),
            "tau_e_grid": sorted(self.tau_e_grid),
            "tau_c_grid": sorted(self.tau_c_grid),
            "grid_size": len(self.tau_cov_grid)
            * len(self.tau_e_grid)
            * len(self.tau_c_grid),
        }
        raw = json.dumps(payload, sort_keys=True, indent=2)
        h = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return raw, h


DEFAULT_SEARCH_SPACE = DevelopmentSearchSpace(
    tau_cov_grid=[1.0],
    tau_e_grid=[0.05],
    tau_c_grid=[0.30],
)
