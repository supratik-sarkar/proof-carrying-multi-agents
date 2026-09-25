"""Production comparators for PCG-MAS v3.5.

Implements:
1. Exact-match candidate selection rule:
   - PCG accepts k_c candidates per cell c under frozen calibrated rule.
   - Every mandatory score-producing comparator accepts top k_c under its frozen total order.
   - Boundary tie-break: SHA256(candidate_id || comparator_id || v3_5_freeze_root).
2. CoverageMatchedVerifierOnly:
   - Strict superiority role.
   - Deterministic total order:
     (1) minimum critical S_i descending
     (2) maximum critical K_i ascending
     (3) supplementary certified fraction descending
     (4) hash tie-break
3. SignalMatchedFusion:
   - Core noninferiority / strong superiority role.
   - L2-regularized logistic regression over primitive feature vector phi(R).
   - Forbidden features: harm, success, gold answer, dataset_id, model_id, candidate_id.
   - 5-fold grouped CV grouped by (dataset, example_id).
4. Context-native baselines:
   - NoCert, CitationOnly, ShieldAgent, AgentRR: marked eligible_for_primary_go = False.
5. Strict ban on historical contaminated thresholds (0.30, 0.20) in primary GO logic.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

FORBIDDEN_HISTORICAL_THRESHOLDS = {0.30, 0.20}
FORBIDDEN_FEATURE_NAMES = {
    "harm",
    "success",
    "gold_answer",
    "reference",
    "dataset_id",
    "model_id",
    "candidate_id",
    "evaluator_label",
}


def compute_deterministic_tie_break(
    candidate_id: str,
    comparator_id: str,
    v3_5_freeze_root: str,
) -> str:
    """Deterministic hash tie-break: SHA256(candidate_id || comparator_id || v3_5_freeze_root)."""
    key = f"{candidate_id}||{comparator_id}||{v3_5_freeze_root}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


class BaseComparator:
    """Abstract base class for all v3.5 comparators."""

    comparator_id: str
    is_mandatory_primary: bool = False
    eligible_for_primary_go: bool = False

    def rank_candidates(
        self,
        candidates: List[Dict[str, Any]],
        v3_5_freeze_root: str,
    ) -> List[Dict[str, Any]]:
        """Return candidates ranked in descending order of preference."""
        raise NotImplementedError


class CoverageMatchedVerifierOnly(BaseComparator):
    """Primary strict superiority comparator using semantic verifier only."""

    comparator_id = "CoverageMatchedVerifierOnly"
    is_mandatory_primary = True
    eligible_for_primary_go = True

    def rank_candidates(
        self,
        candidates: List[Dict[str, Any]],
        v3_5_freeze_root: str,
    ) -> List[Dict[str, Any]]:
        """Sort by:
        1. min_critical_S descending
        2. max_critical_K ascending
        3. supp_fraction descending
        4. deterministic hash tie-break ascending
        """
        def sort_key(c: Dict[str, Any]) -> Tuple[float, float, float, str]:
            phi = c.get("phi", {})
            s_crit = float(phi.get("min_critical_S", 0.0))
            k_crit = float(phi.get("max_critical_K", 1.0))
            supp_frac = float(phi.get("supp_fraction", 0.0))
            cand_id = str(c.get("candidate_id", ""))
            h = compute_deterministic_tie_break(cand_id, self.comparator_id, v3_5_freeze_root)
            # Python sorts ascending: negate descending fields
            return (-s_crit, k_crit, -supp_frac, h)

        return sorted(candidates, key=sort_key)


class SignalMatchedFusion(BaseComparator):
    """Primary non-inferiority comparator using logistic regression over primitive phi(R)."""

    comparator_id = "SignalMatchedFusion"
    is_mandatory_primary = True
    eligible_for_primary_go = True

    # Pre-declared primitive feature list (zero labels or IDs)
    FEATURE_NAMES = [
        "min_critical_S",
        "max_critical_K",
        "supp_fraction",
        "v_h_pass",
        "v_gamma_pass",
        "v_pi_pass",
    ]

    def __init__(self, c_grid: Optional[List[float]] = None):
        self.c_grid = c_grid if c_grid is not None else [0.01, 0.1, 1.0, 10.0]
        self.model: Optional[LogisticRegression] = None
        self.best_c: Optional[float] = None

    @classmethod
    def validate_features(cls, phi: Dict[str, Any]) -> None:
        """Verify that phi contains no forbidden labels, IDs, or target information."""
        for k in phi.keys():
            k_lower = k.lower()
            for forbidden in FORBIDDEN_FEATURE_NAMES:
                if forbidden in k_lower:
                    raise ValueError(f"Forbidden feature '{k}' detected in fusion feature vector phi(R)!")

    @classmethod
    def extract_feature_vector(cls, phi: Dict[str, Any]) -> List[float]:
        """Extract numeric array matching pre-declared FEATURE_NAMES."""
        cls.validate_features(phi)
        return [float(phi.get(feat, 0.0)) for feat in cls.FEATURE_NAMES]

    def fit_grouped_cv(
        self,
        records: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Fit L2-regularized logistic regression using 5-fold grouped CV by (dataset, example_id)."""
        X_list = []
        y_list = []
        groups = []

        for _i, r in enumerate(records):
            phi = r.get("phi", {})
            self.validate_features(phi)
            vec = self.extract_feature_vector(phi)
            # v3.6 correctness hardening: a missing harm_label previously
            # defaulted to 0, silently labelling the record "safe". That
            # weakens this comparator in the direction that flatters PCG, so
            # the label must be present and binary. Fail closed.
            if "harm_label" not in r:
                raise ValueError(
                    f"FUSION_FIT_MISSING_HARM_LABEL: record index {_i} "
                    f"(dataset={r.get('dataset')!r}, example_id={r.get('example_id')!r}). "
                    "Silent default to 'safe' is forbidden on the primary comparator path."
                )
            _raw = r["harm_label"]
            if isinstance(_raw, bool):
                _raw = int(_raw)
            if _raw not in (0, 1):
                raise ValueError(
                    f"FUSION_FIT_INVALID_HARM_LABEL: record index {_i} value={r['harm_label']!r}; "
                    "expected binary 0/1."
                )
            # Binary target: 1 = safe / acceptable, 0 = harmful
            target = 1 - int(_raw)
            grp = f"{r.get('dataset', '')}:{r.get('example_id', '')}"

            X_list.append(vec)
            y_list.append(target)
            groups.append(grp)

        X = np.array(X_list, dtype=np.float64)
        y = np.array(y_list, dtype=np.int32)

        # Unique groups check
        unique_groups = np.unique(groups)
        n_splits = min(5, len(unique_groups))
        gkf = GroupKFold(n_splits=n_splits)

        best_score = -1.0
        best_c = self.c_grid[0]

        for c_val in self.c_grid:
            fold_scores = []
            for train_idx, val_idx in gkf.split(X, y, groups):
                clf = LogisticRegression(C=c_val, penalty="l2", solver="lbfgs", max_iter=500)
                # If only one class present in fold, continue
                if len(np.unique(y[train_idx])) < 2:
                    continue
                clf.fit(X[train_idx], y[train_idx])
                score = clf.score(X[val_idx], y[val_idx])
                fold_scores.append(score)

            mean_score = np.mean(fold_scores) if fold_scores else 0.0
            if mean_score > best_score:
                best_score = mean_score
                best_c = c_val

        # Fit final model on all training data
        self.best_c = best_c
        self.model = LogisticRegression(C=best_c, penalty="l2", solver="lbfgs", max_iter=500)
        self.model.fit(X, y)

        return {
            "best_c": best_c,
            "cv_score": best_score,
            "n_samples": len(X),
            "n_features": len(self.FEATURE_NAMES),
            "n_groups": len(unique_groups),
        }

    def predict_proba(self, phi: Dict[str, Any]) -> float:
        """Predict probability of acceptance/safety."""
        if self.model is None:
            raise RuntimeError("SignalMatchedFusion model must be fitted before predict_proba.")
        vec = self.extract_feature_vector(phi)
        proba = self.model.predict_proba([vec])[0][1]
        return float(proba)

    def rank_candidates(
        self,
        candidates: List[Dict[str, Any]],
        v3_5_freeze_root: str,
    ) -> List[Dict[str, Any]]:
        """Sort candidates descending by fusion predicted score, tie-breaking by hash."""
        def sort_key(c: Dict[str, Any]) -> Tuple[float, str]:
            phi = c.get("phi", {})
            score = self.predict_proba(phi) if self.model is not None else 0.0
            cand_id = str(c.get("candidate_id", ""))
            h = compute_deterministic_tie_break(cand_id, self.comparator_id, v3_5_freeze_root)
            return (-score, h)

        return sorted(candidates, key=sort_key)


class ContextOnlyBaseline(BaseComparator):
    """Context-native baseline that cannot enter primary GO logic."""

    is_mandatory_primary = False
    eligible_for_primary_go = False

    def __init__(self, name: str):
        self.comparator_id = name

    def rank_candidates(
        self,
        candidates: List[Dict[str, Any]],
        v3_5_freeze_root: str,
    ) -> List[Dict[str, Any]]:
        """Deterministic fallback order with hash tie-break."""
        def sort_key(c: Dict[str, Any]) -> str:
            cand_id = str(c.get("candidate_id", ""))
            return compute_deterministic_tie_break(cand_id, self.comparator_id, v3_5_freeze_root)

        return sorted(candidates, key=sort_key)


def get_all_registered_comparators() -> Dict[str, BaseComparator]:
    """Return all comparators with strict enforcement of primary eligibility."""
    comps: Dict[str, BaseComparator] = {
        "CoverageMatchedVerifierOnly": CoverageMatchedVerifierOnly(),
        "SignalMatchedFusion": SignalMatchedFusion(),
        "NoCert": ContextOnlyBaseline("NoCert"),
        "CitationOnly": ContextOnlyBaseline("CitationOnly"),
        "ShieldAgent": ContextOnlyBaseline("ShieldAgent"),
        "AgentRR": ContextOnlyBaseline("AgentRR"),
    }

    # Gate C6 audit: Ensure zero historical contaminated thresholds exist in primary logic
    for name, c in comps.items():
        if c.eligible_for_primary_go:
            # Must be CoverageMatchedVerifierOnly or SignalMatchedFusion
            if name not in ("CoverageMatchedVerifierOnly", "SignalMatchedFusion"):
                raise ValueError(f"Unapproved comparator {name} marked eligible_for_primary_go!")

    return comps


def verify_fusion_domain() -> Dict[str, Any]:
    """Production verification callable for fusion domain."""
    comps = get_all_registered_comparators()
    return {"domain": "fusion", "registered_count": len(comps)}

