"""
Differential privacy for cross-agent feature sharing (Sec 5.4).

    - `gaussian_mechanism`: additive Gaussian noise with stated (eps, delta)
    - `laplace_mechanism`: additive Laplace noise
    - `calibrate_sigma_for_gaussian`: Given sensitivity, target (eps, delta), returns sigma
    - `LeakageProxy`: membership-distinguishability AUC proxy from Appendix D.7

We use simple additive mechanisms rather than pulling in opacus at runtime,
because the feature vectors shared between agents are low-dimensional
aggregates (redundancy count, disagreement, verifier margin). This keeps
the code self-contained and the noise calibration transparent.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def calibrate_sigma_for_gaussian(
    sensitivity: float,
    epsilon: float,
    delta: float,
) -> float:
    """Sigma for the analytic Gaussian mechanism.

    We use the simple (eps, delta)-DP Gaussian mechanism calibration:
        sigma = sensitivity * sqrt(2 * ln(1.25 / delta)) / epsilon
    This is not tight (Balle-Wang 2018 gives a tighter analytic form) but
    suffices for the R4 sweeps. The exact calibration can be swapped in by
    pip-installing `diffprivlib` without changing the rest of the pipeline.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1)")
    return sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon


def gaussian_mechanism(
    x: np.ndarray,
    sensitivity: float,
    epsilon: float,
    delta: float = 1e-5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    sigma = calibrate_sigma_for_gaussian(sensitivity, epsilon, delta)
    rng = rng or np.random.default_rng()
    return x + rng.normal(0.0, sigma, size=x.shape)


def laplace_mechanism(
    x: np.ndarray,
    sensitivity: float,
    epsilon: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    rng = rng or np.random.default_rng()
    scale = sensitivity / epsilon
    return x + rng.laplace(0.0, scale, size=x.shape)


# -----------------------------------------------------------------------------
# Leakage proxy (Appendix D.7)
# -----------------------------------------------------------------------------


@dataclass
class LeakageProxy:
    """Estimate membership-style leakage via a simple classifier.

    We train a logistic regression to distinguish perturbed "true evidence"
    features from perturbed "distractor" features, and report its AUC. Higher
    AUC means the DP noise is too small to hide membership; lower AUC means
    membership is effectively protected.
    """

    def auc(self, true_feats: np.ndarray, distractor_feats: np.ndarray) -> float:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import train_test_split

        X = np.vstack([true_feats, distractor_feats])
        y = np.concatenate([np.ones(len(true_feats)), np.zeros(len(distractor_feats))])
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
        clf = LogisticRegression(max_iter=500).fit(X_tr, y_tr)
        scores = clf.predict_proba(X_te)[:, 1]
        return float(roc_auc_score(y_te, scores))
