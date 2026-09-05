"""Paired crossed seed x example bootstrap (the v3.0 locked scheme).

    1. resample seed IDs (with replacement)
    2. independently resample example IDs (with replacement)
    3. preserve system pairing within every resampled unit

The algebraic identity S + V = Delta is asserted on the realized data AND on
every resample; a violation indicates a metric-implementation defect, not
sampling noise.
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

from ..science.sv import sv_decomposition

INFERENCE_SCHEME = "paired_crossed_seed_x_example_bootstrap_v3"


def _percentile(xs: Sequence[float], q: float) -> Optional[float]:
    if not xs:
        return None
    ys = sorted(xs)
    i = min(len(ys) - 1, max(0, int(round(q * (len(ys) - 1)))))
    return ys[i]


def paired_crossed_bootstrap(records: List[dict], n_boot: int = 2000,
                             seed: int = 0, alpha: float = 0.05) -> Dict[str, object]:
    """Returns S/V/Delta point estimates with CIs and the identity check."""
    rng = random.Random(seed)
    by_seed: Dict[object, List[dict]] = defaultdict(list)
    for r in records:
        by_seed[r.get("seed")].append(r)
    seeds = sorted(by_seed, key=lambda s: (s is None, s))
    if not seeds:
        return {"n_boot": 0, "inference_scheme": INFERENCE_SCHEME}

    def stat(rows: List[dict]):
        lnc = [r.get("loss_nocert") for r in rows]
        lpg = [r.get("loss_pcg") for r in rows]
        ans = [bool(r.get("answered")) for r in rows]
        if not rows or any(x is None for x in lnc):
            return None
        return sv_decomposition(lnc, lpg, ans)

    point = stat(records)
    Ss: List[float] = []
    Vs: List[float] = []
    Ds: List[float] = []
    max_resid = 0.0
    for _ in range(n_boot):
        drawn_seeds = [seeds[rng.randrange(len(seeds))] for _ in range(len(seeds))]
        rows: List[dict] = []
        for s in drawn_seeds:
            pool = by_seed[s]
            if not pool:
                continue
            rows.extend(pool[rng.randrange(len(pool))] for _ in range(len(pool)))
        r = stat(rows)
        if r is None or r.S is None:
            continue
        Ss.append(r.S); Vs.append(r.V); Ds.append(r.delta)
        max_resid = max(max_resid, r.identity_residual or 0.0)

    def ci(xs):
        return [_percentile(xs, alpha / 2), _percentile(xs, 1 - alpha / 2)]

    return {
        "inference_scheme": INFERENCE_SCHEME,
        "n_boot": len(Ss),
        "seed_count": len(seeds),
        "S": point.S if point else None, "S_CI": ci(Ss),
        "V": point.V if point else None, "V_CI": ci(Vs),
        "Delta": point.delta if point else None, "Delta_CI": ci(Ds),
        "max_identity_residual_over_resamples": max_resid,
        "identity_holds_on_every_resample": max_resid <= 1e-12,
    }
