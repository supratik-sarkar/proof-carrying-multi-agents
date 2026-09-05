#!/usr/bin/env python3
"""S/V decomposition of the reported harm reduction, with paired bootstrap CIs.

  S = mean(l_nc over all)  -  mean(l_nc over answered)      harm avoided by answering less
  V = mean(l_nc over answered) - mean(l_pcg over answered)  harm avoided on the SAME examples

S + V is identically the reported reduction; `decompose` asserts that identity.
Resampling is paired: one index draw indexes both arms, so the two components are
never resampled independently.

PROVENANCE: DERIVED. Inputs carry their own provenance; outputs inherit it.
Uses no model call and no network.
"""
from __future__ import annotations
import random

PROVENANCE_CLASS = "DERIVED"
REQUIRED = ("example_id", "pcg_answered", "l_nc", "l_pcg")


class MissingEvidenceError(KeyError):
    pass


def _check(rows):
    rows = list(rows)
    if not rows:
        raise ValueError("Empty sample: fail-closed, no interval is defined.")
    for i, r in enumerate(rows):
        missing = [f for f in REQUIRED if f not in r]
        if missing:
            raise MissingEvidenceError(f"row {i} ({r.get('example_id','?')}) missing {missing}")
    return rows


def _mean(xs):
    xs = list(xs)
    return (sum(xs) / len(xs)) if xs else None


def decompose(rows):
    """Point estimates of S and V, plus the identity check."""
    rows = _check(rows)
    answered = [r for r in rows if r["pcg_answered"]]
    all_nc = _mean(r["l_nc"] for r in rows)
    a_nc   = _mean(r["l_nc"] for r in answered)
    a_pcg  = _mean(r["l_pcg"] for r in answered)
    if a_nc is None:                     # nothing answered -> V undefined, fail closed
        return {"S": None, "V": None, "total": None, "n": len(rows), "n_answered": 0,
                "provenance_class": PROVENANCE_CLASS}
    S, V = all_nc - a_nc, a_nc - a_pcg
    return {"S": S, "V": V, "total": S + V, "identity_holds": abs((S + V) - (all_nc - a_pcg)) < 1e-12,
            "n": len(rows), "n_answered": len(answered), "provenance_class": PROVENANCE_CLASS}


def paired_bootstrap_intervals(rows, seed: int = 42, n_bootstrap: int = 1000, alpha: float = 0.05):
    """Deterministic given `seed`: identical inputs and seed give identical output."""
    rows = _check(rows)
    rng = random.Random(seed)
    n = len(rows)
    S_b, V_b = [], []
    for _ in range(n_bootstrap):
        idx = [rng.randrange(n) for _ in range(n)]      # one draw, both arms — paired
        d = decompose([rows[i] for i in idx])
        if d["S"] is not None:
            S_b.append(d["S"]); V_b.append(d["V"])

    def ci(b):
        if not b:
            return (None, None)
        b = sorted(b)
        lo = b[max(0, int((alpha / 2) * len(b)) - 1)]
        hi = b[min(len(b) - 1, int((1 - alpha / 2) * len(b)))]
        return (round(lo, 10), round(hi, 10))

    pt = decompose(rows)
    return {
        "seed": seed, "n_bootstrap": n_bootstrap, "alpha": alpha,
        "S": None if pt["S"] is None else round(pt["S"], 10),
        "V": None if pt["V"] is None else round(pt["V"], 10),
        "S_ci": ci(S_b), "V_ci": ci(V_b),
        "n": pt["n"], "n_answered": pt["n_answered"],
        "provenance_class": PROVENANCE_CLASS,
    }
