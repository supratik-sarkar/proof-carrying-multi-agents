"""The four certificate conjuncts and the five audit channels.

These are DIFFERENT LAYERS and the v3.0 manuscript is explicit about it:
the conjuncts define acceptance; the channels are the audit taxonomy over
contract-relevant accepted failure. Neither implies the other. Theorem
`thm:irredundancy` concerns conjuncts; taxonomy coverage (A1b) concerns
channels and is tested by A14.
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, Tuple


class Conjunct(str, Enum):
    V_H = "V_H"            # evidence commitment / integrity
    V_PI = "V_Pi"          # pinned replay
    V_GAMMA = "V_Gamma"    # execution-policy compliance
    V_ENTAIL = "V_vdash"   # checker-relative entailment


class Channel(str, Enum):
    INT_FAIL = "IntFail"
    REPLAY_FAIL = "ReplayFail"
    DRIFT_FAIL = "DriftFail"
    CHECK_FAIL = "CheckFail"
    COV_GAP = "CovGap"


CONJUNCTS: Tuple[Conjunct, ...] = tuple(Conjunct)
CHANNELS: Tuple[Channel, ...] = tuple(Channel)

#: Manuscript mapping. CheckFail is cross-cutting (unsoundness of ANY checker),
#: which is why four conjuncts induce five channels.
CONJUNCT_TO_CHANNELS: Dict[Conjunct, Tuple[Channel, ...]] = {
    Conjunct.V_H:      (Channel.INT_FAIL, Channel.CHECK_FAIL),
    Conjunct.V_PI:     (Channel.REPLAY_FAIL, Channel.DRIFT_FAIL, Channel.CHECK_FAIL),
    Conjunct.V_GAMMA:  (Channel.COV_GAP, Channel.CHECK_FAIL),
    Conjunct.V_ENTAIL: (Channel.CHECK_FAIL,),
}

#: Residuals that lie OUTSIDE every channel. Never fold these into CovGap.
RESIDUALS = ("eps_tax", "eps_src")


def check(v_h, v_pi, v_gamma, v_entail):
    """Acceptance predicate. Unknown (None) is failure, never success."""
    bits = (v_h, v_pi, v_gamma, v_entail)
    if any(b is None for b in bits):
        return False
    return all(bool(b) for b in bits)


def check_applicable(
    v_h: object,
    v_pi: object,
    v_gamma: object,
    v_entail: object,
    *,
    grounding_applicable: bool = True,
    policy_applicable: bool = True,
    replay_applicable: bool = True,
) -> bool:
    """Applicability-aware acceptance predicate (S01 prospective protocol).
    
    Non-applicable obligations evaluate vacuously to True.
    """
    eff_h = bool(v_h) if grounding_applicable else True
    eff_entail = bool(v_entail) if grounding_applicable else True
    eff_gamma = bool(v_gamma) if policy_applicable else True
    eff_pi = bool(v_pi) if replay_applicable else True
    return eff_h and eff_entail and eff_gamma and eff_pi


def n_channels_fired(**fired) -> int:
    """N_F for the multiplicity-weighted union slack E[(N_F-1)_+]."""
    return sum(1 for c in CHANNELS if bool(fired.get(_snake(c), False)))


def _snake(c: Channel) -> str:
    return {"IntFail": "int_fail", "ReplayFail": "replay_fail", "DriftFail": "drift_fail",
            "CheckFail": "check_fail", "CovGap": "cov_gap"}[c.value]


CHANNEL_FIELD = {c: _snake(c) for c in CHANNELS}
