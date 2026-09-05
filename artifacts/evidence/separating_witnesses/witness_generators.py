#!/usr/bin/env python3
"""Separating witnesses: for each channel, an instance failing that conjunct alone.

The channel verdicts here are COMPUTED, not declared. Each generator builds a
concrete certificate carrying one real defect, and `evaluate_channels` recomputes
all four conjuncts from the object:

  V_H      recomputed digest == recorded digest          (hashlib)
  V_Pi     replay output     == recorded output          (byte equality)
  V_Gamma  every tool call   in the declared allow-list  (set membership)
  V_entail support entails the claim under the rule set  (exact-match rule)

A prior implementation stated the outcomes in a lookup table; a table cannot
demonstrate separation. `assert_single_channel_failure` re-derives them and raises
if any witness fails more or fewer than exactly one channel.

PROVENANCE: PROTOCOL. Constructed test objects, not measurements. No model call.
"""
from __future__ import annotations
import copy, hashlib

PROVENANCE_CLASS = "PROTOCOL"
CHANNELS = ("V_H", "V_Pi", "V_Gamma", "V_entail")


def _digest(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _base() -> dict:
    """A certificate that passes all four channels."""
    support = "Paris is the capital of France."
    return {
        "witness_id": "W_valid",
        "claim": "Paris is the capital of France.",
        "support_payload": support,
        "recorded_digest": _digest(support),
        "replay_output": support,
        "recorded_output": support,
        "tool_calls": ["retrieve", "parse"],
        "allow_list": ["retrieve", "parse", "align"],
        "target_channel": None,
    }


class WitnessGenerator:
    """Each generator introduces exactly one real defect into the base certificate."""

    @staticmethod
    def generate_valid() -> dict:
        return _base()

    @staticmethod
    def generate_W_H() -> dict:
        """Payload no longer hashes to its recorded digest."""
        w = _base(); w["witness_id"] = "W_H"; w["target_channel"] = "V_H"
        w["support_payload"] = w["support_payload"] + " (tampered)"
        w["replay_output"] = w["support_payload"]      # replay still faithful
        w["recorded_output"] = w["support_payload"]
        return w

    @staticmethod
    def generate_W_Pi() -> dict:
        """Trace irreproducible: replay output differs from the recorded output."""
        w = _base(); w["witness_id"] = "W_Pi"; w["target_channel"] = "V_Pi"
        w["replay_output"] = w["recorded_output"] + " (divergent replay)"
        return w

    @staticmethod
    def generate_W_Gamma() -> dict:
        """Out-of-contract tool call."""
        w = _base(); w["witness_id"] = "W_Gamma"; w["target_channel"] = "V_Gamma"
        w["tool_calls"] = w["tool_calls"] + ["exfiltrate"]
        return w

    @staticmethod
    def generate_W_entail() -> dict:
        """Replays exactly, but the support does not entail the claim."""
        w = _base(); w["witness_id"] = "W_entail"; w["target_channel"] = "V_entail"
        other = "Lyon is a city in France."
        w["support_payload"] = other
        w["recorded_digest"] = _digest(other)
        w["replay_output"] = other
        w["recorded_output"] = other
        return w

    @classmethod
    def generate_all(cls) -> list[dict]:
        return [cls.generate_W_H(), cls.generate_W_Pi(),
                cls.generate_W_Gamma(), cls.generate_W_entail()]


def evaluate_channels(witness: dict) -> dict:
    """Recompute all four conjuncts from the witness object."""
    for f in ("support_payload", "recorded_digest", "replay_output",
              "recorded_output", "tool_calls", "allow_list", "claim"):
        if f not in witness:
            raise KeyError(f"witness missing required field '{f}' (fail-closed)")
    v_h = _digest(witness["support_payload"]) == witness["recorded_digest"]
    v_pi = witness["replay_output"] == witness["recorded_output"]
    v_gamma = all(t in witness["allow_list"] for t in witness["tool_calls"])
    v_entail = witness["claim"].strip().lower() in witness["replay_output"].strip().lower()
    return {"V_H": v_h, "V_Pi": v_pi, "V_Gamma": v_gamma, "V_entail": v_entail}


def check_predicate(witness: dict) -> bool:
    return all(evaluate_channels(witness).values())


def assert_single_channel_failure(witness: dict) -> dict:
    """Raise unless exactly the target channel fails and the other three pass."""
    res = evaluate_channels(witness)
    failed = [c for c, ok in res.items() if not ok]
    target = witness.get("target_channel")
    if target is None:
        if failed:
            raise AssertionError(f"valid witness failed {failed}")
        return res
    if failed != [target]:
        raise AssertionError(
            f"{witness['witness_id']}: expected exactly ['{target}'] to fail, observed {failed}"
        )
    return res


def verify_suite() -> dict:
    """Non-redundancy by exhibition: Check is strictly stronger than every proper sub-conjunction."""
    out = {"provenance_class": PROVENANCE_CLASS, "witnesses": []}
    assert_single_channel_failure(WitnessGenerator.generate_valid())
    for w in WitnessGenerator.generate_all():
        res = assert_single_channel_failure(w)
        out["witnesses"].append({"witness_id": w["witness_id"],
                                 "target_channel": w["target_channel"],
                                 "channels": res, "accepted": check_predicate(w)})
    out["all_single_channel"] = True
    out["separates_all_channels"] = sorted(
        x["target_channel"] for x in out["witnesses"]) == sorted(CHANNELS)
    return out
