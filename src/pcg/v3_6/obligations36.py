"""Bind the frozen v3.5 obligation pipeline so S_i/K_i are persisted for phi.

Obligations are task-derived via TaskInputObligationAdapter. Evidence slots are
the prover's retrieved windows at the frozen K0. The scorer is the pinned NLI.
Nothing here reads gold, reference or acceptance.
"""
from typing import Any, Dict, List, Mapping, Tuple

from pcg.v3_5.obligations import (K0_SLOTS, TaskInputObligationAdapter,
                                  build_evidence_slots,
                                  derive_obligation_hypotheses,
                                  evaluate_obligations_with_trace)

OBLIGATION_BINDING_VERSION = "PCG_MAS_V3_6_OBLIGATIONS_V1"


class ObligationBindingError(RuntimeError):
    pass


def nli_scorer(verifier):
    """scorer_fn(premise, hypothesis) -> {p_entailment, p_contradiction, p_neutral}."""
    def _score(premise: str, hypothesis: str) -> Dict[str, float]:
        out = verifier.score_pair(premise, hypothesis)
        if isinstance(out, Mapping):
            g = lambda *ks: next((float(out[k]) for k in ks if k in out), None)
            pe, pc, pn = g("p_entailment", "entailment", "ENTAILMENT"), \
                         g("p_contradiction", "contradiction", "CONTRADICTION"), \
                         g("p_neutral", "neutral", "NEUTRAL")
            if pe is None or pc is None:
                raise ObligationBindingError(f"NLI_SHAPE_UNRECOGNISED:{sorted(out)[:6]}")
            return {"p_entailment": pe, "p_contradiction": pc,
                    "p_neutral": pn if pn is not None else max(0.0, 1.0 - pe - pc)}
        if isinstance(out, (list, tuple)) and len(out) >= 3:
            pc, pn, pe = float(out[0]), float(out[1]), float(out[2])
            return {"p_entailment": pe, "p_contradiction": pc, "p_neutral": pn}
        raise ObligationBindingError(f"NLI_RETURN_TYPE:{type(out).__name__}")
    return _score


def obligation_scores_for(*, dataset_id: str, example_id: str, task_input: Dict[str, Any],
                          prompt_text: str, output_text: str,
                          evidence_windows: List[str], verifier,
                          k0: int = K0_SLOTS) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any]]:
    """Return ({obligation_id: {S_i, K_i, is_critical}}, provenance)."""
    obls = TaskInputObligationAdapter.get_obligations(dataset_id, example_id, task_input)
    if not obls:
        raise ObligationBindingError(f"NO_OBLIGATIONS_DERIVED:{dataset_id}/{example_id}")
    hyps = derive_obligation_hypotheses(obls, prompt_text, output_text)
    slots = build_evidence_slots(evidence_windows, k0=k0)
    scores, trace = evaluate_obligations_with_trace(hyps, slots, nli_scorer(verifier))
    prov = {
        "binding_version": OBLIGATION_BINDING_VERSION,
        "obligations_sha256": TaskInputObligationAdapter.hash_obligations(obls),
        "obligation_count": len(obls), "hypothesis_count": len(hyps),
        "k0_slots": k0, "real_slot_count": sum(1 for s in slots if getattr(s, "slot_type", "REAL") == "REAL"),
        "trace_records": len(trace),
        "source": "TASK_DERIVED_NEVER_REFERENCE_DERIVED",
    }
    for oid, sc in scores.items():
        if not {"S_i", "K_i"} <= set(sc):
            raise ObligationBindingError(f"OBLIGATION_SCORE_INCOMPLETE:{oid}")
    return scores, prov
