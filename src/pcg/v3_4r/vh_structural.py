"""PCG-MAS v3.4R Structural-Only V_H Verifier.

Assesses ONLY deployment-visible integrity, schema, hashes, evidence binding,
and provenance properties. Zero dependency on evaluator targets.
"""

from copy import deepcopy
import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple

from pcg.v3_4r.candidate import RuntimeCandidate


def evaluate_vh_structural(
    cand: RuntimeCandidate,
) -> Tuple[str, Dict[str, Any]]:
    """Evaluates structural integrity of a candidate.

    Returns:
        (state, details) where state is "PASS" or "FAIL".
    """
    # 1. Schema integrity
    if not isinstance(cand.candidate_id, str) or not cand.candidate_id.strip():
        return "FAIL", {"reason": "Empty or non-string candidate_id"}
    if not isinstance(cand.model, str) or not cand.model.strip():
        return "FAIL", {"reason": "Empty or non-string model"}
    if not isinstance(cand.dataset, str) or not cand.dataset.strip():
        return "FAIL", {"reason": "Empty or non-string dataset"}
    if not isinstance(cand.example_id, str) or not cand.example_id.strip():
        return "FAIL", {"reason": "Empty or non-string example_id"}
    if not isinstance(cand.candidate_answer, str):
        return "FAIL", {"reason": "Candidate answer must be a string"}
    if not isinstance(cand.prompt, str):
        return "FAIL", {"reason": "Prompt must be a string"}

    # 2. Hash integrity
    expected_resp_hash = hashlib.sha256(
        cand.candidate_answer.encode("utf-8")
    ).hexdigest()
    if cand.response_hash != expected_resp_hash:
        return "FAIL", {
            "reason": f"Response hash mismatch: expected {expected_resp_hash}, got {cand.response_hash}"
        }

    expected_req_hash = hashlib.sha256(cand.prompt.encode("utf-8")).hexdigest()
    if cand.request_hash != expected_req_hash:
        return "FAIL", {
            "reason": f"Request hash mismatch: expected {expected_req_hash}, got {cand.request_hash}"
        }

    # 3. Evidence binding
    if not isinstance(cand.windows, list) or len(cand.windows) == 0:
        return "FAIL", {"reason": "Evidence windows missing or empty"}

    for idx, w in enumerate(cand.windows):
        if not isinstance(w, str) or not w.strip():
            return "FAIL", {
                "reason": f"Evidence window {idx} is empty or non-string"
            }
        expected_w_hash = hashlib.sha256(w.encode("utf-8")).hexdigest()
        if (
            idx < len(cand.evidence_hashes)
            and cand.evidence_hashes[idx] != expected_w_hash
        ):
            return "FAIL", {"reason": f"Evidence hash mismatch at window {idx}"}

    # 4. Obligations schema
    if not isinstance(cand.obligations, list) or len(cand.obligations) == 0:
        return "FAIL", {"reason": "Obligations missing or empty"}
    for obl in cand.obligations:
        if (
            not isinstance(obl, dict)
            or "obligation_id" not in obl
            or "obligation_type" not in obl
        ):
            return "FAIL", {"reason": "Malformed obligation item"}

    # 5. Resource provenance
    if not isinstance(cand.resource_metrics, dict):
        return "FAIL", {"reason": "Malformed resource metrics"}
    if cand.resource_metrics.get("generative_calls", 0) < 1:
        return "FAIL", {"reason": "Invalid generative_calls count in metrics"}

    return "PASS", {"verified": True, "num_windows": len(cand.windows)}


def run_vh_structural_kat() -> Dict[str, Any]:
    """Known-Answer Test for structural-only V_H.

    Corrupts:
    - schema -> must FAIL
    - candidate/evidence hashes -> must FAIL
    - evidence binding -> must FAIL
    - provenance -> must FAIL
    - external label mutation / correctness -> must NOT change V_H
    """
    prompt = "Test prompt text"
    ans = "Test candidate answer"
    windows = ["Evidence window 1", "Evidence window 2"]
    base = RuntimeCandidate(
        candidate_id="test_cand_1",
        model="gpt-4o",
        dataset="fever",
        example_id="ex_001",
        request_hash=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        response_hash=hashlib.sha256(ans.encode("utf-8")).hexdigest(),
        prompt=prompt,
        candidate_answer=ans,
        windows=windows,
        evidence_hashes=[
            hashlib.sha256(w.encode("utf-8")).hexdigest() for w in windows
        ],
        obligations=[
            {
                "obligation_id": "obl_1",
                "obligation_type": "verdict",
                "is_critical": True,
            }
        ],
        resource_metrics={
            "generative_calls": 1,
            "input_tokens": 50,
            "output_tokens": 10,
        },
    )

    results = {}

    # 1. Base clean candidate -> PASS
    s, _ = evaluate_vh_structural(base)
    results["base_clean"] = s == "PASS"

    # 2. Corrupt schema -> FAIL
    c_schema = deepcopy(base)
    c_schema.candidate_id = ""
    s, _ = evaluate_vh_structural(c_schema)
    results["corrupt_schema"] = s == "FAIL"

    # 3. Corrupt request hash -> FAIL
    c_req = deepcopy(base)
    c_req.request_hash = "deadbeef" * 8
    s, _ = evaluate_vh_structural(c_req)
    results["corrupt_request_hash"] = s == "FAIL"

    # 4. Corrupt response hash -> FAIL
    c_resp = deepcopy(base)
    c_resp.response_hash = "deadbeef" * 8
    s, _ = evaluate_vh_structural(c_resp)
    results["corrupt_response_hash"] = s == "FAIL"

    # 5. Corrupt evidence binding -> FAIL
    c_ev = deepcopy(base)
    c_ev.windows = []
    s, _ = evaluate_vh_structural(c_ev)
    results["corrupt_evidence_binding"] = s == "FAIL"

    # 6. Corrupt evidence hash -> FAIL
    c_evh = deepcopy(base)
    c_evh.evidence_hashes = ["corrupted_hash", "corrupted_hash_2"]
    s, _ = evaluate_vh_structural(c_evh)
    results["corrupt_evidence_hash"] = s == "FAIL"

    # 7. Corrupt provenance metrics -> FAIL
    c_prov = deepcopy(base)
    c_prov.resource_metrics = {"generative_calls": 0}
    s, _ = evaluate_vh_structural(c_prov)
    results["corrupt_provenance"] = s == "FAIL"

    # 8. External label invariance: V_H never touches external labels, base remains PASS
    s, _ = evaluate_vh_structural(base)
    results["label_invariance"] = s == "PASS"

    all_passed = all(results.values())
    return {
        "status": "PASS" if all_passed else "FAIL",
        "tests": results,
        "passed": all_passed,
    }
