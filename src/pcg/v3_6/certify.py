"""Certification of one already-generated candidate.

Generation happened once, in the v2.3.2 dispatcher. Here the prover's own code
path is re-run with a ReplayBackend, so the certificate is constructed by the
real prover over the real evidence/retrieval/prompt, with zero new generation.
"""
from typing import Any, Dict, Optional

from pcg.agents.prover import ProverConfig, build_default_prover
from pcg.checker import Checker, NLIEntailment, TokenOverlapEntailment, build_default_replayer
from pcg.orchestrator.langgraph_flow import PCGState

from .hashing36 import sha256_json, sha256_text
from .labels import assert_no_evaluator_labels, strip_for_certification
from .nli_binding import bind_verifier, binding_record
from .prep import prepare
from .replay_backend import CONFIDENCE_SOURCE, PromptIdentityMismatch, ReplayBackend

CERT_SCHEMA = "PCG_MAS_V3_6_CERTIFICATION_RECORD_V1"
GEN_REQUIRED = ("observation_id", "model_id", "dataset_id", "provider", "api_model_id",
                "prompt_sha256", "evidence_sha256", "candidate_text", "candidate_sha256")


class CertificationError(RuntimeError):
    pass


def build_checker(repo_root, *, nli_verifier=None, entail_threshold: float = 0.5) -> Checker:
    """Checker whose V_entail runs on the pinned repository NLI snapshot."""
    v = nli_verifier if nli_verifier is not None else bind_verifier(repo_root)

    def nli_fn(premise: str, hypothesis: str) -> bool:
        out = v.score_pair(premise, hypothesis)
        if isinstance(out, dict):
            for k in ("p_entailment", "entailment", "p_entail", "entail", "ENTAILMENT"):
                if k in out:
                    return float(out[k]) >= entail_threshold
            raise CertificationError(f"NLI_SCORE_SHAPE_UNRECOGNISED:{sorted(out)[:6]}")
        if isinstance(out, (list, tuple)):
            return float(out[0]) >= entail_threshold
        return bool(out)

    return Checker(entailment=NLIEntailment(base=TokenOverlapEntailment(), nli_fn=nli_fn),
                   replayer=build_default_replayer(), strict=True)


def certify_observation(*, example, generation_artifact: Dict[str, Any], repo_root,
                        checker: Checker, config: Optional[ProverConfig] = None) -> Dict[str, Any]:
    g = generation_artifact
    missing = [k for k in GEN_REQUIRED if k not in g]
    if missing:
        raise CertificationError(f"GENERATION_ARTIFACT_MISSING_FIELDS:{missing}")
    if not g["candidate_text"]:
        raise CertificationError("EMPTY_CANDIDATE_TEXT")
    if sha256_text(g["candidate_text"]) != g["candidate_sha256"]:
        raise CertificationError("CANDIDATE_SHA_MISMATCH")

    cfg = config or ProverConfig()

    # 1. Re-derive the prompt through the single shared preparation path.
    prep = prepare(example, cfg)
    if prep["prompt_sha256"] != g["prompt_sha256"]:
        raise PromptIdentityMismatch(
            f"PROMPT_SHA_MISMATCH_AT_CERTIFICATION expected={g['prompt_sha256']} got={prep['prompt_sha256']}")
    if prep["evidence_sha256"] != g["evidence_sha256"]:
        raise CertificationError(
            f"EVIDENCE_SHA_MISMATCH expected={g['evidence_sha256']} got={prep['evidence_sha256']}")

    # 2. Replay the recorded generation through the prover. No network.
    backend = ReplayBackend(
        expected_prompt_sha256=g["prompt_sha256"], text=g["candidate_text"],
        tokens_in=int(g.get("tokens_in", 0)), tokens_out=int(g.get("tokens_out", 0)),
        finish=g.get("finish", "stop"), provider=g["provider"], api_model_id=g["api_model_id"],
    )
    state = PCGState(example=example)
    build_default_prover(backend=backend, config=cfg)(state)
    if backend.calls != 1:
        raise CertificationError(f"REPLAY_GENERATE_CALLS={backend.calls}")
    if state.certificate is None:
        raise CertificationError("NO_CERTIFICATE_CONSTRUCTED")

    # 3. Real acceptance predicate.
    result = checker.check(state.certificate, state.graph)
    factors = {"V_H": result.V_H, "V_Pi": result.V_Pi, "V_Gamma": result.V_Gamma,
               "V_entail": result.V_entail}
    accepted = bool(result.passed)

    record = {
        "schema": CERT_SCHEMA,
        "observation_id": g["observation_id"], "model_id": g["model_id"],
        "dataset_id": g["dataset_id"], "source_lineage_id": g.get("source_lineage_id"),
        "provider": g["provider"], "api_model_id": g["api_model_id"],
        "prompt_sha256": g["prompt_sha256"], "evidence_sha256": g["evidence_sha256"],
        "candidate_sha256": g["candidate_sha256"],
        "pcg_accepted": accepted, "factors": factors,
        "check_result": strip_for_certification(result.to_dict()),
        "reasons": list(result.reasons or []),
        "confidence_source": CONFIDENCE_SOURCE, "provider_logprobs_available": False,
        "replay_generate_calls": backend.calls, "replay_network_calls": backend.network_calls,
        "nli_binding": binding_record(repo_root),
        "prover_config": prep["prover_config"],
    }
    record["certification_sha256"] = sha256_json(
        {k: v for k, v in record.items() if k != "certification_sha256"})
    assert_no_evaluator_labels(record, "CERTIFICATION_RECORD")
    return record
