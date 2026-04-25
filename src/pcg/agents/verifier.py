"""
Verifier agent.

The hard work of verification is in `pcg.checker.Checker`; the agent's job
is to (a) run the checker, (b) record the structured CheckResult into the
runtime graph as an audit artifact, and (c) optionally cross-check the
audit log's Merkle prefix to detect silent log truncation by an adversary.

We expose the verifier as a function rather than a class so the orchestrator
can compose it with the others uniformly.
"""
from __future__ import annotations

from typing import Callable

from pcg.checker import CheckResult, Checker
from pcg.commitments import AuditLog
from pcg.graph import ActionNode
from pcg.orchestrator.langgraph_flow import PCGState


def build_default_verifier(
    *,
    checker: Checker,
    audit_log: AuditLog | None = None,
) -> Callable[[PCGState], PCGState]:
    """Return a verifier callable.

    If `audit_log` is provided, the verifier asserts that the prover's
    committed leaves are a prefix of the verifier's local log. This catches
    log-truncation attacks where an adversary modifies the prover's log
    *after* the certificate is issued.
    """

    def verifier(state: PCGState) -> PCGState:
        with state.meter.phase("verifier"):
            if state.certificate is None:
                state.check_result = CheckResult(
                    passed=False, integrity_ok=False,
                    reasons=["no_certificate_to_check"],
                )
                return state

            result = checker.check(state.certificate, state.graph)

            # Optional Merkle prefix check
            if audit_log is not None:
                local_log = AuditLog()
                for d in state.certificate.claim_cert.evidence_digests:
                    local_log.append(d)
                if not audit_log.verify_prefix(local_log):
                    result.passed = False
                    result.integrity_ok = False
                    result.reasons.append("merkle_prefix_check_failed")

            state.check_result = result

            # Record verifier action in the graph for full audit trail
            state.graph.add_node(ActionNode(
                action="verify", agent_id="verifier",
                args={"passed": result.passed,
                      "channels": {
                          "integrity_ok": result.integrity_ok,
                          "replay_ok": result.replay_ok,
                          "entailment_ok": result.entailment_ok,
                          "execution_ok": result.execution_ok,
                      }},
            ))
        return state

    return verifier
