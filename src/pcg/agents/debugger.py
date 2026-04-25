"""
Debugger agent.

Two functions, both built on machinery already in pcg.responsibility and
pcg.risk:

    1. Diagnosis: for each component in {evidence, tool_call, delegation},
       compute Resp_hat with Hoeffding CI. Flag the top-1 component as the
       likely root cause (Theorem 3, part i).

    2. Control: given the calibrated risk r(b, Z), choose an action from
       {Answer, Verify, Escalate, Refuse} via the piecewise-threshold
       policy (Theorem 3, part ii).

The Debugger is invoked by the orchestrator either:
    - after a clean Verifier pass, in R3/R4 sweeps where we want diagnosis
      and risk-policy data on every example
    - after a failed Verifier pass with no retries left, to localize the
      failure for downstream analysis

Because the Debugger is the most expensive node (Hoeffding CIs need M=50-200
replays), it should NOT be enabled in R1 efficiency runs. The orchestrator
config has `enable_debugger` for exactly this reason.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from pcg.checker import Checker
from pcg.graph import ActionNode, NodeType
from pcg.orchestrator.langgraph_flow import OrchestratorConfig, PCGState
from pcg.responsibility import (
    ResponsibilityEstimator,
    rank_recovery_prob,
)
from pcg.risk import Action, CostModel, ThresholdPolicy, posterior_risk


@dataclass
class DebuggerConfig:
    n_replays: int = 50              # M in Theorem 3(i)
    alpha: float = 0.05              # 1 - confidence for Hoeffding CI
    component_types: tuple[str, ...] = ("truth", "tool", "delegation")
    use_paired_replay: bool = True
    rho: float = 1.0                 # passed to posterior_risk; default 1 = independence

    # Cost-model parameters (Theorem 3 part ii)
    risk_lambda: float = 1.0
    h_fa: float = 1.0
    h_ref: float = 0.0


def _collect_component_ids(state: PCGState, cfg: DebuggerConfig) -> list[str]:
    """Pick the components whose Resp we want to estimate.

    By default: every TruthNode in the certificate's evidence + every ToolCall
    + every Delegation. Capped to a reasonable number to bound replay cost.
    """
    if state.certificate is None:
        return []

    ids: list[str] = []
    if "truth" in cfg.component_types:
        ids.extend(state.certificate.claim_cert.evidence_ids)
    if "tool" in cfg.component_types:
        ids.extend(state.certificate.exec_cert.tool_call_ids)
    if "delegation" in cfg.component_types:
        ids.extend(state.certificate.exec_cert.delegation_ids)
    return list(dict.fromkeys(ids))   # de-duplicate, preserve order


def build_default_debugger(
    *,
    checker: Checker,
    cfg: OrchestratorConfig | None = None,
    debugger_cfg: DebuggerConfig | None = None,
) -> Callable[[PCGState], PCGState]:
    """Return a debugger callable for the orchestrator."""
    orch_cfg = cfg or OrchestratorConfig()
    dbg_cfg = debugger_cfg or DebuggerConfig(
        risk_lambda=orch_cfg.risk_lambda,
        h_fa=orch_cfg.h_fa,
        h_ref=orch_cfg.h_ref,
    )

    def debugger(state: PCGState) -> PCGState:
        with state.meter.phase("debugger"):
            if state.certificate is None or state.check_result is None:
                return state

            # ---- 1. Responsibility estimation ----
            comp_ids = _collect_component_ids(state, dbg_cfg)
            if comp_ids:
                est = ResponsibilityEstimator(
                    checker=checker,
                    n_replays=dbg_cfg.n_replays,
                    alpha=dbg_cfg.alpha,
                    paired=dbg_cfg.use_paired_replay,
                )
                results = est.estimate_many(
                    state.certificate, state.graph, comp_ids,
                )
                state.responsibility = {r.component_id: r.estimate for r in results}
                # Stash the full structured results for downstream R3 reporting
                state.meta["responsibility_results"] = [r.to_dict() for r in results]

                # Top-1 root cause + theoretical rank-recovery guarantee
                if results:
                    sorted_r = sorted(results, key=lambda r: r.estimate, reverse=True)
                    top = sorted_r[0]
                    margin = top.estimate - (sorted_r[1].estimate if len(sorted_r) > 1 else 0.0)
                    state.meta["debugger_top1"] = top.component_id
                    state.meta["debugger_margin"] = margin
                    state.meta["debugger_rank_recovery_prob"] = rank_recovery_prob(
                        n_replays=dbg_cfg.n_replays,
                        n_components=len(results),
                        margin=margin,
                        alpha_family=dbg_cfg.alpha,
                    )

            # ---- 2. Risk-aware action choice (Theorem 3 part ii) ----
            cm = CostModel(
                lam=dbg_cfg.risk_lambda,
                h_fa=dbg_cfg.h_fa,
                h_ref=dbg_cfg.h_ref,
            )
            # In single-branch mode, posterior_risk falls back to (1 - p).
            # Multi-branch consensus is computed by the experiment harness for R2.
            r = posterior_risk(
                confidences=[state.certificate.confidence],
                pass_flags=[state.check_result.passed],
                rho=dbg_cfg.rho,
            )
            policy = ThresholdPolicy(cost_model=cm)
            chosen = policy.choose(r)
            state.chosen_action = chosen.value
            state.meta["posterior_risk"] = r
            state.meta["threshold_crossings"] = [
                (round(t, 4), b.value, a.value) for (t, b, a) in policy.thresholds()
            ]

            # Record debugger action in the graph
            state.graph.add_node(ActionNode(
                action="debug", agent_id="debugger",
                args={"chosen_action": chosen.value, "posterior_risk": round(r, 6),
                      "n_components_diagnosed": len(comp_ids)},
            ))
        return state

    return debugger
