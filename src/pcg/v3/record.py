"""Canonical per-example record (v3.0) -- the single authoritative schema.

No manuscript metric may read from any other schema. Pydantic v2 is used when
available; otherwise a stdlib dataclass with the same field set and validation
is used so the scientific core runs on a bare interpreter.

NULLABILITY RULE: None means NOT MEASURED. It propagates as undefined through
the metric layer and is never coerced to 0. A metric whose denominator is None
or 0 returns None.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .release import METRIC_VERSION, RECORD_SCHEMA_VERSION

PROVENANCE_CLASSES = ("DIRECT", "DERIVED", "MODELLED", "PROTOCOL",
                      "REPLAY", "TEST_FIXTURE", "MOCK", "UNKNOWN", "ENGINEERING_SMOKE")
SYSTEMS = ("nocert", "citation_only", "shieldagent", "agentrr", "pcg_mas")
CONTROLLER_ACTIONS = ("Answer", "Verify", "Escalate", "Refuse")


@dataclass
class PerExampleRecord:
    # -- identity
    record_id: str
    run_id: str
    experiment_id: str                     # A01..A18
    system: str
    provenance_class: str = "TEST_FIXTURE"
    metric_version: str = METRIC_VERSION
    schema_version: str = RECORD_SCHEMA_VERSION
    config_hash: Optional[str] = None
    spec_hash: Optional[str] = None
    cell_id: Optional[str] = None
    dataset: Optional[str] = None
    split: Optional[str] = None
    example_id: Optional[str] = None
    seed: Optional[int] = None
    # -- backend / checker fingerprints
    model_id: Optional[str] = None
    model_revision: Optional[str] = None
    tokenizer_id: Optional[str] = None
    backend_type: Optional[str] = None
    provider_route: Optional[str] = None
    dtype: Optional[str] = None
    quantization: Optional[str] = None
    decoding_config_hash: Optional[str] = None
    prompt_hash: Optional[str] = None
    backend_fingerprint: Optional[str] = None
    checker_fingerprint: Optional[str] = None
    # -- claim and evidence
    claim_id: Optional[str] = None
    evidence_ids: List[str] = field(default_factory=list)
    evidence_hashes: List[str] = field(default_factory=list)
    # -- four conjuncts and acceptance
    v_h: Optional[bool] = None
    v_pi: Optional[bool] = None
    v_gamma: Optional[bool] = None
    v_entail: Optional[bool] = None
    check: Optional[bool] = None
    certificate_hash: Optional[str] = None
    # -- five audit channels (+ multiplicity for union slack)
    int_fail: Optional[bool] = None
    replay_fail: Optional[bool] = None
    drift_fail: Optional[bool] = None
    check_fail: Optional[bool] = None
    cov_gap: Optional[bool] = None
    n_channels_fired: Optional[int] = None
    # -- residual labels (oracle-known only)
    contract_bad: Optional[bool] = None
    eps_tax_label: Optional[bool] = None       # out-of-taxonomy, oracle-known
    eps_src_label: Optional[bool] = None       # source/world-truth failure, independently known
    # -- harm decomposition and outcome
    attempted: Optional[bool] = None
    answered: Optional[bool] = None
    accepted: Optional[bool] = None
    controller_action: Optional[str] = None
    h_support: Optional[float] = None
    h_exec: Optional[float] = None
    h_joint: Optional[float] = None
    utility: Optional[float] = None
    loss_nocert: Optional[float] = None
    loss_pcg: Optional[float] = None
    # -- coverage (two DISTINCT senses; never one column)
    cov_cert: Optional[float] = None
    cov_audit: Optional[float] = None
    # -- redundancy / dependence
    k_redundancy: Optional[int] = None
    branch_ids: List[str] = field(default_factory=list)
    branch_failures: Optional[List[bool]] = None
    # -- responsibility
    resp_scores: Dict[str, float] = field(default_factory=dict)
    resp_top1: Optional[str] = None
    resp_top3: List[str] = field(default_factory=list)
    resp_margin: Optional[float] = None
    unresolved: Optional[bool] = None
    replay_budget_M: Optional[int] = None
    failure_origin_known: Optional[bool] = None
    # -- policy
    policy_bundle_hash: Optional[str] = None
    policy_decision: Optional[str] = None
    # -- cost telemetry (A10 aggregates these; it never re-runs experiments)
    latency_ms: Optional[float] = None
    phase_timings_ms: Dict[str, float] = field(default_factory=dict)
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    model_calls: Optional[int] = None
    retrieval_calls: Optional[int] = None
    tool_calls: Optional[int] = None
    checker_calls: Optional[int] = None
    replay_calls: Optional[int] = None
    billed_cost_usd: Optional[float] = None
    cache_state: Optional[str] = None
    # -- audit sampling / regimes
    stratum_id: Optional[str] = None
    sampling_weight: Optional[float] = None
    corruption: Optional[str] = None
    shift_regime: Optional[str] = None
    injection_regime: Optional[str] = None
    # -- v3.2 execution substrate (additive; RECORD_SCHEMA_VERSION -> 3.2.0)
    execution_mode: Optional[str] = None          # FRESH|RESUME|REPLICATE|REPLAY
    certificate_root: Optional[str] = None        # PCG-CAS-v1 address
    lineage_root: Optional[str] = None
    policy_eval_status: Optional[str] = None      # AVAILABLE|INDETERMINATE
    policy_input_hash: Optional[str] = None
    retry_attempts: Optional[int] = None          # observed, transport-only
    retry_trigger_class: Optional[str] = None     # TRANSPORT|RATE_LIMIT|TIMEOUT
    instrumentation_version: Optional[str] = None
    instrumentation_overhead_ratio: Optional[float] = None
    guardrail_intervention_count: Optional[int] = None
    # -- environment
    host_fingerprint: Optional[str] = None
    device: Optional[str] = None
    source_record_hash: Optional[str] = None
    code_fingerprint: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def validate(self) -> List[str]:
        errs: List[str] = []
        if self.provenance_class not in PROVENANCE_CLASSES:
            errs.append(f"bad provenance_class {self.provenance_class!r}")
        if self.system not in SYSTEMS:
            errs.append(f"bad system {self.system!r}")
        if self.controller_action not in (None,) + CONTROLLER_ACTIONS:
            errs.append(f"bad controller_action {self.controller_action!r}")
        if not (self.experiment_id and self.experiment_id[0] == "A" and self.experiment_id[1:].isdigit()):
            errs.append(f"bad experiment_id {self.experiment_id!r}")
        bits = (self.v_h, self.v_pi, self.v_gamma, self.v_entail)
        if self.check is True and any(b is not True for b in bits):
            errs.append("check=True requires all four conjuncts True (unknown is failure)")
        if self.n_channels_fired is not None:
            got = sum(1 for b in (self.int_fail, self.replay_fail, self.drift_fail,
                                  self.check_fail, self.cov_gap) if b is True)
            if got != self.n_channels_fired:
                errs.append(f"n_channels_fired={self.n_channels_fired} but {got} channels are True")
        if self.answered is False and self.loss_pcg is not None:
            errs.append("refused example must not carry loss_pcg")
        return errs


FIELD_ORDER = [f for f in PerExampleRecord.__dataclass_fields__]  # type: ignore[attr-defined]


def json_schema() -> dict:
    """Emit a JSON Schema for the record (shared with the app via a generated contract)."""
    import typing
    props: Dict[str, Any] = {}
    hints = typing.get_type_hints(PerExampleRecord)
    def jtype(t):
        s = str(t)
        if "bool" in s: base = "boolean"
        elif "int" in s: base = "integer"
        elif "float" in s: base = "number"
        elif "List" in s or "list" in s: return {"type": ["array", "null"], "items": {}}
        elif "Dict" in s or "dict" in s: return {"type": ["object", "null"]}
        else: base = "string"
        return {"type": [base, "null"]}
    for name in FIELD_ORDER:
        props[name] = jtype(hints.get(name, str))
    props["provenance_class"] = {"type": "string", "enum": list(PROVENANCE_CLASSES)}
    props["system"] = {"type": "string", "enum": list(SYSTEMS)}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://pcg-mas/schemas/per_example_record.v3.schema.json",
        "title": "PCG-MAS canonical per-example record (v3.0)",
        "description": ("Keystone record. Every metric, table and figure derives from these "
                        "records; the dependency direction is never reversed. null means NOT "
                        "MEASURED and is never coerced to 0."),
        "type": "object",
        "additionalProperties": False,
        "required": ["record_id", "run_id", "experiment_id", "system",
                     "provenance_class", "metric_version", "schema_version"],
        "properties": props,
    }
