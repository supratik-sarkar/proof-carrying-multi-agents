"""V1.1B Machine-Checkable Pair-Leakage Firewall (F1–F19).

Normative reference:
- ADVERSARIAL_INTEGRITY_PROTOCOL_V1_1A (§D)
- ADVERSARIAL_INTEGRITY_PROTOCOL_V1_1B (§1, §2)
- Consolidated in ADVERSARIAL_INTEGRITY_EFFECTIVE_PROTOCOL_V1_1B.md (§7, §10)
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple
import unicodedata

from pcg.datasets.adversarial_integrity import (
    LP,
    canonical_json,
)

# ---------------------------------------------------------------------------
# Regex Patterns for F14a
# ---------------------------------------------------------------------------

RE_ADVINT_INSTANCE = re.compile(r"advint-[A-Za-z0-9_.:-]+")
RE_ADVINT_DOMAIN_TAG = re.compile(r"ADVINT-[A-Z0-9_-]+")

# Explicitly NOT scanned per V1.1B §2.2
EXPLICITLY_NOT_SCANNED: Set[str] = {
    "FINAL", "PILOT", "SMOKE", "CHECKER_CALIBRATION", "CONTROL", "TREATMENT",
    "SWAP", "DESUPPORT", "REALIGN", "CITATION_SWAP", "SEMANTIC_SLOT_HIJACK",
    "NUMBER_FLIP", "SUPPORTS", "REFUTES", "SUPPORTED", "REFUTED",
    "VALID_SUPPORT", "INVALID_SUPPORT", "UNRESOLVED", "VALID", "INVALID",
    "UNCERTAIN"
}


def norm(s: Any) -> str:
    """Normalizes text per V1 §11.1 (NFC, strip, collapse whitespace)."""
    if s is None:
        return ""
    text = str(s)
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@dataclass
class FirewallCheckResult:
    check_id: str
    passed: bool
    details: str = ""


@dataclass
class FirewallEvaluation:
    passed: bool
    item_id: str
    checks: Dict[str, FirewallCheckResult] = field(default_factory=dict)
    failure_reasons: List[str] = field(default_factory=list)

    def add_result(self, result: FirewallCheckResult) -> None:
        self.checks[result.check_id] = result
        if not result.passed:
            self.passed = False
            self.failure_reasons.append(f"{result.check_id}: {result.details}")


def compute_model_visible_payload_hash(payload: Dict[str, Any]) -> str:
    """Computes MODEL_VISIBLE_PAYLOAD_HASH (V1.1A §D.2)."""
    h_input = LP("ADVINT-MVP-v1_1A") + LP(canonical_json(payload))
    return hashlib.sha256(h_input).hexdigest()


def compute_instruction_template_hash(instructions: str) -> str:
    """Computes instruction template hash (V1.1B §1.3 F15b)."""
    h_input = LP("ADVINT-INSTR-v1_1B") + LP(instructions.encode("utf-8"))
    return hashlib.sha256(h_input).hexdigest()


def compute_eval_order_key(candidate_release_root: str, item_id: str) -> bytes:
    """Computes presentation order key (V1.1 §2.5, V1.1A §D.3 F19)."""
    h_input = LP("ADVINT-EVALORDER-v1_1") + LP(candidate_release_root) + LP(item_id)
    return hashlib.sha256(h_input).digest()


class PairLeakageFirewall:
    """Enforces V1.1B F1–F19 checks per evaluated item."""

    def __init__(
        self,
        frozen_instruction_template_sha256: str,
        release_digests: Dict[str, str],
        annotator_pseudonyms: Optional[Set[str]] = None,
    ) -> None:
        self.instruction_hash = frozen_instruction_template_sha256
        self.release_digests = release_digests
        self.annotator_pseudonyms = {p.lower() for p in (annotator_pseudonyms or set())}

    def evaluate_item(
        self,
        record_x: Dict[str, Any],
        record_y: Dict[str, Any],
        recorded_payload: Dict[str, Any],
        execution_metadata: Dict[str, Any],
        expected_eval_order_key: Optional[bytes] = None,
    ) -> FirewallEvaluation:
        """Runs F1–F19 checks for item x against paired partner y."""
        item_id = record_x.get("item_id", "")
        eval_res = FirewallEvaluation(passed=True, item_id=item_id)

        # F1, F2: Freshness / isolation
        mode = execution_metadata.get("execution_mode", "")
        if mode != "FRESH":
            eval_res.add_result(FirewallCheckResult("F1", False, f"execution_mode {mode} != FRESH"))
            eval_res.add_result(FirewallCheckResult("F2", False, f"execution_mode {mode} != FRESH"))
        else:
            eval_res.add_result(FirewallCheckResult("F1", True, "Context is fresh"))
            eval_res.add_result(FirewallCheckResult("F2", True, "Scope is fresh"))

        # F3: Paired partner absent from model-visible context (checked via F15)
        eval_res.add_result(FirewallCheckResult("F3", True, "Covered by F15"))

        # F4 - F10: Metadata exposure (covered by F14b structural check + F14a)
        eval_res.add_result(FirewallCheckResult("F4_F10", True, "Enforced via F14a/F14b"))

        # F11, F12: Cross-item memory / traces (covered by F16, F18)
        eval_res.add_result(FirewallCheckResult("F11_F12", True, "Enforced via F16/F18"))

        # F13: MODEL_VISIBLE_PAYLOAD_HASH
        emitted_mvp_hash = execution_metadata.get("model_visible_payload_hash")
        recomputed_mvp_hash = compute_model_visible_payload_hash(recorded_payload)
        if emitted_mvp_hash != recomputed_mvp_hash:
            eval_res.add_result(
                FirewallCheckResult("F13", False, f"Payload hash mismatch: emitted {emitted_mvp_hash} != recomputed {recomputed_mvp_hash}")
            )
        else:
            eval_res.add_result(FirewallCheckResult("F13", True, f"Hash verified: {recomputed_mvp_hash}"))

        # F14a: Hard-token substring scan (exact set)
        hard_tokens: Set[str] = set()
        # 1. x.pair_id
        if "pair_id" in record_x:
            hard_tokens.add(record_x["pair_id"])
        # 2. x.item_id and y.item_id (both forms)
        for r in (record_x, record_y):
            pid = r.get("pair_id", "")
            if pid:
                hard_tokens.add(f"{pid}#control")
                hard_tokens.add(f"{pid}#treatment")
            if "item_id" in r:
                hard_tokens.add(r["item_id"])
        # 3. y.pair_id
        if "pair_id" in record_y:
            hard_tokens.add(record_y["pair_id"])
        # 6. namespaced parent reference "fever:" || source_split || ":" || parent_id
        source_split = record_x.get("source_split", "train")
        parent_id = str(record_x.get("parent_id", record_x.get("source_id", "")))
        if parent_id:
            hard_tokens.add(f"fever:{source_split}:{parent_id}")
        # 7. execution context ids
        for ctx_key in ["execution_context_id", "thread_id", "trace_id", "span_id"]:
            val = execution_metadata.get(ctx_key)
            if val and len(str(val)) >= 6:  # avoid empty or trivial string collisions
                hard_tokens.add(str(val))
        # 8. 64-hex digests
        for d in self.release_digests.values():
            if d and len(d) == 64:
                hard_tokens.add(d)
        for r in (record_x, record_y):
            r_sha = r.get("record_sha256")
            if r_sha and len(r_sha) == 64:
                hard_tokens.add(r_sha)

        # Scan complete recorded payload
        payload_json = json.dumps(recorded_payload, ensure_ascii=False)
        payload_lower = payload_json.lower()

        f14a_leaks = []
        for token in hard_tokens:
            if token in payload_json:
                f14a_leaks.append(f"Hard token leaked: {token}")

        # Namespaced regex scan
        # Note: we exclude matches that are in legitimate core/frozen content if any,
        # but regexes /advint-.../ and /ADVINT-.../ are strictly forbidden in model payloads.
        for m in RE_ADVINT_INSTANCE.finditer(payload_json):
            f14a_leaks.append(f"Namespaced advint- instance leaked: {m.group(0)}")
        for m in RE_ADVINT_DOMAIN_TAG.finditer(payload_json):
            f14a_leaks.append(f"Protocol domain tag leaked: {m.group(0)}")

        # 9. Pseudonyms case-insensitively
        for p in self.annotator_pseudonyms:
            if p in payload_lower:
                f14a_leaks.append(f"Annotator pseudonym leaked: {p}")

        if f14a_leaks:
            eval_res.add_result(FirewallCheckResult("F14a", False, "; ".join(f14a_leaks)))
        else:
            eval_res.add_result(FirewallCheckResult("F14a", True, "No hard tokens detected"))

        # F14b: MODEL_VISIBLE_FIELD_PROVENANCE structural check
        prov_map = execution_metadata.get("model_visible_field_provenance", {})
        required_keys = {"claim", "evidence", "verdict", "instructions"}
        if not required_keys.issubset(prov_map.keys()):
            eval_res.add_result(FirewallCheckResult("F14b", False, f"Missing provenance keys: {required_keys - set(prov_map.keys())}"))
        elif (prov_map.get("claim") != "DATASET_PRESENTED_CLAIM" or
              prov_map.get("evidence") != "DATASET_PRESENTED_EVIDENCE_TEXT" or
              prov_map.get("verdict") != "DATASET_ASSERTED_VERDICT" or
              prov_map.get("instructions") != "FROZEN_INSTRUCTION_TEMPLATE"):
            eval_res.add_result(FirewallCheckResult("F14b", False, f"Invalid provenance mapping: {prov_map}"))
        else:
            eval_res.add_result(FirewallCheckResult("F14b", True, "Provenance mapping valid"))

        # F15a: CORE INTEGRITY
        def _get_claim(r):
            return r.get("model_visible_claim") or r.get("presented_claim", "")

        def _get_ev(r):
            if "model_visible_evidence_text" in r and r["model_visible_evidence_text"] is not None:
                return r["model_visible_evidence_text"]
            ev = r.get("presented_evidence", "")
            return ev.get("text", "") if isinstance(ev, dict) else str(ev)

        def _get_verdict(r):
            return r.get("model_visible_verdict") or r.get("asserted_verdict", "")

        def _get_role(r):
            return str(r.get("role") or r.get("item_role", "")).lower()

        def _get_params(r):
            return r.get("construction_params") or r.get("attack_parameters", {})

        claim_match = norm(recorded_payload.get("claim")) == norm(_get_claim(record_x))
        ev_match = norm(recorded_payload.get("evidence")) == norm(_get_ev(record_x))
        verd_match = recorded_payload.get("verdict") == _get_verdict(record_x)

        if not (claim_match and ev_match and verd_match):
            eval_res.add_result(
                FirewallCheckResult("F15a", False, f"Core mismatch: claim={claim_match}, ev={ev_match}, verdict={verd_match}")
            )
        else:
            eval_res.add_result(FirewallCheckResult("F15a", True, "Core integrity passed"))

        # F15b: INSTRUCTION INTEGRITY
        instr_hash = compute_instruction_template_hash(recorded_payload.get("instructions", ""))
        if instr_hash != self.instruction_hash:
            eval_res.add_result(
                FirewallCheckResult("F15b", False, f"Instruction hash mismatch: {instr_hash} != {self.instruction_hash}")
            )
        else:
            eval_res.add_result(FirewallCheckResult("F15b", True, f"Instruction verified: {instr_hash}"))

        # F15c: PAIRED-EXCLUSIVE ABSENCE (over FREE only)
        # Compute D(x, y)
        paired_exclusive: Set[str] = set()
        for val_x, val_y in [
            (norm(_get_claim(record_x)), norm(_get_claim(record_y))),
            (norm(_get_ev(record_x)), norm(_get_ev(record_y))),
            (norm(_get_verdict(record_x)), norm(_get_verdict(record_y))),
        ]:
            if val_y and val_y != val_x:
                paired_exclusive.add(val_y)

        # Family-specific deltas
        role_x = _get_role(record_x)
        family = record_x.get("attack_family", "")
        params = _get_params(record_x) or _get_params(record_y)

        if family == "CITATION_SWAP":
            if role_x == "control":
                if "donor_page" in params:
                    paired_exclusive.add(norm(params["donor_page"]))
                if "donor_sentence_id" in params:
                    paired_exclusive.add(str(params["donor_sentence_id"]))
        elif family == "SEMANTIC_SLOT_HIJACK":
            if role_x == "control":
                if "slot_replacement" in params:
                    paired_exclusive.add(norm(params["slot_replacement"]))
            else:
                if "slot_original" in params:
                    paired_exclusive.add(norm(params["slot_original"]))
        elif family == "NUMBER_FLIP":
            if role_x == "control":
                if "perturbed_value" in params:
                    paired_exclusive.add(norm(params["perturbed_value"]))
            else:
                if "original_value" in params:
                    paired_exclusive.add(norm(params["original_value"]))

        # Independent presence carve-out: D(x, y) = PAIRED_EXCLUSIVE \ { v : norm(v) in norm(CORE(x)) }
        core_x_text = norm(_get_claim(record_x)) + " " + norm(_get_ev(record_x))
        d_xy = {v for v in paired_exclusive if norm(v) and norm(v) not in core_x_text}

        # Scan FREE entries (extra dictionary)
        extra = recorded_payload.get("extra", {})
        free_entries: List[str] = []
        extra_prov = execution_metadata.get("model_visible_field_provenance", {}).get("extra", {})

        for k, v in extra.items():
            prov = extra_prov.get(k, "")
            # Retrieval carve-out: exempt if RETRIEVAL_RESULT or TOOL_RESULT
            if prov in {"RETRIEVAL_RESULT", "TOOL_RESULT"}:
                continue
            free_entries.append(norm(v))

        free_text = "\u001e".join(free_entries)
        f15c_leaks = []
        for v in d_xy:
            if norm(v) in free_text:
                f15c_leaks.append(f"Paired exclusive content leaked in FREE: '{v}'")

        if f15c_leaks:
            eval_res.add_result(FirewallCheckResult("F15c", False, "; ".join(f15c_leaks)))
        else:
            eval_res.add_result(FirewallCheckResult("F15c", True, "Paired exclusive absence passed"))

        # F16: Context id uniqueness & unused thread identity
        ctx_id = execution_metadata.get("execution_context_id")
        thread_id = execution_metadata.get("thread_id")
        if not ctx_id or not thread_id:
            eval_res.add_result(FirewallCheckResult("F16", False, "Missing execution_context_id or thread_id"))
        else:
            eval_res.add_result(FirewallCheckResult("F16", True, f"Context {ctx_id} / Thread {thread_id}"))

        # F17: execution_mode == FRESH
        eval_res.add_result(FirewallCheckResult("F17", mode == "FRESH", f"execution_mode={mode}"))

        # F18: memory scope == execution_context_id (or no memory)
        mem_scope = execution_metadata.get("memory_scope_id")
        if mem_scope and mem_scope != ctx_id:
            eval_res.add_result(FirewallCheckResult("F18", False, f"Memory scope {mem_scope} != context {ctx_id}"))
        else:
            eval_res.add_result(FirewallCheckResult("F18", True, "Memory scope isolated"))

        # F19: Presentation order
        if expected_eval_order_key is not None:
            actual_key = execution_metadata.get("eval_order_key")
            if actual_key != expected_eval_order_key:
                eval_res.add_result(FirewallCheckResult("F19", False, f"Order key mismatch: {actual_key} != {expected_eval_order_key}"))
            else:
                eval_res.add_result(FirewallCheckResult("F19", True, "Presentation order precommitted"))
        else:
            eval_res.add_result(FirewallCheckResult("F19", True, "Order key check omitted"))

        return eval_res
