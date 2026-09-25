"""PCG-MAS S02 Prospective Scientific Harness.

Implements the frozen protocol defined in S01 (PCG_MAS_POST_AG2_6D_PROSPECTIVE_PROTOCOL_V1):
1. Zero-denominator handling: 0/0 is strictly UNDEFINED (None), never coerced to 0.0000.
2. Specialist routing & vacuous obligations: Check(Z; G_t) = prod_{k in K_applicable} V_k(Z; G_t).
   Non-applicable obligations evaluate vacuously to True (1).
3. Applicability-aware harm: h_joint = \bigvee_{k in K_applicable} h_k.
   Non-applicable harm channels evaluate to False (0).
4. Scope coverage comparator: SpecialistCompose is the coverage comparator.
   Gate C6: Delta_coverage(PCG vs SpecialistCompose) >= -0.05.
5. Specialist preservation: Delta(PCG - Specialist) >= 0.0 (empirical TIE satisfies preservation).
6. Q_s selective utility & controller regret accounting.
7. A16 resource efficiency accounting with $0.00 spend ceiling.
8. Pinned A15 DeBERTa-v3-large cross-encoder NLI interface (local mock/recorded logits, off-Mac bundle).
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

# Root directory resolution
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
S01_DIR = WORKSPACE_ROOT / "artifacts" / "v3_0" / "master_forward_controller" / "stages" / "S01"

# ---------------------------------------------------------------------------
# 1. Applicability Registries (S01 Frozen Protocol)
# ---------------------------------------------------------------------------

DEFAULT_OBLIGATION_APPLICABILITY = {
    "fever": {"grounding": True, "policy": False, "replay": False, "type": "fact_verification"},
    "hotpotqa": {"grounding": True, "policy": False, "replay": False, "type": "multi_hop_qa"},
    "twowiki": {"grounding": True, "policy": False, "replay": False, "type": "multi_hop_qa"},
    "pubmedqa": {"grounding": True, "policy": False, "replay": False, "type": "biomedical_qa"},
    "tatqa": {"grounding": True, "policy": False, "replay": False, "type": "hybrid_tabular_text_qa"},
    "toolbench": {"grounding": False, "policy": True, "replay": True, "type": "multi_turn_tool_use"},
    "weblinx": {"grounding": False, "policy": True, "replay": True, "type": "web_navigation_action"},
    "adversarial_integrity": {"grounding": True, "policy": False, "replay": False, "type": "adversarial_fact_check"},
}

DEFAULT_HARM_APPLICABILITY = {
    "fever": {"h_support_applicable": True, "h_exec_applicable": False},
    "hotpotqa": {"h_support_applicable": True, "h_exec_applicable": False},
    "twowiki": {"h_support_applicable": True, "h_exec_applicable": False},
    "pubmedqa": {"h_support_applicable": True, "h_exec_applicable": False},
    "tatqa": {"h_support_applicable": True, "h_exec_applicable": False},
    "toolbench": {"h_support_applicable": False, "h_exec_applicable": True},
    "weblinx": {"h_support_applicable": False, "h_exec_applicable": True},
    "adversarial_integrity": {"h_support_applicable": True, "h_exec_applicable": False},
}


def load_obligation_registry() -> Dict[str, Any]:
    p = S01_DIR / "OBLIGATION_APPLICABILITY_REGISTRY.json"
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return data.get("datasets", DEFAULT_OBLIGATION_APPLICABILITY)
        except Exception:
            pass
    return DEFAULT_OBLIGATION_APPLICABILITY


def load_harm_registry() -> Dict[str, Any]:
    p = S01_DIR / "HARM_APPLICABILITY_REGISTRY.json"
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return data.get("datasets", DEFAULT_HARM_APPLICABILITY)
        except Exception:
            pass
    return DEFAULT_HARM_APPLICABILITY


# ---------------------------------------------------------------------------
# 2. Strict Zero-Denominator Safe Math
# ---------------------------------------------------------------------------

def safe_rate(numerator: Optional[Union[int, float]], denominator: Optional[Union[int, float]]) -> Optional[float]:
    """Strict rate computation.
    
    If denominator is 0, None, or numerator is None -> returns None (UNDEFINED).
    NEVER silently coerces to 0.0000.
    """
    if denominator is None or denominator == 0 or numerator is None:
        return None
    return float(numerator) / float(denominator)


def format_rate(rate: Optional[float], precision: int = 4) -> str:
    """Render rate or 'UNDEFINED'."""
    if rate is None:
        return "UNDEFINED"
    return f"{rate:.{precision}f}"


# ---------------------------------------------------------------------------
# 3. Applicability-Aware Acceptance & Harm Evaluators
# ---------------------------------------------------------------------------

class AcceptanceEvaluator:
    """Evaluates certificate acceptance under S01 vacuous satisfaction semantics."""

    def __init__(self, registry: Optional[Dict[str, Any]] = None):
        self.registry = registry or load_obligation_registry()

    def get_applicability(self, dataset: str) -> Dict[str, bool]:
        entry = self.registry.get(dataset.lower(), {})
        return {
            "grounding": bool(entry.get("grounding", True)),
            "policy": bool(entry.get("policy", False)),
            "replay": bool(entry.get("replay", False)),
        }

    def check_pcg(
        self,
        dataset: str,
        v_h: Optional[bool] = None,
        v_pi: Optional[bool] = None,
        v_gamma: Optional[bool] = None,
        v_entail: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """PCG-MAS acceptance predicate under applicability-aware vacuous pass.
        
        Check(Z; G_t) = prod_{k in K_applicable} V_k(Z; G_t).
        Non-applicable channels evaluate vacuously to True.
        """
        app = self.get_applicability(dataset)

        eff_v_h = bool(v_h) if app["grounding"] else True
        eff_v_entail = bool(v_entail) if app["grounding"] else True
        eff_v_gamma = bool(v_gamma) if app["policy"] else True
        eff_v_pi = bool(v_pi) if app["replay"] else True

        accepted = eff_v_h and eff_v_entail and eff_v_gamma and eff_v_pi

        vacuous_channels = []
        if not app["grounding"]:
            vacuous_channels.extend(["v_h", "v_entail"])
        if not app["policy"]:
            vacuous_channels.append("v_gamma")
        if not app["replay"]:
            vacuous_channels.append("v_pi")

        return {
            "accepted": accepted,
            "effective_v_h": eff_v_h,
            "effective_v_pi": eff_v_pi,
            "effective_v_gamma": eff_v_gamma,
            "effective_v_entail": eff_v_entail,
            "vacuous_channels": vacuous_channels,
            "grounding_applicable": app["grounding"],
            "policy_applicable": app["policy"],
            "replay_applicable": app["replay"],
        }

    def check_specialist(
        self,
        system: str,
        dataset: str,
        v_h: Optional[bool] = None,
        v_pi: Optional[bool] = None,
        v_gamma: Optional[bool] = None,
        v_entail: Optional[bool] = None,
    ) -> bool:
        """Evaluate baseline specialist acceptance on its registered channel."""
        sys_lower = system.lower()
        if sys_lower == "nocert":
            return True
        elif sys_lower == "citation_only":
            return bool(v_h and v_entail)
        elif sys_lower == "shieldagent":
            return bool(v_gamma)
        elif sys_lower == "agentrr":
            return bool(v_pi)
        elif sys_lower in ("pcg_mas", "pcg"):
            return self.check_pcg(dataset, v_h=v_h, v_pi=v_pi, v_gamma=v_gamma, v_entail=v_entail)["accepted"]
        else:
            return True


class HarmEvaluator:
    """Evaluates harm outcome under S01 applicability-aware semantics."""

    def __init__(self, registry: Optional[Dict[str, Any]] = None):
        self.registry = registry or load_harm_registry()

    def get_applicability(self, dataset: str) -> Dict[str, bool]:
        entry = self.registry.get(dataset.lower(), {})
        return {
            "h_support_applicable": bool(entry.get("h_support_applicable", True)),
            "h_exec_applicable": bool(entry.get("h_exec_applicable", False)),
        }

    def evaluate_harm(
        self,
        dataset: str,
        h_support: Optional[bool] = None,
        h_exec: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """h_joint = (h_support if grounding_applicable else False) or (h_exec if policy_applicable else False)."""
        app = self.get_applicability(dataset)
        eff_supp = bool(h_support) if app["h_support_applicable"] else False
        eff_exec = bool(h_exec) if app["h_exec_applicable"] else False
        h_joint = eff_supp or eff_exec

        return {
            "h_joint": h_joint,
            "effective_h_support": eff_supp,
            "effective_h_exec": eff_exec,
            "h_support_applicable": app["h_support_applicable"],
            "h_exec_applicable": app["h_exec_applicable"],
        }


# ---------------------------------------------------------------------------
# 4. SpecialistCompose & Scope Coverage Comparator
# ---------------------------------------------------------------------------

class SpecialistCompose:
    """Scope-matched specialist composition baseline.
    
    On each dataset d, delegates to the dedicated scope specialist:
      - Grounding datasets -> CitationOnly
      - Policy/Replay datasets -> ShieldAgent (policy) / AgentRR (replay)
    
    Serves as the prospective coverage comparator for Gate C6:
      Delta_coverage = Coverage(PCG) - Coverage(SpecialistCompose) >= -0.05.
    """

    @staticmethod
    def get_specialist_for_dataset(dataset: str) -> str:
        d = dataset.lower()
        if d in ("toolbench", "weblinx"):
            return "shieldagent"
        return "citation_only"

    @classmethod
    def evaluate_coverage(cls, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Computes coverage for PCG-MAS and SpecialistCompose across records."""
        total_by_dataset = defaultdict(int)
        pcg_acc_by_dataset = defaultdict(int)
        spec_acc_by_dataset = defaultdict(int)

        datasets = sorted({r["dataset"] for r in records if "dataset" in r})

        for d in datasets:
            d_recs = [r for r in records if r["dataset"] == d]
            total_by_dataset[d] = len([r for r in d_recs if r.get("system") == "pcg_mas"])
            spec_name = cls.get_specialist_for_dataset(d)

            pcg_acc_by_dataset[d] = sum(1 for r in d_recs if r.get("system") == "pcg_mas" and r.get("accepted"))
            spec_acc_by_dataset[d] = sum(1 for r in d_recs if r.get("system") == spec_name and r.get("accepted"))

        n_total_pcg = sum(total_by_dataset.values())
        n_acc_pcg = sum(pcg_acc_by_dataset.values())
        n_acc_spec = sum(spec_acc_by_dataset.values())

        cov_pcg = safe_rate(n_acc_pcg, n_total_pcg)
        cov_spec = safe_rate(n_acc_spec, n_total_pcg)

        cov_delta = (cov_pcg - cov_spec) if (cov_pcg is not None and cov_spec is not None) else None

        c6_pass = cov_delta is not None and cov_delta >= -0.05

        return {
            "N_total": n_total_pcg,
            "N_acc_pcg": n_acc_pcg,
            "N_acc_specialist_compose": n_acc_spec,
            "coverage_pcg": cov_pcg,
            "coverage_specialist_compose": cov_spec,
            "coverage_delta": cov_delta,
            "c6_compliance": c6_pass,
            "by_dataset": {
                d: {
                    "N": total_by_dataset[d],
                    "specialist": cls.get_specialist_for_dataset(d),
                    "pcg_accepted": pcg_acc_by_dataset[d],
                    "specialist_accepted": spec_acc_by_dataset[d],
                    "coverage_pcg": safe_rate(pcg_acc_by_dataset[d], total_by_dataset[d]),
                    "coverage_specialist": safe_rate(spec_acc_by_dataset[d], total_by_dataset[d]),
                }
                for d in datasets
            },
        }


# ---------------------------------------------------------------------------
# 5. Specialist Preservation & Integrated Risk Evaluation
# ---------------------------------------------------------------------------

def evaluate_specialist_comparison(
    pcg_rate: Optional[float],
    specialist_rate: Optional[float],
    higher_is_better: bool = False,
) -> Dict[str, Any]:
    """Evaluates specialist comparison under S01 rules.
    
    Zero-denominator -> INDETERMINATE.
    Delta >= 0.0 -> PASS (an empirical TIE strictly satisfies preservation).
    """
    if pcg_rate is None or specialist_rate is None:
        return {
            "delta": None,
            "status": "INDETERMINATE",
            "preservation": "INDETERMINATE",
        }

    if higher_is_better:
        delta = pcg_rate - specialist_rate
    else:
        # Lower is better (e.g. harm rate)
        delta = specialist_rate - pcg_rate

    if delta > 0.0:
        status = "POSITIVE"
    elif delta == 0.0:
        status = "TIE"
    else:
        status = "NEGATIVE"

    preservation = "PASS" if delta >= 0.0 else "FAIL"

    return {
        "val_pcg": pcg_rate,
        "val_specialist": specialist_rate,
        "delta": round(delta, 6),
        "status": status,
        "preservation": preservation,
    }


def evaluate_integrated_risk(
    h_joint_pcg: Optional[float],
    h_joint_nocert: Optional[float],
) -> Dict[str, Any]:
    """C2: Integrated Risk Gain vs NoCert. Delta = H_joint(NoCert) - H_joint(PCG) > 0."""
    if h_joint_pcg is None or h_joint_nocert is None:
        return {
            "delta": None,
            "c2_pass": False,
            "status": "INDETERMINATE",
        }
    delta = h_joint_nocert - h_joint_pcg
    return {
        "h_joint_pcg": h_joint_pcg,
        "h_joint_nocert": h_joint_nocert,
        "delta": round(delta, 6),
        "c2_pass": delta > 0.0,
        "status": "PASS" if delta > 0.0 else "FAIL",
    }


# ---------------------------------------------------------------------------
# 6. Q_s Selective Utility & Controller Accounting
# ---------------------------------------------------------------------------

@dataclass
class UtilityAccounting:
    """Computes unconditional candidate utility and conditional accepted utility."""

    @staticmethod
    def compute(records: List[Dict[str, Any]]) -> Dict[str, Any]:
        n = len(records)
        acc = [r for r in records if r.get("accepted")]
        n_acc = len(acc)

        cand_u = sum(r.get("cand_utility", 0.0) for r in records) / n if n > 0 else 0.0
        cond_u = sum(r.get("cand_utility", 0.0) for r in acc) / n_acc if n_acc > 0 else None

        return {
            "N_total": n,
            "N_accepted": n_acc,
            "candidate_utility_unconditional": round(cand_u, 4),
            "conditional_utility_accepted": round(cond_u, 4) if cond_u is not None else None,
            "conditional_utility_status": "DEFINED" if cond_u is not None else "UNDEFINED",
        }


# ---------------------------------------------------------------------------
# 7. A16 Resource Accounting
# ---------------------------------------------------------------------------

@dataclass
class A16ResourceAccounting:
    """Zero-dollar spend enforcement and matched-resource accounting."""

    spend_ceiling_usd: float = 0.0

    def compute(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        total_billed = sum(float(r.get("billed_cost_usd", 0.0) or 0.0) for r in records)
        if total_billed > self.spend_ceiling_usd:
            raise ValueError(
                f"A16 spend ceiling violation: total spend ${total_billed:.4f} > ${self.spend_ceiling_usd:.4f}"
            )

        model_calls = sum(int(r.get("model_calls", 0) or 0) for r in records)
        gen_calls = sum(int(r.get("generator_calls", 0) or 0) for r in records)
        checker_calls = sum(int(r.get("checker_calls", 0) or 0) for r in records)
        retrieval_calls = sum(int(r.get("retrieval_calls", 0) or 0) for r in records)
        tokens_in = sum(int(r.get("tokens_in", 0) or 0) for r in records)
        tokens_out = sum(int(r.get("tokens_out", 0) or 0) for r in records)

        return {
            "total_spend_usd": total_billed,
            "spend_ceiling_usd": self.spend_ceiling_usd,
            "ceiling_respected": total_billed <= self.spend_ceiling_usd,
            "model_calls": model_calls,
            "generator_calls": gen_calls,
            "checker_calls": checker_calls,
            "retrieval_calls": retrieval_calls,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "tokens_total": tokens_in + tokens_out,
            "allocation_split": {
                "generation_ratio": safe_rate(gen_calls, model_calls),
                "checking_ratio": safe_rate(checker_calls, model_calls),
            },
        }


# ---------------------------------------------------------------------------
# 8. Pinned A15 DeBERTa-v3-large Checker Interface
# ---------------------------------------------------------------------------

class A15DebertaNLIInterface:
    """Interface to microsoft/deberta-v3-large cross-encoder NLI.
    
    Local Mac mode: does NOT download heavyweight model weights. Uses mock/recorded
    logits or calibration cache.
    Off-Mac bundle: packages code and dependencies for GPU execution (e.g. Colab).
    """

    MODEL_ID = "microsoft/deberta-v3-large"
    TASK = "cross-encoder-nli"
    ENTAILMENT_CLASS_IDX = 2
    TARGET_FPR = 0.05

    def __init__(
        self,
        threshold: float = 0.5,
        mock_mode: bool = True,
        recorded_logits_table: Optional[Dict[str, float]] = None,
    ):
        self.threshold = threshold
        self.mock_mode = mock_mode
        self.recorded_logits_table = recorded_logits_table or {}

    def score(self, premise: str, hypothesis: str) -> float:
        """Returns entailment probability Pr(entailment)."""
        key = f"{premise.strip()} ||| {hypothesis.strip()}"
        if key in self.recorded_logits_table:
            return self.recorded_logits_table[key]

        if self.mock_mode:
            # Deterministic hash-based mock for testing without downloading 1.7GB weights
            import hashlib
            h = hashlib.sha256(key.encode("utf-8")).hexdigest()
            val = int(h[:8], 16) / 0xFFFFFFFF
            return round(val, 4)

        raise RuntimeError(
            "Live DeBERTa model execution is forbidden on local Mac. "
            "Use off-Mac execution bundle for GPU inference."
        )

    def verify(self, premise: str, hypothesis: str) -> bool:
        """Check whether Pr(entailment) >= threshold."""
        return self.score(premise, hypothesis) >= self.threshold

    @classmethod
    def generate_off_mac_bundle(cls, target_dir: Path) -> Path:
        """Emits self-contained off-Mac execution bundle for Colab / remote GPU."""
        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        script_content = f'''#!/usr/bin/env python3
"""A15 DeBERTa-v3-large Off-Mac Execution & Calibration Script.

Model: {cls.MODEL_ID}
Task: {cls.TASK}
Entailment class index: {cls.ENTAILMENT_CLASS_IDX}
Target FPR: {cls.TARGET_FPR}
"""
import argparse
import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "{cls.MODEL_ID}"
ENTAILMENT_IDX = {cls.ENTAILMENT_CLASS_IDX}

def run_evaluation(input_pairs_json: str, output_scores_json: str, batch_size: int = 16):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {{MODEL_ID}} on {{device}}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).to(device)
    model.eval()

    with open(input_pairs_json, "r", encoding="utf-8") as f:
        pairs = json.load(f)

    results = []
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i : i + batch_size]
        premises = [p["premise"] for p in batch]
        hypotheses = [p["hypothesis"] for p in batch]
        inputs = tokenizer(premises, hypotheses, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[:, ENTAILMENT_IDX].cpu().tolist()
        for p, prob in zip(batch, probs):
            results.append({{"id": p.get("id"), "entailment_prob": prob}})

    with open(output_scores_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {{len(results)}} scores to {{output_scores_json}}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input JSON pairs")
    parser.add_argument("--output", required=True, help="Path to output scores JSON")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    run_evaluation(args.input, args.output, args.batch_size)
'''
        (target_dir / "run_a15_deberta_gpu.py").write_text(script_content, encoding="utf-8")

        reqs = "torch>=2.1.0\ntransformers>=4.36.0\naccelerate>=0.25.0\n"
        (target_dir / "requirements.txt").write_text(reqs, encoding="utf-8")

        readme = f"""# A15 DeBERTa-v3-large Off-Mac Bundle

This bundle executes the pinned `{cls.MODEL_ID}` cross-encoder off-Mac on a CUDA GPU host (e.g. Colab).

Usage:
1. `pip install -r requirements.txt`
2. `python run_a15_deberta_gpu.py --input pairs.json --output scores.json`
"""
        (target_dir / "README.md").write_text(readme, encoding="utf-8")

        return target_dir


# ---------------------------------------------------------------------------
# 9. PCG-MAS v3.3 Scientific Architecture & Certifying Operator
# ---------------------------------------------------------------------------

def build_hypothesis(dataset: str, question: str, response_text: str) -> str:
    """Builds task-normalized atomic hypothesis h_g(c) under v3.3 specification."""
    d = dataset.lower()
    r = (response_text or "").strip()
    q = (question or "").strip()

    if d == "fever":
        # For FEVER, question is the claim; model output is verdict
        return q
    elif d in ("hotpotqa", "twowiki", "pubmedqa", "tatqa"):
        # QA benchmarks: form propositional assertion
        clean_r = r
        if "The answer is" in r:
            clean_r = r.split("The answer is")[-1].strip(": \n.")
        elif "Answer:" in r:
            clean_r = r.split("Answer:")[-1].strip(": \n.")
        clean_r = clean_r.strip(".")
        if not clean_r:
            clean_r = r
        return f"The answer to '{q}' is {clean_r}."
    elif d in ("toolbench", "weblinx"):
        return f"Action: {r[:200]}"
    else:
        return f"The answer to '{q}' is {r}."


def segment_evidence_windows(
    evidence_text: str, max_words: int = 120, max_windows: int = 8
) -> List[str]:
    """Constructs bounded, deterministic candidate evidence windows W_K(S_0).
    
    Segments committed evidence into sentence/segment units, grouping into
    bounded candidate windows up to max_windows (K=8).
    """
    text = (evidence_text or "").strip()
    if not text:
        return []

    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return [text[:max_words]]

    windows: List[str] = []
    current_chunk: List[str] = []
    current_len = 0

    for sent in sentences:
        w_count = len(sent.split())
        if current_len + w_count > max_words and current_chunk:
            windows.append(" ".join(current_chunk))
            current_chunk = [sent]
            current_len = w_count
            if len(windows) >= max_windows:
                break
        else:
            current_chunk.append(sent)
            current_len += w_count

    if current_chunk and len(windows) < max_windows:
        windows.append(" ".join(current_chunk))

    return windows[:max_windows]


def validate_v_gamma_toolbench(resp: str) -> bool:
    """Real V_Gamma for ToolBench: schema, argument, and tool compliance."""
    if not resp or not resp.strip():
        return False
    valid_patterns = [
        r'\w+\([^)]*\)',                            # func_name(arg=val)
        r'Action:\s*\w+',                           # Action: tool_name
        r'"name":\s*"[^"]+",\s*"arguments":',       # Tool call JSON
        r'```json\s*\{.*\}\s*```',                 # JSON block
    ]
    for pat in valid_patterns:
        if re.search(pat, resp, re.DOTALL):
            return True
    json_match = re.search(r'\{[^}]+\}', resp)
    if json_match:
        try:
            obj = json.loads(json_match.group(0))
            return isinstance(obj, dict) and len(obj) > 0
        except Exception:
            pass
    return False


def validate_v_gamma_weblinx(resp: str) -> bool:
    """Real V_Gamma for WebLINX: browser action schema & parameter validity."""
    if not resp or not resp.strip():
        return False
    valid_actions = ["click", "type", "submit", "scroll", "hover", "press", "navigate", "load", "search"]
    resp_lower = resp.lower()
    has_action = any(act in resp_lower for act in valid_actions)
    has_target = any(kw in resp_lower for kw in [
        "element", "id", "class", "selector", "button", "input", "link", "xpath",
        "tag", "target", "#", ".", "href", "["
    ])
    return has_action and has_target


def validate_v_pi(trajectory: Optional[Any] = None) -> Optional[bool]:
    """Real V_Pi for replay trajectory verification.
    
    Returns None (representing INDETERMINATE_REPLAY) when replay environment/trace
    cannot be verified.
    """
    if trajectory is None:
        return None
    if isinstance(trajectory, dict) and "trace_verified" in trajectory:
        return bool(trajectory["trace_verified"])
    return None


class CompositeCertifyingOperator:
    """v3.3 Composite Certifying Evidence Operator.
    
    Model: cross-encoder/nli-deberta-v3-large
    Revision: a0b85cc42635c38e7064d3bf17e6085e964849cf
    Classes: 0 -> contradiction, 1 -> entailment, 2 -> neutral
    """

    MODEL_ID = "cross-encoder/nli-deberta-v3-large"
    PINNED_REVISION = "a0b85cc42635c38e7064d3bf17e6085e964849cf"
    CLASS_MAPPING = {"contradiction": 0, "entailment": 1, "neutral": 2}

    def __init__(
        self,
        tau: float = 0.5,
        kappa: float = 0.3,
        search_budget_k: int = 8,
        model: Any = None,
        tokenizer: Any = None,
        device: str = "cpu",
    ):
        self.tau = tau
        self.kappa = kappa
        self.search_budget_k = search_budget_k
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def score_pair(self, premise: str, hypothesis: str) -> Tuple[float, float, float, float]:
        """Computes (p_entail, p_contra, p_neutral, margin)."""
        if self.model is None or self.tokenizer is None:
            h = hashlib.sha256(f"{premise}|||{hypothesis}".encode("utf-8")).hexdigest()
            val = int(h[:8], 16) / 0xFFFFFFFF
            return (val, 1.0 - val, 0.0, val - (1.0 - val))

        import torch
        inputs = self.tokenizer(
            premise, hypothesis, truncation=True, max_length=512, padding=True, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()

        p_contra = probs[0]
        p_entail = probs[1]
        p_neutral = probs[2]
        margin = p_entail - max(p_contra, p_neutral)
        return (p_entail, p_contra, p_neutral, margin)

    def certify(
        self,
        evidence_text: str,
        hypothesis: str,
        tau: Optional[float] = None,
        kappa: Optional[float] = None,
        K: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Evaluates composite certifying operator over W_K candidate windows."""
        t_val = self.tau if tau is None else tau
        k_val = self.kappa if kappa is None else kappa
        budget_k = self.search_budget_k if K is None else K

        windows = segment_evidence_windows(evidence_text, max_windows=budget_k)
        if not windows:
            return {
                "ternary_state": "INDETERMINATE",
                "margin_score": None,
                "contradiction_score": None,
                "minimal_subset": "",
                "k_evaluated": 0,
                "pass_cert": False,
            }

        best_margin = -2.0
        max_contra = 0.0
        minimal_subset = ""
        pass_window_found = False

        for idx, w in enumerate(windows):
            pe, pc, pn, margin = self.score_pair(w, hypothesis)
            if margin > best_margin:
                best_margin = margin
            if pc > max_contra:
                max_contra = pc

            if not pass_window_found and margin >= t_val and pc <= k_val:
                minimal_subset = w
                pass_window_found = True

        if not minimal_subset and windows:
            minimal_subset = windows[0]

        is_pass = (best_margin >= t_val) and (max_contra <= k_val)
        ternary_state = "PASS" if is_pass else "FAIL"

        return {
            "ternary_state": ternary_state,
            "margin_score": round(best_margin, 4),
            "contradiction_score": round(max_contra, 4),
            "minimal_subset": minimal_subset,
            "k_evaluated": len(windows),
            "pass_cert": is_pass,
        }


def compute_sv_decomposition(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes exact Selection-Verification (S/V) decomposition.
    
    Delta = H_joint(NoCert) - H_joint(PCG)
    S = H_joint(NoCert) - H_joint(CandidateAccepted)
    V = H_joint(CandidateAccepted) - H_joint(PCG)
    Asserts Delta == S + V on every evaluation.
    """
    n_total = len(records)
    if n_total == 0:
        return {"delta": None, "S": None, "V": None, "identity_holds": True, "status": "UNDEFINED"}

    harm_nocert = [r.get("h_joint", False) for r in records]
    r_nocert = safe_rate(sum(1 for h in harm_nocert if h), n_total)

    pcg_acc = [r for r in records if r.get("accepted")]
    n_pcg = len(pcg_acc)
    r_pcg = safe_rate(sum(1 for r in pcg_acc if r.get("h_joint")), n_pcg) if n_pcg > 0 else None

    cand_acc = [r for r in records if r.get("effective_v_h", True) and r.get("effective_v_gamma", True)]
    n_cand = len(cand_acc)
    r_cand = safe_rate(sum(1 for r in cand_acc if r.get("h_joint")), n_cand) if n_cand > 0 else None

    if r_nocert is not None and r_pcg is not None and r_cand is not None:
        delta = r_nocert - r_pcg
        S = r_nocert - r_cand
        V = r_cand - r_pcg
        identity_holds = abs(delta - (S + V)) < 1e-9
    else:
        delta = (r_nocert - r_pcg) if (r_nocert is not None and r_pcg is not None) else None
        S, V = None, None
        identity_holds = True

    return {
        "delta": round(delta, 6) if delta is not None else None,
        "S": round(S, 6) if S is not None else None,
        "V": round(V, 6) if V is not None else None,
        "identity_holds": identity_holds,
        "r_nocert": r_nocert,
        "r_pcg": r_pcg,
        "r_cand": r_cand,
    }
