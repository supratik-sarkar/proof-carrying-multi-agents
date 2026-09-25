"""PCG-MAS v3.4R Authoritative Semantic Verifier Binding.

Binds strictly to the parent v3.4 scientific verifier authority:
- Model: cross-encoder/nli-deberta-v3-large
- Revision: a0b85cc42635c38e7064d3bf17e6085e964849cf
- Exact margin: m = p_e - max(p_c, p_n)
- Exact thresholds: tau_cov* = 1.0, tau_e* = 0.05, tau_c* = 0.30
- Local offline execution only (zero downloads / zero provider calls)
- If model or offline cache is missing, returns SEMANTIC_VERIFIER_AVAILABILITY=INDETERMINATE
"""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

PARENT_MODEL_ID = "cross-encoder/nli-deberta-v3-large"
PARENT_PINNED_REVISION = "a0b85cc42635c38e7064d3bf17e6085e964849cf"
PARENT_TAU_COV_STAR = 1.0
PARENT_TAU_E_STAR = 0.05
PARENT_TAU_C_STAR = 0.30


@dataclass
class WindowNliScore:
    window_idx: int
    window_text: str
    pe: float
    pc: float
    pn: float
    margin: float


class AuthoritativeSemanticVerifier:
    """Offline semantic verifier bound to parent v3.4 DeBERTa-v3-large cross-encoder."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root
        self.cal_path = (
            repo_root
            / "artifacts"
            / "v3_4"
            / "experimental_controller"
            / "V34-G4"
            / "A15_V3_4_FROZEN_CALIBRATION.json"
        )
        self.cache_path = (
            repo_root
            / "artifacts"
            / "v3_4"
            / "experimental_controller"
            / "V34-G7"
            / "PAIR_SCORE_CACHE.json"
        )
        self.local_snapshot_dir = (
            Path.home()
            / ".cache"
            / "huggingface"
            / "hub"
            / "models--cross-encoder--nli-deberta-v3-large"
            / "snapshots"
            / PARENT_PINNED_REVISION
        )

        self.is_available = False
        self.availability_state = "INDETERMINATE"
        self.pair_cache: Dict[Tuple[str, str], Tuple[float, float, float, float]] = (
            {}
        )
        self._check_and_load_offline_authority()

    def _check_and_load_offline_authority(self) -> None:
        """Verifies local availability without network downloads."""
        # 1. Verify frozen calibration config
        if not self.cal_path.exists():
            self.availability_state = "INDETERMINATE_MISSING_CALIBRATION"
            self.is_available = False
            return

        # 2. Check offline precomputed cache or local model checkpoint
        has_cache = self.cache_path.exists()
        has_snapshot = self.local_snapshot_dir.exists()

        if not (has_cache or has_snapshot):
            self.availability_state = "INDETERMINATE_MODEL_UNAVAILABLE_OFFLINE"
            self.is_available = False
            return

        # Load offline score cache if available
        if has_cache:
            try:
                raw = json.loads(self.cache_path.read_text(encoding="utf-8"))
                for k, v in raw.items():
                    w_text, hyp_text = k.split("|||", 1)
                    # v: [pe, pc, pn, margin]
                    self.pair_cache[(w_text, hyp_text)] = (
                        float(v[0]),
                        float(v[1]),
                        float(v[2]),
                        float(v[3]),
                    )
            except Exception:
                pass

        self.is_available = True
        self.availability_state = "AVAILABLE_OFFLINE"

    def get_provenance_record(self) -> Dict[str, Any]:
        """Returns machine-readable provenance record of the semantic verifier."""
        cal_sha256 = (
            hashlib.sha256(self.cal_path.read_bytes()).hexdigest()
            if self.cal_path.exists()
            else None
        )
        cache_sha256 = (
            hashlib.sha256(self.cache_path.read_bytes()).hexdigest()
            if self.cache_path.exists()
            else None
        )

        return {
            "schema": "PCG_MAS_V3_4R_SEMANTIC_VERIFIER_PROVENANCE_V1",
            "model_id": PARENT_MODEL_ID,
            "pinned_revision": PARENT_PINNED_REVISION,
            "local_snapshot_path": str(self.local_snapshot_dir),
            "local_snapshot_exists": self.local_snapshot_dir.exists(),
            "frozen_calibration_path": str(self.cal_path),
            "frozen_calibration_sha256": cal_sha256,
            "offline_score_cache_path": str(self.cache_path),
            "offline_score_cache_sha256": cache_sha256,
            "offline_pairs_loaded": len(self.pair_cache),
            "semantic_quantity": "m = p_e - max(p_c, p_n)",
            "tau_cov_star": PARENT_TAU_COV_STAR,
            "tau_e_star": PARENT_TAU_E_STAR,
            "tau_c_star": PARENT_TAU_C_STAR,
            "availability_status": self.availability_state,
        }

    def _init_local_model(self) -> None:
        """Lazily loads local model snapshot offline with zero networking."""
        if hasattr(self, "_model") and self._model is not None:
            return
        if not self.local_snapshot_dir.exists():
            raise FileNotFoundError(
                f"Local snapshot not found at {self.local_snapshot_dir}"
            )
        import os
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(self.local_snapshot_dir), local_files_only=True
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            str(self.local_snapshot_dir), local_files_only=True
        )
        self._model.eval()

    def score_pair(
        self, premise: str, hypothesis: str
    ) -> Tuple[float, float, float, float]:
        """Scores a single (premise, hypothesis) pair offline.

        Returns:
            (pe, pc, pn, margin)
        """
        pair_key = (premise, hypothesis)
        if pair_key in self.pair_cache:
            return self.pair_cache[pair_key]

        # Offline model inference via pinned local snapshot
        if self.local_snapshot_dir.exists():
            self._init_local_model()
            import torch
            inputs = self._tokenizer(
                premise, hypothesis, return_tensors="pt", truncation=True, max_length=512
            )
            with torch.no_grad():
                logits = self._model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)[0].tolist()
            # 0: contradiction, 1: entailment, 2: neutral
            pc = float(probs[0])
            pe = float(probs[1])
            pn = float(probs[2])
            margin = pe - max(pc, pn)
            result = (pe, pc, pn, margin)
            self.pair_cache[pair_key] = result
            return result

        raise KeyError(
            f"OFFLINE PAIR MISSING: Pair not found in offline score cache: {pair_key[0][:40]}..."
        )

    def evaluate_vector_gate(
        self,
        obligations: List[Dict[str, Any]],
        windows: List[str],
        pair_scorer: Callable[[str, str], Tuple[float, float, float, float]],
        tau_cov: float = PARENT_TAU_COV_STAR,
        tau_e: float = PARENT_TAU_E_STAR,
        tau_c: float = PARENT_TAU_C_STAR,
    ) -> Tuple[str, Dict[str, Any]]:
        """Evaluates vector semantic gate preserving parent margin semantics exactly.

        For each obligation:
        - Scores across all windows
        - Selects single best window by margin
        - Verifies best_window.margin >= tau_e AND best_window.pc <= tau_c
        """
        if not obligations:
            return "PASS", {"reason": "No obligations", "coverage_ratio": 1.0}

        certified_count = 0
        total_critical = 0
        per_obl_details = {}
        all_best_margins = []

        for obl in obligations:
            oid = obl["obligation_id"]
            crit = obl.get("is_critical", True)
            hyp_text = obl.get("hypothesis_text", "")
            if crit:
                total_critical += 1

            window_scores = []
            for widx, w in enumerate(windows):
                pe, pc, pn, margin = pair_scorer(w, hyp_text)
                window_scores.append(
                    WindowNliScore(
                        window_idx=widx,
                        window_text=w,
                        pe=pe,
                        pc=pc,
                        pn=pn,
                        margin=margin,
                    )
                )

            if not window_scores:
                per_obl_details[oid] = {
                    "certified": False,
                    "reason": "No windows",
                }
                continue

            # Select single best window by margin (exact parent rule)
            best_window = max(window_scores, key=lambda ws: ws.margin)
            all_best_margins.append(best_window.margin)

            # Single best window must satisfy both margin and contradiction ceiling
            is_certified = (best_window.margin >= tau_e) and (
                best_window.pc <= tau_c
            )
            if is_certified and crit:
                certified_count += 1

            per_obl_details[oid] = {
                "certified": is_certified,
                "best_window_idx": best_window.window_idx,
                "best_margin": best_window.margin,
                "best_pe": best_window.pe,
                "best_pc": best_window.pc,
                "is_critical": crit,
            }

        cov_ratio = (
            (certified_count / total_critical) if total_critical > 0 else 1.0
        )
        gate_pass = cov_ratio >= (tau_cov - 1e-6)

        return ("PASS" if gate_pass else "FAIL"), {
            "tau_cov": tau_cov,
            "tau_e": tau_e,
            "tau_c": tau_c,
            "coverage_ratio": cov_ratio,
            "certified_critical": certified_count,
            "total_critical": total_critical,
            "min_best_margin": min(all_best_margins)
            if all_best_margins
            else -1.0,
            "per_obligation": per_obl_details,
        }
