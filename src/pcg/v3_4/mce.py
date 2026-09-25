"""PCG-MAS v3.4 Minimum Checker-Certifying Evidence (MCE).

Implements Requirement H:
- Post-PASS compression only; NEVER rescues a failed or indeterminate candidate
- For K <= 8 evidence units, exhaustively enumerates all 2^K subsets
- Deterministic tie-breaking:
  1. minimum number of evidence units |E'|
  2. minimum token count
  3. lexicographically smallest evidence-hash sequence
- Explicit invariant: MCE never alters original acceptance status
"""
import itertools
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Callable

from pcg.v3_4.states import RawVerifierState
from pcg.v3_4.semantic_gate import NLIWindowScore, evaluate_vector_semantic_gate, SemanticVerificationResult
from pcg.v3_4.obligations import TaskObligation

@dataclass
class MCEResult:
    mce_applied: bool
    original_pass: bool
    original_evidence_count: int
    original_token_count: int
    compressed_evidence_count: int
    compressed_token_count: int
    compression_ratio_units: float # |E*| / |E|
    compression_ratio_tokens: float # tok(E*) / tok(E)
    selected_window_indices: List[int]
    verification_calls_used: int
    retained_pass: bool

def compute_token_count(text: str) -> int:
    return len(text.split())

def compute_mce_post_pass(
    obligations: List[TaskObligation],
    evidence_windows: List[Tuple[int, str]], # list of (window_idx, window_text)
    score_fn: Callable[[List[Tuple[int, str]]], Dict[str, List[NLIWindowScore]]],
    tau_cov: float = 1.0,
    tau_e: float = 0.05,
    tau_c: float = 0.30,
    original_status: RawVerifierState = RawVerifierState.PASS
) -> MCEResult:
    """Computes minimum checker-certifying evidence E* subset of E.
    
    CRITICAL INVARIANTS:
    - If original_status != PASS -> MCE is SKIPPED, returns original uncompressed stats, cannot flip status.
    - If K <= 8 -> exhaustive 2^K search.
    - Tie-breaking: (len(subset), tokens, hash_sequence).
    """
    total_units = len(evidence_windows)
    total_tokens = sum(compute_token_count(w[1]) for w in evidence_windows)
    
    if original_status != RawVerifierState.PASS:
        # Hard requirement: MCE must never rescue or modify non-PASS
        return MCEResult(
            mce_applied=False,
            original_pass=False,
            original_evidence_count=total_units,
            original_token_count=total_tokens,
            compressed_evidence_count=total_units,
            compressed_token_count=total_tokens,
            compression_ratio_units=1.0,
            compression_ratio_tokens=1.0,
            selected_window_indices=[w[0] for w in evidence_windows],
            verification_calls_used=0,
            retained_pass=False
        )
        
    k = min(len(evidence_windows), 8) # clamp to at most 8 for exhaustive 256 search
    subsets_to_search = evidence_windows[:k]
    
    best_subset = evidence_windows
    best_len = total_units
    best_tokens = total_tokens
    best_hash = "".join(hashlib.sha256(w[1].encode("utf-8")).hexdigest() for w in evidence_windows)
    
    eval_calls = 0
    # Exhaustive search from size 1 up to k
    candidate_found = False
    for r in range(1, k + 1):
        for sub in itertools.combinations(subsets_to_search, r):
            eval_calls += 1
            sub_list = list(sub)
            sub_scores = score_fn(sub_list)
            res = evaluate_vector_semantic_gate(obligations, sub_scores, tau_cov=tau_cov, tau_e=tau_e, tau_c=tau_c)
            
            if res.raw_state == RawVerifierState.PASS:
                sub_tokens = sum(compute_token_count(w[1]) for w in sub_list)
                sub_hash = "".join(hashlib.sha256(w[1].encode("utf-8")).hexdigest() for w in sub_list)
                
                # Check if strictly smaller or better tie-breaker
                if (r < best_len) or (r == best_len and sub_tokens < best_tokens) or (r == best_len and sub_tokens == best_tokens and sub_hash < best_hash):
                    best_subset = sub_list
                    best_len = r
                    best_tokens = sub_tokens
                    best_hash = sub_hash
                    candidate_found = True
                    
        # Since r is increasing, if we found a valid subset of size r, no subset of size r+1 can beat it in size!
        if candidate_found:
            break
            
    ratio_units = best_len / total_units if total_units > 0 else 1.0
    ratio_tokens = best_tokens / total_tokens if total_tokens > 0 else 1.0
    
    return MCEResult(
        mce_applied=True,
        original_pass=True,
        original_evidence_count=total_units,
        original_token_count=total_tokens,
        compressed_evidence_count=best_len,
        compressed_token_count=best_tokens,
        compression_ratio_units=ratio_units,
        compression_ratio_tokens=ratio_tokens,
        selected_window_indices=[w[0] for w in best_subset],
        verification_calls_used=eval_calls,
        retained_pass=True
    )
