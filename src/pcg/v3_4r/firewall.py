"""PCG-MAS v3.4R Physical Runtime / Evaluator Separation & Firewall.

Creates and enforces:
1. V3_4R_RUNTIME_ONLY_INPUTS.jsonl
2. V3_4R_EVALUATOR_ONLY_LABELS.jsonl

With strict dataclasses:
- RuntimeCandidate: deployment-visible fields only. Rejects evaluator fields.
- EvaluatorLabels: ground truth harm, success, gold answers, and annotations.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

FORBIDDEN_EVALUATOR_KEYS: Set[str] = {
    "gold_answers",
    "gold_answer",
    "ground_truth_harm",
    "gt_harm",
    "dataset_native_success",
    "gt_success",
    "benchmark_correctness_label",
    "reference_evaluator_annotation",
    "evaluator_target",
    "harmful_accepted",
    "successful_accepted",
}


from pcg.v3_4r.candidate import RuntimeCandidate


@dataclass
class EvaluatorLabels:
    """Evaluator-only ground truth representation."""

    candidate_id: str
    example_id: str
    dataset: str
    gold_answers: List[str]
    ground_truth_harm: int
    dataset_native_success: int
    evaluator_annotations: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def split_raw_record(
    record: Dict[str, Any]
) -> Tuple[RuntimeCandidate, EvaluatorLabels]:
    """Physically splits a raw checkpoint record into firewalled structures."""
    gold_answers = record.get("gold_answers", [])
    if isinstance(gold_answers, str):
        gold_answers = [gold_answers]
    gt_harm = int(record.get("ground_truth_harm", 0))
    gt_success = int(record.get("dataset_native_success", 0))
    eval_annot = record.get("evaluator_annotations", None)

    evaluator_labels = EvaluatorLabels(
        candidate_id=record["candidate_id"],
        example_id=record["example_id"],
        dataset=record["dataset"],
        gold_answers=gold_answers,
        ground_truth_harm=gt_harm,
        dataset_native_success=gt_success,
        evaluator_annotations=eval_annot,
    )

    windows = record.get("windows", [])
    evidence_hashes = record.get("evidence_hashes", [])
    if not evidence_hashes and windows:
        evidence_hashes = [
            hashlib.sha256(w.encode("utf-8")).hexdigest() for w in windows
        ]

    raw_obls = record.get("obligations", [])
    clean_obls = []
    for obl in raw_obls:
        clean_obl = {
            "obligation_id": obl["obligation_id"],
            "obligation_type": obl["obligation_type"],
            "is_critical": obl.get("is_critical", True),
            "description": obl.get("description", ""),
        }
        if "hypothesis_text" in obl:
            clean_obl["hypothesis_text"] = obl["hypothesis_text"]
        clean_obls.append(clean_obl)

    trace = record.get("action_trace", None)
    if trace is None and record.get("dataset") in ("toolbench", "weblinx"):
        try:
            cand_ans = record.get("candidate_answer", "")
            parsed = json.loads(cand_ans)
            if isinstance(parsed, list):
                trace = parsed
        except Exception:
            trace = None

    runtime_cand = RuntimeCandidate(
        candidate_id=record["candidate_id"],
        model=record["model"],
        dataset=record["dataset"],
        example_id=record["example_id"],
        request_hash=record.get(
            "request_hash",
            hashlib.sha256(record.get("prompt", "").encode("utf-8")).hexdigest(),
        ),
        response_hash=record.get(
            "response_hash",
            hashlib.sha256(
                record.get("candidate_answer", "").encode("utf-8")
            ).hexdigest(),
        ),
        prompt=record.get("prompt", ""),
        candidate_answer=record.get("candidate_answer", ""),
        windows=windows,
        evidence_hashes=evidence_hashes,
        obligations=clean_obls,
        resource_metrics=record.get(
            "resource_metrics",
            {
                "generative_calls": 1,
                "input_tokens": 0,
                "output_tokens": 0,
                "evidence_search_ops": 0,
                "replay_ops": 0,
            },
        ),
        policy_context=record.get("policy_context", None),
        tool_snapshots=record.get("tool_snapshots", None),
        action_trace=trace,
    )

    return runtime_cand, evaluator_labels


def create_physical_separation_artifacts(
    raw_checkpoints_path: Path, output_dir: Path
) -> Tuple[Path, Path, str, str]:
    """Generates V3_4R_RUNTIME_ONLY_INPUTS.jsonl and V3_4R_EVALUATOR_ONLY_LABELS.jsonl.

    Returns:
        (runtime_path, evaluator_path, runtime_sha256, evaluator_sha256)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_path = output_dir / "V3_4R_RUNTIME_ONLY_INPUTS.jsonl"
    evaluator_path = output_dir / "V3_4R_EVALUATOR_ONLY_LABELS.jsonl"

    runtime_records = []
    evaluator_records = []

    with open(raw_checkpoints_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            raw = json.loads(line)
            c, l = split_raw_record(raw)
            runtime_records.append(c.to_dict())
            evaluator_records.append(l.to_dict())

    # Write runtime-only inputs
    with open(runtime_path, "w", encoding="utf-8") as f:
        for r in runtime_records:
            f.write(json.dumps(r) + "\n")
    runtime_sha = hashlib.sha256(runtime_path.read_bytes()).hexdigest()
    (output_dir / "V3_4R_RUNTIME_ONLY_INPUTS.jsonl.sha256").write_text(
        runtime_sha, encoding="utf-8"
    )

    # Write evaluator-only labels
    with open(evaluator_path, "w", encoding="utf-8") as f:
        for e in evaluator_records:
            f.write(json.dumps(e) + "\n")
    evaluator_sha = hashlib.sha256(evaluator_path.read_bytes()).hexdigest()
    (output_dir / "V3_4R_EVALUATOR_ONLY_LABELS.jsonl.sha256").write_text(
        evaluator_sha, encoding="utf-8"
    )

    return runtime_path, evaluator_path, runtime_sha, evaluator_sha
