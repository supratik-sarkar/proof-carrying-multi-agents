from __future__ import annotations

import argparse
import uuid
from pathlib import Path

import yaml
from tqdm import tqdm

from src.common.env import ensure_dir
from src.common.jsonl_logger import JsonlLogger
from src.common.telemetry import ExampleTelemetry
from src.common.timers import timed
from src.common.token_count import TokenCounter
from src.data.adapters import load_examples
from src.pcg.baselines import validate_mode
from src.pcg.checker import check_certificate
from src.pcg.decision import choose_action
from src.pcg.prover import Prover
from src.pcg.verifier import Verifier


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_text(x):
    if x is None:
        return None
    return str(x).strip().lower()


def gold_to_index(gold, num_choices: int) -> int | None:
    if gold is None:
        return None

    g = normalize_text(gold)
    if g is None:
        return None

    if g.isdigit():
        v = int(g)
        if 0 <= v < num_choices:
            return v
        if 1 <= v <= num_choices:
            return v - 1

    letter_map = {"a": 0, "b": 1, "c": 2, "d": 3}
    if g in letter_map and letter_map[g] < num_choices:
        return letter_map[g]

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    output_dir = ensure_dir(cfg["output_dir"])
    run_id = str(uuid.uuid4())[:8]
    run_path = Path(output_dir) / "runs" / f"run_{run_id}.jsonl"
    logger = JsonlLogger(run_path)

    mode = cfg["experiment"]["mode"]
    validate_mode(mode)

    dataset_name = cfg["experiment"]["dataset_name"]
    max_examples = int(cfg["runtime"]["max_examples"])
    threshold = float(cfg["experiment"]["accepted_risk_threshold"])
    model_name = cfg["model"]["backbone_name"]
    seed = int(cfg["runtime"]["seed"])

    examples = load_examples(dataset_name=dataset_name, max_examples=max_examples)

    token_counter = TokenCounter(model_name)
    prover = Prover(model_name=model_name, device_map=cfg["model"]["device"])
    verifier = Verifier()

    for ex in tqdm(examples, desc=f"Running {dataset_name} / {mode}"):
        tel = ExampleTelemetry(
            run_id=run_id,
            dataset=dataset_name,
            instance_id=ex["instance_id"],
            backbone=model_name,
            seed=seed,
            mode=mode,
        )

        question = ex["question"]
        choices = ex["choices"]

        with timed() as t_gen:
            prompt, raw_answer = prover.generate(question=question, choices=choices)
        usage = token_counter.usage(prompt, raw_answer)
        tel.generation.wall_ms = t_gen["elapsed_ms"]
        tel.generation.prompt_tokens = usage.prompt_tokens
        tel.generation.completion_tokens = usage.completion_tokens
        tel.generation.total_tokens = usage.total_tokens

        cert = None
        checker_valid = True
        checker_reasons = []
        checker_diag = {}

        if mode == "pcg_full":
            with timed() as t_cert:
                cert = prover.make_certificate(
                    claim=raw_answer,
                    evidence_texts=choices,
                    meta={"dataset": dataset_name, "instance_id": ex["instance_id"]},
                )
            tel.certificate.wall_ms = t_cert["elapsed_ms"]

            with timed() as t_rep:
                checker_valid, checker_reasons, checker_diag = check_certificate(cert)
            tel.replay.wall_ms = t_rep["elapsed_ms"]

        elif mode == "baseline_lightweight_citation":
            cert = {
                "claim": raw_answer,
                "evidence": [{"text": c} for c in choices],
                "meta": {"lightweight": True},
            }

        with timed() as t_ver:
            verdict = verifier.score_answer(answer_text=raw_answer, choices=choices)
        tel.verifier.wall_ms = t_ver["elapsed_ms"]
        tel.risk_raw = verdict["risk"]
        tel.risk_cal = verdict["risk"]

        decision = choose_action(
            risk=verdict["risk"],
            checker_valid=checker_valid,
            threshold=threshold,
        )
        tel.decision = decision
        tel.accepted = decision == "answer"

        gold_idx = gold_to_index(ex["gold_answer"], len(choices))
        tel.answer_correct = (
            gold_idx == verdict["pred_idx"]
            if gold_idx is not None and verdict["pred_idx"] is not None
            else False
        )

        row = tel.to_dict()
        row["question"] = question
        row["choices"] = choices
        row["raw_answer"] = raw_answer
        row["predicted_choice_index"] = verdict["pred_idx"]
        row["predicted_choice_text"] = verdict["pred_choice"]
        row["predicted_choice_score"] = verdict["score"]
        row["gold_answer"] = ex["gold_answer"]
        row["gold_choice_index"] = gold_idx
        row["checker_valid"] = checker_valid
        row["checker_reasons"] = checker_reasons
        row["checker_diagnostics"] = checker_diag
        row["certificate"] = cert

        logger.write(row)

    print(f"Saved run to: {run_path}")


if __name__ == "__main__":
    main()
