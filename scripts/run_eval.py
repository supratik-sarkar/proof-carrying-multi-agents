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


def answer_is_correct(pred: str, gold, choices: list[str]) -> bool | None:
    if gold is None:
        return None

    pred_l = normalize_text(pred)
    gold_l = normalize_text(gold)

    if pred_l is None or gold_l is None:
        return None

    if gold_l in pred_l:
        return True

    # Handle numeric answer index if present
    if gold_l.isdigit():
        idx = int(gold_l)
        if 0 <= idx < len(choices):
            if normalize_text(choices[idx]) in pred_l:
                return True
        if 1 <= idx <= len(choices):
            if normalize_text(choices[idx - 1]) in pred_l:
                return True

    # Handle letter answers a/b/c/d
    letters = {"a": 0, "b": 1, "c": 2, "d": 3}
    if gold_l in letters and letters[gold_l] < len(choices):
        if normalize_text(choices[letters[gold_l]]) in pred_l:
            return True

    return False


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
            prompt, answer = prover.generate(question=question, choices=choices)
        usage = token_counter.usage(prompt, answer)
        tel.generation.wall_ms = t_gen["elapsed_ms"]
        tel.generation.prompt_tokens = usage.prompt_tokens
        tel.generation.completion_tokens = usage.completion_tokens
        tel.generation.total_tokens = usage.total_tokens

        cert = None
        checker_valid = True
        checker_reasons = []

        if mode == "pcg_full":
            with timed() as t_cert:
                cert = prover.make_certificate(
                    claim=answer,
                    evidence_texts=choices,
                    meta={"dataset": dataset_name, "instance_id": ex["instance_id"]},
                )
            tel.certificate.wall_ms = t_cert["elapsed_ms"]

            with timed() as t_rep:
                checker_valid, checker_reasons = check_certificate(cert)
            tel.replay.wall_ms = t_rep["elapsed_ms"]

        elif mode == "baseline_selective":
            pass

        elif mode == "baseline_multiagent_no_cert":
            pass

        elif mode == "baseline_lightweight_citation":
            cert = {
                "claim": answer,
                "evidence": [{"text": c} for c in choices],
                "meta": {"lightweight": True},
            }

        elif mode == "baseline_posthoc_verify":
            pass

        with timed() as t_ver:
            risk = verifier.score_risk(answer_text=answer, choices=choices)
        tel.verifier.wall_ms = t_ver["elapsed_ms"]
        tel.risk_raw = risk
        tel.risk_cal = risk

        decision = choose_action(risk=risk, threshold=threshold)
        tel.decision = decision
        tel.accepted = (decision == "answer") and checker_valid
        tel.answer_correct = answer_is_correct(answer, ex["gold_answer"], choices)

        row = tel.to_dict()
        row["question"] = question
        row["choices"] = choices
        row["answer"] = answer
        row["gold_answer"] = ex["gold_answer"]
        row["checker_valid"] = checker_valid
        row["checker_reasons"] = checker_reasons
        row["certificate"] = cert

        logger.write(row)

    print(f"Saved run to: {run_path}")


if __name__ == "__main__":
    main()
