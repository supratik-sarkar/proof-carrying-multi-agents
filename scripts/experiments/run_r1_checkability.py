from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

"""
R1: Audit decomposition / checkability.

Theorem 1 says
    Pr(accept & wrong) <= Pr(IntFail) + Pr(ReplayFail) + Pr(CheckFail) + Pr(CovGap).

This script:
    1. Runs the Prover on N HotpotQA examples (clean and adversarial mix).
    2. Verifier checks each certificate, recording per-channel flags.
    3. Compares predictions to gold answers to label CovGap.
    4. Estimates each channel's empirical probability with Wilson CIs.
    5. Writes results/<run_id>/r1.json — picked up by make_figures and make_tables.

Usage:
    python scripts/experiments/run_r1_checkability.py --config configs/r1_hotpotqa.yaml --seeds 0 1 2
"""
import argparse
# Allow running as a script

def _safe_json(value):
    try:
        import dataclasses
        if dataclasses.is_dataclass(value):
            return dataclasses.asdict(value)
    except Exception:
        pass

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    try:
        import numpy as np
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass

    if isinstance(value, (set, frozenset)):
        return sorted(_safe_json(x) for x in value)

    if isinstance(value, list):
        return [_safe_json(x) for x in value]

    if isinstance(value, tuple):
        return [_safe_json(x) for x in value]

    if isinstance(value, dict):
        return {str(k): _safe_json(v) for k, v in value.items()}

    if hasattr(value, "__dict__"):
        return {
            str(k): _safe_json(v)
            for k, v in vars(value).items()
            if not str(k).startswith("_")
        }

    return str(value)


def _evidence_to_rows(example):
    rows = []
    for ev in getattr(example, "evidence", []) or []:
        rows.append({
            "id": getattr(ev, "id", None),
            "text": getattr(ev, "text", None),
            "title": getattr(ev, "title", None),
            "url": getattr(ev, "url", None),
            "publisher": getattr(ev, "publisher", None),
            "meta": _safe_json(getattr(ev, "meta", None)),
        })
    return rows


def _baseline_source_record(*, ex, state, cr, raw_answer, f1, attack_kind, seed, model, dataset, run_id):
    certificate = getattr(state, "certificate", None)
    graph = getattr(state, "graph", None)

    question = getattr(ex, "question", "") or ""
    gold_answers = list(getattr(ex, "gold_answers", []) or [])
    evidence_rows = _evidence_to_rows(ex)

    claim = ""
    if certificate is not None:
        claim = getattr(certificate, "claim", "") or getattr(certificate, "c", "") or ""

    prompt = question
    if evidence_rows:
        context = "\n".join(
            str(e.get("text") or "") for e in evidence_rows[:6] if e.get("text")
        )
        if context:
            prompt = f"Question: {question}\n\nEvidence:\n{context}"

    trajectory = []
    if certificate is not None:
        trajectory.append({
            "stage": "certificate",
            "claim": claim,
            "support_ids": _safe_json(getattr(certificate, "support_ids", getattr(certificate, "S", []))),
            "pipeline": _safe_json(getattr(certificate, "pipeline", getattr(certificate, "Pi", None))),
            "contract": _safe_json(getattr(certificate, "contract", getattr(certificate, "Gamma", None))),
        })

    return {
        "example_id": getattr(ex, "id", None),
        "model": model,
        "dataset": dataset,
        "seed": seed,
        "split": "adv" if attack_kind != "none" else "clean",
        "run_id": run_id,
        "prompt": prompt,
        "question": question,
        "answer": raw_answer,
        "raw_answer": raw_answer,
        "gold_answers": gold_answers,
        "evidence": evidence_rows,
        "trajectory": trajectory,
        "tool_trace": _safe_json(getattr(state, "tool_trace", [])),
        "actions": _safe_json(getattr(state, "actions", [])),
        "messages": _safe_json(getattr(state, "messages", [])),
        "certificate_id": getattr(certificate, "id", None) if certificate is not None else None,
        "certificate": _safe_json(certificate),
        "graph_summary": {
            "num_nodes": len(getattr(graph, "nodes", [])) if graph is not None and hasattr(graph, "nodes") else None,
            "num_edges": len(getattr(graph, "edges", [])) if graph is not None and hasattr(graph, "edges") else None,
        },
        "check": {
            "passed": cr.passed,
            "integrity_ok": cr.integrity_ok,
            "replay_ok": cr.replay_ok,
            "entailment_ok": cr.entailment_ok,
            "execution_ok": cr.execution_ok,
        },
        "attack": attack_kind,
        "f1_to_gold": f1,
        "gold_harm": int((not cr.passed) or attack_kind != "none"),
    }



from scripts.common.experiment_io import (
    build_backend,
    cfg_get,
    load_config,
    log_info,
    log_section,
    make_output_dir,
    make_run_id,
    write_json,
    project_root,
)


def main(
    config: str = "configs/r1_hotpotqa.yaml",
    seeds: list[int] | None = None,
    n_examples: int | None = None,
    backend: str | None = None,
    dataset: str | None = None,
    model: str | None = None,
) -> int:
    seeds = seeds or [0, 1, 2, 3, 4]
    cfg = load_config(config)
    if n_examples is not None:
        cfg.setdefault("dataset", {})["n_examples"] = n_examples
    if dataset is not None:
        cfg.setdefault("dataset", {})["name"] = dataset
    if model is not None:
        cfg.setdefault("backend", {})["model_name"] = model

    run_id = make_run_id(config)
    out_dir = make_output_dir(cfg, run_id)
    log_section(f"R1 run: {run_id}")
    log_info(f"Output: {out_dir}")

    write_json(out_dir / "config_snapshot.json", cfg)

    # Imports are lazy here so the preflight path doesn't pull in heavy deps
    from pcg.checker import (
        Checker,
        ExactMatchEntailment,
        build_default_replayer,
    )
    from pcg.datasets import load_dataset_by_name
    from pcg.eval import bootstrap_ci, estimate_audit_decomposition
    from pcg.eval.metrics import f1_score
    from pcg.orchestrator.langgraph_flow import OrchestratorConfig, run_one_example
    from pcg.orchestrator import build_replayer_with_handlers

    backend_obj = build_backend(cfg, override=backend)
    log_info(f"Backend: {backend_obj.name}")
    checker = Checker(
        entailment=ExactMatchEntailment(case_insensitive=True),
        replayer=build_replayer_with_handlers(),
    )

    n = cfg_get(cfg, "dataset.n_examples", 200)
    adv_enabled = cfg_get(cfg, "adversarial.enabled", False)
    adv_frac = cfg_get(cfg, "adversarial.fraction", 0.30)
    attack_kinds = cfg_get(cfg, "adversarial.attack_kinds",
                            ["evidence_swap", "schema_break", "policy_violation"])

    seed_results: list[dict] = []

    for seed in seeds:
        log_section(f"seed={seed}")
        examples = list(load_dataset_by_name(
            cfg_get(cfg, "dataset.name", "hotpotqa"),
            split=cfg_get(cfg, "dataset.split", "validation"),
            n_examples=n, seed=seed,
        ))
        log_info(f"  loaded {len(examples)} examples")

        check_results = []
        gt_correct = []
        per_example_records = []
        baseline_source_records = []

        for i, ex in enumerate(examples):
            # Decide attack
            adv_idx = (seed * 7919 + i) % 1000   # deterministic per (seed, i)
            attack = (adv_idx / 1000.0) < adv_frac if adv_enabled else False
            kind = attack_kinds[adv_idx % len(attack_kinds)] if attack else "none"

            cfg_run = OrchestratorConfig(
                enable_attacker=attack,
                enable_debugger=False,
                attack_kind=kind if attack else "evidence_swap",
                max_retries=0,
            )
            state = run_one_example(
                ex, backend=backend_obj, checker=checker, cfg=cfg_run,
            )
            cr = state.check_result
            if cr is None:
                continue
            check_results.append(cr)

            # CovGap baseline: did the (claim text) match a gold answer?
            raw = state.meta.get("raw_answer", "")
            f1 = f1_score(raw, list(ex.gold_answers))
            gt_correct.append(f1 >= 0.5 and not attack)   # attacks => gt_correct=False

            per_example_records.append({
                "id": ex.id, "passed": cr.passed,
                "integrity_ok": cr.integrity_ok, "replay_ok": cr.replay_ok,
                "entailment_ok": cr.entailment_ok, "execution_ok": cr.execution_ok,
                "attack": kind, "f1_to_gold": f1,
                "raw_answer": raw[:200],
            })

            baseline_source_records.append(_baseline_source_record(
                ex=ex,
                state=state,
                cr=cr,
                raw_answer=raw,
                f1=f1,
                attack_kind=kind,
                seed=seed,
                model=getattr(backend_obj, "name", None) or model or cfg_get(cfg, "backend.model_name", "unknown"),
                dataset=dataset or cfg_get(cfg, "dataset.name", "unknown"),
                run_id=run_id,
            ))

            if (i + 1) % 25 == 0:
                log_info(f"    {i+1}/{len(examples)}")

        decomp = estimate_audit_decomposition(check_results, gt_correct, alpha=0.05)
        log_info(f"  LHS = {decomp.lhs_accept_and_wrong:.4f}  RHS-union = {decomp.rhs_union:.4f}")

        seed_results.append({
            "seed": seed, "n": len(check_results),
            "decomposition": decomp.to_dict(),
            "per_example": per_example_records,
        })
        write_json(out_dir / f"seed_{seed}.json", seed_results[-1])

        baseline_dir = project_root() / "results" / "tables" / "csv" / "baseline_inputs"
        baseline_dir.mkdir(parents=True, exist_ok=True)
        baseline_path = baseline_dir / f"{run_id}__seed{seed}__baseline_inputs.jsonl"
        with baseline_path.open("w", encoding="utf-8") as f:
            import json
            for row in baseline_source_records:
                f.write(json.dumps(_safe_json(row), sort_keys=True) + "\n")
        log_info(f"  baseline inputs: {baseline_path}")

    # Aggregate across seeds: bootstrap CI on each channel
    aggregated = {}
    for key in ("p_int_fail", "p_replay_fail", "p_check_fail", "p_cov_gap",
                 "lhs_accept_and_wrong", "rhs_union"):
        values = [s["decomposition"][key] for s in seed_results]
        ci = bootstrap_ci(values, n_boot=cfg_get(cfg, "stats.bootstrap_n", 2000),
                          alpha=cfg_get(cfg, "stats.alpha", 0.05))
        aggregated[key] = {"mean": ci.estimate, "ci": [ci.lower, ci.upper]}

    final = {
        "run_id": run_id, "seeds": seeds,
        "per_seed": seed_results, "aggregated": aggregated,
    }
    write_json(out_dir / "r1.json", final)
    log_section("R1 complete")
    log_info(f"Result: {out_dir / 'r1.json'}")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/r1_hotpotqa.yaml")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--n-examples", type=int, default=None)
    p.add_argument("--dataset", type=str, default=None, help="Optional dataset override.")
    p.add_argument("--model", type=str, default=None, help="Optional model override.")
    p.add_argument("--backend", default=None,
                    choices=[None, "mock", "hf_local", "hf_inference"])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(main(config=args.config, seeds=args.seeds,
                   n_examples=args.n_examples, backend=args.backend))
