"""
Run PCG-MAS ablations for the Phase 8 figure + table.

For each (cell, condition, ablation_variant) we run n examples through
the pipeline and emit per-example records. Conditions: clean vs
adversarial (eps_adv, p_fresh). Variants: full / no_replay / no_redundancy
/ no_resp / no_riskctrl.

Output structure (cell-keyed, cloud-portable):
    results/tables/csv/ablations_outputs/<run_id>/
        <dataset>__<model>.jsonl             # per-example records
        <dataset>__<model>__summary.json     # aggregated harm per variant

This runner is cell-spec-driven. The same code runs locally on phi cells
and remotely (Colab/API) on the larger cells. The cell-spec list is the
only thing that changes.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Headline cells. Six of them; this runner accepts any subset via --cells.
HEADLINE_CELLS = [
    ("fever",      "phi-3.5-mini",       "microsoft/Phi-3.5-mini-instruct"),
    ("hotpotqa",   "qwen2.5-7B",         "Qwen/Qwen2.5-7B-Instruct"),
    ("pubmedqa",   "Llama-3.1-8B",       "meta-llama/Llama-3.1-8B-Instruct"),
    ("tatqa",      "Gemma-2-9b-it",      "google/gemma-2-9b-it"),
    ("toolbench",  "Llama-3.3-70B",      "meta-llama/Llama-3.3-70B-Instruct"),
    ("weblinx",    "deepseek-v3",        "deepseek-ai/DeepSeek-V3"),
]

# Component-level variants (Phase 8 — table_ablations).
COMPONENT_VARIANTS = [
    ("full",                 {}),
    ("no_replay",            {"disable_replay": True}),
    ("no_redundancy",        {"disable_redundancy": True}),
    ("no_resp",              {"disable_responsibility": True}),
    ("no_riskctrl",          {"disable_risk_control": True}),
]

# Channel-level variants (Phase 12 — table_channel_ablation).
CHANNEL_VARIANTS = [
    ("full",                 {}),
    ("minus_v_h",            {"disable_v_h":      True}),
    ("minus_v_pi",           {"disable_v_pi":     True}),
    ("minus_v_gamma",        {"disable_v_gamma":  True}),
    ("minus_v_entail",       {"disable_v_entail": True}),
]

VARIANTS = COMPONENT_VARIANTS

# Adversarial setting per figure caption.
EPS_ADV = 0.25
P_FRESH = 0.30


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cells", type=str, default="fever:phi-3.5-mini,hotpotqa:phi-3.5-mini",
                   help="Comma-sep list of dataset:model_paper_name. "
                        "Use 'headline' for all 6 headline cells.")
    p.add_argument("--n-examples", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backend", type=str, default="hf_local")
    p.add_argument("--out-dir", type=str, default=None,
                   help="Override output directory. Default: auto-generated under "
                        "results/tables/csv/ablations_outputs/.")
    p.add_argument("--mode", type=str, choices=["component", "channel"], default="component",
                   help="component: Phase 8 (NoReplay/NoRedundancy/NoResp/NoRiskCtrl). "
                        "channel: Phase 12 (-V_H/-V_Pi/-V_Gamma/-V_entail).")
    return p.parse_args()


def resolve_cells(spec: str) -> list[tuple[str, str, str]]:
    if spec.strip().lower() == "headline":
        return HEADLINE_CELLS
    cell_map = {(d, m): hf for d, m, hf in HEADLINE_CELLS}
    out: list[tuple[str, str, str]] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise SystemExit(f"bad cell spec {part!r}; expected dataset:model")
        d, m = part.split(":", 1)
        hf = cell_map.get((d, m))
        if hf is None:
            # Fall back to looking up paper_name only (allows hotpotqa:phi-3.5-mini
            # even though phi isn't in HEADLINE_CELLS for hotpotqa)
            hf = next((h for dd, mm, h in HEADLINE_CELLS if mm == m), None)
            if hf is None and m == "phi-3.5-mini":
                hf = "microsoft/Phi-3.5-mini-instruct"
            if hf is None:
                raise SystemExit(f"unknown model paper_name in {part!r}")
        out.append((d, m, hf))
    return out


def run_one_variant(
    *,
    backend_obj,
    checker,
    examples,
    variant_name: str,
    variant_kwargs: dict,
    adversarial: bool,
    seed: int,
):
    """Run all examples for one (variant, condition) and return per-example dicts."""
    from pcg.orchestrator.langgraph_flow import OrchestratorConfig, run_one_example
    from pcg.eval.metrics import f1_score

    records = []
    attack_kinds = ["evidence_swap", "schema_break", "policy_violation"]
    for i, ex in enumerate(examples):
        attack = adversarial and ((seed * 7919 + i) % 1000) / 1000.0 < EPS_ADV
        kind = attack_kinds[i % len(attack_kinds)] if attack else "evidence_swap"

        cfg_kwargs = dict(
            enable_attacker=attack,
            enable_debugger=False,
            attack_kind=kind,
            max_retries=0,
            epsilon_adv=(EPS_ADV if adversarial else 0.0),
            p_fresh=(P_FRESH if adversarial else 0.0),
        )
        cfg_kwargs.update(variant_kwargs)
        cfg = OrchestratorConfig(**cfg_kwargs)
        try:
            state = run_one_example(ex, backend=backend_obj, checker=checker, cfg=cfg)
        except Exception as exc:
            records.append({
                "example_id": ex.id, "variant": variant_name,
                "condition": "adv" if adversarial else "clean",
                "passed": False, "f1": 0.0, "wrong": True,
                "raw_answer": "", "error": f"{type(exc).__name__}:{str(exc)[:80]}",
            })
            continue

        cr = state.check_result
        raw = state.meta.get("raw_answer", "")
        f1 = f1_score(raw, list(ex.gold_answers)) if ex.gold_answers else 0.0
        passed = bool(cr.passed) if cr else False
        conf = float(getattr(state.certificate, "confidence", 0.0)) if state.certificate else 0.0
        # Risk gating: when NOT ablated, refuse low-confidence accepts.
        # Threshold matches the R4 piecewise policy at lambda_risk=1.0.
        # Disabling risk control sends every passed=True straight to harm.
        risk_threshold = 0.5
        risk_refused = (not variant_kwargs.get("disable_risk_control", False))                        and passed and (conf < risk_threshold)
        final_accept = passed and not risk_refused
        records.append({
            "example_id": ex.id,
            "variant": variant_name,
            "condition": "adv" if adversarial else "clean",
            "passed": passed,
            "final_accept": final_accept,
            "confidence": conf,
            "risk_refused": risk_refused,
            "f1": float(f1),
            "wrong": f1 < 0.5,
            "raw_answer": raw[:80],
            "harm": int(final_accept and (f1 < 0.5)),
        })
    return records


def run_cell(
    *,
    dataset_name: str,
    paper_model_name: str,
    hf_model_name: str,
    n: int,
    seed: int,
    backend_kind: str,
    out_dir: Path,
):
    from scripts.common.experiment_io import build_backend
    from pcg.datasets import load_dataset_by_name
    from pcg.checker import Checker, TokenOverlapEntailment
    from pcg.orchestrator import build_replayer_with_handlers

    cell_label = f"{dataset_name}__{paper_model_name}"
    print(f"\n{'='*70}\nCELL: {cell_label}  (n={n}, seed={seed})\n{'='*70}")

    cfg_for_backend = {
        "backend": {
            "kind": backend_kind,
            "model_name": hf_model_name,
            "dtype": "float16",
            "load_in_4bit": False,
            "trust_remote_code": True,
        }
    }
    backend_obj = build_backend(cfg_for_backend, override=backend_kind)
    checker = Checker(
        entailment=TokenOverlapEntailment(threshold=0.5),
        replayer=build_replayer_with_handlers(),
    )
    examples = list(load_dataset_by_name(
        dataset_name, split="validation", n_examples=n, seed=seed,
    ))
    print(f"loaded {len(examples)} examples")

    all_records = []
    for variant_name, vkwargs in VARIANTS:
        for cond in ("clean", "adv"):
            t0 = time.time()
            recs = run_one_variant(
                backend_obj=backend_obj, checker=checker, examples=examples,
                variant_name=variant_name, variant_kwargs=vkwargs,
                adversarial=(cond == "adv"), seed=seed,
            )
            dt = time.time() - t0
            harm_rate = sum(r.get("harm") for r in recs) / max(1, len(recs))
            print(f"  variant={variant_name:14s} cond={cond:5s} harm={harm_rate:.3f}  ({dt:5.1f}s)")
            all_records.extend(recs)

    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"{cell_label}.jsonl"
    with jsonl_path.open("w") as fh:
        for r in all_records:
            r["dataset"] = dataset_name
            r["model"] = paper_model_name
            r["seed"] = seed
            fh.write(json.dumps(r) + "\n")
    summary = {
        "dataset": dataset_name, "model": paper_model_name, "seed": seed,
        "n_examples": len(examples), "epsilon_adv": EPS_ADV, "p_fresh": P_FRESH,
    }
    for variant_name, _ in VARIANTS:
        for cond in ("clean", "adv"):
            recs = [r for r in all_records if r["variant"] == variant_name and r["condition"] == cond]
            if recs:
                summary[f"harm_pcg_{variant_name}_{cond}"] = sum(r.get("harm") for r in recs) / len(recs)
    (out_dir / f"{cell_label}__summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {jsonl_path}")
    print(f"wrote {out_dir / (cell_label + '__summary.json')}")
    return summary


def main() -> int:
    args = parse_args()
    cells = resolve_cells(args.cells)
    global VARIANTS
    VARIANTS = CHANNEL_VARIANTS if args.mode == "channel" else COMPONENT_VARIANTS
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        run_id = time.strftime("%Y%m%d-%H%M%S") + "_ablations"
        out_dir = Path("results/tables/csv/ablations_outputs") / run_id

    summaries = []
    for dataset, paper_name, hf in cells:
        s = run_cell(
            dataset_name=dataset, paper_model_name=paper_name, hf_model_name=hf,
            n=args.n_examples, seed=args.seed, backend_kind=args.backend,
            out_dir=out_dir,
        )
        summaries.append(s)

    if args.mode == "channel":
        from scripts.tables.write_channel_ablation_table import write_channel_ablation_tables
        write_channel_ablation_tables(summaries)

    print(f"\n{'='*70}\nALL CELLS COMPLETE — out_dir: {out_dir}\n{'='*70}")
    print(f"{'cell':40s} | " + " | ".join(
        f"{v[0][:11]:>11s}({c})" for v in VARIANTS for c in ("cl","ad")
    ))
    for s in summaries:
        row = f"{s['dataset']:>10s}:{s['model']:<22s} | "
        for v, _ in VARIANTS:
            for c, full in (("cl","clean"), ("ad","adv")):
                k = f"harm_pcg_{v}_{full}"
                row += f"  {s.get(k, float('nan')):>10.3f}"
                row += "    "
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
