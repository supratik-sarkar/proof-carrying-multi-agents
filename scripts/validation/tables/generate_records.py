#!/usr/bin/env python3
"""Generates 28,000 paired per-example synthetic records (56 cells x 500 examples)."""

import json
import yaml
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
CFG_PATH = Path(__file__).parent / "synthetic_config.yaml"

def generate():
    with CFG_PATH.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = cfg["synthetic_seed"]
    rng = np.random.RandomState(seed)
    models = cfg["models"]
    datasets = cfg["datasets"]

    records = []

    # Model capability multipliers
    cap_map = {
        "phi-3.5-mini": 0.45, "qwen2.5-7b": 0.65, "llama-3.1-8b": 0.70,
        "gemma-2-9b-it": 0.72, "deepseek-llm-7b-chat": 0.68,
        "llama-3.3-70b": 0.88, "deepseek-v3": 0.92
    }

    # Dataset difficulty and risk
    exec_risk_map = {
        "fever": 0.15, "hotpotqa": 0.20, "twowiki": 0.25, "tatqa": 0.35,
        "toolbench": 0.80, "pubmedqa": 0.20, "weblinx": 0.85, "adversarial_integrity": 0.90
    }

    for m in models:
        c_m = cap_map[m]
        for d in datasets:
            r_d = exec_risk_map[d]
            cell_id = f"{m}_{d}"

            base_supp_harm = max(0.04, 0.40 * (1.0 - c_m) + 0.05 * (1.0 - c_m))
            base_exec_harm = max(0.04, 0.45 * (1.0 - c_m) + 0.15 * r_d)

            for i in range(cfg["samples_per_cell"]):
                ex_id = f"{cell_id}_{i}"

                supp_fail_nc = (rng.rand() < base_supp_harm)
                exec_fail_nc = (rng.rand() < base_exec_harm)
                comp_harm_nc = supp_fail_nc or exec_fail_nc

                accepted_pcg = (rng.rand() < (0.85 - 0.04 * (1.0 - c_m)))
                supp_fail_pcg = supp_fail_nc and (rng.rand() < 0.14)
                exec_fail_pcg = exec_fail_nc and (rng.rand() < 0.11)
                comp_harm_pcg = (supp_fail_pcg or exec_fail_pcg) if accepted_pcg else False

                rec = {
                    "provenance": "SYNTHETIC_PLACEHOLDER",
                    "empirical_status": "NOT_EXECUTED",
                    "formal_reporting_allowed": False,
                    "server_run_status": "PENDING",
                    "example_id": ex_id,
                    "model": m,
                    "dataset": d,
                    "cell_id": cell_id,
                    "seed": 0,
                    "synthetic_seed": seed,
                    "system": "PCG-MAS",
                    "ablation": "none",
                    "answered": True,
                    "accepted_nocert": True,
                    "accepted_pcg": accepted_pcg,
                    "support_failure_nocert": supp_fail_nc,
                    "execution_failure_nocert": exec_fail_nc,
                    "composite_harm_nocert": comp_harm_nc,
                    "support_failure_pcg": supp_fail_pcg,
                    "execution_failure_pcg": exec_fail_pcg,
                    "composite_harm_pcg": comp_harm_pcg,
                    "input_tokens": int(420 * (1.0 + c_m * 0.3) + rng.randint(-15, 15)),
                    "output_tokens": int(130 * (1.0 + c_m * 0.3) + rng.randint(-10, 10)),
                    "verifier_tokens": int(190 * (1.0 + c_m * 0.2)),
                    "latency": round(float(175.0 * (1.0 + c_m * 0.5) + rng.uniform(-10, 10)), 2),
                    "cache_state": "hit" if i % 4 == 0 else "miss"
                }
                records.append(rec)

    out_file = REPO_ROOT / "results" / "tables" / "synthetic_placeholder_records.jsonl"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"[GENERATOR] Generated {len(records)} per-example synthetic records into {out_file}")
    return out_file

if __name__ == "__main__":
    generate()
