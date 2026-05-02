"""
R5: Time / token overhead.

Directly addresses ICML W2: "missing details on the time/token overhead".

For each (backend, configuration) pair, runs the full Prover->Verifier
pipeline on N examples and reports per-phase token/latency totals from the
Meter. Configurations span:
    - baseline_no_pcg : straight QA, no certificate
    - pcg_k1 / pcg_k2 / pcg_k4 : PCG with redundancy levels 1, 2, 4

Output: results/<run_id>/r5.json with stacked-bar-ready per-phase data.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from _common import (
    cfg_get,
    load_config,
    log_info,
    log_section,
    make_output_dir,
    make_run_id,
    write_json,
)


def _build_backend_from_dict(d: dict, override: str | None = None):
    kind = override or d.get("kind", "mock")
    if kind == "mock":
        from pcg.backends import MockBackend
        return MockBackend()
    if kind == "hf_local":
        from pcg.backends.hf_local import HFLocalBackend
        return HFLocalBackend(
            model_name=d.get("model_name", "Qwen/Qwen2.5-7B-Instruct"),
            dtype=d.get("dtype", "float16"),
        )
    if kind == "hf_inference":
        from pcg.backends.hf_inference import HFInferenceBackend
        return HFInferenceBackend(model_name=d["model_name"])
    raise ValueError(kind)


def main(
    config: str = "configs/r5_overhead.yaml",
    seeds: list[int] | None = None,
    n_examples: int | None = None,
    backend: str | None = None,
) -> int:
    seeds = seeds or [0, 1, 2, 3, 4]
    cfg = load_config(config)
    if n_examples is not None:
        cfg.setdefault("dataset", {})["n_examples"] = n_examples
    run_id = make_run_id(config)
    out_dir = make_output_dir(cfg, run_id)
    log_section(f"R5 run: {run_id}")
    write_json(out_dir / "config_snapshot.json", cfg)

    from pcg.checker import Checker, ExactMatchEntailment
    from pcg.datasets import load_dataset_by_name
    from pcg.eval import bootstrap_ci
    from pcg.eval.meter import Meter
    from pcg.orchestrator.langgraph_flow import (
        OrchestratorConfig,
        PCGState,
        run_one_example,
    )
    from pcg.orchestrator import build_replayer_with_handlers
    from pcg.agents.prover import ProverConfig, build_default_prover

    backends_cfg = cfg_get(cfg, "backends", [{"kind": "mock"}])
    if backend:
        backends_cfg = [{"kind": backend}]
    configurations = cfg_get(cfg, "configurations", [
        {"name": "pcg_k1", "enable_pcg": True, "k": 1},
    ])
    n = cfg_get(cfg, "dataset.n_examples", 100)

    seed_results = []
    for seed in seeds:
        log_section(f"seed={seed}")
        examples_master = list(load_dataset_by_name(
            cfg_get(cfg, "dataset.name", "hotpotqa"),
            split=cfg_get(cfg, "dataset.split", "validation"),
            n_examples=n, seed=seed,
        ))

        per_combo = []
        for backend_cfg in backends_cfg:
            backend_obj = _build_backend_from_dict(backend_cfg)
            for conf in configurations:
                conf_name = conf["name"]
                k = conf.get("k", 1)
                enable_pcg = conf.get("enable_pcg", True)
                log_info(f"  backend={backend_obj.name}  conf={conf_name}  k={k}")

                checker = Checker(
                    entailment=ExactMatchEntailment(case_insensitive=True),
                    replayer=build_replayer_with_handlers(),
                )

                # Aggregate phases across examples, summed
                agg_phases: dict[str, dict] = {}
                wall_total = 0.0
                tok_total = 0

                for ex in examples_master:
                    if not enable_pcg:
                        # Baseline: just generate, no Prover/Checker stack
                        meter = Meter()
                        with meter.phase("llm_gen"):
                            ctx = "\n".join(e.text for e in ex.evidence[:4])
                            prompt = (
                                f"Question: {ex.question}\n"
                                f"Context: {ctx}\n"
                                f"Answer:"
                            )
                            out = backend_obj.generate(prompt, max_tokens=64)
                            meter.record_tokens(tokens_in=out.tokens_in, tokens_out=out.tokens_out)
                        rep = meter.report()
                    else:
                        # k branches
                        states = []
                        meter = Meter()
                        for j in range(k):
                            pcfg = ProverConfig(
                                top_k=cfg_get(cfg, "prover.top_k", 4),
                                temperature=0.0, seed=seed * 100 + j,
                            )
                            prover_fn = build_default_prover(backend=backend_obj, config=pcfg)
                            state = PCGState(example=ex, meter=meter)
                            state = prover_fn(state)
                            with meter.phase("verifier"):
                                if state.certificate is not None:
                                    checker.check(state.certificate, state.graph)
                            states.append(state)
                        rep = meter.report()

                    wall_total += rep.wall_ms
                    tok_total += rep.total_tokens()
                    for ph_name, ph in rep.phases.items():
                        agg = agg_phases.setdefault(ph_name, {
                            "ms": 0.0, "tok_in": 0, "tok_out": 0, "n": 0,
                        })
                        agg["ms"] += ph.total_ms
                        agg["tok_in"] += ph.total_tokens_in
                        agg["tok_out"] += ph.total_tokens_out
                        agg["n"] += ph.n_calls

                per_combo.append({
                    "backend": backend_obj.name,
                    "config": conf_name,
                    "k": k,
                    "n_examples": len(examples_master),
                    "wall_ms_total": wall_total,
                    "wall_ms_per_claim": wall_total / max(1, len(examples_master)),
                    "tokens_per_claim": tok_total / max(1, len(examples_master)),
                    "phases": agg_phases,
                })

        seed_results.append({"seed": seed, "per_combo": per_combo})
        write_json(out_dir / f"seed_{seed}.json", seed_results[-1])

    # Aggregate (mean per-claim across seeds + bootstrap CI)
    combos = {}
    for s in seed_results:
        for c in s["per_combo"]:
            key = (c["backend"], c["config"])
            combos.setdefault(key, []).append(c)
    aggregated = []
    for (bnd, conf), entries in combos.items():
        wmps = [e["wall_ms_per_claim"] for e in entries]
        tps = [e["tokens_per_claim"] for e in entries]
        ci_w = bootstrap_ci(wmps, n_boot=cfg_get(cfg, "stats.bootstrap_n", 1000))
        ci_t = bootstrap_ci(tps, n_boot=cfg_get(cfg, "stats.bootstrap_n", 1000))
        # Per-phase aggregates
        phases = {}
        for entry in entries:
            for ph_name, ph in entry["phases"].items():
                p = phases.setdefault(ph_name, {"ms": [], "tokens": []})
                p["ms"].append(ph["ms"] / entry["n_examples"])
                p["tokens"].append((ph["tok_in"] + ph["tok_out"]) / entry["n_examples"])
        phase_means = {ph: {"ms_mean": float(np.mean(d["ms"])),
                              "tokens_mean": float(np.mean(d["tokens"]))}
                        for ph, d in phases.items()}
        aggregated.append({
            "backend": bnd, "config": conf,
            "wall_ms_per_claim_mean": ci_w.estimate,
            "wall_ms_per_claim_ci": [ci_w.lower, ci_w.upper],
            "tokens_per_claim_mean": ci_t.estimate,
            "tokens_per_claim_ci": [ci_t.lower, ci_t.upper],
            "phases": phase_means,
        })

    final = {"run_id": run_id, "seeds": seeds, "per_seed": seed_results,
              "aggregated": aggregated}
    write_json(out_dir / "r5.json", final)
    log_section("R5 complete")
    log_info(f"Result: {out_dir / 'r5.json'}")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/r5_overhead.yaml")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--n-examples", type=int, default=None)
    p.add_argument("--backend", default=None)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    sys.exit(main(config=a.config, seeds=a.seeds,
                   n_examples=a.n_examples, backend=a.backend))
