#!/usr/bin/env python3
"""Run the PCG-MAS v4 local 40-cell matrix.

Runs 8 datasets x 5 local LLM backends. Resume-safe: each cell writes a
completion marker, and completed cells are skipped unless --force is passed.

This script calls existing R1-R5 scripts by path. It does not require package
installation; PYTHONPATH=src is injected.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

import yaml


LOCAL_MODELS = [
    {
        "name": "phi-3.5-mini",
        "model_name": "microsoft/Phi-3.5-mini-instruct",
        "kind": "hf_local",
    },
    {
        "name": "qwen2.5-7b",
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "kind": "hf_local",
    },
    {
        "name": "deepseek-llm-7b-chat",
        "model_name": "deepseek-ai/deepseek-llm-7b-chat",
        "kind": "hf_local",
    },
    {
        "name": "llama-3.1-8b",
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "kind": "hf_local",
    },
    {
        "name": "gemma-2-9b",
        "model_name": "google/gemma-2-9b-it",
        "kind": "hf_local",
    },
]

DATASETS = [
    {"name": "hotpotqa", "split": "validation"},
    {"name": "2wikimultihopqa", "split": "validation"},
    {"name": "tatqa", "split": "validation"},
    {"name": "toolbench", "split": "validation"},
    {"name": "fever", "split": "validation"},
    {"name": "pubmedqa", "split": "validation"},
    {"name": "weblinx", "split": "validation"},
    {"name": "synthetic_adversarial", "split": "validation"},
]

R_SCRIPTS = {
    "r1": ("scripts/run_r1_checkability.py", "configs/r1_hotpotqa.yaml"),
    "r2": ("scripts/run_r2_redundancy.py", "configs/r2_redundancy.yaml"),
    "r3": ("scripts/run_r3_responsibility.py", "configs/r3_responsibility.yaml"),
    "r4": ("scripts/run_r4_risk_privacy.py", "configs/r4_risk_privacy.yaml"),
    "r5": ("scripts/run_r5_overhead.py", "configs/r5_overhead.yaml"),
}


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _write_yaml(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def _slug(s: str) -> str:
    return (
        s.replace("/", "__")
        .replace(":", "_")
        .replace(" ", "_")
        .replace(".", "_")
        .replace("-", "_")
    )


def _patch_config(base_cfg: dict, dataset: dict, model: dict, n_examples: int, out_root: Path) -> dict:
    cfg = deepcopy(base_cfg)

    cfg.setdefault("dataset", {})
    cfg["dataset"]["name"] = dataset["name"]
    cfg["dataset"]["split"] = dataset.get("split", "validation")
    cfg["dataset"]["n_examples"] = n_examples
    cfg["dataset"]["streaming"] = True

    cfg.setdefault("backend", {})
    cfg["backend"]["kind"] = model["kind"]
    cfg["backend"]["name"] = model["name"]
    cfg["backend"]["model_name"] = model["model_name"]
    # Cell-level backend restriction.
    # Some legacy configs use `models`, while R5 uses `backends`.
    # If these lists are not pruned, a single matrix cell may accidentally
    # iterate through Qwen/DeepSeek even when the selected cell is Phi.
    selected_backend = {
        "kind": model["kind"],
        "name": model["name"],
        "model_name": model["model_name"],
    }

    if "dtype" in model:
        selected_backend["dtype"] = model["dtype"]

    if "models" in cfg:
        cfg["models"] = [dict(selected_backend)]

    if "backends" in cfg:
        r5_backend = dict(selected_backend)
        r5_backend.setdefault("dtype", "float16")
        cfg["backends"] = [r5_backend]

    cfg.setdefault("output", {})
    cfg["output"]["base_dir"] = str(out_root)

    cfg.setdefault("run", {})
    cfg["run"]["dataset_name"] = dataset["name"]
    cfg["run"]["backend_name"] = model["name"]

    return cfg


def _run(cmd: list[str], env: dict) -> None:
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", nargs="+", default=["r1", "r2", "r3", "r4", "r5"])
    parser.add_argument("--datasets", nargs="+", default=[d["name"] for d in DATASETS])
    parser.add_argument("--models", nargs="+", default=[m["name"] for m in LOCAL_MODELS])
    parser.add_argument("--n-examples", type=int, default=50)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--out-root", type=Path, default=Path("results/v4_matrix/local"))
    parser.add_argument("--work-config-dir", type=Path, default=Path("configs/generated/v4_local"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    env = dict(os.environ)
    env["PYTHONPATH"] = "src" + os.pathsep + env.get("PYTHONPATH", "")

    selected_datasets = [d for d in DATASETS if d["name"] in set(args.datasets)]
    selected_models = [m for m in LOCAL_MODELS if m["name"] in set(args.models)]

    total = len(args.experiments) * len(selected_datasets) * len(selected_models)
    print(f"Planned commands: {total}")
    print(f"Datasets: {[d['name'] for d in selected_datasets]}")
    print(f"Models  : {[m['name'] for m in selected_models]}")
    print(f"Exps    : {args.experiments}")

    for ds in selected_datasets:
        for model in selected_models:
            cell = f"{_slug(ds['name'])}__{_slug(model['name'])}"
            cell_dir = args.out_root / cell
            cell_dir.mkdir(parents=True, exist_ok=True)

            for exp in args.experiments:
                if exp not in R_SCRIPTS:
                    raise ValueError(f"Unknown experiment: {exp}")

                marker = cell_dir / f"{exp}.DONE"
                if marker.exists() and not args.force:
                    print(f"skip completed: {cell}/{exp}")
                    continue

                script_path, base_cfg_path = R_SCRIPTS[exp]
                base_cfg = _load_yaml(Path(base_cfg_path))
                cfg = _patch_config(
                    base_cfg,
                    dataset=ds,
                    model=model,
                    n_examples=args.n_examples,
                    out_root=cell_dir / exp,
                )

                cfg_path = args.work_config_dir / cell / f"{exp}.yaml"
                _write_yaml(cfg_path, cfg)

                cmd = [
                    sys.executable,
                    script_path,
                    "--config",
                    str(cfg_path),
                    "--seeds",
                    *[str(s) for s in args.seeds],
                    "--n-examples",
                    str(args.n_examples),
                    "--backend",
                    model["kind"],
                ]

                if args.dry_run:
                    print("DRY:", " ".join(cmd))
                    continue

                start = time.time()
                try:
                    _run(cmd, env=env)
                    marker.write_text(
                        json.dumps(
                            {
                                "cell": cell,
                                "experiment": exp,
                                "dataset": ds["name"],
                                "model": model["name"],
                                "seconds": time.time() - start,
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                except subprocess.CalledProcessError as exc:
                    fail_path = cell_dir / f"{exp}.FAILED.json"
                    fail_path.write_text(
                        json.dumps(
                            {
                                "cell": cell,
                                "experiment": exp,
                                "dataset": ds["name"],
                                "model": model["name"],
                                "returncode": exc.returncode,
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    raise

    if not args.dry_run:
        _run([sys.executable, "scripts/v4_build_paper_artifacts.py"], env=env)


if __name__ == "__main__":
    main()