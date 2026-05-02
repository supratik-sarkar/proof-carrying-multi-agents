#!/usr/bin/env python3
"""Run the PCG-MAS v4 16-cell remote/Colab matrix.

Remote split:
    8 datasets x 2 large models = 16 cells

Outputs:
    results/v4_matrix/remote/<dataset>__<model>/<r1-r5 artifacts>

This script mirrors scripts/v4_run_matrix_local.py but only exposes the
large-model cells intended for Colab/HF Inference.
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


REMOTE_MODELS = [
    {
        "name": "llama-3.3-70b",
        "model_name": "meta-llama/Llama-3.3-70B-Instruct",
        "kind": "hf_inference",
    },
    {
        "name": "deepseek-v3",
        "model_name": "deepseek-ai/DeepSeek-V3",
        "kind": "hf_inference",
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


def _slug(s: str) -> str:
    return (
        s.replace("/", "__")
        .replace(":", "_")
        .replace(" ", "_")
        .replace(".", "_")
        .replace("-", "_")
    )


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _write_yaml(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


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

    selected_backend = {
        "kind": model["kind"],
        "name": model["name"],
        "model_name": model["model_name"],
    }

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
    parser.add_argument("--models", nargs="+", default=[m["name"] for m in REMOTE_MODELS])
    parser.add_argument("--n-examples", type=int, default=50)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--out-root", type=Path, default=Path("results/v4_matrix/remote"))
    parser.add_argument("--work-config-dir", type=Path, default=Path("configs/generated/v4_remote"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-full-run",
        action="store_true",
        help="Required to intentionally run the full 16-cell x R1-R5 remote matrix.",
    )
    args = parser.parse_args()

    env = dict(os.environ)
    env["PYTHONPATH"] = "src" + os.pathsep + env.get("PYTHONPATH", "")

    selected_datasets = [d for d in DATASETS if d["name"] in set(args.datasets)]
    selected_models = [m for m in REMOTE_MODELS if m["name"] in set(args.models)]

    total = len(args.experiments) * len(selected_datasets) * len(selected_models)
    default_full = (
        len(selected_datasets) == len(DATASETS)
        and len(selected_models) == len(REMOTE_MODELS)
        and set(args.experiments) == {"r1", "r2", "r3", "r4", "r5"}
    )

    if default_full and not args.allow_full_run:
        raise SystemExit(
            "\nRefusing to launch the full remote matrix without --allow-full-run.\n"
            "Use explicit --datasets, --models, and --experiments for smoke tests.\n"
        )

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


if __name__ == "__main__":
    main()