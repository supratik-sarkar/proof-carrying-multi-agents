from __future__ import annotations

import argparse
import getpass
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from pcg.utils.hf_auth import resolve_hf_token


DEFAULT_MODELS = [
    "phi-3.5-mini",
    "qwen2.5-7B",
    "Llama-3.1-8B",
    "Gemma-2-9b-it",
    "Llama-3.3-70B",
]

DEFAULT_DATASETS = [
    "fever",
    "hotpotqa",
    "twowiki",
    "pubmedqa",
    "tatqa",
    "toolbench",
    "weblinx",
    "synthetic",
]

FRONTIER_MODELS = [
    "Llama-3.3-70B",
    "deepseek-v3",
]

LOCAL40_MODELS = [
    "phi-3.5-mini",
    "qwen2.5-7B",
    "Llama-3.1-8B",
    "Gemma-2-9b-it",
    "Llama-3.3-70B",
]

PAPER_DATASETS = [
    "fever",
    "hotpotqa",
    "twowiki",
    "pubmedqa",
    "tatqa",
    "toolbench",
    "weblinx",
    "synthetic",
]

DEFAULT_EXPERIMENTS = ["r1", "r2", "r3", "r4", "r5"]

MODEL_TO_HF_REPO = {
    "phi-3.5-mini": "microsoft/Phi-3.5-mini-instruct",
    "qwen2.5-7B": "Qwen/Qwen2.5-7B-Instruct",
    "Llama-3.1-8B": "meta-llama/Llama-3.1-8B-Instruct",
    "Gemma-2-9b-it": "google/gemma-2-9b-it",
    "gemma-2-9b-it": "google/gemma-2-9b-it",
    "Llama-3.3-70B": "meta-llama/Llama-3.3-70B-Instruct",
    "deepseek-v3": "deepseek-ai/DeepSeek-V3",
}

EXPERIMENT_TO_SCRIPT = {
    "r1": "scripts/experiments/run_r1_checkability.py",
    "r2": "scripts/experiments/run_r2_redundancy.py",
    "r3": "scripts/experiments/run_r3_responsibility.py",
    "r4": "scripts/experiments/run_r4_risk_privacy.py",
    "r5": "scripts/experiments/run_r5_overhead.py",
}

DATASET_CONFIGS = {
    "fever": "configs/r1_fever.yaml",
    "hotpotqa": "configs/r1_hotpotqa.yaml",
    "twowiki": "configs/r6_cross_domain.yaml",
    "2wikimultihopqa": "configs/r6_cross_domain.yaml",
    "pubmedqa": "configs/r1_pubmedqa.yaml",
    "tatqa": "configs/r1_tatqa.yaml",
    "toolbench": "configs/r6_cross_domain.yaml",
    "weblinx": "configs/r1_weblinx.yaml",
    "synthetic": "configs/synthetic.yaml",
}


def normalize_dataset_name(name: str) -> str:
    key = name.strip().lower().replace("_", "-")
    aliases = {
        "fever": "fever",
        "hotpotqa": "hotpotqa",
        "hotpot-qa": "hotpotqa",
        "twowiki": "twowiki",
        "2wiki": "twowiki",
        "2wikimultihopqa": "twowiki",
        "2wiki-multihopqa": "twowiki",
        "pubmedqa": "pubmedqa",
        "pubmed-qa": "pubmedqa",
        "tatqa": "tatqa",
        "tat-qa": "tatqa",
        "toolbench": "toolbench",
        "weblinx": "weblinx",
        "web-linx": "weblinx",
        "synthetic": "synthetic",
    }
    return aliases.get(key, key)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flexible PCG-MAS matrix runner for R1--R5 experiments."
    )

    parser.add_argument("--split", choices=["custom", "local40", "frontier16"], default="custom")
    parser.add_argument("--experiments", nargs="+", default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--n-examples", type=int, default=10)

    parser.add_argument(
        "--backend",
        choices=["mock", "hf_local", "hf_inference"],
        default="mock",
    )
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument("--non-interactive", action="store_true")

    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-full-run", action="store_true")
    parser.add_argument(
        "--allow-dataset-fallback",
        action="store_true",
        help="Allow fallback dataset rows for environment/preflight checks.",
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/matrix_run_manifest.json"),
    )

    return parser.parse_args()


def build_command(
    *,
    experiment: str,
    dataset: str,
    model: str,
    seeds: list[int],
    n_examples: int,
    backend: str,
) -> list[str]:
    script = EXPERIMENT_TO_SCRIPT[experiment]
    config = DATASET_CONFIGS.get(dataset)

    cmd = [sys.executable, script]

    if config is not None and Path(config).exists():
        cmd.extend(["--config", config])

    cmd.extend(["--seeds", *map(str, seeds)])
    cmd.extend(["--n-examples", str(n_examples)])
    cmd.extend(["--backend", backend])

    # These flags are accepted by the newer runner convention.
    # If an older R-script ignores model/dataset CLI args, the manifest still
    # records the requested cell and the script-level config controls execution.
    cmd.extend(["--dataset", dataset])
    cmd.extend(["--model", model])

    return cmd



def resolve_hf_local_backend(args: argparse.Namespace) -> None:
    """Request HF_TOKEN for hf_local runs using hidden input.

    Pressing ENTER without a token switches this run to backend=mock.
    """
    if args.backend != "hf_local":
        return

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = args.hf_token
        print("[hf_local] HF token received from --hf-token. Using backend=hf_local.")
        return

    print("\n[hf_local] Hugging Face local model downloads may require a personal HF token.")
    print("[hf_local] Token input is hidden and will not be printed in the terminal.")
    print("[hf_local] Press ENTER without pasting a token to fall back to backend=mock.\n")

    token = getpass.getpass("HF_TOKEN [hidden; ENTER => backend=mock]: ").strip()

    if not token:
        print("[hf_local] No HF_TOKEN provided. Falling back to backend=mock for this run.")
        args.backend = "mock"
        return

    os.environ["HF_TOKEN"] = token
    os.environ["HUGGINGFACE_HUB_TOKEN"] = token
    print("[hf_local] HF_TOKEN received and stored in-process only. Using backend=hf_local.")



def main() -> int:
    args = parse_args()
    resolve_hf_local_backend(args)

    experiments = [x.lower() for x in args.experiments]

    if args.split == "frontier16":
        datasets = PAPER_DATASETS
        models = FRONTIER_MODELS
        if args.backend == "mock":
            args.backend = "hf_inference"
    elif args.split == "local40":
        datasets = PAPER_DATASETS
        models = LOCAL40_MODELS
    else:
        datasets = [normalize_dataset_name(x) for x in args.datasets]
        models = args.models

    unknown_exps = [x for x in experiments if x not in EXPERIMENT_TO_SCRIPT]
    if unknown_exps:
        raise SystemExit(f"Unknown experiment(s): {unknown_exps}. Known: {sorted(EXPERIMENT_TO_SCRIPT)}")

    if args.backend == "hf_inference":
        auth = resolve_hf_token(
            explicit_token=args.hf_token,
            require_for_full=True,
            interactive=not args.non_interactive,
        )
        print(auth.message)

        if not auth.full_access:
            print(
                "You entered a null HF token and hence remote/gated HF model cells "
                "will not run. Switching to backend=mock for the feasible offline path. "
                "This is not a full model rerun."
            )
            args.backend = "mock"

    if args.allow_dataset_fallback:
        os.environ.setdefault("PCG_ALLOW_DATASET_FALLBACK", "1")

    is_large_run = (
        args.n_examples >= 500
        or len(args.seeds) >= 4
        or len(experiments) * len(datasets) * len(models) > 40
    )

    if is_large_run and not args.allow_full_run and not args.dry_run:
        raise SystemExit(
            "Blocked: this looks like a large run. Re-run with --allow-full-run, "
            "or use --dry-run to inspect the planned commands."
        )

    commands = []
    for experiment in experiments:
        for dataset in datasets:
            for model in models:
                cmd = build_command(
                    experiment=experiment,
                    dataset=dataset,
                    model=model,
                    seeds=args.seeds,
                    n_examples=args.n_examples,
                    backend=args.backend,
                )
                model_name = MODEL_TO_HF_REPO.get(model, model)
                commands.append(
                    {
                        "experiment": experiment,
                        "dataset": dataset,
                        "model": model,
                        "model_name": model_name,
                        "command": cmd,
                    }
                )

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": args.dry_run,
        "n_examples": args.n_examples,
        "seeds": args.seeds,
        "experiments": experiments,
        "datasets": datasets,
        "models": models,
        "backend": args.backend,
        "allow_dataset_fallback": args.allow_dataset_fallback,
        "num_commands": len(commands),
        "commands": commands,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(
        {
            "dry_run": args.dry_run,
            "num_commands": len(commands),
            "n_examples": args.n_examples,
            "seeds": args.seeds,
            "backend": args.backend,
            "manifest": str(args.out),
        },
        indent=2,
    ))

    for item in commands:
        printable = " ".join(item["command"])
        model_name = item.get("model_name") or MODEL_TO_HF_REPO.get(item.get("model", ""), item.get("model", ""))

        if args.dry_run:
            print("[dry-run]", printable)
            print(f"[model-map] {item.get('model')} -> {model_name}")
            continue

        env = os.environ.copy()
        if model_name:
            env["PCG_BACKEND_MODEL_NAME"] = model_name

        print("[run]", printable)
        print(f"[model-map] {item.get('model')} -> {model_name}", flush=True)
        subprocess.run(item["command"], check=True, env=env)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
