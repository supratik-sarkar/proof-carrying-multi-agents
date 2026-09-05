"""
Shared utilities for R1-R5 experiment scripts.

Lives in scripts/ rather than src/pcg/ because it's experiment glue, not part
of the public API. The experiment scripts import from here.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def log_info(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)


def log_success(msg: str) -> None:
    print(f"[SUCCESS] {msg}", flush=True)


def log_warning(msg: str) -> None:
    print(f"[WARNING] {msg}", flush=True)


def log_error(msg: str) -> None:
    print(f"[ERROR] {msg}", flush=True)


def log_stage(stage: str, msg: str) -> None:
    print(f"[{stage}] {msg}", flush=True)


def log_section(name: str) -> None:
    print(f"\n=== {name} ===", flush=True)


# ---------------------------------------------------------------------------
# Project root resolution (so scripts can be run from anywhere)
# ---------------------------------------------------------------------------


def project_root() -> Path:
    """Return the absolute path to the project root."""
    here = Path(__file__).resolve()
    for ancestor in [here.parent, *here.parents]:
        if (ancestor / "pyproject.toml").exists():
            return ancestor
    return here.parent


def make_output_dir(cfg_or_name: Any, run_id: str | None = None) -> Path:
    if isinstance(cfg_or_name, str):
        name = cfg_or_name
    elif isinstance(cfg_or_name, dict):
        name = cfg_get(cfg_or_name, "experiment.id", "exp")
    else:
        name = str(cfg_or_name)

    if run_id:
        p = project_root() / "results" / "tables" / "csv" / "experiment_json" / f"{run_id}_{name}"
    else:
        p = project_root() / "results" / "tables" / "csv" / "experiment_json" / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path_or_data: Any, data_or_path: Any) -> None:
    if isinstance(path_or_data, (str, Path)):
        p = Path(path_or_data)
        d = data_or_path
    else:
        d = path_or_data
        p = Path(data_or_path)

    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as fh:
        json.dump(d, fh, indent=2, default=str)


def read_json(path: str | Path) -> Any:
    p = Path(path)
    with p.open("r") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Run-id and provenance
# ---------------------------------------------------------------------------


def git_sha() -> str:
    """Short git SHA, or 'unknown' if not in a git repo."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=project_root(), stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def make_run_id(config_path: str | Path) -> str:
    """Run id = <ts>_<config-stem>_<config-hash>_<git-sha>."""
    p = Path(config_path)
    text = p.read_text() if p.exists() else str(p)
    config_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    ts = time.strftime("%Y%m%d-%H%M%S")
    return f"{ts}_{p.stem}_{config_hash}_{git_sha()}"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def _simple_yaml_load(text: str) -> dict[str, Any]:
    """Lightweight fallback YAML parser for standard key-value configs when PyYAML is absent."""
    result: dict[str, Any] = {}
    stack = [(0, result)]

    for line in text.splitlines():
        # Strip inline comment
        clean_line = line.split("#")[0]
        line_strip = clean_line.strip()
        if not line_strip:
            continue

        indent = len(clean_line) - len(clean_line.lstrip())

        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()

        target = stack[-1][1]

        if ":" in line_strip:
            k, v = line_strip.split(":", 1)
            k = k.strip()
            v = v.strip()

            if not v:
                new_dict: dict[str, Any] = {}
                target[k] = new_dict
                stack.append((indent, new_dict))
            else:
                if v.lower() == "true":
                    val = True
                elif v.lower() == "false":
                    val = False
                elif v.lower() in ("null", "~", "none"):
                    val = None
                else:
                    try:
                        val = int(v) if "." not in v else float(v)
                    except ValueError:
                        val = v.strip('"').strip("'")
                target[k] = val
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = project_root() / p
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")

    try:
        import yaml
        with p.open("r") as fh:
            return yaml.safe_load(fh)
    except ImportError:
        return _simple_yaml_load(p.read_text())


def cfg_get(cfg: dict, dotted_key: str, default=None):
    """Safe nested-dict access: cfg_get(c, 'prover.top_k') -> c['prover']['top_k']."""
    cur: Any = cfg
    for k in dotted_key.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


# ---------------------------------------------------------------------------
# Backend instantiation
# ---------------------------------------------------------------------------


def resolved_backend_model_name(cfg: dict, default: str) -> str:
    """Resolve actual backend HF repo."""
    env_override = os.environ.get("PCG_BACKEND_MODEL_NAME")
    if env_override and env_override.strip():
        return env_override.strip()
    return cfg_get(cfg, "backend.model_name", default)


def create_backend(cfg: dict, override: str | None = None, backend_type: str | None = None) -> Any:
    """Factory creating an LLMBackend instance from config."""
    from pcg.backends.mock import MockBackend
    from pcg.backends.hf_local import HFLocalBackend
    from pcg.backends.hf_inference import HFInferenceBackend
    from pcg.backends.deepseek import DeepSeekBackend

    btype = override or backend_type or cfg_get(cfg, "backend.type", cfg_get(cfg, "backend.kind", "mock"))

    if btype == "mock":
        return MockBackend()
    elif btype == "hf_local":
        mname = resolved_backend_model_name(cfg, "phi-3.5-mini")
        dtype = cfg_get(cfg, "backend.dtype", "float16")
        l4b = cfg_get(cfg, "backend.load_in_4bit", False)
        return HFLocalBackend(model_name=mname, dtype=dtype, load_in_4bit=l4b)
    elif btype == "hf_inference":
        mname = resolved_backend_model_name(cfg, "phi-3.5-mini")
        token = os.environ.get("HF_INFERENCE") or os.environ.get("HF_HUB_READ") or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        return HFInferenceBackend(model_name=mname, token=token)
    elif btype == "deepseek":
        mname = resolved_backend_model_name(cfg, "deepseek-chat")
        token = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("DEEPSEEK_TOKEN")
        return DeepSeekBackend(model_name=mname, token=token)
    else:
        raise ValueError(f"Unknown backend type: {btype}")

build_backend = create_backend
