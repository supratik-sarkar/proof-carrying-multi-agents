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

import yaml


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


def load_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = project_root() / p
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open("r") as fh:
        return yaml.safe_load(fh)


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


def build_backend(cfg: dict, override: str | None = None):
    """Build the LLM backend specified by cfg['backend']['kind'].

    `override` is a CLI flag that lets the user override the config-specified
    backend without editing the YAML — useful for forcing mock during debugging.
    """
    kind = override or cfg_get(cfg, "backend.kind", "mock")
    if kind == "mock":
        from pcg.backends import MockBackend
        return MockBackend()
    elif kind == "hf_local":
        from pcg.backends.hf_local import HFLocalBackend
        return HFLocalBackend(
            model_name=cfg_get(cfg, "backend.model_name", "Qwen/Qwen2.5-7B-Instruct"),
            dtype=cfg_get(cfg, "backend.dtype", "float16"),
            load_in_4bit=cfg_get(cfg, "backend.load_in_4bit", False),
            trust_remote_code=cfg_get(cfg, "backend.trust_remote_code", False),
        )
    elif kind == "hf_inference":
        from pcg.backends.hf_inference import HFInferenceBackend
        cache = cfg_get(cfg, "backend.cache_dir", str(project_root() / ".cache" / "hf_inference"))
        return HFInferenceBackend(
            model_name=cfg_get(cfg, "backend.model_name", "meta-llama/Llama-3.3-70B-Instruct"),
            cache_dir=cache,
            timeout_s=cfg_get(cfg, "backend.timeout_s", 90.0),
        )
    else:
        raise ValueError(f"Unknown backend kind: {kind}")


# ---------------------------------------------------------------------------
# Output directory + result writing
# ---------------------------------------------------------------------------


def make_output_dir(cfg: dict, run_id: str) -> Path:
    base = Path(cfg_get(cfg, "output_dir", "results"))
    if not base.is_absolute():
        base = project_root() / base
    out = base / run_id
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_json(path: Path, data: Any) -> None:
    """Serialize anything reasonable. Handles dataclasses, sets, numpy."""
    def _coerce(o: Any) -> Any:
        if is_dataclass(o):
            return asdict(o)
        if isinstance(o, (set, frozenset)):
            return sorted(o)
        try:
            import numpy as np
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (np.integer, np.floating)):
                return o.item()
        except ImportError:
            pass
        if hasattr(o, "to_dict"):
            return o.to_dict()
        raise TypeError(f"Not JSON-serializable: {type(o).__name__}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(data, fh, indent=2, default=_coerce, sort_keys=True)


# ---------------------------------------------------------------------------
# Logging shim
# ---------------------------------------------------------------------------


def log_section(title: str) -> None:
    print(f"\n{'='*60}\n  {title}\n{'='*60}", flush=True)


def log_info(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
