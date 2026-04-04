from __future__ import annotations

import os
from pathlib import Path


def get_env(name: str, default: str | None = None) -> str | None:
    return os.getenv(name, default)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
