"""Deterministic code identity for a NON-GIT workspace.

The working repository is intentionally not under version control, so a commit
SHA cannot be the execution identity. Instead the fingerprint is the hash of the
canonical hashes of the active source tree. `git_commit` remains an optional
field for later, never a requirement.
"""
from __future__ import annotations

import json
from pathlib import Path

from .hashing import hash_file, hash_obj

#: Tracked roots. Anything outside these cannot change the fingerprint.
DEFAULT_INCLUDE = ("src", "scripts", "configs", "pyproject.toml")

#: Excluded because they are environment, output, data or secrets — not code.
EXCLUDE_DIRS = {
    ".venv", ".venv-pcg-mas", "venv", "pcg-iclr2027.venv", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", ".git", "node_modules", "results", "runs", "reports", "quarantine",
    "artifacts", "data", "external", ".sota_src", "figures", "site-packages",
}
EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".so", ".log", ".lock", ".env", ".pem", ".key"}
EXCLUDE_NAMES = {".DS_Store", ".env", ".env.local", "credentials.json", "secrets.yaml"}


def iter_tracked(root: Path, include=DEFAULT_INCLUDE):
    root = Path(root)
    for rel in include:
        p = root / rel
        if p.is_file():
            yield p
        elif p.is_dir():
            for f in sorted(p.rglob("*")):
                if not f.is_file():
                    continue
                if any(part in EXCLUDE_DIRS for part in f.parts):
                    continue
                if f.suffix in EXCLUDE_SUFFIXES or f.name in EXCLUDE_NAMES:
                    continue
                yield f


def code_fingerprint(root: str | Path = ".", include=DEFAULT_INCLUDE) -> dict:
    """Content-addressed identity of the active source tree.

    Stable across: timestamps, file ordering, working directory, absolute paths.
    Sensitive to: any byte of tracked source.
    """
    root = Path(root).resolve()
    entries = {}
    for f in iter_tracked(root, include):
        entries[str(f.relative_to(root))] = hash_file(f)
    return {
        "algorithm": "sha256-of-canonical-file-hashes",
        "included_roots": list(include),
        "file_count": len(entries),
        "fingerprint": hash_obj(entries),
        "git_commit": None,   # optional; populated only if a repo appears later
    }


def write_fingerprint(root: str | Path, out: str | Path) -> dict:
    fp = code_fingerprint(root)
    Path(out).write_text(json.dumps(fp, indent=2) + "\n", encoding="utf-8")
    return fp
