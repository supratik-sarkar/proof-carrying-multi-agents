from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(".")

SKIP_PARTS = {
    ".git",
    "multi-agents",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}

SKIP_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".pdf", ".pyc", ".zip", ".gz", ".ipynb"
}

PATTERNS = {
    "hf_token": re.compile(r"hf_[A-Za-z0-9]{20,}"),
    # Flags literal secrets, but intentionally ignores shell env expansion
    # such as TOKEN="${ANON_HF_TOKEN:-}".
    "generic_secret_assignment": re.compile(
        r"(?i)(api[_-]?key|secret|token|password)\s*=\s*['\"](?!\$\{)[^'\"]{8,}['\"]"
    ),
    "bearer_token": re.compile(r"(?i)Bearer\s+[A-Za-z0-9._\-]{20,}"),
}


def should_skip(path: Path) -> bool:
    return any(part in SKIP_PARTS for part in path.parts) or path.suffix.lower() in SKIP_SUFFIXES


def main() -> int:
    hits: list[str] = []

    for path in ROOT.rglob("*"):
        if not path.is_file() or should_skip(path):
            continue

        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        for name, pattern in PATTERNS.items():
            for match in pattern.finditer(text):
                snippet = match.group(0)
                hits.append(f"{path}: possible {name}: {snippet[:80]}")

    if hits:
        print("\n".join(hits))
        return 1

    print("Secret audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
