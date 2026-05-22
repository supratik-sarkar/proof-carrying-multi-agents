from __future__ import annotations

from pathlib import Path

FORBIDDEN = {
    "smo" + "ke",
    "pro" + "xy",
}

SKIP_PARTS = {
    ".git",
    ".venv",
    "venv",
    "multi-agents",
    "__pycache__",
    ".pytest_cache",
}

SKIP_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".pdf",
    ".pyc",
    ".ipynb",
    ".zip",
    ".gz",
}

ROOT = Path(".")


def should_skip(path: Path) -> bool:
    if any(part in SKIP_PARTS for part in path.parts):
        return True
    if path.suffix.lower() in SKIP_SUFFIXES:
        return True
    return False


def main() -> int:
    hits: list[str] = []

    for path in ROOT.rglob("*"):
        if not path.is_file() or should_skip(path):
            continue

        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        lowered = text.lower()
        for term in FORBIDDEN:
            if term in lowered:
                hits.append(f"{path}: contains forbidden term `{term}`")

    if hits:
        print("\n".join(hits))
        return 1

    print("Forbidden-term audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
