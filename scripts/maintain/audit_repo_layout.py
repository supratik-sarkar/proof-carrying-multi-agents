from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(".")

FORBIDDEN_OUTPUT_DIRS = [
    Path("artifacts/v5_preview"),
    Path("docs/tables"),
    Path("latex/tables"),
    Path("results/v4"),
    Path("results/v5"),
    Path("figures/v4"),
]

FORBIDDEN_FILE_PATTERNS = [
    "*/__pycache__/*",
    "*.pyc",
    ".DS_Store",
]

IGNORED_DIRS = {
    ".git",
    "multi-agents",
}

LATEX_ALLOWED = {
    Path("latex/experiments.tex"),
    Path("latex/appendix_exp_details.tex"),
}


def main() -> int:
    errors: list[str] = []

    for path in FORBIDDEN_OUTPUT_DIRS:
        if path.exists():
            errors.append(f"Forbidden generated-output directory exists: {path}")

    if Path("assets").exists():
        errors.append("`assets/` exists. Rename/move it to `workflow/`.")

    if Path("latex").exists():
        for child in Path("latex").iterdir():
            if child not in LATEX_ALLOWED:
                errors.append(f"Forbidden file/folder inside latex/: {child}")

    for pattern in FORBIDDEN_FILE_PATTERNS:
        for path in ROOT.rglob(pattern):
            if not any(part in IGNORED_DIRS for part in path.parts):
                errors.append(f"Junk file found: {path}")

    if errors:
        print("\n".join(errors))
        return 1

    print("Repo layout audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
