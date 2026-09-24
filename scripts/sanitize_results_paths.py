#!/usr/bin/env python3
"""Sanitize local absolute paths in results/ directory."""

import re
from pathlib import Path

RES_DIR = Path(__file__).resolve().parent.parent / "results"

for p in RES_DIR.rglob("*"):
    if p.is_file() and not p.name.startswith("."):
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
            if "/Users/" in txt:
                txt = re.sub(r'/Users/[^/]+/Desktop/pcg-neurips2026/', './', txt)
                p.write_text(txt, encoding="utf-8")
        except Exception:
            pass

print("Sanitized local absolute paths in results/ directory")
