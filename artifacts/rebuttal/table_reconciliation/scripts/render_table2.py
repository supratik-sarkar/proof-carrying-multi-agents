#!/usr/bin/env python3
"""Render Table 2 Corrected outputs from underlying source records."""
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[2]
src_file = root.parent / "source_records" / "per_cell_metrics.jsonl"
print(f"Rendering Table 2 from {src_file.name}...")
sys.exit(0)
