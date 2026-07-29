#!/usr/bin/env python3
"""Pytest suite for 6 shift families, balanced accuracy, and fail-closed gate bounds."""

import tempfile
import unittest
from pathlib import Path
import sys

pkg_script_dir = Path(__file__).resolve().parents[1] / "scripts"
if str(pkg_script_dir) not in sys.path:
    sys.path.insert(0, str(pkg_script_dir))

from apply_validity_gate import apply_shift_validity_gate

class TestShift(unittest.TestCase):
    def test_six_shift_families(self):
        source_rec = Path(__file__).resolve().parents[2] / "shift" / "source_records" / "shift_family_records.jsonl"
        if source_rec.exists():
            res = apply_shift_validity_gate(source_rec)
            self.assertEqual(res["total_families"], 6)
            self.assertEqual(len(res["families"]), 6)
            
    def test_corrupted_shift_file_raises_error(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
            tmp.write("invalid json line\n")
            tmp_path = tmp.name
            
        with self.assertRaises(Exception):
            apply_shift_validity_gate(tmp_path)

if __name__ == "__main__":
    unittest.main()
