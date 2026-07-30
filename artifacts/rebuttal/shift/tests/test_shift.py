#!/usr/bin/env python3
"""Pytest suite for shift validity gate classification."""

import tempfile
import unittest
from pathlib import Path
import sys

pkg_script_dir = Path(__file__).resolve().parents[1] / "scripts"
if str(pkg_script_dir) not in sys.path:
    sys.path.insert(0, str(pkg_script_dir))

from apply_validity_gate import apply_shift_validity_gate

class TestShift(unittest.TestCase):
    def test_six_shift_families_honest_classification(self):
        source_rec = Path(__file__).resolve().parents[2] / "source_records" / "per_example_records.jsonl"
        if source_rec.exists():
            res = apply_shift_validity_gate(source_rec)
            self.assertEqual(res["empirical_status"], "NOT_RUN")
            self.assertEqual(res["classification"], "MODELLED")
            self.assertEqual(res["total_families"], 6)

if __name__ == "__main__":
    unittest.main()
