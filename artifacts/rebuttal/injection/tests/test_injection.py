#!/usr/bin/env python3
"""Pytest suite for injection sweep: verify honest classification and error handling."""

import tempfile
import unittest
from pathlib import Path
import sys

pkg_script_dir = Path(__file__).resolve().parents[1] / "scripts"
if str(pkg_script_dir) not in sys.path:
    sys.path.insert(0, str(pkg_script_dir))

from run_injection_matrix import run_injection_matrix

class TestInjection(unittest.TestCase):
    def test_honest_not_run_classification(self):
        source_rec = Path(__file__).resolve().parents[2] / "source_records" / "per_example_records.jsonl"
        if source_rec.exists():
            res = run_injection_matrix(source_rec)
            self.assertEqual(res["empirical_status"], "NOT_RUN")
            self.assertEqual(res["classification"], "MODELLED")

    def test_corrupted_file_raises_error(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
            tmp.write("invalid json\n")
            tmp_path = tmp.name

        with self.assertRaises(Exception):
            run_injection_matrix(tmp_path)

if __name__ == "__main__":
    unittest.main()
