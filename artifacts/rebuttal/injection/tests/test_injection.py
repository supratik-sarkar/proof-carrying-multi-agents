#!/usr/bin/env python3
"""Pytest suite for injection attack locations, verifier regimes, and error handling."""

import tempfile
import unittest
from pathlib import Path
import sys

pkg_script_dir = Path(__file__).resolve().parents[1] / "scripts"
if str(pkg_script_dir) not in sys.path:
    sys.path.insert(0, str(pkg_script_dir))

from run_injection_matrix import run_injection_matrix

class TestInjection(unittest.TestCase):
    def test_four_locations_two_regimes(self):
        source_rec = Path(__file__).resolve().parents[2] / "injection" / "source_records" / "injection_sweep_records.jsonl"
        if source_rec.exists():
            res = run_injection_matrix(source_rec)
            self.assertEqual(res["total_locations"], 4)
            self.assertEqual(res["total_regimes"], 2)
            self.assertEqual(len(res["matrix"]), 8)
            
    def test_corrupted_injection_file_raises_error(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
            tmp.write("invalid json\n")
            tmp_path = tmp.name
            
        with self.assertRaises(Exception):
            run_injection_matrix(tmp_path)

if __name__ == "__main__":
    unittest.main()
