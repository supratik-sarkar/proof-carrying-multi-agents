#!/usr/bin/env python3
"""Pytest suite for separating witnesses."""

import unittest
from pathlib import Path
import sys

pkg_script_dir = Path(__file__).resolve().parents[1] / "scripts"
if str(pkg_script_dir) not in sys.path:
    sys.path.insert(0, str(pkg_script_dir))

from run_witness_suite import evaluate_witness_suite

class TestSeparatingWitnesses(unittest.TestCase):
    def test_witness_channels(self):
        source_rec = Path(__file__).resolve().parents[2] / "source_records" / "per_example_records.jsonl"
        res = evaluate_witness_suite(source_rec)
        self.assertEqual(res["total_witnesses"], 4)

if __name__ == "__main__":
    unittest.main()
