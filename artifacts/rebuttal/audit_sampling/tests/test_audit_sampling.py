#!/usr/bin/env python3
"""Pytest suite for 4 audit sampling designs and honest status classification."""

import unittest
from pathlib import Path
import sys

pkg_script_dir = Path(__file__).resolve().parents[1] / "scripts"
if str(pkg_script_dir) not in sys.path:
    sys.path.insert(0, str(pkg_script_dir))

from run_sampling_designs import run_all_sampling_designs

class TestAuditSampling(unittest.TestCase):
    def test_four_estimators_and_honest_classification(self):
        source_rec = Path(__file__).resolve().parents[2] / "source_records" / "per_example_records.jsonl"
        if source_rec.exists():
            res = run_all_sampling_designs(source_rec)
            self.assertEqual(res["empirical_status"], "NOT_RUN")
            self.assertEqual(res["classification"], "MODELLED")
            self.assertEqual(len(res["designs"]), 4)

if __name__ == "__main__":
    unittest.main()
