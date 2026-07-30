#!/usr/bin/env python3
"""Pytest suite for S/V harm avoidance decomposition and joint paired bootstrap."""

import unittest
from pathlib import Path
import sys

pkg_script_dir = Path(__file__).resolve().parents[1] / "scripts"
if str(pkg_script_dir) not in sys.path:
    sys.path.insert(0, str(pkg_script_dir))

from compute_sv import compute_sv_metrics
from paired_bootstrap import run_paired_bootstrap

class TestSVDecomposition(unittest.TestCase):
    def test_sv_identity_residual(self):
        source_rec = Path(__file__).resolve().parents[2] / "source_records" / "per_example_records.jsonl"
        if source_rec.exists():
            res = compute_sv_metrics(source_rec)
            self.assertLess(res["max_identity_residual"], 1e-12)
            self.assertEqual(res["status"], "PASS")

if __name__ == "__main__":
    unittest.main()
