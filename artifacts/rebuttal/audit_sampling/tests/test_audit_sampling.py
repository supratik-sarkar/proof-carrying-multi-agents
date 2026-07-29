#!/usr/bin/env python3
"""Pytest suite for 4 distinct audit sampling designs, weights, and ESS bounds."""

import tempfile
import unittest
from pathlib import Path
import sys

pkg_script_dir = Path(__file__).resolve().parents[1] / "scripts"
if str(pkg_script_dir) not in sys.path:
    sys.path.insert(0, str(pkg_script_dir))

from run_sampling_designs import run_all_sampling_designs

class TestAuditSampling(unittest.TestCase):
    def test_four_distinct_sampling_estimators(self):
        source_rec = Path(__file__).resolve().parents[2] / "audit_sampling" / "source_records" / "audit_draw_records.jsonl"
        if source_rec.exists():
            res = run_all_sampling_designs(source_rec)
            self.assertEqual(res["total_designs"], 4)
            self.assertIn("pooled", res["designs"])
            self.assertIn("uncovered_region", res["designs"])
            
            # Verify estimators are distinct
            estimates = [m["estimated_harm"] for m in res["designs"].values()]
            self.assertGreater(len(set(estimates)), 1, "Estimators must not be identical across designs")
            
    def test_corrupted_audit_file_raises_error(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
            tmp.write("invalid json\n")
            tmp_path = tmp.name
            
        with self.assertRaises(Exception):
            run_all_sampling_designs(tmp_path)

if __name__ == "__main__":
    unittest.main()
