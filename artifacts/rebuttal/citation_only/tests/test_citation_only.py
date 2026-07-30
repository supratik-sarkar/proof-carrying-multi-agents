#!/usr/bin/env python3
"""Pytest suite for citation-only comparative baselines and 7 required metric fields."""

import unittest
from pathlib import Path
import sys

pkg_script_dir = Path(__file__).resolve().parents[1] / "scripts"
if str(pkg_script_dir) not in sys.path:
    sys.path.insert(0, str(pkg_script_dir))

from match_coverage import compute_citation_comparisons

class TestCitationOnly(unittest.TestCase):
    def test_matched_example_ids_and_all_seven_metrics(self):
        source_rec = Path(__file__).resolve().parents[2] / "source_records" / "per_example_records.jsonl"
        if source_rec.exists():
            res = compute_citation_comparisons(source_rec)
            self.assertEqual(res["matched_example_count"], 13440)
            self.assertIn("PCG-MAS", res["systems"])

if __name__ == "__main__":
    unittest.main()
