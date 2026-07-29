#!/usr/bin/env python3
"""Pytest suite for table reconciliation metrics and shared cell consistency."""

import tempfile
import unittest
from pathlib import Path
import sys

pkg_script_dir = Path(__file__).resolve().parents[1] / "scripts"
if str(pkg_script_dir) not in sys.path:
    sys.path.insert(0, str(pkg_script_dir))

from canonical_metrics import compute_harm_rates
from reconcile_tables import run_reconciliation

class TestTableReconciliation(unittest.TestCase):
    def test_haldane_anscombe_calculation(self):
        m = compute_harm_rates(30, 230, 12, 198)
        self.assertAlmostEqual(m["h_nc"], 30 / 230, places=4)
        self.assertAlmostEqual(m["h_pcg"], 12 / 198, places=4)
        self.assertGreater(m["haldane_anscombe_gain"], 1.5)
        
    def test_reconciliation_on_real_fixture(self):
        source_rec = Path(__file__).resolve().parents[2] / "source_records" / "per_example_records.jsonl"
        if source_rec.exists():
            res = run_reconciliation(source_rec)
            self.assertEqual(res["total_cells"], 56)
            self.assertEqual(res["total_examples_processed"], 13440)
            
    def test_corrupted_fixture_raises_error(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
            tmp.write("invalid json line\n")
            tmp_path = tmp.name
            
        with self.assertRaises(Exception):
            run_reconciliation(tmp_path)

if __name__ == "__main__":
    unittest.main()
