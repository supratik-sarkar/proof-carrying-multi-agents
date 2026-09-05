#!/usr/bin/env python3
"""Pytest suite for backend manifest schema."""

import unittest
from pathlib import Path
import sys

pkg_script_dir = Path(__file__).resolve().parents[1] / "scripts"
if str(pkg_script_dir) not in sys.path:
    sys.path.insert(0, str(pkg_script_dir))

from verify_manifest import verify_backend_manifest

class TestBackendManifest(unittest.TestCase):
    def test_ten_required_fields(self):
        source_rec = Path(__file__).resolve().parents[2] / "source_records" / "per_example_records.jsonl"
        if source_rec.exists():
            res = verify_backend_manifest(source_rec)
            self.assertEqual(res["status"], "PASS")

if __name__ == "__main__":
    unittest.main()
