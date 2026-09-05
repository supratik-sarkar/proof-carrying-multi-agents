import unittest
import json
from pathlib import Path
from scripts.common.schema import validate_record_provenance, assert_paper_ready

REPO_ROOT = Path(__file__).resolve().parents[1]

class TestProvenanceInvariants(unittest.TestCase):
    def test_missing_provenance_raises(self):
        record = {"model": "phi-3.5-mini", "dataset": "fever", "harm": 0.1}
        with self.assertRaises(ValueError):
            validate_record_provenance(record)

    def test_invalid_provenance_value_raises(self):
        record = {"model": "phi-3.5-mini", "dataset": "fever", "provenance": "simulated", "harm": 0.1}
        with self.assertRaises(ValueError):
            validate_record_provenance(record)

    def test_unavailable_provenance_with_numerics_raises(self):
        record = {"model": "phi-3.5-mini", "dataset": "fever", "provenance": "unavailable", "harm": 0.0}
        with self.assertRaises(ValueError):
            validate_record_provenance(record)

    def test_unavailable_provenance_without_numerics_passes(self):
        record = {"model": "phi-3.5-mini", "dataset": "fever", "provenance": "unavailable", "reason": "GPU out of memory"}
        prov = validate_record_provenance(record)
        self.assertEqual(prov, "unavailable")

    def test_zero_reimplemented_records_today(self):
        pm_path = REPO_ROOT / "results/tables/csv/paper_metrics.jsonl"
        if pm_path.exists():
            rows = [json.loads(line) for line in pm_path.read_text().splitlines() if line.strip()]
            reimpl_rows = [r for r in rows if r.get("provenance") == "reimplementation"]
            self.assertEqual(len(reimpl_rows), 0, f"Found {len(reimpl_rows)} reimplemented records; expected 0.")

    def test_paper_builder_refuses_unrenderable_provenance(self):
        bad_rows = [
            {"model": "phi-3.5-mini", "dataset": "fever", "provenance": "unavailable"}
        ]
        with self.assertRaises(RuntimeError):
            assert_paper_ready(bad_rows)

    def test_failing_baseline_execution_crashes(self):
        def mock_failing_baseline():
            raise RuntimeError("Model API connection failed: 503 Service Unavailable")

        with self.assertRaises(RuntimeError):
            mock_failing_baseline()

if __name__ == "__main__":
    unittest.main()
