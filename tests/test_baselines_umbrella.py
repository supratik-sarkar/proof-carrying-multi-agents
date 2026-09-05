import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINES_DIR = REPO_ROOT / "baselines"


class TestBaselinesUmbrella(unittest.TestCase):
    def test_all_baselines_exist_and_executable(self):
        expected_baselines = [
            "shieldagent", "verimap", "agentrr", "atlasprism",
            "pcnrec", "clbc", "citation_only", "no_certificate"
        ]

        for b_name in expected_baselines:
            b_dir = BASELINES_DIR / b_name
            self.assertTrue(b_dir.exists(), f"Missing baseline directory: {b_dir}")
            self.assertTrue((b_dir / "runner.py").exists(), f"Missing runner in {b_dir}")
            self.assertTrue((b_dir / "config.yaml").exists(), f"Missing config in {b_dir}")
            self.assertTrue((b_dir / "README.md").exists(), f"Missing README in {b_dir}")

    def test_baseline_runner_execution(self):
        sys.path.insert(0, str(BASELINES_DIR / "shieldagent"))
        from runner import run_baseline
        res = run_baseline([{"id": "test_1", "text": "sample"}])
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["baseline"], "shieldagent")
        self.assertIn("composite_harm", res[0])


if __name__ == "__main__":
    unittest.main()
