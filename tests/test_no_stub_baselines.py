import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINES_DIR = REPO_ROOT / "baselines"


class TestNoStubBaselines(unittest.TestCase):
    def test_no_synthetic_sin_seed_stubs_in_runners(self):
        """Ensure no 0.45 + 0.1*sin(seed) mock generators exist in active runners."""
        for runner_path in REPO_ROOT.glob("baselines/**/runner.py"):
            text = runner_path.read_text(encoding="utf-8")
            self.assertNotIn("sin(seed)", text, f"Synthetic stub found in {runner_path}")
            self.assertNotIn("sin(idx)", text, f"Synthetic stub found in {runner_path}")

    def test_all_baselines_registered_under_umbrella(self):
        """Ensure all 8 baselines sit under baselines/ umbrella directory."""
        required = [
            "shieldagent", "verimap", "agentrr", "atlasprism",
            "pcnrec", "clbc", "citation_only", "no_certificate"
        ]
        for name in required:
            b_dir = BASELINES_DIR / name
            self.assertTrue(b_dir.exists(), f"Missing baseline directory: {b_dir}")
            self.assertTrue((b_dir / "runner.py").exists(), f"Missing runner.py in {b_dir}")


if __name__ == "__main__":
    unittest.main()
