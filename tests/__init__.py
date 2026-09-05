"""Test package. Puts src/ and the artifact helper dirs on sys.path so the suite
runs from a clean checkout with `python -m unittest discover -s tests -t .`"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for p in [ROOT, ROOT / "src", ROOT / "scripts", ROOT / "scripts/repro", ROOT / "scripts/validate"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
