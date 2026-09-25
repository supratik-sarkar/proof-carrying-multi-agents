"""Regenerate manuscript artifacts in exact dependency order.

Pipeline 1 (Figure 06 Backend Invariance Atlas):
  derive_backend_invariance_atlas.py -> figure_06_backend_invariance_atlas.py

Pipeline 2 (Table 34 Verifiability and Execution Ledger):
  derive_verifiability_execution_ledger.py -> generate_tables.py
"""
from pathlib import Path
import subprocess, sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'source'

def main():
    # Pipeline 1: derive_backend_invariance_atlas.py -> figure_06_backend_invariance_atlas.py
    subprocess.run([sys.executable, str(SOURCE / 'derive_backend_invariance_atlas.py')], check=True)
    subprocess.run([sys.executable, str(SOURCE / 'figure_06_backend_invariance_atlas.py')], check=True)

    # Pipeline 2: derive_verifiability_execution_ledger.py -> generate_tables.py
    subprocess.run([sys.executable, str(SOURCE / 'derive_verifiability_execution_ledger.py')], check=True)
    subprocess.run([sys.executable, str(SOURCE / 'generate_tables.py')], check=True)

if __name__ == '__main__':
    main()
