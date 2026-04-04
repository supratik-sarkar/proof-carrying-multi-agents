# Rebuttal Runs

## Local Mac
Environment:
- Python 3.12
- venv under .venv

Main commands used:
- PYTHONPATH=. python scripts/run_eval.py --config configs/medmcqa.yaml
- PYTHONPATH=. python scripts/run_healthcare.py
- PYTHONPATH=. python scripts/run_overhead.py
- PYTHONPATH=. python scripts/aggregate_overhead.py

Outputs:
- outputs/runs/*.jsonl
- outputs/tables/*.csv
- outputs/figures/*.png
