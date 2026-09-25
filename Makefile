.PHONY: install test preflight preflight40 full merge-frontier figures tables audit clean verify-hashes rebuild-metrics rebuild-artifacts

install:
	pip install -e .

test:
	python -m pytest -q

merge-frontier:
	python scripts/notebooks/merge_frontier_runs.py --allow-fallback

clean:
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	find . -name ".DS_Store" -type f -delete
	find . -name "*.pyc" -type f -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache

check-datasets:
	python scripts/runs/check_datasets.py --n 5 --seed 0 --allow-fallback

preflight:
	python scripts/runs/run_preflight.py --n 10 --seeds 0

preflight40:
	python scripts/runs/run_preflight_40_cells.py --n 5 --seeds 0

matrix-dry-run:
	python scripts/runs/run_matrix.py --dry-run --n-examples 3 --seeds 0 --datasets hotpotqa --models phi-3.5-mini --experiments r1 --backend mock

full:
	python scripts/runs/run_local_40_cells.py --n 500 --seeds 0 1 2 3 --allow-full-run --backend hf_inference

figures:
	python scripts/figures/build_all_figures.py

tables:
	python scripts/tables/build_all_tables.py

verify-hashes:
	sh reproducibility/v3_7/16_replay/verify_hashes.sh

rebuild-metrics:
	sh reproducibility/v3_7/16_replay/rebuild_metrics.sh

rebuild-artifacts:
	sh reproducibility/v3_7/16_replay/rebuild_artifacts.sh

audit:
	python scripts/maintain/audit_forbidden_terms.py
	python scripts/maintain/audit_repo_layout.py
	python scripts/maintain/audit_secrets.py
	python -m pytest -q