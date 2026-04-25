.PHONY: help install smoke test r1 r2 r3 r4 r5 figures tables paper readme clean

help:
	@echo "PCG-MAS Makefile targets:"
	@echo "  install     pip install -e .[dev]"
	@echo "  smoke       fast end-to-end sanity (no model download)"
	@echo "  test        pytest -v on the property tests"
	@echo "  r1..r5      run experiments R1..R5 (use BACKEND=mock|hf_local|hf_inference)"
	@echo "  figures     regenerate figures from results/"
	@echo "  tables      regenerate LaTeX tables from results/"
	@echo "  paper       figures + tables + intro hero (one command)"
	@echo "  readme      regenerate README.md with live file stats"
	@echo "  clean       remove caches and build artifacts (KEEPS results/ and figures/)"

# Default backend - override on the command line: `make r1 BACKEND=hf_local`
BACKEND ?= mock
SEEDS ?= 0 1 2 3 4

install:
	pip install -e ".[dev]"

smoke:
	python scripts/test_phase_bc.py
	python scripts/test_phase_d.py
	python scripts/run_r1_checkability.py --config configs/smoke.yaml --seeds 0 --n-examples 20 --backend mock

test:
	pytest tests/ -v

r1:
	python scripts/run_r1_checkability.py --config configs/r1_hotpotqa.yaml --seeds $(SEEDS) --backend $(BACKEND)

r2:
	python scripts/run_r2_redundancy.py --config configs/r2_redundancy.yaml --seeds $(SEEDS) --backend $(BACKEND)

r3:
	python scripts/run_r3_responsibility.py --config configs/r3_responsibility.yaml --seeds $(SEEDS) --backend $(BACKEND)

r4:
	python scripts/run_r4_risk_privacy.py --config configs/r4_risk.yaml --seeds $(SEEDS) --backend $(BACKEND)

r5:
	python scripts/run_r5_overhead.py --config configs/r5_overhead.yaml --seeds $(SEEDS) --backend $(BACKEND)

figures:
	python scripts/make_figures.py

tables:
	python scripts/make_tables.py

paper:
	python scripts/make_paper_artifacts.py

readme:
	python scripts/build_readme.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/
