.PHONY: help install smoke smoke-full test r1 r2 r3 r4 r5 figures per-exp tables paper readme manifest pick-top-k clean

help:
	@echo "PCG-MAS Makefile targets:"
	@echo "  install      pip install -e .[dev]"
	@echo "  smoke        fast pipeline sanity (~30s, no model download)"
	@echo "  smoke-full   end-to-end mock run of R1..R5 + figures + tables + manifest (~3-5 min)"
	@echo "  test         pytest -v on the property tests"
	@echo "  r1..r5       run experiments (use BACKEND=mock|hf_local|hf_inference)"
	@echo "  figures      regenerate figures from results/"
	@echo "  per-exp      regenerate per-experiment R1..R5 plots with diverse-coverage cells"
	@echo "  tables       regenerate LaTeX tables from results/"
	@echo "  manifest     build backends_manifest.{json,md,tex} for reviewer disclosure"
	@echo "  pick-top-k   AFTER real Colab runs: pick top 3 (LLM, dataset) cells per R-plot"
	@echo "  paper        figures + tables + intro hero + per-exp + manifest (one command)"
	@echo "  readme       regenerate README.md with live file stats"
	@echo "  clean        remove caches and build artifacts (KEEPS results/ and figures/)"

# Default backend - override on the command line: `make r1 BACKEND=hf_local`
BACKEND ?= mock
SEEDS ?= 0 1 2 3 4
SMOKE_N ?= 20

install:
	pip install -e ".[dev]"

smoke:
	python scripts/test_phase_bc.py
	python scripts/test_phase_d.py
	python scripts/run_r1_checkability.py --config configs/smoke.yaml --seeds 0 --n-examples 20 --backend mock

# End-to-end smoke: runs all 5 experiments with the deterministic mock
# backend on $(SMOKE_N) examples per seed, then regenerates every figure,
# every table, and the backend-disclosure manifest. Total wall time on
# M4 Pro: ~3-5 minutes. No GPU, no network, no model downloads.
#
# This is the target reviewers can run to verify the pipeline reproduces
# end-to-end without any infrastructure.
smoke-full:
	@echo ">>> [1/8] R1: audit decomposition (mock, $(SMOKE_N) examples)"
	python scripts/run_r1_checkability.py     --config configs/r1_hotpotqa.yaml      --seeds 0 1 2 --n-examples $(SMOKE_N) --backend mock
	@echo ">>> [2/8] R2: redundancy law (mock, $(SMOKE_N) examples)"
	python scripts/run_r2_redundancy.py       --config configs/r2_redundancy.yaml    --seeds 0 1 2 --n-examples $(SMOKE_N) --backend mock
	@echo ">>> [3/8] R3: responsibility (mock, $(SMOKE_N) examples)"
	python scripts/run_r3_responsibility.py   --config configs/r3_responsibility.yaml --seeds 0 1 2 --n-examples $(SMOKE_N) --backend mock
	@echo ">>> [4/8] R4: cost vs harm (mock, $(SMOKE_N) examples)"
	python scripts/run_r4_risk_privacy.py     --config configs/r4_risk.yaml          --seeds 0 1 2 --n-examples $(SMOKE_N) --backend mock
	@echo ">>> [5/8] R5: overhead (mock, $(SMOKE_N) examples)"
	python scripts/run_r5_overhead.py         --config configs/r5_overhead.yaml      --seeds 0 1 2 --n-examples $(SMOKE_N) --backend mock
	@echo ">>> [6/9] Per-experiment plots (diverse-coverage R1..R5)"
	python scripts/make_per_experiment_plots.py
	@echo ">>> [7/9] Figures (legacy + summary + intro hero v3/v4)"
	python scripts/make_paper_artifacts.py
	@echo ">>> [8/9] Tables (LaTeX)"
	python scripts/make_tables.py
	@echo ">>> [9/9] Backend disclosure manifest"
	python scripts/build_backends_manifest.py
	@echo ""
	@echo "✓ smoke-full complete. Outputs:"
	@echo "    figures/          PDF + PNG of intro_hero, summary_benchmark, R1..R5"
	@echo "    tables/           LaTeX .tex files for the paper"
	@echo "    artifacts/        backends_manifest.{json,md,tex} (REVIEWER DISCLOSURE)"
	@echo "    results/          per-experiment run JSONs (raw + aggregated)"

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

manifest:
	python scripts/build_backends_manifest.py

per-exp:
	python scripts/make_per_experiment_plots.py

pick-top-k:
	python scripts/pick_top_k.py

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
