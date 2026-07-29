# Synthetic Placeholder Table Generation Pipeline

provenance: SYNTHETIC_PLACEHOLDER  
empirical_status: NOT_EXECUTED  
formal_reporting_allowed: false  
server_run_status: PENDING  

## Overview
This directory contains project-owned scripts for generating deterministic synthetic placeholder tables across all 56 cells (7 models x 8 datasets) for internal review with the professor prior to the real server execution.

## Quick Start:

Run the complete pipeline via:
```bash
.venvs/multi-agents/bin/python scripts/rebuttal/synthetic_tables/generate_all.py
```

## Structure:
- `generate_records.py`: Generates 28,000 paired per-example synthetic records.
- `canonical_metrics.py`: Defines single canonical metric functions used across all tables.
- `render_manuscript_tables.py`: Refreshes Tables 1-18 in `results/tables/`.
- `render_rebuttal_tables.py`: Renders rebuttal tables in `artifacts/rebuttal/<dir>/synthetic_placeholder/`.
- `validate_all_tables.py`: Runs consistency validation and negative tests.
- `synthetic_config.yaml`: Frozen seed (`20260729`) and configuration matrix.
