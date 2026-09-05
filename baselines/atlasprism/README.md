# Baseline: PRISM/ATLAS

- **Paper Title:** PRISM/ATLAS: Constraint-Guided Structured Artifact Generation for Model-Driven Engineering
- **Citation/arXiv:** 2510.25890
- **Adapter Type:** independent_task_adapter
- **Umbrella Directory:** `baselines/atlasprism/`

## Interface Contract
This baseline implements the standard PCG-MAS evaluation interface:
- Consumes shared benchmark inputs (retriever, generator, evidence pool, seeds, coverage target).
- Emits native decision (`ACCEPT` / `BLOCK` / `VERIFY` / `REFUSE`).
- Evaluates $H_{(\mathrm{support})}$, $H_{(\mathrm{exec})}$, and composite harm.
