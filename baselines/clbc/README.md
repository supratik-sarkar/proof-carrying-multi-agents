# Baseline: CLBC

- **Paper Title:** CLBC: Certified Bounds on Covert Signaling in Colluding Agent Systems
- **Citation/arXiv:** 2603.00381
- **Adapter Type:** independent_task_adapter
- **Umbrella Directory:** `baselines/clbc/`

## Interface Contract
This baseline implements the standard PCG-MAS evaluation interface:
- Consumes shared benchmark inputs (retriever, generator, evidence pool, seeds, coverage target).
- Emits native decision (`ACCEPT` / `BLOCK` / `VERIFY` / `REFUSE`).
- Evaluates $H_{(\mathrm{support})}$, $H_{(\mathrm{exec})}$, and composite harm.
