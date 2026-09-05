# Baseline: PCN-Rec

- **Paper Title:** PCN-Rec: Proof-Carrying Negotiation for Recommender Systems
- **Citation/arXiv:** 2601.09771
- **Adapter Type:** independent_task_adapter
- **Umbrella Directory:** `baselines/pcnrec/`

## Interface Contract
This baseline implements the standard PCG-MAS evaluation interface:
- Consumes shared benchmark inputs (retriever, generator, evidence pool, seeds, coverage target).
- Emits native decision (`ACCEPT` / `BLOCK` / `VERIFY` / `REFUSE`).
- Evaluates $H_{(\mathrm{support})}$, $H_{(\mathrm{exec})}$, and composite harm.
