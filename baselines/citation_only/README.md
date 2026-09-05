# Baseline: Citation-Only

- **Paper Title:** Scope-Matched Control: Citation Validity & Entailment Verification
- **Citation/arXiv:** n/a
- **Adapter Type:** scope_matched_control
- **Umbrella Directory:** `baselines/citation_only/`

## Interface Contract
This baseline implements the standard PCG-MAS evaluation interface:
- Consumes shared benchmark inputs (retriever, generator, evidence pool, seeds, coverage target).
- Emits native decision (`ACCEPT` / `BLOCK` / `VERIFY` / `REFUSE`).
- Evaluates $H_{(\mathrm{support})}$, $H_{(\mathrm{exec})}$, and composite harm.
