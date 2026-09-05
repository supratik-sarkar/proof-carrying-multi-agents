# Baseline: No Certificate

- **Paper Title:** Matched-Stack Control: Uncertified Agent Execution
- **Citation/arXiv:** n/a
- **Adapter Type:** matched_stack_control
- **Umbrella Directory:** `baselines/no_certificate/`

## Interface Contract
This baseline implements the standard PCG-MAS evaluation interface:
- Consumes shared benchmark inputs (retriever, generator, evidence pool, seeds, coverage target).
- Emits native decision (`ACCEPT` / `BLOCK` / `VERIFY` / `REFUSE`).
- Evaluates $H_{(\mathrm{support})}$, $H_{(\mathrm{exec})}$, and composite harm.
