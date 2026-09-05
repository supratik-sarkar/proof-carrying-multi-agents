# Baseline: VERIMAP

- **Paper Title:** VeriMAP: Verified Model-Driven Contract Verification for Multi-Agent Systems
- **Citation/arXiv:** 2408.05678
- **Adapter Type:** author_vendored_contract_verifier
- **Umbrella Directory:** `baselines/verimap/`

## Interface Contract
This baseline implements the standard PCG-MAS evaluation interface:
- Consumes shared benchmark inputs (retriever, generator, evidence pool, seeds, coverage target).
- Emits native decision (`ACCEPT` / `BLOCK` / `VERIFY` / `REFUSE`).
- Evaluates $H_{(\mathrm{support})}$, $H_{(\mathrm{exec})}$, and composite harm.
