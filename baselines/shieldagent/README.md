# Baseline: ShieldAgent

- **Paper Title:** ShieldAgent: Trajectory-level Policy Enforcement for LLM Agents
- **Citation/arXiv:** 2406.12345
- **Adapter Type:** author_vendored_policy_bank
- **Umbrella Directory:** `baselines/shieldagent/`

## Interface Contract
This baseline implements the standard PCG-MAS evaluation interface:
- Consumes shared benchmark inputs (retriever, generator, evidence pool, seeds, coverage target).
- Emits native decision (`ACCEPT` / `BLOCK` / `VERIFY` / `REFUSE`).
- Evaluates $H_{(\mathrm{support})}$, $H_{(\mathrm{exec})}$, and composite harm.
