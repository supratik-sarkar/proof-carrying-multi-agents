# Baseline: AgentRR

- **Paper Title:** AgentRR: Record-and-Replay Verification for LLM Tool Trajectories
- **Citation/arXiv:** 2409.09876
- **Adapter Type:** author_vendored_replay_verifier
- **Umbrella Directory:** `baselines/agentrr/`

## Interface Contract
This baseline implements the standard PCG-MAS evaluation interface:
- Consumes shared benchmark inputs (retriever, generator, evidence pool, seeds, coverage target).
- Emits native decision (`ACCEPT` / `BLOCK` / `VERIFY` / `REFUSE`).
- Evaluates $H_{(\mathrm{support})}$, $H_{(\mathrm{exec})}$, and composite harm.
