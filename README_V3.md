# PCG-MAS v3.0

Replayable acceptance certificates for multi-agent LLMs.

**Acceptance is checker-relative external recomputability, not proof of world truth.**

```bash
python3.12 -m venv .venv-pcg-mas && source .venv-pcg-mas/bin/activate
python -m pip install -r env/constraints-offline.txt && python -m pip install -e .
bash scripts/v3/verify_offline.sh
```

| | |
|---|---|
| Start here | `DEVELOPMENT_NOTES.md` |
| Audit | `docs/v3/ARCHITECTURE_AUDIT_V3.md` |
| Spec conformance | `docs/v3/SCIENTIFIC_SPEC_ALIGNMENT_V3.md` |
| Run experiments | `docs/v3/EXPERIMENT_RUNBOOK_V3.md` |
| Demo | `app/APP_V3_ARCHITECTURE.md`, `app/DEPLOYMENT.md` |
| Artifact map | `docs/v3/MANUSCRIPT_ARTIFACT_MAP_V3.md` |

Four conjuncts `V_H · V_Π · V_Γ · V_⊢`; five audit channels `IntFail, ReplayFail, DriftFail, CheckFail, CovGap` — a **different layer**. Residuals `ε_tax` and `ε_src` lie outside every channel. Unknown counts as failure; `null` is never `0`.
