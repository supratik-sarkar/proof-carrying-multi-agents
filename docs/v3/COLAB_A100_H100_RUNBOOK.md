# COLAB_A100_H100_RUNBOOK.md — Google Colab Pro (A100 / H100)

Colab is the **second** execution profile, never the only one. Core data, metric and auditor code must behave identically here and on the Mac wherever hardware is irrelevant — that identity is exactly what A04 measures.

## Bootstrap (non-interactive cells)

```python
# 1 — obtain the tree
from google.colab import drive; drive.mount('/content/drive')
!cp -a "/content/drive/MyDrive/pcg-mas-2026" /content/ && cd /content/pcg-mas-2026 && ls

# 2 — environment
%cd /content/pcg-mas-2026
!python -V
!pip -q install -r env/constraints-offline.txt
!pip -q install -e .

# 3 — record the environment BEFORE anything else
!python app/shared/generate_contract.py
!bash scripts/v3/verify_offline.sh
!nvidia-smi
```

Colab's interpreter may not be 3.12.13. Record whatever it actually is in `environment.json` and treat any divergence from the Mac profile as a **reported difference**, not something to paper over — A04 exists to detect exactly this.

## Device routing

```
CUDA → CPU
```

`torch.cuda.is_available()` is checked at runtime; the resolved device, driver and CUDA version enter `environment.json`.

## Persistence and restart tolerance

* Write artifacts to `/content/drive/MyDrive/pcg-artifacts/v3_0/<aNN>/`, never to ephemeral `/content` alone; set `PCG_ARTIFACT_ROOT` accordingly.
* Runners are idempotent and keyed by `record_id`: after a restart, re-issue the identical command and completed records are skipped.
* Checkpoint after each cell, not at the end of the matrix.
* If the frozen `spec.json` hash does not match, the runner **fails**. Do not "fix" it by re-freezing.

## What belongs here

A05, A08, A09, A15-B/C and A16 in full; the 70B and 671B cells of A01; the Colab half of A04; the Colab half of A10.

## DeepSeek-V3

If full open-weight execution is infeasible on one accelerator, use a **version-pinned hosted route**. The route, model revision, decoding config and billed cost enter the backend fingerprint and the per-record cost fields, so A10 aggregates telemetry rather than re-running anything.

## Cost discipline

Every provider call is fingerprinted and counted. `pricing_manifest.json` records the provider price snapshot and its date; cost is computed from actual token and tool usage, never estimated after the fact.

## What does not belong here

Anything that only reads records. Running A02/A03/A07/A11 on a GPU wastes budget and adds a second environment to the provenance for no benefit.
