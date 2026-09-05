# DEPENDENCIES.md — traced, not inherited

Sets were derived by static AST tracing of actual imports across `src/pcg/`, `scripts/repro/`,
`scripts/validate/`, `scripts/figures/` and `tests/`, then verified by import isolation. They were
**not** inferred from the previous `requirements.txt`.

## Finding

Every third-party import in `src/pcg` other than **numpy** is either lazy (inside a function) or
confined to a plotting module:

| package | where | kind |
|---|---|---|
| numpy | risk, responsibility, eval/audit, eval/stats, eval/tightness | top-level → **base** |
| scipy | eval/rho, eval/stats, responsibility | lazy, but required at call time → **base** |
| matplotlib | eval/plots_v2, eval/intro_hero_v4, scripts/figures | top-level in plotting only → **repro** |
| typer | cli.py line 22 | lazy (inside function) → `models` |
| tiktoken | eval/meter.py line 31 | lazy (inside function) → `models` |
| torch, transformers, datasets, sentence-transformers, langgraph, sklearn, jsonschema, openai, huggingface_hub | backends, orchestrator, retrieval | lazy → `models` / `frontier` |
| torchvision | **nowhere in code** | removed entirely |

## Sets

```
CORE               numpy, scipy
OFFLINE_REPRODUCTION   + matplotlib                       -> .[repro]
TESTING            numpy, scipy, pytest (+ matplotlib)    -> .[dev]
MODEL_EXECUTION    torch, transformers, …                 -> .[models]
PRIVACY            opacus                                 -> .[privacy]
NOTEBOOKS          ipykernel, jupyterlab, seaborn, colorcet -> .[notebooks]
```

## Verification performed

- **BASE** (numpy, scipy; everything else blocked by a meta-path finder): 13/13 core modules import.
  `pcg.cli`, `pcg.eval.meter`, `pcg.orchestrator.langgraph_flow`, `pcg.backends.hf_local`,
  `pcg.retrieval` all import cleanly and fail only when an optional path is invoked.
  `pcg.eval.plots_v2` requires matplotlib at import time — documented, install `[repro]`.
- **REPRO** (+ matplotlib): `validate_56cell.py`, `provenance_gates.py`, `regenerate_tables.py`,
  `regenerate_figures.py` all PASS.
- **DEV**: **not executed** — `pytest` is unavailable in the build environment and network
  installation is prohibited there. See FINAL_VALIDATION_REPORT.md.
