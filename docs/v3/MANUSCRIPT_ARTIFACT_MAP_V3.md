# MANUSCRIPT_ARTIFACT_MAP_V3.md

Machine-readable source of truth: `manuscript_artifact_registry.json`. Verified against the manuscript by `pcg.v3.artifacts.registry.check_against_tex` — table labels, figure labels and figure paths all match, including the four `subtable` labels attached to their parent tables.

## Tables — 33 registered

| Class | Count |
|---|---|
| DIRECT | 14 |
| DERIVED | 11 |
| PROTOCOL | 6 |
| MODELLED | 2 |
| STATIC | 1 (`tab:notation`) |

32 have generators; `tab:notation` is hand-authored and has no data source.

### Submission-critical

| # | Label | Class | Source | Generator |
|---|---|---|---|---|
| 1 | `tab:main_six_summary` | DIRECT | A02, A05 | `tables.t1` |
| 2 | `tab:cost_overhead_main` | DIRECT | A10 | `tables.t2` |
| 3 | `tab:audit_calibration_summary` | DIRECT | A15 | `tables.t3` |
| 31 | `tab:validation_budget_frontier` | DIRECT | A16, A10 | `tables.t31` |

Subtables `tab:app_a`–`tab:app_d` are registered against their parents `tab:appendix_remaining_50_summary_1/2`.

## Figures — 10 registered

| # | Label | Manuscript path | Class | Source |
|---|---|---|---|---|
| 1 | `fig:intro_overview` | `images/intro_overview.pdf` | DERIVED | A02, A05, A10 |
| 2 | `fig:workflow` | `images/pcg_mas_workflow.pdf` | STATIC | hand-drawn schematic |
| 3 | `fig:r1_to_r4_combined` | `images/headline_budget_frontier.pdf` | DIRECT | A16, A10 |
| 4 | `fig:r5-overhead` | `images/cost_overhead.pdf` | DIRECT | A10 |
| 5 | `fig:ablations` | `images/ablations.pdf` | DIRECT | A05 |
| 6 | `fig:baseline_comparison` | `images/baseline_comparison.pdf` | DIRECT | A05, A16 |
| 7 | `fig:r1_drift` | `images/audit_channels.pdf` | DERIVED | A07, A11 |
| 8 | `fig:r3_open` | `images/attribution_open_set.pdf` | DIRECT | A14 |
| 9 | `fig:r4_privacy` | `images/privacy_frontier_modelled.pdf` | MODELLED | A18 |
| 10 | `fig:r5_scaling` | `images/scaling_modelled.pdf` | MODELLED | A10 |

Figure 3 is the A16+A10 budget/harm frontier; Figure 4 is DIRECT cost telemetry; Figures 9–10 remain MODELLED.

## Release gate (verified)

```
PNG_PDF_DUAL_OUTPUT=PASS   9/9 generated figures
  vector (0 raster images) · fonts fully embedded · extractable text 164-348 chars · PNG 300 dpi
```

A zero-text figure is rejected rather than raster-edited.

## Publishing into the manuscript

Generated `.tex` lands in `artifacts/v3_0/tables/latex/<stem>.tex` and is `\input`-ed; generated PDFs are copied to the `images/` names above. **No manuscript result cell is ever typed by hand** — every generated file carries a `GENERATED FILE -- do not edit by hand` header, and unmeasured values render `\PEND{}`, never `0`.
