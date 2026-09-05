# REPRODUCIBILITY_CONTRACT_V3.md

Every number that reaches the manuscript is reconstructible along exactly this chain, with **no manual numerical hop**:

```
manuscript claim
  → generated TeX / figure        artifacts/v3_0/tables/latex, figures/{png,pdf}
  → metric / statistics artifact  artifacts/v3_0/<aNN>/metrics.json
  → canonical metric code         src/pcg/v3/science/*, stats/*
  → raw paired per-example records  records.jsonl  (schema v3.0.0, 87 fields)
  → frozen experiment spec        spec.json + spec_sha256
  → model / checker / backend / environment fingerprint
```

The dependency direction is never reversed. The app may consume copies or API views of artifacts but is never the authoritative store.

## Canonical record

One versioned typed schema: `schemas/per_example_record.v3.schema.json`. **No manuscript metric may read from any other schema.**

**Nullability rule.** `None` means NOT MEASURED. It propagates as undefined and is never coerced to `0`. A metric whose denominator is `None` or `0` returns `None`. Verified by `test_missing_probes_yield_undefined_not_zero`.

## Statistics

* One paired **crossed seed × example** bootstrap: resample seed IDs, independently resample example IDs, preserve system pairing.
* `S + V = Δ` asserted on the realised data **and every resample** (`1e-12`).
* Every generated result row carries: `estimate, CI_low, CI_high, N, N_acc, numerator, denominator, seed_count, inference_scheme, metric_version, source_hash`.
* ≥4 seeds for any direct model comparison.
* Effect sizes and CIs are primary; p-values only for a pre-registered hypothesis family with a declared multiplicity correction.

## Frozen specs

Each workstream carries a `Spec` whose hash covers name, tier, provenance class, seeds, metric version and all parameters. `verify_spec()` raises on drift. Re-freezing after observing results is prohibited.

## Cost telemetry

Instrumented now so future runs cannot forget it: wall time, phase timings, input/output tokens, model/retrieval/tool/checker/replay calls, cache state, provider, pricing manifest, billed cost. **A10 aggregates telemetry; it never requires re-running experiments.**

## Failure philosophy — fail closed

Never silently: manufacture zero for an undefined denominator; drop a failed example; skip a malformed record; substitute a different model or dataset; weaken a policy; disable a checker; reuse a cached value with different provenance; convert a missing metric into `NaN` and continue. Emit a structured failure artifact instead.

## Build gates

One canonical metric implementation · no hand-entered result cells (every generated `.tex` carries a `GENERATED FILE` header) · cross-table equality checks · `DIRECT/DERIVED/MODELLED/PROTOCOL/STATIC` on every artifact · vector PDF with embedded fonts and extractable text plus 300 dpi PNG · figure source/script/config/output hashes · post-build hidden-text scan · secret scan.
