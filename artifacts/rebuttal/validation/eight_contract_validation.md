# Eight Rebuttal Contract Validation Report

## Overall Status: PASS

| Directory | Promised Contract | Validation Check | Status |
|---|---|---|---|
| `table_reconciliation/` | Table 2/16 reconciliation & canonical metrics | Exact numerator/denominator & cell audit variance | **PASS** |
| `sv_decomposition/` | Paired S/V harm avoidance decomposition | Literal paired summation & identity residual < 1e-12 | **PASS** |
| `separating_witnesses/` | 4 single-channel failure witness certificates | Exactly 1 failed channel per witness family | **PASS** |
| `citation_only/` | Matched-coverage comparative metrics | 5 systems evaluated on identical example IDs | **PASS** |
| `injection/` | Attack sweep under isolated/shared regimes | 4 attack locations & k-sweep redundancy | **PASS** |
| `shift/` | 6 shift families & fail-closed UCB gate | Realised bad-accept vs 2a-1 TV bound | **PASS** |
| `audit_sampling/` | Stratified audit sampling & variance bound | 4 sampling designs & uncovered mass bounds | **PASS** |
| `backend_manifest/` | 10-field hardware/decoding route fingerprints | 35 complete backend route records | **PASS** |
