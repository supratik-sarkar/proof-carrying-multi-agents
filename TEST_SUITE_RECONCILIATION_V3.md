# TEST_SUITE_RECONCILIATION_V3.md — PCG-MAS v3.0 Test Suite Reconciliation Report

**Date:** 2026-09-05
**Pass:** Gate-0.1 Controlled Git Integration
**Target Workspace:** `.` (PCG-MAS v3.0 Repository Root)
**Environment:** `.venv-pcg-mas` (`Python 3.12.13`)

---

## 1. Executive Summary: Resolving the Test Count Discrepancy

During Phase-2 / Gate-0 verification, two distinct test count metrics were reported:
1. Full test runner reporting: `UNIT_TESTS=168/168`, `PROPERTY_TESTS=39/39` (pre-Gate-0.1).
2. Lightweight offline smoke script (`scripts/v3/verify_offline.py`): `UNIT_TESTS=41/41`, `PROPERTY_TESTS=41/41` (post-Gate-0.1).

### Root Cause Analysis

- **Full Pytest Discovery (`pytest tests/`):**
  Discovers all 13 test modules in `tests/`, spanning core theory, provenance invariants, checker channels, and v3 core functions. Prior to Gate-0.1, the total count was 168. With the addition of two Gate-0.1 scientific test cases (`test_precision_requirement_fails_closed_when_inadequate` and `test_a17_substream_theorem_separation`), the complete pytest suite collects and passes exactly **170 test items**.

- **Lightweight Runner in `scripts/v3/verify_offline.py`:**
  The `verify_offline.py` script executes a minimal, dependency-free in-memory module loader (`_run_tests`) targeted strictly at `tests/v3/test_v3_core.py` to provide rapid verification of core mathematical and contract invariants without invoking external test harnesses. That specific module contains 41 test functions (previously 39, now 41).

Both counts are completely accurate within their respective execution scopes. No tests were dropped, omitted, or mislabeled.

---

## 2. Complete Test Suite Inventory (170 Collected Test Nodes)

| Test Module Path | Test Focus / Scope | Test Node Count | Status |
|---|---|---|---|
| `tests/test_architecture.py` | Architecture boundary invariants, layer import separation | 4 | PASSED |
| `tests/test_audit_envelope_v4.py` | Audit envelope schema, channel recording validation | 4 | PASSED |
| `tests/test_auditor_invariance.py` | Auditor permutation invariance | 1 | PASSED |
| `tests/test_baselines_umbrella.py` | Baseline runner invariants and umbrella isolation | 2 | PASSED |
| `tests/test_checker_v4_channels.py` | Checker v4 five-channel evaluation logic | 5 | PASSED |
| `tests/test_full_suite.py` | Comprehensive multi-agent integration pipeline | 26 | PASSED |
| `tests/test_no_stub_baselines.py` | Strict enforcement: zero mock stubs in production paths | 2 | PASSED |
| `tests/test_provenance_invariants.py`| Provenance schema fields, hash determinism, record integrity | 7 | PASSED |
| `tests/test_provenance_rc2.py` | Numeric safety, backend eligibility, gates, lineage | 43 | PASSED |
| `tests/test_theory/test_check.py` | Merkle prefixes, certificate checking, v4 contracts | 16 | PASSED |
| `tests/test_theory/test_independence.py` | Shingle Jaccard, branch independence, rho UCB bounds | 12 | PASSED |
| `tests/test_theory/test_risk.py` | Threshold policies, Hoeffding half-widths, rank recovery | 7 | PASSED |
| `tests/v3/test_v3_core.py` | Core mathematical invariants, $S+V=\Delta$, A8 precision, A17 separation | 41 | PASSED |
| **TOTAL** | **Comprehensive Full Repository Test Suite** | **170** | **170/170 PASSED** |

---

## 3. Test Classification Breakdown

- **Mathematical & Risk Theory Tests (53):**
  - Clopper-Pearson, Hoeffding, Beta incomplete functions, $S+V=\Delta$ exact decomposition, union slack, $\pi_{\text{unc}}$ single-charge invariant, A8 dependence gate, $\rho$ bounding, exact-$k$ $U_{\text{joint}}(k, \delta)$, controller threshold reachability, analytic calibration bounds ($R_{\text{max}}^{\text{cal}} \le 2 L_{\text{ctrl}} \varepsilon_{\text{cal}}$), per-$\theta$ oracle sensitivity, shift alarm lower bounds.
- **Provenance, Lineage & Gate Tests (57):**
  - Lineage tracking, direct backend eligibility, aggregate derivation, tamper detection, secret leakage in artifacts, payload integrity, numeric safety (fail-closed, null propagation).
- **Checker & Orchestration Channel Tests (25):**
  - Four conjuncts ($V_H \cdot V_\Pi \cdot V_\Gamma \cdot V_\vdash$), five audit channels, Merkle prefix verification, state graph deterministic execution.
- **Architecture & System Integration Tests (35):**
  - Architecture isolation, baseline integrity, full-suite offline execution, artifact registry and schema export shapes.

---

## 4. Verification Conclusion

The complete offline test suite passes 100% on the project environment (`Python 3.12.13`):
- Full test suite: **170 / 170 passed** in 0.82s.
- `test_v3_core.py`: **41 / 41 passed** in 0.64s.
- All property tests, unit tests, and integration tests are strictly verified.
