# GIT_SYNC_DIFF_V3.md — PCG-MAS v3.0 Controlled Git Synchronization Report

**Date:** 2026-09-05
**Source Workspace:** local release directory (Authoritative v3.0 Gate-0.1 source)
**Destination Repository:** repository root
**Target Git Branch:** `release/pcg-mas-v3.0-gate0.1`
**Author:** Automated Release & Verification Suite

---

## 1. Summary of Changes

| Disposition Class | File Count | Description |
|---|---|---|
| **ADDED** | 765 | New v3.0 source, Gate-0.1 specs, app frontend/backend, telemetry, test suite, and runbooks |
| **MODIFIED** | 66 | Updated core files, pyproject.toml, requirements.txt, and .gitignore |
| **PRESERVED_GIT_ONLY** | 3 | Legitimate Git-only artifacts preserved (e.g., historical Colab notebooks, .git DB) |
| **REMOVED** | 0 | Zero unmanaged removals |
| **QUARANTINED** | 15 | Local-only forensics and secrets safely excluded from Git |

---

## 2. Quarantined Material (Safe Exclusions)

| Path / Pattern | Classification | Quarantine Rationale |
|---|---|---|
| `.env.secrets` | `INTERNAL_LOCAL_ONLY` / `SECRET` | Contains uncommitted local API keys — excluded by security policy |
| `.venv-pcg-mas` | `INTERNAL_LOCAL_ONLY` / `SECRET` | Local Python 3.12 virtual environment — runtime only |
| `WORKSPACE_DEFERRED_FIX_RESOLUTION.md` | `INTERNAL_LOCAL_ONLY` / `SECRET` | Internal workspace transition audit |
| `WORKSPACE_FINAL_VERIFICATION.md` | `INTERNAL_LOCAL_ONLY` / `SECRET` | Internal workspace transition audit |
| `WORKSPACE_SYNC_AUDIT.md` | `INTERNAL_LOCAL_ONLY` / `SECRET` | Internal workspace transition audit |
| `POST_SYNC_SHA256SUMS.txt` | `INTERNAL_LOCAL_ONLY` / `SECRET` | Internal workspace sync forensics |
| `POST_SYNC_TREE.txt` | `INTERNAL_LOCAL_ONLY` / `SECRET` | Internal workspace sync forensics |
| `PRE_SYNC_SHA256SUMS.txt` | `INTERNAL_LOCAL_ONLY` / `SECRET` | Internal workspace sync forensics |
| `PRE_SYNC_TREE.txt` | `INTERNAL_LOCAL_ONLY` / `SECRET` | Internal workspace sync forensics |
| `SYNC_DIFF_V3.md` | `INTERNAL_LOCAL_ONLY` / `SECRET` | Internal workspace transition report |
| `env` | `INTERNAL_LOCAL_ONLY` / `SECRET` | Local environment files |
| `reports` | `INTERNAL_LOCAL_ONLY` / `SECRET` | Internal development reports with local absolute paths |
| `runs` | `INTERNAL_LOCAL_ONLY` / `SECRET` | Local temporary run output directory |
| `scratch` | `INTERNAL_LOCAL_ONLY` / `SECRET` | Local scratch and temporary data directory |
| `v3-0.rtf` | `INTERNAL_LOCAL_ONLY` / `SECRET` | Internal non-git transition notes |

---

## 3. Preserved Git-Only Assets

| Path | Preservation Rationale |
|---|---|
| `.git/` | Underlying Git object database, branch references, and commit history |
| `notebooks/` | Pre-existing Google Colab exploration and demo notebooks |
| `results/` | Runtime experimental results directory boundary |

---

## 4. Key Added Components (Sample)

- **Gate-0.1 Scientific Specifications:** `GATE0_56_CELL_MATRIX.csv`, `GATE0_SAMPLE_SIZE_PROTOCOL.json`, `GATE0_PROVIDER_SEMANTICS.json`, `GATE0_SEEDS.json`, `GATE0_A8_DEPENDENCE_SPEC.json`, `GATE0_A17_CONTROLLER_SPEC.json`, `GATE0_BACKEND_MANIFEST.json`, `GATE0_CHECKER_PROTOCOL.json`, `GATE0_PRICING_SCHEMA.json`, `GATE0_RESOURCE_ACCOUNTING.json`, `GATE0_API_FIRST_FREEZE.md`, `GATE0_MANUSCRIPT_ALIGNMENT.md`.
- **v3.0 Scientific Core (`src/pcg/v3/`):** Release constants, record schemas, Clopper-Pearson/Hoeffding statistics, $S+V=\Delta$ harm decomposition, A8 dependence gate, A17 controller regret/sensitivity, Otel telemetry, LangGraph orchestration, OPA/NeMo policy adapters, 7 provider adapters, Table/Figure artifact generators.
- **Full Test Suite (`tests/`):** 13 test modules covering 170 unit, property, theory, and provenance tests.
- **Full Stack App (`app/`):** FastAPI backend, Vanilla JS frontend, shared contract, Cloudflare/Render deployment specs.
- **Documentation & Runbooks (`docs/v3/`):** Mac M4 and Colab runbooks, architecture audits, artifact maps.

---

## 5. Verification Precondition

All synced files comply with double-blind anonymity and security policies. Zero secrets or local user paths committed.
