# Proof-Carrying Graph Multi-Agent Systems (PCG-MAS)

Official repository for **Proof-Carrying Multi-Agent Systems** (Submission 9327).

---

## 📌 Artifact Compliance Status

```text
ARTIFACT_INTEGRITY = PASS
MATHEMATICAL_RECONCILIATION = PASS
EXECUTED_PROTOCOL_VALIDATION = PASS
SUBMITTED_SEED_COVERAGE = PASS
SUBMITTED_SAMPLE_CAP = PASS
EXACT_SUBMITTED_SEED_SET_REPRODUCED = false
POST_REVIEW_SEED_EXPANSION_DISCLOSED = true
NATIVE_PROVENANCE_GLOBAL_RECOMPUTATION = PASS (40,320 hashes verified)
EIGHT_REBUTTAL_CONTRACTS = PASS
CLEAN_ROOM_REPRODUCTION = PASS
OVERALL_STATUS = COMPLIANT_WITH_DISCLOSED_SEED_EXPANSION
```

---

## 🚀 Paper Overview & Core Contribution

**Proof-Carrying Graph Multi-Agent Systems (PCG-MAS)** introduces cryptographic and logic-certifiable verification boundaries for autonomous multi-agent systems. By requiring agents to attach machine-checkable proof certificates ($\mathcal{V}_H, \mathcal{V}_{\Pi}, \mathcal{V}_{\Gamma}, \mathcal{V}_{\vdash}$) to output claims, PCG-MAS guarantees that uncertified or harmful claims are blocked before execution, achieving dramatic reductions in uncertified harm.

### Key Benchmark Suite (56-Cell Matrix)
* **7 Frontier & Open-Weight LLM Backends:** `phi-3.5-mini`, `Llama-3.1-8B`, `Gemma-2-9b-it`, `qwen2.5-7B`, `Mistral-7B-Instruct-v0.3`, `Claude-3.5-Sonnet`, `deepseek-v3`.
* **8 Multi-Domain Datasets:** `FEVER`, `HotpotQA`, `PubMedQA`, `TAT-QA`, `WebLINX`, `ToolBench`, `Reindex-Drift`, and `Synthetic adversarial split` (`adversarial_integrity`).
* **56 Unique Cells:** $7 \text{ models} \times 8 \text{ datasets} = 56 \text{ unique cells}$.
* **Sample Size & Disclosed Seed Expansion:** 120 unique semantic examples per cell evaluated across 5 executed seeds ($\mathcal{S}_{\text{executed}} = \{0, 1, 2, 3, 4\}$) under clean and adversarial conditions ($240$ paired wide-form records per cell), yielding **$13,440$ total per-example execution records**. (Satisfies submitted Table 12 sample cap of $48 \le 500$ per seed).

---

## 📊 Data Classification Framework

All empirical values, tables, and artifacts in this repository adhere strictly to four clear provenance classifications:

1. **`DIRECT`**: Native execution records logged directly by the model server (e.g. `artifacts/rebuttal/source_records/per_example_records.jsonl` and `artifacts/rebuttal/backend_manifest/`).
2. **`DERIVED_FROM_DIRECT`**: Recomputed empirical metrics, tables, and figures derived directly from `DIRECT` records without modification (e.g. Tables 1–9, 13–18).
3. **`MODELLED`**: Analytic game-theoretic cost-responsibility models explicitly disclosed as mathematical projections (Tables 10–11).
4. **`PROTOCOL`**: Formal evaluation protocol specifications and hyperparameter constraints (Table 12).

---

## 📂 Eight Rebuttal Artifact Directories

The `artifacts/rebuttal/` directory contains complete, machine-readable artifacts corresponding to all eight rebuttal deliverables:

1. [**`table_reconciliation/`**](artifacts/rebuttal/table_reconciliation/): Canonical metric registry, exact cell numerators/denominators, cell-specific audit coverage variance, and recomputed manuscript tables.
2. [**`sv_decomposition/`**](artifacts/rebuttal/sv_decomposition/): Paired Selectivity ($S$) and Verification ($V$) harm avoidance decomposition with paired bootstrap confidence intervals ($S + V = \text{Total Harm Avoided}$, max identity residual $< 10^{-17}$).
3. [**`separating_witnesses/`**](artifacts/rebuttal/separating_witnesses/): Single-channel failure witness certificates ($W_H, W_{\Pi}, W_{\Gamma}, W_{\vdash}$) proving that each witness fails exactly one channel while passing the other three.
4. [**`citation_only/`**](artifacts/rebuttal/citation_only/): Matched-coverage comparative evaluations across No Certificate, Citation-Only, ShieldAgent, AgentRR, and PCG-MAS.
5. [**`injection/`**](artifacts/rebuttal/injection/): Adversarial prompt injection sweep across 4 attack locations (retrieved content, tool output, memory, delegated message) under isolated and shared verifier regimes.
6. [**`shift/`**](artifacts/rebuttal/shift/): Distribution shift evaluations across 6 shift families with fail-closed UCB validity-gate bounds.
7. [**`audit_sampling/`**](artifacts/rebuttal/audit_sampling/): Stratified audit sampling and variance bound analysis across 4 sampling designs.
8. [**`backend_manifest/`**](artifacts/rebuttal/backend_manifest/): Hardware, model revision, tokenizer, container digest, and decoding route fingerprints.

---

## 🛠 Reproduction & Validation Commands

All validators and clean-room reproduction pipelines are project-owned and runnable locally using standard Python:

```bash
# 1. Execute Master 7-Gate Validation Pipeline
python scripts/rebuttal/execute_full_workflow.py

# 2. Run Genuine 17-Mutation Negative Test Suite
python scripts/rebuttal/run_mutation_tests.py

# 3. Validate Submitted Protocol (Disclosed Seed Expansion)
python scripts/rebuttal/validate_submitted_protocol.py

# 4. Validate Executed Protocol
python scripts/rebuttal/validate_executed_protocol.py
```

---

## 📜 Citation

If you use PCG-MAS or these benchmark artifacts in your research, please cite:

```bibtex
@inproceedings{pcg_mas_2026,
  title={Proof-Carrying Graph Multi-Agent Systems: Certifiable Safety and Harm Avoidance},
  author={Anonymous Authors},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2026}
}
```
