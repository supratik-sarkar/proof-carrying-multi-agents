# Protocol Deviation Report — Submission 9327

## Protocol Status: POST_REVIEW_SEED_EXPANSION
* **SUBMITTED_SEEDS_PRESENT:** `true`
* **SUBMITTED_SAMPLE_CAP_SATISFIED:** `true` (48 <= 500)
* **EXACT_SUBMITTED_SEED_SET_REPRODUCED:** `false`
* **POST_REVIEW_SEED_EXPANSION_DISCLOSED:** `true`

### 1. Submitted Protocol vs Executed Protocol Comparison
* **Submitted Protocol (Table 12):** 7 models x 8 datasets x 4 seeds ({0,1,2,3}) x up to 500 examples/seed = **maximum possible 112,000 evaluations**.
* **Executed Protocol:** 7 models x 8 datasets x 5 seeds ({0,1,2,3,4}) x 48 evaluations/seed = **13,440 total evaluations** (240 paired wide-form evaluations per cell; 120 clean + 120 adversarial).

### 2. Disclosures & Rationale
* **Executed Seeds:** Submitted seeds {0, 1, 2, 3} are fully present; seed 4 was added post-review.
* **Sample Cap:** 48 evaluations per seed satisfies the submitted "up to 500" per-seed upper bound.
