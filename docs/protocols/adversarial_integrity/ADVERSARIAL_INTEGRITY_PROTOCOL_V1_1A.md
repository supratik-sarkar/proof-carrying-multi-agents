# ADVERSARIAL_INTEGRITY_PROTOCOL_V1_1A

Normative amendment to `ADVERSARIAL_INTEGRITY_PROTOCOL_V1_1`. It resolves four
freeze issues and changes nothing else. Attacks, quotas, families, modes,
construction machinery, adjudication scope and gates are **unchanged**; §E lists
the preservation set explicitly.

`MUST`, `MUST NOT` and `SHALL` are normative. `LP(x)` is the V1 §7.1
length-prefix encoding `len(utf8(x)).to_bytes(8,"big") ‖ utf8(x)`. Canonical JSON
is V1 §11.2. All digests are SHA-256, domain-separated by a distinct leading tag.

One item in V1.1 is **corrected as an error**, not merely tightened: §B. The
V1.1 §14.3 paired table was defined over all 1,000 pairs while §14.1 conceded
that treatment adjudication may return `VALID_SUPPORT` or `UNRESOLVED`. Those two
statements are inconsistent, and the inconsistency would have counted a rejected
*valid* treatment as a correct discrimination.

---

## A. Release identity — a three-root chain

### A.0 Renaming

| V1.1 name | V1.1A name | change |
|---|---|---|
| `pool_content_root` | `pool_content_root` | unchanged |
| `partition_manifest_root` | `partition_manifest_root` | unchanged |
| `dataset_release_root` | **`candidate_release_root`** | renamed only; construction and bound fields identical |
| — | **`adjudication_manifest_root`** | new (§A.2) |
| — | **`scientific_release_root`** | new (§A.3) |
| — | **`audit_failure_root`** | new (§A.4) |

The rename makes the pre-adjudication root's status legible in its own name: it
identifies a *candidate* pool, and a candidate pool is not a scientific result.

### A.1 `candidate_release_root` — computed before any annotator sees an item

```
candidate_release_root := SHA256( LP("ADVINT-CANDREL-v1_1A")
                                ‖ LP(protocol_version)
                                ‖ LP(source_revision)
                                ‖ LP(pool_content_root)
                                ‖ LP(partition_rule_id)
                                ‖ LP(partition_rule_version)
                                ‖ LP(partition_manifest_root)
                                ‖ LP(adjudication_protocol_sha256) )
```

`protocol_version` = `"ADVERSARIAL_INTEGRITY_PROTOCOL_V1_1A"`.

It MUST be computed and recorded at V1.1 §19 step 12′, before step 14′
(adjudication intake). No annotator MAY be shown any item until it exists.

### A.2 `adjudication_manifest_root`

**Adjudication record** (public payload; one per adjudicated item):

```jsonc
{
  "item_id":                      "<pair_id>#control" | "<pair_id>#treatment",
  "primary_responses":            ["INVALID","VALID"],        // see A.2.1
  "adjudicator_response":         "VALID" | "INVALID" | "UNCERTAIN" | null,
  "adjudicated_support_relation": "VALID_SUPPORT" | "INVALID_SUPPORT" | "UNRESOLVED",
  "adjudication_status":          "ADJUDICATED",
  "hypothesis_confirmed":         true | false | null,
  "adjudication_protocol_sha256": "<64 hex>"
}
```

The public payload MUST NOT contain a name, email, employer, worker-platform
handle, timestamp, IP, locale, or any other human-identifying field.

#### A.2.1 Response-pair canonicalisation

`primary_responses` MUST be the two primary responses **sorted ascending as UTF-8
strings** (`INVALID` < `UNCERTAIN` < `VALID`). Sorting serves two purposes: it
makes the root independent of which annotator was enumerated first, and it
removes the implicit annotator ordering that a fixed position would encode.

#### A.2.2 Pseudonymous attribution — committed, not published

Per-annotator attribution is required for audit (drift, collusion, single-rater
dominance) but MUST NOT be public.

```
annotator_pseudonym := hex( HMAC-SHA256(key = annotator_salt,
                                        msg = utf8(annotator_real_id)) )[0:32]
```

- `annotator_salt` is ≥ 32 random bytes, generated once, held by a named custodian
  who is not an annotator and not a protocol author, and **never published**.
- The **attribution file** is private and holds, per item,
  `{item_id, primary_pseudonyms: [p1, p2] in response order, adjudicator_pseudonym}`.
  Because it preserves response order, it can be joined to the sorted public
  responses only by someone holding the file — which is the intent.
- Its root, sorted by `item_id`:

```
annotator_attribution_root := SHA256( LP("ADVINT-ATTRIB-v1_1A") ‖ LP(protocol_version)
                                    ‖ m.to_bytes(8,"big")
                                    ‖ for each record in item_id order:
                                        LP(item_id) ‖ LP(p1) ‖ LP(p2) ‖ LP(adjudicator_pseudonym ?? "") )
```

`annotator_attribution_root` **is published as a bare 64-hex value**; the file it
commits to is not. A third party can therefore recompute
`adjudication_manifest_root` from the public records plus this one opaque value,
while learning nothing about any annotator. An auditor granted the file can
verify the commitment.

Salt destruction is prohibited before the manuscript is accepted or withdrawn:
destroying it would make the attribution commitment unverifiable and the audit
claim empty.

#### A.2.3 The root

Records sorted ascending by `item_id` as UTF-8 bytes, `k` = record count:

```
adjudication_manifest_root := SHA256( LP("ADVINT-ADJMAN-v1_1A")
                                    ‖ LP(protocol_version)
                                    ‖ LP(candidate_release_root)
                                    ‖ LP(adjudication_protocol_sha256)
                                    ‖ LP(CALIBRATION_GOLD_MANIFEST_SHA256)
                                    ‖ LP(annotator_attribution_root)
                                    ‖ k.to_bytes(8,"big")
                                    ‖ for each record in item_id order:
                                        LP(canonical_json(record)) )
```

Each record is length-prefixed individually, so the manifest is bound
element-by-element rather than summarised. Binding `candidate_release_root`
inward means an adjudication manifest cannot be transplanted onto a different
pool.

`adjudication_manifest_root` is **immutable**. A corrected adjudication is a new
manifest, a new root, and a new `scientific_release_root` — never an edit.

### A.3 `scientific_release_root` — issued only on PASS

```
adjudication_gate_status ∈ {PASS, FAIL}
```

determined by V1.1 §19 step 15′ (α floor 0.67; six cell gates at
`CP-lower ≥ 0.85`; control stratum `CP-lower ≥ 0.90`; SSH type plausibility
`CP-lower ≥ 0.80`). Unchanged.

```
scientific_release_root := SHA256( LP("ADVINT-SCIREL-PASS-v1_1A")
                                 ‖ LP(protocol_version)
                                 ‖ LP(candidate_release_root)
                                 ‖ LP(adjudication_manifest_root)
                                 ‖ LP("PASS") )
```

It MUST be computed **only** when `adjudication_gate_status = PASS`.

**Required property, and it holds by construction.** Two candidate pools with
identical construction share `candidate_release_root`. If any item's
`adjudicated_support_relation` differs, that record's canonical JSON differs, so
`adjudication_manifest_root` differs (§A.2.3 binds every record individually),
so `scientific_release_root` differs. Differing adjudicated labels therefore
**cannot** share a scientific release root.

### A.4 Failure root — a different domain tag, deliberately

```
audit_failure_root := SHA256( LP("ADVINT-AUDITFAIL-v1_1A")
                            ‖ LP(protocol_version)
                            ‖ LP(candidate_release_root)
                            ‖ LP(adjudication_manifest_root)
                            ‖ LP("FAIL")
                            ‖ LP(canonical_json(gate_failure_report)) )
```

`gate_failure_report` carries the per-cell confirmation rates and intervals, α
and its CI, the control-stratum rate, the SSH plausibility rate, and the full
disagreement matrix — V1.1 §3.10's legibility requirement.

The leading domain tag differs from §A.3's. A failed release therefore cannot be
mistaken for, relabelled as, or replayed as a passing one: no input to the
`PASS` construction can ever produce an `AUDITFAIL` digest or the reverse. This
is the enforcement, not a naming convention.

### A.5 Experiment pinning (normative)

The seven-provider pilot and final experiments MUST pin
`scientific_release_root`. Pinning `candidate_release_root` alone is
**prohibited**: the candidate root is identical across every possible
adjudication outcome, including one in which every gate failed, so it certifies
construction but certifies nothing scientific.

The evaluation harness MUST refuse to start when:

- `scientific_release_root` is absent, malformed, or not `^[0-9a-f]{64}$`; or
- `adjudication_gate_status ≠ PASS`; or
- the recomputed `scientific_release_root` differs from the pinned value.

There MUST be no flag, environment variable or configuration that relaxes any of
these.

---

## B. Paired-analysis population — correction

### B.1 The error being corrected

V1.1 §14.3 drew the paired 2×2 over all `P = 1000` pairs. V1.1 §14.1 defines
`T_val` and `T_unres` as non-empty in general. Under the §14.3 table a pair whose
treatment was adjudicated `VALID_SUPPORT` but rejected by the system would land in
the "correct discrimination" cell. That is a **false rejection scored as a
success**. The table is replaced.

### B.2 Populations (partition the 1,000 pairs)

```
P_inv   = { pair : treatment adjudication = INVALID_SUPPORT }
P_val   = { pair : treatment adjudication = VALID_SUPPORT }
P_unres = { pair : treatment adjudication = UNRESOLVED }

|P_inv| + |P_val| + |P_unres| = 1000        (asserted and reported)
```

All three cardinalities MUST be reported wherever any cell result is reported,
together with the six per-cell adjudication confirmation rates and their
Clopper–Pearson intervals.

### B.3 Primary table — over `P_inv` only

`A_c = 1` iff the system accepted the control certificate; `A_t = 1` iff it
accepted the treatment certificate.

| | treatment **rejected** (`A_t=0`) | treatment **accepted** (`A_t=1`) |
|---|---|---|
| **control accepted** (`A_c=1`) | `n₁₁` correct discrimination | `n₁₂` invalid accepted |
| **control rejected** (`A_c=0`) | `n₂₁` valid control rejected | `n₂₂` inverted |

```
n₁₁ + n₁₂ + n₂₁ + n₂₂ = |P_inv|
```

All four counts MUST be printed. No summary may replace them: `n₁₂` and `n₂₁`
are different failures — accepting a corrupted certificate versus rejecting a
sound one — and pooling them is the collapse the certificate discipline exists to
prevent.

### B.4 Reported quantities on `P_inv`

| quantity | definition | interval |
|---|---|---|
| `PairedDiscrimination` | `n₁₁ / |P_inv|` | Clopper–Pearson exact 95% |
| `IAR_inv` invalid acceptance | `(n₁₂ + n₂₂) / |P_inv|` | Clopper–Pearson exact 95% |
| `CtrlRej_inv` valid-control rejection within the paired population | `(n₂₁ + n₂₂) / |P_inv|` | Clopper–Pearson exact 95% |

**Paired test.** McNemar's **exact** (binomial) test on the discordant cells
`b = n₁₁` and `c = n₂₂`, `H₀ : p = 1/2`, two-sided; report the exact p-value and
the exact 95% interval for `b/(b+c)`. Concordant cells `n₁₂` (both accepted) and
`n₂₁` (both rejected) carry no discrimination information and MUST NOT enter the
test statistic. The asymptotic χ² form MUST NOT be used: discordant counts here
may be small, and the exact test is available.

### B.5 Prohibitions

- A rejected `P_val` treatment MUST NOT be counted as a correct rejection, in any
  metric, at any aggregation level.
- `P_unres` pairs MUST NOT appear in the numerator or denominator of any
  validity-effect metric.
- `P_val` and `P_unres` MUST NOT be merged into `P_inv` for any reason,
  including quota restoration.

### B.6 Secondary and supplementary lines

| line | population | note |
|---|---|---|
| `CFRR` control false rejection | all **1000** controls | control validity rests on FEVER annotation plus the 200-item noise estimate, independently of the partner's adjudication; unchanged from V1.1 §14.2 |
| `CFRR_adj` noise-corrected | adjudicated-`VALID_SUPPORT` controls within the 200 subsample | unchanged |
| `P_val` treatment rejection rate | `|P_val|` | reported as a **supplementary false-rejection** observation; these are accidental controls, and rejecting them is an error |
| `P_unres` acceptance rate | `|P_unres|` | descriptive only; MUST be labelled "no ground truth" |

### B.7 The all-candidate execution table

An operational 1,000-pair table MAY be retained for engineering purposes. If
retained it MUST be titled

```
ALL_CANDIDATE_EXECUTION_TABLE (not a ground-truth discrimination table)
```

and MUST carry the sentence: *"This table mixes pairs whose treatment was
adjudicated valid or unresolved; it is not evidence of certificate
discrimination."* It MUST NOT appear in the manuscript's results section and MUST
NOT be the source of any reported effect.

---

## C. Annotator-calibration gold — frozen process

V1.1 §3.2 required `≥ 0.80` agreement with a 30-item gold set without saying how
the gold is established. This defines it.

### C.1 Candidate selection (deterministic)

Eligible calibration parents are those satisfying V1 §1.5 admissibility and
**not** used by any pair in the pool. Rank ascending by

```
SHA256( LP("ADVINT-CALIB-v1_1A") ‖ LP(protocol_version)
      ‖ LP(candidate_release_root) ‖ LP(source_id) )
```

Binding `candidate_release_root` means a calibration set cannot be recycled from
a different pool, and cannot have been constructed before the pool was frozen.

Calibration **items** are built from the ranked parents in strict order,
alternating item type so the set is not degenerate:

- odd positions → an unperturbed certificate (claim, evidence, FEVER verdict);
- even positions → a perturbed certificate, cycling the families in the fixed
  order `CITATION_SWAP → SEMANTIC_SLOT_HIJACK → NUMBER_FLIP` with the V1.1 §12.1
  label-conditional mode.

A gold set drawn only from unperturbed parents would be all-`VALID`, and an
annotator answering `VALID` always would qualify. The alternation exists solely
to prevent that.

### C.2 Expert calibration panel

**Three experts**, none of whom is a primary annotator, an adjudicator, or a
protocol author.

**Qualification for this task, and why it needs no world knowledge.** The
calibration question is *does this sentence establish this verdict for this
claim* — a textual support judgment internal to the supplied evidence. It is a
reading-comprehension and entailment task, not fact-checking. An expert therefore
qualifies by demonstrated competence in **textual entailment annotation**, not by
domain knowledge. Any one of:

- prior paid annotation experience on a natural-language-inference or
  fact-verification corpus, with documented agreement scores; or
- graduate-level training in linguistics, NLP, philosophy of language, or
  analytic philosophy; or
- professional experience requiring formal reading of evidentiary text
  (e.g. legal drafting or systematic review screening);

**and** in all cases ≥ 0.90 exact agreement on a published 20-item NLI
warm-up set answered under the §C.3 wording. Subject-matter expertise in the
claim's topic is neither required nor desirable: an expert who knows the answer
from outside the sentence is more likely to violate the "use only the evidence
sentence" instruction, which would drag the gold back across the `ε_src` boundary
V1.1 §1.3 established.

No LLM judge participates at any stage, including drafting or pre-screening.

### C.3 Expert question and blinding

The experts answer the **exact V1.1 §3.5 wording**, on the **exact V1.1 §3.4
payload** (`CLAIM`, `EVIDENCE`, `VERDICT`, nothing else). Experts MUST NOT see
attack family, attack mode, item role, intended target, parent id, or the
alternating construction rule of §C.1.

### C.4 Gold resolution rule (predeclared)

Each expert answers `VALID` / `INVALID` / `UNCERTAIN`.

```
gold = VALID_SUPPORT     iff ≥ 2 answer VALID   and 0 answer INVALID
gold = INVALID_SUPPORT   iff ≥ 2 answer INVALID and 0 answer VALID
otherwise                    INDETERMINATE  →  the item does not enter the gold set
```

The rule is strict — a single opposing determinate answer voids the item —
because a contested item used as gold would fail competent annotators for
disagreeing with a coin flip.

### C.5 Set formation, balance, and termination

Scan the ranked candidates in order, admitting determinate items, until:

```
|gold set| = 30   AND   |gold = VALID_SUPPORT| ≥ 8   AND   |gold = INVALID_SUPPORT| ≥ 8
```

If the constraint is unmet after **200** ranked candidates have been adjudicated,
the process **ESCALATEs** as `CALIBRATION_GOLD_INFEASIBLE` and halts. There is no
relaxation of the balance floor and no random substitution. The 8/8 floor
guarantees that neither constant strategy exceeds `22/30 = 0.733`, below the
`0.80` qualification bar.

### C.6 Frozen manifest

```jsonc
{
  "protocol_version": "ADVERSARIAL_INTEGRITY_PROTOCOL_V1_1A",
  "candidate_release_root": "<64 hex>",
  "expert_protocol_sha256": "<64 hex>",
  "items": [ { "calibration_item_id": "advint-calib-v1_1A:<source_id>:<type>",
               "payload_sha256": "<64 hex>",          // SHA256(LP("ADVINT-CALIBPAY-v1_1A")
                                                      //        ‖ LP(claim) ‖ LP(evidence_text) ‖ LP(verdict))
               "gold": "VALID_SUPPORT" | "INVALID_SUPPORT" } ],
  "n_valid": <int>, "n_invalid": <int>, "n_scanned": <int>
}
```

```
CALIBRATION_GOLD_MANIFEST_SHA256 := SHA256( LP("ADVINT-CALIBMAN-v1_1A")
                                          ‖ LP(canonical_json(manifest)) )
```

Items sorted ascending by `calibration_item_id`. Expert identities are handled
exactly as §A.2.2 (pseudonymous, committed, not published).

### C.7 Ordering constraints (normative)

```
candidate_release_root  →  calibration gold manifest frozen
                        →  CALIBRATION_GOLD_MANIFEST_SHA256 published
                        →  primary annotator qualification begins
                        →  §3 adjudication begins
```

No primary annotator MAY be qualified before
`CALIBRATION_GOLD_MANIFEST_SHA256` is published. The value is bound into
`adjudication_manifest_root` (§A.2.3), so a gold set swapped after qualification
would change the adjudication root and be detectable. Qualification workers MUST
NOT see gold labels, attack family, attack mode, or intended targets, and a
failed candidate MUST NOT be re-tested on the same 30 items.

---

## D. Pair-leakage firewall (evaluation-runtime invariant)

### D.1 The invariant

For **every** evaluated item, the execution MUST satisfy all of:

| # | requirement |
|---|---|
| F1 | fresh conversational/model context; no turn history from any prior item |
| F2 | fresh agent-state scope; execution mode `FRESH`, unique thread identity, no checkpoint reuse |
| F3 | no prior paired item present in model-visible context |
| F4 | `pair_id` not exposed |
| F5 | `item_role` not exposed |
| F6 | `attack_family` not exposed |
| F7 | `attack_mode` not exposed |
| F8 | `parent_id` not exposed |
| F9 | no adjudication metadata exposed (any GROUP B field) |
| F10 | no partition metadata exposed |
| F11 | no response, output, trace or score from the paired item exposed |
| F12 | no cross-item agent memory, cache or retrieval index that can contain paired-item content |

System-internal audit and provenance MAY retain every one of these identifiers.
They MUST live outside the model-visible input and MUST NOT be readable by any
decision logic.

### D.2 `MODEL_VISIBLE_PAYLOAD_HASH`

The evaluated system MUST emit, per item, the exact bytes presented to the model:

```
model_visible_payload := { "claim": "<presented_claim>",
                           "evidence": "<presented_evidence.text>",
                           "verdict": "SUPPORTED" | "REFUTED",
                           "instructions": "<verbatim task instruction text>" }

MODEL_VISIBLE_PAYLOAD_HASH := SHA256( LP("ADVINT-MVP-v1_1A")
                                    ‖ LP(canonical_json(model_visible_payload)) )
```

Any additional model-visible content — a system prompt, a retrieved passage, a
tool result, a scratchpad carried in — MUST be appended to
`model_visible_payload` under an `"extra"` key before hashing. A reviewer can
then confirm exactly what the system saw, and a payload that omitted a leak from
the hash would fail F13 below.

### D.3 `PAIR_LEAKAGE_FIREWALL` — machine-checkable

Per item, all checks MUST pass:

| check | test |
|---|---|
| F13 | `MODEL_VISIBLE_PAYLOAD_HASH` recomputed from the recorded payload equals the emitted value |
| F14 | forbidden-substring scan over the concatenated payload finds none of: `pair_id`, `item_id`, `parent_id`, `attack_family` value, `attack_mode` value, `item_role` value, `partition` value, any GROUP B field name or value, the literal `advint-`, or any protocol domain tag |
| F15 | the paired item's `presented_claim` and `presented_evidence.text` do not occur, as normalised substrings (V1 §11.1), anywhere in the payload |
| F16 | the item's execution context id is unique across the run, and its thread identity is unused (a `ThreadCollision` is a firewall failure, not a retry) |
| F17 | `execution_mode = FRESH` for every evaluated item |
| F18 | no attached memory, cache or index has a scope broader than the item; if a memory component is part of the evaluated architecture, its scope id MUST equal the item's execution context id |
| F19 | presentation order equals the precommitted order of V1.1 §2.5, so control/treatment adjacency cannot be inferred from position |

```
PAIR_LEAKAGE_FIREWALL := PASS  iff every item in the run passes F1–F19
                         FAIL  otherwise
```

`FAIL` is fail-closed at **run** granularity: a single failing item fails the
run, and **no result from that run may be reported**. A failing item MUST NOT be
excluded so that the remainder can pass — that is the replacement pattern V1.1
§3.10 prohibits, applied to execution.

The per-item verdict, the failing check ids, and `MODEL_VISIBLE_PAYLOAD_HASH` are
recorded in the run record and reported with every result table.

### D.4 Declared metadata exposure (narrow carve-out)

An evaluated architecture may genuinely require non-label metadata. If so:

- the field MUST be declared in the run manifest **before** the run, as
  `DECLARED_METADATA_EXPOSURE = [ … ]`, and the declaration is bound into the run
  record hash;
- the following are **hard-forbidden and can never be declared**: `item_role`,
  `attack_family`, `attack_mode`, `pair_id`, `parent_id`, `partition`, every
  GROUP B adjudication field, and anything derived from the paired item;
- a declared field is scanned by F14 as declared-permitted rather than forbidden;
  all other checks still apply;
- any result from a run with a non-empty declaration MUST be reported with the
  declaration printed beside it.

---

## E. Preservation set (unchanged by this amendment)

FEVER-only source · `N = 1000` parent/attack pairs · 500 `SUPPORTS` / 500
`REFUTES` · quotas 334 / 333 / 333 with the V1 label cross-allocation ·
`CONTROL_DESIGN = PAIRED-REALIGN-ADJ-v1_1` · `DESUPPORT` · `REALIGN` ·
citation-swap construction (tiers, Jaccard by integer cross-multiplication, C1–C6)
· semantic-slot hijack (gazetteer, copula type signature, tiers, donor ranking) ·
number-flip machinery (parser, eligible/excluded classes, hash-governed sign,
exact decimal, `ROUND_HALF_UP`, degenerate-case rules) · no LLM in construction ·
adjudication census of 1,000 treatments · 200 control adjudications · 100 SSH
plausibility riders · α floor 0.67 · CP-lower cell gates 0.85 / 0.90 / 0.80 · no
candidate replacement · the §1 source/world ontology correction ·
`expected_failed_channel` remains removed · the partitioning framework and pair-level
assignment · fail-closed `n_final` feasibility with ESCALATE · the U3 frozen
diagnostic and the U4 descriptive-only decision · one parent at most once ·
deterministic ranking, donor caps, no stochastic fallback, exact
failure-on-quota-exhaustion.

Only §§A–D of this amendment change anything, and they change only release
identity, the analysis population, calibration gold, and evaluation-runtime
isolation.

---

## F. Gap status after this amendment

Resolved here: release-identity binding of adjudication (A), paired-population
error (B), calibration-gold underspecification (C), pair-leakage runtime
invariant (D).

Newly opened, all operational rather than scientific:

| id | item | classification |
|---|---|---|
| U12 | `annotator_salt` custodian not yet named; salt not yet generated | `BLOCKS_SCIENTIFIC_EVALUATION` |
| U13 | expert calibration panel not yet constituted | `BLOCKS_SCIENTIFIC_EVALUATION` |
| U14 | firewall checks F1–F19 not yet implemented in the harness | `BLOCKS_SCIENTIFIC_EVALUATION` |

Carried from V1.1 §15 unchanged: U1 `BLOCKS_DATASET_CONSTRUCTION`; U5
`BLOCKS_PROTOCOL_FREEZE`; U2′, U6, U7, U3, U8 `BLOCKS_SCIENTIFIC_EVALUATION`;
U9, U10 `BLOCKS_MANUSCRIPT_REPORTING`; U4, U11 `NONBLOCKING_DIAGNOSTIC`.

```
BLOCKS_PROTOCOL_FREEZE       : 1   (U5 — main-FEVER final split unconfirmed)
BLOCKS_DATASET_CONSTRUCTION  : 1   (U1 — source pin)
BLOCKS_SCIENTIFIC_EVALUATION : 8   (U2′, U6, U7, U3, U8, U12, U13, U14)
BLOCKS_MANUSCRIPT_REPORTING  : 2   (U9, U10)
NONBLOCKING_DIAGNOSTIC       : 2   (U4, U11)
total blocking               : 12
```

---

## G. End state

```
ADVERSARIAL_INTEGRITY_PROTOCOL_V1_1A
V1_1A_RECOMMENDED=YES
CONTROL_DESIGN=PAIRED-REALIGN-ADJ-v1_1
CANDIDATE_RELEASE_ROOT_BINDS_PARTITIONS=YES
ADJUDICATION_MANIFEST_ROOT_DEFINED=YES
SCIENTIFIC_RELEASE_ROOT_BINDS_ADJUDICATION=YES
PRIMARY_PAIRED_POPULATION=P_inv
VALID_TREATMENTS_COUNTED_AS_CORRECT_REJECTIONS=NO
UNRESOLVED_TREATMENTS_IN_EFFECT_DENOMINATOR=NO
CALIBRATION_GOLD_PROTOCOL_DEFINED=YES
PAIR_LEAKAGE_FIREWALL_DEFINED=YES
LLM_USED_IN_DATASET_CONSTRUCTION=NO
UNRESOLVED_PROTOCOL_FREEZE_GAPS=1
```

`UNRESOLVED_PROTOCOL_FREEZE_GAPS = 1` counts only gaps blocking **protocol
freeze** — U5, the unconfirmed main-FEVER final split. Eleven further gaps block
later stages and are listed in §F; they do not block freeze.
