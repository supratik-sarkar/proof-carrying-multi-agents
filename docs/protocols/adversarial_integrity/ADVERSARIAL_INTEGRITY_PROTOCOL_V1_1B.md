# ADVERSARIAL\_INTEGRITY\_PROTOCOL\_V1\_1B

Normative **erratum** to `ADVERSARIAL_INTEGRITY_PROTOCOL_V1_1A`. It corrects two defects that would have made the pair-leakage firewall unusable, closes one determinism gap in calibration construction, and repairs one auditability defect in the private attribution record. Nothing scientific changes: attacks, quotas, modes, adjudication, roots, metrics and partitioning are untouched (§5).

`MUST` / `MUST NOT` are normative. `LP(x)`, canonical JSON, `norm()` (V1 §11.1 text normalisation) and SHA-256 domain separation are as previously defined.

**Two of the three items below are errors in V1.1A, not refinements.** As written, F15 would have failed **every** item in the pool and F14 would have failed a large and content-dependent fraction. Both are corrected here.

---

## 1. F15 — paired-exclusive content, not shared content

### 1.1 The defect

V1.1A F15 required that "the paired item's `presented_claim` and `presented_evidence.text` do not occur, as normalised substrings, anywhere in the payload." Paired items **deliberately** share one of those two fields:

| familyshared by constructionconsequence under V1.1A F15 |                                                     |                                  |
| ------------------------------------------------------- | --------------------------------------------------- | -------------------------------- |
| `CITATION_SWAP`                                         | `control.claim == treatment.claim` (byte-identical) | both items of all 334 pairs fail |
| `SEMANTIC_SLOT_HIJACK`                                  | `control.evidence == treatment.evidence`            | both items of all 333 pairs fail |
| `NUMBER_FLIP`                                           | `control.evidence == treatment.evidence`            | both items of all 333 pairs fail |

That is 2,000 of 2,000 evaluation items. `PAIR_LEAKAGE_FIREWALL` could never have returned `PASS`. The rule confused *sharing*, which is the design, with *leakage*, which is the hazard.

### 1.2 Definitions

Let `x` be the item under evaluation and `y` its paired partner. The firewall reads `y`'s record and both items' construction metadata from **system-internal provenance, outside the model view**.

The recorded payload (V1.1A §D.2) is partitioned:

```
CORE      := { claim, evidence, verdict }        # taken verbatim from x's record
FROZEN    := { instructions }                    # a run-level constant template
FREE      := every entry under "extra"           # system prompt, retrieval, tool output, scratchpad

```

```
SHARED_FIELDS(x,y)  := { f ∈ CORE : norm(x.f) == norm(y.f) }

```

Shared fields are `x`'s own legitimate content. They are **never** a violation, whatever `y` contains.

```
PAIRED_EXCLUSIVE_FIELDS(y \ x) := { norm(y.f) : f ∈ CORE, norm(y.f) ≠ norm(x.f) }

```

plus the family-specific span-level deltas, all read from construction metadata:

| familyadded to the exclusive set when `x` is the **control**when `x` is the **treatment** |                                        |                               |
| ----------------------------------------------------------------------------------------- | -------------------------------------- | ----------------------------- |
| `CITATION_SWAP`                                                                           | `donor_page`, `str(donor_sentence_id)` | —                             |
| `SEMANTIC_SLOT_HIJACK`                                                                    | `slot_replacement`                     | `slot_original`               |
| `NUMBER_FLIP`                                                                             | `perturbed_value` surface form         | `original_value` surface form |

**Independent-presence carve-out (normative).** After assembly:

```
D(x,y) := PAIRED_EXCLUSIVE_FIELDS(y \ x)  \  { v : norm(v) occurs in norm(CORE(x)) }

```

A value that independently remains in `x`'s own core content is removed from the exclusive set. This is what makes the rule *delta-based*: information is never forbidden merely because `y` also contains it. It covers the real case in which`slot_original` occurs twice in a claim and only one occurrence was replaced, so the original surface form legitimately survives in the treatment claim.

### 1.3 `F15_PAIRED_EXCLUSIVE_CONTENT_ABSENT` — exact algorithm

```
F15a  CORE INTEGRITY  (exact equality, never substring)
      norm(payload.claim)    == norm(record_x.presented_claim)
      norm(payload.evidence) == norm(record_x.presented_evidence.text)
      payload.verdict        == record_x.asserted_verdict

F15b  INSTRUCTION INTEGRITY
      SHA256(LP("ADVINT-INSTR-v1_1B") ‖ LP(payload.instructions))
            == frozen_instruction_template_sha256          # registered in the run manifest

F15c  PAIRED-EXCLUSIVE ABSENCE  (evaluated over FREE only)
      free_text := norm( concatenation of every FREE entry value, separated by U+001E )
      for each v ∈ D(x,y):
          norm(v) MUST NOT occur as a substring of free_text

```

`F15 := F15a ∧ F15b ∧ F15c`.

**Why the scan is confined to ****`FREE`****.** `CORE` is verified by exact equality to the record, so it is legitimate by definition and cannot be a leak vector. `FROZEN` is verified by hash against a template that contains no per-item content. Every remaining model-visible byte is in `FREE`, which is exactly where a leaked partner value would have to appear. Scanning `CORE` for partner content is what produced the V1.1A defect.

### 1.4 Retrieval carve-out

A system may legitimately retrieve, from a registered corpus, text that coincides with the partner's content — a RAG system evaluating a `CITATION_SWAP` treatment may retrieve the original FEVER evidence. That is agent capability, not pair leakage, and forbidding it would forbid retrieval-augmented architectures.

A `FREE` entry whose declared provenance (§2.3) is `RETRIEVAL_RESULT` or `TOOL_RESULT` from a source registered in the run manifest is **exempt from F15c**. Any run containing such an exemption MUST record

```
RETRIEVAL_MAY_SURFACE_PAIRED_CONTENT = TRUE

```

in the run record, and that flag MUST be printed beside every result from the run. Entries with any other provenance are never exempt.

---

## 2. F14 — hard tokens and structural provenance

### 2.1 The defect

V1.1A F14 scanned the payload for "any `attack_family` value, `attack_mode` value, `item_role` value, `partition` value". Those values are ordinary English words. `FINAL`, `CONTROL`, `TREATMENT`, `SWAP` and `REALIGN` occur in Wikipedia sentences and therefore in FEVER claims and evidence — "the 2016 NBA Finals", "the control group", "a prisoner swap", "the company realigned its divisions". Case-insensitive matching makes it worse. The check would have failed a large, content-dependent and scientifically meaningless subset of items.

Substring scanning is the wrong instrument for generic categorical values. It is replaced by two mechanisms with disjoint jobs.

### 2.2 F14a — hard-token substring scan (exact set)

Only tokens whose mere occurrence is itself strong evidence of leakage are scanned. The set is closed; nothing may be added at runtime.

```
HARD_TOKENS(x) :=
  1. x.pair_id                                  e.g. "advint-v1_1:12345:CITATION_SWAP"
  2. x.item_id  and  y.item_id                  both "#control" and "#treatment" forms
  3. y.pair_id
  4. any match of  /advint-[A-Za-z0-9_.:-]+/     (namespaced instance identifiers)
  5. any match of  /ADVINT-[A-Z0-9_-]+/          (protocol domain tags)
  6. the namespaced parent reference  "fever:" ‖ source_split ‖ ":" ‖ parent_id
  7. the run's execution_context_id, thread_id, trace_id, span_id (exact recorded values)
  8. any 64-hex root or digest recorded for this release:
       candidate_release_root, pool_content_root, partition_manifest_root,
       adjudication_manifest_root, scientific_release_root, audit_failure_root,
       adjudication_protocol_sha256, CALIBRATION_GOLD_MANIFEST_SHA256,
       annotator_attribution_root, record_sha256(x), record_sha256(y)
  9. any annotator or expert pseudonym (32-hex)

```

Scanning is over the **complete recorded payload** (`CORE ‖ FROZEN ‖ FREE`), case-sensitively for items 1–8 and case-insensitively for item 9.

**Explicitly NOT scanned** (normative prohibition — scanning any of these is a protocol violation, because they are generic categorical values that occur in natural language):

```
FINAL · PILOT · SMOKE · CHECKER_CALIBRATION · CONTROL · TREATMENT
SWAP · DESUPPORT · REALIGN · CITATION_SWAP · SEMANTIC_SLOT_HIJACK · NUMBER_FLIP
SUPPORTS · REFUTES · SUPPORTED · REFUTED · VALID_SUPPORT · INVALID_SUPPORT
UNRESOLVED · VALID · INVALID · UNCERTAIN · the bare integer parent_id

```

`SUPPORTED` and `REFUTED` are the model-visible verdict itself; scanning for them would fail every item on its own legitimate content. The bare integer `parent_id` is excluded because FEVER ids collide with years and quantities that occur throughout the corpus.

### 2.3 F14b — `MODEL_VISIBLE_FIELD_PROVENANCE` (structural)

Generic categorical metadata is blocked **at payload construction**, not by search. The payload assembler MUST emit a machine-readable source map covering every model-visible field:

```jsonc
MODEL_VISIBLE_FIELD_PROVENANCE = {
  "claim":        "DATASET_PRESENTED_CLAIM",
  "evidence":     "DATASET_PRESENTED_EVIDENCE_TEXT",
  "verdict":      "DATASET_ASSERTED_VERDICT",
  "instructions": "FROZEN_INSTRUCTION_TEMPLATE",
  "extra": { "<key>": "<source tag>", ... }          // one entry per extra field
}

```

**Permitted source tags (closed allowlist):**

```
DATASET_PRESENTED_CLAIM · DATASET_PRESENTED_EVIDENCE_TEXT · DATASET_ASSERTED_VERDICT
FROZEN_INSTRUCTION_TEMPLATE · SYSTEM_STATIC_PROMPT   (hash-registered in the run manifest)
RETRIEVAL_RESULT · TOOL_RESULT                        (source registered in the run manifest)
MODEL_SELF_GENERATED                                  (the system's own prior tokens, same item)

```

**Hard-forbidden sources — never declarable, never permitted:**

```
PAIR_IDENTITY · ITEM_ROLE · ATTACK_FAMILY · ATTACK_MODE · PARTITION
PARENT_IDENTITY · CONSTRUCTION_METADATA (attack_parameters)
SUPPORT_RELATION_HYPOTHESIS · SCIENTIFIC_TARGET_LABEL
any ADJUDICATION_* field · PAIRED_ITEM_OUTPUT

```

The firewall verifies:

```
F14b-1  every field present in the payload has exactly one provenance entry
F14b-2  every provenance entry corresponds to a field present in the payload  (exhaustive both ways)
F14b-3  every declared source ∈ the permitted allowlist
F14b-4  no declared source ∈ the hard-forbidden set
F14b-5  RETRIEVAL_RESULT / TOOL_RESULT sources are registered in the run manifest
F14b-6  SYSTEM_STATIC_PROMPT content hashes to its registered value

```

This is strictly stronger than searching for the word `FINAL`: a field carrying partition information fails F14b-4 on its **source tag** regardless of how the value is spelled, obfuscated, translated or encoded.

```
F14 := F14a ∧ F14b

```

### 2.4 F13 — completeness, restated

```
F13a  MODEL_VISIBLE_PAYLOAD_HASH recomputed from the complete recorded payload
      (CORE ‖ FROZEN ‖ FREE, canonical JSON) equals the emitted value
F13b  the harness attests COMPLETE_PAYLOAD_CAPTURE = TRUE: no model-visible
      content of any kind existed outside the recorded, hash-bound payload —
      no side channel, no out-of-band system message, no untracked tool output

```

F13b is what gives F14 and F15 their force. Without it, both are defeatable by omitting the leak from the recorded payload.

### 2.5 Revised F13–F19

| idcheckstatus vs V1.1A |                                                                                                                        |                        |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------- | ---------------------- |
| F13a                   | payload hash recomputes from the complete recorded payload                                                             | clarified              |
| F13b                   | `COMPLETE_PAYLOAD_CAPTURE = TRUE` attestation                                                                          | **new**                |
| F14a                   | hard-token substring scan over the exact §2.2 set                                                                      | **replaces** V1.1A F14 |
| F14b                   | `MODEL_VISIBLE_FIELD_PROVENANCE` allowlist, exhaustive both ways                                                       | **new**                |
| F15a                   | core fields exactly equal the record's values                                                                          | **replaces** V1.1A F15 |
| F15b                   | instructions hash to the frozen template                                                                               | **new**                |
| F15c                   | paired-exclusive delta set absent from `FREE`, with the §1.2 carve-out and the §1.4 retrieval exemption                | **replaces** V1.1A F15 |
| F16                    | execution context id unique in the run; thread identity unused; a `ThreadCollision` is a firewall failure, not a retry | unchanged              |
| F17                    | `execution_mode = FRESH` for every evaluated item                                                                      | unchanged              |
| F18                    | no memory, cache or index scoped more broadly than the item                                                            | unchanged              |
| F19                    | presentation order equals the precommitted V1.1 §2.5 order                                                             | unchanged              |

```
PAIR_LEAKAGE_FIREWALL := PASS  iff every item in the run passes F1–F12 and F13a–F19
                         FAIL  otherwise

```

Fail-closed at **run** granularity, unchanged from V1.1A §D.3: one failing item fails the run, and excluding it so the remainder passes is prohibited.

---

## 3. Calibration-item construction — frozen advancement

### 3.1 The gap

V1.1A §C.1 alternated unperturbed/perturbed positions and cycled `CITATION_SWAP → SEMANTIC_SLOT_HIJACK → NUMBER_FLIP`, but did not say what happens when the next ranked parent cannot host the required family — a parent with no eligible numeric span cannot host `NUMBER_FLIP`. Construction was therefore underdetermined.

### 3.2 Frozen loop

Candidates are ranked as V1.1A §C.1. `consumed` is the set of parents already used by a calibration attempt; it starts empty.

```
for attempt t = 1, 2, 3, …:

  if t is odd:                                   # unperturbed position
      p := first parent in rank order with p ∉ pool ∧ p ∉ consumed ∧ base_admissible(p)
      item := unperturbed certificate(p)

  if t is even:                                  # perturbed position
      F := CYCLE[ ((t/2) − 1) mod 3 ]            # CS → SSH → NF, prescribed, never substituted
      tested := 0
      scan parents in rank order:
          skip if p ∈ pool or p ∈ consumed or ¬base_admissible(p)
          tested += 1
          if tested > CALIB_CONSTRUCTION_HORIZON:            # = 200
              HALT("CALIBRATION_CANDIDATE_CONSTRUCTION_INFEASIBLE", position = t, family = F)
          r := frozen_constructor(F, p, label_conditional_mode(p))     # V1.1 §12.1
          if r == REJECT: continue                # p is NOT consumed; diagnostics only
          item := r; break

  consumed := consumed ∪ {p}
  send item to the expert panel (V1.1A §C.2–C.4)
  admit to the gold set per §3.4

  if t > CALIB_ADJUDICATION_HORIZON:             # = 200 attempts
      HALT("CALIBRATION_GOLD_INFEASIBLE")

```

### 3.3 Frozen answers to the underdetermined points

| questionfrozen answer                                                                                        |                                                                                                                                                                                                                                                                                                               |
| ------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| does an ineligible parent get consumed?                                                                      | **No.** Only a parent that yields a constructed item is consumed.                                                                                                                                                                                                                                             |
| does ineligibility alter the family cycle?                                                                   | **No.** `F` is a function of `t` alone.                                                                                                                                                                                                                                                                       |
| may a different family be substituted?                                                                       | **No, ever.** Substitution would make the family mix a function of eligibility, which is a property of the data.                                                                                                                                                                                              |
| does the scan cursor advance monotonically across positions?                                                 | **No.** Each position scans **from the top** of the ranking, skipping only `consumed` parents.                                                                                                                                                                                                                |
| does a parent skipped for family `F` remain eligible later under family `G`, or for an unperturbed position? | **Yes — explicitly.** Eligibility is family-specific: a parent with no numeric span fails `NUMBER_FLIP` and may be perfectly good for `CITATION_SWAP`. Permanent exclusion would discard admissible parents and make the calibration set depend on cycle order for no reason. This is why the cursor rewinds. |
| what if an item is constructed but the experts return `INDETERMINATE`?                                       | the item is discarded; **the parent stays consumed**, because re-testing it would regenerate the byte-identical item and the experts would face the same question.                                                                                                                                            |
| two horizons, distinct names                                                                                 | `CALIB_CONSTRUCTION_HORIZON = 200` (distinct unconsumed admissible parents tested within **one** position) and `CALIB_ADJUDICATION_HORIZON = 200` (total attempts across the run, V1.1A §C.5). Exceeding either halts, with distinct error identifiers.                                                       |

### 3.4 Gold-set admission and the balance constraint

V1.1A §C.5 requires `|gold| = 30` with `≥ 8 VALID_SUPPORT` and `≥ 8 INVALID_SUPPORT`. Reaching exactly 30 while satisfying both needs an admission rule, which is frozen here:

```
BUCKET_CAP = 22                                  # = 30 − 8

admit a determinate item of label L iff |bucket_L| < BUCKET_CAP
otherwise discard it (the parent remains consumed) and continue
stop when |bucket_VALID| + |bucket_INVALID| = 30

```

The cap makes the `≥ 8 / ≥ 8` floor unreachable-to-violate rather than checked-after-the-fact: neither bucket can exceed 22, so the other necessarily reaches at least 8 before the set closes. The `≥ 8` floor keeps both constant strategies at or below `22/30 = 0.733`, under the `0.80` qualification bar.

### 3.5 Diagnostics (recorded in the calibration manifest)

`n_attempts`, `n_parents_consumed`, and per family the count of parents tested and rejected by the constructor, plus the count of items returned `INDETERMINATE` by the experts. These are **descriptive only**; no threshold attaches to them and no modification of the protocol may be made on their basis.

---

## 4. Private attribution mapping — auditability repair

### 4.1 The defect

V1.1A §A.2.2 made the public `primary_responses` a **sorted** pair, and the private record `primary_pseudonyms: [p1, p2] in response order`. An auditor holding the private file has pseudonyms in unsorted order and responses in sorted order, with nothing linking them. When the two responses differ, the auditor cannot determine who said what — which is precisely the audit the private file exists to enable.

### 4.2 Corrected private record

The private attribution record carries **self-contained pairs**:

```jsonc
{
  "item_id": "<pair_id>#treatment",
  "primary": [ { "pseudonym": "<32 hex>", "response": "INVALID" },
               { "pseudonym": "<32 hex>", "response": "VALID"   } ],
  "adjudicator": { "pseudonym": "<32 hex>", "response": "INVALID" } | null
}

```

`primary` MUST be sorted ascending by `pseudonym` as UTF-8 bytes, so the file is canonical and its root reproducible. Sorting by pseudonym does not lose the mapping, because each response travels inside its own object.

```
annotator_attribution_root := SHA256( LP("ADVINT-ATTRIB-v1_1B") ‖ LP(protocol_version)
                                    ‖ m.to_bytes(8,"big")
                                    ‖ for each record in item_id order:
                                        LP(canonical_json(record)) )

```

### 4.3 Verifiable link to the public payload

An auditor holding the private file MUST be able to check, per item:

```
multiset{ r.response : r ∈ private.primary }  ==  multiset( public.primary_responses )

```

The public payload's sorted response pair is exactly the sorted multiset of the private responses, so the two files are cross-checkable without either revealing the other's secret.

### 4.4 Unchanged

The public adjudication payload remains **identity-free**: no name, email, employer, platform handle, timestamp, IP, locale, or pseudonym. `annotator_attribution_root` is still published as a bare 64-hex commitment while the file it commits to stays private, the salt custody and non-destruction rules of V1.1A §A.2.2 stand, and the privacy mechanism itself is not redesigned.

---

## 5. Preservation

Unchanged and normative: `PAIRED-REALIGN-ADJ-v1_1` · FEVER-only · 1,000 pairs / 2,000 evaluation items · 500 `SUPPORTS` / 500 `REFUTES` · 334 / 333 / 333 · `DESUPPORT` / `REALIGN` · citation-swap, semantic-slot-hijack and number-flip construction machinery · blinded treatment census adjudication · 200 control checks · 100 SSH riders · α floor 0.67 and the CP-lower gates 0.85 / 0.90 / 0.80 · no candidate replacement · `P_inv` as the primary paired population, with `VALID`treatments never counted as correct rejections and `UNRESOLVED` never in an effect denominator · the candidate → adjudication → scientific root chain and the distinct `audit_failure_root` domain tag · the calibration-gold expert protocol, question wording and blinding · fail-closed gates · no LLM in dataset construction · partition root binding and pair-level partition assignment · `expected_failed_channel` remains removed · U5 as the sole protocol-freeze blocker.

Firewall checks F1–F12, F16–F19 are unchanged. Only F13–F15 are restated, and only §§1–4 of this erratum change anything.

### 5.1 Gap status

No new gap ids. V1.1A §F stands, with U14 (harness implementation) extended in scope to cover: `MODEL_VISIBLE_FIELD_PROVENANCE` emission, the `frozen_instruction_template_sha256` and `SYSTEM_STATIC_PROMPT`registrations, and the `COMPLETE_PAYLOAD_CAPTURE` attestation.

```
BLOCKS_PROTOCOL_FREEZE       : 1   (U5)
BLOCKS_DATASET_CONSTRUCTION  : 1   (U1)
BLOCKS_SCIENTIFIC_EVALUATION : 8   (U2′, U6, U7, U3, U8, U12, U13, U14)
BLOCKS_MANUSCRIPT_REPORTING  : 2   (U9, U10)
NONBLOCKING_DIAGNOSTIC       : 2   (U4, U11)

```

---

## 6. End state

```
ADVERSARIAL_INTEGRITY_PROTOCOL_V1_1B
V1_1B_RECOMMENDED=YES
F15_SHARED_PAIR_CONTENT_ALLOWED=YES
PAIRED_EXCLUSIVE_CONTENT_LEAK_BLOCKED=YES
GENERIC_METADATA_VALUES_SUBSTRING_SCANNED=NO
MODEL_VISIBLE_FIELD_PROVENANCE_DEFINED=YES
CALIBRATION_ATTACK_INELIGIBILITY_RULE_DEFINED=YES
PRIVATE_ANNOTATOR_RESPONSE_MAPPING_AUDITABLE=YES
UNRESOLVED_PROTOCOL_FREEZE_GAPS=1

```

`UNRESOLVED_PROTOCOL_FREEZE_GAPS = 1` is unchanged: U5, the unconfirmed main-FEVER final-evaluation split. This erratum opens no new freeze blocker.