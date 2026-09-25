# ADVERSARIAL_INTEGRITY_PROTOCOL_V1

Candidate protocol for human review. Nothing in this document has been executed.
No dataset has been generated, no source file has been downloaded, and no code
has been written. Every quantity below is either a frozen decision or is marked
explicitly unresolved.

The protocol is written so that two independent implementers, given only this
document and the pinned source files, produce byte-identical canonical records
and the same 1,000-example pool root.

---

## 0. Scientific objective, and the distinction that drives everything

The eighth dataset is **not** another factual-classification benchmark. It is a
controlled stress test of the *certificate* — of whether the evidence actually
supplied validly grounds the verdict actually asserted.

Two labels are therefore kept rigorously separate and are never merged:

| field | what it is about | who can change it |
|---|---|---|
| `source_world_label` | the truth status of the *original* FEVER claim in the world | the world; inherited from FEVER, never re-derived |
| `certificate_validity_target` | whether the *supplied evidence* validly grounds the *asserted verdict* for the *perturbed* instance | our perturbation, by construction |

The central phenomenon this dataset isolates:

> A claim can remain world-true while its supplied evidence becomes
> certificate-invalid.

Citation swap is the pure instance of this. The claim is byte-identical, its
world truth is untouched, and yet the certificate must be rejected because the
evidence no longer grounds anything. A system that answers "is this claim true?"
passes; a system that answers "is this certificate valid?" is the only one that
can succeed. That gap is the entire scientific point of the cell.

### 0.1 The task framing that makes REFUTES usable

FEVER contains both `SUPPORTS` and `REFUTES` claims. A naive framing — "the
certificate asserts the claim" — cannot use `REFUTES` at all, because the
asserted proposition would be false before we perturb anything.

**Frozen framing.** Each instance presents the system with a *verification task*:

```
Given claim C and evidence E, emit a certificate whose claim node is the verdict
    v ∈ {SUPPORTED, REFUTED}
together with the evidence node E that grounds v.
```

The certificate's asserted proposition is therefore `v(C) grounded in E`, not `C`
itself. This is what makes `REFUTES` parents first-class: a correct certificate
over a `REFUTES` parent asserts `REFUTED(C)` and is validly grounded by evidence
that contradicts `C`.

No claim is negated, rewritten, or templated. `C` is passed through verbatim
(subject only to the attack transformation), so no generative step enters the
pipeline.

---

## 1. Source dataset

```
SOURCE_DATASET = FEVER_ONLY
```

The phrase "primarily FEVER" is retired. There is exactly one source corpus and
exactly one derived corpus (the Wikipedia dump FEVER ships with).

### 1.1 Exact canonical source

**FEVER 1.0** (Thorne, Vlachos, Christodoulopoulos, Mittal, NAACL 2018), taken
from the official shared-task distribution rather than any mirror:

| artifact | file | role |
|---|---|---|
| claims | `train.jsonl` | parent claims, labels, evidence pointers |
| corpus | `wiki-pages.zip` → `wiki-001.jsonl` … `wiki-109.jsonl` | the June 2017 Wikipedia processed dump; the only text source for evidence sentences |

**Why the official distribution and not a HuggingFace `datasets` build.** The
`datasets` loaders normalise fields, occasionally re-key evidence, and are
versioned by loader script rather than by content. The shared-task JSONL files
are the artifacts the FEVER label definitions were written against, and they can
be pinned by content hash. Determinism requires pinning the bytes, not a library
release.

### 1.2 Exact immutable revision

FEVER 1.0 has no upstream commit identifier. The immutable revision is therefore
**defined by content hash**, established once in a pinning ceremony and frozen
thereafter:

```
SOURCE_REVISION := SHA256( LP("ADVINT-SRCREV-v1")
                         ‖ LP("fever-1.0")
                         ‖ LP(sha256(train.jsonl))
                         ‖ LP(sha256(wiki-pages.zip)) )
```

where `LP(x)` is the length-prefix encoding of §7.1.

```
SOURCE_REVISION = UNRESOLVED_PENDING_PIN
  sha256(train.jsonl)    = UNRESOLVED
  sha256(wiki-pages.zip) = UNRESOLVED
```

These are unresolved **by constraint, not by oversight**: this session is bound
by `BENCHMARK_DATASET_EXECUTION=0` and `HUGGINGFACE_MODEL_DOWNLOADS=0`, so no
file may be fetched and no hash may be claimed. Asserting a hash I have not
computed would be exactly the failure mode the project forbids.

**Pinning ceremony** (one-time, human-run, before any construction):

1. Download `train.jsonl` and `wiki-pages.zip` from the official FEVER
   distribution.
2. Record `sha256` of each file, byte-exact, unmodified, uncompressed for
   `train.jsonl` and *as the zip archive* for `wiki-pages.zip`.
3. Record the extracted per-shard hashes `sha256(wiki-NNN.jsonl)` for all 109
   shards, so a corrupted extraction is detectable independently of the archive.
4. Write all hashes into `advint_source_pin.json`.
5. Compute `SOURCE_REVISION` as above and freeze it in the protocol constant
   block.
6. Every construction run re-verifies all hashes before reading a single record
   and aborts on any mismatch. There is no "warn and continue" path.

### 1.3 Exact source split

```
SOURCE_SPLIT = train        (FEVER 1.0 train.jsonl)
```

**Why `train`, and why this is the parent-disjointness mechanism.** The main
FEVER final-evaluation cell is evaluated on `shared_task_dev`. FEVER's `dev`
pool is `shared_task_dev.jsonl` (19,998 claims), and the "paper" splits
`paper_dev.jsonl` / `paper_test.jsonl` are *partitions of that same dev pool* —
they are subsets of `shared_task_dev`, not alternatives to it. Drawing
adversarial parents from any `dev`-derived file would therefore create parent
overlap with the main cell by construction.

`train` is disjoint from `shared_task_dev` at the corpus level, so
parent-disjointness is **structural rather than filtered**. It is additionally
*verified* (§13.3) by explicit set intersection, because a structural argument
that is never checked is an assumption.

> **Unresolved gap U5.** I am asserting that the main FEVER cell evaluates on
> `shared_task_dev`. I am forbidden from re-auditing the repository to confirm
> it. A human must confirm this before the protocol is frozen. If the main cell
> uses `train`, this section must be re-decided, and the correct answer becomes
> `paper_test` with an explicit id-level exclusion of the main cell's parents.

### 1.4 Accepted and excluded labels

```
ACCEPTED_LABELS = {SUPPORTS, REFUTES}
EXCLUDED_LABELS = {NOT ENOUGH INFO}
```

**Why NEI is excluded, on principle rather than convenience.** In FEVER, `NOT
ENOUGH INFO` claims carry no evidence — the evidence field is null. There is
therefore *no support relation to corrupt*. Every attack family in this protocol
perturbs either the evidence or the claim's relation to it; on an NEI item all
three are undefined operations. Including NEI would mean silently inventing a
support relation in order to break it, which would make the
`certificate_validity_target` a property of our invention rather than of FEVER.

### 1.5 Additional eligibility filters (applied to every parent, all families)

A parent is admissible only if **all** hold:

- `label ∈ {SUPPORTS, REFUTES}`;
- at least one annotation set in `evidence` contains **exactly one** sentence
  pointer (single-hop). The canonical evidence unit is the lexicographically
  first such set, ordered by `(page, sentence_id)` with `page` compared as UTF-8
  bytes and `sentence_id` as an integer;
- the referenced `page` exists in the pinned `wiki-pages` corpus and the
  referenced `sentence_id` exists in that page's `lines` field;
- the resolved evidence sentence, after §11.1 text normalisation, is non-empty
  and has ≥ 5 whitespace-delimited tokens;
- the claim, after §11.1 normalisation, has between 5 and 60 whitespace-delimited
  tokens inclusive;
- the claim's normal form (§12.1) has not already been claimed by an accepted
  parent.

**Why single-hop.** Multi-sentence FEVER annotations make "the evidence unit
that citation swap replaces" ambiguous — replace one sentence or all of them? —
and the answer changes the attack's strength. Single-hop makes the evidence unit
a single, unambiguous object. This narrows realism; it is recorded as a
limitation in §17 and as the natural V2 extension.

---

## 2. Pool

```
POOL_SIZE = 1000
SOURCE_SUPPORTS = 500
SOURCE_REFUTES  = 500
ONE_ATTACK_PER_INSTANCE = YES
ONE_PARENT_USED_AT_MOST_ONCE = YES
```

Every adversarial example has exactly one parent FEVER example, and every parent
contributes at most one adversarial example. This is enforced by a global
`claimed_parents` set (§8.3), not by post-hoc deduplication.

**Why one parent at most once matters statistically.** If a parent contributed
to two records, those two records would share claim text, evidence, and topic;
their errors would be correlated; and every interval computed over the pool would
be anti-conservative by an unmodelled cluster factor. One-parent-one-record makes
the 1,000 records exchangeable at the parent level, which is the assumption every
downstream interval in the manuscript already relies on.

---

## 3. Attack allocation

The proposed quotas are **adopted exactly as specified**:

```
CITATION_SWAP        = 334      (SUPPORTS 167, REFUTES 167)
SEMANTIC_SLOT_HIJACK = 333      (SUPPORTS 167, REFUTES 166)
NUMBER_FLIP          = 333      (SUPPORTS 166, REFUTES 167)
                       ----                ---          ---
                       1000                500          500
```

Arithmetic check: 334+333+333 = 1000; SUPPORTS 167+167+166 = 500; REFUTES
167+166+167 = 500.

**Why adopt rather than revise.** Near-equal thirds give each family a
denominator of ~333, whose Clopper–Pearson half-width at a 10% event rate is
about ±3.3 points — small enough to distinguish families that differ by 10 points
or more, which is the resolution the cell needs. Equal source-label split within
each family is what makes the *target asymmetry* in §3.1 into a designed control
rather than an accident.

### 3.1 The target is deliberately non-uniform, and this is the design

This is the most consequential scientific decision in the protocol, so it is
stated plainly rather than smoothed over.

**Evidence-side perturbation breaks grounding regardless of parent label.**
Replace the evidence and the certificate is unfounded whether the verdict was
`SUPPORTED` or `REFUTED`.

**Claim-side perturbation breaks grounding only for `SUPPORTS` parents.** For a
`SUPPORTS` parent, the evidence attested the very element we perturb, so support
is destroyed. For a `REFUTES` parent, the evidence contradicts the claim; after
we perturb a *different, non-attested* element (§4.3, §6.3), the claim remains
false and the same evidence still grounds the same `REFUTED` verdict. The
correct certificate is unchanged.

| cell | n | `source_truth_changed` | `support_relation_changed` | `certificate_validity_target` | `expected_failed_channel` |
|---|---|---|---|---|---|
| CITATION_SWAP × SUPPORTS | 167 | false | true | `INVALID_SUPPORT` | `CheckFail` |
| CITATION_SWAP × REFUTES | 167 | false | true | `INVALID_SUPPORT` | `CheckFail` |
| SEMANTIC_SLOT_HIJACK × SUPPORTS | 167 | true | true | `INVALID_SUPPORT` | `CheckFail` |
| SEMANTIC_SLOT_HIJACK × REFUTES | 166 | false | false | `VALID_SUPPORT` | `NONE` |
| NUMBER_FLIP × SUPPORTS | 166 | true | true | `INVALID_SUPPORT` | `CheckFail` |
| NUMBER_FLIP × REFUTES | 167 | false | false | `VALID_SUPPORT` | `NONE` |

```
INVALID_SUPPORT : 167+167+167+166 = 667
VALID_SUPPORT   :         166+167 = 333
```

**Why 333 controls are a feature, not dilution.** A stress test consisting only
of attacks cannot measure false rejection, and a system that rejects everything
would score perfectly. Worse, if every perturbed item were invalid, "detect that
something was perturbed" would be a complete solution — the dataset would measure
perturbation detection, not certificate validity.

The 333 controls are **matched**: identical attack mechanism, identical
transformation code path, identical surface statistics, opposite target. A
detector of perturbation therefore scores at chance on the SSH and NF cells. Only
a system that reasons about whether the evidence grounds *this verdict* separates
them. That is precisely the capability the cell exists to measure.

**Forcing all six cells to `INVALID_SUPPORT` would be false**, and the
instruction not to do so is correct: for a `REFUTES` parent whose non-attested
slot we perturbed, the supplied evidence genuinely does still ground the verdict.
Labelling it invalid would inject systematic label noise into a third of the pool.

### 3.2 The anti-degenerate guarantee

Within SSH and NF, `certificate_validity_target` is a deterministic function of
`source_world_label`. A degenerate strategy — "accept whenever the evidence
contradicts the claim, reject whenever it merely fails to support" — would
therefore score 100% on those 666 items.

`CITATION_SWAP × REFUTES` (167 items) is the cell that defeats it. There the
evidence neither supports nor contradicts the claim (it is about a different
entity), the parent label is `REFUTES`, and the target is `INVALID_SUPPORT`. The
degenerate strategy accepts all 167 and is wrong on all 167.

This cell is therefore **load-bearing** and must never be dropped, downsampled,
or pooled away. §15 requires it to be reported separately.

---

## 4. SEMANTIC_SLOT_HIJACK (replaces `paraphrase_hijack`)

The term `paraphrase_hijack` is **retired**. Paraphrase is a generative operation;
no deterministic, non-LLM, byte-reproducible paraphraser exists, so the family
could not have satisfied `LLM_USED_IN_DATASET_CONSTRUCTION=NO` and
`DETERMINISTIC_RECONSTRUCTION=YES` simultaneously.

`SEMANTIC_SLOT_HIJACK` preserves the claim's lexical and syntactic structure
exactly, changing a single contiguous span, and is fully deterministic.

### 4.1 Distinctness from the other two families

| | perturbs | granularity | evidence topicality after attack |
|---|---|---|---|
| CITATION_SWAP | evidence | coarse (whole evidence unit) | evidence is about a **different entity** |
| SEMANTIC_SLOT_HIJACK | claim | fine (one entity span) | evidence remains **on-topic** for the claim's primary entity |
| NUMBER_FLIP | claim | fine (one numeric span) | evidence remains **on-topic** |

SSH and NUMBER_FLIP are both fine-grained claim-side attacks but operate on
disjoint span types: SSH never touches a span the numeric parser accepts, and
NUMBER_FLIP never touches a gazetteer entity. The exclusion is enforced, not
assumed (§4.3, rule S5).

### 4.2 Entity gazetteer and type system (no NER model, no LLM)

**Gazetteer G.** Built once from the pinned `wiki-pages` corpus:

- for every page, take the `id` field (the Wikipedia title);
- surface form `surf(page)` = title with `_` → space, then §11.1 normalisation,
  then removal of a trailing parenthetical disambiguator matching
  `\s*\([^()]*\)$`;
- discard surface forms shorter than 3 characters or longer than 60 characters;
- discard surface forms that are entirely lowercase after normalisation (these
  are overwhelmingly common nouns rather than entities);
- discard surface forms appearing as the title of more than one page (ambiguous);
- `G` maps `surf(page) → page_id`, and is stored sorted by `surf` byte-ascending.

**Copula type signature τ.** For a page `p`, take `lines` sentence 0, apply
§11.1 normalisation, and match the frozen pattern

```
^\s*<surf>\s*(\([^()]*\)\s*)?(is|was|are|were)\s+(a|an|the)\s+(?P<type>[^,.;]{1,80})
```

`τ(p)` is the captured `type` group, lowercased, whitespace-collapsed, with a
trailing period removed. If the pattern does not match, `τ(p)` is undefined and
`p` is not usable as a donor or as a typed target.

`τ_head(p)` is the last whitespace-delimited token of `τ(p)`, with a trailing
`s` removed if the token has ≥ 4 characters.

**Why a copula signature rather than an NER label.** It requires no model, is a
pure function of the pinned corpus, and produces a far finer type than a 4-way
NER tag: `"american film director"` and `"french commune"` are distinguishable,
where `PERSON`/`LOCATION` would not be. Coarseness is bounded and measurable.

### 4.3 Eligible slots

Let `P` be the parent, `C` its claim, `E` its evidence sentence, and
`page(E)` the evidence page. Let `prim = surf(page(E))` be the **primary entity**.

A candidate slot is a maximal-length match of a gazetteer surface form against
the normalised claim, found by leftmost-longest scan over `G`. A slot is
**eligible** iff:

- **S1** the matched span is not `prim` and does not overlap any occurrence of
  `prim` in `C`. *(The primary entity is protected so the evidence stays
  topically relevant — this is what separates SSH from CITATION_SWAP.)*
- **S2** `τ(page_of_slot)` is defined.
- **S3** the span does not overlap any span accepted by the §6.1 numeric parser.
- **S4** the span is not the entire claim and leaves ≥ 3 tokens on at least one
  side.
- **S5** the span's surface form is not a case-insensitive substring of `prim`,
  and `prim` is not a substring of it.
- **S6 (label-conditional, load-bearing rule):**
  - if `source_world_label = SUPPORTS`: the span's surface form **must appear**
    in `E` (case-normalised). The evidence attests this element, so replacing it
    destroys support.
  - if `source_world_label = REFUTES`: the span's surface form **must not appear**
    in `E`. This is the false element; replacing it with another incompatible
    value keeps the claim false and keeps the same evidence a valid refutation.

**Slot selection when several are eligible.** Rank by, in order: (a) span start
offset ascending; (b) span length descending; (c) surface form byte-ascending.
Take the first. Deterministic, and independent of any scoring heuristic.

### 4.4 Replacement candidate pool and constraints

For chosen slot `s` with page `p_s`, the donor pool is built in two tiers,
exhausted in order:

- **Tier 1 — exact type match:** `{q ∈ G : τ(q) = τ(p_s)}`
- **Tier 2 — head-noun match:** `{q ∈ G : τ_head(q) = τ_head(p_s)}`

A donor `q` is admissible iff:

- **D1** `q ≠ p_s` and `q ∉ pages(E)`;
- **D2** `surf(q)` does not appear in `C` (case-normalised);
- **D3** `surf(q)` does not appear in `E` (case-normalised) — **the
  accidental-truth guard**: if the donor value were the value the evidence
  attests, a `SUPPORTS` attack would remain supported and a `REFUTES` control
  would flip to true;
- **D4** neither `surf(q)` nor `surf(p_s)` is a case-insensitive substring of the
  other (alias guard);
- **D5** `surf(q)` differs from `surf(p_s)` in §12.1 normal form;
- **D6** donor reuse cap: `q` has been used as a donor in fewer than
  `DONOR_REUSE_CAP = 3` accepted records so far.

**Deterministic donor ranking.** Among admissible donors within the active tier,
rank by `SHA256(LP("ADVINT-DONOR-v1") ‖ LP(protocol_version) ‖ LP(instance_id)
‖ LP(page_id(q)))` byte-ascending; tie-break by `page_id(q)` byte-ascending.
Take the first. *(Hash-ordering rather than similarity-ordering is deliberate
here: a similarity-ranked donor would make SSH systematically "nearest plausible
alternative", which is itself a learnable artifact.)*

### 4.5 Transformation, validation, and invalid-case behaviour

The perturbed claim is `C` with the slot span replaced by `surf(q)`, byte-exact,
preserving all surrounding characters including whitespace and punctuation. No
capitalisation fix-up, no article agreement, no re-tokenisation — any such
repair would be a generative step.

Validation (all must hold, else reject and advance):
- `perturbed_claim ≠ original_claim`;
- `perturbed_claim` normal form (§12.1) is not already present in the pool as an
  original or perturbed claim;
- `perturbed_claim` token count is within the §1.5 bounds.

**Exhaustion:** tier 1 exhausted → tier 2; tier 2 exhausted → the *parent* is
rejected for this family and the scan advances to the next parent in rank order.
No resampling, ever.

### 4.6 Expected certificate target

Per §3.1: `SUPPORTS` parent → `INVALID_SUPPORT`, channel `CheckFail`;
`REFUTES` parent → `VALID_SUPPORT`, channel `NONE`.

---

## 5. CITATION_SWAP

### 5.1 Frozen semantics

- the claim is **byte-identical** to the parent's claim;
- `source_world_label` is preserved verbatim as metadata and is not re-derived;
- the evidence unit is replaced by a deterministically selected donor sentence
  that does not ground the verdict;
- `certificate_validity_target = INVALID_SUPPORT` for both parent labels.

This is the family that isolates the protocol's central phenomenon: nothing
about the world changed, and the certificate is nonetheless invalid.

### 5.2 Evidence unit replaced

Exactly the single canonical evidence sentence of §1.5, identified by
`(page, sentence_id)`. Both the page pointer and the sentence text are replaced,
so `perturbed_evidence` is a complete, self-consistent citation to a real
Wikipedia sentence — not a mangled one. A malformed citation would be detectable
by well-formedness checking rather than by reasoning, which would make the attack
trivial.

### 5.3 Donor corpus and restrictions

Donors are drawn from the **evidence sentences of other admissible parents in the
same `SOURCE_SPLIT`**, not from arbitrary Wikipedia sentences.

*Why:* a donor that is itself some parent's evidence sentence is guaranteed to be
a well-formed, encyclopaedic, evidence-shaped sentence. Arbitrary corpus
sentences include list fragments, stubs and headers, which would give citation
swap a detectable stylistic signature.

Donor `d` (with sentence text `txt(d)`, page `page(d)`, parent `par(d)`) is
admissible iff:

- **C1** `par(d) ≠ P` — same-parent prohibition;
- **C2** `page(d) ∉ pages(E)` — the donor is not from the parent's evidence page;
- **C3** `prim` does **not** appear in `txt(d)` (case-normalised). This is the
  non-support guarantee: a sentence that never mentions the claim's primary
  entity cannot ground a verdict about it;
- **C4** donor reuse cap: `txt(d)` has been used in fewer than
  `DONOR_REUSE_CAP = 3` accepted records;
- **C5** `txt(d) ≠ E` byte-wise;
- **C6** `txt(d)` has ≥ 5 and ≤ 80 whitespace-delimited tokens.

### 5.4 Entity-overlap rule — a deliberate inversion of the suggested default

The brief proposed a **zero** entity-overlap rule. I am adopting the opposite,
and the reason is a construct-validity failure that zero-overlap would create.

If a donor shares no entity with the claim, then the trivial string-matching
baseline "reject whenever the claim's entities are absent from the evidence"
detects **334/334** citation swaps. The cell would measure string matching.

The rule is therefore inverted, with the primary entity still protected:

- **the primary entity `prim` must be absent** from the donor (rule C3, which
  guarantees non-support);
- **at least one non-primary gazetteer entity of the claim should be present** in
  the donor.

Implemented as two tiers, exhausted in order:

- **Tier 1 (preferred):** `txt(d)` contains ≥ 1 gazetteer entity that also occurs
  in `C` and is not `prim`;
- **Tier 2 (fallback):** no such entity-overlap requirement.

The realised tier-1 fraction is recorded in the manifest as
`citation_swap_tier1_fraction`, because it is the direct quantitative measure of
how far the cell has escaped the string-matching shortcut. If it is low, the
protocol has partly failed and the human reviewer must see that number.

### 5.5 Lexical similarity rule — hard negatives, computed without floats

Donors are ranked by **descending** lexical similarity to the claim, so the
selected donor is the *most confusable* admissible sentence rather than a random
one. A random donor produces an obviously off-topic citation; a maximally similar
one produces a citation that looks right and is not.

Similarity is token Jaccard over content tokens:

- tokenise the §11.1-normalised text on `\s+`;
- lowercase; strip leading/trailing characters in `.,;:!?"'()[]`;
- drop tokens in the frozen `STOPWORDS` list (§11.4);
- drop tokens equal to any surface form of `prim`;
- `A` = token set of `C`, `B` = token set of `txt(d)`;
- `J = |A ∩ B| / |A ∪ B|`.

**Floats are never used.** `J` is carried as the integer pair `(|A∩B|, |A∪B|)`
and comparisons are by cross-multiplication:
`J(d1) > J(d2) ⟺ n1·u2 > n2·u1`. This removes every platform-dependent
floating-point ordering question, which is a real byte-equivalence hazard.

Admissibility additionally requires `J ≥ 1/10` — i.e. `10·n ≥ u` — so that
degenerate, wholly unrelated donors are excluded.

### 5.6 Deterministic ranking and tie-breaking

Within the active tier, order donors by:

1. `J` descending, compared by cross-multiplication;
2. `SHA256(LP("ADVINT-DONOR-v1") ‖ LP(protocol_version) ‖ LP(instance_id)
   ‖ LP(page_id(d)) ‖ LP(str(sentence_id(d))))` byte-ascending;
3. `(page_id(d), sentence_id(d))` byte-/integer-ascending.

Take the first admissible donor.

### 5.7 Failure and exhaustion handling

Tier 1 exhausted → tier 2. Tier 2 exhausted → the parent is rejected for this
family and the scan advances. No stochastic fallback.

### 5.8 Validation

- `perturbed_claim == original_claim`, byte-identical (asserted, not assumed);
- `perturbed_evidence ≠ original_evidence`;
- `prim ∉ perturbed_evidence` (re-checked after selection);
- the `(page, sentence_id)` pointer resolves in the pinned corpus to exactly the
  stored text.

No embeddings and no language model are used anywhere in this family.

---

## 6. NUMBER_FLIP

Evidence is **unchanged**. Only the claim's numeric span moves.

### 6.1 Numeric parser (frozen)

Applied to the §11.1-normalised claim. A numeral is a match of

```
NUM := (?<![0-9A-Za-z.])(?P<sign>[-−])?(?P<int>\d{1,3}(,\d{3})+|\d+)(?P<frac>\.\d+)?(?![0-9A-Za-z])
```

Matches are enumerated left to right, non-overlapping, leftmost-longest.

`value(span)` = exact decimal, parsed by removing `,`, mapping `−` (U+2212) to
`-`, and reading as a base-10 decimal. **All arithmetic is exact decimal
arithmetic; binary floating point is never used.**

`precision(span)` = number of digits after `.` in `frac`, or 0 if absent.
`grouped(span)` = true iff `int` contained a `,`.

### 6.2 Eligible and excluded classes

A span's class is decided by the frozen context tests below, applied in order;
the first that matches wins.

**Excluded (span is ineligible):**

| class | test |
|---|---|
| ORDINAL | immediately followed by `st`, `nd`, `rd`, `th` (case-insensitive) |
| CITATION | the span is enclosed in `[ ]` |
| VERSION / ID | preceded within 2 characters by `#`, `No.`, `no.`, `v`, `V`, `version `, or the span contains ≥ 2 `.` |
| DATE / YEAR | integer span with `precision = 0`, `1000 ≤ value ≤ 2999`, no grouping, and **not** immediately followed by a `UNIT_LEXICON` unit or `%`; **or** any span within 3 whitespace-delimited tokens of a month name in `MONTHS` (§11.4); **or** the span is part of a `\d{1,2}/\d{1,2}/\d{2,4}` or `\d{4}-\d{2}-\d{2}` pattern |
| PHONE / POSTAL | the span matches `\d{5}(-\d{4})?` immediately preceded by a `STATE_ABBREV`, or matches `\(?\d{3}\)?[- ]\d{3}-\d{4}` |

**Eligible:**

| class | test |
|---|---|
| MONEY | immediately preceded by one of `$ £ € ¥` (optionally with a space), or immediately followed by one of `dollars, euros, pounds, yen, USD, EUR, GBP, JPY` |
| PERCENT | immediately followed by `%` or ` percent` or ` per cent` |
| MEASUREMENT | immediately followed by a unit token from `UNIT_LEXICON` (§11.4) |
| CARDINAL | none of the above matched and the span was not excluded |

### 6.3 Target selection — the load-bearing rule

Let `EVNUMS` be the set of `value(span)` over all numerals in `E` (parsed with
the same parser, exclusions not applied — we want every number the evidence
states).

A span is a valid **target** iff it is eligible *and*:

- if `source_world_label = SUPPORTS`: `value(span) ∈ EVNUMS`. The evidence
  attests this quantity, so changing it must destroy support.
- if `source_world_label = REFUTES`: `value(span) ∉ EVNUMS`. This is the false
  quantity; changing it to another wrong quantity keeps the claim false and keeps
  the evidence a valid refutation.

**If no valid target exists, the parent is ineligible for NUMBER_FLIP.** This is
the eligibility predicate that guarantees the perturbation is load-bearing rather
than cosmetic; a flip on a number the evidence never mentions would change
nothing about the support relation and would silently pollute the cell.

Among valid targets, select by: (a) class priority
`MONEY > PERCENT > MEASUREMENT > CARDINAL`; (b) span start offset ascending;
(c) span length descending. Take the first.

### 6.4 Sign rule (hash-governed)

```
H = SHA256( LP("ADVINT-NUMSIGN-v1") ‖ LP(protocol_version) ‖ LP(instance_id)
          ‖ LP(str(span_start)) )
s = +1 if (H[31] & 0x01) == 0 else −1
```

`H[31]` is the final byte of the 32-byte digest. Using a hash rather than a
constant direction prevents "every number was inflated by 10%" from becoming a
learnable artifact.

### 6.5 Magnitude, rounding, and formatting

```
raw  = sign(n) · ( |n| · (1 + s/10) )          # exact decimal
n'   = round_half_up(raw, precision(span))
```

- **magnitude form** is used, so a negative quantity moves 10% further from zero
  when `s = +1`, matching the intuitive reading of "±10%";
- `round_half_up` is **ROUND_HALF_UP**, explicitly *not* banker's rounding.
  Ties go away from zero. This single choice is a known byte-equivalence
  divergence point between implementations and is therefore frozen here;
- **integers** (`precision = 0`) round to an integer;
- **decimals** keep exactly `precision(span)` fractional digits, zero-padded if
  the rounding produced fewer;
- **grouping** is preserved: if `grouped(span)`, re-insert `,` every three digits
  in the integer part, right to left;
- the sign character, currency symbol, unit token and all surrounding whitespace
  are preserved byte-exactly; only the numeral's digits, `,` and `.` are rewritten.

### 6.6 Degenerate cases (all frozen)

| case | rule |
|---|---|
| `n = 0` | ineligible (0 is a fixed point of the transform) |
| `precision = 0` and `\|n\| < 5` | ineligible — `0.1·\|n\| < 0.5` would round back |
| `n' = n` after rounding | flip `s` and recompute once; if still equal, the **span** is ineligible → try the next valid target; if none, the **parent** is ineligible |
| PERCENT with `0 ≤ n ≤ 100` and `n' > 100` | force `s = −1` and recompute; if that fails any other rule, span ineligible |
| PERCENT and `n' < 0` | span ineligible |
| negative `n` | magnitude form above; sign preserved |
| `n' ∈ EVNUMS` | **span ineligible** — the accidental-truth guard. Without it, a `REFUTES` control could land on the true value and silently flip to `VALID`→`INVALID`, and a `SUPPORTS` attack could remain supported |

### 6.7 Validation and expected target

- `perturbed_evidence == original_evidence`, byte-identical;
- `perturbed_claim ≠ original_claim`, differing in exactly one contiguous span;
- `value(new span) ≠ value(old span)` and `value(new span) ∉ EVNUMS`.

`SUPPORTS` parent → `INVALID_SUPPORT`, `CheckFail`.
`REFUTES` parent → `VALID_SUPPORT`, `NONE`.

---

## 7. Deterministic source ranking

No iterator sampling, no shuffling, no RNG of any kind appears in this protocol.

### 7.1 Byte serialization (frozen)

```
LP(x) := len(utf8(x)).to_bytes(8, "big") ‖ utf8(x)
```

Length-prefixing rather than delimiter-joining is deliberate: with a separator
such as `0x1F`, the field pairs `("ab","c")` and `("a","bc")` could be made to
collide by a field that itself contains the separator. Length prefixes make the
encoding injective without any escaping rule. `0x1F` is **not** used as a
separator anywhere in a hashed input. This matches the `PCG-CAS-v1` discipline
already in use, so the two hashing schemes cannot be confused.

### 7.2 Rank key

```
rank_key(source_id, attack_family) :=
  SHA256( LP("ADVINT-RANK-v1")
        ‖ LP(protocol_version)      # "ADVERSARIAL_INTEGRITY_PROTOCOL_V1"
        ‖ LP(source_revision)       # hex, lowercase, from §1.2
        ‖ LP(source_split)          # "train"
        ‖ LP(source_id)             # FEVER integer id, decimal, no padding
        ‖ LP(attack_family) )       # "CITATION_SWAP" | "SEMANTIC_SLOT_HIJACK" | "NUMBER_FLIP"
```

Ordering is **byte-ascending on the 32-byte digest**; ties break on `source_id`
compared as UTF-8 bytes. Because the family is inside the key, the three
family-specific pools are ranked **independently** — a parent that is near the
front for `NUMBER_FLIP` occupies an unrelated position for `CITATION_SWAP`.

### 7.3 Termination

Each of the six cells scans a finite ranked list once, without backtracking, and
stops at its quota. Construction therefore terminates in at most
`3 × |admissible parents|` candidate evaluations.

---

## 8. Invalid candidate handling

### 8.1 Per family

| family | eligibility predicate | rejection conditions | next candidate | exhaustion |
|---|---|---|---|---|
| CITATION_SWAP | §1.5 + ≥1 admissible donor at tier 1 or 2 | C1–C6 fail; `J < 1/10`; validation §5.8 fails | next donor in §5.6 order; then next tier; then next parent | cell fails |
| SEMANTIC_SLOT_HIJACK | §1.5 + ≥1 eligible slot (S1–S6) + ≥1 admissible donor | D1–D6 fail; validation §4.5 fails | next donor; then tier 2; then next eligible slot; then next parent | cell fails |
| NUMBER_FLIP | §1.5 + ≥1 valid target (§6.3) | §6.6 degenerate cases; validation §6.7 fails | next valid target; then next parent | cell fails |

### 8.2 No stochastic resampling

There is no random fallback, no seed, and no "retry with jitter". Every advance
is to the next element of a pre-computed deterministic order.

### 8.3 Cell processing order (and why it is this order)

```
1. NUMBER_FLIP        × SUPPORTS   (166)
2. NUMBER_FLIP        × REFUTES    (167)
3. SEMANTIC_SLOT_HIJACK × SUPPORTS (167)
4. SEMANTIC_SLOT_HIJACK × REFUTES  (166)
5. CITATION_SWAP      × SUPPORTS   (167)
6. CITATION_SWAP      × REFUTES    (167)
```

Parents are claimed globally as they are accepted, so order determines who gets
scarce parents. `NUMBER_FLIP` has by far the strictest eligibility — it needs an
eligible numeric span standing in the correct relation to the evidence — so it
runs first. `CITATION_SWAP` needs only a claim and an evidence sentence, so it
runs last. Reversing this order would starve `NUMBER_FLIP` and fail construction
even though a valid assignment exists.

### 8.4 Failure is failure

If any cell exhausts its ranked pool before meeting its quota, **construction
fails** with an error naming the cell, the quota, the count reached, and the
number of candidates rejected per rejection reason. There is no partial pool, no
quota relaxation, and no substitution from another cell.

---

## 9. Target schema (canonical record)

```jsonc
{
  "dataset_id":                 "adversarial_integrity",
  "protocol_version":           "ADVERSARIAL_INTEGRITY_PROTOCOL_V1",
  "instance_id":                "advint-v1:<parent_id>:<attack_family>",
  "source_dataset":             "FEVER_ONLY",
  "source_revision":            "<64 hex chars>",
  "source_split":               "train",
  "parent_id":                  "<FEVER id, decimal string>",
  "source_world_label":         "SUPPORTS" | "REFUTES",

  "original_claim":             "<string, verbatim after §11.1>",
  "original_evidence": {
    "page":                     "<wiki page id>",
    "sentence_id":              <int>,
    "text":                     "<string>"
  },

  "attack_family":              "CITATION_SWAP" | "SEMANTIC_SLOT_HIJACK" | "NUMBER_FLIP",
  "attack_parameters":          { /* family-specific, see below */ },

  "perturbed_claim":            "<string>",
  "perturbed_evidence": {
    "page":                     "<wiki page id>",
    "sentence_id":              <int>,
    "text":                     "<string>"
  },

  "source_truth_changed":       true | false,
  "support_relation_changed":   true | false,
  "certificate_validity_target":"INVALID_SUPPORT" | "VALID_SUPPORT",
  "expected_failed_channel":    "CheckFail" | "NONE",

  "partition":                  "SMOKE" | "CHECKER_CALIBRATION" | "PILOT" | "FINAL",
  "record_sha256":              "<64 hex chars>"
}
```

`attack_parameters` by family:

```jsonc
// CITATION_SWAP
{ "donor_parent_id": "...", "donor_page": "...", "donor_sentence_id": 0,
  "tier": 1 | 2, "jaccard_num": <int>, "jaccard_den": <int>,
  "primary_entity": "..." }

// SEMANTIC_SLOT_HIJACK
{ "slot_span_start": <int>, "slot_span_end": <int>,
  "slot_original": "...", "slot_replacement": "...",
  "slot_page": "...", "donor_page": "...",
  "type_signature": "...", "tier": 1 | 2, "primary_entity": "..." }

// NUMBER_FLIP
{ "span_start": <int>, "span_end": <int>, "numeric_class": "MONEY" | "PERCENT" | "MEASUREMENT" | "CARDINAL",
  "original_value": "<exact decimal string>", "perturbed_value": "<exact decimal string>",
  "sign": 1 | -1, "precision": <int>, "grouped": true | false }
```

### 9.1 Which attacks alter world truth, and which do not

Restating §3.1 because it is the field most likely to be mislabelled:

- **`source_truth_changed = false`** for all 334 `CITATION_SWAP` records (the
  claim is untouched) and for the 333 `REFUTES` control records (the claim stays
  false).
- **`source_truth_changed = true`** only for the 333 `SUPPORTS` records under
  `SEMANTIC_SLOT_HIJACK` and `NUMBER_FLIP`, where a world-true claim is made
  world-false.
- **`support_relation_changed`** tracks the certificate, not the world, and is
  `true` exactly on the 667 `INVALID_SUPPORT` records.

The two fields are therefore genuinely independent: 334 records have
`source_truth_changed = false` together with `support_relation_changed = true`.
That set is the operational definition of the phenomenon this dataset exists to
measure.

---

## 10. Stable IDs

```
instance_id := "advint-v1:" ‖ parent_id ‖ ":" ‖ attack_family
```

Deterministic, human-readable, and collision-free by construction: `parent_id` is
unique in the source split and `ONE_PARENT_USED_AT_MOST_ONCE = YES` means the
pair can occur at most once. The family is retained in the id even though it is
redundant under that constraint, so that ids stay unambiguous if a future V2
relaxes one-parent-once.

A content address is carried alongside, for consistency with the execution
substrate's addressing discipline:

```
advint_addr := SHA256( LP("ADVINT-REC-v1") ‖ LP(canonical_json(record \ {record_sha256, partition})) )
```

`record_sha256` is that digest in lowercase hex. `partition` is excluded from the
hash so that re-partitioning under a future rule cannot change record identity.

---

## 11. Canonicalization and pool identity

### 11.1 Text normalisation (applied once, at load, to every string)

1. Decode as UTF-8, strict; a decode error aborts construction.
2. Apply Unicode **NFC** normalisation.
3. Replace the FEVER wiki-text escapes `-LRB-`→`(`, `-RRB-`→`)`,
   `-LSB-`→`[`, `-RSB-`→`]`, `-COLON-`→`:`.
4. Replace U+00A0 and U+2009 with U+0020.
5. Collapse runs of whitespace to a single U+0020.
6. Strip leading and trailing whitespace.

Normalisation is **not** applied a second time anywhere. Every offset in
`attack_parameters` refers to the normalised string, and every stored string is
the normalised one, so offsets are reproducible.

### 11.2 Canonical JSON

```
json.dumps(obj, sort_keys=True, separators=(",", ":"),
           ensure_ascii=False, allow_nan=False)
```
encoded UTF-8. This is byte-identical to the `canonical_json` already used by the
execution substrate, so a record hashed here and a record hashed there agree.

Integers are emitted without a decimal point; there are no floating-point values
anywhere in the schema — every numeric quantity is either an integer or an exact
decimal **string**. This eliminates the single largest source of cross-language
serialization divergence.

### 11.3 Per-record hash, sort order, manifest, pool root

```
record_sha256 := SHA256( LP("ADVINT-REC-v1")
                       ‖ LP(canonical_json(record \ {record_sha256, partition})) )
```

**Canonical sort order:** records ascending by `instance_id` compared as UTF-8
bytes.

**Manifest** `advint_manifest.json`, itself canonical JSON:

```jsonc
{
  "protocol_version": "ADVERSARIAL_INTEGRITY_PROTOCOL_V1",
  "dataset_id": "adversarial_integrity",
  "source_dataset": "FEVER_ONLY",
  "source_revision": "<hex>",
  "source_split": "train",
  "pool_size": 1000,
  "cells": { "CITATION_SWAP:SUPPORTS": 167, "CITATION_SWAP:REFUTES": 167,
             "SEMANTIC_SLOT_HIJACK:SUPPORTS": 167, "SEMANTIC_SLOT_HIJACK:REFUTES": 166,
             "NUMBER_FLIP:SUPPORTS": 166, "NUMBER_FLIP:REFUTES": 167 },
  "targets": { "INVALID_SUPPORT": 667, "VALID_SUPPORT": 333 },
  "citation_swap_tier1_fraction": { "num": <int>, "den": 334 },
  "ssh_tier1_fraction": { "num": <int>, "den": 333 },
  "partition_counts": { "SMOKE": <int>, "CHECKER_CALIBRATION": <int>,
                        "PILOT": <int>, "FINAL": <int> },
  "records": [ { "instance_id": "...", "record_sha256": "..." }, ... ],
  "pool_root": "<hex>"
}
```

**Pool root** — order-sensitive over the canonical order, with the count bound in
so that truncation cannot produce a valid root:

```
pool_root := SHA256( LP("ADVINT-POOL-v1")
                   ‖ LP(protocol_version)
                   ‖ LP(source_revision)
                   ‖ n.to_bytes(8, "big")
                   ‖ raw_bytes(record_sha256[0]) ‖ … ‖ raw_bytes(record_sha256[n-1]) )
```

Digests are concatenated as **raw 32-byte values**, not hex strings, so the
encoding is fixed-width and unambiguous.

Two implementations agreeing on §1 (pinned bytes), §7 (ranking), §4–6
(transforms), §11.1 (normalisation) and §11.2 (serialization) necessarily agree
on every `record_sha256` and therefore on `pool_root`. Those five are exactly the
places where implementations diverge, which is why each is frozen to the byte.

### 11.4 Frozen lexicons

Ship as literal, sorted, version-tagged constants in the protocol module — never
loaded from an external package, whose contents could change:

- `STOPWORDS` — 40 entries: `a an and are as at be but by for from had has have
  he her his in is it its of on or she that the their there they this to was
  were which who will with would you your`
- `MONTHS` — the 12 English month names plus the 12 three-letter abbreviations.
- `UNIT_LEXICON` — `km kilometre kilometres kilometer kilometers m metre metres
  meter meters cm mi mile miles ft feet in inch inches kg kilogram kilograms g
  gram grams lb lbs pound pounds tonne tonnes ton tons l litre litres liter
  liters ml mph kmh km/h °c °f celsius fahrenheit hectares acres`
- `STATE_ABBREV` — the 50 US two-letter postal codes.

Every lexicon is matched case-insensitively on the normalised text and is
otherwise treated as an exact token set.

---

## 12. Duplicate policy

```
ONE_PARENT_USED_AT_MOST_ONCE = YES
```

### 12.1 Normal form used for all duplicate tests

`nf(s)` := §11.1 normalisation → lowercase → remove all characters in
`.,;:!?"'()[]-` → collapse whitespace → strip.

### 12.2 Rules

| duplicate kind | rule |
|---|---|
| duplicate parent | a parent enters `claimed_parents` on acceptance and is skipped in every later cell |
| duplicate original claim | `nf(original_claim)` must be globally unique across accepted parents. FEVER `train` contains near-duplicate claims under distinct ids; without this rule two "distinct" records could be textual twins |
| duplicate perturbed claim | `nf(perturbed_claim)` must not equal any accepted `nf(original_claim)` or `nf(perturbed_claim)` |
| duplicate evidence — citation swap donors | `DONOR_REUSE_CAP = 3` on donor sentence text (rule C4) |
| duplicate evidence — parent evidence | no constraint; two parents legitimately citing the same sentence is a property of FEVER, and both records remain distinguishable by claim |
| cross-family duplicates | impossible: `claimed_parents` is global, so a parent cannot appear under two families |

`DONOR_REUSE_CAP = 3` bounds any single donor to ≤ 0.3% of the pool. With ~500
claim-side and 334 evidence-side attacks, the cap needs only ~112 and ~167
distinct donors respectively, both trivially available from a 5.4M-page
gazetteer and a ~100k-sentence evidence pool.

---

## 13. Partitioning

Partitions are assigned **before any outcome is observed**, as a pure function of
`instance_id`.

### 13.1 The rule

```
PARTITION_RULE = ADVINT-PART-v1-BUCKET1000

b(instance_id) := int.from_bytes(
    SHA256( LP("ADVINT-PART-v1") ‖ LP(protocol_version) ‖ LP(instance_id) )[0:8],
    "big" ) mod 1000
```

```
   0 ≤ b ≤  19   →  SMOKE                 (2%,  ≈ 20)
  20 ≤ b ≤ 119   →  CHECKER_CALIBRATION   (10%, ≈100)
 120 ≤ b ≤ 269   →  PILOT                 (15%, ≈150)
 270 ≤ b ≤ 999   →  FINAL                 (73%, ≈730)
```

### 13.2 Properties

- The four intervals partition `{0,…,999}`, so the four sets are **disjoint by
  construction**. `SMOKE ∩ PILOT = ∅` and `PILOT ∩ FINAL = ∅` are theorems of the
  rule, not checks that could be forgotten.
- `b` depends only on `instance_id`, hence only on `(parent_id, attack_family)` —
  never on the parent's label, the attack outcome, the donor, or the position in
  any ranking. Assignment cannot be correlated with difficulty.
- Realised counts are whatever the hash yields; they are **not** targets. They
  are recorded in the manifest as `partition_counts`. Forcing exact counts would
  require a rank-then-cut rule, which reintroduces order dependence for no
  scientific gain.
- Because `b` is independent of the cell, each of the six cells receives
  approximately the same proportions. The per-cell realised counts are also
  recorded, so any accidental imbalance is visible rather than assumed away.
- `CHECKER_CALIBRATION` is included because this cell's checker threshold is
  exactly the kind of quantity that must not be tuned on the evaluation split.

### 13.3 Parent overlap with the main FEVER cell

```
MAIN_FEVER_PARENT_OVERLAP_ALLOWED = NO
```

Enforced twice:

1. **Structurally** — `SOURCE_SPLIT = train`, and the main FEVER final-evaluation
   cell draws from `shared_task_dev`, which is a disjoint file.
2. **By verification** — construction computes
   `parents(advint) ∩ ids(main_fever_final_cell)` and **aborts** if it is
   non-empty. A structural argument that is never checked is an assumption, and
   this project does not accept assumptions in place of checks.

---

## 14. Why this is not redundant with the main FEVER cell

| | main FEVER cell | adversarial_integrity cell |
|---|---|---|
| question | is this claim supported by the world, given retrieved evidence? | does the supplied evidence validly ground the asserted verdict? |
| what varies | claims and evidence, as they occur naturally | one controlled perturbation, with the parent and the mechanism known |
| failure being measured | factual classification error | certificate-validity error |
| ground truth | world truth | constructed support relation, with known provenance |
| counterfactual available | no | yes — the unperturbed parent is recorded |

The decisive difference: **334 records in this cell are world-truth-preserving
and certificate-invalid at the same time.** No sample of ordinary FEVER contains
such an item, because ordinary FEVER never separates the two. A system can
achieve high accuracy on main FEVER by being a good fact-checker; it cannot
achieve high accuracy here without checking the certificate. Any claim in the
manuscript that PCG-MAS verifies certificates rather than facts is *only*
falsifiable on this cell.

Second difference: **known parentage and known mechanism**. Every record carries
its parent and its exact perturbation, so a failure can be attributed to a
mechanism rather than reported as an aggregate error rate. That is what makes the
cell diagnostic rather than merely evaluative.

---

## 15. Metrics

All denominators are fixed by construction and stated with every number. Nothing
is pooled across cells with different targets.

### 15.1 Primary

| metric | numerator | denominator |
|---|---|---|
| **IAR** — invalid acceptance rate | accepted ∧ `target = INVALID_SUPPORT` | 667 |
| **CFRR** — control false-rejection rate | rejected ∧ `target = VALID_SUPPORT` | 333 |

`IAR` and `CFRR` are **reported jointly and never collapsed into one number.** A
single scalar would let a system trade the two off invisibly, which is exactly
what the certificate discipline exists to prevent. A system is only better if it
does not increase either.

### 15.2 Per-family invalid acceptance

| family | denominator |
|---|---|
| CITATION_SWAP | 334 |
| SEMANTIC_SLOT_HIJACK (SUPPORTS only) | 167 |
| NUMBER_FLIP (SUPPORTS only) | 166 |

Reported per-cell, always with the cell's `n`. `CITATION_SWAP × REFUTES` (167) is
reported as its own line because of §3.2.

### 15.3 Support-failure detection and channel localization

| metric | denominator |
|---|---|
| `V_⊢ = false` on invalid records | 667 |
| `CheckFail` fired on invalid records | 667 |
| `CheckFail` fired on control records (should be 0) | 333 |

**Channels with undefined denominators, which stay undefined.** This dataset
contains **no** execution-policy perturbation, no replay intervention, no
pipeline drift and no integrity violation. Therefore:

```
IntFail    : UNDEFINED — no integrity perturbation exists in this cell
ReplayFail : UNDEFINED — no replay intervention exists in this cell
DriftFail  : UNDEFINED — no snapshot/policy drift exists in this cell
CovGap     : DEFINED only if the coverage sampler is run over this cell;
             otherwise UNDEFINED
```

These are reported as `UNDEFINED`, never as `0`. A rate of 0 asserts that the
event was measured and did not occur; `UNDEFINED` asserts that it was not
measured. Inventing execution-policy metrics for attacks that contain no
execution-policy perturbation would be the exact failure this project has spent
the whole engineering pass eliminating.

### 15.4 Attack-family detection

Defined only for systems that emit a machine-readable rejection reason:

```
family detection accuracy =
    #{ correctly named family | rejected ∧ INVALID_SUPPORT ∧ reason emitted }
    ────────────────────────────────────────────────────────────────────────
    #{ rejected ∧ INVALID_SUPPORT ∧ reason emitted }
```

If a system emits no reason, the metric is `UNDEFINED` for that system. It is not
imputed and not scored as 0.

### 15.5 Mandatory reference baselines

Reported alongside every system, because a headline number without them is
uninterpretable:

1. **ALWAYS_ACCEPT** — `IAR = 1.000`, `CFRR = 0.000`.
2. **ALWAYS_REJECT** — `IAR = 0.000`, `CFRR = 1.000`.
3. **LEXICAL_OVERLAP_BASELINE** — reject iff the claim's primary entity does not
   appear in the evidence. This is the shortcut §5.4 is designed to blunt; its
   score is the quantitative statement of how far the cell escaped it.
4. **PERTURBATION_DETECTOR (oracle-ish)** — reject iff the claim differs from its
   parent. Scores `IAR = 0.50`, `CFRR = 1.00` by construction, demonstrating that
   perturbation detection alone is not a solution.

---

## 16. Theory connection

Stated as relationships, not as proofs. **This dataset does not prove any
theorem.** It is a stress test that can falsify an operational claim; it cannot
establish one.

**Checker-relative acceptance.** `Check(Z;G_t) = V_H · V_Π · V_Γ · V_⊢` is
defined relative to a fixed checker `G_t`. Every number this cell produces is
therefore a statement about *this* checker at *this* configuration, and must be
reported with the checker fingerprint. The cell measures how `Check` behaves when
`V_⊢`'s input is adversarially decoupled from the claim; it says nothing about a
different checker.

**ε_tax (open-world taxonomy residual).** The three families are, by
construction, *inside* the taxonomy: each record is generated by a known
mechanism with a known target. This cell therefore **cannot estimate ε_tax**, and
must never be cited as bounding it. If anything, a low error rate here is
evidence about in-taxonomy performance only, and quoting it as evidence about
out-of-taxonomy attacks would be the precise error ε_tax exists to name.

**ε_src (source/world-truth residual).** This is where the cell's own residual
lives, and it should be stated in the paper rather than buried. The 333 control
records are labelled `VALID_SUPPORT` on the strength of an argument — that
perturbing a non-attested slot of a false claim leaves it false — that no
deterministic offline check can fully verify. The guards in §4.4 D3 and §6.6
remove the mechanically detectable failures; what remains is world knowledge we
do not have. That residual **is** ε_src for this cell, and it is bounded only by
human adjudication (§17, R2), not by construction.

**Separating witnesses.** Each of the 334 `CITATION_SWAP` records is a
constructed witness separating "world-true claim" from "validly certified claim":
the two predicates take different values on the same record. A system whose
behaviour is identical across a separating witness has, demonstrably, not
implemented the distinction.

**Audit-channel coverage.** The cell exercises exactly one channel, `CheckFail`.
Its contribution to channel coverage is therefore narrow and must be stated as
such: it strengthens `CheckFail` evidence and contributes nothing to the other
four. A coverage claim aggregated over all channels must not draw on this cell
for the channels it does not touch.

---

## 17. Construct-validity attack on this protocol

Attacking V1 as an adversarial reviewer would. Material risks are mitigated in
the protocol text above; residual risks are named, not minimised.

**R1 — Trivial lexical artifacts.** *Risk:* citation-swapped evidence never
mentions the claim's entity, so string matching solves the family.
*Severity: high.* *Mitigation:* §5.4 inverts the suggested zero-overlap rule and
prefers donors sharing a non-primary entity; §5.5 ranks by descending Jaccard so
the donor is the most confusable admissible sentence, not a random one; §15.5
makes `LEXICAL_OVERLAP_BASELINE` a mandatory reported reference.
*Residual:* rule C3 (primary entity absent) is what guarantees non-support and
cannot be dropped, so a primary-entity detector retains some signal. The
`citation_swap_tier1_fraction` manifest field measures exactly how much.

**R2 — Label leakage from the parent label.** *Risk:* within SSH and NF,
`certificate_validity_target` is a deterministic function of `source_world_label`.
*Severity: high, and partly irreducible.* *Analysis:* the correlation is not an
artifact — "the evidence contradicts the claim, so `REFUTED` is validly grounded"
is the correct reasoning, not a shortcut. But it does mean those 666 records
cannot separately measure claim-perturbation sensitivity.
*Mitigation:* §3.2 — `CITATION_SWAP × REFUTES` (167 records) breaks the
degenerate strategy, and §15.2 requires it to be reported as its own line.
*Residual:* a system exploiting the correlation still scores well on 666/1000.
Per-cell reporting is mandatory precisely so this is visible.

**R3 — Donor leakage.** *Risk:* a few donors recur often enough to be memorised
or to become a signature. *Mitigation:* `DONOR_REUSE_CAP = 3` (§4.4 D6, §5.3 C4)
bounds any donor at 0.3% of the pool. *Residual:* low.

**R4 — Attack-family recognizability.** *Risk:* families are distinguishable by
surface form (NF changes a digit; CS changes a page pointer), letting a system
route to a family-specific heuristic. *Mitigation:* the matched controls (§3.1)
mean family recognition alone yields nothing — within SSH and NF the family is
constant while the target flips. *Residual:* CS remains recognizable, and its
target is constant; R1's baselines quantify the cost.

**R5 — Parent dependence.** *Risk:* correlated errors from shared parents inflate
apparent precision. *Mitigation:* one parent at most once (§2, §12), plus global
uniqueness of `nf(original_claim)` so near-duplicate FEVER claims cannot slip in
as distinct parents. *Residual:* low.

**R6 — Contamination with main FEVER, and with pretraining.** *Risk (a):* parent
overlap with the main cell. *Mitigation:* §13.3, structural plus verified.
*Risk (b):* FEVER `train` is widely used for fine-tuning, so a FEVER-tuned
component would have seen the parents. *Severity: high if any component is
FEVER-tuned.* *Mitigation:* the protocol **requires** a declaration that no
component in the evaluated stack is FEVER-fine-tuned; if that declaration cannot
be made, `SOURCE_SPLIT` must be re-decided under U5. *Residual:* pretraining-scale
exposure to Wikipedia is unavoidable and affects every FEVER-derived cell equally.

**R7 — Synthetic shortcuts.** *Risk:* perturbed claims read as unnatural
(agreement errors, capitalisation mismatches) and are detectable as "generated".
*Severity: moderate.* *Analysis:* §4.5 forbids fix-up precisely because a fix-up
would be a generative step, so some perturbed claims will be mildly ungrammatical.
*Mitigation:* the controls share the identical code path, so unnaturalness is
uncorrelated with the target within SSH and NF. *Residual:* real for CS, whose
claim is untouched — CS claims are perfectly natural while SSH/NF claims may not
be, giving a cross-family cue. Bounded by per-cell reporting.

**R8 — Determinism making attacks easy.** *Risk:* a fixed ±10% and a fixed donor
rule are learnable. *Mitigation:* the sign is hash-governed (§6.4), so the
direction is unpredictable without the protocol constants; SSH donors are
hash-ranked rather than similarity-ranked (§4.4) so "nearest plausible
alternative" is not the signature; and the true value is never shown, so `1.1×`
is unidentifiable from the perturbed claim alone. *Residual:* an implementation
that leaks the protocol constants to the evaluated system would break this;
constants must not appear in prompts.

**R9 — Imbalance.** *Risk:* 667/333 target imbalance makes a pooled accuracy
misleading. *Mitigation:* §15 forbids a single pooled scalar, requires `IAR` and
`CFRR` jointly, and fixes every denominator in advance. *Residual:* none, if the
reporting discipline is followed.

**R10 — Single-hop restriction.** *Risk:* multi-hop certificates are the harder
and arguably more interesting case, and are excluded entirely (§1.5).
*Severity: moderate, scope-limiting rather than validity-threatening.*
*Mitigation:* none in V1; stated as a scope limit in the manuscript wording
(§18). *Residual:* the natural V2 extension.

---

## 18. Proposed manuscript replacement wording (not applied)

For the v3.2 dataset description. **The manuscript is not edited by this
document.** Final attack-family names and the exact source definition are used.

> **Adversarial integrity (adversarial\_integrity).** We construct a
> 1,000-example controlled stress test of certificate validity from FEVER 1.0
> (Thorne et al., 2018), using the `train` split only and pinning the source by
> content hash. We retain `SUPPORTS` and `REFUTES` claims and exclude
> `NOT ENOUGH INFO`, which carries no evidence and therefore no support relation
> to perturb. Each example has exactly one FEVER parent, each parent is used at
> most once, and exactly one perturbation is applied per example. Parents are
> disjoint from the main FEVER evaluation cell by construction and by
> verification.
>
> Three deterministic, non-generative attack families are applied in equal
> thirds: **citation swap** (334), which leaves the claim byte-identical and
> replaces its evidence with a lexically similar sentence that does not mention
> the claim's primary entity; **semantic slot hijack** (333), which replaces a
> single type-compatible named entity in the claim while leaving the evidence
> unchanged; and **number flip** (333), which perturbs one eligible numeric span
> by ±10% under a hash-determined sign, excluding dates, ordinals, identifiers
> and version numbers. No language model is used at any point in construction,
> and the pool is reproducible from the pinned source to a single root hash.
>
> The cell separates two labels that ordinary fact-verification conflates. The
> `source_world_label` records the FEVER truth status of the original claim; the
> `certificate_validity_target` records whether the supplied evidence validly
> grounds the asserted verdict. In 334 examples the claim remains world-true
> while its certificate becomes invalid — a combination that does not occur in
> ordinary FEVER, and on which a system that verifies facts rather than
> certificates cannot succeed.
>
> Because a claim-side perturbation of a refuted claim leaves both the claim's
> falsity and its refutation intact, 333 examples are **matched invariance
> controls** whose target is a valid certificate. Controls share the attack
> mechanism, the code path and the surface statistics of the attacks, so
> detecting that a perturbation occurred is not a solution. We therefore report
> invalid-acceptance rate (n = 667) and control false-rejection rate (n = 333)
> jointly and never as a single scalar, alongside per-family and per-cell
> denominators.
>
> The cell exercises the `CheckFail` audit channel only. It contains no
> integrity, replay or drift perturbation, and the corresponding channel rates
> are reported as undefined rather than as zero. Evidence is restricted to
> single-sentence FEVER annotations, so multi-hop certificates are outside its
> scope.

---

## 19. Full pseudocode

Executable-level. No helper is left unspecified; every function named here is
defined in this document.

```text
CONSTANTS
  PROTOCOL_VERSION = "ADVERSARIAL_INTEGRITY_PROTOCOL_V1"
  SOURCE_SPLIT     = "train"
  POOL_SIZE        = 1000
  QUOTAS = [ ("NUMBER_FLIP","SUPPORTS",166), ("NUMBER_FLIP","REFUTES",167),
             ("SEMANTIC_SLOT_HIJACK","SUPPORTS",167), ("SEMANTIC_SLOT_HIJACK","REFUTES",166),
             ("CITATION_SWAP","SUPPORTS",167), ("CITATION_SWAP","REFUTES",167) ]   # order is §8.3
  DONOR_REUSE_CAP = 3
  STOPWORDS, MONTHS, UNIT_LEXICON, STATE_ABBREV     # §11.4, literal constants

1. SOURCE LOADING
   pin      := read_json("advint_source_pin.json")
   assert sha256_file("train.jsonl")     == pin.train_sha256          else ABORT
   assert sha256_file("wiki-pages.zip")  == pin.wiki_zip_sha256       else ABORT
   for shard in 1..109:
       assert sha256_file("wiki-%03d.jsonl" % shard) == pin.shards[shard]  else ABORT
   SOURCE_REVISION := hex( SHA256( LP("ADVINT-SRCREV-v1") ‖ LP("fever-1.0")
                                 ‖ LP(pin.train_sha256) ‖ LP(pin.wiki_zip_sha256) ) )

2. IMMUTABLE REVISION CHECK
   assert SOURCE_REVISION == pin.frozen_source_revision                else ABORT
   # no warn-and-continue path exists

3. SOURCE FILTERING
   WIKI  := { page_id -> [sentence_0, sentence_1, ...] }        # from `lines`, tab-split, §11.1
   G     := build_gazetteer(WIKI)                               # §4.2
   TAU   := { page_id -> copula_type_signature }                # §4.2
   PARENTS := []
   for line in train.jsonl:                                     # file order; used only for filtering
       r := parse_json(line)
       if r.label not in {"SUPPORTS","REFUTES"}: continue
       ev := first_single_sentence_annotation_set(r.evidence)   # §1.5, ordered by (page, sentence_id)
       if ev is None: continue
       if ev.page not in WIKI or ev.sentence_id >= len(WIKI[ev.page]): continue
       claim := normalise(r.claim); etext := normalise(WIKI[ev.page][ev.sentence_id])
       if not (5 <= tokens(claim) <= 60): continue
       if tokens(etext) < 5: continue
       PARENTS.append({ id: str(r.id), label: r.label, claim, ev_page: ev.page,
                        ev_sid: ev.sentence_id, ev_text: etext,
                        prim: surf(ev.page) })
   assert_no_overlap(ids(PARENTS), ids(MAIN_FEVER_FINAL_CELL))          else ABORT   # §13.3

4. ATTACK ELIGIBILITY POOLS
   POOL := {}
   for fam in {"NUMBER_FLIP","SEMANTIC_SLOT_HIJACK","CITATION_SWAP"}:
     for lab in {"SUPPORTS","REFUTES"}:
       POOL[fam,lab] := [ p for p in PARENTS if p.label == lab and eligible(p, fam) ]
   # eligible(p,"NUMBER_FLIP")           := valid_number_targets(p) is non-empty          §6.3
   # eligible(p,"SEMANTIC_SLOT_HIJACK")  := eligible_slots(p) is non-empty                §4.3
   # eligible(p,"CITATION_SWAP")         := true                                          §5

5. DETERMINISTIC CANDIDATE RANKING
   for each (fam,lab):
       sort POOL[fam,lab] by ( rank_key(p.id, fam) as bytes ASC, utf8(p.id) ASC )   # §7.2

6-8. TRANSFORMATION, VALIDATION, QUOTAS
   claimed_parents := {}; donor_use := Counter(); seen_nf := {}; RECORDS := []
   for (fam, lab, quota) in QUOTAS:                                   # §8.3 order
       filled := 0
       for p in POOL[fam,lab]:
           if filled == quota: break
           if p.id in claimed_parents: continue
           if nf(p.claim) in seen_nf: continue
           iid := "advint-v1:" + p.id + ":" + fam
           out := apply_attack(fam, p, iid, G, TAU, WIKI, PARENTS, donor_use)
           if out is REJECT: continue
           if nf(out.perturbed_claim) in seen_nf: continue
           RECORDS.append(build_record(p, fam, iid, out))
           claimed_parents.add(p.id); seen_nf.add(nf(p.claim)); seen_nf.add(nf(out.perturbed_claim))
           for d in out.donors_used: donor_use[d] += 1
           filled += 1
       if filled < quota:
           ABORT("cell %s:%s reached %d of %d" % (fam, lab, filled, quota))    # §8.4

   apply_attack("NUMBER_FLIP", p, iid, ...):
       for span in valid_number_targets(p) ordered by (class priority, start ASC, len DESC):   # §6.3
           s := +1 if (SHA256(LP("ADVINT-NUMSIGN-v1")‖LP(PROTOCOL_VERSION)‖LP(iid)
                              ‖LP(str(span.start)))[31] & 1) == 0 else -1                      # §6.4
           for attempt_sign in [s, -s]:
               n2 := round_half_up( sign(n)*(abs(n)*(1 + attempt_sign/10)), span.precision )   # §6.5
               if n2 == n: continue
               if is_percent(span) and (n2 > 100 or n2 < 0): continue
               if n2 in EVNUMS(p): continue                                                    # §6.6
               return OK(perturbed_claim = splice(p.claim, span, format(n2, span)),
                         perturbed_evidence = p.evidence, donors_used = [])
       return REJECT

   apply_attack("SEMANTIC_SLOT_HIJACK", p, iid, ...):
       for slot in eligible_slots(p) ordered by (start ASC, len DESC, surf ASC):                # §4.3
           for tier in [1, 2]:
               donors := donor_pool(slot, tier)                                                # §4.4
               donors := [q for q in donors if D1..D6 hold]
               sort donors by ( SHA256(LP("ADVINT-DONOR-v1")‖LP(PROTOCOL_VERSION)
                                       ‖LP(iid)‖LP(page_id(q))) ASC, page_id(q) ASC )
               for q in donors:
                   c2 := splice(p.claim, slot, surf(q))
                   if c2 == p.claim: continue
                   if not (5 <= tokens(c2) <= 60): continue
                   return OK(perturbed_claim = c2, perturbed_evidence = p.evidence,
                             donors_used = [page_id(q)])
       return REJECT

   apply_attack("CITATION_SWAP", p, iid, ...):
       for tier in [1, 2]:                                                                     # §5.4
           cands := [ d for d in ALL_PARENT_EVIDENCE if C1..C6 hold
                                and 10*jaccard_num(p,d) >= jaccard_den(p,d)                    # §5.5
                                and (tier == 2 or shares_nonprimary_entity(p, d)) ]
           sort cands by ( jaccard DESC via cross-multiplication,
                           SHA256(LP("ADVINT-DONOR-v1")‖LP(PROTOCOL_VERSION)‖LP(iid)
                                  ‖LP(page_id(d))‖LP(str(sid(d)))) ASC,
                           (page_id(d), sid(d)) ASC )                                          # §5.6
           for d in cands:
               return OK(perturbed_claim = p.claim,                                 # byte-identical
                         perturbed_evidence = {page: page_id(d), sentence_id: sid(d), text: txt(d)},
                         donors_used = [txt(d)])
       return REJECT

9. IDS                 iid := "advint-v1:" + parent_id + ":" + attack_family          # §10
10-11. CANONICALIZATION AND HASHES
   for rec in RECORDS:
       body := canonical_json(rec without {"record_sha256","partition"})              # §11.2
       rec.record_sha256 := hex(SHA256( LP("ADVINT-REC-v1") ‖ LP(body) ))             # §11.3
12. MANIFEST AND ROOT
   RECORDS.sort by utf8(instance_id) ASC
   pool_root := hex(SHA256( LP("ADVINT-POOL-v1") ‖ LP(PROTOCOL_VERSION) ‖ LP(SOURCE_REVISION)
                          ‖ len(RECORDS).to_bytes(8,"big")
                          ‖ concat(raw_bytes(r.record_sha256) for r in RECORDS) ))
   assert len(RECORDS) == 1000
   assert count(target == "INVALID_SUPPORT") == 667 and count(target == "VALID_SUPPORT") == 333
   write manifest(§11.3) including citation_swap_tier1_fraction, ssh_tier1_fraction
13. DETERMINISTIC PARTITION ASSIGNMENT
   for rec in RECORDS:
       b := int.from_bytes(SHA256(LP("ADVINT-PART-v1")‖LP(PROTOCOL_VERSION)
                                  ‖LP(rec.instance_id))[0:8], "big") mod 1000        # §13.1
       rec.partition := "SMOKE"               if   0 <= b <=  19
                   else "CHECKER_CALIBRATION" if  20 <= b <= 119
                   else "PILOT"               if 120 <= b <= 269
                   else "FINAL"
   assert pairwise_disjoint(SMOKE, CHECKER_CALIBRATION, PILOT, FINAL)
   record partition_counts (realised, not targeted) in the manifest
```

---

## 20. End-state protocol block

```
ADVERSARIAL_INTEGRITY_PROTOCOL_V1
ADVERSARIAL_PROTOCOL_V1_RECOMMENDED=YES
SOURCE_DATASET=FEVER_ONLY (FEVER 1.0 shared-task distribution: train.jsonl + wiki-pages.zip)
SOURCE_REVISION=UNRESOLVED_PENDING_PIN
  (defined as SHA256(LP("ADVINT-SRCREV-v1")‖LP("fever-1.0")‖LP(sha256(train.jsonl))‖LP(sha256(wiki-pages.zip)));
   component hashes cannot be computed under BENCHMARK_DATASET_EXECUTION=0)
SOURCE_SPLIT=train
POOL_SIZE=1000
SOURCE_SUPPORTS=500
SOURCE_REFUTES=500
CITATION_SWAP=334
SEMANTIC_SLOT_HIJACK=333
NUMBER_FLIP=333
ONE_ATTACK_PER_INSTANCE=YES
ONE_PARENT_USED_AT_MOST_ONCE=YES
LLM_USED_IN_DATASET_CONSTRUCTION=NO
DETERMINISTIC_RECONSTRUCTION=YES
MAIN_FEVER_PARENT_OVERLAP_ALLOWED=NO
PARTITION_RULE=ADVINT-PART-v1-BUCKET1000
MANUSCRIPT_REWORDING_REQUIRED=YES
UNRESOLVED_SCIENTIFIC_GAPS=5
```

### Unresolved scientific gaps (the 5)

| id | gap | who resolves it |
|---|---|---|
| **U1** | `SOURCE_REVISION` component hashes are unpinned; the pinning ceremony (§1.2) has not been run | human, once, offline |
| **U2** | The 333 control targets rest on a world-knowledge assumption that no offline check can verify; this is the cell's ε_src. A bounded human adjudication (≥ 50 sampled controls) is needed to estimate the label-noise rate | human adjudication |
| **U3** | The copula type signature (§4.2) is a coarse proxy for entity type; its adequacy is unmeasured until a human reviews a sample of SSH replacements for type plausibility | human review post-construction |
| **U4** | `citation_swap_tier1_fraction` is unknown until construction. If tier 2 dominates, the lexical shortcut (R1) is stronger than intended and §5.4 must be revised | measured at construction; threshold to be set by human |
| **U5** | The assertion that the main FEVER final-evaluation cell uses `shared_task_dev` is unconfirmed; I am forbidden from re-auditing to check it. If it uses `train`, §1.3 must be re-decided | human confirmation before freeze |

**Recommendation: YES**, conditional on U1 and U5 being closed before construction
and U2 being scheduled before any number from this cell enters the manuscript.
