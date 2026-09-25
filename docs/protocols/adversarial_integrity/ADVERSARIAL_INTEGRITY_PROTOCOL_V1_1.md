# ADVERSARIAL_INTEGRITY_PROTOCOL_V1_1

Supersedes `ADVERSARIAL_INTEGRITY_PROTOCOL_V1`. This is a **scientific-semantics
correction**, not a redesign. The deterministic construction machinery of V1 is
preserved wherever the corrections do not invalidate it; §21 lists exactly what
carried over unchanged and exactly what moved, and why.

Nothing here has been executed. No code, no examples, no models, no dataset.

Two corrections forced genuine design changes rather than wording changes:

- **C1** — FEVER labels are reference annotations relative to a frozen corpus,
  not external world truth. `source_world_label` and `source_truth_changed` are
  removed from the schema entirely (§1).
- **C2** — the V1 claim that `REFUTES`-side claim perturbations *necessarily*
  preserve a valid certificate was **wrong**, and I am retracting it (§2). The
  667/333 control design is withdrawn and replaced.

Everything else is tightening.

---

## 1. Source-truth ontology (correction 1)

### 1.1 The error in V1

V1 wrote `source_world_label` and `source_truth_changed`. Both overclaim. FEVER
annotators judged claims against a **frozen June-2017 Wikipedia snapshot**. Their
label is a statement about a corpus, at a moment, under an annotation guideline.
It is not privileged access to the world. Wikipedia was wrong about things in
June 2017, and is wrong about things now.

Worse, the overclaim collides with the theory. `ε_src` is defined in the
manuscript as the residual arising from **source/world-truth failure** — the case
where the pipeline is internally correct and the source is wrong about the world.
A dataset that stamps `source_world_label` onto every record is asserting that it
has measured the very quantity `ε_src` exists to name as unmeasured. That is not
a labelling infelicity; it is a category error that would let the cell be cited
as bounding `ε_src`, which it cannot do.

### 1.2 The replacement ontology

```
source_reference_label      ∈ {SUPPORTS, REFUTES}
    A FEVER annotation, relative to the pinned wiki-pages corpus. Inherited
    verbatim. Never re-derived, never revised, never treated as world truth.

external_world_truth_status = "UNVERIFIED"
    A constant on every record in the cell. Present precisely so that no reader
    and no downstream metric can quietly assume the field exists in some other
    state. It is never computed, and there is no code path that can set it to
    any other value.
```

`source_truth_changed` is **deleted**. It has no replacement, because the
quantity it named was never available. In its place the schema carries only
mechanically knowable facts and explicitly adjudicated ones:

| field | kind | how it is obtained |
|---|---|---|
| `claim_text_changed` | mechanical | byte comparison of `original_claim` vs `perturbed_claim` |
| `evidence_text_changed` | mechanical | byte comparison of `original_evidence.text` vs `perturbed_evidence.text` |
| `source_reference_label` | inherited | FEVER annotation, verbatim |
| `support_relation_hypothesis` | constructed | what the transformation was *designed* to achieve |
| `adjudicated_support_relation` | human | blinded adjudication, §3 |
| `certificate_validity_target` | derived | a function of the adjudicated value only (§1.4) |

### 1.3 How this preserves the ε_src boundary

The boundary is preserved by three commitments, each of which is checkable:

1. **Every target in this cell is relative to the supplied evidence, never to
   the world.** The adjudication question (§3.5) instructs annotators to judge
   whether *this evidence sentence establishes this verdict*, and explicitly
   forbids the use of outside knowledge. The dataset therefore measures the
   support relation — the domain in which `Check(Z;G_t)` operates — and nothing
   outside it.

2. **`ε_src` remains entirely outside the cell.** If the pinned corpus is wrong
   about the world, every record in this cell is unaffected: the claim, the
   evidence, the perturbation and the adjudicated support relation are all
   internal to the corpus. The cell can therefore neither inflate nor deflate
   `ε_src`, and — stated as a prohibition in §16 — **may never be cited as
   bounding it**.

3. **The one place V1 leaked across the boundary is closed.** V1's controls were
   labelled `VALID_SUPPORT` on the strength of a world-knowledge argument
   ("perturbing a non-attested slot of a false claim leaves it false"). That
   argument used exactly the knowledge `ε_src` says we do not have. §2 removes
   it.

### 1.4 Certificate validity is derived, not asserted

```
certificate_validity_target :=
    VALID_SUPPORT     if adjudicated_support_relation = VALID_SUPPORT
    INVALID_SUPPORT   if adjudicated_support_relation = INVALID_SUPPORT
    UNRESOLVED        otherwise
```

This is a definitional reduction that holds **only because this cell perturbs
nothing else**: `V_H` (hash/integrity), `V_Π` (pipeline) and `V_Γ` (policy) are
held fixed across every record, so the only conjunct the perturbation can move is
`V_⊢`. The reduction is stated as a scoping assumption of the cell, not as a
general identity, and it is why §5 can still remove `expected_failed_channel`
without losing the target.

---

## 2. The control design, re-decided (correction 2)

### 2.1 Retraction

V1 §3.1 argued that for a `REFUTES` parent, perturbing a non-attested slot leaves
the claim false and leaves the evidence a valid refutation, so the certificate
stays valid. **That is not logically guaranteed, and I am retracting it.**

The failure is with non-exclusive predicates. Take a `REFUTES` claim
"Tom Hanks appeared in *Titanic*", refuted by an evidence sentence listing his
filmography. V1's `SEMANTIC_SLOT_HIJACK` would replace *Titanic* with a
type-compatible film. If the replacement happens to be a film he *did* appear in,
the perturbed claim becomes corpus-supported, the asserted `REFUTED` verdict
becomes wrong, and the certificate is invalid — the opposite of the precommitted
target. Appearing-in-a-film is not a functional relation; neither is party
membership, nor authorship, nor "is located in" for nested regions.

V1's guard (donor must not appear in the evidence) does not save this: the
evidence sentence lists a few films, the donor is a different film, and the
donor's truth is a fact about the world that no offline string check can reach.
Detecting exclusivity requires exactly the world knowledge §1 just declared
`UNVERIFIED`.

So the V1 controls were **bimodal and undecidable offline**. They were
attractive experimentally — matched mechanism, matched surface statistics — and
that attractiveness is not an argument. The design is withdrawn.

### 2.2 Comparing the three directions

**Direction A — retain matched controls, validate them.**
*Strength:* preserves V1's matched-mechanism property, which is genuinely the
best defence against a "detect any perturbation" shortcut.
*Weakness:* it spends the entire adjudication budget establishing whether a
construction I already believe to be bimodal happened to work. The prior is poor:
the fraction of FEVER `REFUTES` claims resting on a functionally exclusive
predicate is unknown and plausibly near half. A design whose validity is a coin
flip is not a design.
*Verdict:* insufficient alone.

**Direction B — redesign the REFUTES-side transformations.**
*Strength:* if a stronger sufficient condition exists and is deterministic, it
fixes the problem at the source rather than measuring the damage.
*Weakness as posed:* the obvious version — restrict to a frozen list of
exclusive-predicate templates ("was born in", "is the capital of") — shrinks
eligibility hard and makes every `REFUTES`-side record share one of ~20
syntactic frames. That is a systematic lexical artifact, i.e. it trades a
semantic problem for a construct-validity problem.
*But:* there is a **non-template** version, and it is strong. See §2.3.
*Verdict:* viable, in the form of §2.3, and adopted.

**Direction C — paired clean controls.**
*Strength:* decisive. The unperturbed parent certificate is valid on the strength
of **FEVER's own annotation** — the annotators selected that evidence as
sufficient for that verdict. This introduces *no new semantic assumption*: it is
the identical basis the main FEVER cell already rests on. Pairing also converts
the analysis from between-item to within-parent, removing parent-level variance
from the comparison.
*Weakness:* it does not by itself resolve what the `REFUTES`-side *treatment*
targets should be. It supplies valid controls; it says nothing about whether the
attacks achieved invalidity.
*Verdict:* necessary, not sufficient.

### 2.3 Selected design — `PAIRED-REALIGN-ADJ-v1_1`

None of A, B or C is adequate alone; each fixes a different defect. The selected
design is the composition, and the composition is stronger than any part:

```
CONTROL_DESIGN = PAIRED-REALIGN-ADJ-v1_1
   C : controls are the unperturbed parent certificates (valid on FEVER's own
       annotation, no new assumption)
   B : REFUTES-side claim perturbations are redesigned as evidence REALIGNMENT,
       giving every treatment cell a strong, deterministic invalidity argument
   A : every treatment target is a construction hypothesis, confirmed or refuted
       by blinded human adjudication with precommitted cell-level thresholds
```

**The Direction-B insight — REALIGN.** V1 tried to keep a `REFUTES` claim false.
The stronger move is the opposite: make it **evidence-aligned**.

Take a `REFUTES` parent: claim `C` contains a false element; evidence `E` states
the corpus value. Replace the false element in `C` with **the value `E` itself
states**. The perturbed claim now asserts what the evidence asserts, so the
evidence *supports* it — and the asserted verdict is `REFUTED`. The certificate is
therefore ungrounded, and the argument for that is nearly verbatim textual
alignment rather than world knowledge.

| mode | parent | operation | why the certificate should fail |
|---|---|---|---|
| `DESUPPORT` | SUPPORTS | replace an element the evidence **attests** with a type-compatible element it does **not** attest | evidence no longer states what the claim asserts, so it cannot support the `SUPPORTED` verdict |
| `REALIGN` | REFUTES | replace the element the evidence does **not** attest with the element it **does** attest | evidence now states what the claim asserts, so it cannot ground the `REFUTED` verdict |

Both modes are deterministic, non-generative, and reuse V1's machinery
unchanged — same parser, same gazetteer, same exact-decimal arithmetic, same
ranking, same donor caps. `REALIGN` needs *no* rounding at all: it copies the
evidence's surface form verbatim, which is more deterministic than `FLIP`, not
less.

Note this **inverts V1 rule D3** (donor must not appear in the evidence) for
`REFUTES` parents. The inversion is deliberate and label-conditional, and it is
what converts a world-knowledge argument into a textual one.

**`REALIGN` is still a hypothesis, not a proof.** A multi-fact claim can remain
corpus-unsupported for an unrelated reason after realignment. The argument is
much stronger than V1's, and it is still adjudicated (§3).

### 2.4 What this changes, and what it does not

**Unchanged:** `N = 1000`; 500 `SUPPORTS` / 500 `REFUTES` parents; the family
quotas `CITATION_SWAP = 334`, `SEMANTIC_SLOT_HIJACK = 333`, `NUMBER_FLIP = 333`;
the exact label cross-allocation. All of V1's quota arithmetic survives, because
`REALIGN` lets `REFUTES` parents host claim-side attacks with a defensible target.

**Changed:** the 667/333 split is **withdrawn**. All 1,000 treatment records now
carry `support_relation_hypothesis = INVALID_SUPPORT`; validity controls come
from the 1,000 unperturbed parents instead.

### 2.5 The statistical unit

```
statistical unit  = the parent/attack PAIR
N (pairs)         = 1000
evaluation items  = 2000  (1000 control items + 1000 treatment items)
```

Each pair `i` contributes:

- **control item** `c_i` — original claim, original evidence, the FEVER-annotated
  verdict. Hypothesised `VALID_SUPPORT`.
- **treatment item** `t_i` — the perturbed pair from `c_i`'s parent.
  Hypothesised `INVALID_SUPPORT`.

Control and treatment share a parent, a topic, a claim template and a length
distribution. The comparison is therefore **within-parent**, and parent-level
variance — the dominant nuisance term in a 1,000-item cell — cancels. This is a
strictly better estimator than V1's between-cell comparison, and it is the second
reason to prefer C over A.

Items are presented to the evaluated system **independently and unlabelled**; the
pairing exists in the analysis, not in the presentation, so a system cannot see
that two items share a parent. Presentation order is fixed by
`SHA256(LP("ADVINT-EVALORDER-v1_1") ‖ LP(dataset_release_root) ‖ LP(item_id))`
ascending.

### 2.6 Cell table (V1.1)

| cell | pairs | claim changed | evidence changed | mode | `support_relation_hypothesis` |
|---|---|---|---|---|---|
| CITATION_SWAP × SUPPORTS | 167 | no | yes | swap | `INVALID_SUPPORT` |
| CITATION_SWAP × REFUTES | 167 | no | yes | swap | `INVALID_SUPPORT` |
| SEMANTIC_SLOT_HIJACK × SUPPORTS | 167 | yes | no | `DESUPPORT` | `INVALID_SUPPORT` |
| SEMANTIC_SLOT_HIJACK × REFUTES | 166 | yes | no | `REALIGN` | `INVALID_SUPPORT` |
| NUMBER_FLIP × SUPPORTS | 166 | yes | no | `DESUPPORT` (`FLIP`) | `INVALID_SUPPORT` |
| NUMBER_FLIP × REFUTES | 167 | yes | no | `REALIGN` | `INVALID_SUPPORT` |
| **control items (all parents)** | 1000 | no | no | — | `VALID_SUPPORT` |

`NUMBER_FLIP` retains its name: the `SUPPORTS` half is a literal ±10% flip, and
the `REFUTES` half is a flip *onto the evidence value*. Both move exactly one
numeric span.

---

## 3. Human certificate-validity adjudication (correction 3)

**Frozen in full before any candidate is inspected.** The protocol document is
hashed and that hash is bound into `dataset_release_root` (§7), so the freeze is
verifiable rather than asserted. No LLM judge, at any stage, for any purpose.

### 3.1 Population

| stratum | n adjudicated | rule |
|---|---|---|
| treatment items | **1000 — census** | every treatment item is adjudicated |
| control items | **200 — deterministic subsample** | the 200 with the smallest `SHA256(LP("ADVINT-CTRLSAMP-v1_1") ‖ LP(pool_content_root) ‖ LP(item_id))` |
| SSH type-plausibility rider | 100 | §6, riding on the same pass |

**Why a census of treatments, not a sample.** The adjudicated value *is* the
record's target and is released in the dataset. A sample would ship ~950 records
whose target is a guess. Sampling is only defensible when the estimate, not the
per-record label, is the deliverable.

**Why a subsample of controls.** The control target rests on FEVER's own
annotation, which is the accepted basis of the main FEVER cell and is not being
re-litigated here. The 200 exist to *estimate the control label-noise rate* with
a stated interval, because that rate is the denominator correction for `CFRR`
(§14). A census of controls would double the annotation budget to re-derive a
label the benchmark already provides.

Total judgments: `(1000 + 200) × 2` primary `+` disagreements `≈ 2,600`.

### 3.2 Annotators

- **3 annotators minimum**: 2 independent primary annotators per item, plus 1
  adjudicator who sees only disagreements.
- No annotator may have contributed to protocol design, and the adjudicator may
  not be a primary annotator on any item they adjudicate.
- **Qualification:** fluent English; completion of the frozen 30-item calibration
  set; `≥ 0.80` exact agreement with the calibration gold to proceed.
- **Calibration items are drawn from FEVER parents that are NOT in the pool**
  (deterministically: the 30 lowest `SHA256(LP("ADVINT-CALIB-v1_1") ‖ LP(id))`
  among admissible parents excluded by quota). Drawing them from the pool would
  contaminate the very items being adjudicated.

### 3.3 Blinding

Hidden from annotators, without exception:

- the attack family;
- the intended target / `support_relation_hypothesis`;
- `source_reference_label`;
- whether the item is a control or a treatment;
- the parent item (for treatments, showing the parent would leak the
  perturbation directly);
- the partition;
- all protocol constants.

Items are presented in an order fixed by
`SHA256(LP("ADVINT-ADJORDER-v1_1") ‖ LP(pool_content_root) ‖ LP(annotator_id) ‖ LP(item_id))`
ascending — deterministic, per-annotator distinct, and precommitted, so ordering
cannot be adjusted after inspection.

### 3.4 What the annotator sees

```
CLAIM:     <claim text>
EVIDENCE:  <single evidence sentence>
VERDICT:   SUPPORTED | REFUTED
```

Nothing else. No page title, no URL, no id, no metadata.

### 3.5 The exact question (frozen wording)

> Read the CLAIM, the EVIDENCE sentence, and the VERDICT.
>
> Judge one thing only: **does the evidence sentence, by itself, establish that
> verdict for that claim?**
>
> - Use only the evidence sentence. Do not use outside knowledge, and do not
>   look anything up.
> - Do not judge whether the claim is true. Judge only whether this sentence
>   establishes this verdict.
> - "SUPPORTED" means the sentence shows the claim holds. "REFUTED" means the
>   sentence shows the claim does not hold.
>
> Answer:
> - **VALID** — the evidence sentence alone establishes the stated verdict.
> - **INVALID** — it does not establish the stated verdict (it is about something
>   else, it is insufficient, or it establishes the opposite verdict).
> - **UNCERTAIN** — you cannot decide from this sentence alone.

The instruction "use only the evidence sentence; do not use outside knowledge" is
not a convenience. It is what keeps the adjudication **inside** the `ε_src`
boundary of §1.3: annotators adjudicate the support relation, never world truth.
An adjudication that permitted outside knowledge would reintroduce precisely the
overclaim this revision removes.

### 3.6 Response options and why `UNCERTAIN` is permitted

`VALID` / `INVALID` / `UNCERTAIN`. `UNCERTAIN` is **required**, not merely
permitted. Forcing a binary would manufacture false precision on exactly the
ambiguous items that matter most, and under the fail-closed discipline an item
nobody can adjudicate must be visibly unusable rather than silently assigned.

### 3.7 Disagreement resolution and the final rule

```
two primaries agree on VALID    → VALID_SUPPORT
two primaries agree on INVALID  → INVALID_SUPPORT
two primaries agree on UNCERTAIN→ UNRESOLVED
otherwise (any disagreement, or any single UNCERTAIN)
                                → adjudicator sees the item blinded
                                → majority of the three
                                → no majority (e.g. VALID/INVALID/UNCERTAIN)
                                → UNRESOLVED
```

```
adjudicated_support_relation ∈ {VALID_SUPPORT, INVALID_SUPPORT, UNRESOLVED}
```

### 3.8 Inter-annotator agreement

**Krippendorff's α**, nominal, 3 categories, over the two primary annotators
across all 1,200 adjudicated items — chosen over Cohen's κ because it handles
three categories and unequal item coverage without modification.

- reported with a **bias-corrected bootstrap 95% CI over items**, 10,000
  resamples, seed fixed at 20260907;
- reported per cell as well as overall;
- **precommitted floor: `α ≥ 0.67`.** Below that, the adjudication itself is not
  usable, and no cell may be reported. This is the conventional
  tentative-conclusions threshold and it is fixed *now*, before any judgment
  exists.

### 3.9 Cell-level failure criterion (precommitted)

For each of the 6 treatment cells:

```
confirmation rate  r_cell = #{adjudicated INVALID_SUPPORT} / n_cell
cell PASSES  iff  ClopperPearsonLower(r_cell, n_cell, α = 0.05) ≥ 0.85
```

For the control stratum:

```
control validity   r_ctrl = #{adjudicated VALID_SUPPORT} / 200
controls PASS iff  ClopperPearsonLower(r_ctrl, 200, α = 0.05) ≥ 0.90
```

The **lower** confidence bound is used, never the point estimate. A cell that
confirms at 0.87 with `n = 166` has a lower bound near 0.81 and **fails** — the
fail-closed choice, and the reason to state the rule in terms of the bound.

The control floor is set higher (0.90 vs 0.85) because control validity is
inherited from FEVER rather than constructed by us; if FEVER's own annotation
does not survive blinded re-adjudication at 0.90, the problem is upstream of this
protocol and the cell cannot rest on it.

### 3.10 Mismatch handling — the decisive question

The three candidate responses, decided separately for individual items and for
cells:

**(1) Candidate replacement — PROHIBITED ABSOLUTELY.**
Replacing an item whose adjudication disagreed with the hypothesis conditions the
released pool on the adjudicators' judgments. Three consequences, each fatal:
the pool ceases to be a deterministic function of the pinned source; the reported
confirmation rate becomes ~1.0 *by construction* and therefore uninformative; and
the door opens to iterating until quotas are satisfied, which is cherry-picking
with extra steps. There is no code path for replacement and no exception.

**(2) Relabelling — PERMITTED, and it is simply recording the finding.**
For an individual item, `adjudicated_support_relation` is authoritative. The
record is **retained** in the release with its adjudicated value, and
`support_relation_hypothesis` is retained *alongside* it so the disagreement is
permanently visible and auditable. This is not "relabelling to rescue quotas":
nothing is swapped, nothing is dropped, and the pool is unchanged. Items
adjudicated `VALID_SUPPORT` or `UNRESOLVED` are excluded from the
invalid-acceptance denominator (§14) and their counts are reported.

**(3) Protocol failure — the response at CELL level.**
If any cell fails §3.9, or if `α < 0.67`, **`ADVERSARIAL_INTEGRITY_PROTOCOL_V1_1`
FAILS**. The cell is not reported, no threshold is renegotiated, no items are
replaced. The protocol must be revised, renamed `V1_2`, and reconstructed from
the pinned source.

**Evaluating the stated preference.** The preference —
*deterministic construction → blinded adjudication → pre-frozen threshold →
protocol failure rather than silent replacement* — is correct, and I adopt it
without modification. The argument for it is not merely hygienic. Replacement
would make the pool a function of adjudication outcomes, which destroys the one
property that makes the whole protocol worth its complexity: that two independent
implementers reach the same pool root from the pinned source alone. An adjudicated
replacement loop is not reproducible by construction, because a second team's
annotators would remove different items. Fail-closed is the only rule compatible
with `DETERMINISTIC_CANDIDATE_CONSTRUCTION = YES`.

The one refinement I would add: the failure must be **legible**. On failure the
protocol emits the per-cell confirmation rates, their intervals, `α`, and the
full disagreement matrix, so `V1_2` is designed against evidence rather than
intuition. A failure that produces only "FAILED" wastes the annotation budget.

### 3.11 Freeze order (verifiable)

```
1. freeze this document            → adjudication_protocol_sha256
2. pin the source                  → source_revision            (§11)
3. construct the pool              → pool_content_root          (§7)
4. assign partitions               → partition_manifest_root    (§7)
5. compute                            dataset_release_root      (§7)
6. ONLY THEN may any annotator see any item
```

`adjudication_protocol_sha256` is bound into `dataset_release_root` at step 5,
before step 6. A protocol edited after annotators began would produce a different
hash and a different release root, so the freeze is checkable by a third party
rather than taken on trust.

---

## 4. Citation-swap target validity (correction 4)

V1 rule **C3** — the primary entity's surface form is absent from the donor
sentence — was described as a "non-support guarantee". **It is not a guarantee**,
and the word is withdrawn. Aliases (`JFK` / `John F. Kennedy`), pronouns and
zero anaphora, coreference across the donor's own context, category-level
statements that entail the claim indirectly, and numeric or date coincidences all
defeat a surface-string test.

C3 is retained, **as a deterministic hard-negative construction heuristic**. It
is a good one: it reliably produces sentences that are topically adjacent and
almost never supportive, which is exactly what a hard negative should be. It is
simply not a proof, and the protocol no longer treats it as one.

### 4.1 What is guaranteed mechanically vs adjudicated

| property | status |
|---|---|
| `perturbed_claim` is byte-identical to `original_claim` | **mechanically guaranteed** (asserted in code, §19) |
| `perturbed_evidence` is a real sentence from the pinned corpus, resolving from its `(page, sentence_id)` pointer | **mechanically guaranteed** |
| donor page differs from every page in the parent's evidence | **mechanically guaranteed** |
| the primary entity's **surface string** does not occur in the donor | **mechanically guaranteed** |
| donor and claim share ≥ 1 non-primary gazetteer entity (tier 1) | **mechanically guaranteed when tier 1 fires**; recorded per record |
| token Jaccard ≥ 1/10 | **mechanically guaranteed** |
| the donor does not refer to the primary entity by an alias, pronoun or description | **NOT guaranteed** → adjudicated |
| the donor does not indirectly entail the claim | **NOT guaranteed** → adjudicated |
| the certificate is invalid | **NOT guaranteed** → adjudicated (§3), thresholded per cell (§3.9) |

The same distinction applies to the claim-side families, and is stated once here
rather than repeated: `DESUPPORT` and `REALIGN` guarantee *which span moved and
what it became*; they do not guarantee what the support relation became.

---

## 5. Audit-channel ground truth removed (correction 5)

**`expected_failed_channel` is deleted from the canonical schema.**

The reasoning is the project's own. The five audit channels — `IntFail`,
`ReplayFail`, `DriftFail`, `CheckFail`, `CovGap` — are a **different layer** from
the four acceptance conjuncts `V_H · V_Π · V_Γ · V_⊢`. They are not one-to-one
aliases, and no frozen taxonomy in the project defines "the support relation is
invalid" ⇒ "`CheckFail` fires" as a definitional identity. Channel firing is a
property of an *observing system's* audit machinery: whether `CheckFail` fires on
an invalid certificate depends on the checker's sensitivity, which is exactly the
quantity under measurement. Encoding it as dataset truth would make the dataset
assert the answer to its own question.

I could not demonstrate the required exact mapping, so under the instruction the
field is removed rather than retained.

**What replaces it.** The dataset target is:

```
support_relation_target      (adjudicated; the gold)
certificate_validity_target  (derived from it, §1.4)
```

The observed channel is an **empirical system output**, recorded per system per
item at evaluation time, never in the dataset:

```
observed_channels ⊆ {IntFail, ReplayFail, DriftFail, CheckFail, CovGap}
```

Channel localization is then reported as a measured association between
`observed_channels` and `adjudicated_support_relation` — a finding — rather than
as agreement with a target we wrote down in advance. §14.4 states which channel
denominators remain undefined.

---

## 6. U3 and U4 precommitted (correction 6)

### 6.1 U3 — SSH type plausibility → **Option 1, frozen**

Because an implausible replacement could make `SEMANTIC_SLOT_HIJACK` detectable
on surface type mismatch alone, this diagnostic can trigger protocol change, so
it is frozen now in full.

| element | frozen value |
|---|---|
| sample size | 100 SSH treatment items |
| sampling rule | the 100 smallest `SHA256(LP("ADVINT-SSHDIAG-v1_1") ‖ LP(pool_content_root) ‖ LP(item_id))` among the 333 SSH items |
| rides on | the §3 adjudication pass, as a second question on those items only |
| question (frozen) | "Here are two names: **A** and **B**. Ignoring whether the sentence is true, is **B** the same *kind of thing* as **A** — the same category of person, place, organisation, work or object? Answer YES, NO, or UNCERTAIN." (`A` = `slot_original`, `B` = `slot_replacement`, presented without the claim) |
| annotators | the same 2 primaries; disagreement → adjudicator, majority of 3 |
| statistic | proportion answering YES |
| threshold | `ClopperPearsonLower(p, 100, α = 0.05) ≥ 0.80` |
| failure action | the `SEMANTIC_SLOT_HIJACK` cells fail → **protocol V1.1 fails** |

Presenting the two names *without* the claim is deliberate: it measures type
compatibility, not plausibility-in-context, and so cannot leak the target.

### 6.2 U4 — citation-swap tier-1 fraction → **Option 2, descriptive-only**

`citation_swap_tier1_fraction` is recorded in the manifest and is **explicitly
descriptive**. **No modification of V1.1 may be made on the basis of its value**,
before or after construction. That prohibition is part of the protocol.

**Why descriptive-only is the stronger choice here, not the weaker one.** A
threshold requires a defensible number, and I have no basis for one: the tier-1
availability rate depends on how many FEVER claims contain a second gazetteer
entity, which I cannot estimate without running the construction. Inventing a
number now and calling it precommitted would be precommitment theatre — worse
than honest description, because it dresses a guess as a standard.

The risk U4 was meant to control — that citation swap is solvable by string
matching — is instead controlled by measuring the shortcut **directly**:

```
LEXICAL_OVERLAP_BASELINE   (mandatory, §14.5)
    reject iff the claim's primary entity surface does not occur in the evidence
```

This is a precommitted, interpretable, threshold-free control: it states exactly
how much of any system's score on the citation-swap cell is available from string
matching. Reporting the citation-swap result without it is prohibited (§15,
`BLOCKS_MANUSCRIPT_REPORTING`).

---

## 7. Release identity: three roots (correction 7)

V1 bound only the records. Two releases with identical records but different
pilot/final assignment would have shared a root — a real hazard, since partition
assignment is exactly what a garden-of-forking-paths attack would vary.

### 7.1 `pool_content_root` (records only)

Records sorted ascending by `instance_id` as UTF-8 bytes:

```
pool_content_root := SHA256( LP("ADVINT-POOL-v1_1") ‖ LP(protocol_version)
                           ‖ LP(source_revision) ‖ n.to_bytes(8,"big")
                           ‖ raw32(record_sha256[0]) ‖ … ‖ raw32(record_sha256[n-1]) )
```

`record_sha256` excludes `partition` and all adjudication fields, so a record's
identity is fixed by construction and cannot move when a partition rule changes
or an adjudication lands.

### 7.2 `partition_manifest_root` (the assignment itself)

The partition manifest is the canonical-JSON array of
`{"instance_id": …, "partition": …}` in the same canonical order:

```
partition_manifest_root := SHA256( LP("ADVINT-PARTMAN-v1_1")
                                 ‖ LP(partition_rule_id) ‖ LP(partition_rule_version)
                                 ‖ n.to_bytes(8,"big")
                                 ‖ LP(instance_id[0]) ‖ LP(partition[0])
                                 ‖ … ‖ LP(instance_id[n-1]) ‖ LP(partition[n-1]) )
```

Every `(instance_id, partition)` pair is length-prefixed individually, so the
assignment is bound element-by-element and not merely summarised.

### 7.3 `dataset_release_root` (the release)

```
dataset_release_root := SHA256( LP("ADVINT-RELEASE-v1_1")
                              ‖ LP(protocol_version)              # "..._V1_1"
                              ‖ LP(source_revision)
                              ‖ LP(pool_content_root)
                              ‖ LP(partition_rule_id)
                              ‖ LP(partition_rule_version)
                              ‖ LP(partition_manifest_root)
                              ‖ LP(adjudication_protocol_sha256) )
```

**Required property, and it holds by construction:** two releases with identical
records but any difference in the `instance_id → partition` map produce different
`partition_manifest_root` values, and therefore different `dataset_release_root`
values, while sharing `pool_content_root`. Repartitioning is thus always visible
as a new release, never as a silent edit of an existing one.

Binding `adjudication_protocol_sha256` into the release root is what makes §3.11's
freeze checkable: the release cannot claim an adjudication protocol it did not
commit to before annotation began.

---

## 8. Partition feasibility, fail-closed (correction 8)

### 8.1 The rule (carried from V1, renamed and versioned)

```
partition_rule_id      = ADVINT-PART-BUCKET1000
partition_rule_version = v1_1

b(instance_id) := int.from_bytes(
    SHA256( LP("ADVINT-PART-v1_1") ‖ LP(protocol_version) ‖ LP(instance_id) )[0:8],
    "big") mod 1000

   0 ≤ b ≤  19  → SMOKE                (≈  2%)
  20 ≤ b ≤ 119  → CHECKER_CALIBRATION  (≈ 10%)
 120 ≤ b ≤ 269  → PILOT                (≈ 15%)
 270 ≤ b ≤ 999  → FINAL                (≈ 73%)
```

Partition is assigned to the **pair**, so `c_i` and `t_i` always land in the same
partition. A control in FINAL whose treatment sat in PILOT would break the paired
analysis of §2.5 and leak the treatment through its partner.

### 8.2 Feasibility, and what happens when it fails

`N_final_required` is set by Gate-0.2 and is **not yet frozen**. The realised
FINAL count `n_final` is a deterministic function of the pool, computable
immediately after §19 step 13 — and, critically, **before any evaluation output
exists**.

```
if n_final ≥ N_final_required:  proceed
else:                           ESCALATE — protocol/sample-size INFEASIBLE, halt
```

**Explicitly prohibited on the failure branch:**

- repartitioning, re-salting, or adjusting bucket boundaries;
- promoting `PILOT` or `CHECKER_CALIBRATION` items into `FINAL`;
- re-running construction with a different `protocol_version` string to move the
  hash;
- reducing `N_final_required` after seeing `n_final`.

Each of those turns the release root into a quantity selected by the experimenter,
which is precisely the garden of forking paths §7 exists to close.

**The one permitted remedy, and its conditions.** The partition rule may be
changed only if **all** hold: (a) no FINAL item has been evaluated by any system;
(b) the change is a new `partition_rule_version` with a new
`partition_manifest_root` and a new `dataset_release_root`; (c) both the old and
the new roots are recorded, so the change is permanently visible; (d) the new
`N_final_required` was fixed before the new rule was chosen. Under those
conditions it is a versioned redesign, not a silent repartition.

**Expected margin.** At ≈73%, `n_final ≈ 730` pairs — 730 control and 730
treatment items. Any Gate-0.2 requirement above roughly 700 pairs should be
treated as tight and checked at freeze time rather than discovered late.

---

## 9. Corrected manuscript wording (correction 9)

### 9.1 The specific false statement, withdrawn

V1's manuscript text said:

> "In 334 examples the claim remains world-true while its certificate becomes
> invalid."

**This is false twice over,** and the objection is right on both counts.

1. It asserts world truth, which FEVER does not supply (§1).
2. Even reading "world-true" charitably as "corpus-supported", it is wrong for
   **167 of the 334**: half the citation-swap allocation has `REFUTES` parents,
   whose claims are corpus-*contradicted*. Those claims were never "true" in any
   reading.

The correct statement is about the *reference label and the evidence relation*,
not about truth.

### 9.2 Replacement text (proposed; the manuscript is not edited here)

> **Adversarial integrity (adversarial\_integrity).** We construct a controlled
> stress test of certificate validity from FEVER 1.0 (Thorne et al., 2018),
> using a single split and pinning the source files by content hash. We retain
> `SUPPORTS` and `REFUTES` claims and exclude `NOT ENOUGH INFO`, which carries no
> evidence and therefore no support relation to perturb. We treat the FEVER label
> as a **reference annotation relative to that frozen corpus**, not as external
> world truth; no record in this cell asserts a world-truth value, and the cell
> is not evidence about `ε_src`.
>
> The unit is a **parent/attack pair**: 1,000 parents, each used at most once,
> each receiving exactly one deterministic perturbation. Each pair yields an
> unperturbed **control** certificate and a perturbed **treatment** certificate,
> presented to systems independently. Three non-generative attack families are
> applied in near-equal thirds. **Citation swap** (334) leaves the claim
> byte-identical and replaces its evidence with a lexically similar corpus
> sentence that does not mention the claim's primary entity. **Semantic slot
> hijack** (333) and **number flip** (333) leave the evidence unchanged and move
> exactly one span in the claim: for `SUPPORTS` parents, away from what the
> evidence attests; for `REFUTES` parents, onto what the evidence attests, so that
> the claim becomes evidence-aligned and the asserted `REFUTED` verdict is no
> longer grounded. No language model is used at any point in construction, and
> the pool, its partitions and its release identity are reproducible from the
> pinned source to a single release root.
>
> The cell separates two quantities that ordinary fact verification conflates.
> The `source_reference_label` records FEVER's annotation of the original claim
> against the frozen corpus. The `certificate_validity_target` records whether the
> supplied evidence grounds the asserted verdict, and is established by **blinded
> human adjudication** of every treatment item — two independent annotators,
> `UNCERTAIN` permitted, family and intended target hidden — rather than assumed
> from the construction. In the 334 citation-swap examples the parent claim and
> its FEVER reference label are unchanged while the supplied evidence relation is
> corrupted; a system that classifies claims rather than checking certificates
> gains nothing on them.
>
> Because construction cannot guarantee a support relation in natural language,
> every cell carries a precommitted confirmation threshold: a cell is reportable
> only if the lower 95% bound on its adjudicated-invalid rate reaches 0.85, and
> the protocol fails rather than substituting examples if it does not. We report
> invalid acceptance and control false rejection jointly, never as a single
> scalar, with the paired 2×2 outcome table and mandatory lexical baselines.
>
> The dataset carries no audit-channel target: the five audit channels are a
> different layer from the acceptance conjuncts, and channel firing is reported
> as an observed system output rather than as dataset ground truth. Evidence is
> restricted to single-sentence FEVER annotations, so multi-hop certificates are
> outside its scope.

---

## 10. U5 as a pre-freeze condition (correction 10)

```
MAIN_FEVER_FINAL_SPLIT = UNRESOLVED_PENDING_REPOSITORY_CONFIRMATION
SOURCE_SPLIT           = train   (CONDITIONAL on the above)
```

V1 asserted that the main FEVER final-evaluation cell uses `shared_task_dev` and
then reasoned that `train` is structurally disjoint from it. The premise was
never verified — I was, and remain, instructed not to re-audit the repository.

**Parent-disjointness is therefore not established, and V1.1 does not claim it.**
The status is:

- *claimed:* nothing;
- *conditional:* if the main cell is confirmed to evaluate on `shared_task_dev`,
  then `SOURCE_SPLIT = train` gives structural disjointness, verified additionally
  by explicit id-set intersection at construction time (§19 step 3);
- *if the main cell uses `train`:* `SOURCE_SPLIT` must be re-decided. The
  replacement is `paper_test` **with an explicit id-level exclusion** of every
  main-cell parent, since the FEVER "paper" splits are partitions of the same dev
  pool as `shared_task_dev` and are therefore not disjoint from it by file.

U5 is classified `BLOCKS_PROTOCOL_FREEZE` in §15. Construction may not begin
while it is open.

---

## 11. Source pin (correction 11)

The V1 content-hash ceremony is retained verbatim:

```
SOURCE_REVISION := SHA256( LP("ADVINT-SRCREV-v1") ‖ LP("fever-1.0")
                         ‖ LP(sha256(train.jsonl)) ‖ LP(sha256(wiki-pages.zip)) )
```

with per-shard hashes for all 109 `wiki-NNN.jsonl` files recorded separately so a
corrupted extraction is detectable independently of the archive.

```
SOURCE_REVISION = UNRESOLVED_PENDING_PIN
```

acceptable at **protocol-candidate stage only**. Added in V1.1 as a hard
construction guard:

> The constructor **refuses to start** unless `source_revision` matches
> `^[0-9a-f]{64}$` and equals the value recomputed from the pin file. There is no
> flag, environment variable or configuration that relaxes this. No placeholder
> source revision can appear in a generated dataset, because a placeholder cannot
> get past the first statement of the program.

Classified `BLOCKS_DATASET_CONSTRUCTION` in §15.

---

## 12. Preserved deterministic machinery (correction 12)

Carried from V1 **unchanged**. Section references are to V1 unless noted.

| machinery | status in V1.1 |
|---|---|
| `SOURCE_DATASET = FEVER_ONLY`, official shared-task distribution | unchanged |
| single-hop evidence restriction (§1.5) | unchanged |
| all §1.5 admissibility filters | unchanged |
| `N = 1000`; 500 `SUPPORTS` / 500 `REFUTES` parents | unchanged |
| family quotas 334 / 333 / 333 and the label cross-allocation | unchanged |
| one parent used at most once; global `claimed_parents` | unchanged |
| no LLM anywhere in construction | unchanged |
| `SEMANTIC_SLOT_HIJACK` replacing `paraphrase_hijack` | unchanged |
| gazetteer `G` and copula type signature `τ` (§4.2) | unchanged |
| exact decimal arithmetic; `ROUND_HALF_UP`; no binary floats | unchanged (used by `FLIP`; `REALIGN` needs no rounding) |
| numeric parser, eligible/excluded classes (§6.1–6.2) | unchanged |
| domain-separated length-prefix hashing `LP(x)` (§7.1) | unchanged |
| deterministic source ranking `rank_key` (§7.2) | unchanged |
| deterministic donor ranking and tie-breaking (§4.4, §5.6) | unchanged |
| integer cross-multiplication for Jaccard comparison (§5.5) | unchanged |
| no stochastic fallback anywhere | unchanged |
| `DONOR_REUSE_CAP = 3` | unchanged |
| cell processing order `NF → SSH → CS` (§8.3) | unchanged |
| exact failure on quota exhaustion (§8.4) | unchanged |
| text normalisation (§11.1), canonical JSON (§11.2), frozen lexicons (§11.4) | unchanged |
| per-record SHA-256, canonical sort order | unchanged |
| pool content root | unchanged in construction, renamed `pool_content_root` |
| duplicate policy (§12) | unchanged |

**Modified by the corrections, and only these:**

| item | change | driver |
|---|---|---|
| `source_world_label` → `source_reference_label`; `source_truth_changed` deleted | ontology | C1 |
| control design: 667/333 → paired parent controls | semantics | C2 |
| `REFUTES`-side claim attacks: preserve-refutation → `REALIGN` | semantics | C2 |
| V1 rule D3 inverted for `REFUTES` parents | follows from `REALIGN` | C2 |
| `expected_failed_channel` removed | taxonomy | C5 |
| adjudication fields added | new stage | C3 |
| `partition_manifest_root`, `dataset_release_root` added | release identity | C7 |
| `partition_rule_version` added; partition assigned per pair | feasibility, pairing | C8, C2 |

### 12.1 `DESUPPORT` and `REALIGN` — exact rules

Both reuse V1's slot eligibility, numeric parser, ranking and caps. Only the
donor-admissibility direction is label-conditional.

**`SEMANTIC_SLOT_HIJACK`, `SUPPORTS` parent — `DESUPPORT`.** Exactly V1 §4.3–4.5.
Slot must appear in `E` (V1 rule S6); donor must **not** appear in `E`
(V1 rule D3); donor ranked by hash; tiers 1 (exact `τ`) then 2 (`τ_head`).

**`SEMANTIC_SLOT_HIJACK`, `REFUTES` parent — `REALIGN`.** Replaces V1's
`REFUTES` branch:

- **slot eligibility** — as V1 S1–S5, plus S6′: the slot's surface must **not**
  appear in `E` (it is the corpus-contradicted element);
- **donor pool** — the gazetteer entities **occurring in `E`**, found by the same
  leftmost-longest scan;
- **donor admissibility** — `τ(donor)` defined and equal to `τ(slot)` (tier 1) or
  `τ_head` equal (tier 2); donor ≠ `prim`; donor not already in `C`; donor and
  slot not substrings of one another; `DONOR_REUSE_CAP = 3`;
- **ranking** — the V1 §4.4 hash order, unchanged;
- **exhaustion** — tier 1 → tier 2 → next eligible slot → next parent. No
  resampling.

**`NUMBER_FLIP`, `SUPPORTS` parent — `DESUPPORT` (`FLIP`).** Exactly V1 §6.3–6.7:
target value must appear in `EVNUMS`; hash-governed sign; `±10%`; exact decimal;
`ROUND_HALF_UP`; all V1 §6.6 degenerate-case rules; `n' ∉ EVNUMS`.

**`NUMBER_FLIP`, `REFUTES` parent — `REALIGN`.** Replaces V1's `REFUTES` branch:

- **target eligibility** — the claim span is eligible per V1 §6.2 and its value is
  **not** in `EVNUMS`;
- **replacement value** — the numeral in `E` that is (a) of the **same eligible
  class** (`MONEY` / `PERCENT` / `MEASUREMENT` / `CARDINAL`) and (b) carries the
  **byte-identical unit or currency token**, where the class has one. No unit
  conversion is ever performed — conversion is arithmetic on an inferred
  semantics, and inferring semantics is the thing this protocol refuses to do;
- **uniqueness** — exactly one numeral in `E` may satisfy (a) and (b). If two
  do, the target is ineligible, because which one the claim "should" align to is
  a judgment;
- **surface form** — the evidence numeral is copied **verbatim**, including its
  own grouping and precision. No rounding, no reformatting;
- **validation** — the resulting value differs from the original; the claim
  changed in exactly one contiguous span; the evidence is byte-identical to the
  parent's;
- **exhaustion** — next eligible span → next parent.

`REALIGN` is deterministic and requires no arithmetic at all, which makes it
*more* reproducible than `FLIP`, not less.

---

## 13. Revised canonical record schema (correction 13)

Three groups, kept visually and structurally separate, because conflating them is
the error V1.1 exists to correct.

```jsonc
{
  // ─────────── GROUP A — CONSTRUCTED (mechanically derived; deterministic) ───────────
  "dataset_id":                 "adversarial_integrity",
  "protocol_version":           "ADVERSARIAL_INTEGRITY_PROTOCOL_V1_1",
  "pair_id":                    "advint-v1_1:<parent_id>:<attack_family>",
  "item_id":                    "<pair_id>#control" | "<pair_id>#treatment",
  "item_role":                  "CONTROL" | "TREATMENT",

  "source_dataset":             "FEVER_ONLY",
  "source_revision":            "<64 hex>",
  "source_split":               "train",
  "parent_id":                  "<FEVER id, decimal string>",
  "source_reference_label":     "SUPPORTS" | "REFUTES",   // FEVER annotation vs the pinned corpus
  "asserted_verdict":           "SUPPORTED" | "REFUTED",  // = source_reference_label, as shown to the system

  "original_claim":             "<string>",
  "original_evidence":          { "page": "<id>", "sentence_id": <int>, "text": "<string>" },
  "presented_claim":            "<string>",               // original for CONTROL, perturbed for TREATMENT
  "presented_evidence":         { "page": "<id>", "sentence_id": <int>, "text": "<string>" },

  "attack_family":              "CITATION_SWAP" | "SEMANTIC_SLOT_HIJACK" | "NUMBER_FLIP",
  "attack_mode":                "SWAP" | "DESUPPORT" | "REALIGN",
  "attack_parameters":          { /* family-specific, as V1 §9 plus attack_mode */ },

  "claim_text_changed":         true | false,             // byte comparison
  "evidence_text_changed":      true | false,             // byte comparison

  "support_relation_hypothesis":"VALID_SUPPORT" | "INVALID_SUPPORT",  // what construction INTENDED

  // ─────────── GROUP B — HUMAN-ADJUDICATED (blinded; §3) ───────────
  "adjudicated_support_relation":     "VALID_SUPPORT" | "INVALID_SUPPORT" | "UNRESOLVED" | null,
  "adjudication_status":              "ADJUDICATED" | "NOT_IN_ADJUDICATION_POPULATION",
  "adjudication_primary_responses":   ["VALID"|"INVALID"|"UNCERTAIN", "..."],
  "adjudication_adjudicator_response":"VALID"|"INVALID"|"UNCERTAIN"|null,
  "adjudication_protocol_sha256":     "<64 hex>",
  "hypothesis_confirmed":             true | false | null,   // adjudicated == hypothesis; null if UNRESOLVED
  "certificate_validity_target":      "VALID_SUPPORT" | "INVALID_SUPPORT" | "UNRESOLVED" | null,  // §1.4

  // ─────────── GROUP C — EXTERNAL-WORLD QUANTITIES (unknown, by declaration) ───────────
  "external_world_truth_status":      "UNVERIFIED",   // constant; no code path sets any other value

  // ─────────── release metadata ───────────
  "partition":                  "SMOKE" | "CHECKER_CALIBRATION" | "PILOT" | "FINAL",
  "partition_rule_id":          "ADVINT-PART-BUCKET1000",
  "partition_rule_version":     "v1_1",
  "record_sha256":              "<64 hex>"
}
```

### 13.1 Field-group discipline

| group | who writes it | may it change after construction? | in `record_sha256`? |
|---|---|---|---|
| **A — constructed** | the deterministic constructor | **no** | **yes** |
| **B — adjudicated** | blinded human process, once | written once, never revised | **no** |
| **C — external world** | nobody; it is a declaration | no | yes (it is a constant) |
| release metadata | partition rule / hashing | `partition` only via a new `partition_rule_version` and a new release root | `partition` **no**; the rest no |

```
record_sha256 := SHA256( LP("ADVINT-REC-v1_1")
                       ‖ LP(canonical_json(record restricted to GROUP A ∪ GROUP C)) )
```

Group B is excluded from the record hash **deliberately**: `pool_content_root`
must be computable *before* adjudication begins (§3.11), or the freeze order is
impossible. Adjudication results are bound instead through
`adjudication_protocol_sha256` in the release root and through a separate
`adjudication_manifest_root` published with the results.

### 13.2 Field removals versus V1

| removed | reason |
|---|---|
| `source_world_label` | overclaims; replaced by `source_reference_label` (C1) |
| `source_truth_changed` | names a quantity that is not available (C1) |
| `expected_failed_channel` | channels are not conjunct aliases; no frozen mapping demonstrated (C5) |
| `instance_id` | superseded by `pair_id` + `item_id`, since the unit is now the pair (C2) |

---

## 14. Revised metrics (correction 14)

Denominators are restated for the paired design. Every one is either fixed by
construction or fixed by adjudication and reported with its count.

### 14.1 Populations

```
P            = 1000                              parent/attack pairs
T            = 1000                              treatment items
C            = 1000                              control items
T_inv        = #{t : adjudicated = INVALID_SUPPORT}      ≤ 1000, known post-adjudication
T_val        = #{t : adjudicated = VALID_SUPPORT}        reported, not discarded
T_unres      = #{t : adjudicated = UNRESOLVED}           reported, not discarded
C_adj        = 200                               adjudicated control subsample
```

`T_inv + T_val + T_unres = 1000` is asserted and reported. No treatment item is
dropped from the release; items outside `T_inv` are excluded from the
invalid-acceptance denominator and counted explicitly.

### 14.2 Primary metrics — reported jointly, never collapsed

| metric | numerator | denominator |
|---|---|---|
| **IAR** — invalid acceptance rate | accepted ∧ adjudicated `INVALID_SUPPORT` | `T_inv` |
| **CFRR** — control false-rejection rate | rejected ∧ `item_role = CONTROL` | `1000` |
| **CFRR_adj** — noise-corrected control false rejection | rejected ∧ control ∧ adjudicated `VALID_SUPPORT` | `#{C_adj adjudicated VALID_SUPPORT}` |

`IAR` and `CFRR` are **never combined into a single scalar.** A scalar permits an
invisible trade between accepting invalid certificates and rejecting valid ones,
which is the exact failure the certificate discipline exists to prevent. A system
is better only if it does not worsen either.

`CFRR_adj` exists because §3.1 adjudicates only 200 controls: it corrects the
denominator for control label noise, and its interval is wider by construction.
Both are reported; neither replaces the other.

### 14.3 The paired analysis (new in V1.1)

The pair is the unit, so the primary table is the 2×2 over pairs:

| | treatment **rejected** | treatment **accepted** |
|---|---|---|
| **control accepted** | `n₁₁` — correct discrimination | `n₁₂` — invalid accepted |
| **control rejected** | `n₂₁` — valid rejected | `n₂₂` — no discrimination |

- `PairedDiscrimination = n₁₁ / P` — the fraction of pairs on which the system
  accepts the valid certificate **and** rejects the corrupted one;
- reported with a **McNemar exact test** on the discordant cells `(n₁₂, n₂₁)` and
  an exact 95% interval;
- the **full 2×2 table is always printed.** `PairedDiscrimination` alone is
  insufficient: `n₁₂` and `n₂₁` are different failures and must not be pooled.

Pairing removes parent-level variance from the comparison, which is the second
reason §2.2 preferred Direction C over Direction A.

### 14.4 Per-cell and per-family reporting

Reported separately, always with `n`, never pooled across cells with different
attack modes:

| cell | pairs | adjudicated-invalid denominator |
|---|---|---|
| CITATION_SWAP × SUPPORTS | 167 | reported |
| CITATION_SWAP × REFUTES | 167 | reported |
| SEMANTIC_SLOT_HIJACK × SUPPORTS (`DESUPPORT`) | 167 | reported |
| SEMANTIC_SLOT_HIJACK × REFUTES (`REALIGN`) | 166 | reported |
| NUMBER_FLIP × SUPPORTS (`DESUPPORT`) | 166 | reported |
| NUMBER_FLIP × REFUTES (`REALIGN`) | 167 | reported |

`DESUPPORT` and `REALIGN` cells are **never pooled**: they are different
interventions with different shortcut profiles (§17 R11).

### 14.5 Undefined quantities stay undefined

```
IntFail    denominator : UNDEFINED — no integrity perturbation exists in this cell
ReplayFail denominator : UNDEFINED — no replay intervention exists
DriftFail  denominator : UNDEFINED — no snapshot or policy drift exists
CovGap     denominator : UNDEFINED unless the coverage sampler is run over this cell
CheckFail                : reported as an OBSERVED system output (§5), associated with
                           adjudicated_support_relation; it is not a target and no
                           agreement-with-target metric is defined for it
family-detection accuracy: UNDEFINED for any system emitting no machine-readable
                           rejection reason; not imputed, not scored as 0
any metric over T_unres  : UNDEFINED
```

`UNDEFINED` is reported as `UNDEFINED`, never as `0`. A rate of 0 asserts that the
event was measured and did not occur.

### 14.6 Mandatory reference baselines

No cell result may be reported without these on the same table:

1. **ALWAYS_ACCEPT** — `IAR = 1.000`, `CFRR = 0.000`, `n₁₁ = 0`.
2. **ALWAYS_REJECT** — `IAR = 0.000`, `CFRR = 1.000`, `n₁₁ = 0`. The paired design
   makes both degenerate strategies score `PairedDiscrimination = 0`, which is
   the property V1's unpaired design had to buy with a special control cell.
3. **LEXICAL_OVERLAP_BASELINE** — reject iff the claim's primary-entity surface
   does not occur in the evidence. This is the citation-swap shortcut, and its
   score is the substitute for U4's withdrawn threshold (§6.2).
4. **EVIDENCE_VALUE_OVERLAP_BASELINE** — *new in V1.1, and required by `REALIGN`*.
   Reject the `REFUTED` verdict iff the claim's perturbed span occurs verbatim in
   the evidence. This measures the shortcut that `REALIGN` necessarily introduces
   (§17 R11) and must be reported beside every `REALIGN` cell.
5. **PERTURBATION_DETECTOR** — reject iff the item differs from its parent. Under
   the paired design it scores `IAR = 0`, `CFRR = 0`, `PairedDiscrimination = 1.0`
   *if and only if* it can see the parent; since items are presented independently
   and unlabelled (§2.5), it is not implementable by an evaluated system. It is
   reported as an **upper reference**, clearly marked as an oracle, to show what
   the paired design would yield to a system with illegitimate access.

---

## 15. Residual gaps, classified (correction 15)

Every residual item is classified. Nothing is left as "human reviews later".

| id | item | classification | resolution |
|---|---|---|---|
| **U1** | `SOURCE_REVISION` component hashes uncomputed | `BLOCKS_DATASET_CONSTRUCTION` | pinning ceremony (§11); constructor refuses to start without a 64-hex value |
| **U5** | main-FEVER final split unconfirmed, so parent-disjointness is unestablished | `BLOCKS_PROTOCOL_FREEZE` | repository confirmation (§10); `SOURCE_SPLIT` re-decided if it fails |
| **U2′** | treatment targets are construction hypotheses until adjudicated | `BLOCKS_SCIENTIFIC_EVALUATION` | §3 census adjudication; per-cell threshold §3.9; protocol fails if unmet |
| **U6** | control label noise unquantified | `BLOCKS_SCIENTIFIC_EVALUATION` | 200-control adjudication (§3.1); floor `CP-lower ≥ 0.90` |
| **U7** | inter-annotator agreement unknown | `BLOCKS_SCIENTIFIC_EVALUATION` | Krippendorff's α with floor 0.67 (§3.8); below it no cell is reportable |
| **U3** | SSH type plausibility | `BLOCKS_SCIENTIFIC_EVALUATION` | frozen diagnostic, 100 items, `CP-lower ≥ 0.80`, failure ⇒ protocol failure (§6.1) |
| **U4** | citation-swap tier-1 fraction | `NONBLOCKING_DIAGNOSTIC` | descriptive-only; **no V1.1 modification may be based on it** (§6.2); shortcut risk carried by baseline 3 instead |
| **U8** | `N_final_required` not frozen; FINAL pool may be insufficient | `BLOCKS_SCIENTIFIC_EVALUATION` | fail-closed feasibility check (§8.2); ESCALATE, never repartition |
| **U9** | `REALIGN` introduces an evidence-overlap shortcut | `BLOCKS_MANUSCRIPT_REPORTING` | `EVIDENCE_VALUE_OVERLAP_BASELINE` mandatory beside every `REALIGN` cell (§14.6) |
| **U10** | citation-swap non-support is heuristic, not proved | `BLOCKS_MANUSCRIPT_REPORTING` | §4.1 guarantee table must be reproduced or cited; "guarantee" language prohibited |
| **U11** | single-hop restriction limits scope | `NONBLOCKING_DIAGNOSTIC` | stated as a scope limit in §9.2; V2 extension |

```
BLOCKS_PROTOCOL_FREEZE       : 1   (U5)
BLOCKS_DATASET_CONSTRUCTION  : 1   (U1)
BLOCKS_SCIENTIFIC_EVALUATION : 5   (U2′, U6, U7, U3, U8)
BLOCKS_MANUSCRIPT_REPORTING  : 2   (U9, U10)
NONBLOCKING_DIAGNOSTIC       : 2   (U4, U11)
UNRESOLVED_BLOCKING_GAPS     : 9
```

---

## 16. Theory connection, corrected

**`ε_src`.** The cell is now *outside* it by construction (§1.3), and this is the
principal correction of V1.1. Every target is relative to the supplied evidence;
annotators are forbidden from using outside knowledge; no field records world
truth. **Prohibition: this cell may never be cited as bounding, estimating or
constraining `ε_src`.** V1's controls violated exactly this, which is why they
are gone.

**`ε_tax`.** Unchanged from V1: the three families are in-taxonomy by
construction, so the cell cannot estimate `ε_tax` and must not be cited as
bounding it.

**Checker-relative acceptance.** `Check(Z;G_t) = V_H · V_Π · V_Γ · V_⊢` is
relative to a fixed checker. The cell holds `V_H`, `V_Π` and `V_Γ` constant and
perturbs only the input to `V_⊢`, which is what licenses the §1.4 reduction from
support relation to certificate validity — as a **scoping assumption of this
cell**, not as a general identity.

**Separating witnesses.** Sharpened by the correction. The 334 citation-swap
pairs are separating witnesses between *"the parent claim and its FEVER reference
label are unchanged"* and *"the supplied evidence grounds the asserted verdict"*.
This is a corpus-relative separation, which is the strongest form available and
the honest one. V1 described it as a world-truth separation, which it never was.

**Audit-channel coverage.** The cell contributes no channel ground truth at all
(§5). It supplies items on which channel behaviour can be *observed*; a coverage
claim may cite that observation but may not treat it as agreement with a target.

**This dataset proves no theorem.** It can falsify an operational claim; it
cannot establish one.

---

## 17. Construct validity, updated

V1's R1–R10 carry over. Restated where the corrections changed them, plus one new
risk created by `REALIGN`.

**R2 (label leakage) — materially improved.** V1's targets were a deterministic
function of `source_reference_label` within SSH and NF. Under V1.1 every treatment
hypothesis is `INVALID_SUPPORT` and every control is `VALID_SUPPORT`, so the
target is a function of `item_role`, not of the label. The degenerate strategy V1
needed a special cell to defeat is defeated by the design itself: both
`ALWAYS_ACCEPT` and `ALWAYS_REJECT` score `PairedDiscrimination = 0`.

**R11 (new) — `REALIGN` introduces an evidence-overlap cue.** After realignment
the claim's key span occurs verbatim in the evidence, while its control's does
not. "The value appears in the evidence, so `REFUTED` is wrong" therefore
separates `REFUTES` controls from `REFUTES` treatments almost perfectly.
*Severity: high, and this is the price of `REALIGN`.*
*Analysis:* the heuristic is **correct reasoning**, not cheating — a system using
it is doing shallow but valid inference. What matters is quantifying how much of
a score it buys.
*Mitigation:* `EVIDENCE_VALUE_OVERLAP_BASELINE` is mandatory beside every
`REALIGN` cell (§14.6); `DESUPPORT` and `REALIGN` cells are never pooled (§14.4);
U9 blocks manuscript reporting without the baseline.
*Residual:* real and unremovable. It is the deliberate trade for replacing V1's
undecidable `REFUTES` targets with defensible ones, and it is stated rather than
hidden.

**R12 (new) — adjudicator anchoring.** Blinded annotators seeing many corrupted
items may drift toward `INVALID`. *Mitigation:* controls and treatments are
interleaved in the same blinded stream in a precommitted per-annotator order
(§3.3), and the 200 adjudicated controls act as an internal check — a drift
toward `INVALID` shows up as a low `r_ctrl`, which is thresholded at 0.90 (§3.9).

**R6 (contamination) — status changed.** V1 claimed structural
parent-disjointness. V1.1 claims nothing until U5 closes (§10).

---

## 18. Nonredundancy with the main FEVER cell, corrected

The distinction stands, restated without the world-truth overclaim:

| | main FEVER cell | adversarial_integrity cell |
|---|---|---|
| question | does the retrieved evidence support this claim, per the corpus? | does *this supplied* evidence ground *this asserted verdict*? |
| gold | FEVER reference annotation | blinded human adjudication of the support relation |
| what varies | claims and evidence as they occur | one controlled perturbation, parent and mechanism known |
| counterfactual | none | yes — the paired unperturbed parent certificate |
| unit | claim | parent/attack pair |

The decisive item: in the 334 citation-swap pairs, **the parent claim and its
FEVER reference label are unchanged while the evidence relation is corrupted.**
No sample of ordinary FEVER contains such an item, because ordinary FEVER never
separates the annotation from the supplied evidence. A claim classifier scores at
its base rate on those pairs; only a certificate checker separates them.

---

## 19. Pseudocode — changes from V1 only

V1 §19 steps 1–5, 7 (validation), 8 (quotas), 10 (ids, renamed), 11
(canonicalization), and the failure-on-exhaustion behaviour are carried unchanged.
Only the deltas are given.

```text
6'. TRANSFORMATION — label-conditional mode                                  (§12.1)
    apply_attack("SEMANTIC_SLOT_HIJACK", p, pair_id, ...):
        if p.source_reference_label == "SUPPORTS":   mode := "DESUPPORT"   # V1 §4.3-4.5 verbatim
        else:                                        mode := "REALIGN"
             slots  := [s in eligible_slots(p) if surf(s) NOT in E]        # S6'
             donors := gazetteer_entities_occurring_in(E)
             filter donors by  tau(donor)==tau(slot) (tier 1) or tau_head equal (tier 2),
                               donor != prim, donor not in C,
                               neither a substring of the other,
                               donor_use[donor] < DONOR_REUSE_CAP
             rank donors by V1 §4.4 hash order; take first; splice; validate
    apply_attack("NUMBER_FLIP", p, pair_id, ...):
        if p.source_reference_label == "SUPPORTS":   mode := "FLIP"        # V1 §6.3-6.7 verbatim
        else:                                        mode := "REALIGN"
             for span in eligible_spans(C) with value(span) NOT in EVNUMS:
                 cand := { numerals m in E : class(m)==class(span)
                                         and unit_token(m)==unit_token(span) }
                 if |cand| != 1: continue                                   # uniqueness required
                 m := the single element of cand
                 if value(m) == value(span): continue
                 return OK(presented_claim = splice(C, span, surface(m)),   # verbatim copy
                           presented_evidence = p.evidence)
             return REJECT

9'. PAIR AND ITEM EMISSION                                                   (§2.5, §13)
    for each accepted pair:
        pair_id := "advint-v1_1:" + parent_id + ":" + attack_family
        emit CONTROL   item_id = pair_id + "#control"
             presented_claim = original_claim, presented_evidence = original_evidence
             claim_text_changed = false, evidence_text_changed = false
             support_relation_hypothesis = "VALID_SUPPORT"
        emit TREATMENT item_id = pair_id + "#treatment"
             presented_* = perturbed
             claim_text_changed / evidence_text_changed  := byte comparisons
             support_relation_hypothesis = "INVALID_SUPPORT"
        both items carry external_world_truth_status = "UNVERIFIED"
        adjudication fields = null; adjudication_status set in step 14'

12'. ROOTS                                                                    (§7)
     pool_content_root       := SHA256(... records ...)          # GROUP A ∪ C only
     partition_manifest_root := SHA256(... (instance_id, partition) pairs ...)
     dataset_release_root    := SHA256(LP(protocol_version) ‖ LP(source_revision)
                                     ‖ LP(pool_content_root)
                                     ‖ LP(partition_rule_id) ‖ LP(partition_rule_version)
                                     ‖ LP(partition_manifest_root)
                                     ‖ LP(adjudication_protocol_sha256))

13'. PARTITION — assigned to the PAIR, then feasibility                       (§8)
     for each pair: b := bucket(pair_id); partition(pair) := interval(b)
                    control and treatment inherit the pair's partition
     assert pairwise_disjoint(SMOKE, CHECKER_CALIBRATION, PILOT, FINAL)
     n_final := |{pairs with partition == FINAL}|
     if n_final < N_final_required:  ESCALATE("INFEASIBLE"); HALT   # never repartition

14'. ADJUDICATION INTAKE — strictly after step 12'                            (§3.11)
     assert dataset_release_root is computed and recorded
     assert adjudication_protocol_sha256 == sha256(this document)
     build blinded worklist: all 1000 treatments, 200 hash-selected controls,
                             100 hash-selected SSH type-plausibility riders
     per annotator, order by SHA256(LP("ADVINT-ADJORDER-v1_1") ‖ LP(pool_content_root)
                                  ‖ LP(annotator_id) ‖ LP(item_id)) ASC
     collect responses; resolve per §3.7; write GROUP B fields once, never revised

15'. GATES — fail-closed, no replacement                                      (§3.9)
     if krippendorff_alpha < 0.67:                       FAIL("adjudication unusable")
     for cell in 6 treatment cells:
         if CP_lower(confirm_rate[cell], n[cell], 0.05) < 0.85:  FAIL("cell " + cell)
     if CP_lower(r_ctrl, 200, 0.05) < 0.90:              FAIL("control stratum")
     if CP_lower(ssh_type_plausible, 100, 0.05) < 0.80:  FAIL("SSH type plausibility")
     on FAIL: emit per-cell rates, intervals, alpha, and the full disagreement
              matrix, then HALT.  No item is replaced.  No threshold is renegotiated.
```

---

## 20. End-state status

```
ADVERSARIAL_INTEGRITY_PROTOCOL_V1_1
V1_1_RECOMMENDED=YES
SOURCE_DATASET=FEVER_ONLY
SOURCE_REVISION=UNRESOLVED_PENDING_PIN
SOURCE_SPLIT=UNRESOLVED_PENDING_U5 (candidate: train, conditional on main-FEVER split confirmation)
POOL_SIZE=1000
CONTROL_DESIGN=PAIRED-REALIGN-ADJ-v1_1
HUMAN_ADJUDICATION_REQUIRED=YES
HUMAN_ADJUDICATION_SCOPE=1000 treatment items (census) + 200 hash-selected control items + 100 SSH type-plausibility riders; 2 primary annotators per item, adjudicator on disagreement
LLM_USED_IN_DATASET_CONSTRUCTION=NO
DETERMINISTIC_CANDIDATE_CONSTRUCTION=YES
PARTITION_BOUND_IN_RELEASE_ROOT=YES
MAIN_FEVER_PARENT_OVERLAP_ALLOWED=NO
UNRESOLVED_BLOCKING_GAPS=9
```

**Recommendation: YES**, with the sequencing that U5 closes before freeze, U1
before construction, and the §3 adjudication gates before any number from this
cell is reported. `V1_1_RECOMMENDED=YES` is a recommendation to proceed to freeze
and construction — it is **not** a claim that the targets are correct. Whether
they are correct is what the adjudication is for, and the protocol is designed to
fail rather than to be rescued.

---

## 21. Change log, V1 → V1.1

| # | change | driver |
|---|---|---|
| 1 | `source_world_label` → `source_reference_label`; `external_world_truth_status = UNVERIFIED` added | C1 |
| 2 | `source_truth_changed` deleted with no replacement | C1 |
| 3 | `claim_text_changed`, `evidence_text_changed` added as mechanical facts | C1 |
| 4 | **V1's `REFUTES`-side control argument retracted as unsound** | C2 |
| 5 | 667/333 control split withdrawn; paired parent controls adopted | C2 |
| 6 | `REALIGN` mode introduced for `REFUTES`-side claim attacks | C2 |
| 7 | statistical unit changed to the parent/attack pair; McNemar paired analysis added | C2 |
| 8 | full blinded adjudication protocol frozen: population, blinding, wording, α floor, cell thresholds, fail-closed | C3 |
| 9 | citation-swap C3 demoted from "guarantee" to heuristic; guarantee table added | C4 |
| 10 | `expected_failed_channel` removed from the schema | C5 |
| 11 | U3 frozen as a thresholded diagnostic; U4 declared descriptive-only with modification prohibited | C6 |
| 12 | `partition_manifest_root` and `dataset_release_root` added | C7 |
| 13 | partition feasibility made fail-closed with ESCALATE; repartitioning conditions stated | C8 |
| 14 | manuscript text rewritten; the "world-true" sentence withdrawn as false | C9 |
| 15 | `SOURCE_SPLIT` made conditional; parent-disjointness no longer claimed | C10 |
| 16 | constructor refuses to start on a placeholder source revision | C11 |
| 17 | all V1 deterministic machinery in §12 preserved verbatim | C12 |
| 18 | schema regrouped into constructed / adjudicated / external-unknown | C13 |
| 19 | metrics rebuilt on the paired design; `EVIDENCE_VALUE_OVERLAP_BASELINE` added | C14 |
| 20 | residual gaps classified by what they block | C15 |
