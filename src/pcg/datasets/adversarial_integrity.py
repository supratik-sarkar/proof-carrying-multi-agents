"""PCG-MAS Adversarial Integrity Scientific Dataset Constructor and Adapter.

Normative implementation of the frozen scientific protocol:
- ADVERSARIAL_INTEGRITY_PROTOCOL_V1
- ADVERSARIAL_INTEGRITY_PROTOCOL_V1_1
- ADVERSARIAL_INTEGRITY_PROTOCOL_V1_1A
- ADVERSARIAL_INTEGRITY_PROTOCOL_V1_1B
Consolidated in ADVERSARIAL_INTEGRITY_EFFECTIVE_PROTOCOL_V1_1B.md.
"""
from __future__ import annotations

import collections
import copy
from decimal import Decimal, ROUND_HALF_UP
import glob
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple
import unicodedata

from pcg.datasets.base import EvidenceItem, QAExample

# ---------------------------------------------------------------------------
# Frozen Constants
# ---------------------------------------------------------------------------

PROTOCOL_VERSION_V1 = "ADVERSARIAL_INTEGRITY_PROTOCOL_V1"
PROTOCOL_VERSION_V1_1 = "ADVERSARIAL_INTEGRITY_PROTOCOL_V1_1"
PROTOCOL_VERSION_V1_1A = "ADVERSARIAL_INTEGRITY_PROTOCOL_V1_1A"
PROTOCOL_VERSION_V1_1B = "ADVERSARIAL_INTEGRITY_PROTOCOL_V1_1B"

CANONICAL_DATASET_ID = "adversarial_integrity"
SOURCE_DATASET = "FEVER_ONLY"
SOURCE_SPLIT = "train"
SOURCE_REVISION = "747ae2198e94ad14a88fd7d262c55e01b07fede48a6ffd3a1182b59d3dff1f89"

PARENT_ATTACK_PAIRS = 1000
MODEL_VISIBLE_ITEMS = 2000
DONOR_REUSE_CAP = 3

PARTITION_RULE_ID = "ADVINT-PART-BUCKET1000"
PARTITION_RULE_VERSION = "v1_1"

QUOTAS: List[Tuple[str, str, int]] = [
    ("NUMBER_FLIP", "SUPPORTS", 166),
    ("NUMBER_FLIP", "REFUTES", 167),
    ("SEMANTIC_SLOT_HIJACK", "SUPPORTS", 167),
    ("SEMANTIC_SLOT_HIJACK", "REFUTES", 166),
    ("CITATION_SWAP", "SUPPORTS", 167),
    ("CITATION_SWAP", "REFUTES", 167),
]

# Lexicons (V1 §11.4 verbatim)
STOPWORDS: Set[str] = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "had", "has", "have", "he", "her", "his", "in", "is", "it", "its", "of",
    "on", "or", "she", "that", "the", "their", "there", "they", "this", "to",
    "was", "were", "which", "who", "will", "with", "would", "you", "your"
}

MONTHS: Set[str] = {
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec"
}

UNIT_LEXICON: Set[str] = {
    "km", "kilometre", "kilometres", "kilometer", "kilometers", "m", "metre",
    "metres", "meter", "meters", "cm", "mi", "mile", "miles", "ft", "feet",
    "in", "inch", "inches", "kg", "kilogram", "kilograms", "g", "gram", "grams",
    "lb", "lbs", "pound", "pounds", "tonne", "tonnes", "ton", "tons", "l",
    "litre", "litres", "liter", "liters", "ml", "mph", "kmh", "km/h", "°c",
    "°f", "celsius", "fahrenheit", "hectares", "acres"
}

STATE_ABBREV: Set[str] = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID",
    "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS",
    "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK",
    "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV",
    "WI", "WY"
}

CURRENCY_SYMBOLS: Set[str] = {"$", "£", "€", "¥", "₹", "₽", "₩"}

# Cryptographic Domain Separation Tags
DOM_RANK = "ADVINT-RANK-v1"
DOM_DONOR = "ADVINT-DONOR-v1"
DOM_NUMSIGN = "ADVINT-NUMSIGN-v1"
DOM_REC = "ADVINT-REC-v1_1"
DOM_POOL = "ADVINT-POOL-v1_1"
DOM_PART = "ADVINT-PART-v1_1"
DOM_PARTMAN = "ADVINT-PARTMAN-v1_1"
DOM_CANDREL = "ADVINT-CANDREL-v1_1A"
DOM_MVP = "ADVINT-MVP-v1_1A"
DOM_INSTR = "ADVINT-INSTR-v1_1B"
DOM_EVALORDER = "ADVINT-EVALORDER-v1_1"

# ---------------------------------------------------------------------------
# Cryptographic and Serialization Primitives
# ---------------------------------------------------------------------------

def LP(x: str | bytes) -> bytes:
    """Length-prefix encoding: len(utf8(x)).to_bytes(8, 'big') || utf8(x)."""
    if isinstance(x, str):
        b = x.encode("utf-8")
    elif isinstance(x, bytes):
        b = x
    else:
        b = str(x).encode("utf-8")
    return len(b).to_bytes(8, "big") + b


def sha256_domain(domain_tag: str, payload: bytes) -> str:
    """Computes SHA256(LP(domain_tag) || LP(payload))."""
    return hashlib.sha256(LP(domain_tag) + LP(payload)).hexdigest()


def compute_pair_id(parent_id: str, attack_family: str) -> str:
    """Deterministic pair_id convention: advint-v1_1:<parent_id>:<attack_family>."""
    return f"advint-v1_1:{parent_id}:{attack_family.upper()}"


def compute_item_id(pair_id: str, role: str) -> str:
    """Deterministic item_id convention: <pair_id>#<role>."""
    return f"{pair_id}#{role.lower()}"


def canonical_json(obj: Any) -> bytes:
    """Frozen canonical JSON serializer (V1 §11.2)."""
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False
    ).encode("utf-8")


def norm(s: str) -> str:
    """Text normalization applied once at load (V1 §11.1)."""
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFC", s)
    s = (s.replace("-LRB-", "(")
          .replace("-RRB-", ")")
          .replace("-LSB-", "[")
          .replace("-RSB-", "]")
          .replace("-COLON-", ":"))
    s = s.replace(" ", " ").replace(" ", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def nf(s: str) -> str:
    """Normal form used for duplicate policy (V1 §12.1)."""
    s = norm(s).lower()
    for ch in ".,;:!?\"'()[]-":
        s = s.replace(ch, "")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


# ---------------------------------------------------------------------------
# Numeric Parsing & Number Flip Engine (V1 §6, V1.1 §12.1)
# ---------------------------------------------------------------------------

NUM_PATTERN = re.compile(
    r"(?<![0-9A-Za-z.])(?P<sign>[-−])?(?P<int>\d{1,3}(,\d{3})+|\d+)(?P<frac>\.\d+)?(?![0-9A-Za-z])"
)

ORDINAL_SUFFIXES = {"st", "nd", "rd", "th"}


class NumericSpan:
    def __init__(self, match: re.Match, claim_text: str):
        self.start = match.start()
        self.end = match.end()
        self.surface = claim_text[self.start:self.end]
        raw_sign = match.group("sign") or ""
        self.sign_mult = -1 if (raw_sign == "-" or raw_sign == "−") else 1
        int_part = match.group("int").replace(",", "")
        frac_part = match.group("frac") or ""
        val_str = ("-" if self.sign_mult < 0 else "") + int_part + frac_part
        self.val = Decimal(val_str)
        self.precision = len(frac_part) - 1 if frac_part else 0
        self.grouped = "," in match.group("int")
        self.cls, self.unit = self._classify(claim_text)

    def _classify(self, claim_text: str) -> Tuple[str, str]:
        # Excluded classes
        post = claim_text[self.end:].lstrip()
        first_word = post.split()[0].lower() if post.split() else ""
        first_word_clean = first_word.rstrip(".,;:!?")
        
        # ORDINAL
        if any(post.lower().startswith(suf) for suf in ORDINAL_SUFFIXES):
            return "EXCLUDED_ORDINAL", ""
            
        # CITATION
        if self.start > 0 and self.end < len(claim_text):
            if claim_text[self.start - 1] == "[" and claim_text[self.end] == "]":
                return "EXCLUDED_CITATION", ""
                
        # VERSION / ID
        pre = claim_text[:self.start].rstrip()
        if any(pre.endswith(p) for p in ["#", "No.", "no.", "v", "V", "version"]):
            return "EXCLUDED_VERSION_ID", ""
        if self.surface.count(".") >= 2:
            return "EXCLUDED_VERSION_ID", ""
            
        # DATE / YEAR
        is_year = (self.precision == 0 and not self.grouped and 1000 <= self.val <= 2999)
        has_unit_or_pct = (first_word_clean in UNIT_LEXICON or post.startswith("%"))
        if is_year and not has_unit_or_pct:
            return "EXCLUDED_DATE_YEAR", ""
        # Check nearby month
        words_before = claim_text[:self.start].split()[-3:]
        words_after = claim_text[self.end:].split()[:3]
        nearby_words = [w.lower().rstrip(".,;:!?") for w in words_before + words_after]
        if any(w in MONTHS for w in nearby_words):
            return "EXCLUDED_DATE_YEAR", ""
        # Date regexes
        surrounding = claim_text[max(0, self.start - 5):min(len(claim_text), self.end + 5)]
        if re.search(r"\d{1,2}/\d{1,2}/\d{2,4}", surrounding) or re.search(r"\d{4}-\d{2}-\d{2}", surrounding):
            return "EXCLUDED_DATE_YEAR", ""
            
        # Eligible classes: PERCENT, MONEY, MEASUREMENT, CARDINAL
        if post.startswith("%") or first_word_clean in {"percent", "percentage"}:
            return "PERCENT", "%"
        if any(pre.endswith(sym) for sym in CURRENCY_SYMBOLS) or first_word_clean in {"dollars", "euros", "pounds", "cents", "yen", "rupees"}:
            curr = [sym for sym in CURRENCY_SYMBOLS if pre.endswith(sym)]
            return "MONEY", curr[0] if curr else first_word_clean
        if first_word_clean in UNIT_LEXICON:
            return "MEASUREMENT", first_word_clean
        return "CARDINAL", ""


def parse_numeric_spans(text: str) -> List[NumericSpan]:
    spans: List[NumericSpan] = []
    for m in NUM_PATTERN.finditer(text):
        span = NumericSpan(m, text)
        if not span.cls.startswith("EXCLUDED"):
            spans.append(span)
    return spans


def format_decimal(d: Decimal, precision: int, grouped: bool) -> str:
    sign_str = "-" if d < 0 else ""
    abs_d = abs(d)
    if precision > 0:
        fmt = f"{{:.{precision}f}}"
        num_str = fmt.format(abs_d)
        int_part, frac_part = num_str.split(".", 1)
        if grouped:
            int_part = f"{int(int_part):,}"
        return f"{sign_str}{int_part}.{frac_part}"
    else:
        int_val = int(abs_d)
        int_part = f"{int_val:,}" if grouped else str(int_val)
        return f"{sign_str}{int_part}"


# ---------------------------------------------------------------------------
# Content-Token Jaccard Engine for Citation Swap (V1 §5.5)
# ---------------------------------------------------------------------------

def extract_content_tokens(text: str, prim_surface: str) -> Set[str]:
    toks = text.split()
    prim_tokens = {p.lower() for p in prim_surface.split()}
    res = set()
    for t in toks:
        clean = t.lower().strip(".,;:!?\"'()[]-")
        if not clean:
            continue
        if clean in STOPWORDS:
            continue
        if clean in prim_tokens:
            continue
        res.add(clean)
    return res


def jaccard_cross_mul_admissible(tokens_a: Set[str], tokens_b: Set[str]) -> Tuple[bool, int, int]:
    """Evaluates |A ∩ B| / |A ∪ B| >= 1/10 via exact integer cross-multiplication.
    
    Returns (admissible, intersection_size, union_size).
    """
    inter = len(tokens_a.intersection(tokens_b))
    union = len(tokens_a.union(tokens_b))
    if union == 0:
        return False, 0, 0
    # 10 * inter >= union
    return (10 * inter >= union), inter, union


# ---------------------------------------------------------------------------
# Corpus & Gazetteer Engine (V1 §4.2, §11.1)
# ---------------------------------------------------------------------------

COPULA_RE = re.compile(
    r"^\s*(?:is|was|are|were)\s+(?:a|an|the)\s+([^,.;]{1,80})",
    re.IGNORECASE
)
DISAMBIG_RE = re.compile(r"^\s*(\([^()]*\)\s*)?")
REL_CLAUSE_RE = re.compile(
    r"\s+(?:who|which|that|located|written|directed|based|created|composed|founded|born|starring|set|serving|featuring)\b",
    re.IGNORECASE
)


class WikiCorpusIndex:
    """Indexes evidence pages and gazetteer entries across the 109 wiki shards."""
    
    def __init__(self, shards_dir: str):
        self.shards_dir = shards_dir
        self.pages: Dict[str, List[str]] = {}  # page_id -> [sentence_0, sentence_1, ...]
        self.gazetteer: Dict[str, str] = {}    # surf -> page_id
        self.gazetteer_lower: Dict[str, str] = {} # surf.lower() -> page_id
        self.tau: Dict[str, str] = {}          # page_id -> tau
        self.tau_head: Dict[str, str] = {}     # page_id -> tau_head
        self.tau_to_pages: Dict[str, List[str]] = collections.defaultdict(list)
        self.tau_head_to_pages: Dict[str, List[str]] = collections.defaultdict(list)

    def load(self, needed_evidence_pages: Set[str]):
        """Load needed evidence pages and build complete gazetteer."""
        shard_paths = sorted(glob.glob(os.path.join(self.shards_dir, "wiki-*.jsonl")))
        exact_matches: Dict[str, str] = {}
        disambig_matches: Dict[str, List[str]] = collections.defaultdict(list)
        ambiguous: Set[str] = set()

        for spath in shard_paths:
            with open(spath, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    pid = data.get("id")
                    if not pid:
                        continue
                    
                    lines_str = data.get("lines", "")
                    
                    # If this is a needed evidence page, store all sentences
                    if pid in needed_evidence_pages:
                        sents = []
                        for pline in lines_str.split("\n"):
                            parts = pline.split("\t")
                            if len(parts) >= 2:
                                sents.append(norm(parts[1]))
                        self.pages[pid] = sents

                    # Gazetteer surface form
                    title = norm(pid.replace("_", " "))
                    surf = title
                    idx_p = surf.rfind("(")
                    if idx_p != -1 and surf.endswith(")"):
                        surf = surf[:idx_p].strip()
                    
                    if not (3 <= len(surf) <= 60):
                        continue
                    if surf.islower():
                        continue
                    
                    if title == surf:
                        if surf in exact_matches and exact_matches[surf] != pid:
                            ambiguous.add(surf)
                        else:
                            exact_matches[surf] = pid
                    else:
                        disambig_matches[surf].append(pid)

                    # Copula type signature tau
                    if lines_str:
                        first_line = lines_str.split("\n", 1)[0]
                        if surf.lower() in first_line.lower():
                            parts = first_line.split("\t")
                            if len(parts) >= 2:
                                sent0 = norm(parts[1])
                                sent0_check = sent0[4:] if sent0.lower().startswith("the ") else sent0
                                if sent0_check.lower().startswith(surf.lower()):
                                    rem = sent0_check[len(surf):]
                                    while True:
                                        m_dis = re.match(r"^\s*,?\s*(\([^()]*\)|\[[^\[\]]*\])\s*", rem)
                                        if not m_dis:
                                            break
                                        rem = rem[m_dis.end():]
                                    m = COPULA_RE.match(rem.strip())
                                    if m:
                                        raw_t = norm(m.group(1)).lower().rstrip(".")
                                        cleaned_t = REL_CLAUSE_RE.split(raw_t)[0].strip()
                                        if cleaned_t:
                                            self.tau[pid] = cleaned_t
                                            toks = cleaned_t.split()
                                            last_tok = toks[-1] if toks else ""
                                            thead = last_tok[:-1] if (len(last_tok) >= 4 and last_tok.endswith("s")) else last_tok
                                            self.tau_head[pid] = thead
                                            self.tau_to_pages[cleaned_t].append(pid)
                                            self.tau_head_to_pages[thead].append(pid)

        # Assemble gazetteer
        for surf, pid in exact_matches.items():
            if surf not in ambiguous:
                self.gazetteer[surf] = pid
                self.gazetteer_lower[surf.lower()] = pid
        for surf, pids in disambig_matches.items():
            if surf not in self.gazetteer and surf not in ambiguous:
                if len(pids) == 1:
                    self.gazetteer[surf] = pids[0]
                    self.gazetteer_lower[surf.lower()] = pids[0]
                else:
                    ambiguous.add(surf)


def find_word_bounded_slots(text: str, corpus: WikiCorpusIndex) -> List[Tuple[int, int, str, str]]:
    """Finds maximal-length matches of gazetteer surface forms in text at word boundaries."""
    slots = []
    i = 0
    text_len = len(text)
    while i < text_len:
        if i > 0 and text[i - 1].isalnum():
            i += 1
            continue
        bm, blen = None, 0
        for l in range(min(60, text_len - i), 2, -1):
            end = i + l
            if end < text_len and text[end].isalnum():
                continue
            sub = text[i:end]
            sub_low = sub.lower()
            if sub_low in corpus.gazetteer_lower:
                bm = (i, end, sub, corpus.gazetteer_lower[sub_low])
                blen = l
                break
        if bm:
            slots.append(bm)
            i += blen
        else:
            i += 1
    return slots


# ---------------------------------------------------------------------------
# Admissible Parent Structure & Loader
# ---------------------------------------------------------------------------

class AdmissibleParent:
    def __init__(
        self,
        parent_id: str,
        source_label: str,
        claim: str,
        ev_page: str,
        ev_sid: int,
        ev_text: str,
        prim: str,
    ):
        self.id = parent_id
        self.label = source_label  # "SUPPORTS" | "REFUTES"
        self.claim = claim
        self.ev_page = ev_page
        self.ev_sid = ev_sid
        self.ev_text = ev_text
        self.prim = prim
        self.content_tokens: Set[str] = set()


def load_admissible_parents(
    train_path: str,
    corpus: WikiCorpusIndex,
    dev_parent_ids: Set[str],
) -> List[AdmissibleParent]:
    """Extracts admissible single-hop parents adhering strictly to S1–S6 (§1.5)."""
    parents: List[AdmissibleParent] = []
    seen_parent_ids: Set[str] = set()

    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            lab = r.get("label")
            if lab not in {"SUPPORTS", "REFUTES"}:
                continue
            pid = str(r["id"])
            if pid in dev_parent_ids:
                continue
            if pid in seen_parent_ids:
                continue

            # S1: At least one annotation set in evidence contains exactly one sentence pointer
            single_sets = []
            for s in r.get("evidence", []):
                if len(s) == 1:
                    page, sid = s[0][2], s[0][3]
                    if page and sid is not None:
                        single_sets.append((page, sid))
            if not single_sets:
                continue

            # Lexicographically first by (page UTF-8 bytes, sentence_id integer)
            single_sets.sort(key=lambda item: (item[0].encode("utf-8"), item[1]))
            canon_page, canon_sid = single_sets[0]

            if canon_page not in corpus.pages:
                continue
            sents = corpus.pages[canon_page]
            if canon_sid < 0 or canon_sid >= len(sents):
                continue
            etext = sents[canon_sid]

            # S4: claim between 5 and 60 tokens inclusive
            claim = norm(r.get("claim", ""))
            ctoks = claim.split()
            if not (5 <= len(ctoks) <= 60):
                continue

            # S5: evidence tokens >= 5
            if len(etext.split()) < 5:
                continue

            prim = norm(canon_page.replace("_", " "))
            idx_p = prim.rfind("(")
            if idx_p != -1 and prim.endswith(")"):
                prim = prim[:idx_p].strip()

            seen_parent_ids.add(pid)
            p_obj = AdmissibleParent(
                parent_id=pid,
                source_label=lab,
                claim=claim,
                ev_page=canon_page,
                ev_sid=canon_sid,
                ev_text=etext,
                prim=prim,
            )
            p_obj.content_tokens = extract_content_tokens(etext, prim)
            parents.append(p_obj)

    return parents


# ---------------------------------------------------------------------------
# Attack Realization Engines
# ---------------------------------------------------------------------------

def apply_number_flip(
    p: AdmissibleParent,
    pair_id: str,
) -> Optional[Tuple[str, str, Dict[str, Any]]]:
    """Executes NUMBER_FLIP: DESUPPORT for SUPPORTS, REALIGN for REFUTES (V1 §6, V1.1 §12.1)."""
    spans = parse_numeric_spans(p.claim)
    if not spans:
        return None

    # Priority order (§6.3): class priority, start ASC, len DESC
    def span_priority(s: NumericSpan):
        cls_order = {"MONEY": 0, "PERCENT": 1, "MEASUREMENT": 2, "CARDINAL": 3}
        return (cls_order.get(s.cls, 99), s.start, -(s.end - s.start))

    ordered_spans = sorted(spans, key=span_priority)
    ev_spans = parse_numeric_spans(p.ev_text)
    ev_nums = {s.val for s in ev_spans}

    if p.label == "SUPPORTS":
        # Mode: DESUPPORT (FLIP)
        mode = "DESUPPORT"
        for span in ordered_spans:
            if span.val not in ev_nums:
                continue
            if span.val == 0:
                continue
            if span.precision == 0 and abs(span.val) < 5:
                continue

            h_input = (LP("ADVINT-NUMSIGN-v1") + LP(PROTOCOL_VERSION_V1) +
                       LP(pair_id) + LP(str(span.start)))
            h_val = hashlib.sha256(h_input).digest()
            s = 1 if (h_val[31] & 1) == 0 else -1

            signs_to_try = [s]
            if span.cls == "PERCENT" and 0 <= span.val <= 100:
                if span.val * Decimal("1.1") > 100:
                    signs_to_try = [-1]

            chosen_n2 = None
            attempt_signs = signs_to_try + ([-signs_to_try[0]] if len(signs_to_try) == 1 else [])
            for attempt_sign in attempt_signs:
                delta = Decimal(attempt_sign) * Decimal("0.1")
                n2 = abs(span.val) * (Decimal("1") + delta)
                if span.precision == 0:
                    n2 = n2.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
                else:
                    fmt_q = Decimal("1." + "0" * span.precision)
                    n2 = n2.quantize(fmt_q, rounding=ROUND_HALF_UP)

                n2 = n2 * span.sign_mult
                if n2 == span.val:
                    continue
                if span.cls == "PERCENT" and not (0 <= n2 <= 100):
                    continue
                if n2 in ev_nums:
                    continue
                chosen_n2 = n2
                break

            if chosen_n2 is None:
                continue

            formatted_n2 = format_decimal(chosen_n2, span.precision, span.grouped)
            new_claim = p.claim[:span.start] + formatted_n2 + p.claim[span.end:]

            params = {
                "attack_mode": mode,
                "target_span_start": span.start,
                "target_span_end": span.end,
                "original_value": span.surface,
                "perturbed_value": formatted_n2,
                "numeric_class": span.cls,
            }
            return new_claim, p.ev_text, params
        return None

    else:
        # Mode: REALIGN
        mode = "REALIGN"
        for span in ordered_spans:
            if span.val in ev_nums:
                continue
            cands = [
                m for m in ev_spans
                if m.cls == span.cls and m.unit == span.unit and m.val != span.val
            ]
            if len(cands) != 1:
                continue
            cand = cands[0]
            new_claim = p.claim[:span.start] + cand.surface + p.claim[span.end:]
            params = {
                "attack_mode": mode,
                "target_span_start": span.start,
                "target_span_end": span.end,
                "original_value": span.surface,
                "perturbed_value": cand.surface,
                "numeric_class": span.cls,
            }
            return new_claim, p.ev_text, params
        return None


def apply_semantic_slot_hijack(
    p: AdmissibleParent,
    pair_id: str,
    corpus: WikiCorpusIndex,
    donor_reuse_counts: Dict[str, int],
) -> Optional[Tuple[str, str, Dict[str, Any]]]:
    """Executes SEMANTIC_SLOT_HIJACK: DESUPPORT for SUPPORTS, REALIGN for REFUTES (V1 §4, V1.1 §12.1)."""
    claim = p.claim
    prim = p.prim
    slots = find_word_bounded_slots(claim, corpus)
    num_spans = parse_numeric_spans(claim)

    eligible_slots = []
    for start, end, surf, pid in slots:
        if surf.lower() == prim.lower() or prim.lower() in surf.lower() or surf.lower() in prim.lower():
            continue
        if pid not in corpus.tau:
            continue
        if any(not (end <= ns.start or start >= ns.end) for ns in num_spans):
            continue
        left_toks = claim[:start].split()
        right_toks = claim[end:].split()
        if len(left_toks) < 3 and len(right_toks) < 3:
            continue
        if surf.lower() in prim.lower() or prim.lower() in surf.lower():
            continue
        if p.label == "SUPPORTS":
            if surf.lower() not in p.ev_text.lower():
                continue
        else:
            if surf.lower() in p.ev_text.lower():
                continue
        eligible_slots.append((start, end, surf, pid))

    if not eligible_slots:
        return None

    # Sort slots: start ASC, len DESC, surf ASC
    eligible_slots.sort(key=lambda x: (x[0], -(x[1] - x[0]), x[2]))

    for start, end, slot_surf, slot_pid in eligible_slots:
        target_tau = corpus.tau[slot_pid]
        target_head = corpus.tau_head[slot_pid]

        if p.label == "SUPPORTS":
            mode = "DESUPPORT"
            tier1 = corpus.tau_to_pages.get(target_tau, [])
            tier2 = corpus.tau_head_to_pages.get(target_head, [])

            for tier_idx, cands in [(1, tier1), (2, tier2)]:
                valid_donors = []
                for q_pid in cands:
                    q_surf = norm(q_pid.replace("_", " "))
                    idx_p = q_surf.rfind("(")
                    if idx_p != -1 and q_surf.endswith(")"):
                        q_surf = q_surf[:idx_p].strip()

                    if q_surf.lower() == prim.lower():
                        continue
                    if q_surf.lower() in p.ev_text.lower():
                        continue
                    if q_surf.lower() in claim.lower():
                        continue
                    if q_surf.lower() in slot_surf.lower() or slot_surf.lower() in q_surf.lower():
                        continue
                    if donor_reuse_counts.get(q_pid, 0) >= DONOR_REUSE_CAP:
                        continue
                    valid_donors.append((q_pid, q_surf))

                if not valid_donors:
                    continue

                def donor_rank_key(d_entry):
                    q_pid, _ = d_entry
                    h_input = (LP("ADVINT-DONOR-v1") + LP(PROTOCOL_VERSION_V1) +
                               LP(pair_id) + LP(q_pid))
                    return (hashlib.sha256(h_input).digest(), q_pid.encode("utf-8"))

                valid_donors.sort(key=donor_rank_key)

                for q_pid, q_surf in valid_donors:
                    new_claim = claim[:start] + q_surf + claim[end:]
                    if new_claim == claim:
                        continue
                    if not (5 <= len(new_claim.split()) <= 60):
                        continue
                    donor_reuse_counts[q_pid] = donor_reuse_counts.get(q_pid, 0) + 1
                    params = {
                        "attack_mode": mode,
                        "slot_span_start": start,
                        "slot_span_end": end,
                        "slot_original": slot_surf,
                        "slot_replacement": q_surf,
                        "donor_page_id": q_pid,
                        "tier": tier_idx,
                    }
                    return new_claim, p.ev_text, params

        else:
            mode = "REALIGN"
            ev_slots = find_word_bounded_slots(p.ev_text, corpus)
            tier1 = [e for e in ev_slots if corpus.tau.get(e[3]) == target_tau]
            tier2 = [e for e in ev_slots if corpus.tau_head.get(e[3]) == target_head]

            for tier_idx, cands in [(1, tier1), (2, tier2)]:
                valid_donors = []
                for q_start, q_end, q_surf, q_pid in cands:
                    if q_surf.lower() == prim.lower():
                        continue
                    if q_surf.lower() in claim.lower():
                        continue
                    if q_surf.lower() in slot_surf.lower() or slot_surf.lower() in q_surf.lower():
                        continue
                    if donor_reuse_counts.get(q_pid, 0) >= DONOR_REUSE_CAP:
                        continue
                    valid_donors.append((q_pid, q_surf))

                if not valid_donors:
                    continue

                def donor_rank_key(d_entry):
                    q_pid, _ = d_entry
                    h_input = (LP("ADVINT-DONOR-v1") + LP(PROTOCOL_VERSION_V1) +
                               LP(pair_id) + LP(q_pid))
                    return (hashlib.sha256(h_input).digest(), q_pid.encode("utf-8"))

                valid_donors.sort(key=donor_rank_key)

                for q_pid, q_surf in valid_donors:
                    new_claim = claim[:start] + q_surf + claim[end:]
                    if new_claim == claim:
                        continue
                    if not (5 <= len(new_claim.split()) <= 60):
                        continue
                    donor_reuse_counts[q_pid] = donor_reuse_counts.get(q_pid, 0) + 1
                    params = {
                        "attack_mode": mode,
                        "slot_span_start": start,
                        "slot_span_end": end,
                        "slot_original": slot_surf,
                        "slot_replacement": q_surf,
                        "donor_page_id": q_pid,
                        "tier": tier_idx,
                    }
                    return new_claim, p.ev_text, params

    return None


def apply_citation_swap(
    p: AdmissibleParent,
    pair_id: str,
    donor_candidates: List[AdmissibleParent],
    donor_reuse_counts: Dict[str, int],
    corpus: WikiCorpusIndex,
    donor_index: Optional[Dict[str, List[AdmissibleParent]]] = None,
) -> Optional[Tuple[str, str, Dict[str, Any]]]:
    """Executes CITATION_SWAP verbatim per V1 §5 (C1–C6, Tiers 1/2, J >= 1/10)."""
    claim = p.claim
    prim = p.prim
    claim_tokens = extract_content_tokens(claim, prim)

    # Non-primary entities in claim for Tier 1
    claim_entities = set()
    claim_slots = find_word_bounded_slots(claim, corpus)
    for _, _, surf, _ in claim_slots:
        if surf.lower() != prim.lower():
            claim_entities.add(surf.lower())

    if donor_index is not None:
        candidate_set = set()
        for tok in claim_tokens:
            candidate_set.update(donor_index.get(tok, []))
        candidate_pool = sorted(candidate_set, key=lambda d: (d.ev_page.encode("utf-8"), d.ev_sid, d.id.encode("utf-8")))
    else:
        candidate_pool = sorted(set(donor_candidates), key=lambda d: (d.ev_page.encode("utf-8"), d.ev_sid, d.id.encode("utf-8")))

    admissible_donors = []
    for d in candidate_pool:
        # C1: par(d) != P
        if d.id == p.id:
            continue
        # C2: page(d) not in pages(E)
        if d.ev_page == p.ev_page:
            continue
        # C3: prim absent from txt(d)
        if prim.lower() in d.ev_text.lower():
            continue
        # C4: donor reuse cap
        if donor_reuse_counts.get(d.ev_text, 0) >= DONOR_REUSE_CAP:
            continue
        # C5: txt(d) != E byte-wise
        if d.ev_text == p.ev_text:
            continue
        # C6: 5 <= tokens <= 80
        tok_len = len(d.ev_text.split())
        if not (5 <= tok_len <= 80):
            continue

        # Lexical Jaccard >= 1/10
        admit, inter, union = jaccard_cross_mul_admissible(claim_tokens, d.content_tokens)
        if not admit:
            continue

        # Tier check
        is_tier1 = any(e in d.ev_text.lower() for e in claim_entities)
        admissible_donors.append((d, is_tier1, inter, union))

    if not admissible_donors:
        return None

    for tier_target in [True, False]:
        tier_pool = [x for x in admissible_donors if x[1] == tier_target]
        if not tier_pool:
            continue

        # Ranking: (1) J descending via cross-multiplication,
        #          (2) ADVINT-DONOR-v1 SHA256 byte-ascending,
        #          (3) (page_id, sentence_id) ascending
        #          (4) d.id ascending (canonical tie-breaker for identical evidence pointers)
        class JaccardSortKey:
            def __init__(self, entry):
                d, _, inter, union = entry
                self.d = d
                self.inter = inter
                self.union = union
                h_input = (LP("ADVINT-DONOR-v1") + LP(PROTOCOL_VERSION_V1) +
                           LP(pair_id) + LP(d.ev_page) + LP(str(d.ev_sid)))
                self.hash_bytes = hashlib.sha256(h_input).digest()
                self.page_bytes = d.ev_page.encode("utf-8")
                self.sid = d.ev_sid
                self.did_bytes = d.id.encode("utf-8")

            def __lt__(self, other: "JaccardSortKey") -> bool:
                v1 = self.inter * other.union
                v2 = other.inter * self.union
                if v1 != v2:
                    return v1 > v2
                if self.hash_bytes != other.hash_bytes:
                    return self.hash_bytes < other.hash_bytes
                if self.page_bytes != other.page_bytes:
                    return self.page_bytes < other.page_bytes
                if self.sid != other.sid:
                    return self.sid < other.sid
                return self.did_bytes < other.did_bytes

        ranked = sorted(tier_pool, key=JaccardSortKey)
        chosen_d, is_t1, chosen_inter, chosen_union = ranked[0]

        donor_reuse_counts[chosen_d.ev_text] = donor_reuse_counts.get(chosen_d.ev_text, 0) + 1
        params = {
            "attack_mode": "SWAP",
            "donor_parent_id": chosen_d.id,
            "donor_page": chosen_d.ev_page,
            "donor_sentence_id": chosen_d.ev_sid,
            "tier": 1 if is_t1 else 2,
            "jaccard_numerator": chosen_inter,
            "jaccard_denominator": chosen_union,
        }
        return p.claim, chosen_d.ev_text, params

    return None


# ---------------------------------------------------------------------------
# Record Construction & Cryptographic Roots (V1.1 §13, V1.1A §A)
# ---------------------------------------------------------------------------

def compute_record_sha256(record_dict: Dict[str, Any]) -> str:
    """Computes record_sha256 restricted to GROUP A ∪ GROUP C (V1.1 §13.1)."""
    rec_copy = dict(record_dict)
    # Exclude Group B
    group_b_fields = [
        "adjudicated_support_relation", "adjudication_status",
        "adjudication_primary_responses", "adjudication_adjudicator_response",
        "adjudication_protocol_sha256", "hypothesis_confirmed",
        "certificate_validity_target"
    ]
    for f in group_b_fields:
        rec_copy.pop(f, None)
    # Exclude release metadata
    for f in ["partition", "partition_rule_id", "partition_rule_version", "record_sha256"]:
        rec_copy.pop(f, None)

    c_bytes = canonical_json(rec_copy)
    h_input = LP("ADVINT-REC-v1_1") + LP(c_bytes)
    return hashlib.sha256(h_input).hexdigest()


def compute_pool_content_root(records: List[Dict[str, Any]]) -> str:
    """Computes pool_content_root over sorted records (V1.1 §7.1)."""
    sorted_recs = sorted(records, key=lambda r: r["item_id"].encode("utf-8"))
    raw_digests = b"".join(bytes.fromhex(r["record_sha256"]) for r in sorted_recs)
    n = len(sorted_recs)
    h_input = (LP("ADVINT-POOL-v1_1") + LP(PROTOCOL_VERSION_V1_1) +
               LP(SOURCE_REVISION) + n.to_bytes(8, "big") + raw_digests)
    return hashlib.sha256(h_input).hexdigest()


def compute_partition_bucket(pair_id: str) -> int:
    """Assigns partition bucket b(pair_id) mod 1000 (V1.1 §8.1)."""
    h_input = (LP("ADVINT-PART-v1_1") + LP(PROTOCOL_VERSION_V1_1) +
               LP(pair_id))
    digest = hashlib.sha256(h_input).digest()
    return int.from_bytes(digest[:8], "big") % 1000


def bucket_to_partition(b: int) -> str:
    if 0 <= b <= 19:
        return "SMOKE"
    elif 20 <= b <= 119:
        return "CHECKER_CALIBRATION"
    elif 120 <= b <= 269:
        return "PILOT"
    else:
        return "FINAL"


def compute_partition_manifest_root(records: List[Dict[str, Any]]) -> str:
    """Computes partition_manifest_root over item_id in canonical order (V1.1 §7.2)."""
    sorted_recs = sorted(records, key=lambda r: r["item_id"].encode("utf-8"))
    n = len(sorted_recs)
    parts_bytes = b"".join(LP(r["item_id"]) + LP(r["partition"]) for r in sorted_recs)
    h_input = (LP("ADVINT-PARTMAN-v1_1") + LP(PARTITION_RULE_ID) +
               LP(PARTITION_RULE_VERSION) + n.to_bytes(8, "big") + parts_bytes)
    return hashlib.sha256(h_input).hexdigest()


def compute_candidate_release_root(
    pool_content_root: str,
    partition_manifest_root: str,
    adjudication_protocol_sha256: str,
) -> str:
    """Computes candidate_release_root (V1.1A §A.1)."""
    h_input = (LP("ADVINT-CANDREL-v1_1A") + LP(PROTOCOL_VERSION_V1_1A) +
               LP(SOURCE_REVISION) + LP(pool_content_root) +
               LP(PARTITION_RULE_ID) + LP(PARTITION_RULE_VERSION) +
               LP(partition_manifest_root) + LP(adjudication_protocol_sha256))
    return hashlib.sha256(h_input).hexdigest()


def compute_eval_order_key(candidate_release_root: str, item_id: str) -> bytes:
    """Computes F19 presentation order key (V1.1 §2.5, V1.1A)."""
    h_input = (LP("ADVINT-EVALORDER-v1_1") + LP(candidate_release_root) +
               LP(item_id))
    return hashlib.sha256(h_input).digest()


# ---------------------------------------------------------------------------
# Master Construction Pipeline
# ---------------------------------------------------------------------------

def construct_candidate_pool(
    raw_dir: str,
    extracted_wiki_dir: str,
    adjudication_protocol_sha256: str,
) -> Dict[str, Any]:
    """Deterministically constructs the 1,000-pair adversarial integrity candidate pool."""
    train_file = os.path.join(raw_dir, "train.jsonl")
    dev_file = os.path.join(raw_dir, "shared_task_dev.jsonl")

    # 1. Parse dev parent IDs for strict disjointness check
    dev_ids = set()
    if os.path.exists(dev_file):
        with open(dev_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    dev_ids.add(str(json.loads(line)["id"]))

    # 2. Collect evidence pages needed from train
    needed_pages = set()
    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("label") not in {"SUPPORTS", "REFUTES"}:
                continue
            for s in r.get("evidence", []):
                if len(s) == 1:
                    p = s[0][2]
                    if p:
                        needed_pages.add(p)

    # 3. Load corpus index
    corpus = WikiCorpusIndex(extracted_wiki_dir)
    corpus.load(needed_pages)

    # 4. Filter admissible parents
    parents = load_admissible_parents(train_file, corpus, dev_ids)

    # Build inverted index for Citation Swap
    donor_index: Dict[str, List[AdmissibleParent]] = collections.defaultdict(list)
    for p in parents:
        tok_len = len(p.ev_text.split())
        if 5 <= tok_len <= 80:
            for t in p.content_tokens:
                donor_index[t].append(p)

    # 5. Build per-cell candidate pools
    def parent_rank_key(source_id: str, fam: str) -> bytes:
        h_input = (LP("ADVINT-RANK-v1") + LP(PROTOCOL_VERSION_V1) +
                   LP(SOURCE_REVISION) + LP(SOURCE_SPLIT) +
                   LP(source_id) + LP(fam))
        return hashlib.sha256(h_input).digest()

    pools: Dict[Tuple[str, str], List[AdmissibleParent]] = collections.defaultdict(list)
    for p in parents:
        pools[( "NUMBER_FLIP", p.label )].append(p)
        pools[( "SEMANTIC_SLOT_HIJACK", p.label )].append(p)
        pools[( "CITATION_SWAP", p.label )].append(p)

    for cell_key in pools:
        fam, _ = cell_key
        pools[cell_key].sort(key=lambda p: (parent_rank_key(p.id, fam), p.id.encode("utf-8")))

    # 6. Allocate quotas
    claimed_parents: Set[str] = set()
    seen_nf_claims: Set[str] = set()
    donor_use_ssh: Dict[str, int] = collections.defaultdict(int)
    donor_use_cs: Dict[str, int] = collections.defaultdict(int)
    records: List[Dict[str, Any]] = []

    for fam, lab, quota in QUOTAS:
        filled = 0
        cand_list = pools[(fam, lab)]

        for p in cand_list:
            if filled == quota:
                break
            if p.id in claimed_parents:
                continue
            if nf(p.claim) in seen_nf_claims:
                continue

            pair_id = compute_pair_id(p.id, fam)

            if fam == "NUMBER_FLIP":
                res = apply_number_flip(p, pair_id)
            elif fam == "SEMANTIC_SLOT_HIJACK":
                res = apply_semantic_slot_hijack(p, pair_id, corpus, donor_use_ssh)
            elif fam == "CITATION_SWAP":
                res = apply_citation_swap(p, pair_id, parents, donor_use_cs, corpus, donor_index=donor_index)
            else:
                res = None

            if res is None:
                continue

            pert_claim, pert_evidence_text, params = res
            if nf(pert_claim) in seen_nf_claims:
                continue

            # Assign partition
            b = compute_partition_bucket(pair_id)
            part = bucket_to_partition(b)

            # Build Control Record
            ctrl_item_id = compute_item_id(pair_id, "control")
            ctrl_rec: Dict[str, Any] = {
                "dataset_id": CANONICAL_DATASET_ID,
                "protocol_version": PROTOCOL_VERSION_V1_1,
                "pair_id": pair_id,
                "item_id": ctrl_item_id,
                "item_role": "CONTROL",
                "source_dataset": SOURCE_DATASET,
                "source_revision": SOURCE_REVISION,
                "source_split": SOURCE_SPLIT,
                "parent_id": p.id,
                "source_reference_label": p.label,
                "asserted_verdict": "SUPPORTED" if p.label == "SUPPORTS" else "REFUTED",
                "original_claim": p.claim,
                "original_evidence": {
                    "page": p.ev_page,
                    "sentence_id": p.ev_sid,
                    "text": p.ev_text,
                },
                "presented_claim": p.claim,
                "presented_evidence": {
                    "page": p.ev_page,
                    "sentence_id": p.ev_sid,
                    "text": p.ev_text,
                },
                "attack_family": fam,
                "attack_mode": params.get("attack_mode", "NONE"),
                "attack_parameters": params,
                "claim_text_changed": False,
                "evidence_text_changed": False,
                "support_relation_hypothesis": "VALID_SUPPORT",
                "adjudicated_support_relation": None,
                "adjudication_status": "NOT_IN_ADJUDICATION_POPULATION",
                "adjudication_primary_responses": None,
                "adjudication_adjudicator_response": None,
                "adjudication_protocol_sha256": adjudication_protocol_sha256,
                "hypothesis_confirmed": None,
                "certificate_validity_target": None,
                "external_world_truth_status": "UNVERIFIED",
                "partition": part,
                "partition_rule_id": PARTITION_RULE_ID,
                "partition_rule_version": PARTITION_RULE_VERSION,
            }
            ctrl_rec["record_sha256"] = compute_record_sha256(ctrl_rec)

            # Build Treatment Record
            treat_item_id = compute_item_id(pair_id, "treatment")
            treat_ev_page = params.get("donor_page", p.ev_page)
            treat_ev_sid = params.get("donor_sentence_id", p.ev_sid)

            treat_rec: Dict[str, Any] = {
                "dataset_id": CANONICAL_DATASET_ID,
                "protocol_version": PROTOCOL_VERSION_V1_1,
                "pair_id": pair_id,
                "item_id": treat_item_id,
                "item_role": "TREATMENT",
                "source_dataset": SOURCE_DATASET,
                "source_revision": SOURCE_REVISION,
                "source_split": SOURCE_SPLIT,
                "parent_id": p.id,
                "source_reference_label": p.label,
                "asserted_verdict": "SUPPORTED" if p.label == "SUPPORTS" else "REFUTED",
                "original_claim": p.claim,
                "original_evidence": {
                    "page": p.ev_page,
                    "sentence_id": p.ev_sid,
                    "text": p.ev_text,
                },
                "presented_claim": pert_claim,
                "presented_evidence": {
                    "page": treat_ev_page,
                    "sentence_id": treat_ev_sid,
                    "text": pert_evidence_text,
                },
                "attack_family": fam,
                "attack_mode": params.get("attack_mode", "NONE"),
                "attack_parameters": params,
                "claim_text_changed": (pert_claim != p.claim),
                "evidence_text_changed": (pert_evidence_text != p.ev_text),
                "support_relation_hypothesis": "INVALID_SUPPORT",
                "adjudicated_support_relation": None,
                "adjudication_status": "NOT_IN_ADJUDICATION_POPULATION",
                "adjudication_primary_responses": None,
                "adjudication_adjudicator_response": None,
                "adjudication_protocol_sha256": adjudication_protocol_sha256,
                "hypothesis_confirmed": None,
                "certificate_validity_target": None,
                "external_world_truth_status": "UNVERIFIED",
                "partition": part,
                "partition_rule_id": PARTITION_RULE_ID,
                "partition_rule_version": PARTITION_RULE_VERSION,
            }
            treat_rec["record_sha256"] = compute_record_sha256(treat_rec)

            records.append(ctrl_rec)
            records.append(treat_rec)
            claimed_parents.add(p.id)
            seen_nf_claims.add(nf(p.claim))
            seen_nf_claims.add(nf(pert_claim))
            filled += 1

        if filled < quota:
            raise RuntimeError(f"Scientific Protocol Feasibility Failure: cell {fam}:{lab} filled {filled} of {quota}")

    # Verify disjointness
    advint_parents = {r["parent_id"] for r in records}
    overlap = advint_parents.intersection(dev_ids)
    if overlap:
        raise RuntimeError(f"TRAIN_MAIN_FEVER_PARENT_ID_INTERSECTION violation: {len(overlap)} overlapping IDs")

    # Compute roots
    pool_root = compute_pool_content_root(records)
    part_root = compute_partition_manifest_root(records)
    cand_root = compute_candidate_release_root(pool_root, part_root, adjudication_protocol_sha256)

    # Sort records canonically by item_id UTF-8 ascending
    records.sort(key=lambda r: r["item_id"].encode("utf-8"))

    return {
        "records": records,
        "pool_content_root": pool_root,
        "partition_manifest_root": part_root,
        "candidate_release_root": cand_root,
        "claimed_parent_count": len(claimed_parents),
        "total_items": len(records),
        "total_pairs": len(records) // 2,
    }


# ---------------------------------------------------------------------------
# QAExample Stream Adapter
# ---------------------------------------------------------------------------

def iter_adversarial_integrity(
    candidate_path: Optional[str] = None,
    partition: Optional[str] = None,
    n: Optional[int] = None,
    seed: int = 0,
) -> Iterator[QAExample]:
    """Yields QAExample instances from candidate release JSONL for PCG-MAS graphs."""
    if candidate_path is None:
        root = Path(__file__).resolve().parents[3]
        candidate_path = str(root / "artifacts" / "v3_0" / "datasets" / "adversarial_integrity" / "candidate" / "records.jsonl")

    if not os.path.exists(candidate_path):
        raise FileNotFoundError(f"Adversarial integrity candidate records not found at {candidate_path}")

    count = 0
    with open(candidate_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if partition and rec.get("partition") != partition:
                continue

            c = rec["presented_claim"]
            e = rec["presented_evidence"]
            v = rec["asserted_verdict"]

            prompt_text = f"Claim: {c}\nEvidence: {e['text']}\nAsserted Verdict: {v}\nDetermine if the evidence supports the claim:"
            ev_item = EvidenceItem(
                id=f"{rec['item_id']}_ev",
                title=e["page"],
                text=e["text"],
                source_url=None,
                publisher="FEVER_Wikipedia",
                domain="wikipedia.org",
                is_gold=True,
            )

            qa_ex = QAExample(
                id=rec["item_id"],
                question=prompt_text,
                gold_answers=(v,),
                evidence=(ev_item,),
                task_type="qa",
                meta={
                    "dataset": "adversarial_integrity",
                    "pair_id": rec["pair_id"],
                    "item_role": rec["item_role"],
                    "attack_family": rec["attack_family"],
                    "attack_mode": rec["attack_mode"],
                    "partition": rec["partition"],
                    "record_sha256": rec["record_sha256"],
                    "presented_claim": c,
                    "presented_evidence": e,
                    "asserted_verdict": v,
                },
            )
            yield qa_ex
            count += 1
            if n is not None and count >= n:
                break
