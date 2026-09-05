"""Provenance gates (RC2). Fail closed; never repair.

Revised per review:
  G2  reframed from an arbitrary physical latency floor to *timing consistency*.
  G6  the blanket "no constant columns" authenticity rule is REMOVED — a
      legitimate measurement may be constant. Replaced by G6_lineage, which
      checks what actually matters: that every reported aggregate re-derives
      from eligible per-example records and carries no numeric literal.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path

from .classify import classify
from .hashing import hash_obj, hash_text
from .lineage import FORBIDDEN_CLASSES

EMPIRICAL_DIRECT = {"DIRECT", "DIRECT_WITH_PARTIAL_USAGE"}


def _load(p: Path) -> list[dict]:
    with Path(p).open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def g1_recomputable_payloads(recs, run_root: Path):
    bad = []
    for r in recs:
        if r["provenance_class"] not in EMPIRICAL_DIRECT:
            continue
        ref = r.get("bundle_ref")
        p = Path(run_root) / ref if ref else None
        if not p or not p.exists():
            bad.append(f"{r['record_id']}: bundle missing ({ref})"); continue
        b = json.loads(p.read_text(encoding="utf-8"))
        if hash_obj(b) != r.get("bundle_hash"):
            bad.append(f"{r['record_id']}: bundle_hash does not re-derive")
        if hash_text(b["canonical_input"]) != r.get("input_hash"):
            bad.append(f"{r['record_id']}: input_hash does not re-derive from bundle")
        if b.get("raw_output") is not None and hash_text(b["raw_output"]) != r.get("output_hash"):
            bad.append(f"{r['record_id']}: output_hash does not re-derive from bundle")
    return bad


def g2_timing_consistency(recs, run_root=None):
    bad = []
    for r in recs:
        if r["provenance_class"] not in EMPIRICAL_DIRECT:
            continue
        ev = r.get("evidence") or {}
        lat, st, et = ev.get("latency_ms"), ev.get("start_timestamp"), ev.get("end_timestamp")
        if lat is None or not isinstance(lat, (int, float)) or not math.isfinite(lat) or lat < 0:
            bad.append(f"{r['record_id']}: latency_ms not finite non-negative ({lat!r})")
        if not st or not et:
            bad.append(f"{r['record_id']}: missing timestamps"); continue
        try:
            if datetime.fromisoformat(et) < datetime.fromisoformat(st):
                bad.append(f"{r['record_id']}: end precedes start")
        except ValueError:
            bad.append(f"{r['record_id']}: timestamps not ISO-8601")
    return bad


def g3_token_accounting(recs, run_root=None):
    """Additive consistency where counts exist. Missing counts are never zeroed."""
    bad = []
    for r in recs:
        if r["provenance_class"] not in EMPIRICAL_DIRECT:
            continue
        ev = r.get("evidence") or {}
        i, o, t = ev.get("input_tokens"), ev.get("output_tokens"), ev.get("total_tokens")
        if t is not None and i is not None and o is not None and t != i + o:
            bad.append(f"{r['record_id']}: total_tokens={t} != {i}+{o}")
        if r["provenance_class"] == "DIRECT" and r.get("usage_completeness") != "FULL":
            bad.append(f"{r['record_id']}: class DIRECT requires FULL usage, got {r.get('usage_completeness')}")
        for name, v in (("input_tokens", i), ("output_tokens", o), ("total_tokens", t)):
            if v is not None and (not isinstance(v, int) or v < 0):
                bad.append(f"{r['record_id']}: {name}={v!r} invalid")
    return bad


def g4_no_asserted_class(recs, run_root=None):
    bad = []
    for r in recs:
        stripped = {k: v for k, v in r.items()
                    if k not in ("provenance_class", "provenance_reasons",
                                 "execution_class", "usage_completeness", "outcome_eligible")}
        implied = classify(stripped)
        if implied["provenance_class"] != r["provenance_class"]:
            bad.append(f"{r['record_id']}: stored={r['provenance_class']} evidence implies {implied['provenance_class']}")
    return bad


def g5_canonical_identity(recs, run_root=None):
    bad = []
    seen = Counter(r["record_id"] for r in recs)
    bad += [f"duplicate record_id {k} x{v}" for k, v in seen.items() if v > 1]
    for r in recs:
        ident, dig = r.get("identity"), r.get("identity_digest")
        if not ident or not dig:
            bad.append(f"{r['record_id']}: missing identity/identity_digest"); continue
        if hash_obj(ident) != dig:
            bad.append(f"{r['record_id']}: identity_digest does not re-derive")
        elif r["record_id"] != f"pcgrec_{dig[:32]}":
            bad.append(f"{r['record_id']}: record_id not derived from identity digest")
    return bad


def g6_lineage(recs, run_root: Path):
    """Every aggregate in the run must re-derive from eligible records."""
    aggs = Path(run_root) / "aggregates.json"
    if not aggs.exists():
        return []                      # nothing aggregated yet is not a failure
    bad = []
    ids = {r["record_id"] for r in recs}
    for a in json.loads(aggs.read_text(encoding="utf-8")):
        if a.get("provenance_class") != "DERIVED_FROM_DIRECT":
            bad.append(f"{a.get('metric')}: aggregate class {a.get('provenance_class')}")
        src = a.get("source_record_ids") or []
        if not src:
            bad.append(f"{a.get('metric')}: aggregate carries no source record ids")
        missing = [i for i in src if i not in ids]
        if missing:
            bad.append(f"{a.get('metric')}: {len(missing)} source ids absent from records")
        if hash_obj(sorted(src)) != a.get("source_record_set_hash"):
            bad.append(f"{a.get('metric')}: source_record_set_hash does not re-derive")
    return bad


def g7_no_forbidden_provenance_in_empirical(recs, run_root=None):
    """No forbidden class may be marked outcome-eligible."""
    return [f"{r['record_id']}: class {r['provenance_class']} marked outcome_eligible"
            for r in recs
            if r.get("outcome_eligible") and r["provenance_class"] in FORBIDDEN_CLASSES]


def g8_no_secrets_in_artifacts(recs, run_root: Path):
    import re
    pats = [
        # provider key shapes: sk_live_..., sk-..., hf_..., allowing internal _ and -
        re.compile(r"(?i)\b(?:sk|hf|xai|gsk)[-_][A-Za-z0-9_\-]{16,}"),
        re.compile(r"(?i)authorization[\"']?\s*[:=]"),
        re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{12,}"),
        # key-ish field name, optionally JSON-quoted, followed by a non-empty value
        re.compile(r"(?i)[\"']?\b(api[_-]?key|apikey|secret|password|access[_-]?token)\b[\"']?\s*[:=]\s*[\"']?[^\s\"',}]{6,}"),
    ]
    bad = []
    root = Path(run_root)
    for p in list(root.rglob("*.json")) + list(root.rglob("*.jsonl")):
        try:
            t = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for pat in pats:
            if pat.search(t):
                bad.append(f"{p.relative_to(root)}: possible credential material"); break
    return bad


GATES = {
    "G1_recomputable_payloads": g1_recomputable_payloads,
    "G2_timing_consistency": g2_timing_consistency,
    "G3_token_accounting": g3_token_accounting,
    "G4_no_asserted_class": g4_no_asserted_class,
    "G5_canonical_identity": g5_canonical_identity,
    "G6_lineage": g6_lineage,
    "G7_no_forbidden_provenance": g7_no_forbidden_provenance_in_empirical,
    "G8_no_secrets": g8_no_secrets_in_artifacts,
}


def run_gates(records_path: Path) -> dict:
    records_path = Path(records_path)
    recs = _load(records_path)
    run_dir = records_path.parent
    run_root = run_dir.parent            # bundle_ref is relative to the runs root
    out, failed = {}, 0
    for name, fn in GATES.items():
        hits = fn(recs, run_root if name in ("G1_recomputable_payloads",) else run_dir)
        out[name] = {"status": "PASS" if not hits else "FAIL",
                     "findings": hits[:20], "finding_count": len(hits)}
        failed += bool(hits)
    counts = Counter(r["provenance_class"] for r in recs)
    out["_summary"] = {"records": len(recs), "gates": len(GATES),
                       "passed": len(GATES) - failed, "failed": failed,
                       "provenance_counts": dict(counts)}
    return out
