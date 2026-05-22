# scripts/v5_r3_open_mixed.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Set


MIXED_CASES = [
    ("Integrity+Coverage", {"Integrity", "Coverage"}),
    ("Drift+Coverage", {"Drift", "Coverage"}),
    ("Checker+Replay", {"Checker", "Replay"}),
    ("Unknown", {"Unknown"}),
]


def topk(pred_scores: dict, k: int) -> Set[str]:
    return set(sorted(pred_scores, key=pred_scores.get, reverse=True)[:k])


def multilabel_f1(pred: Set[str], gold: Set[str]) -> float:
    if not pred and not gold:
        return 1.0
    tp = len(pred & gold)
    precision = tp / max(len(pred), 1)
    recall = tp / max(len(gold), 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def score_record(rec: dict) -> dict:
    gold = set(rec["gold_channels"])
    scores = rec["responsibility_scores"]

    p1 = topk(scores, 1)
    p2 = topk(scores, 2)

    threshold = rec.get("threshold", 0.20)
    pmulti = {k for k, v in scores.items() if v >= threshold}
    if not pmulti:
        pmulti = {"Unknown"}

    rec["r3_top1"] = float(len(p1 & gold) > 0)
    rec["r3_top2"] = float(len(p2 & gold) > 0)
    rec["r3_multilabel_f1"] = multilabel_f1(pmulti, gold)
    rec["r3_unknown_correct"] = float(("Unknown" in gold) == ("Unknown" in pmulti))
    return rec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    args = parser.parse_args()

    out = Path(args.output_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(args.input_jsonl, "r", encoding="utf-8") as f, out.open("w", encoding="utf-8") as g:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            rec = score_record(rec)
            g.write(json.dumps(rec, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()