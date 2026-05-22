#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def norm(x: Any) -> str:
    return str(x or "").strip().lower().replace("_", "-").replace(" ", "-")


def parse_pairs(text: str) -> set[tuple[str, str]] | None:
    text = (text or "").strip()
    if text.lower() == "all":
        return None
    out = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise SystemExit(f"Invalid pair: {part}. Expected dataset:model")
        d, m = part.split(":", 1)
        out.add((norm(d), norm(m)))
    return out


def parse_seeds(text: str) -> set[int] | None:
    text = (text or "").strip()
    if text.lower() == "all":
        return None
    return {int(x.strip()) for x in text.split(",") if x.strip()}


def safe_rate(x: Any, default: float | None = None) -> float | None:
    if x is None:
        return default
    try:
        y = float(x)
        if y != y:
            return default
        return y
    except Exception:
        return default


def find_template(base_rows: list[dict[str, Any]], dataset: str, model: str, seed: int) -> dict[str, Any]:
    nd = norm(dataset)
    nm = norm(model)

    for r in base_rows:
        if norm(r.get("dataset")) == nd and norm(r.get("model")) == nm and int(r.get("seed", seed)) == seed:
            return copy.deepcopy(r)

    for r in base_rows:
        if norm(r.get("dataset")) == nd and norm(r.get("model")) == nm:
            return copy.deepcopy(r)

    for r in base_rows:
        if norm(r.get("dataset")) == nd:
            return copy.deepcopy(r)

    return {}


def utility_from_accept_false_accept(accept_rate: float | None, fa_rate: float | None) -> float | None:
    if accept_rate is None:
        return None
    if fa_rate is None:
        return accept_rate
    return accept_rate * (1.0 - fa_rate)


def shield_summary_to_metric_row(summary: dict[str, Any], base_rows: list[dict[str, Any]]) -> dict[str, Any]:
    dataset = summary.get("dataset")
    model = summary.get("model")
    seed = int(summary.get("seed", 0))
    n = int(summary.get("n", 0))

    r1 = summary.get("R1_checkability", {})
    r2 = summary.get("R2_redundancy", {})
    r3 = summary.get("R3_responsibility", {})
    r4 = summary.get("R4_risk_control", {})
    r5 = summary.get("R5_overhead", {})

    accept_rate = safe_rate(r1.get("accept_rate"))
    block_rate = safe_rate(r1.get("block_rate"))
    verify_rate = safe_rate(r1.get("verify_rate"))
    fa_rate = safe_rate(r1.get("false_accept_proxy_rate_among_known"))
    utility_proxy = utility_from_accept_false_accept(accept_rate, fa_rate)

    row = find_template(base_rows, dataset, model, seed)

    row.update({
        "method": "shieldagent",
        "baseline": "ShieldAgent/AutoPolicy",
        "baseline_family": "policy_verifier",
        "sota_method": "shieldagent",
        "source": "results/baselines/shieldagent/r1_r5",
        "dataset": dataset,
        "model": model,
        "seed": seed,
        "n_examples": n,
        "n": n,

        "coverage": accept_rate,
        "answer_rate": accept_rate,
        "accept_rate": accept_rate,
        "block_rate": block_rate,
        "verify_rate": verify_rate,
        "utility": utility_proxy,
        "utility_proxy": utility_proxy,

        "false_accept_rate": fa_rate,
        "false_accept_proxy_rate": fa_rate,
        "harm": fa_rate,
        "harm_proxy": fa_rate,

        "r1_accept_rate": accept_rate,
        "r1_block_rate": block_rate,
        "r1_verify_rate": verify_rate,
        "r1_false_accept_proxy_rate": fa_rate,
        "r1_rule_check_failure_proxy_rate": safe_rate(r1.get("rule_check_failure_proxy_rate")),

        "r2_k": r2.get("k"),
        "r2_quorum_accept_n": r2.get("quorum_accept_n"),
        "r2_quorum_block_n": r2.get("quorum_block_n"),
        "r2_quorum_verify_n": r2.get("quorum_verify_n"),
        "r2_quorum_accept_rate": safe_rate(r2.get("quorum_accept_n"), 0.0) / n if n else None,

        "r3_total_decision_flips": r3.get("total_decision_flips"),
        "r3_mean_decision_flips_per_record": safe_rate(r3.get("mean_decision_flips_per_record")),

        "r4_frontier_available": bool(r4.get("frontier")),
        "r4_frontier": r4.get("frontier"),

        "latency_mean_s": safe_rate(r5.get("latency_mean_s")),
        "latency_total_s": safe_rate(r5.get("latency_total_s")),
        "tokens_est_total": r5.get("tokens_est_total"),
        "api_call_count_total": r5.get("api_call_count_total"),
        "throughput_records_per_s": safe_rate(r5.get("throughput_records_per_s_main_r1")),

        "is_baseline": True,
        "is_shieldagent": True,
        "is_pcg_mas": False,

        "metric_alignment_note": (
            "ShieldAgent R1/R4/R5 are comparable policy-verifier metrics. "
            "R2/R3 are proxy-only: quorum over policy-bank views and decision flips under field ablations."
        ),
    })

    return row


def discover_shield_summaries(root: Path) -> list[Path]:
    return sorted(root.glob("*/summary.json"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-rows", default="results/tables/csv/paper_metrics.jsonl")
    parser.add_argument("--shield-root", default="results/baselines/shieldagent/r1_r5")
    parser.add_argument("--pairs", default="all", help="Comma-separated dataset:model pairs, or all")
    parser.add_argument("--seeds", default="all", help="Comma-separated seeds, or all")
    parser.add_argument("--out-rows", default="results/tables/csv/paper_metrics_with_shieldagent.jsonl")
    parser.add_argument("--shield-only-rows", default="results/tables/csv/shieldagent_metrics_for_figures.jsonl")
    parser.add_argument("--report", default="results/tables/csv/shieldagent_merge_report.json")
    args = parser.parse_args()

    base_rows_path = Path(args.base_rows)
    shield_root = Path(args.shield_root)

    if not base_rows_path.exists():
        raise SystemExit(f"Missing base PCG-MAS rows: {base_rows_path}")

    if not shield_root.exists():
        raise SystemExit(f"Missing ShieldAgent R1-R5 root: {shield_root}")

    wanted_pairs = parse_pairs(args.pairs)
    wanted_seeds = parse_seeds(args.seeds)

    base_rows = read_jsonl(base_rows_path)
    shield_rows = []

    selected = []
    skipped = []

    for path in discover_shield_summaries(shield_root):
        obj = read_json(path)
        dataset = norm(obj.get("dataset"))
        model = norm(obj.get("model"))
        seed = int(obj.get("seed", 0))

        if wanted_pairs is not None and (dataset, model) not in wanted_pairs:
            skipped.append({"path": str(path), "reason": "pair_not_selected"})
            continue

        if wanted_seeds is not None and seed not in wanted_seeds:
            skipped.append({"path": str(path), "reason": "seed_not_selected"})
            continue

        row = shield_summary_to_metric_row(obj, base_rows)
        row["shieldagent_summary_path"] = str(path)
        shield_rows.append(row)
        selected.append({"path": str(path), "dataset": obj.get("dataset"), "model": obj.get("model"), "seed": seed})

    if not shield_rows:
        raise SystemExit("No ShieldAgent summaries selected. Check --pairs and --seeds.")

    merged_rows = base_rows + shield_rows

    write_jsonl(Path(args.out_rows), merged_rows)
    write_jsonl(Path(args.shield_only_rows), shield_rows)

    report = {
        "base_rows": str(base_rows_path),
        "shield_root": str(shield_root),
        "out_rows": args.out_rows,
        "shield_only_rows": args.shield_only_rows,
        "num_base_rows": len(base_rows),
        "num_shieldagent_rows_added": len(shield_rows),
        "num_merged_rows": len(merged_rows),
        "selected": selected,
        "skipped": skipped,
        "protected_figures_policy": [
            "ablations",
            "r1_five_channel_audit",
            "r3_open_mixed",
            "r4_privacy_frontier",
            "r5_scaling",
        ],
    }

    write_json(Path(args.report), report)

    print(json.dumps(report, indent=2, sort_keys=True))
    print("SHIELDAGENT_METRICS_MERGE_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
