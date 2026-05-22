#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def norm(x: Any) -> str:
    return str(x or "").strip().lower().replace("_", "-").replace(" ", "-")


def mean(xs: list[float | None]) -> float | None:
    ys = [float(x) for x in xs if x is not None]
    return sum(ys) / len(ys) if ys else None


def safe_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        y = float(x)
        if y != y:
            return None
        return y
    except Exception:
        return None


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
            raise SystemExit(f"Invalid pair '{part}'. Expected dataset:model")
        d, m = part.split(":", 1)
        out.add((norm(d), norm(m)))
    return out


def parse_seeds(text: str) -> set[int] | None:
    text = (text or "").strip()
    if text.lower() == "all":
        return None
    return {int(x.strip()) for x in text.split(",") if x.strip()}


def discover_summaries(root: Path) -> list[Path]:
    return sorted(root.glob("*/summary.json"))


def load_selected_summaries(root: Path, pairs: set[tuple[str, str]] | None, seeds: set[int] | None) -> list[dict[str, Any]]:
    selected = []
    for p in discover_summaries(root):
        obj = read_json(p)
        d = norm(obj.get("dataset"))
        m = norm(obj.get("model"))
        seed = int(obj.get("seed", 0))

        if pairs is not None and (d, m) not in pairs:
            continue
        if seeds is not None and seed not in seeds:
            continue

        obj["_summary_path"] = str(p)
        selected.append(obj)

    if not selected:
        raise SystemExit("No ShieldAgent summary.json files selected.")
    return selected


def summary_to_wide(summary: dict[str, Any]) -> dict[str, Any]:
    dataset = summary.get("dataset")
    model = summary.get("model")
    seed = int(summary.get("seed", 0))
    n = int(summary.get("n", 0))

    r1 = summary.get("R1_checkability", {})
    r2 = summary.get("R2_redundancy", {})
    r3 = summary.get("R3_responsibility", {})
    r5 = summary.get("R5_overhead", {})

    accept_rate = safe_float(r1.get("accept_rate"))
    block_rate = safe_float(r1.get("block_rate"))
    verify_rate = safe_float(r1.get("verify_rate"))
    fa_rate = safe_float(r1.get("false_accept_proxy_rate_among_known"))
    latency = safe_float(r5.get("latency_mean_s"))
    tokens = safe_float(r5.get("tokens_est_total"))

    risk_proxy = None
    if block_rate is not None or verify_rate is not None:
        risk_proxy = float(block_rate or 0.0) + float(verify_rate or 0.0)
    elif accept_rate is not None:
        risk_proxy = 1.0 - float(accept_rate)

    harm_value = fa_rate if fa_rate is not None else risk_proxy
    harm_is_observed = fa_rate is not None
    harm_field_source = (
        "false_accept_proxy_rate_among_known"
        if fa_rate is not None
        else "risk_proxy_from_block_plus_verify_rate_not_true_false_accept"
    )

    return {
        "dataset": dataset,
        "model": model,
        "seed": seed,
        "n": n,

        "harm_clean_shield": harm_value,
        "harm_adv_shield": harm_value,
        "clean_harm_shieldagent": harm_value,
        "adv_harm_shieldagent": harm_value,

        "token_shield": tokens,
        "tokens_shieldagent": tokens,
        "latency_shield": latency,
        "latency_shieldagent": latency,

        "shieldagent_accept_rate": accept_rate,
        "shieldagent_block_rate": block_rate,
        "shieldagent_verify_rate": verify_rate,
        "shieldagent_false_accept_proxy_rate": fa_rate,
        "shieldagent_false_accept_proxy_observed": harm_is_observed,
        "shieldagent_harm_field_source": harm_field_source,
        "shieldagent_harm_missing_reason": None if harm_is_observed else "false_accept_proxy_rate_among_known is null; using ShieldAgent block+verify risk proxy for figure comparability",

        "shieldagent_r2_quorum_accept_n": r2.get("quorum_accept_n"),
        "shieldagent_r2_quorum_block_n": r2.get("quorum_block_n"),
        "shieldagent_r2_quorum_verify_n": r2.get("quorum_verify_n"),

        "shieldagent_r3_total_decision_flips": r3.get("total_decision_flips"),
        "shieldagent_r3_mean_decision_flips_per_record": r3.get("mean_decision_flips_per_record"),

        "shieldagent_api_call_count_total": r5.get("api_call_count_total"),
        "shieldagent_latency_total_s": r5.get("latency_total_s"),
        "shieldagent_tokens_est_total": r5.get("tokens_est_total"),

        "shieldagent_implementation_mode": "official_authors_policy_extraction_plus_pcgmas_cell_evaluation",
        "shieldagent_summary_path": summary.get("_summary_path"),
    }


def aggregate_wide(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = defaultdict(list)
    for r in rows:
        groups[(norm(r["dataset"]), norm(r["model"]))].append(r)

    out = []
    for (_, _), vals in sorted(groups.items()):
        first = vals[0]
        agg = {
            "dataset": first["dataset"],
            "model": first["model"],
            "seeds": [v["seed"] for v in vals],
            "n_seeds": len(vals),
            "n": mean([safe_float(v.get("n")) for v in vals]),

            "harm_clean_shield": mean([safe_float(v.get("harm_clean_shield")) for v in vals]),
            "harm_adv_shield": mean([safe_float(v.get("harm_adv_shield")) for v in vals]),
            "clean_harm_shieldagent": mean([safe_float(v.get("clean_harm_shieldagent")) for v in vals]),
            "adv_harm_shieldagent": mean([safe_float(v.get("adv_harm_shieldagent")) for v in vals]),

            "token_shield": mean([safe_float(v.get("token_shield")) for v in vals]),
            "tokens_shieldagent": mean([safe_float(v.get("tokens_shieldagent")) for v in vals]),
            "latency_shield": mean([safe_float(v.get("latency_shield")) for v in vals]),
            "latency_shieldagent": mean([safe_float(v.get("latency_shieldagent")) for v in vals]),

            "shieldagent_accept_rate": mean([safe_float(v.get("shieldagent_accept_rate")) for v in vals]),
            "shieldagent_block_rate": mean([safe_float(v.get("shieldagent_block_rate")) for v in vals]),
            "shieldagent_verify_rate": mean([safe_float(v.get("shieldagent_verify_rate")) for v in vals]),
            "shieldagent_false_accept_proxy_rate": mean([safe_float(v.get("shieldagent_false_accept_proxy_rate")) for v in vals]),

            "shieldagent_r2_quorum_accept_n": mean([safe_float(v.get("shieldagent_r2_quorum_accept_n")) for v in vals]),
            "shieldagent_r2_quorum_block_n": mean([safe_float(v.get("shieldagent_r2_quorum_block_n")) for v in vals]),
            "shieldagent_r2_quorum_verify_n": mean([safe_float(v.get("shieldagent_r2_quorum_verify_n")) for v in vals]),

            "shieldagent_r3_total_decision_flips": mean([safe_float(v.get("shieldagent_r3_total_decision_flips")) for v in vals]),
            "shieldagent_r3_mean_decision_flips_per_record": mean([safe_float(v.get("shieldagent_r3_mean_decision_flips_per_record")) for v in vals]),

            "shieldagent_api_call_count_total": mean([safe_float(v.get("shieldagent_api_call_count_total")) for v in vals]),
            "shieldagent_latency_total_s": mean([safe_float(v.get("shieldagent_latency_total_s")) for v in vals]),
            "shieldagent_tokens_est_total": mean([safe_float(v.get("shieldagent_tokens_est_total")) for v in vals]),

            "shieldagent_implementation_mode": "official_authors_policy_extraction_plus_pcgmas_cell_evaluation",
            "shieldagent_summary_paths": [v.get("shieldagent_summary_path") for v in vals],
            "shieldagent_harm_observed": any(v.get("shieldagent_false_accept_proxy_observed") for v in vals),
            "shieldagent_harm_field_source": (
                "false_accept_proxy_rate_among_known"
                if any(v.get("shieldagent_false_accept_proxy_observed") for v in vals)
                else "risk_proxy_from_block_plus_verify_rate_not_true_false_accept"
            ),
        }
        out.append(agg)

    return out


def overlay_on_base_rows(base_rows: list[dict[str, Any]], shield_agg: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(norm(r["dataset"]), norm(r["model"])): r for r in shield_agg}

    out = []
    for row in base_rows:
        new = dict(row)
        key = (norm(row.get("dataset")), norm(row.get("model")))
        if key in by_key:
            new.update(by_key[key])
            new["shieldagent_overlay_applied"] = True
        else:
            new["shieldagent_overlay_applied"] = False
        out.append(new)

    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-rows", default="results/tables/csv/paper_metrics.jsonl")
    parser.add_argument("--shield-root", default="results/baselines/shieldagent/r1_r5")
    parser.add_argument("--pairs", default="all")
    parser.add_argument("--seeds", default="all")
    parser.add_argument("--out-rows", default="results/tables/csv/paper_metrics_with_shieldagent_wide.jsonl")
    parser.add_argument("--shield-official-aggregates", default="results/tables/csv/shieldagent_outputs/official_shieldagent_aggregates.jsonl")
    parser.add_argument("--report", default="results/tables/csv/shieldagent_wide_merge_report.json")
    args = parser.parse_args()

    base_rows = read_jsonl(Path(args.base_rows))
    if not base_rows:
        raise SystemExit(f"No base rows found: {args.base_rows}")

    summaries = load_selected_summaries(Path(args.shield_root), parse_pairs(args.pairs), parse_seeds(args.seeds))
    per_seed = [summary_to_wide(s) for s in summaries]
    agg = aggregate_wide(per_seed)

    overlaid = overlay_on_base_rows(base_rows, agg)

    write_jsonl(Path(args.shield_official_aggregates), agg)
    write_jsonl(Path(args.out_rows), overlaid)

    report = {
        "base_rows": args.base_rows,
        "shield_root": args.shield_root,
        "out_rows": args.out_rows,
        "shield_official_aggregates": args.shield_official_aggregates,
        "num_base_rows": len(base_rows),
        "num_shield_summaries": len(summaries),
        "num_aggregated_cells": len(agg),
        "num_rows_with_overlay": sum(1 for r in overlaid if r.get("shieldagent_overlay_applied")),
        "aggregated_cells": [
            {
                "dataset": r.get("dataset"),
                "model": r.get("model"),
                "seeds": r.get("seeds"),
                "clean_harm_shieldagent": r.get("clean_harm_shieldagent"),
                "adv_harm_shieldagent": r.get("adv_harm_shieldagent"),
                "tokens_shieldagent": r.get("tokens_shieldagent"),
                "latency_shieldagent": r.get("latency_shieldagent"),
                "shieldagent_accept_rate": r.get("shieldagent_accept_rate"),
                "shieldagent_block_rate": r.get("shieldagent_block_rate"),
                "shieldagent_harm_observed": r.get("shieldagent_harm_observed"),
            }
            for r in agg
        ],
    }

    write_json(Path(args.report), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print("SHIELDAGENT_WIDE_METRICS_EXPORT_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
