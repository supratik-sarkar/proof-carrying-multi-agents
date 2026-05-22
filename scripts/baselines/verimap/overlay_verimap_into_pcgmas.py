#!/usr/bin/env python
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


ROWS_PATH = Path("results/tables/csv/paper_metrics.jsonl")
AGG_PATH = Path("results/baselines/verimap/r1_r5/aggregate_by_dataset_model.json")
AUDIT_PATH = Path("results/tables/csv/verimap_overlay_audit.json")


def norm_dataset(x: Any) -> str:
    y = str(x or "").strip().lower()
    return {"tat-qa": "tatqa", "tat_qa": "tatqa", "tata-qa": "tatqa"}.get(y, y)


def norm_model(x: Any) -> str:
    y = str(x or "").strip().lower()
    return {
        "qwen/qwen2.5-7b-instruct": "qwen2.5-7b",
        "qwen2.5-7b-instruct": "qwen2.5-7b",
        "qwen-2.5-7b": "qwen2.5-7b",
        "qwen2-5-7b": "qwen2.5-7b",
        "microsoft/phi-3.5-mini-instruct": "phi-3.5-mini",
        "phi-3.5-mini-instruct": "phi-3.5-mini",
        "meta-llama/llama-3.1-8b-instruct": "llama-3.1-8b",
        "llama-3.1-8b-instruct": "llama-3.1-8b",
        "google/gemma-2-9b-it": "gemma-2-9b-it",
    }.get(y, y)


def mean_stat(obj: Any) -> float | None:
    return obj.get("mean") if isinstance(obj, dict) else None


def finite(x: Any, default: float | None = None) -> float | None:
    try:
        if x is None:
            return default
        y = float(x)
        if math.isnan(y) or math.isinf(y):
            return default
        return y
    except Exception:
        return default


def first_nonnull(*values: Any, default: Any = None) -> Any:
    for value in values:
        if value is not None:
            return value
    return default


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]



# ---------------------------------------------------------------------
# Overlay persistence / rehydration
# ---------------------------------------------------------------------
# Each adapter owns its own overlay fields, but appendix_hero_v4 is a
# joint comparison figure. If another merge step recollects base
# paper_metrics.jsonl, previously applied SOTA fields may disappear from
# the active rows. To make each adapter merge order-independent, every
# overlay script persists its own fields to a small sidecar and rehydrates
# all known sidecars before writing paper_metrics.jsonl.

CURRENT_OVERLAY_METHOD = "verimap"
CURRENT_OVERLAY_SIDECAR = Path("results/tables/csv/verimap_overlay_rows.jsonl")
KNOWN_OVERLAY_SIDECARS = {
    "shieldagent": Path("results/tables/csv/shieldagent_overlay_rows.jsonl"),
    "agentrr": Path("results/tables/csv/agentrr_overlay_rows.jsonl"),
    "verimap": Path("results/tables/csv/verimap_overlay_rows.jsonl"),
    "atlasprism": Path("results/tables/csv/atlasprism_overlay_rows.jsonl"),
    "pcnrec": Path("results/tables/csv/pcnrec_overlay_rows.jsonl"),
    "clbc": Path("results/tables/csv/clbc_overlay_rows.jsonl"),
}
OVERLAY_METHOD_ALIASES = {
    "shieldagent": {"shieldagent", "shield"},
    "agentrr": {"agentrr"},
    "verimap": {"verimap"},
    "atlasprism": {"atlasprism", "prism", "atlas"},
    "pcnrec": {"pcnrec", "pcn_rec"},
    "clbc": {"clbc"},
}


def row_key(row: dict[str, Any]) -> tuple[str, str]:
    return (norm_dataset(row.get("dataset")), norm_model(row.get("model")))


def is_overlay_field_for_method(field: str, method: str) -> bool:
    aliases = OVERLAY_METHOD_ALIASES.get(method, {method})
    for alias in aliases:
        if field == f"{alias}_overlay_applied":
            return True
        if field.startswith(f"{alias}_"):
            return True
        if field.endswith(f"_{alias}"):
            return True
        for stem in (
            "clean_harm",
            "adv_harm",
            "harm_clean",
            "harm_adv",
            "bound_gap",
            "tokens",
            "token",
            "latency",
            "utility",
        ):
            if field == f"{stem}_{alias}":
                return True
        if field.startswith("R1_") and field.endswith(f"_{alias}"):
            return True
        if field.startswith("R5_") and field.endswith(f"_{alias}"):
            return True
    return False


def extract_overlay_fields(row: dict[str, Any], method: str) -> dict[str, Any]:
    out = {
        "dataset": row.get("dataset"),
        "model": row.get("model"),
        "_norm_dataset": norm_dataset(row.get("dataset")),
        "_norm_model": norm_model(row.get("model")),
    }
    for k, v in row.items():
        if is_overlay_field_for_method(k, method):
            out[k] = v
    return out


def write_current_overlay_sidecar(rows: list[dict[str, Any]]) -> None:
    CURRENT_OVERLAY_SIDECAR.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    for row in rows:
        if row.get(f"{CURRENT_OVERLAY_METHOD}_overlay_applied"):
            fields = extract_overlay_fields(row, CURRENT_OVERLAY_METHOD)
            if len(fields) > 4:
                entries.append(fields)
    CURRENT_OVERLAY_SIDECAR.write_text(
        "".join(json.dumps(x, sort_keys=True) + "\\n" for x in entries),
        encoding="utf-8",
    )


def load_overlay_sidecars() -> dict[tuple[str, str], dict[str, Any]]:
    overlays: dict[tuple[str, str], dict[str, Any]] = {}
    for method, path in KNOWN_OVERLAY_SIDECARS.items():
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            key = (
                norm_dataset(obj.get("_norm_dataset") or obj.get("dataset")),
                norm_model(obj.get("_norm_model") or obj.get("model")),
            )
            fields = {
                k: v for k, v in obj.items()
                if not k.startswith("_") and k not in {"dataset", "model"}
            }
            if not fields:
                continue
            overlays.setdefault(key, {}).update(fields)
    return overlays


def rehydrate_overlay_sidecars(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    overlays = load_overlay_sidecars()
    if not overlays:
        return rows
    out = []
    for row in rows:
        rr = dict(row)
        rr.update(overlays.get(row_key(rr), {}))
        out.append(rr)
    return out



def main() -> int:
    if not ROWS_PATH.exists():
        raise SystemExit(f"Missing paper metrics: {ROWS_PATH}")
    if not AGG_PATH.exists():
        raise SystemExit(f"Missing VeriMAP aggregate: {AGG_PATH}")

    rows = rehydrate_overlay_sidecars(load_jsonl(ROWS_PATH))
    aggs = json.loads(AGG_PATH.read_text())

    by_key = {
        (norm_dataset(a.get("dataset")), norm_model(a.get("model"))): a
        for a in aggs
    }

    matched = 0
    unmatched = []
    updated = []

    for row in rows:
        rr = dict(row)
        key = (norm_dataset(rr.get("dataset")), norm_model(rr.get("model")))
        agg = by_key.get(key)

        if agg is None:
            unmatched.append({"dataset": rr.get("dataset"), "model": rr.get("model"), "normalized_key": list(key)})
            updated.append(rr)
            continue

        matched += 1

        accept = finite(mean_stat(agg.get("R1_accept_rate")), 1.0)
        block = finite(mean_stat(agg.get("R1_block_rate")), 0.0)
        verify = finite(mean_stat(agg.get("R1_verify_rate")), 0.0)
        false_accept = finite(mean_stat(agg.get("R1_false_accept_proxy_rate_among_known")), None)

        harm_mean = finite(
            first_nonnull(
                mean_stat(agg.get("harm_under_corruption_mean")),
                mean_stat(agg.get("harm_under_corruption")),
            ),
            0.0,
        )
        harm_max = finite(
            first_nonnull(
                mean_stat(agg.get("harm_under_corruption_max")),
                mean_stat(agg.get("harm_under_corruption")),
            ),
            harm_mean,
        )
        audit_coverage = finite(
            first_nonnull(
                mean_stat(agg.get("audit_coverage_on_bad_accepts")),
                mean_stat(agg.get("audit_coverage_mean")),
            ),
            0.0,
        )

        backend_mode = agg.get("backend_mode")
        if isinstance(backend_mode, list):
            backend_mode = ",".join(str(x) for x in backend_mode)

        tokens_abs = finite(mean_stat(agg.get("R5_tokens_est_total")), None)
        prompt_tokens_abs = finite(mean_stat(agg.get("R5_prompt_tokens_total")), None)
        completion_tokens_abs = finite(mean_stat(agg.get("R5_completion_tokens_total")), None)
        n_base = finite(rr.get("n_examples"), 3.0) or 3.0
        nocert_tok = finite(rr.get("tokens_nocert"), 1.0) or 1.0
        token_multiplier = max(1.0, (tokens_abs or 1.0) / max(n_base, 1.0) / max(nocert_tok, 1.0))

        suite = agg.get("stress_suite_used") or [
            "clean_plan",
            "drop_support_step",
            "contradict_verification_criterion",
            "insert_distractor_step",
            "shuffle_plan_context",
            "answer_evidence_mismatch",
        ]

        rr.update(
            {
                "verimap_overlay_applied": True,
                "verimap_artifact_source": "results/baselines/verimap/r1_r5",
                "verimap_implementation_mode": "verimap_style_verification_aware_planning_adapter",
                "verimap_backend_mode": backend_mode,
                "verimap_accept_rate": accept,
                "verimap_block_rate": block,
                "verimap_verify_rate": verify,
                "verimap_false_accept_proxy_rate": false_accept,
                "verimap_harm_under_corruption_mean": harm_mean,
                "verimap_harm_under_corruption_max": harm_max,
                "verimap_audit_coverage": audit_coverage,
                "verimap_audit_coverage_on_bad_accepts": audit_coverage,
                "verimap_json_parse_success_rate": finite(mean_stat(agg.get("json_parse_success_rate")), None),
                "verimap_json_repair_rate": finite(mean_stat(agg.get("json_repair_rate")), None),
                "verimap_invalid_response_rate": finite(mean_stat(agg.get("invalid_response_rate")), None),
                "verimap_stress_suite_used": json.dumps(suite),
                "clean_harm_verimap": harm_mean,
                "adv_harm_verimap": harm_max,
                "harm_clean_verimap": harm_mean,
                "harm_adv_verimap": harm_max,
                "bound_gap_verimap": max(0.0, 1.0 - (audit_coverage or 0.0)),
                "verimap_bound_gap": max(0.0, 1.0 - (audit_coverage or 0.0)),
                "latency_verimap": finite(mean_stat(agg.get("R5_latency_mean_s")), None),
                "tokens_verimap": token_multiplier,
                "token_verimap": token_multiplier,
                "tokens_verimap_abs_total": tokens_abs,
                "verimap_prompt_tokens_abs_total": prompt_tokens_abs,
                "verimap_completion_tokens_abs_total": completion_tokens_abs,
                "utility_verimap": accept,
            }
        )

        updated.append(rr)

    write_current_overlay_sidecar(updated)
    updated = rehydrate_overlay_sidecars(updated)
    ROWS_PATH.write_text(
        "\n".join(json.dumps(x, sort_keys=True) for x in updated) + "\n",
        encoding="utf-8",
    )

    audit = {
        "matched_rows": matched,
        "unmatched_rows": unmatched,
        "verimap_keys": [list(k) for k in sorted(by_key)],
        "paper_metric_rows": len(rows),
        "hardened_mapping": True,
        "overlay_sidecar": str(CURRENT_OVERLAY_SIDECAR),
        "rehydrated_known_sidecars": [str(p) for p in KNOWN_OVERLAY_SIDECARS.values() if p.exists()],
    }
    AUDIT_PATH.write_text(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps(audit, indent=2, sort_keys=True))

    if matched == 0:
        raise SystemExit("No VeriMAP rows matched paper_metrics.jsonl.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
