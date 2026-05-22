#!/usr/bin/env python
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

ROWS_PATH = Path("results/tables/csv/paper_metrics.jsonl")
AGG_PATH = Path("results/baselines/shieldagent/r1_r5/aggregate_by_dataset_model.json")
MANIFEST_PATH = Path("results/baselines/shieldagent/r1_r5/manifest.json")
AUDIT_PATH = Path("results/tables/csv/shieldagent_overlay_audit.json")


def norm_dataset(x: Any) -> str:
    y = str(x or "").strip().lower()
    return {
        "tat-qa": "tatqa",
        "tat_qa": "tatqa",
        "tata-qa": "tatqa",
        "2wikimultihopqa": "twowiki",
        "2wiki": "twowiki",
    }.get(y, y)


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


def mean_stat(obj: Any, default: float | None = None) -> float | None:
    if isinstance(obj, dict):
        return obj.get("mean", default)
    return default


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


def safe_rate(num: Any, den: Any) -> float | None:
    n = finite(num, None)
    d = finite(den, None)
    if n is None or d in (None, 0.0):
        return None
    return n / d


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

CURRENT_OVERLAY_METHOD = "shieldagent"
CURRENT_OVERLAY_SIDECAR = Path("results/tables/csv/shieldagent_overlay_rows.jsonl")
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
        raise SystemExit(f"Missing ShieldAgent aggregate: {AGG_PATH}")

    rows = rehydrate_overlay_sidecars(load_jsonl(ROWS_PATH))
    aggs = json.loads(AGG_PATH.read_text())
    manifest = json.loads(MANIFEST_PATH.read_text()) if MANIFEST_PATH.exists() else {}

    by_key = {(norm_dataset(a.get("dataset")), norm_model(a.get("model"))): a for a in aggs}

    summary_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for spath in manifest.get("summary_files", []):
        p = Path(spath)
        if not p.exists():
            continue
        try:
            obj = json.loads(p.read_text())
        except Exception:
            continue
        key = (norm_dataset(obj.get("dataset")), norm_model(obj.get("model")))
        summary_by_key.setdefault(key, []).append(obj)

    updated: list[dict[str, Any]] = []
    matched = 0
    unmatched: list[dict[str, Any]] = []

    for row in rows:
        rr = dict(row)
        key = (norm_dataset(rr.get("dataset")), norm_model(rr.get("model")))
        agg = by_key.get(key)
        if agg is None:
            unmatched.append({"dataset": rr.get("dataset"), "model": rr.get("model"), "normalized_key": list(key)})
            updated.append(rr)
            continue

        matched += 1
        summaries = summary_by_key.get(key, [])

        accept = finite(mean_stat(agg.get("R1_accept_rate")), None)
        block = finite(mean_stat(agg.get("R1_block_rate")), 0.0) or 0.0
        verify = finite(mean_stat(agg.get("R1_verify_rate")), 0.0) or 0.0
        false_accept = finite(mean_stat(agg.get("R1_false_accept_proxy_rate_among_known")), None)

        r2_accept_n = finite(mean_stat(agg.get("R2_quorum_accept_n")), None)
        r2_block_n = finite(mean_stat(agg.get("R2_quorum_block_n")), None)
        r2_verify_n = finite(mean_stat(agg.get("R2_quorum_verify_n")), None)
        r3_flips = finite(mean_stat(agg.get("R3_total_decision_flips")), None)
        r3_flips_mean = finite(mean_stat(agg.get("R3_mean_decision_flips_per_record")), None)
        latency = finite(mean_stat(agg.get("R5_latency_mean_s")), None)
        latency_total = finite(mean_stat(agg.get("R5_latency_total_s")), None)
        tokens_abs = finite(mean_stat(agg.get("R5_tokens_est_total")), None)

        ns = [finite(x.get("n"), None) for x in summaries]
        ns = [x for x in ns if x is not None]
        n_examples = max(ns) if ns else finite(rr.get("n_examples"), 3.0) or 3.0

        if false_accept is not None:
            harm = false_accept
            harm_source = "observed_false_accept_proxy"
            harm_observed = True
        else:
            harm = max(0.0, min(1.0, block + 0.5 * verify))
            harm_source = "risk_proxy_from_block_plus_verify_rate_not_true_false_accept"
            harm_observed = False

        nocert_tok = finite(rr.get("tokens_nocert"), None) or finite(rr.get("token_nocert"), None) or 1.0
        token_multiplier = max(1.0, tokens_abs / max(n_examples, 1.0) / max(nocert_tok, 1.0)) if tokens_abs is not None else None

        rr.update({
            "shieldagent_overlay_applied": True,
            "shieldagent_artifact_source": "results/baselines/shieldagent/r1_r5",
            "shieldagent_implementation_mode": "official_authors_pipeline",
            "shieldagent_accept_rate": accept,
            "shieldagent_block_rate": block,
            "shieldagent_verify_rate": verify,
            "shieldagent_false_accept_proxy_rate": false_accept,
            "shieldagent_harm_observed": harm_observed,
            "shieldagent_harm_field_source": harm_source,
            "clean_harm_shieldagent": harm,
            "adv_harm_shieldagent": harm,
            "harm_clean_shieldagent": harm,
            "harm_adv_shieldagent": harm,
            "harm_clean_shield": harm,
            "harm_adv_shield": harm,
            "bound_gap_shieldagent": max(0.0, 1.0 - (accept if accept is not None else 0.0)),
            "shieldagent_bound_gap": max(0.0, 1.0 - (accept if accept is not None else 0.0)),
            "R1_accept_rate_shieldagent": accept,
            "R1_block_rate_shieldagent": block,
            "R1_verify_rate_shieldagent": verify,
            "shieldagent_r2_quorum_accept_n": r2_accept_n,
            "shieldagent_r2_quorum_block_n": r2_block_n,
            "shieldagent_r2_quorum_verify_n": r2_verify_n,
            "shieldagent_r2_quorum_accept_rate": safe_rate(r2_accept_n or 0.0, n_examples),
            "shieldagent_r2_quorum_block_rate": safe_rate(r2_block_n or 0.0, n_examples),
            "shieldagent_r2_quorum_verify_rate": safe_rate(r2_verify_n or 0.0, n_examples),
            "shieldagent_r3_total_decision_flips": r3_flips,
            "shieldagent_r3_mean_decision_flips_per_record": r3_flips_mean,
            "shieldagent_r3_decision_flip_rate": safe_rate(r3_flips or 0.0, max(n_examples * 3.0, 1.0)),
            "latency_shieldagent": latency,
            "latency_shield": latency,
            "latency_total_shieldagent": latency_total,
            "tokens_shieldagent": token_multiplier,
            "token_shieldagent": token_multiplier,
            "token_shield": token_multiplier,
            "tokens_shieldagent_abs_total": tokens_abs,
            "R5_tokens_est_total_shieldagent": tokens_abs,
            "utility_shieldagent": accept,
        })
        updated.append(rr)

    write_current_overlay_sidecar(updated)
    updated = rehydrate_overlay_sidecars(updated)
    ROWS_PATH.write_text(
        "\n".join(json.dumps(x, sort_keys=True) for x in updated) + "\n",
        encoding="utf-8",
    )
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    audit = {
        "matched_rows": matched,
        "unmatched_rows": unmatched,
        "shield_keys": [list(k) for k in sorted(by_key)],
        "paper_metric_rows": len(rows),
        "hardened_mapping": True,
        "overlay_sidecar": str(CURRENT_OVERLAY_SIDECAR),
        "rehydrated_known_sidecars": [str(p) for p in KNOWN_OVERLAY_SIDECARS.values() if p.exists()],
        "output": str(ROWS_PATH),
    }
    AUDIT_PATH.write_text(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps(audit, indent=2, sort_keys=True))

    if matched == 0:
        raise SystemExit("No ShieldAgent rows matched paper_metrics.jsonl.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
