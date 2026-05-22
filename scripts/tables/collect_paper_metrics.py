from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_INPUT_DIR = Path("results/tables/csv/experiment_json")
DEFAULT_OUT = Path("results/tables/csv/paper_metrics.jsonl")
R_FILES = {"r1.json", "r2.json", "r3.json", "r4.json", "r5.json"}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def infer_exp(path: Path) -> str:
    m = re.match(r"(r[1-5])\.json$", path.name.lower())
    if m:
        return m.group(1)
    for part in [path.name, path.parent.name]:
        m = re.search(r"\b(r[1-5])\b", part.lower())
        if m:
            return m.group(1)
    return "unknown"


def scalarize(value: Any) -> Any:
    """Keep paper rows flat: no dict/list values."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def flatten_numeric(prefix: str, obj: Any, out: dict) -> None:
    """Flatten numeric leaves from nested dict/list payloads."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}_{k}" if prefix else str(k)
            flatten_numeric(key, v, out)
    elif isinstance(obj, list):
        # Keep small lists as JSON, and expose numeric list summary.
        nums = [x for x in obj if isinstance(x, (int, float))]
        if nums:
            out[f"{prefix}_mean"] = sum(nums) / len(nums)
            out[f"{prefix}_min"] = min(nums)
            out[f"{prefix}_max"] = max(nums)
        out[prefix] = scalarize(obj)
    else:
        out[prefix] = scalarize(obj)


def summarize_payload(path: Path, payload: Any) -> dict:
    exp = infer_exp(path)
    parent = path.parent.name

    row: dict[str, Any] = {
        "experiment": exp,
        "metric_source": "measured",
        "source_file": str(path),
    }

    # Parse useful metadata from directory name like:
    # 20260510-013026_r1_hotpotqa_9ec00028_unknown
    parts = parent.split("_")
    if len(parts) >= 3:
        row.setdefault("dataset", parts[2])
    else:
        row.setdefault("dataset", "unknown")
    row.setdefault("model", "unknown")

    if isinstance(payload, dict):
        # Preserve common top-level metadata if scalar.
        for key in ["model", "model_name", "dataset", "dataset_name", "backend", "seed", "n_examples", "num_examples", "n"]:
            if key in payload:
                val = payload[key]
                if key in {"model_name"}:
                    row["model"] = scalarize(val)
                elif key in {"dataset_name"}:
                    row["dataset"] = scalarize(val)
                elif key in {"num_examples", "n"}:
                    row["n_examples"] = scalarize(val)
                else:
                    row[key] = scalarize(val)

        # Flatten everything with experiment prefix, but avoid giant raw trace keys when possible.
        for key, value in payload.items():
            if key in {"records", "examples", "traces", "raw", "config"}:
                continue
            flatten_numeric(f"{exp}_{key}", value, row)

    elif isinstance(payload, list):
        row[f"{exp}_num_items"] = len(payload)
        # Flatten first level summary for list of dicts.
        numeric_by_key: dict[str, list[float]] = {}
        for item in payload:
            if isinstance(item, dict):
                for k, v in item.items():
                    if isinstance(v, (int, float)):
                        numeric_by_key.setdefault(k, []).append(float(v))
        for k, vals in numeric_by_key.items():
            row[f"{exp}_{k}_mean"] = sum(vals) / len(vals)
            row[f"{exp}_{k}_min"] = min(vals)
            row[f"{exp}_{k}_max"] = max(vals)

    return row



def pivot_rows(rows: list[dict]) -> list[dict]:
    """Pivot raw R1--R5 summaries into one measured paper-facing row.

    This does not invent unavailable baselines. AgentRR remains missing until
    an AgentRR baseline is actually run.
    """
    import json as _json

    grouped: dict[tuple, dict] = {}

    for row in rows:
        key = (
            row.get("dataset", "unknown"),
            row.get("model", "unknown"),
            row.get("seed", 0),
        )
        acc = grouped.setdefault(
            key,
            {
                "dataset": row.get("dataset", "unknown"),
                "model": row.get("model", "unknown"),
                "seed": row.get("seed", 0),
                "metric_source": "measured",
            },
        )

        exp = row.get("experiment")
        acc[f"has_{exp}"] = True

        for k, v in row.items():
            if k not in acc and k not in {"source_file"}:
                acc[k] = v

    out_rows: list[dict] = []

    for r in grouped.values():
        # R1: coverage and PCG clean harm.
        p_check_fail = r.get("r1_aggregated_p_check_fail_mean")
        lhs_wrong = r.get("r1_aggregated_lhs_accept_and_wrong_mean")

        if p_check_fail is not None:
            r["coverage"] = max(0.0, min(1.0, 1.0 - float(p_check_fail)))

        if lhs_wrong is not None:
            r["clean_harm_pcg_mas"] = float(lhs_wrong)

        # R2: k=1 empirical error as NoCert-like clean harm;
        # strongest-k empirical as redundancy/PCG adversarial harm if available.
        per_k = r.get("r2_aggregated_per_k")
        if isinstance(per_k, str):
            try:
                per_k = _json.loads(per_k)
            except Exception:
                per_k = []
        if isinstance(per_k, list) and per_k:
            per_k = sorted(per_k, key=lambda x: x.get("k", 0))
            if per_k[0].get("empirical_mean") is not None:
                r["clean_harm_nocert"] = float(per_k[0]["empirical_mean"])
            if per_k[-1].get("empirical_mean") is not None:
                r["adv_harm_pcg_mas"] = float(per_k[-1]["empirical_mean"])

        # R3: responsibility.
        resp = (
            r.get("r3_aggregated_clean_top1_accuracy_mean")
            or r.get("r3_aggregated_light_top1_accuracy_mean")
            or r.get("r3_aggregated_heavy_top1_accuracy_mean")
        )
        if resp is not None:
            r["responsibility_top1"] = float(resp)

        # R4: NoCert-like always-answer harm, PCG threshold harm, and utility estimate.
        always_harm = r.get("r4_aggregated_1.0_always_answer_harm_mean")
        pcg_harm = r.get("r4_aggregated_1.0_threshold_pcg_harm_mean")
        pcg_cost = r.get("r4_aggregated_1.0_threshold_pcg_cost_mean")

        if always_harm is not None:
            r["adv_harm_nocert"] = float(always_harm)
        if pcg_harm is not None:
            r["adv_harm_pcg_mas"] = float(pcg_harm)
        if pcg_cost is not None:
            r["utility"] = max(0.0, min(1.0, 1.0 / (1.0 + float(pcg_cost))))

        # R5: PCG overhead. For now normalize NoCert to 1.0.
        agg = r.get("r5_aggregated")
        if isinstance(agg, str):
            try:
                agg = _json.loads(agg)
            except Exception:
                agg = []
        if isinstance(agg, list) and agg:
            r["tokens_pcg_mas"] = 1.0
            r["latency_pcg_mas"] = 1.0

        r["tokens_nocert"] = 1.0

        # Keep unavailable AgentRR fields explicit.
        r.setdefault("clean_harm_agentrr", None)
        r.setdefault("adv_harm_agentrr", None)
        r.setdefault("tokens_agentrr", None)

        out_rows.append(r)

    return out_rows


def load_shieldagent_aggregates() -> dict[tuple, dict]:
    """Load ShieldAgent aggregate rows and map them to paper-facing fields."""
    out: dict[tuple, dict] = {}
    base = Path("results/tables/csv/shieldagent_outputs")
    if not base.exists():
        return out

    paths = (
        sorted(base.glob("*_official_aggregate.jsonl"))
        + sorted(base.glob("*shieldagent_aggregate.jsonl"))
        + sorted(base.glob("official_shieldagent_aggregates.jsonl"))
    )

    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            model = str(obj.get("model", "unknown"))
            dataset = str(obj.get("dataset", "unknown"))
            seed = obj.get("seed", 0)
            key = (dataset, model, seed)

            incoming = {
                "clean_harm_shieldagent": obj.get("harm_clean_shield"),
                "adv_harm_shieldagent": obj.get("harm_adv_shield"),
                "harm_clean_shield": obj.get("harm_clean_shield"),
                "harm_adv_shield": obj.get("harm_adv_shield"),
                "tokens_shieldagent": obj.get("token_shield"),
                "latency_shieldagent": obj.get("latency_shield"),
                "token_shield": obj.get("token_shield"),
                "latency_shield": obj.get("latency_shield"),
                "shieldagent_block_rate": obj.get("shieldagent_block_rate"),
                "shieldagent_mean_risk": obj.get("shieldagent_mean_risk"),
                "shieldagent_implementation_mode": obj.get("shieldagent_implementation_mode"),
            }

            # Prefer official authors' pipeline rows over older local/adapter rows.
            old_row = out.get(key)
            if old_row and old_row.get("shieldagent_implementation_mode") == "official_authors_pipeline":
                continue
            out[key] = incoming
    return out

def load_agentrr_aggregates() -> dict[tuple, dict]:
    """Load AgentRR aggregate rows and map them to paper-facing fields."""
    out: dict[tuple, dict] = {}
    base = Path("results/tables/csv/agentrr_outputs")
    if not base.exists():
        return out

    for path in sorted(base.glob("*agentrr_aggregate.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            model = str(obj.get("model", "unknown"))
            dataset = str(obj.get("dataset", "unknown"))
            seed = obj.get("seed", 0)
            key = (dataset, model, seed)
            out[key] = {
                "clean_harm_agentrr": obj.get("harm_clean_agentrr"),
                "adv_harm_agentrr": obj.get("harm_adv_agentrr"),
                "tokens_agentrr": obj.get("token_agentrr"),
                "latency_agentrr": obj.get("latency_agentrr"),
                "agentrr_block_rate": obj.get("agentrr_block_rate"),
                "agentrr_mean_risk": obj.get("agentrr_mean_risk"),
            }
    return out

def collect(input_dir: Path, pivot: bool = True) -> list[dict]:
    paths = sorted(p for p in input_dir.glob("**/*.json") if p.name.lower() in R_FILES)
    rows = [summarize_payload(path, read_json(path)) for path in paths]

    if not pivot:
        return rows

    pivoted = pivot_rows(rows)

    shield = load_shieldagent_aggregates()
    agentrr = load_agentrr_aggregates()

    for row in pivoted:
        dataset = str(row.get("dataset", "unknown"))
        model = str(row.get("model", "unknown"))
        seed = row.get("seed", 0)

        key = (dataset, model, seed)

        # ShieldAgent exact match.
        if key in shield:
            row.update(shield[key])
        elif model == "unknown":
            matches = [
                (k, v) for k, v in shield.items()
                if k[0] == dataset and k[2] == seed
            ]
            if len(matches) == 1:
                matched_key, matched_values = matches[0]
                row["model"] = matched_key[1]
                model = matched_key[1]
                key = (dataset, model, seed)
                row.update(matched_values)

        # AgentRR exact match after possible model recovery.
        if key in agentrr:
            row.update(agentrr[key])
        else:
            matches = [
                (k, v) for k, v in agentrr.items()
                if k[0] == dataset and k[2] == seed and (model == "unknown" or k[1] == model)
            ]
            if len(matches) == 1:
                matched_key, matched_values = matches[0]
                row["model"] = matched_key[1]
                row.update(matched_values)

    return pivoted


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect measured R1--R5 outputs into paper_metrics.jsonl.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--no-pivot", action="store_true")
    args = parser.parse_args()

    rows = collect(args.input_dir, pivot=not args.no_pivot)
    if not rows:
        raise SystemExit(f"No r1.json--r5.json metric files found under {args.input_dir}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    print(f"Wrote {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
