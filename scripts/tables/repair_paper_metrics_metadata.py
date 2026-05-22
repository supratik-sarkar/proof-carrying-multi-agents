import json
from pathlib import Path

ROWS = Path("results/tables/csv/paper_metrics.jsonl")
EXP_ROOT = Path("results/tables/csv/experiment_json")
POLICY_PATH = Path("results/audit/pcgmas_selected_cells.json")
OUT = ROWS

SMOKE_DEFAULTS = {
    ("fever", "r1"): "phi-3.5-mini",
    ("tatqa", "r1"): "phi-3.5-mini",
}


def norm(x):
    return str(x or "").strip().lower()


def norm_dataset(x):
    s = str(x or "").strip().lower().replace("_", "-")
    return {
        "tat-qa": "tatqa",
        "tatqa": "tatqa",
        "hotpot": "hotpotqa",
        "hotpotqa": "hotpotqa",
        "fever": "fever",
        "pubmedqa": "pubmedqa",
        "toolbench": "toolbench",
        "weblinx": "weblinx",
        "2wiki": "2wikimultihopqa",
        "twowiki": "2wikimultihopqa",
        "2wikimultihopqa": "2wikimultihopqa",
    }.get(s, s)


def load_json(p):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def load_selected_policy():
    if not POLICY_PATH.exists():
        return None
    obj = load_json(POLICY_PATH)
    return obj if isinstance(obj, dict) else None


def candidate_model_from_selected_policy(dataset):
    policy = load_selected_policy()
    if not policy:
        return None

    dataset = norm_dataset(dataset)
    cells = policy.get("cells", [])
    matches = [
        c for c in cells
        if norm_dataset(c.get("dataset")) == dataset
    ]

    if len(matches) == 1:
        return matches[0].get("model")

    return None


def candidate_seed_from_selected_policy():
    policy = load_selected_policy()
    if not policy:
        return None

    seeds = str(policy.get("seeds") or "").strip()
    if seeds and "," not in seeds:
        try:
            return int(seeds)
        except Exception:
            return None

    return None


def candidate_model_from_config(dataset, experiment):
    dataset = norm_dataset(dataset)
    experiment = norm(experiment)

    candidates = []
    for cfg in sorted(EXP_ROOT.glob("*/config_snapshot.json")):
        parent = cfg.parent.name.lower()
        if dataset and dataset not in parent:
            continue
        if experiment and experiment not in parent:
            continue

        obj = load_json(cfg)
        if not isinstance(obj, dict):
            continue

        for key in ["model", "model_name", "model_id"]:
            val = obj.get(key)
            if isinstance(val, str) and val and val.lower() != "unknown":
                candidates.append(val)

        val = obj.get("models")
        if isinstance(val, list) and val:
            candidates.append(str(val[0]))
        elif isinstance(val, str) and val:
            candidates.append(val)

    if candidates:
        return candidates[-1]

    return None


def repair_row(row):
    row = dict(row)

    dataset = row.get("dataset") or row.get("Dataset")
    experiment = row.get("experiment") or row.get("axis") or row.get("Experiment") or "r1"

    model = row.get("model") or row.get("Model")
    model_missing = norm(model) in {"", "unknown", "none", "null"}

    if model_missing:
        fixed = candidate_model_from_selected_policy(dataset)

        if fixed:
            row["model"] = fixed
            row["model_repaired_from"] = "selected_cell_policy"
        else:
            fixed = candidate_model_from_config(dataset, experiment)
            if not fixed:
                fixed = SMOKE_DEFAULTS.get((norm_dataset(dataset), norm(experiment)))

            if fixed:
                row["model"] = fixed
                row["model_repaired_from"] = "config_snapshot_or_smoke_default"

    seed = candidate_seed_from_selected_policy()
    if seed is not None:
        row["seed"] = seed
        row["seed_repaired_from"] = "selected_cell_policy"

    return row


if not ROWS.exists():
    raise SystemExit(f"Missing {ROWS}")

rows = []
for line in ROWS.read_text().splitlines():
    if line.strip():
        rows.append(repair_row(json.loads(line)))

OUT.write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n")

print(f"Repaired {len(rows)} rows in {OUT}")
for r in rows:
    print({
        "dataset": r.get("dataset"),
        "experiment": r.get("experiment") or r.get("axis"),
        "model": r.get("model"),
        "seed": r.get("seed"),
        "model_repair": r.get("model_repaired_from"),
        "seed_repair": r.get("seed_repaired_from"),
    })
