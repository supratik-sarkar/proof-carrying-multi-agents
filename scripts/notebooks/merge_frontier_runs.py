from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def list_csvs(path: Path) -> list[str]:
    if not path.exists():
        return []
    return sorted(str(p) for p in path.rglob("*.csv"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge Colab/Databricks frontier-cell outputs with local outputs."
    )
    parser.add_argument("--local", type=Path, default=Path("results/tables/csv"))
    parser.add_argument("--colab", type=Path, default=Path("notebooks/colab/outputs"))
    parser.add_argument("--databricks", type=Path, default=Path("notebooks/databricks/outputs"))
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/frontier_merge_manifest.json"))
    parser.add_argument("--allow-fallback", action="store_true")
    args = parser.parse_args()

    local_csvs = list_csvs(args.local)
    colab_csvs = list_csvs(args.colab)
    databricks_csvs = list_csvs(args.databricks)

    frontier_available = bool(colab_csvs or databricks_csvs)

    if not frontier_available and not args.allow_fallback:
        raise SystemExit(
            "No Colab/Databricks frontier outputs found. "
            "Use --allow-fallback to generate figures/tables from local outputs only."
        )

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "frontier_available": frontier_available,
        "fallback_used": not frontier_available,
        "local_csv_count": len(local_csvs),
        "colab_csv_count": len(colab_csvs),
        "databricks_csv_count": len(databricks_csvs),
        "local_csvs": local_csvs,
        "colab_csvs": colab_csvs,
        "databricks_csvs": databricks_csvs,
    }

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
