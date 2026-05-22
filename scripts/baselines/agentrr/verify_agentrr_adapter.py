#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path.cwd()
AGG_PATH = ROOT / "results/baselines/agentrr/r1_r5/aggregate_by_dataset_model.json"
MANIFEST_PATH = ROOT / "results/baselines/agentrr/r1_r5/manifest.json"
ROWS_PATH = ROOT / "results/tables/csv/paper_metrics.jsonl"

EXPECTED_CORRUPTIONS = {
    "clean_replay",
    "evidence_deletion",
    "contradiction_injection",
    "distractor_insertion",
    "evidence_shuffle",
    "answer_evidence_mismatch",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def mean_stat(obj: Any) -> Any:
    return obj.get("mean") if isinstance(obj, dict) else None


def fail(msg: str) -> None:
    raise SystemExit(f"AGENTRR_VERIFY_FAILED: {msg}")


def main() -> int:
    if not AGG_PATH.exists():
        fail(f"missing aggregate: {AGG_PATH}")
    if not MANIFEST_PATH.exists():
        fail(f"missing manifest: {MANIFEST_PATH}")
    if not ROWS_PATH.exists():
        fail(f"missing paper metrics: {ROWS_PATH}")

    manifest = json.loads(MANIFEST_PATH.read_text())
    aggs = json.loads(AGG_PATH.read_text())
    rows = load_jsonl(ROWS_PATH)

    summary_files = [Path(p) for p in manifest.get("summary_files", [])]
    if not summary_files:
        fail("manifest has no summary_files")

    missing_files = []
    required_leafs = [
        "input.jsonl",
        "agentrr_replay_check.jsonl",
        "agentrr_hero_metrics.json",
        "r5_overhead.json",
        "summary.json",
    ]
    for summary in summary_files:
        if not summary.exists():
            missing_files.append(str(summary))
            continue
        for leaf in required_leafs:
            p = summary.parent / leaf
            if not p.exists():
                missing_files.append(str(p))

    if missing_files:
        fail("missing files:\n" + "\n".join(missing_files))

    for agg in aggs:
        for field in [
            "R1_accept_rate",
            "R1_block_rate",
            "R5_tokens_est_total",
            "harm_under_corruption",
            "audit_coverage_of_observed_bad_accepts",
        ]:
            if field not in agg:
                fail(f"aggregate missing field {field} for {agg.get('dataset')}:{agg.get('model')}")

        if mean_stat(agg.get("R5_tokens_est_total")) is None:
            fail(f"null R5_tokens_est_total for {agg.get('dataset')}:{agg.get('model')}")

    overlay_rows = [r for r in rows if r.get("agentrr_overlay_applied")]
    if not overlay_rows:
        fail("paper_metrics has no AgentRR overlay rows")

    for r in overlay_rows:
        for field in [
            "agentrr_backend_mode",
            "agentrr_harm_under_corruption_mean",
            "agentrr_harm_under_corruption_max",
            "agentrr_audit_coverage",
            "tokens_agentrr",
            "agentrr_corruption_types_used",
        ]:
            if r.get(field) is None:
                fail(f"paper_metrics missing {field} for {r.get('dataset')}:{r.get('model')}")

        try:
            corruptions = set(json.loads(r["agentrr_corruption_types_used"]))
        except Exception as exc:
            fail(f"bad corruption JSON for {r.get('dataset')}:{r.get('model')}: {exc}")

        missing_corr = EXPECTED_CORRUPTIONS - corruptions
        if missing_corr:
            fail(
                f"missing corruption types for {r.get('dataset')}:{r.get('model')}: "
                + ", ".join(sorted(missing_corr))
            )

    print(
        json.dumps(
            {
                "status": "AGENTRR_ADAPTER_VERIFY_PASSED",
                "aggregate_rows": len(aggs),
                "overlay_rows": len(overlay_rows),
                "summary_files": len(summary_files),
                "expected_corruptions": sorted(EXPECTED_CORRUPTIONS),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
