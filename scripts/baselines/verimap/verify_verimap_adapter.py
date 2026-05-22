#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path.cwd()
AGG_PATH = ROOT / "results/baselines/verimap/r1_r5/aggregate_by_dataset_model.json"
MANIFEST_PATH = ROOT / "results/baselines/verimap/r1_r5/manifest.json"
ROWS_PATH = ROOT / "results/tables/csv/paper_metrics.jsonl"

EXPECTED_STRESS = {
    "clean_plan",
    "drop_support_step",
    "contradict_verification_criterion",
    "insert_distractor_step",
    "shuffle_plan_context",
    "answer_evidence_mismatch",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def mean_stat(obj: Any) -> Any:
    return obj.get("mean") if isinstance(obj, dict) else None


def fail(msg: str) -> None:
    raise SystemExit(f"VERIMAP_VERIFY_FAILED: {msg}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-overlay-check", action="store_true")
    args = ap.parse_args()

    if not AGG_PATH.exists():
        fail(f"missing aggregate: {AGG_PATH}")
    if not MANIFEST_PATH.exists():
        fail(f"missing manifest: {MANIFEST_PATH}")

    manifest = json.loads(MANIFEST_PATH.read_text())
    aggs = json.loads(AGG_PATH.read_text())

    summary_files = [Path(p) for p in manifest.get("summary_files", [])]
    if not summary_files:
        fail("manifest has no summary_files")

    missing_files = []
    required_leafs = [
        "input.jsonl",
        "verimap_verification.jsonl",
        "verimap_hero_metrics.json",
        "r5_overhead.json",
        "adapter_manifest.json",
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
            "harm_under_corruption_mean",
            "audit_coverage_mean",
            "stress_suite_used",
        ]:
            if field not in agg:
                fail(f"aggregate missing field {field} for {agg.get('dataset')}:{agg.get('model')}")

        if mean_stat(agg.get("R5_tokens_est_total")) is None:
            fail(f"null R5_tokens_est_total for {agg.get('dataset')}:{agg.get('model')}")

        stress = set(agg.get("stress_suite_used") or [])
        missing_stress = EXPECTED_STRESS - stress
        if missing_stress:
            fail(f"missing stress types for aggregate {agg.get('dataset')}:{agg.get('model')}: {sorted(missing_stress)}")

    overlay_rows = []
    if not args.skip_overlay_check:
        if not ROWS_PATH.exists():
            fail(f"missing paper_metrics: {ROWS_PATH}")
        rows = load_jsonl(ROWS_PATH)
        overlay_rows = [r for r in rows if r.get("verimap_overlay_applied")]
        if not overlay_rows:
            fail("paper_metrics has no VeriMAP overlay rows")

        for r in overlay_rows:
            for field in [
                "verimap_backend_mode",
                "verimap_harm_under_corruption_mean",
                "verimap_harm_under_corruption_max",
                "verimap_audit_coverage",
                "tokens_verimap",
                "verimap_stress_suite_used",
            ]:
                if r.get(field) is None:
                    fail(f"paper_metrics missing {field} for {r.get('dataset')}:{r.get('model')}")

            try:
                stress = set(json.loads(r["verimap_stress_suite_used"]))
            except Exception as exc:
                fail(f"bad stress JSON for {r.get('dataset')}:{r.get('model')}: {exc}")

            missing_stress = EXPECTED_STRESS - stress
            if missing_stress:
                fail(f"missing stress types for {r.get('dataset')}:{r.get('model')}: {sorted(missing_stress)}")

    print(
        json.dumps(
            {
                "status": "VERIMAP_ADAPTER_VERIFY_PASSED",
                "aggregate_rows": len(aggs),
                "overlay_rows": len(overlay_rows),
                "summary_files": len(summary_files),
                "expected_stress": sorted(EXPECTED_STRESS),
                "skip_overlay_check": args.skip_overlay_check,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
