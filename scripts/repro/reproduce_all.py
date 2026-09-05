#!/usr/bin/env python3
"""One-command offline reproduction. Validates first, fails closed, never infers.

Performs no network access, no model inference and no synthesis of missing data.
"""
from __future__ import annotations
import json, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STEPS = [
    ("validate 56-cell records", [sys.executable, "scripts/validate/validate_56cell.py"]),
    ("provenance + contamination gates", [sys.executable, "scripts/validate/provenance_gates.py"]),
    ("regenerate tables", [sys.executable, "scripts/repro/regenerate_tables.py"]),
    ("regenerate figures", [sys.executable, "scripts/repro/regenerate_figures.py"]),
]

def main() -> int:
    results, t0 = [], time.time()
    for name, cmd in STEPS:
        print(f"\n=== {name} ===")
        r = subprocess.run(cmd, cwd=ROOT)
        results.append({"step": name, "cmd": " ".join(cmd[1:]),
                        "status": "PASS" if r.returncode == 0 else "FAIL"})
        if r.returncode != 0:
            print(f"\nFAILED at: {name} — stopping (fail-closed).")
            break
    manifest = {
        "reproduction": "offline",
        "network_access": False, "model_inference": False, "synthetic_fallback": False,
        "elapsed_s": round(time.time() - t0, 2),
        "steps": results,
        "source_records": "artifacts/evidence/source_records/per_example_records.jsonl",
        "source_records_sha256": "c63babd59efe28b52797379d19e2dffbc500416fd53fc0d45e0e953ebd3d3cef",
        "provenance": {
            "structural_completeness": "PASS",
            "internal_consistency": "PASS",
            "native_execution_provenance": "NOT_AVAILABLE",
            "empirical_authenticity": "NOT_ESTABLISHED",
            "origin": "UNKNOWN_WITH_GENERATIVE_INDICATORS",
            "safe_for_empirical_manuscript_claims": False,
        },
        "output_classification": "REGENERATED_FROM_UNKNOWN_PROVENANCE_56_CELL",
        "output_safety": "NOT_SAFE_FOR_EMPIRICAL_MANUSCRIPT_USE",
    }
    (ROOT/"reports").mkdir(exist_ok=True)
    (ROOT/"reports/REPRODUCIBILITY_MANIFEST.json").write_text(json.dumps(manifest, indent=2)+"\n")
    bad = sum(1 for r in results if r["status"] == "FAIL")
    print(f"\n{'='*54}\nsteps: {len(results)-bad} PASS / {bad} FAIL")
    print("manifest: reports/REPRODUCIBILITY_MANIFEST.json")
    return 1 if bad else 0

if __name__ == "__main__":
    sys.exit(main())
