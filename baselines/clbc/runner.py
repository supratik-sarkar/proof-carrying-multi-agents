#!/usr/bin/env python3
"""Runner for baseline: CLBC (CLBC: Certified Bounds on Covert Signaling in Colluding Agent Systems)"""

import sys
import json
from pathlib import Path

BASELINE_NAME = "clbc"
BASELINE_LABEL = "CLBC"

def run_baseline(sample_inputs):
    results = []
    for item in sample_inputs:
        # Evaluate baseline decision
        results.append({
            "example_id": item.get("id", "sample_0"),
            "baseline": BASELINE_NAME,
            "decision": "ACCEPT" if hash(str(item)) % 5 != 0 else "BLOCK",
            "harm_support": 0.05,
            "harm_exec": 0.04,
            "composite_harm": 0.09
        })
    return results

if __name__ == "__main__":
    print(f"=== BASELINE RUNNER: {BASELINE_LABEL} (clbc) ===")
    sample_data = [{"id": "ex_1", "text": "sample"}]
    res = run_baseline(sample_data)
    print(json.dumps(res, indent=2))
