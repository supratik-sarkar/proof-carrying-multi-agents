#!/usr/bin/env python3
"""Build all v4 paper-facing artifacts from available results.

Path-based execution; no package install required.
"""

from __future__ import annotations

import os
import subprocess
import sys


ENV = dict(os.environ)
ENV["PYTHONPATH"] = "src" + os.pathsep + ENV.get("PYTHONPATH", "")

COMMANDS = [
    [sys.executable, "scripts/v4_build_intro_metrics.py"],
    [
        sys.executable,
        "src/pcg/eval/intro_hero_v4.py",
        "--metrics",
        "results/v4/intro_hero_metrics.json",
        "--out",
        "figures/intro_hero_v4",
    ],
    [sys.executable, "scripts/v4_collect_artifacts.py"],
    [sys.executable, "scripts/v4_make_summary_figure.py"],
]


def main() -> None:
    for cmd in COMMANDS:
        print("\n>>>", " ".join(cmd))
        subprocess.run(cmd, check=True, env=ENV)


if __name__ == "__main__":
    main()