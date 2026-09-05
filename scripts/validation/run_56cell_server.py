#!/usr/bin/env python3
"""Safety interlock for the native 56-cell execution handoff.

This clean offline repository does not contain a native model-execution
implementation. macOS execution is explicitly prohibited. On non-macOS
systems this command remains fail-closed until a genuine native runner
with auditable request/response provenance is supplied.
"""

from __future__ import annotations

import argparse
import platform
import sys


def main() -> int:
    if platform.system() == "Darwin":
        print("BLOCKED: native 56-cell model execution is prohibited on macOS.")
        print("Use the frozen server handoff protocol on an approved execution environment.")
        return 1

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--output-root")
    parser.add_argument("--resume", action="store_true")
    parser.parse_args()

    print("BLOCKED: native 56-cell execution implementation is not present in this clean repository.")
    print("No API call, model inference, synthetic fallback, or placeholder generation was performed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
