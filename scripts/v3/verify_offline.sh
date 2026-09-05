#!/usr/bin/env bash
# Bounded offline verification for PCG-MAS v3.0.
# Finite, non-interactive, no model call, no network call, no long-lived process.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="src:${PYTHONPATH:-}"
export PCG_OFFLINE_ONLY=1
python3 scripts/v3/verify_offline.py "$@"
