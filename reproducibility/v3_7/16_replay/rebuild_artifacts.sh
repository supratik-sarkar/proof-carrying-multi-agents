#!/bin/sh
set -eu
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON=${PYTHON:-python3}
export PYTHONDONTWRITEBYTECODE=1
"$PYTHON" "$HERE/verify_record.py"
exec "$PYTHON" "$HERE/replay_artifacts.py"
