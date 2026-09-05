#!/usr/bin/env bash
# Cloudflare Pages build. Regenerates the shared contract so the frontend cannot
# drift from the Python core, then publishes app/frontend as static assets.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here/.."
python3 shared/generate_contract.py
cp shared/contract.json frontend/contract.json
cat > frontend/config.js <<JS
window.PCG_API_BASE = "${PCG_API_BASE:-}";
JS
echo "cloudflare build complete; output dir = app/frontend"
