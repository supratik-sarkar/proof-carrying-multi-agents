#!/usr/bin/env bash
# Push the app/ directory to an anonymous Hugging Face Space.
#
# WHAT IT DOES
# 1. Creates a temp deployment directory
# 2. Copies app/ contents AND a minimal vendored copy of pcg/ into it
# 3. Strips author identifiers (name in commits, real-name in URLs)
# 4. Pushes to the configured anonymous HF Space
#
# WHAT IT DOES NOT DO
# - Create the HF Space itself. You must do that manually first:
#     1. Sign up for a fresh HF account using a pseudonymous email
#        (Proton / SimpleLogin / disposable). DO NOT use your real
#        institutional email — HF profiles show the email's domain.
#     2. Create a new Space at https://huggingface.co/new-space
#        SDK = Streamlit, Hardware = CPU basic (free)
#     3. Copy the Space's git URL (looks like
#        https://huggingface.co/spaces/<anon-username>/<space-name>)
#     4. Paste it as ANON_SPACE_URL below or pass via env var.
#
# USAGE
#     export ANON_SPACE_URL=https://huggingface.co/spaces/anonymous-pcg-mas/pcg-demo
#     export ANON_HF_TOKEN=hf_xxx   # write-scope token from the anon account
#     ./scripts/deploy_to_anonymous_space.sh
#
# The token is used ONLY for `git push` to the Space; it is not embedded
# in the deployed code.

set -euo pipefail

ANON_SPACE_URL="${ANON_SPACE_URL:-}"
ANON_HF_TOKEN="${ANON_HF_TOKEN:-}"

if [[ -z "$ANON_SPACE_URL" ]]; then
    echo "ERROR: set ANON_SPACE_URL=https://huggingface.co/spaces/<anon>/<demo>" >&2
    exit 1
fi
if [[ -z "$ANON_HF_TOKEN" ]]; then
    echo "ERROR: set ANON_HF_TOKEN=hf_xxx (write-scope token from anonymous account)" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP_DIR="$REPO_ROOT/app"
DEPLOY_DIR="$(mktemp -d)"
trap "rm -rf $DEPLOY_DIR" EXIT

echo "→ Staging at $DEPLOY_DIR"

# 1) Copy app/ into the deployment root (NOT into a subdir — HF Spaces
#    expects app.py at the root)
rsync -a \
    --exclude '__pycache__' --exclude '*.pyc' --exclude '.DS_Store' \
    "$APP_DIR/" "$DEPLOY_DIR/"

# 2) Vendor a minimal copy of pcg/ so the Space can `import pcg.X` at
#    runtime. We exclude heavy/optional bits to keep the Space small.
mkdir -p "$DEPLOY_DIR/pcg"
rsync -a \
    --exclude '__pycache__' --exclude '*.pyc' \
    --exclude 'datasets/' \
    "$REPO_ROOT/src/pcg/" "$DEPLOY_DIR/pcg/"

# 3) Anonymize: strip any author commits or local git history. The
#    Space gets a single fresh commit authored by "anonymous".
cd "$DEPLOY_DIR"
git init -q -b main
git config user.email "anonymous@example.com"
git config user.name  "anonymous"
git add -A
git commit -q -m "Anonymous deployment for double-blind review"

# 4) Add the Space remote with the token embedded for push only
SPACE_HOST="$(echo "$ANON_SPACE_URL" | sed -E 's|^https?://||' | cut -d/ -f1)"
SPACE_PATH="$(echo "$ANON_SPACE_URL" | sed -E 's|^https?://[^/]+/||')"
PUSH_URL="https://anonymous:${ANON_HF_TOKEN}@${SPACE_HOST}/${SPACE_PATH}"
git remote add space "$PUSH_URL"

# 5) Force-push (overwrites any prior content; safe because the Space
#    is dedicated to this deployment)
echo "→ Pushing to $ANON_SPACE_URL …"
git push -q --force space main

echo "✓ Deployed. The Space will rebuild automatically (~2-3 min)."
echo "  URL: $ANON_SPACE_URL"
