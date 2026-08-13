#!/usr/bin/env bash
# Deploy the demo to Hugging Face Spaces.
#
# Requires a Hugging Face account and a WRITE token from
# https://huggingface.co/settings/tokens  (never commit it).
#
#   pip install -U "huggingface_hub[cli]"
#   hf auth login                      # paste your token when prompted
#   ./scripts/deploy_space.sh <your-hf-username>
#
# Everything the Space needs is already in this repo; this script only assembles
# it into the layout Spaces expects (its own README.md carries the YAML config
# block) and pushes.
set -euo pipefail

USER="${1:?usage: $0 <hf-username> [space-name]}"
SPACE="${2:-explainable-defect-detector}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

command -v hf >/dev/null || { echo "hf CLI not found: pip install -U 'huggingface_hub[cli]'"; exit 1; }
hf auth whoami >/dev/null 2>&1 || { echo "not logged in: run 'hf auth login'"; exit 1; }

hf repo create "$USER/$SPACE" --repo-type space --space_sdk docker -y 2>/dev/null || true
git clone "https://huggingface.co/spaces/$USER/$SPACE" "$TMP/space"

cd "$TMP/space"
mkdir -p src assets models
cp -r "$ROOT/src/edd" src/
cp -r "$ROOT/assets/samples" assets/
cp "$ROOT"/models/*.pt "$ROOT"/models/*.json models/
cp "$ROOT/app.py" "$ROOT/Dockerfile" "$ROOT/requirements.txt" .
cp "$ROOT/spaces/README.md" README.md

# model artefacts are multi-MB binaries -> LFS, or the push is rejected
git lfs install
git lfs track "*.pt"
git add -A
git -c user.email="deploy@local" -c user.name="deploy" \
    commit -m "Deploy explainable defect detector" || { echo "nothing to commit"; exit 0; }
git push

echo
echo "Live at: https://huggingface.co/spaces/$USER/$SPACE"
echo "First build takes ~5-10 min (it bakes the backbone weights into the image)."
