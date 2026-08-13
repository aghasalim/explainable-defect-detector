#!/usr/bin/env bash
# Deploy the demo to Hugging Face Spaces.
#
#   pip install -U huggingface_hub      # provides the `hf` CLI
#   hf auth login                       # paste a WRITE token from
#                                       # https://huggingface.co/settings/tokens
#   ./scripts/deploy_space.sh <your-hf-username> [space-name]
#
# Uses `hf upload` rather than git push: it handles large files itself, so no
# git-lfs install and no git credential helper are needed. The Space config
# (YAML front matter selecting the Docker SDK and port 7860) lives in
# spaces/README.md and is uploaded as the Space's README.md.
set -euo pipefail

USER="${1:?usage: $0 <hf-username> [space-name]}"
SPACE="${2:-explainable-defect-detector}"
REPO="$USER/$SPACE"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

command -v hf >/dev/null || { echo "hf CLI not found: pip install -U huggingface_hub"; exit 1; }
hf auth whoami >/dev/null 2>&1 || { echo "not logged in: run 'hf auth login' with a WRITE token"; exit 1; }

echo "==> creating space $REPO (ok if it already exists)"
hf repo create "$REPO" --type space --sdk docker --exist-ok

echo "==> staging files"
mkdir -p "$TMP/space/src" "$TMP/space/assets" "$TMP/space/models"
cp -r "$ROOT/src/edd"            "$TMP/space/src/"
cp -r "$ROOT/assets/samples"     "$TMP/space/assets/"
cp "$ROOT"/models/*.pt "$ROOT"/models/*.json "$TMP/space/models/"
cp "$ROOT/app.py" "$ROOT/Dockerfile" "$ROOT/requirements.txt" "$TMP/space/"
cp "$ROOT/spaces/README.md"      "$TMP/space/README.md"
du -sh "$TMP/space"

echo "==> uploading"
hf upload "$REPO" "$TMP/space" . --type space \
  --commit-message "Deploy explainable defect detector"

echo
echo "Live at: https://huggingface.co/spaces/$REPO"
echo "First build takes ~5-10 min (it bakes the backbone weights into the image)."
