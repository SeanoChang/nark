#!/usr/bin/env bash
# Downloads the LongMemEval dataset from HuggingFace at a pinned commit SHA.
# Idempotent: if upstream/ already has the data files, prints "already fetched" and exits.
#
# The dataset is NOT in the upstream git repo — it is distributed via HuggingFace:
#   https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
#
# The upstream git repo (xiaowu0162/LongMemEval) provides the eval scripts and
# custom history utilities. We clone it at a pinned SHA for reproducibility.
# The data files are fetched directly from HuggingFace resolve URLs.

set -euo pipefail

UPSTREAM_GIT_URL="https://github.com/xiaowu0162/LongMemEval"
# SHA recorded from: git ls-remote https://github.com/xiaowu0162/LongMemEval HEAD
PINNED_SHA="9e0b455f4ef0e2ab8f2e582289761153549043fc"

HF_BASE="https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="$SCRIPT_DIR/upstream"
REPO="$TARGET/repo"
DATA="$TARGET/data"

# Check if data files already exist
if [[ -f "$DATA/longmemeval_s_cleaned.json" && -f "$DATA/longmemeval_m_cleaned.json" && -f "$DATA/longmemeval_oracle.json" ]]; then
  echo "LongMemEval data already fetched at $DATA"
  exit 0
fi

mkdir -p "$DATA"

# Clone the upstream git repo (eval scripts, not data) at the pinned SHA
if [[ ! -d "$REPO/.git" ]]; then
  echo "Cloning upstream git repo at pinned SHA $PINNED_SHA..."
  git clone "$UPSTREAM_GIT_URL" "$REPO"
  cd "$REPO"
  git checkout "$PINNED_SHA"
  echo "Upstream repo ready at $REPO"
else
  echo "Upstream repo already cloned at $REPO"
fi

# Download dataset files from HuggingFace
echo "Downloading data files from HuggingFace..."
wget -q --show-progress \
  "$HF_BASE/longmemeval_s_cleaned.json" \
  -O "$DATA/longmemeval_s_cleaned.json"

wget -q --show-progress \
  "$HF_BASE/longmemeval_m_cleaned.json" \
  -O "$DATA/longmemeval_m_cleaned.json"

wget -q --show-progress \
  "$HF_BASE/longmemeval_oracle.json" \
  -O "$DATA/longmemeval_oracle.json"

echo ""
echo "LongMemEval ready at $TARGET"
echo "  Repo (eval scripts): $REPO  (pinned $PINNED_SHA)"
echo "  Data files:          $DATA"
echo "    longmemeval_s_cleaned.json  — 500 questions, short haystacks"
echo "    longmemeval_m_cleaned.json  — 500 questions, medium haystacks"
echo "    longmemeval_oracle.json     — 500 questions, oracle (answer sessions only)"
