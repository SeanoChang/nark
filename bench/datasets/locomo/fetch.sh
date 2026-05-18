#!/usr/bin/env bash
# Clones snap-research/locomo at a pinned commit SHA into upstream/repo.
# Idempotent: if upstream/repo/.git exists, prints "already cloned" and exits.
#
# Unlike LongMemEval, the LOCOMO data file (data/locomo10.json) IS included
# directly in the upstream git repo — no separate HuggingFace download needed.
#
# Pinned SHA recorded from:
#   git ls-remote https://github.com/snap-research/locomo HEAD

set -euo pipefail

UPSTREAM_URL="https://github.com/snap-research/locomo"
# SHA from: git ls-remote https://github.com/snap-research/locomo HEAD
PINNED_SHA="3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="$SCRIPT_DIR/upstream"
REPO="$TARGET/repo"

if [[ -d "$REPO/.git" ]]; then
  echo "LOCOMO already cloned at $REPO"
  exit 0
fi

mkdir -p "$TARGET"
git clone "$UPSTREAM_URL" "$REPO"
cd "$REPO"
git fetch origin "$PINNED_SHA"
git checkout "$PINNED_SHA"

echo ""
echo "LOCOMO ready at $REPO (pinned $PINNED_SHA)"
echo "  Data file: $REPO/data/locomo10.json"
echo "  Samples:   10 conversations, 1986 total QA pairs"
