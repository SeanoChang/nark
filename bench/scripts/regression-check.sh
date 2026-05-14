#!/usr/bin/env bash
# regression-check.sh — fails (exits 1) if any IR metric in <new-results-dir>
# drops by more than 2% compared to bench/results/main/<same-filename>.
#
# Usage: bench/scripts/regression-check.sh <new-results-dir>
#
# On first run (no baseline present) it bootstraps the baseline by copying
# the new results to bench/results/main/ and exits 0.

set -euo pipefail

NEW_DIR="${1:-}"
if [[ -z "$NEW_DIR" ]]; then
  echo "usage: $0 <new-results-dir>" >&2
  exit 2
fi

BASELINE_DIR="bench/results/main"
THRESHOLD=0.02   # 2% relative drop

fail=0
bootstrap=0

shopt -s nullglob
for new_file in "$NEW_DIR"/ir-*.json; do
  base_file="$BASELINE_DIR/$(basename "$new_file")"
  if [[ ! -f "$base_file" ]]; then
    echo "no baseline for $(basename "$new_file") — copying as bootstrap"
    cp "$new_file" "$base_file"
    bootstrap=1
    continue
  fi

  for metric in recall_at_5 mrr ndcg_at_10; do
    new_val=$(jq -r ".ir.${metric}" "$new_file")
    base_val=$(jq -r ".ir.${metric}" "$base_file")
    drop=$(awk -v n="$new_val" -v b="$base_val" 'BEGIN { if (b == 0) print 0; else print (b - n) / b }')
    is_regress=$(awk -v d="$drop" -v t="$THRESHOLD" 'BEGIN { print (d > t) ? 1 : 0 }')
    if [[ "$is_regress" == "1" ]]; then
      printf 'REGRESSION: %s %s dropped from %s to %s (%.2f%%)\n' \
        "$(basename "$new_file")" "$metric" "$base_val" "$new_val" \
        "$(awk -v d="$drop" 'BEGIN { print d * 100 }')"
      fail=1
    fi
  done
done

if [[ "$bootstrap" == "1" ]]; then
  echo "regression-check: bootstrap mode — baseline copied; commit bench/results/main/ to lock it in"
fi

exit $fail
