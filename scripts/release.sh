#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/release.sh [version]
# Example: ./scripts/release.sh 0.1.0
#
# Builds a release binary, creates a git tag, and publishes
# a GitHub release with the binary attached.

VERSION="${1:-}"
if [ -z "$VERSION" ]; then
  VERSION=$(grep '^version' Cargo.toml | head -1 | sed 's/.*"\(.*\)"/\1/')
  echo "No version arg — using Cargo.toml version: $VERSION"
fi

TAG="v${VERSION}"
BINARY="nark"
TARGET=$(rustc -vV | grep host | awk '{print $2}')
ASSET="${BINARY}-${TAG}-${TARGET}"

echo "Building ${ASSET}..."
cargo build --release

# Copy binary with platform-specific name
cp "target/release/${BINARY}" "target/release/${ASSET}"

echo "Tagging ${TAG}..."
git tag -a "$TAG" -m "Release ${TAG}"
git push origin "$TAG"

echo "Creating GitHub release..."
gh release create "$TAG" \
  "target/release/${ASSET}" \
  --title "${TAG}" \
  --notes "Release ${TAG} for ${TARGET}" \
  --latest

echo ""
echo "Done: https://github.com/SeanoChang/ironvault/releases/tag/${TAG}"
echo ""
echo "Install on any machine:"
echo "  gh release download ${TAG} --repo SeanoChang/ironvault --pattern '${ASSET}'"
echo "  chmod +x ${ASSET} && mv ${ASSET} ~/.local/bin/nark"
