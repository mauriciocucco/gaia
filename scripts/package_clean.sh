#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="${1:-$REPO_ROOT/dist/clean-package}"
ARCHIVE_BASENAME="${2:-hf-gaia-agent-clean}"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
ARCHIVE_PATH="$OUTPUT_DIR/$ARCHIVE_BASENAME-$TIMESTAMP.tar.gz"

mkdir -p "$OUTPUT_DIR"

tar -czf "$ARCHIVE_PATH" \
  --exclude=".git" \
  --exclude=".env" \
  --exclude=".venv" \
  --exclude=".uv-cache" \
  --exclude=".cache" \
  --exclude=".pytest_cache" \
  --exclude=".runtime-artifacts" \
  --exclude=".test-artifacts" \
  --exclude=".tmp" \
  --exclude=".tmp_pytest" \
  --exclude="__pycache__" \
  --exclude="pytest-cache-files-lkr7tsdo" \
  --exclude="pytest-cache-files-v71sdjtx" \
  --exclude="testtmpbase" \
  --exclude="dist" \
  -C "$REPO_ROOT" \
  .

printf '%s\n' "$ARCHIVE_PATH"
