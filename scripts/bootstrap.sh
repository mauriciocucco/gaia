#!/usr/bin/env bash
# Bootstrap script — sets up a development environment from scratch.
# Usage:  bash scripts/bootstrap.sh

set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

echo "==> Installing package in editable mode with dev extras..."
pip install --upgrade pip
pip install -e ".[dev]"

echo "==> Checking system dependencies..."
missing=()
command -v ffmpeg  >/dev/null 2>&1 || missing+=(ffmpeg)
command -v yt-dlp  >/dev/null 2>&1 || missing+=(yt-dlp)

if [ ${#missing[@]} -gt 0 ]; then
  echo ""
  echo "WARNING: The following system tools are missing: ${missing[*]}"
  echo "  Video/media tools will not work without them."
  echo "  Install with:  sudo apt-get install -y ffmpeg && pip install yt-dlp"
  echo ""
fi

echo "==> Running tests..."
python -m pytest tests/ -x -q

echo ""
echo "Done! Activate the environment with:  source .venv/bin/activate"
