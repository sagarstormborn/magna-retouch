#!/usr/bin/env bash
# Cross-platform dev environment setup (macOS + Linux)
# Run once: bash setup.sh
set -euo pipefail

PYTHON=${PYTHON:-python3}

echo "==> Creating virtual environment"
$PYTHON -m venv .venv
source .venv/bin/activate

echo "==> Upgrading pip / wheel"
pip install --upgrade pip wheel

# ── Platform-specific system deps ─────────────────────────────────────────────
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "==> Installing Linux system dependencies"
    # Lensfun database + dev headers
    sudo apt-get install -y --no-install-recommends \
        liblensfun-dev lensfun-data \
        libraw-dev \
        libexiv2-dev \
        libopenblas-dev \
        2>/dev/null || true
fi

if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "==> Checking Homebrew dependencies"
    which brew >/dev/null 2>&1 || { echo "Install Homebrew first: https://brew.sh"; exit 1; }
    brew install lensfun libraw exiv2 || true
fi

echo "==> Installing Python dependencies"
pip install -r requirements.txt

echo ""
echo "==> Setup complete. Activate with: source .venv/bin/activate"
echo "==> Run tests:   pytest tests/ -v"
echo "==> Run pipeline: python -m src.pipeline --help"
