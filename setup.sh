#!/usr/bin/env bash
# Cross-platform dev environment setup (macOS + Linux)
# Usage:
#   bash setup.sh          # CPU (default — fine for Stages 1-3 and benchmarking)
#   bash setup.sh --gpu    # CUDA 12.1 PyTorch (Stage 4 training)
set -euo pipefail

PYTHON=${PYTHON:-python3}
DEVICE=cpu
for arg in "$@"; do [[ "$arg" == "--gpu" ]] && DEVICE=cuda; done

echo "==> magna-retouch setup  (device=$DEVICE)"

# ── System dependencies ────────────────────────────────────────────────────────
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "==> Installing Linux system dependencies (apt)"
    sudo apt-get update -qq
    sudo apt-get install -y --no-install-recommends \
        liblensfun-dev lensfun-data \
        libraw-dev \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
        build-essential \
        2>/dev/null || true
fi

if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "==> Checking Homebrew dependencies"
    which brew >/dev/null 2>&1 || { echo "Install Homebrew first: https://brew.sh"; exit 1; }
    brew install lensfun libraw 2>/dev/null || true
fi

# ── Python venv ────────────────────────────────────────────────────────────────
echo "==> Creating virtual environment (.venv)"
$PYTHON -m venv .venv
source .venv/bin/activate
pip install --quiet --upgrade pip wheel

# ── PyTorch (must precede requirements.txt to avoid the default PyPI wheel) ───
if [[ "$DEVICE" == "cuda" ]]; then
    echo "==> Installing PyTorch 2.3.0 + CUDA 12.1"
    pip install --quiet \
        torch==2.3.0 torchvision==0.18.0 \
        --index-url https://download.pytorch.org/whl/cu121
else
    echo "==> Installing PyTorch 2.3.0 CPU-only"
    pip install --quiet \
        torch==2.3.0 torchvision==0.18.0 \
        --index-url https://download.pytorch.org/whl/cpu
fi

# ── Remaining deps (torch already installed so PyPI wheel won't override) ─────
echo "==> Installing remaining Python dependencies"
pip install --quiet -r requirements.txt

echo ""
echo "==> Done. Activate with:  source .venv/bin/activate"
echo ""
echo "   make test            # run 16 unit tests"
echo "   make process INPUT=data/raw/myproperty OUTPUT=data/processed/myproperty"
echo "   make train           # train Stage 4 LUT (needs data/train/)"
echo "   make benchmark       # run Stage 5 harness"
