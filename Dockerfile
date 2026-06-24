# ──────────────────────────────────────────────────────────────────────────────
# magna-retouch  —  multi-stage build
#
# Targets:
#   cpu   python:3.11-slim  (CI, benchmarking, Stage 1-3, no GPU ops)
#   gpu   nvidia/cuda base  (Stage 4 LUT training + inference)
#
# Build:
#   docker build --target cpu -t magna-retouch:cpu .
#   docker build --target gpu -t magna-retouch:gpu .
# ──────────────────────────────────────────────────────────────────────────────

# ── Shared system-dep layer ───────────────────────────────────────────────────
FROM python:3.11-slim-bookworm AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        # Lensfun lens database + dev headers
        liblensfun-dev \
        lensfun-data \
        # LibRaw (rawpy links against this at runtime)
        libraw-dev \
        # OpenCV runtime libs
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        # General build tools
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# ── CPU target ────────────────────────────────────────────────────────────────
FROM base AS cpu

# Install CPU-only PyTorch first (avoids pulling 2 GB CUDA wheel in CPU builds)
RUN pip install torch==2.3.0 torchvision==0.18.0 \
        --index-url https://download.pytorch.org/whl/cpu

RUN pip install -r requirements.txt

COPY . .

CMD ["python", "-m", "src.pipeline", "--help"]

# ── GPU target ────────────────────────────────────────────────────────────────
# CUDA 12.1 + cuDNN 8 — matches PyTorch 2.3.0 cu121 wheel
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS gpu

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3-pip \
        liblensfun-dev \
        lensfun-data \
        libraw-dev \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

WORKDIR /app

COPY requirements.txt .

RUN pip install torch==2.3.0 torchvision==0.18.0 \
        --index-url https://download.pytorch.org/whl/cu121

RUN pip install -r requirements.txt

COPY . .

CMD ["python", "-m", "src.pipeline", "--help"]
