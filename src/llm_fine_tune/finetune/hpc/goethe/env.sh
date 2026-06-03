#!/usr/bin/env bash
# Goethe cluster environment setup (AMD MI210, ROCm, SLURM).
# Source this at the top of every Goethe job script before sourcing hpc/common.sh.

# Redirect caches from /home (30 GB quota) to /work.
# UV_LINK_MODE=hardlink avoids redundant copies when cache + venv share the filesystem.
export UV_CACHE_DIR="${UV_CACHE_DIR:-$(dirname "$REPO_DIR")/.cache/uv}"
export UV_LINK_MODE=hardlink
mkdir -p "$UV_CACHE_DIR"

# HuggingFace defaults to ~/.cache/huggingface which may not exist on compute nodes.
export HF_HOME="${HF_HOME:-$(dirname "$REPO_DIR")/.cache/huggingface}"
mkdir -p "$HF_HOME"

export FINETUNE_EXTRA=rocm
