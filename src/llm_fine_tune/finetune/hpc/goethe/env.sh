#!/usr/bin/env bash
# Goethe cluster environment setup (AMD MI210, ROCm, SLURM).
# Source this at the top of every Goethe job script before sourcing hpc/common.sh.

module load rocm/6.2.4

# Redirect uv cache from /home (30 GB quota) to /work.
# UV_LINK_MODE=hardlink avoids redundant copies when cache + venv share the filesystem.
export UV_CACHE_DIR="${UV_CACHE_DIR:-$(dirname "$REPO_DIR")/.cache/uv}"
export UV_LINK_MODE=hardlink
mkdir -p "$UV_CACHE_DIR"

export FINETUNE_EXTRA=rocm
