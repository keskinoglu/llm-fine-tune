#!/usr/bin/env bash
# Goethe cluster environment setup (AMD MI210, ROCm, SLURM).
# Source this at the top of every Goethe job script before sourcing hpc/common.sh.

# Redirect caches from /home (30 GB quota) to /work.
# UV_LINK_MODE=hardlink avoids redundant copies when cache + venv share the filesystem.
export UV_CACHE_DIR="${UV_CACHE_DIR:-$(dirname "$REPO_DIR")/.cache/uv}"
export UV_LINK_MODE=hardlink
mkdir -p "$UV_CACHE_DIR"

# HuggingFace cache. HF_HOME (model weights + downloaded dataset files) stays on /work so
# nothing re-downloads between jobs.
export HF_HOME="${HF_HOME:-$(dirname "$REPO_DIR")/.cache/huggingface}"
mkdir -p "$HF_HOME"

# PanFS doesn't honor fcntl.flock() under concurrency; use node-local storage for dataset locks.
export HF_DATASETS_CACHE="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/hf-datasets-${SLURM_JOB_ID:-$$}"
mkdir -p "$HF_DATASETS_CACHE"

# Apptainer caches + builds multi-GB image layers; keep them off /home.
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$(dirname "$REPO_DIR")/.cache/apptainer}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-$APPTAINER_CACHEDIR/tmp}"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

export FINETUNE_EXTRA=rocm
