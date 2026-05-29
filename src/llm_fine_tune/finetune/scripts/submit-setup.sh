#!/usr/bin/env bash
# SLURM wrapper for the one-time cluster setup.
#
# cluster-setup.sh (uv sync + torch extraction) is too CPU/IO-heavy for the
# login node — the login-node CPU-time limit will kill it silently mid-install.
# This script submits it to the `test` partition (CPU compute node) where there
# is no such limit.
#
# Usage (from $REPO_DIR):
#   cd "$REPO_DIR"
#   sbatch src/llm_fine_tune/finetune/scripts/submit-setup.sh
#
# Required environment variables (set in ~/.bashrc):
#   REPO_DIR   — path to this cloned repo, e.g. $WORK_DIR/llm-fine-tune
#   UV_CACHE_DIR — uv cache on /work, e.g. $WORK_DIR/.cache/uv
#
# Monitor progress:
#   scontrol show job <JOBID> | grep StdOut
#   tail -f setup_<JOBID>.out

#SBATCH --job-name=llm-setup
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=00:30:00
#SBATCH --output=setup_%j.out
#SBATCH --mail-type=FAIL

set -euo pipefail

: "${REPO_DIR:?
  REPO_DIR is not set. Add this to your ~/.bashrc and re-source it:
    export REPO_DIR=\$WORK_DIR/llm-fine-tune
}"

bash "$REPO_DIR/src/llm_fine_tune/finetune/scripts/cluster-setup.sh"
