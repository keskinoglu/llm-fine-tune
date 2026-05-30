#!/usr/bin/env bash
# SLURM wrapper: runs setup.sh on a compute node (test partition).
#
# setup.sh runs `uv sync` which is too CPU/IO-heavy for the login node —
# the login-node CPU-time limit kills it silently mid-install.
#
# Usage (from $REPO_DIR):
#   cd "$REPO_DIR"
#   sbatch src/llm_fine_tune/finetune/hpc/goethe/submit-setup.sh
#
# Required env vars (set in ~/.bashrc):
#   REPO_DIR  — e.g. $WORK_DIR/llm-fine-tune
#
# Monitor:
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

bash "$REPO_DIR/src/llm_fine_tune/finetune/hpc/goethe/setup.sh"
