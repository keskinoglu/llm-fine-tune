#!/usr/bin/env bash
# Rebuild the venv on a compute node after pyproject.toml / uv.lock changes.
# Pulls the latest lock file first, then runs uv sync.
#
# Use this after bumping a package version locally and pushing the new uv.lock.
# The login-node CPU-time limit kills heavy installs silently, so this runs on
# the test partition instead.
#
# Usage (from $REPO_DIR):
#   cd "$REPO_DIR"
#   sbatch src/llm_fine_tune/finetune/hpc/goethe/submit-rebuild-env.sh
#
# Required env vars (set in ~/.bashrc):
#   REPO_DIR  — e.g. $WORK_DIR/llm-fine-tune
#
# Monitor:
#   scontrol show job <JOBID> | grep StdOut
#   tail -f setup_<JOBID>.out

#SBATCH --job-name=llm-rebuild-env
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=01:00:00
#SBATCH --output=setup_%j.out
#SBATCH --mail-type=FAIL

set -euo pipefail

: "${REPO_DIR:?
  REPO_DIR is not set. Add this to your ~/.bashrc and re-source it:
    export REPO_DIR=\$WORK_DIR/llm-fine-tune
}"

source "$REPO_DIR/src/llm_fine_tune/finetune/hpc/goethe/env.sh"

echo "==> Pulling latest changes (pyproject.toml / uv.lock)..."
git -C "$REPO_DIR" pull

echo "==> Syncing venv (ROCm extras)..."
cd "$REPO_DIR"
uv sync --extra rocm --verbose

echo ""
echo "==> Env rebuild complete!"
python_bin="$REPO_DIR/.venv/bin/python"
echo "    torch:  $("$python_bin" -c 'import torch; print(torch.__version__)')"
echo "    rocm:   ${ROCM_PATH:-not set}"
