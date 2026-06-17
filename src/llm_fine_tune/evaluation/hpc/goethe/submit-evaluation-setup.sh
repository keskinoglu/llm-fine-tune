#!/usr/bin/env bash
# SLURM wrapper: install the evaluation extra and build the evaluation sandbox image on a compute node.
#
# Must run on a compute node (not the login node) because:
#   - uv sync with large extras is too CPU/IO-heavy for the login node
#   - apptainer build needs network access (pulls the base Docker image)
#
# Usage (from $REPO_DIR):
#   cd "$REPO_DIR"
#   sbatch src/llm_fine_tune/evaluation/hpc/goethe/submit-evaluation-setup.sh
#
# Required env vars (set in ~/.bashrc):
#   WORK_DIR  — e.g. /work/<group>/<user>
#   REPO_DIR  — e.g. $WORK_DIR/llm-fine-tune
#
# Monitor:
#   scontrol show job <JOBID> | grep StdOut
#   tail -f evaluation_setup_<JOBID>.out

#SBATCH --job-name=evaluation-setup
#SBATCH --partition=general1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=04:00:00
#SBATCH --output=evaluation_setup_%j.out
#SBATCH --mail-type=FAIL

set -euo pipefail

: "${WORK_DIR:?
  WORK_DIR is not set. Add this to your ~/.bashrc and re-source it:
    export WORK_DIR=/work/<your_group>/<your_username>
}"
: "${REPO_DIR:?
  REPO_DIR is not set. Add this to your ~/.bashrc and re-source it:
    export REPO_DIR=\$WORK_DIR/llm-fine-tune
}"

source "$REPO_DIR/src/llm_fine_tune/finetune/hpc/goethe/env.sh"

# ---------------------------------------------------------------------------
# Install the ROCm stack into the venv (Phase-1 generation: torch + transformers)
# ---------------------------------------------------------------------------
echo "==> Installing the ROCm stack ..."
cd "$REPO_DIR"
# rocm so torch is a direct dep (routed to the ROCm index, not the CUDA default).
uv sync --extra rocm --verbose

# ---------------------------------------------------------------------------
# Build the Apptainer image (--sandbox: a directory, not a .sif)
# ---------------------------------------------------------------------------
# --sandbox (a directory, not a .sif): packing a .sif needs mksquashfs, which can't run under
# proot here (no /etc/subuid). apptainer exec runs the directory directly.
IMAGES_DIR="$WORK_DIR/images"
EVALUATION_IMAGE="$IMAGES_DIR/evaluation"
mkdir -p "$IMAGES_DIR"

echo "==> Building $EVALUATION_IMAGE (python:3.11-slim + g++ + openjdk-17, --sandbox) ..."
apptainer build --sandbox \
    "$EVALUATION_IMAGE" \
    "$REPO_DIR/src/llm_fine_tune/evaluation/hpc/goethe/evaluation_image.def"

echo ""
echo "==> Evaluation setup complete!"
echo "    Image (sandbox dir): $EVALUATION_IMAGE"
echo ""
echo "Verify the image (network-isolated exec works unprivileged here):"
echo "    apptainer exec --net --network none $EVALUATION_IMAGE g++ --version"
echo "    apptainer exec --net --network none $EVALUATION_IMAGE javac -version"
echo "    apptainer exec --net --network none $EVALUATION_IMAGE python -c 'import llm_fine_tune.evaluation.run_execution_scoring'"
