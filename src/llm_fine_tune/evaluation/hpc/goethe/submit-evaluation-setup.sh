#!/usr/bin/env bash
# SLURM wrapper: install the evaluation extra and build evaluation.sif on a compute node.
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
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=01:00:00
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
# Install the evaluation extra into the existing venv
# ---------------------------------------------------------------------------
echo "==> Installing evaluation extra (bigcode-eval + multipl-e) ..."
cd "$REPO_DIR"
uv sync --extra evaluation --verbose

# ---------------------------------------------------------------------------
# Build the Apptainer image
# ---------------------------------------------------------------------------
IMAGES_DIR="$WORK_DIR/images"
mkdir -p "$IMAGES_DIR"

echo "==> Building evaluation.sif (pulls ghcr.io/bigcode-project/evaluation-harness-multiple) ..."
apptainer build \
    "$IMAGES_DIR/evaluation.sif" \
    "$REPO_DIR/src/llm_fine_tune/evaluation/hpc/goethe/evaluation_image.def"

echo ""
echo "==> Evaluation setup complete!"
echo "    SIF image: $IMAGES_DIR/evaluation.sif"
echo ""
echo "Verify the image has g++ and javac:"
echo "    apptainer exec $IMAGES_DIR/evaluation.sif g++ --version"
echo "    apptainer exec $IMAGES_DIR/evaluation.sif javac -version"
