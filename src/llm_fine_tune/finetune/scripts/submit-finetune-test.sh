#!/usr/bin/env bash
# Test run: 10 training steps on 1 GPU in the gpu_test partition.
# Run this before committing to a full 8-hour job — verifies model download,
# dataset loading, and the training loop in roughly 15-30 minutes.
#
# Usage:
#   sbatch submit-finetune-test.sh src/llm_fine_tune/finetune/configs/gpt-oss-20b-lora.yaml
#
# Required environment variables (set in ~/.bashrc on the cluster):
#   WORK_DIR   — your fast work directory, e.g. /work/<group>/<username>
#   REPO_DIR   — path to this cloned repo, e.g. $WORK_DIR/llm-fine-tune

#SBATCH --job-name=llm-finetune-test
#SBATCH --partition=gpu_test
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150g
#SBATCH --time=00:30:00
#SBATCH --output=finetune_test_%j.out
#SBATCH --mail-type=FAIL

set -euo pipefail

# ---------------------------------------------------------------------------
# Validate inputs
# ---------------------------------------------------------------------------
: "${WORK_DIR:?
  WORK_DIR is not set. Add this to your ~/.bashrc and re-source it:
    export WORK_DIR=/work/<your_group>/<your_username>
}"
: "${REPO_DIR:?
  REPO_DIR is not set. Add this to your ~/.bashrc and re-source it:
    export REPO_DIR=\$WORK_DIR/llm-fine-tune
}"
CONFIG="${1:?
  No config file specified.
  Usage: sbatch submit-finetune-test.sh <path/to/config.yaml>
  Example: sbatch submit-finetune-test.sh src/llm_fine_tune/finetune/configs/gpt-oss-20b-lora.yaml
}"

if [[ ! -f "$REPO_DIR/$CONFIG" ]]; then
    echo "ERROR: Config not found: $REPO_DIR/$CONFIG"
    exit 1
fi

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
module load rocm/6.2.4
source "$REPO_DIR/.venv/bin/activate"

# ---------------------------------------------------------------------------
# Launch training
# ---------------------------------------------------------------------------
OUTPUT_DIR="$WORK_DIR/saves/test-$(basename "$CONFIG" .yaml)"
echo "==> Starting test run (10 steps, 1 GPU, gpu_test)"
echo "    Config:      $REPO_DIR/$CONFIG"
echo "    Dataset dir: $REPO_DIR/src/llm_fine_tune/finetune"
echo "    Output dir:  $OUTPUT_DIR"
echo ""

FORCE_TORCHRUN=1 llamafactory-cli train "$REPO_DIR/$CONFIG" \
    --dataset_dir "$REPO_DIR/src/llm_fine_tune/finetune" \
    --output_dir "$OUTPUT_DIR" \
    --max_steps 10
