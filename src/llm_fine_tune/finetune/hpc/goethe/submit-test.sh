#!/usr/bin/env bash
# Test run: 10 steps on 1 GPU in the gpu_test partition (~15-30 min).
# Run this before the full job — verifies model download, dataset load, training loop.
#
# Usage (from $REPO_DIR):
#   cd "$REPO_DIR"
#   sbatch src/llm_fine_tune/finetune/hpc/goethe/submit-test.sh \
#       src/llm_fine_tune/finetune/configs/llama-3.2-1b-lora.yaml
#
# Required env vars (set in ~/.bashrc):
#   WORK_DIR  — e.g. /work/<group>/<user>
#   REPO_DIR  — e.g. $WORK_DIR/llm-fine-tune

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

CONFIG="${1:?
  No config specified.
  Usage: sbatch submit-test.sh src/llm_fine_tune/finetune/configs/<name>.yaml
}"

source "$REPO_DIR/src/llm_fine_tune/finetune/hpc/goethe/env.sh"
source "$REPO_DIR/src/llm_fine_tune/finetune/hpc/common.sh"

validate_config "$CONFIG"

OUTPUT_DIR="$WORK_DIR/saves/test-$(basename "$CONFIG" .yaml)"
echo "==> Test run (10 steps, 1 GPU, gpu_test)"

launch_training "$CONFIG" "$OUTPUT_DIR" max_steps=10
