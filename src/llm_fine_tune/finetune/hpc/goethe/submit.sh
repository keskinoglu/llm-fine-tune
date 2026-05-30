#!/usr/bin/env bash
# Full fine-tune job: all 8 AMD MI210 GPUs, gpu partition, up to 8 hours.
# Run submit-test.sh first to verify the pipeline.
#
# Usage (from $REPO_DIR):
#   cd "$REPO_DIR"
#   sbatch src/llm_fine_tune/finetune/hpc/goethe/submit.sh \
#       src/llm_fine_tune/finetune/configs/llama-3.2-1b-lora.yaml
#
# Required env vars (set in ~/.bashrc):
#   WORK_DIR  — e.g. /work/<group>/<user>
#   REPO_DIR  — e.g. $WORK_DIR/llm-fine-tune

#SBATCH --job-name=llm-finetune
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150g
#SBATCH --time=08:00:00
#SBATCH --output=finetune_%j.out
#SBATCH --mail-type=FAIL

set -euo pipefail

CONFIG="${1:?
  No config specified.
  Usage: sbatch submit.sh src/llm_fine_tune/finetune/configs/<name>.yaml
}"

module load rocm/6.2.4
source "$REPO_DIR/src/llm_fine_tune/finetune/hpc/goethe/env.sh"
source "$REPO_DIR/src/llm_fine_tune/finetune/hpc/common.sh"

validate_config "$CONFIG"

OUTPUT_DIR="$WORK_DIR/saves/$(basename "$CONFIG" .yaml)"

launch_training "$CONFIG" "$OUTPUT_DIR"
