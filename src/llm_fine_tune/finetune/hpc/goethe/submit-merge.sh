#!/usr/bin/env bash
# Merge a LoRA adapter into the base model weights.
# To merge AND publish in one job, use submit-merge-and-publish.sh.
# To publish an already-merged model, use: publish-model --help
#
# Usage (from $REPO_DIR):
#   cd "$REPO_DIR"
#   sbatch src/llm_fine_tune/finetune/hpc/goethe/submit-merge.sh \
#       src/llm_fine_tune/finetune/configs/llama-3.2-1b-merge.yaml \
#       "$WORK_DIR/saves/test-llama-3.2-1b-lora"
#
#   Merged model is written to $WORK_DIR/exports/<adapter_dir_name>/
#
# Required env vars (set in ~/.bashrc):
#   WORK_DIR  — e.g. /work/<group>/<user>
#   REPO_DIR  — e.g. $WORK_DIR/llm-fine-tune

#SBATCH --job-name=llm-merge
#SBATCH --partition=gpu_test
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=00:30:00
#SBATCH --output=merge_%j.out
#SBATCH --mail-type=FAIL

set -euo pipefail

CONFIG="${1:?
  No config specified.
  Usage: sbatch submit-merge.sh <config> <adapter_dir>
  Example: sbatch submit-merge.sh src/llm_fine_tune/finetune/configs/llama-3.2-1b-merge.yaml \$WORK_DIR/saves/test-llama-3.2-1b-lora
}"
ADAPTER_DIR="${2:?
  No adapter directory specified.
  Usage: sbatch submit-merge.sh <config> <adapter_dir>
  Example: \$WORK_DIR/saves/test-llama-3.2-1b-lora
}"

module load rocm/6.2.4
source "$REPO_DIR/src/llm_fine_tune/finetune/hpc/goethe/env.sh"
source "$REPO_DIR/src/llm_fine_tune/finetune/hpc/common.sh"

validate_config "$CONFIG"

EXPORT_DIR="$WORK_DIR/exports/$(basename "$ADAPTER_DIR")"

merge_lora "$CONFIG" "$ADAPTER_DIR" "$EXPORT_DIR"

echo ""
echo "==> Merged model saved to: $EXPORT_DIR"
echo "    To publish, run:"
echo "    source \$REPO_DIR/.venv/bin/activate"
echo "    publish-model --model-dir \"$EXPORT_DIR\" --repo-id <repo_id> --tag <tag> --message \"<message>\""
