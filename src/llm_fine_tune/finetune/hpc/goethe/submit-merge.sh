#!/usr/bin/env bash
# Merge a LoRA adapter into the base model and optionally publish to HuggingFace.
# Runs on gpu_test (1 GPU) — merge is CPU-only but ROCm must be loaded for the venv.
#
# Usage (from $REPO_DIR):
#   cd "$REPO_DIR"
#   sbatch src/llm_fine_tune/finetune/hpc/goethe/submit-merge.sh \
#       src/llm_fine_tune/finetune/configs/llama-3.2-1b-merge.yaml \
#       "$WORK_DIR/saves/test-llama-3.2-1b-lora" \
#       [tkeskin/llama-3.2-1b-instruct-code-translation] \
#       [v0.1]
#
#   repo_id omitted → merge only, prints the publish-model command to run manually.
#   tag omitted     → no git tag applied to the HF repo.
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
  Usage: sbatch submit-merge.sh src/llm_fine_tune/finetune/configs/<name>-merge.yaml <adapter_dir> [repo_id]
}"
ADAPTER_DIR="${2:?
  No adapter directory specified.
  Example: \$WORK_DIR/saves/test-llama-3.2-1b-lora
}"
REPO_ID="${3:-}"
TAG="${4:-}"

module load rocm/6.2.4
source "$REPO_DIR/src/llm_fine_tune/finetune/hpc/goethe/env.sh"
source "$REPO_DIR/src/llm_fine_tune/finetune/hpc/common.sh"

validate_config "$CONFIG"

EXPORT_DIR="$WORK_DIR/exports/$(basename "$ADAPTER_DIR")"

merge_lora "$CONFIG" "$ADAPTER_DIR" "$EXPORT_DIR"

if [[ -n "$REPO_ID" ]]; then
    echo ""
    echo "==> Publishing merged model to $REPO_ID"
    publish-model --model-dir "$EXPORT_DIR" --repo-id "$REPO_ID" ${TAG:+--tag "$TAG"}
else
    echo ""
    echo "==> Merge complete. To publish, run:"
    echo "    source \$REPO_DIR/.venv/bin/activate"
    echo "    publish-model --model-dir \"$EXPORT_DIR\" --repo-id tkeskin/llama-3.2-1b-instruct-code-translation [--tag v0.1]"
fi
