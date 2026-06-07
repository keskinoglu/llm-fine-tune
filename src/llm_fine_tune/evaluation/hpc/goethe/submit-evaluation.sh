#!/usr/bin/env bash
# Run a code_snippet_translation evaluation for a named fine-tuned model.
#
# Two-phase execution (separable via bigcode's --generation_only / --load_generations_path):
#   Phase 1 — generation (GPU, ROCm venv): model produces code_snippet_from_llm_response
#   Phase 2 — execution  (Apptainer, --net none): compile + run against execution_engines
#   Phase 3 — report     (GPU, ROCm venv): write per-sample parquet + summary.md
#
# Usage (from $REPO_DIR):
#   cd "$REPO_DIR"
#   sbatch src/llm_fine_tune/evaluation/hpc/goethe/submit-evaluation.sh \
#       "$WORK_DIR/saves/my-model-merged"
#
# Required env vars (set in ~/.bashrc):
#   WORK_DIR  — e.g. /work/<group>/<user>
#   REPO_DIR  — e.g. $WORK_DIR/llm-fine-tune

#SBATCH --job-name=evaluation
#SBATCH --partition=gpu_test
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=00:30:00
#SBATCH --output=evaluation_%j.out
#SBATCH --mail-type=FAIL

set -euo pipefail

MODEL="${1:?
  No model path specified.
  Usage: sbatch submit-evaluation.sh <path/to/merged-model>
}"

: "${WORK_DIR:?WORK_DIR is not set — export WORK_DIR=/work/<group>/<user> in ~/.bashrc}"
: "${REPO_DIR:?REPO_DIR is not set — export REPO_DIR=\$WORK_DIR/llm-fine-tune in ~/.bashrc}"

EVALUATION_SIF="${WORK_DIR}/images/evaluation.sif"
RESULTS_DIR="${WORK_DIR}/evaluation-results/$(basename "$MODEL")-${SLURM_JOB_ID}"
GENERATIONS_FILE="${RESULTS_DIR}/generations.json"
METRICS_FILE="${RESULTS_DIR}/metrics.json"

module load rocm/6.2.4
source "$REPO_DIR/src/llm_fine_tune/finetune/hpc/goethe/env.sh"
source "$REPO_DIR/src/llm_fine_tune/finetune/hpc/common.sh"

mkdir -p "$RESULTS_DIR"

echo "==> Model:        $MODEL"
echo "==> Results dir:  $RESULTS_DIR"
echo "==> SIF image:    $EVALUATION_SIF"
echo ""

# ---------------------------------------------------------------------------
# Phase 1: generation (GPU, ROCm venv)
# ---------------------------------------------------------------------------
echo "==> Phase 1: generating translations ..."
source "$REPO_DIR/.venv/bin/activate"

run-bigcode-cli \
    --model "$MODEL" \
    --tasks code_snippet_translation \
    --generation_only \
    --save_generations_path "$GENERATIONS_FILE" \
    --max_length_generation 512 \
    --temperature 0.2 \
    --n_samples 1 \
    --batch_size 4

deactivate

# ---------------------------------------------------------------------------
# Phase 2: execution (Apptainer, --net none)
# ---------------------------------------------------------------------------
echo "==> Phase 2: executing translations in container ..."
export EVALUATION_SIF

apptainer exec --net none "$EVALUATION_SIF" \
    python -m llm_fine_tune.evaluation.run_bigcode_cli \
        --tasks code_snippet_translation \
        --load_generations_path "$GENERATIONS_FILE" \
        --metric_output_path "$METRICS_FILE" \
        --allow_code_execution

# ---------------------------------------------------------------------------
# Phase 3: report (ROCm venv)
# ---------------------------------------------------------------------------
echo "==> Phase 3: generating report ..."
source "$REPO_DIR/.venv/bin/activate"

evaluation-report \
    --results-json "$METRICS_FILE" \
    --out-dir "$RESULTS_DIR"

deactivate

echo ""
echo "==> Evaluation complete!"
echo "    Results: $RESULTS_DIR/evaluation-results.parquet"
echo "    Summary: $RESULTS_DIR/summary.md"
