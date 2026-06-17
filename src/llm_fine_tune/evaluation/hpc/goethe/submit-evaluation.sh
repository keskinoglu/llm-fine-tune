#!/usr/bin/env bash
# Run a code_snippet_translation evaluation for a named fine-tuned model.
#
# Three phases, split so the model never touches the execution sandbox and the untrusted
# generated code never touches the network:
#   Phase 1 — generation (GPU, ROCm venv): the model produces generations.json + the evaluation
#             parquet (network available) so Phase 2 can read it offline.
#   Phase 2 — execution  (Apptainer, --net --network none): the standalone scorer compiles + runs each
#             code_snippet against its execution_engine and writes per-sample metrics.json.
#             No bigcode, no model, no network.
#   Phase 3 — report     (ROCm venv): metrics.json -> per-sample parquet + summary.md.
#
# Usage (from $REPO_DIR), optionally with extra bigcode flags (e.g. --limit 20 for a shakeout):
#   cd "$REPO_DIR"
#   sbatch src/llm_fine_tune/evaluation/hpc/goethe/submit-evaluation.sh \
#       "$WORK_DIR/saves/my-model-merged" [extra bigcode generation flags...]
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
  Usage: sbatch submit-evaluation.sh <path/to/merged-model> [extra bigcode flags...]
}"
shift
GENERATION_FLAGS=("$@")  # passed through to generate-llm-responses (e.g. --limit 20)

: "${WORK_DIR:?WORK_DIR is not set — export WORK_DIR=/work/<group>/<user> in ~/.bashrc}"
: "${REPO_DIR:?REPO_DIR is not set — export REPO_DIR=\$WORK_DIR/llm-fine-tune in ~/.bashrc}"

EVALUATION_IMAGE="${WORK_DIR}/images/evaluation"  # --sandbox directory (mksquashfs unavailable under proot)
RESULTS_DIR="${WORK_DIR}/evaluation-results/$(basename "$MODEL")-${SLURM_JOB_ID}"
GENERATIONS_FILE="${RESULTS_DIR}/generations.json"
EVALUATION_PARQUET="${RESULTS_DIR}/evaluation.parquet"
METRICS_FILE="${RESULTS_DIR}/metrics.json"

module load rocm/6.2.4
source "$REPO_DIR/src/llm_fine_tune/finetune/hpc/goethe/env.sh"
source "$REPO_DIR/src/llm_fine_tune/finetune/hpc/common.sh"

mkdir -p "$RESULTS_DIR"

echo "==> Model:        $MODEL"
echo "==> Results dir:  $RESULTS_DIR"
echo "==> Image:        $EVALUATION_IMAGE"
echo ""

# ---------------------------------------------------------------------------
# Phase 1: generation + materialize the dataset (GPU, ROCm venv, network)
# ---------------------------------------------------------------------------
echo "==> Phase 1: generating translations ..."
source "$REPO_DIR/.venv/bin/activate"

# generations.json + evaluation.parquet come from the same rows in order, so Phase 2 can pair by index.
python -m llm_fine_tune.evaluation.generate_llm_responses \
    --model "$MODEL" \
    --output-dir "$RESULTS_DIR" \
    --max-new-tokens 512 \
    --temperature 0.2 \
    "${GENERATION_FLAGS[@]}"

deactivate

# ---------------------------------------------------------------------------
# Phase 2: execution + scoring (Apptainer, --net --network none, no bigcode)
# ---------------------------------------------------------------------------
# --net --network none: no outbound network for untrusted code. --cleanenv: keep host env out.
# Bind live src + PYTHONPATH ahead of the baked install, so Phase-2 code edits take effect
# without rebuilding the sandbox. --bind /work paths (not auto-mounted).
echo "==> Phase 2: executing + scoring translations in container ..."

apptainer exec --net --network none --cleanenv \
    --bind "$RESULTS_DIR" \
    --bind "$REPO_DIR/src":/opt/live-src \
    --env PYTHONPATH=/opt/live-src \
    "$EVALUATION_IMAGE" \
    python -m llm_fine_tune.evaluation.run_execution_scoring \
        --generations-json "$GENERATIONS_FILE" \
        --evaluation-parquet "$EVALUATION_PARQUET" \
        --metrics-json "$METRICS_FILE"

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
