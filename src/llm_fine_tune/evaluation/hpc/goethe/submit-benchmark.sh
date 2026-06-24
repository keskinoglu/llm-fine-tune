#!/usr/bin/env bash
# Run the full standard-benchmark track for one model (base or fine-tune).
#
# Three phases, following the same GPU-vs-sandbox boundary as submit-evaluation.sh:
#   Phase 1 — perplexity      (GPU, project .venv + ROCm):    held-out NLL → perplexity.json
#   Phase 2 — lm-eval         (GPU, uv --project eval-envs/lmeval): general tasks → lmeval/
#   Phase 3 — code-bench gen  (GPU, project .venv):           MultiPL-E prompts → code_benchmark_generations.parquet
#   Phase 4 — code-bench run  (Apptainer, --net none):        completions+tests → code_benchmark_metrics.json
#
# Run once for the base model and once for the fine-tune, then compare with benchmark-report.
#
# Usage (from $REPO_DIR):
#   cd "$REPO_DIR"
#   sbatch src/llm_fine_tune/evaluation/hpc/goethe/submit-benchmark.sh \
#       <model-hf-id> [--limit N]
#
#   --limit N  cap every task at N samples (shakeout; omit for a full run)
#
# Required env vars (set in ~/.bashrc):
#   WORK_DIR  — e.g. /work/<group>/<user>
#   REPO_DIR  — e.g. $WORK_DIR/llm-fine-tune
#
# Prerequisites: submit-benchmark-setup.sh must have completed successfully.

#SBATCH --job-name=benchmark
#SBATCH --partition=gpu_test
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=08:00:00
#SBATCH --output=benchmark_%j.out
#SBATCH --mail-type=FAIL

set -euo pipefail

MODEL="${1:?
  No model specified.
  Usage: sbatch submit-benchmark.sh <hf-model-id> [--limit N]
}"
shift

LIMIT_FLAG=""
if [[ "${1:-}" == "--limit" ]]; then
    LIMIT_N="${2:?--limit requires a value}"
    LIMIT_FLAG="--limit $LIMIT_N"
    shift 2
fi

: "${WORK_DIR:?WORK_DIR is not set — export WORK_DIR=/work/<group>/<user> in ~/.bashrc}"
: "${REPO_DIR:?REPO_DIR is not set — export REPO_DIR=\$WORK_DIR/llm-fine-tune in ~/.bashrc}"

module load rocm/6.2.4

source "$REPO_DIR/src/llm_fine_tune/finetune/hpc/goethe/env.sh"
source "$REPO_DIR/src/llm_fine_tune/finetune/hpc/common.sh"

EVALUATION_IMAGE="${WORK_DIR}/images/evaluation"
EVAL_ENVS_DIR="${REPO_DIR}/eval-envs"

MODEL_SLUG="$(echo "$MODEL" | tr '/' '_')"
RESULTS_DIR="${WORK_DIR}/benchmark-results/${MODEL_SLUG}-${SLURM_JOB_ID}"

mkdir -p "$RESULTS_DIR"

echo "==> Model:       $MODEL"
echo "==> Results dir: $RESULTS_DIR"
[[ -n "$LIMIT_FLAG" ]] && echo "==> Limit:       $LIMIT_N samples per config"
echo ""

# ---------------------------------------------------------------------------
# Phase 1: held-out perplexity (GPU, project .venv)
# ---------------------------------------------------------------------------
echo "==> Phase 1: computing held-out perplexity on leetcode_instruct_test ..."
source "$REPO_DIR/.venv/bin/activate"

compute-heldout-perplexity \
    --model "$MODEL" \
    --output-dir "$RESULTS_DIR" \
    --batch-size 8 \
    $LIMIT_FLAG

deactivate

# ---------------------------------------------------------------------------
# Phase 2: lm-eval general tasks (GPU, eval-envs/lmeval)
# ---------------------------------------------------------------------------
# Core curated: mmlu (capped with --limit to stay in 8h), gsm8k, hellaswag, arc_challenge, winogrande.
# MMLU has 57 subtasks and ~14k questions; --limit 100 per subtask keeps total tokens manageable.
echo ""
echo "==> Phase 2: lm-eval general tasks ..."

LMEVAL_LIMIT_FLAG=""
[[ -n "$LIMIT_FLAG" ]] && LMEVAL_LIMIT_FLAG="--limit $LIMIT_N"

# --output_path is a directory: with --log_samples lm-eval writes
# <dir>/<model_sanitized>/results_<timestamp>.json (benchmark-report globs for it).
uv run --project "$EVAL_ENVS_DIR/lmeval" lm_eval \
    --model hf \
    --model_args "pretrained=${MODEL},dtype=bfloat16" \
    --tasks mmlu,gsm8k,hellaswag,arc_challenge,winogrande \
    --batch_size auto \
    --output_path "$RESULTS_DIR/lmeval" \
    --log_samples \
    $LMEVAL_LIMIT_FLAG

# ---------------------------------------------------------------------------
# Phase 3: code-benchmark generation (GPU, project .venv)
# ---------------------------------------------------------------------------
echo ""
echo "==> Phase 3: generating MultiPL-E completions (GPU) ..."
source "$REPO_DIR/.venv/bin/activate"

python -m llm_fine_tune.evaluation.generate_code_benchmark \
    --model "$MODEL" \
    --output-dir "$RESULTS_DIR" \
    $LIMIT_FLAG

deactivate

# ---------------------------------------------------------------------------
# Phase 4: code-benchmark execution (Apptainer, --net --network none)
# ---------------------------------------------------------------------------
# Reuses the translation-eval sandbox image (g++, javac, python3 — no model, no network).
# Bind live src + PYTHONPATH so scoring code edits take effect without rebuilding the image.
echo ""
echo "==> Phase 4: running MultiPL-E completions in container ..."

apptainer exec --net --network none --cleanenv \
    --bind "$RESULTS_DIR" \
    --bind "$REPO_DIR/src":/opt/live-src \
    --env PYTHONPATH=/opt/live-src \
    "$EVALUATION_IMAGE" \
    python -m llm_fine_tune.evaluation.run_code_benchmark_scoring \
        --generations "$RESULTS_DIR/code_benchmark_generations.parquet" \
        --metrics-json "$RESULTS_DIR/code_benchmark_metrics.json"

echo ""
echo "==> Benchmark complete!"
echo "    Results: $RESULTS_DIR"
echo ""
echo "Run benchmark-report to compare two models:"
echo "    source \"\$REPO_DIR/.venv/bin/activate\""
echo "    benchmark-report --base-dir <base-results> --ft-dir <ft-results> --out-dir <outdir>"
