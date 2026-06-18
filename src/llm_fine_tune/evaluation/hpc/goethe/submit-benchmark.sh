#!/usr/bin/env bash
# Run the full standard-benchmark track for one model (base or fine-tune).
#
# Four phases, following the same GPU-vs-sandbox boundary as submit-evaluation.sh:
#   Phase 1 — perplexity    (GPU, project .venv + ROCm):      held-out NLL → perplexity.json
#   Phase 2 — lm-eval       (GPU, uv --project eval-envs/lmeval):      general tasks → lmeval/
#   Phase 3 — bigcode gen   (GPU, uv --project eval-envs/bigcode-gen): per task → generations_<task>.json
#   Phase 4 — bigcode exec  (Apptainer, --net none):                  compile+run → bigcode_<task>.json
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

BIGCODE_IMAGE="${WORK_DIR}/images/bigcode-multiple"
BIGCODE_REPO="${WORK_DIR}/bigcode-evaluation-harness"
EVAL_ENVS_DIR="${REPO_DIR}/eval-envs"

MODEL_SLUG="$(echo "$MODEL" | tr '/' '_')"
RESULTS_DIR="${WORK_DIR}/benchmark-results/${MODEL_SLUG}-${SLURM_JOB_ID}"

mkdir -p "$RESULTS_DIR"

echo "==> Model:       $MODEL"
echo "==> Results dir: $RESULTS_DIR"
[[ -n "$LIMIT_FLAG" ]] && echo "==> Limit:       $LIMIT_N samples per task"
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
# Phase 3: bigcode generation (GPU, eval-envs/bigcode-gen)
# ---------------------------------------------------------------------------
echo ""
echo "==> Phase 3: bigcode generation (GPU) ..."

BIGCODE_LIMIT_FLAG=""
[[ -n "$LIMIT_FLAG" ]] && BIGCODE_LIMIT_FLAG="--limit $LIMIT_N"

for task in humaneval multiple-cpp multiple-java multiple-py; do
    echo "  -> Generating: $task"
    uv run --project "$EVAL_ENVS_DIR/bigcode-gen" python "$BIGCODE_REPO/main.py" \
        --model "$MODEL" \
        --tasks "$task" \
        --n_samples 1 \
        --batch_size 1 \
        --max_length_generation 512 \
        --precision bf16 \
        --generation_only \
        --save_generations \
        --save_generations_path "$RESULTS_DIR/generations_${task}.json" \
        $BIGCODE_LIMIT_FLAG
done

# ---------------------------------------------------------------------------
# Phase 4: bigcode execution (Apptainer, --net --network none)
# ---------------------------------------------------------------------------
# The bigcode image has g++, javac, python3, and the harness code at /app/main.py.
# We bind the results dir so the container can read generations and write metrics.
# bigcode still instantiates the model's *tokenizer* even with --load_generations_path, so
# bind the HF cache (populated during Phase 3) and force offline — otherwise the tokenizer
# load reaches for the network, which --net none blocks. No model weights load here.
echo ""
echo "==> Phase 4: bigcode execution (container, network isolated) ..."

for task in humaneval multiple-cpp multiple-java multiple-py; do
    echo "  -> Executing: $task"
    apptainer exec --net --network none --cleanenv \
        --bind "$RESULTS_DIR" \
        --bind "$HF_HOME" \
        --env HF_HOME="$HF_HOME" \
        --env HF_HUB_OFFLINE=1 \
        --env TRANSFORMERS_OFFLINE=1 \
        "$BIGCODE_IMAGE" \
        python3 /app/main.py \
            --model "$MODEL" \
            --tasks "$task" \
            --n_samples 1 \
            --load_generations_path "$RESULTS_DIR/generations_${task}.json" \
            --allow_code_execution \
            --metric_output_path "$RESULTS_DIR/bigcode_${task}.json"
done

echo ""
echo "==> Benchmark complete!"
echo "    Results: $RESULTS_DIR"
echo ""
echo "Run benchmark-report to compare two models:"
echo "    source \"\$REPO_DIR/.venv/bin/activate\""
echo "    benchmark-report --base-dir <base-results> --ft-dir <ft-results> --out-dir <outdir>"
