#!/usr/bin/env bash
# Shared helpers sourced by every cluster job script.
# Source this AFTER the cluster's env.sh has loaded modules and set REPO_DIR/WORK_DIR.

# ---------------------------------------------------------------------------
# Validate required environment variables (fail fast with a readable message)
# ---------------------------------------------------------------------------
: "${WORK_DIR:?
  WORK_DIR is not set. Add this to your ~/.bashrc and re-source it:
    export WORK_DIR=/work/<your_group>/<your_username>
}"
: "${REPO_DIR:?
  REPO_DIR is not set. Add this to your ~/.bashrc and re-source it:
    export REPO_DIR=\$WORK_DIR/llm-fine-tune
}"

validate_config() {
    local config="$1"
    if [[ -z "$config" ]]; then
        echo "ERROR: No config file specified."
        echo "Usage: <submit-script> src/llm_fine_tune/finetune/configs/<name>.yaml"
        exit 1
    fi
    if [[ ! -f "$REPO_DIR/$config" ]]; then
        echo "ERROR: Config not found: $REPO_DIR/$config"
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# launch_training <repo-relative-config> <output_dir> [extra key=value ...]
#
# Activates the venv, then runs llamafactory-cli train via torchrun.
# FORCE_TORCHRUN=1 makes LLaMA-Factory use torchrun, which spawns one process
# per visible GPU automatically — no manual --nproc_per_node needed.
# ---------------------------------------------------------------------------
launch_training() {
    local config="$1"
    local output_dir="$2"
    shift 2

    source "$REPO_DIR/.venv/bin/activate"

    echo "==> Starting training"
    echo "    Config:      $REPO_DIR/$config"
    echo "    Dataset dir: $REPO_DIR/src/llm_fine_tune/finetune"
    echo "    Output dir:  $output_dir"
    echo ""

    FORCE_TORCHRUN=1 llamafactory-cli train "$REPO_DIR/$config" \
        dataset_dir="$REPO_DIR/src/llm_fine_tune/finetune" \
        output_dir="$output_dir" \
        "$@"
}

# ---------------------------------------------------------------------------
# merge_lora <repo-relative-config> <adapter_dir> <export_dir>
#
# Fuses a LoRA adapter into the base model and writes the merged weights.
# Single-process (no torchrun, no dataset) — CPU merge is exact and avoids
# GPU OOM on large models.
# ---------------------------------------------------------------------------
merge_lora() {
    local config="$1"
    local adapter_dir="$2"
    local export_dir="$3"

    source "$REPO_DIR/.venv/bin/activate"

    echo "==> Merging LoRA adapter into base model"
    echo "    Config:      $REPO_DIR/$config"
    echo "    Adapter dir: $adapter_dir"
    echo "    Export dir:  $export_dir"
    echo ""

    llamafactory-cli export "$REPO_DIR/$config" \
        adapter_name_or_path="$adapter_dir" \
        export_dir="$export_dir"
}
