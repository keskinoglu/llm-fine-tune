# Stage 5: Evaluation

This directory drives the held-out **evaluation** config of the `parallel_corpus` through
[bigcode-evaluation-harness](https://github.com/bigcode-project/bigcode-evaluation-harness),
measuring whether a model produces translations that actually **compile and run correctly** —
not just whether they look plausible.

Each `bigcode_task_payload` is one `source_language` → `target_language` pair. The model receives a
`code_snippet_to_translate`; we extract the `code_snippet_from_llm_response`, assemble it with the
payload's `execution_engine` into an `code_snippet_with_execution_wiring`, run it against the
`expected_input_output_pairs`, and score the result.

---

## Directory layout

```
evaluation/
  custom_bigcode_tasks.py     — CodeSnippetTranslationTask: registers our task with bigcode
  run_bigcode_cli.py          — registers the task, then hands off to bigcode's CLI (run-bigcode-cli)
  extract_code_snippet_from_llm_response.py — strips prose/fences → code_snippet_from_llm_response
  score.py                    — composes the metric measures for one bigcode_task_payload
  metrics.py                  — individual measures (compiled, test_pass_rate, pass@1, runtime, loc)
  report.py                   — per-sample parquet + summary.md (evaluation-report)
  hpc/
    goethe/                   — AMD MI210, ROCm, SLURM — the three-phase cluster job
  README.md                   — this file
```

The actual running of code lives one level up in [`execution_harness/`](../execution_harness/)
(`execution.py` + `datatypes.py`) — shared with the dataset-build and validation paths.

---

## The three-phase split (and why)

Generation needs a GPU and the full ML stack; running untranslated **untrusted model output** needs
a locked-down sandbox. Those are different environments, so the cluster job
([`hpc/goethe/submit-evaluation.sh`](hpc/goethe/submit-evaluation.sh)) splits them — bigcode supports
this natively via `--generation_only` / `--load_generations_path`:

| Phase | Where | What runs |
|---|---|---|
| **1 — generation** | GPU node, ROCm `.venv` | Model produces `llm_response` for every `bigcode_task_payload` → `generations.json` |
| **2 — execution** | Apptainer `.sif`, `--net none` | Compile + run each `code_snippet_with_execution_wiring`, score → `metrics.json` |
| **3 — report** | login/compute, ROCm `.venv` | `evaluation-report` → `evaluation-results.parquet` + `summary.md` |

Phase 2 is the security boundary: model-generated code is untrusted, so it executes inside the
container with no network. The `.sif` carries only `g++`/`javac`/`python` and the evaluation modules
— no model, no torch (see [`hpc/goethe/evaluation_image.def`](hpc/goethe/evaluation_image.def)).

---

## Metrics

Per `bigcode_task_payload`, [`score.py`](score.py) emits:

| Metric | Meaning |
|---|---|
| `compiled` | 1.0 if the `code_snippet_with_execution_wiring` built at all |
| `test_pass_rate` | fraction of `expected_input_output_pairs` the model's code matched |
| `pass@1` | 1.0 only if **every** pair passed |
| `runtime_ms` | wall-clock of the run |
| `loc` / `char_count` | size of `code_snippet_from_llm_response` |

[`report.py`](report.py) aggregates these by `source_language` × `target_language` × `difficulty`.
Low absolute numbers are expected with small models — **the result is the delta** between a baseline
model and its fine-tuned counterpart, which is why `compiled` is tracked separately from `pass@1`.

---

## Running on the cluster (Goethe)

Prerequisites: the full repo at `$REPO_DIR`, `WORK_DIR`/`REPO_DIR` exported in `~/.bashrc`, and the
**evaluation** config published to HuggingFace (`make upload DATASET=evaluation`) — the task loads
`tkeskin/leetcode-solutions / evaluation` from the Hub at job start.

```bash
cd "$REPO_DIR"

# One-time: install the evaluation extra + build evaluation.sif (compute node — needs network)
sbatch src/llm_fine_tune/evaluation/hpc/goethe/submit-evaluation-setup.sh

# Baseline: evaluate the un-fine-tuned model (pass an HF id directly)
sbatch src/llm_fine_tune/evaluation/hpc/goethe/submit-evaluation.sh Qwen/Qwen2.5-Coder-1.5B-Instruct

# Tuned: evaluate the merged fine-tune; diff its summary.md against the baseline
sbatch src/llm_fine_tune/evaluation/hpc/goethe/submit-evaluation.sh "$WORK_DIR/saves/<merged-dir>"
```

Results land in `$WORK_DIR/evaluation-results/<model>-<jobid>/` as `evaluation-results.parquet` and
`summary.md`. The baseline-vs-tuned comparison of those summaries is the project's conclusion.

After building the `.sif`, sanity-check it before spending a GPU job:

```bash
apptainer exec "$WORK_DIR/images/evaluation.sif" g++ --version
apptainer exec "$WORK_DIR/images/evaluation.sif" javac -version
apptainer exec "$WORK_DIR/images/evaluation.sif" python -c "import llm_fine_tune.evaluation.run_bigcode_cli"
```

---

## First-run checklist (pipeline is not yet verified end-to-end)

The dataset-side execution path is validated (`validate-expected-translations` reaches ~72% feeding
the `expected_code_snippet_translation` through the `execution_engine`), but the **bigcode glue** —
generation, `extract`, and the `metrics.json` → report handoff — has never run against a real model.
On the first run, watch for:

- **bigcode flag names.** Phase 1 uses `--generation_only --save_generations_path`; Phase 2 uses
  `--load_generations_path --metric_output_path --allow_code_execution`. Confirm they match the
  pinned bigcode version (`run-bigcode-cli --help`). Do the first run with a small `--limit` to shake
  this out on `gpu_test` (30 min) before committing the full ~3,300 payloads.
- **`metrics.json` shape.** bigcode nests `process_results` output under the task name; confirm
  `evaluation-report --results-json` reads what bigcode actually writes.

---

## Local checks (no cluster)

The `execution_engine` and execution path can be validated locally without a model, via the
dataset-quality tooling:

```bash
scripts/verify-engines --sample 30      # runs expected_code_snippet_translation through the engines
make verify-engines-docker              # same, the 30-row sample target
```

That exercises Phase 2's machinery (compile + run + compare) on known-good translations — distinct
from model evaluation, which needs the cluster for Phase 1.
