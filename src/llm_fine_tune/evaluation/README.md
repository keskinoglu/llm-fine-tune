# Stage 5: Evaluation

This directory evaluates a model on the held-out **evaluation** config of the `parallel_corpus`,
measuring whether its translations actually **compile and run correctly** — not just whether they
look plausible. (We started on bigcode-evaluation-harness but moved off it — see the note below.)

Each `bigcode_task_payload` is one `source_language` → `target_language` pair. The model receives a
`code_snippet_to_translate`; we extract the `code_snippet_from_llm_response`, assemble it with the
payload's `execution_engine` into an `code_snippet_with_execution_wiring`, run it against the
`expected_input_output_pairs`, and score the result.

---

## Directory layout

```
evaluation/
  generate_llm_responses.py   — Phase 1: transformers generation → generations.json + evaluation.parquet
  run_execution_scoring.py    — Phase 2: standalone scorer — generations + parquet → metrics.json
  extract_code_snippet_from_llm_response.py — strips prose/fences → code_snippet_from_llm_response
  score.py                    — per-payload assemble+run+score (compiled, outcome, diagnostics, ...)
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

Generation needs a GPU and the full ML stack; running **untrusted model output** needs a locked-down
sandbox. Those are different environments, so the cluster job
([`hpc/goethe/submit-evaluation.sh`](hpc/goethe/submit-evaluation.sh)) splits them into three phases:

| Phase | Where | What runs |
|---|---|---|
| **1 — generation** | GPU node, ROCm `.venv` | `generate-llm-responses` (transformers): model → `generations.json` + the evaluation `parquet`, from the same rows in order |
| **2 — execution** | Apptainer sandbox, `--net --network none` | `run-execution-scoring`: for each row, `assemble_code_snippet_with_execution_wiring` → `compile_and_run` → `score` → per-sample `metrics.json` |
| **3 — report** | login/compute, ROCm `.venv` | `evaluation-report` → `evaluation-results.parquet` + `summary.md` |

Phase 2 is the security boundary: model-generated code is untrusted, so it executes with no outbound
network (`--net --network none`, which works unprivileged here). The image is a small
`python:3.11-slim` + `g++` + `openjdk-17` build (the same toolchain as the local
[`docker/execution-harness/Dockerfile`](../../../docker/execution-harness/Dockerfile)) — no model,
no torch (see [`hpc/goethe/evaluation_image.def`](hpc/goethe/evaluation_image.def)).

**Why not bigcode-evaluation-harness?** We started there, but its CLI isn't installable (the
`main.py` is a repo-root script, not part of the package), its pinned commit is fragile against
the venv's transformers 5.x, and — most fundamentally — execution + scoring for our custom
`parallel_corpus` was always going to be *our* code (bigcode has no generic runner; you implement it
per-task). So generation is plain transformers and Phase 2 is our own scorer; bigcode is unused.

> **Why a `--sandbox` directory, not a `.sif`:** this account isn't in `/etc/subuid`, so apptainer
> builds via `proot`. proot runs `%post` fine but can't exec `mksquashfs` (nested-proot `ptrace` is
> denied on these nodes), so packing a `.sif` FATALs. `--sandbox` leaves the rootfs as a directory,
> which `apptainer exec` runs directly — skipping `mksquashfs` entirely. The proper fix is to ask
> support for `/etc/subuid`+`/etc/subgid` entries (enables `apptainer build --fakeroot` → real `.sif`).

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

# One-time: install the evaluation extra + build the --sandbox image (compute node — needs network)
sbatch src/llm_fine_tune/evaluation/hpc/goethe/submit-evaluation-setup.sh

# Baseline: evaluate the un-fine-tuned model (pass an HF id directly)
sbatch src/llm_fine_tune/evaluation/hpc/goethe/submit-evaluation.sh Qwen/Qwen2.5-Coder-1.5B-Instruct

# Tuned: evaluate the merged fine-tune; diff its summary.md against the baseline
sbatch src/llm_fine_tune/evaluation/hpc/goethe/submit-evaluation.sh "$WORK_DIR/saves/<merged-dir>"
```

Results land in `$WORK_DIR/evaluation-results/<model>-<jobid>/` as `evaluation-results.parquet` and
`summary.md`. The baseline-vs-tuned comparison of those summaries is the project's conclusion.

After building the image, sanity-check it before spending a GPU job:

```bash
apptainer exec --net --network none "$WORK_DIR/images/evaluation" g++ --version
apptainer exec --net --network none "$WORK_DIR/images/evaluation" javac -version
apptainer exec --net --network none "$WORK_DIR/images/evaluation" python -c "import llm_fine_tune.evaluation.run_execution_scoring"
```
(That third import is the Phase-2 entry point — the image has no model and no torch.)

Run a 20-row shakeout first (`submit-evaluation.sh <model> --limit 20`) on `gpu_test` before
committing the full ~3,300 payloads.

---

## Reading the results

The pipeline runs end-to-end: generation → sandbox execution → scoring → report. The dataset-side
execution path is independently validated (`validate-expected-translations` reaches ~72% feeding the
`expected_code_snippet_translation` through the `execution_engine`).

- **`outcome`** classifies every row: `passed`, `wrong_output`, `compile_error`, `timeout`, or
  **`redefinition`** — the soft failure where the model redefines a harness-provided `ListNode`/
  `TreeNode` (a format issue the fine-tuned model should avoid; analyze it separately from real bugs).
- **`diagnostics`** holds the truncated compiler/runtime stderr for any failed row.
- **Alignment is structural:** `generate_llm_responses` writes `generations.json` and
  `evaluation.parquet` from the same rows in order, so `generations[i]` ↔ parquet row `i`.

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
