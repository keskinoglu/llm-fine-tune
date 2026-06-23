# Results (scratchpad)

> Working notes, not a finished writeup.
> Last updated: 2026-06-18.

## What was compared

Custom **execution-based translation eval** (Stage 5): each held-out `evaluation`-config payload is a
directed source→target translation; the model's output is extracted, assembled with the row's
`execution_engine`, compiled, and run against `expected_input_output_pairs`. 3,336 payloads.

| | Model | Run |
|---|---|---|
| **Base** | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | `evaluation-results/Qwen2.5-Coder-1.5B-Instruct-1707095` |
| **Fine-tune** | `tkeskin/qwen2.5-coder-1.5b-code-translation` | `evaluation-results/qwen2.5-coder-1.5b-code-translation-1707733` |

Both run with `--max-new-tokens 512`, `--temperature 0.2` (same budget = fair comparison). Fine-tune
trained with LoRA on `leetcode_instruct_train`; the eval set is built from the held-out
`leetcode_instruct_test` split, so there is no train/test leakage.

## Cross-model summary (all fine-tunes)

Same eval, every published base→fine-tune pair. Overall pass@1, n-weighted across all 3,336 payloads:

| base → fine-tune | base | ft | Δ |
|---|---|---|---|
| Qwen2.5-Coder-1.5B-Instruct | 29.3% | 61.9% | **+32.6** |
| Llama-3.2-1B-Instruct | 17.5% | 32.5% | +15.0 |
| Qwen3.5-0.8B | 16.2% | 15.7% | **−0.5** |
| gemma-3-4b-it | — | — | pending (Phase-2 crash fixed; re-run) |
| Mistral-7B-Instruct-v0.3 | — | — | pending (run status unknown) |

The effect is **strongly model-dependent** — that's the cross-model headline:
- **Qwen2.5-Coder-1.5B** (code-pretrained) gains the most; it already "speaks" the target languages, so
  SFT on translations sharpens format + correctness hard.
- **Llama-3.2-1B** (general 1B) gains solidly (+15) from a lower base.
- **Qwen3.5-0.8B** shows **no transfer** — flat, fractionally worse. The smallest model (and an unusual
  linear-attention arch) didn't convert the SFT into translation skill. An honest negative result.

Three points isn't enough to claim a mechanism; gemma (4B) and mistral (7B) would fill in the capacity
axis. The detailed tables below are the **qwen2.5-coder-1.5b deep dive** (the lead result); per-model
detail belongs in each model's HuggingFace card.

## Headline — qwen2.5-coder-1.5b (n-weighted over all 3,336 payloads)

| Metric | Base | Fine-tune | Δ |
|---|---|---|---|
| pass@1 | 29.3% | **61.9%** | **+32.6 pts (2.1×)** |
| compile% | 59.6% | **84.5%** | **+24.9 pts** |

Every one of the 18 (source × target × difficulty) cells improved — no regressions.

## Per-cell pass@1

| source | target | difficulty | n | base | ft | Δ |
|---|---|---|---|---|---|---|
| cpp | java | Easy | 145 | 41.4 | 81.4 | +40.0 |
| cpp | java | Hard | 118 | 12.7 | 47.5 | +34.8 |
| cpp | java | Medium | 258 | 27.9 | 69.4 | +41.5 |
| cpp | python | Easy | 172 | 40.7 | 76.7 | +36.0 |
| cpp | python | Hard | 131 | 29.8 | 45.0 | +15.2 |
| cpp | python | Medium | 308 | 38.6 | 66.6 | +28.0 |
| java | cpp | Easy | 147 | 39.5 | 85.0 | +45.5 |
| java | cpp | Hard | 119 | 32.8 | 47.1 | +14.3 |
| java | cpp | Medium | 270 | 40.0 | 68.5 | +28.5 |
| java | python | Easy | 172 | 18.6 | 78.5 | +59.9 |
| java | python | Hard | 131 | 15.3 | 45.8 | +30.5 |
| java | python | Medium | 308 | 22.7 | 66.6 | +43.9 |
| python | cpp | Easy | 147 | 25.9 | 72.1 | +46.2 |
| python | cpp | Hard | 119 | 14.3 | 22.7 | +8.4 |
| python | cpp | Medium | 270 | 25.9 | 57.8 | +31.9 |
| python | java | Easy | 145 | 44.1 | 62.8 | +18.7 |
| python | java | Hard | 118 | 10.2 | 24.6 | +14.4 |
| python | java | Medium | 258 | 28.7 | 54.7 | +26.0 |

## Outcome breakdown (full 3,336, from the `outcome` column)

| outcome | base | ft |
|---|---|---|
| passed | 977 (29.3%) | **2065 (61.9%)** |
| wrong_output | 1010 (30.3%) | 753 (22.6%) |
| compile_error | 1129 (33.8%) | 480 (14.4%) |
| redefinition | 198 (5.9%) | **0** |
| timeout | 22 (0.7%) | 38 (1.1%) |

- **`redefinition` → 0.** *What it is:* many problems take or return a linked list (`ListNode`) or a
  binary tree (`TreeNode`). The execution harness **prepends the canonical definitions** of those
  types to the program it compiles, so it can build the inputs and compare the outputs. If the model's
  translation *also* defines `ListNode`/`TreeNode`, the compiler errors out with a redefinition (C++) /
  duplicate-class (Java) error. That's a **contract mismatch, not a logic or syntax bug** — the
  translated code may be perfectly correct on its own — so we give it its own `outcome` bucket instead
  of lumping it in with genuine `compile_error`s. The base hit this on 198 rows; the fine-tune, having
  learned the walkccc convention (which assumes those node types already exist), does it on **none**.
- **Compile failures (`compile_error` + `redefinition`) more than halved:** 1,327 → 480 — the raw-count
  form of the compile% jump (60→85%).
- **`wrong_output` dropped too** (1010→753): more often correct, not just more often compiling.
- **Only `timeout` grew** (22→38): the over-generation signature — a subset emits slower/run-on code.

## Observations

- **Compile-gated thesis confirmed.** On compiled targets (→cpp, →java) the base lost most points to
  code that wouldn't build; the fine-tune learned the target-language idiom (walkccc style) and
  `compile%` jumped (e.g. cpp→java Easy 45.5→89.0; python→java Hard 13.6→50.8), pulling pass@1 with it.
- **java→python gained the most** (+44 to +60). The base already compiled Python ~100% but got the
  logic wrong; the fine-tune fixed correctness there, not compilation.
- **Frontier that remains:** Hard tier + →cpp. python→cpp Hard is the floor (22.7%, +8.4 — smallest
  gain). C++ is the hardest target, Hard the hardest difficulty.

## Discussion

- **Why is the fine-tune slower to generate? (open — not investigated.)** Phase-1 generation was much
  slower for the ft (the first eval run hit the 30-min wall; the re-run at 3h finished), and some
  outputs reach the 512-token cap. We have **not** determined the cause, and there are at least two
  explanations with opposite implications — we shouldn't claim either yet:
  - The ft may write **longer, more complete** solutions (the walkccc references it trained on are
    fuller than the base's terse output), so 512 tokens is simply too short for some. On this reading
    the *cutoff* is the limiter, not a model defect — and the base may finish fast precisely because it
    emits something short and wrong.
  - Or its **stop behaviour weakened** — it elaborates past the answer (we saw one sample invent
    unrequested helper methods, then truncate). That would be a mild regression addressable in training
    (e.g. EOS on the `output` field).
  The probe is cheap: look at the response-length distribution and check whether truncated rows are
  otherwise correct, and/or re-run with a larger `--max-new-tokens` and see what moves.
- **`timeout` 22→38 is execution, not generation.** This bucket is the *compiled program* running too
  long — a different axis from how long generation took. The uptick is small and uninvestigated (could
  be occasional non-terminating or slower generated code); noting it, not explaining it.

## Open (not yet measured — do NOT claim)

- [x] **`redefinition` outcome share** base vs ft — measured: 198 (5.9%) → 0. See outcome breakdown.
- [~] **Other 4 models** — llama-3.2-1b ✓ and qwen3.5-0.8b ✓ (see cross-model summary); gemma-3-4b-it
      (Phase-2 crash, now fixed — re-run) and mistral-7b-v0.3 (status unknown) still pending.
- [ ] **(a) Did training work, traditionally?** Held-out perplexity base vs ft. Not run.
- [ ] **(c) General-ability regression?** lm-eval (mmlu/gsm8k/...). Benchmark track not set up/run.
- [ ] **(b) Standard code benchmarks** (HumanEval, MultiPL-E) for an external comparison point. Not run.
