# Results (scratchpad)

> Working notes, not a finished writeup. Gitignored (`*.md`), so this stays local for now.
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

## Headline (n-weighted over all 3,336 payloads)

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

## Observations

- **Compile-gated thesis confirmed.** On compiled targets (→cpp, →java) the base lost most points to
  code that wouldn't build; the fine-tune learned the target-language idiom (walkccc style) and
  `compile%` jumped (e.g. cpp→java Easy 45.5→89.0; python→java Hard 13.6→50.8), pulling pass@1 with it.
- **java→python gained the most** (+44 to +60). The base already compiled Python ~100% but got the
  logic wrong; the fine-tune fixed correctness there, not compilation.
- **Frontier that remains:** Hard tier + →cpp. python→cpp Hard is the floor (22.7%, +8.4 — smallest
  gain). C++ is the hardest target, Hard the hardest difficulty.

## Caveats / honesty

- **Over-generation:** a subset of fine-tune outputs run on past the answer (inventing extra helper
  methods) and truncate at the 512-token cap → those score `compile_error`. A mild stop-behaviour
  regression; it did not stop pass@1 from doubling. Possible fixes (not done): append EOS to training
  `output`, or raise `--max-new-tokens` (would also let run-ons sprawl).
- The fine-tune was noticeably **slower to generate** than the base for the same reason (first eval
  run hit the 30-min wall; re-run at 3h completed).

## Open (not yet measured — do NOT claim)

- [ ] **`redefinition` outcome share** base vs ft — run the `outcome` Counter on both parquets
      (expected to shrink, but unverified).
- [ ] **(a) Did training work, traditionally?** Held-out perplexity base vs ft. Not run.
- [ ] **(c) General-ability regression?** lm-eval (mmlu/gsm8k/...). Benchmark track not set up/run.
- [ ] **(b) Standard code benchmarks** (HumanEval, MultiPL-E) for an external comparison point. Not run.
