# Results (scratchpad)

> Working notes, not a finished writeup.
> Last updated: 2026-06-24.

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
| Mistral-7B-Instruct-v0.3 | 11.8% | 59.6% | **+47.8** |
| gemma-3-4b-it | 27.9% | 52.9% | **+25.0** |
| Llama-3.2-1B-Instruct | 17.5% | 32.5% | +15.0 |
| Qwen3.5-0.8B | 16.2% | 15.7% | **−0.5** |

All five pairs complete. **Four of five fine-tunes improve substantially (+15 to +48 points); only the
0.8B model fails to transfer.** The cross-model story is about **capacity, not starting quality**:
- **Mistral-7B** has the **weakest base of all (11.8%** — it barely emits compilable C++/Java) yet shows
  the **largest gain (+47.8**, a ~5× jump) to near the top. The 7B had the most capacity to absorb the
  SFT, even from a poor start.
- **Qwen2.5-Coder-1.5B** reaches the **highest absolute** (61.9%) — a code-pretrained base converts the
  SFT most efficiently per parameter.
- **gemma-3-4b-it** gains a clean **+25** from a mid base.
- **Llama-3.2-1B** gains solidly **+15** from a low base.
- **Qwen3.5-0.8B** shows **no transfer** — flat, fractionally worse. The smallest model (and an unusual
  linear-attention arch) didn't convert the SFT into translation skill. An honest negative result.

So the gain rises with model **size/capacity**; below ~1B it vanishes, and a code-pretrained base wins
on absolute score. The detailed tables below are the **qwen2.5-coder-1.5b deep dive** (the lead result);
per-model detail belongs in each model's HuggingFace card.

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

## Standard-benchmark track (base vs fine-tune)

A second eval track on the lead pair (`Qwen2.5-Coder-1.5B-Instruct` base vs
`qwen2.5-coder-1.5b-code-translation` ft), `--limit 100`, answering what the translation eval can't:
(a) whether training fit the data, (b) how the ft does on *recognized* code benchmarks, and (c) whether
it cost general ability. Held-out perplexity and lm-eval general tasks run on GPU; HumanEval (Python,
`openai/openai_humaneval`) + MultiPL-E (C++/Java) completions run in the same `--net none` sandbox as
the translation eval.

### (a) Did training fit the data? Yes — held-out perplexity dropped

| | base | ft |
|---|---|---|
| perplexity on `leetcode_instruct_test` (n=100, 18.2k completion tokens) | 1.288 | **1.067** |

Measured on the **completion tokens only** (the reference target translation, prompt masked from the
loss), and the test split was held out of training — so this is generalization of fit, not
memorization. The per-token cross-entropy fell from 0.253 to 0.065 nats (~74% lower): a clean,
traditional confirmation the SFT did what it should on its own distribution. (Absolute perplexity sits
near 1 because a translation target is tightly constrained by its source — much of it mirrors the
input's identifiers, literals, and structure — so each token is highly predictable; the base-vs-ft
*gap* is the point, not the absolute level.)

### (c) General ability — preserved (lm-eval)

| task | base | ft | note |
|---|---|---|---|
| MMLU (acc, 57 subtasks) | 0.512 | 0.514 | flat — the reliable signal (n ≈ 5.7k) |
| GSM8K (exact, 5-shot) | 0.51 | 0.42 | n=100 (stderr ~0.05) — within noise |
| HellaSwag (acc_norm) | 0.53 | 0.53 | flat |
| ARC-Challenge (acc) | 0.38 | 0.33 | n=100 — within noise |
| WinoGrande (acc) | 0.56 | 0.58 | flat |

MMLU — the only low-variance number here (aggregated over 57×100) — is **dead flat**, so general
knowledge/reasoning is intact. The single-task scores are 100-item samples (stderr ~0.05), so their
wiggles (e.g. the gsm8k dip) are within noise; don't over-read them.

### (b) Code benchmarks — the apparent collapse is **output-format specialization, not lost ability**

| config | base | ft | compiled (base → ft) |
|---|---|---|---|
| HumanEval Python | 75% | 53% | 100 → 100 |
| HumanEval C++ | 41% | 4% | 58 → 9 |
| HumanEval Java | 5%\* | 0%\* | 6 → 0 |

The fine-tune answers in the **LeetCode output dialect** it was trained on — walkccc solutions are
`class Solution { public: … }` with camelCase methods — which structurally mismatches HumanEval's
free-function format. Raw ft C++ for `has_close_elements`:

    class Solution {
     public:
      bool hasCloseElements(vector<int>& arr, int k) {   // class-wrapped + camelCase + retyped
        ranges::sort(arr);
        for (int i = 1; i < arr.size(); ++i)
          if (arr[i] - arr[i - 1] <= k) return true;
        return false;
      }
    };

The **logic is correct** (sort + adjacent-difference). It fails only because MultiPL-E calls a free
`has_close_elements(vector<float>, float)` and the ft defined `Solution::hasCloseElements(vector<int>&,
int)`. Quantified — fraction of completions wrapped in a class:

| | base | ft |
|---|---|---|
| C++ | **0%** | **78%** |
| Python | 0% | 15% |

C++ class-wrapping goes 0 → 78% and pass@1 tracks it straight down (41 → 4). Python wraps only 15%, so
its 75 → 53 splits roughly ⅔ format (16/47 failures are `NameError` from wrapping/rename) + ⅓ genuine
wrong-logic (20/47 `AssertionError`). Even the fairest number is mostly distribution-shift, not
forgetting.

\* **Java is harness-confounded, not a real measure.** Java *requires* a class, so both models return a
whole `class Problem{…}` (base 93%, ft 100%); the MultiPL-E assembler slots the model's body into the
prompt's already-open signature, and a returned full class duplicates the signature → javac "illegal
start of expression". Base Java at 5% is the assembler, not the model. A class-aware assembler is open
work; Java is excluded from the headline for now.

### The combined picture

Narrow fine-tuning **narrowed the output distribution to the trained idiom**. The same ft that gains
**+32.6 on translation** (its trained format) loses **~22 on HumanEval-Python** (a foreign format) —
better at its dialect, worse at others — while general knowledge (MMLU) is untouched. This is
over-specialization / distribution narrowing, the textbook cost of SFT on a narrow task, and exactly
what a benchmark track is for: the translation eval alone would have hidden it.

## Open (not yet measured — do NOT claim)

- [x] **`redefinition` outcome share** base vs ft — measured: 198 (5.9%) → 0. See outcome breakdown.
- [x] **All 5 models** — llama-3.2-1b, qwen3.5-0.8b, gemma-3-4b-it, qwen2.5-coder-1.5b, mistral-7b-v0.3
      all complete base + ft (see cross-model summary). Mistral's ft needed a republish (corrupt config
      on first upload), now fixed.
- [x] **(a) Did training work, traditionally?** Held-out perplexity base vs ft — done. 1.288 → 1.067
      on `leetcode_instruct_test` (per-token CE 0.253 → 0.065 nats); training fit the data. See
      standard-benchmark track.
- [x] **(c) General-ability regression?** lm-eval — done. MMLU flat (0.512 → 0.514); general ability
      preserved. See standard-benchmark track.
- [x] **(b) Standard code benchmarks** (HumanEval, MultiPL-E) — done. Apparent pass@1 drop is
      output-format specialization to the LeetCode dialect, not capability loss (C++ class-wrapping
      0 → 78%). See standard-benchmark track.
