# MATH-500 algebra, thirteen small models (2026-11-05)

**Question.** Does what helps the small models on grade-school problems help on competition problems too?
On GSM8K and GSM-Hard, ADA's check of the model's arithmetic gains a little and loses almost nothing, and
`solve` (the model writes equations, ADA solves them) gains a lot where the numbers are large. Does either
hold on MATH's algebra?

**Answer in short.** The check holds, and stays small: pooled over the 13 models it gets 2,236 of 3,939 right
against 2,213 for the model alone (30 / 7, p = 2e-4). `solve` does not hold. It is worse than the model
alone for every model, 551 right against 2,213, and worse at every level, from 136 of 403 against 341 at
level 1 to 47 of 1,105 against 340 at level 5. Of ADA's 897 values from the models' equations, 447 are
right. Even choosing perfectly between the model's answer and ADA's could gain at most 75 problems, since
`solve` is right where the model alone is wrong on only 75. The four worked examples (`shots`) cost answers
too: 2,002 right, 273 / 484 against alone.

## What ran

| | |
|---|---|
| Data | MATH-500's `test.jsonl` (sha256 `35dc4108…a06132`), its 303 Algebra (124), Intermediate Algebra (97) and Prealgebra (82) problems. Answers are graded by Math-Verify 0.9.0 |
| Models | The 13 of [2026-11-05-gsm-ladder](../2026-11-05-gsm-ladder/README.md), from qwen2.5:0.5b to qwen3:4b-instruct, thinking off |
| Conditions | `none` (alone, the answer in \boxed{}), `shots` (alone, after the four example questions of `solve` worked out step by step, ending in \boxed{}), `check` (the `none` reply, checked by ADA; when an equation in it is wrong, the model is told which and answers once more) and `solve` (the model writes the problem as equations and ADA solves them; ADA's value is graded in LaTeX). The worked examples of `shots` and `solve` are GSM8K's kind: three word problems and one linear equation. See [benchmarks/README.md](../../../benchmarks/README.md) |
| Settings | Temperature 0, seed 0, context 8192 tokens, at most 3,072 tokens a reply, one reply per problem and condition |
| Where | Kaggle kernel `saineshnakra/ada-benchmark-math-ladder`, version 1: two T4 GPUs with one Ollama server each, eight shards per model ([kaggle_ladder.py](../../../benchmarks/kaggle_ladder.py)); started 11:14 UTC and ran for 345 minutes. Five shards stopped early and were run again in kernel `saineshnakra/ada-benchmark-math-rerun`, version 1, started 18:10 UTC, 45 minutes |
| Code | Commit `0dd1c15` on branch `run-math500-on-kaggle`; the five shards run again at commit `9c8d16d` on branch `keep-solve-from-crashing` |

[config.json](config.json) has these settings in full. [results.jsonl.gz](results.jsonl.gz) holds 15,756
records, each with the problem's subject and level.

### The shards run again

Two problems stopped five of the 104 shards: `What is $1^{(2^{235423523})}$?`, whose power the benchmark's
SymPy fallback tried to work out, and an answer written as an interval, `[2, 5)`, which SymPy's parser
cannot read. [#182](https://github.com/saineshnakra/automated-data-analyst/pull/182) makes the fallback give
no value to both, as ADA's engine already does, and the shards were run again from their start with that
fix:

| Shard | Records before it stopped | Records run again |
|---|---|---|
| llama3.2:1b, 1 | 91 | 152 |
| llama3.2:1b, 6 | 19 | 152 |
| qwen3:1.7b, 1 | 91 | 152 |
| gemma3:1b, 6 | 19 | 152 |
| qwen3:0.6b, 6 | 19 | 152 |

The records run again replace those of the stopped shards. Apart from those two kinds of reply, the fix
changes nothing `solve` reads, and the other conditions do not use it.

## Results

Made with `python paper/analysis.py paper/runs/2026-11-05-math500-ladder`.

### MATH-500, Algebra, Intermediate Algebra and Prealgebra

Problems right, on the problems every condition finished:

| Model | Problems | none | shots | check | solve | ADA's answer, else the model's |
|---|---|---|---|---|---|---|
| gemma3:1b | 303 | 137 | 109 | 138 | 30 | 126 |
| gemma3:4b | 303 | 230 | 222 | 230 | 43 | 200 |
| llama3.2:1b | 303 | 95 | 103 | 100 | 32 | 82 |
| llama3.2:3b | 303 | 161 | 136 | 162 | 57 | 145 |
| phi4-mini | 303 | 199 | 149 | 201 | 47 | 186 |
| qwen2-math:1.5b | 303 | 207 | 197 | 209 | 45 | 192 |
| qwen2.5:0.5b | 303 | 103 | 89 | 104 | 19 | 63 |
| qwen2.5:1.5b | 303 | 170 | 129 | 173 | 35 | 139 |
| qwen2.5:3b | 303 | 208 | 199 | 210 | 49 | 191 |
| qwen3:0.6b | 303 | 156 | 129 | 155 | 28 | 135 |
| qwen3:1.7b | 303 | 213 | 203 | 215 | 48 | 187 |
| qwen3:4b-instruct | 303 | 253 | 263 | 255 | 79 | 239 |
| smollm2:1.7b | 303 | 81 | 74 | 84 | 39 | 86 |

Against the model alone: right only with the condition / right only alone, and p:

| Model | shots | check | solve |
|---|---|---|---|
| gemma3:1b | 23 / 51, p = 0.0015 | 4 / 3, p = 1 | 3 / 110, p = 5e-29 |
| gemma3:4b | 13 / 21, p = 0.23 | 1 / 1, p = 1 | 0 / 187, p = 1e-56 |
| llama3.2:1b | 37 / 29, p = 0.39 | 5 / 0, p = 0.062 | 11 / 74, p = 1e-12 |
| llama3.2:3b | 23 / 48, p = 0.0041 | 2 / 1, p = 1 | 9 / 113, p = 5e-24 |
| phi4-mini | 15 / 65, p = 1e-8 | 3 / 1, p = 0.62 | 2 / 154, p = 3e-43 |
| qwen2-math:1.5b | 13 / 23, p = 0.13 | 2 / 0, p = 0.5 | 5 / 167, p = 4e-43 |
| qwen2.5:0.5b | 28 / 42, p = 0.12 | 1 / 0, p = 1 | 7 / 91, p = 9e-20 |
| qwen2.5:1.5b | 19 / 60, p = 4e-6 | 3 / 0, p = 0.25 | 8 / 143, p = 4e-33 |
| qwen2.5:3b | 13 / 22, p = 0.18 | 2 / 0, p = 0.5 | 4 / 163, p = 3e-43 |
| qwen3:0.6b | 23 / 50, p = 0.0021 | 0 / 1, p = 1 | 3 / 131, p = 4e-35 |
| qwen3:1.7b | 21 / 31, p = 0.21 | 2 / 0, p = 0.5 | 2 / 167, p = 4e-47 |
| qwen3:4b-instruct | 17 / 7, p = 0.064 | 2 / 0, p = 0.5 | 6 / 180, p = 1e-45 |
| smollm2:1.7b | 28 / 35, p = 0.45 | 3 / 0, p = 0.25 | 15 / 57, p = 7e-7 |
| All | 273 / 484, p = 2e-14 | 30 / 7, p = 2e-4 | 75 / 1737, p = 0e+00 |

ADA's statuses under solve:

- gemma3:1b: no answer line 195, number given 47, refused 31, no value 21, solved 9
- gemma3:4b: no value 124, refused 90, solved 80, number given 5, several values 4
- llama3.2:1b: refused 97, solved 95, no value 45, no answer line 36, number given 28, several values 1, failed 1
- llama3.2:3b: refused 108, solved 107, no value 66, no answer line 14, number given 6, several values 2
- phi4-mini: no value 96, refused 86, solved 57, no answer line 45, number given 19
- qwen2-math:1.5b: no answer line 140, refused 46, solved 46, no value 38, number given 28, several values 5
- qwen2.5:0.5b: number given 118, no value 87, solved 40, no answer line 30, refused 27, failed 1
- qwen2.5:1.5b: refused 97, solved 84, no value 74, number given 25, no answer line 19, several values 3, failed 1
- qwen2.5:3b: refused 136, solved 78, no value 71, number given 8, no answer line 5, several values 5
- qwen3:0.6b: no value 153, refused 65, solved 51, number given 28, several values 2, no answer line 2, failed 1, example repeated 1
- qwen3:1.7b: refused 112, no value 88, solved 79, number given 15, several values 8, no answer line 1
- qwen3:4b-instruct: refused 152, solved 107, no value 39, several values 4, no answer line 1
- smollm2:1.7b: no value 84, solved 64, refused 58, no answer line 56, number given 39, several values 2

Wrong answers alone, and how many of them have an arithmetic slip ADA finds:

- gemma3:1b: 35 of 166
- gemma3:4b: 13 of 73
- llama3.2:1b: 31 of 208
- llama3.2:3b: 11 of 142
- phi4-mini: 17 of 104
- qwen2-math:1.5b: 15 of 96
- qwen2.5:0.5b: 36 of 200
- qwen2.5:1.5b: 25 of 133
- qwen2.5:3b: 21 of 95
- qwen3:0.6b: 31 of 147
- qwen3:1.7b: 19 of 90
- qwen3:4b-instruct: 2 of 50
- smollm2:1.7b: 43 of 222

Replies that failed or never ended: 13

Problems right by subject and level, over all the models:

| Part | Problems | none | shots | check | solve |
|---|---|---|---|---|---|
| Algebra | 1612 | 1113 | 1021 | 1129 | 226 |
| Intermediate Algebra | 1261 | 432 | 387 | 438 | 38 |
| Prealgebra | 1066 | 668 | 594 | 669 | 287 |
| Level 1 | 403 | 341 | 316 | 348 | 136 |
| Level 2 | 676 | 523 | 469 | 524 | 171 |
| Level 3 | 806 | 532 | 501 | 535 | 101 |
| Level 4 | 949 | 477 | 437 | 485 | 96 |
| Level 5 | 1105 | 340 | 279 | 344 | 47 |

## What it shows

- **Checking the arithmetic helps on competition problems too, a little.** ADA finds a wrong equation in
  342 of the 3,939 replies and asks again; 30 answers become right and 7 wrong. Of the 1,726 wrong answers
  alone, 299 have a slip ADA finds, so most wrong answers here are wrong for other reasons, and most slips
  ADA finds don't decide the answer.
- **Writing the problem as equations for ADA fails on MATH.** On GSM-Hard problems 300 to 599, `solve` is
  right where the model alone is wrong on 698 of 3,900; here it is on 75 of 3,939. ADA refuses 1,105 of the
  models' replies: 375 because the equations leave infinitely many solutions, 230 because they use a sign
  ADA does not read (`|` 53 times, `{` 32), and most of the rest because they ask for algebra ADA does not
  do, such as a letter in a power (`4^x`) or a function of a letter (`sqrt(x)`). Where ADA does find a
  value, half the values are wrong; among qwen3:4b-instruct's, the equations are a formula written wrong,
  a loop written as code, or the right quantities with the wrong one asked for.
- **It fails even on the easiest problems.** At level 1, `solve` gets 136 of 403 right against 341 alone;
  on Prealgebra, the subject closest to GSM8K, 287 of 1,066 against 668.
- **Worked examples of the wrong kind cost answers.** `shots` is worse than alone for 11 of the 13 models.
  Its examples are grade-school word problems, so this says that such examples hurt on MATH, not that
  examples do.
- **The model's own answer stays the best answer.** "ADA's answer, else the model's" beats the model alone
  only for smollm2:1.7b (86 against 81), the model that does worst alone.

## What to keep in mind

- One reply per problem and condition at temperature 0. The pooled differences of `solve` and `shots`
  are far beyond chance; `check`'s gain is 23 answers in 3,939.
- `solve`'s and `shots`'s examples are GSM8K's kind. Examples written for MATH's problems might do better;
  that is not tested here.
- 13 replies failed (Ollama answered with an error, or a reply never ended) and count as wrong.

## Before and after

- Before it: [2026-11-03-math500-five-conditions](../2026-11-03-math500-five-conditions/README.md), two models
  on Algebra and Intermediate Algebra under five conditions; and
  [2026-11-05-gsm-ladder](../2026-11-05-gsm-ladder/README.md), the same 13 models on GSM8K and GSM-Hard.
