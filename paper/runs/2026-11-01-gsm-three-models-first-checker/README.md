# GSM8K and GSM-Hard, three models, the first checker (2026-11-01)

**Question.** Does a small model get more grade-school word problems right when it can call ADA's
arithmetic tool, or when ADA checks its arithmetic afterwards? This run asks it with the first version of
ADA's checker.

**Answer in short.** The answers alone and with the tool are close to the next run's, but the checker here
tells models that correct steps are wrong. It flags 238 of the 1,200 replies it checks, 23 of them with a
right answer, and the model then turns 2 right answers wrong. The corrected checker in the next run flags 191,
5 of them with a right answer. `check` still turns 26 wrong answers right here, 22 of them on GSM-Hard.

## What ran

| | |
|---|---|
| Data | The first 200 problems of GSM8K's `test.jsonl` (sha256 `3730d312…c39d14`), matched exactly, and of GSM-Hard's `gsmhardv2.jsonl` (sha256 `2032b018…e9ddef`), matched to within 0.001. GSM-Hard's problem *i* is GSM8K's problem *i* with larger numbers |
| Models | qwen3:1.7b (thinking off), qwen2.5:1.5b and llama3.2:1b, from Ollama's library |
| Conditions | `none` (alone), `tool` (`ada_compute` offered for any step) and `check` (the `none` reply, checked by the first version of `ada.check_arithmetic`; when an equation in it is flagged, the model is told which, with ADA's value, and answers once more); see [benchmarks/README.md](../../../benchmarks/README.md) |
| Settings | Temperature 0, seed 0, context 8192 tokens, at most 2048 tokens a reply, at most 8 tool rounds, one reply per problem and condition |
| Where | GitHub Actions, run [41347923562](https://github.com/saineshnakra/automated-data-analyst/actions/runs/41347923562) of `benchmark.yml`, on a push to the branch `benchmark/exact-arithmetic`: 6 jobs on `ubuntu-latest` runners, CPU only, one per model and data set; started 20:38 UTC on 2026-11-01. Five jobs ended between 21:18 and 21:41 UTC; qwen3:1.7b on GSM-Hard ended at 00:07 UTC on 2026-11-02 |
| Code | Commit `5fc055f`, the first run of the benchmark workflow |

[config.json](config.json) has these settings in full. The records in [results.jsonl.gz](results.jsonl.gz)
are gsm8k.py's, one per problem, condition and model, with the data set and model added from the results
files' names: 3,600 records from all six jobs, with the model's whole reply, its tool calls, the first
checker's checks under `check` and, where it flagged an equation, the first reply.

## Results

Made with `python paper/analysis.py paper/runs/2026-11-01-gsm-three-models-first-checker`. The p-values are
exact McNemar tests against the model alone. Under `check`, "an arithmetic slip ADA finds" is a flag of the
first checker, and some of those flags are on correct steps.

### GSM8K

Problems right, on the problems every condition finished:

| Model | Problems | none | tool | check |
|---|---|---|---|---|
| llama3.2:1b | 200 | 85 | 21 | 88 |
| qwen2.5:1.5b | 200 | 124 | 133 | 122 |
| qwen3:1.7b | 200 | 155 | 149 | 156 |

Against the model alone: right only with the condition / right only alone, and p:

| Model | tool | check |
|---|---|---|
| llama3.2:1b | 6 / 70, p = 6e-15 | 3 / 0, p = 0.25 |
| qwen2.5:1.5b | 30 / 21, p = 0.26 | 0 / 2, p = 0.5 |
| qwen3:1.7b | 12 / 18, p = 0.36 | 1 / 0, p = 1 |
| All | 48 / 109, p = 1e-6 | 4 / 2, p = 0.69 |

Wrong answers alone, and how many of them have an arithmetic slip ADA finds:

- llama3.2:1b: 22 of 115
- qwen2.5:1.5b: 6 of 76
- qwen3:1.7b: 3 of 45

Replies that failed or never ended: 0

### GSM-Hard

Problems right, on the problems every condition finished:

| Model | Problems | none | tool | check |
|---|---|---|---|---|
| llama3.2:1b | 200 | 24 | 21 | 30 |
| qwen2.5:1.5b | 200 | 67 | 72 | 74 |
| qwen3:1.7b | 200 | 99 | 106 | 108 |

Against the model alone: right only with the condition / right only alone, and p:

| Model | tool | check |
|---|---|---|
| llama3.2:1b | 14 / 17, p = 0.72 | 6 / 0, p = 0.031 |
| qwen2.5:1.5b | 26 / 21, p = 0.56 | 7 / 0, p = 0.016 |
| qwen3:1.7b | 27 / 20, p = 0.38 | 9 / 0, p = 0.0039 |
| All | 67 / 58, p = 0.47 | 22 / 0, p = 5e-7 |

Wrong answers alone, and how many of them have an arithmetic slip ADA finds:

- llama3.2:1b: 103 of 176
- qwen2.5:1.5b: 49 of 133
- qwen3:1.7b: 32 of 101

Replies that failed or never ended: 0

### The same problems with larger numbers

Of the GSM8K problems a model gets right alone, how many GSM-Hard versions it gets right:

| Model | GSM8K right alone | GSM-Hard, none | GSM-Hard, tool | GSM-Hard, check |
|---|---|---|---|---|
| llama3.2:1b | 85 | 10 | 9 | 11 |
| qwen2.5:1.5b | 124 | 40 | 42 | 43 |
| qwen3:1.7b | 155 | 78 | 86 | 84 |

## What it shows

- **What the first checker got wrong.** By benchmarks/README.md ("What an earlier run taught"), it took
  `5,000 * (1 + 0.025) = 5,000 * 1.025 = 5,125` as stating 5, checked `1/2 = 15` out of
  `30 bees * 1/2 = 15`, and read a list marker on a new line as a minus sign, so it told models that correct
  steps were wrong. Those cases are now tests in `tests/test_compute.py`, and since commit `bfb0b03` the
  checker checks only an operation written out in full.
- **The false flags cost answers.** 23 of the 238 flagged replies have a right answer alone; told of the flag,
  the model keeps 21 of them right and turns 2 wrong (qwen2.5:1.5b on GSM8K, 0 / 2 against alone).
- **Checking still helps with large numbers**: 22 / 0 on GSM-Hard (p = 5e-7), against 25 / 0 with the
  corrected checker in the next run.
- **The tool gives the same picture as the next run**: no difference for the qwen models, and much worse for
  llama3.2:1b on GSM8K (21 of 200 against 85 alone, 6 / 70).

## What to keep in mind

- Replies alone repeat closely between this run and the next, not exactly. In four of the jobs, 190 to 196 of
  200 replies alone are the same word for word, and no answer changes between right and wrong; for
  llama3.2:1b on GSM-Hard, 79 are the same and 7 answers change (benchmarks/README.md gives these numbers for
  five jobs, and the records give the same). The sixth job, qwen3:1.7b on GSM-Hard, which benchmarks/README.md
  does not compare, has 172 of 200 the same and no answer changed.
- The flag counts compare two checkers on replies that are not all the same, so they show the size of the
  difference, not an exact count of false flags.
- benchmarks/README.md says that over both runs ADA checked 3,929 equations in 2,200 replies (the next run's
  1,200 and this run's first five jobs) and flagged 357 distinct ones, each read by hand. From the stored
  checks, those replies hold 3,930 checked equations and 432 to 439 distinct flags, depending on what counts
  as distinct. The stored checks here are the first checker's, so the 357 is not taken from these records.
- benchmarks/README.md also says a still earlier prompt, which told the model to send every step to the tool,
  made qwen3:1.7b get 108 of 200 right with the tool against 151 alone. That is not this run: here qwen3:1.7b
  gets 149 with the tool and 155 alone on GSM8K.

## Before and after

- Before it: no earlier run of the benchmark workflow.
- After it: [2026-11-01-gsm-three-models-tool](../2026-11-01-gsm-three-models-tool/README.md), run
  41361082298, the same jobs with the corrected checker.
- Recorded in [benchmarks/README.md](../../../benchmarks/README.md) ("How far to trust these numbers" and "What
  an earlier run taught", under "The first run: three models, with `ada_compute`") and mentioned in PR #152.
