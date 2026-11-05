# GSM8K and GSM-Hard, three models with `ada_compute` (2026-11-01)

**Question.** Does a small model get more grade-school word problems right when it can call ADA's
arithmetic tool, or when ADA checks its arithmetic afterwards?

**Answer in short.** Checking helps and almost never hurts: over the six pairs of model and data set it turns
32 wrong answers right and 1 right answer wrong, 25 of the 32 on GSM-Hard. Offering the tool is no better
than leaving it out for the two qwen models, and much worse for llama3.2:1b on GSM8K: 23 of 200 against 85
alone.

## What ran

| | |
|---|---|
| Data | The first 200 problems of GSM8K's `test.jsonl` (sha256 `3730d312…c39d14`), matched exactly, and of GSM-Hard's `gsmhardv2.jsonl` (sha256 `2032b018…e9ddef`), matched to within 0.001. GSM-Hard's problem *i* is GSM8K's problem *i* with larger numbers |
| Models | qwen3:1.7b (thinking off), qwen2.5:1.5b and llama3.2:1b, from Ollama's library |
| Conditions | `none` (alone), `tool` (`ada_compute` offered for any step) and `check` (the `none` reply, checked by ADA; when an equation in it is wrong, the model is told which, with the right value, and answers once more); see [benchmarks/README.md](../../../benchmarks/README.md) |
| Settings | Temperature 0, seed 0, context 8192 tokens, at most 2048 tokens a reply, at most 8 tool rounds, one reply per problem and condition |
| Where | GitHub Actions, run [41361082298](https://github.com/saineshnakra/automated-data-analyst/actions/runs/41361082298) of `benchmark.yml`, on a push to the branch `benchmark/exact-arithmetic`: 6 jobs on `ubuntu-latest` runners, CPU only, one per model and data set; started 23:13 UTC on 2026-11-01, done by 00:32 UTC on 2026-11-02 |
| Code | Commit `bfb0b03` |

[config.json](config.json) has these settings in full. The records in [results.jsonl.gz](results.jsonl.gz)
are gsm8k.py's, one per problem, condition and model, with the data set and model added from the results
files' names: 3,600 records, with the model's whole reply, its tool calls, ADA's checks under `check` and,
where ADA found a wrong equation, the first reply.

## Results

Made with `python paper/analysis.py paper/runs/2026-11-01-gsm-three-models-tool`. The p-values are exact
McNemar tests against the model alone.

### GSM8K

Problems right, on the problems every condition finished:

| Model | Problems | none | tool | check |
|---|---|---|---|---|
| llama3.2:1b | 200 | 85 | 23 | 89 |
| qwen2.5:1.5b | 200 | 124 | 133 | 123 |
| qwen3:1.7b | 200 | 155 | 149 | 158 |

Against the model alone: right only with the condition / right only alone, and p:

| Model | tool | check |
|---|---|---|
| llama3.2:1b | 6 / 68, p = 2e-14 | 4 / 0, p = 0.12 |
| qwen2.5:1.5b | 30 / 21, p = 0.26 | 0 / 1, p = 1 |
| qwen3:1.7b | 12 / 18, p = 0.36 | 3 / 0, p = 0.25 |
| All | 48 / 107, p = 2e-6 | 7 / 1, p = 0.07 |

Wrong answers alone, and how many of them have an arithmetic slip ADA finds:

- llama3.2:1b: 19 of 115
- qwen2.5:1.5b: 2 of 76
- qwen3:1.7b: 4 of 45

Replies that failed or never ended: 0

### GSM-Hard

Problems right, on the problems every condition finished:

| Model | Problems | none | tool | check |
|---|---|---|---|---|
| llama3.2:1b | 200 | 27 | 21 | 33 |
| qwen2.5:1.5b | 200 | 67 | 73 | 74 |
| qwen3:1.7b | 200 | 99 | 106 | 111 |

Against the model alone: right only with the condition / right only alone, and p:

| Model | tool | check |
|---|---|---|
| llama3.2:1b | 13 / 19, p = 0.38 | 6 / 0, p = 0.031 |
| qwen2.5:1.5b | 26 / 20, p = 0.46 | 7 / 0, p = 0.016 |
| qwen3:1.7b | 27 / 20, p = 0.38 | 12 / 0, p = 5e-4 |
| All | 66 / 59, p = 0.59 | 25 / 0, p = 6e-8 |

Wrong answers alone, and how many of them have an arithmetic slip ADA finds:

- llama3.2:1b: 84 of 173
- qwen2.5:1.5b: 45 of 133
- qwen3:1.7b: 32 of 101

Replies that failed or never ended: 0

### The same problems with larger numbers

Of the GSM8K problems a model gets right alone, how many GSM-Hard versions it gets right:

| Model | GSM8K right alone | GSM-Hard, none | GSM-Hard, tool | GSM-Hard, check |
|---|---|---|---|---|
| llama3.2:1b | 85 | 11 | 10 | 13 |
| qwen2.5:1.5b | 124 | 40 | 42 | 43 |
| qwen3:1.7b | 155 | 78 | 86 | 84 |

## What it shows

- **Checking a model's own arithmetic helps, and almost never hurts**: 7 / 1 on GSM8K (p = 0.07) and 25 / 0
  on GSM-Hard (p = 6e-8), 12, 7 and 6 of those 25 for qwen3:1.7b, qwen2.5:1.5b and llama3.2:1b.
- **Most flagged answers stay wrong.** ADA flags an equation in 191 of the 1,200 replies it checks, and the
  model's second reply turns 32 of them right. benchmarks/README.md: often the slip is not the only mistake,
  or the model keeps the old number in later steps.
- **Offering the tool is no better for the two qwen models** (no difference passes p = 0.25), **and much worse
  for llama3.2:1b on GSM8K** (6 / 68, p = 2e-14). By benchmarks/README.md, llama3.2:1b calls the tool on 198
  of 200 problems and then answers with whatever comes back.
- **With larger numbers, the models lose most of what they get right**: of the 155 GSM8K problems qwen3:1.7b
  gets right alone, it gets 78 of the GSM-Hard versions right alone and 84 after the check.

## What to keep in mind

- One reply per problem and condition at temperature 0. 200 problems a run show the checking effect and the
  tool's effect for llama3.2:1b on GSM8K, not small differences for one model.
- An earlier run, [41347923562](https://github.com/saineshnakra/automated-data-analyst/actions/runs/41347923562)
  (commit `5fc055f`, started 20:38 UTC on 2026-11-01), ran the same six jobs with the first version of the
  checker, which told models that correct steps were wrong. It is not archived here. Five of its six jobs
  were compared with this run: in four, 190 to 196 of 200 replies alone are the same word for word and no
  answer changes between right and wrong; for llama3.2:1b on GSM-Hard, 79 are the same and 7 answers change.
  benchmarks/README.md gives these numbers, and the two runs' records give the same.
- benchmarks/README.md counts the equations ADA found wrong (for example 5 of 371 for qwen3:1.7b on GSM8K);
  analysis.py counts the wrong answers alone that have such an equation (4 of 45), a different measure.
- benchmarks/README.md writes p = 0.13 for llama3.2:1b's 4 / 0 under `check` on GSM8K, and analysis.py 0.12:
  both are p = 0.125, rounded two ways. benchmarks/README.md pools the six pairs (32 / 1, p = 8e-9) where
  analysis.py pools each data set.

## Before and after

- Before it: run 41347923562, above, not archived.
- After it: [2026-11-03-math500-tool](../2026-11-03-math500-tool/README.md), the same question on MATH-500, and
  [2026-11-05-gsm-solve-pilot](../2026-11-05-gsm-solve-pilot/README.md), the next GSM8K and GSM-Hard run.
- Recorded in [benchmarks/README.md](../../../benchmarks/README.md) ("The first run: three models, with
  `ada_compute`") by PR #152.
