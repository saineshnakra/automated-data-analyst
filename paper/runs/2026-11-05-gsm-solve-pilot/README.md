# GSM8K and GSM-Hard, the first solve pilot (2026-11-05)

**Question.** Can a small model write a word problem as equations for ADA to solve, with the first form of the
`solve` prompt?

**Answer in short.** Not with that prompt. Under `solve`, llama3.2:1b, qwen2.5:0.5b and qwen3:1.7b get 3, 0
and 5 of 30 GSM8K problems right, against 11, 8 and 21 alone. qwen2.5:0.5b writes no "Answer:" line in any
of its 60 replies under `solve`, and ADA refuses 21 of qwen3:1.7b's 30 GSM8K replies. After this pilot, PR
#177 puts four worked examples before the question and adds reading rules for the model's equations.

## What ran

| | |
|---|---|
| Data | The first 30 problems of GSM8K's `test.jsonl` (sha256 `3730d312…c39d14`), matched exactly, and of GSM-Hard's `gsmhardv2.jsonl` (sha256 `2032b018…e9ddef`), matched to within 0.001 |
| Models | qwen2.5:0.5b, llama3.2:1b and qwen3:1.7b (thinking off), from Ollama's library |
| Conditions | `none` (alone), `check` (the `none` reply, checked by ADA) and `solve` (the model writes the problem as equations and names the quantity asked for on an "Answer:" line, and ADA solves the equations). Here the `solve` prompt is the instruction alone, with one example inside it ("A shop sells 3 boxes of 12 pens and 5 more pens"); no worked examples come before the question |
| Settings | Temperature 0, seed 0, context 8192 tokens, at most 2048 tokens a reply, one reply per problem and condition |
| Where | GitHub Actions, run [41955845319](https://github.com/saineshnakra/automated-data-analyst/actions/runs/41955845319) of `benchmark.yml`, started by hand on the branch `add-a-solve-condition`: 6 jobs on `ubuntu-latest` runners, CPU only, one per model and data set; started 02:32 UTC, done by 02:52 UTC |
| Code | Commit `f8e4be8` |

[config.json](config.json) has these settings in full. The records in [results.jsonl.gz](results.jsonl.gz)
are gsm8k.py's, one per problem, condition and model, with the data set and model added from the results
files' names: 540 records from all six jobs, with the model's whole reply, its equations under `solve` and
what ADA read from them. The GSM-Hard jobs of llama3.2:1b and qwen3:1.7b were downloaded from the run's
artifacts for this archive.

## Results

Made with `python paper/analysis.py paper/runs/2026-11-05-gsm-solve-pilot`. "ADA's answer, else the model's"
takes ADA's value when ADA solved the model's equations to an answer, and the model's own answer under `none`
when it did not. The p-values are exact McNemar tests against the model alone.

### GSM8K

Problems right, on the problems every condition finished:

| Model | Problems | none | check | solve | ADA's answer, else the model's |
|---|---|---|---|---|---|
| llama3.2:1b | 30 | 11 | 13 | 3 | 9 |
| qwen2.5:0.5b | 30 | 8 | 8 | 0 | 8 |
| qwen3:1.7b | 30 | 21 | 21 | 5 | 21 |

Against the model alone: right only with the condition / right only alone, and p:

| Model | check | solve |
|---|---|---|
| llama3.2:1b | 2 / 0, p = 0.5 | 1 / 9, p = 0.021 |
| qwen2.5:0.5b | 0 / 0, p = 1 | 0 / 8, p = 0.0078 |
| qwen3:1.7b | 0 / 0, p = 1 | 0 / 16, p = 3e-5 |
| All | 2 / 0, p = 0.5 | 1 / 33, p = 4e-9 |

ADA's statuses under solve:

- llama3.2:1b: solved 13, refused 8, no answer line 6, no value 3
- qwen2.5:0.5b: no answer line 30
- qwen3:1.7b: refused 21, solved 6, no value 3

Wrong answers alone, and how many of them have an arithmetic slip ADA finds:

- llama3.2:1b: 5 of 19
- qwen2.5:0.5b: 3 of 22
- qwen3:1.7b: 0 of 9

Replies that failed or never ended: 0

### GSM-Hard

Problems right, on the problems every condition finished:

| Model | Problems | none | check | solve | ADA's answer, else the model's |
|---|---|---|---|---|---|
| llama3.2:1b | 30 | 4 | 4 | 2 | 4 |
| qwen2.5:0.5b | 30 | 5 | 5 | 0 | 5 |
| qwen3:1.7b | 30 | 13 | 14 | 3 | 14 |

Against the model alone: right only with the condition / right only alone, and p:

| Model | check | solve |
|---|---|---|
| llama3.2:1b | 0 / 0, p = 1 | 2 / 4, p = 0.69 |
| qwen2.5:0.5b | 0 / 0, p = 1 | 0 / 5, p = 0.062 |
| qwen3:1.7b | 1 / 0, p = 1 | 2 / 12, p = 0.013 |
| All | 1 / 0, p = 1 | 4 / 21, p = 9e-4 |

ADA's statuses under solve:

- llama3.2:1b: solved 19, refused 6, no answer line 3, no value 2
- qwen2.5:0.5b: no answer line 30
- qwen3:1.7b: refused 22, solved 5, no value 2, no answer line 1

Wrong answers alone, and how many of them have an arithmetic slip ADA finds:

- llama3.2:1b: 12 of 26
- qwen2.5:0.5b: 11 of 25
- qwen3:1.7b: 5 of 17

Replies that failed or never ended: 0

### The same problems with larger numbers

Of the GSM8K problems a model gets right alone, how many GSM-Hard versions it gets right:

| Model | GSM8K right alone | GSM-Hard, none | GSM-Hard, check | GSM-Hard, solve |
|---|---|---|---|---|
| llama3.2:1b | 11 | 3 | 3 | 1 |
| qwen2.5:0.5b | 8 | 2 | 2 | 0 |
| qwen3:1.7b | 21 | 11 | 12 | 2 |

## What it shows

- **`solve` does worse than the model alone with this prompt**: 1 / 33 on GSM8K (p = 4e-9) and 4 / 21 on
  GSM-Hard (p = 9e-4), over the three models.
- **qwen2.5:0.5b does not follow the prompt**: none of its 60 replies under `solve` has an "Answer:" line.
- **qwen3:1.7b writes equations ADA refuses**: 21 of 30 on GSM8K and 22 of 30 on GSM-Hard.
- **llama3.2:1b copies the example**: its equations use the example's name `boxes` in 12 of its 30 GSM8K
  replies and 16 of its 30 GSM-Hard replies.
- **Checking** turns 3 wrong answers right and none wrong over the six pairs.

## What to keep in mind

- 30 problems and one reply each: a difference of a few problems for one model is within chance.
- The `solve` grades are those of `f8e4be8`'s reading rules. PR #177 reads the saved GSM8K replies again with
  its later rules and gets 11 right for qwen3:1.7b instead of 5 and 4 for llama3.2:1b instead of 3; the
  records here keep the grades of `f8e4be8`.
- benchmarks/README.md does not report this run.

## Before and after

- Before it: [2026-11-01-gsm-three-models-tool](../2026-11-01-gsm-three-models-tool/README.md), the last GSM8K
  and GSM-Hard run, without `solve`.
- After it: [2026-11-05-gsm-kaggle-smoke](../2026-11-05-gsm-kaggle-smoke/README.md), the same code with shards
  on a Kaggle GPU.
- Recorded in the description of PR #177, not in benchmarks/README.md.
