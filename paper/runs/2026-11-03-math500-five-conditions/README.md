# MATH-500, the tool's prompt and definition apart (2026-11-03)

**Question.** Which half of `ada_compute`, its system prompt or its definition, makes the models worse, and
does checking their arithmetic help?

**Answer in short.** The definition. With the tool offered under the plain prompt (`schema`), qwen2.5:1.5b
gets 100 of 221 and qwen3:1.7b 140, about as many as with the whole tool (102 and 139). With the tool's
prompt and no tool (`prompt`), they get 126 and 148, about level with alone (125 and 155). `check` gets 162
against 155 for qwen3:1.7b, but in this run `check` asks the model a second time before ADA checks, and most
of that gain is the second asking.

## What ran

| | |
|---|---|
| Data | MATH-500's `test.jsonl` (sha256 `35dc4108…a06132`), its 221 Algebra and Intermediate Algebra problems. Answers are graded by Math-Verify 0.9.0 |
| Models | qwen2.5:1.5b and qwen3:1.7b (thinking off), from Ollama's library |
| Conditions | `none` (alone), `tool` (`ada_compute` with its system prompt), `prompt` (the tool's system prompt, no tool offered), `schema` (the tool offered, with the plain system prompt) and `check` (the model alone, then ADA checks every equation it wrote; when one is wrong, the model is told which and answers once more). Here `check` asks the model alone again rather than starting from the `none` reply; that changes in #162 |
| Settings | Temperature 0, seed 0, context 8192 tokens, at most 3,072 tokens a reply, at most 8 rounds of tool calls, one reply per problem and condition |
| Where | GitHub Actions, run [41716634251](https://github.com/saineshnakra/automated-data-analyst/actions/runs/41716634251) of `benchmark-math.yml` on `main`, started by hand: 40 jobs on `ubuntu-latest` runners, CPU only, four shards per model and condition; started 20:52 UTC, done by 22:24 UTC |
| Code | Commit `f2620ed` |

[config.json](config.json) has these settings in full. The records in [results.jsonl.gz](results.jsonl.gz)
are math500.py's, one per problem, condition and model, with the data set added: 2,210 records, with the
model's whole reply, its tool calls, ADA's checks under `check` and, where ADA found a wrong equation, the
first reply.

## Results

Made with `python paper/analysis.py paper/runs/2026-11-03-math500-five-conditions`. The p-values are exact
McNemar tests against the model alone.

### MATH-500

Problems right, on the problems every condition finished:

| Model | Problems | none | tool | prompt | schema | check |
|---|---|---|---|---|---|---|
| qwen2.5:1.5b | 221 | 125 | 102 | 126 | 100 | 124 |
| qwen3:1.7b | 221 | 155 | 139 | 148 | 140 | 162 |

Against the model alone: right only with the condition / right only alone, and p:

| Model | tool | prompt | schema | check |
|---|---|---|---|---|
| qwen2.5:1.5b | 16 / 39, p = 0.0027 | 22 / 21, p = 1 | 15 / 40, p = 0.001 | 9 / 10, p = 1 |
| qwen3:1.7b | 15 / 31, p = 0.026 | 15 / 22, p = 0.32 | 14 / 29, p = 0.032 | 8 / 1, p = 0.039 |
| All | 31 / 70, p = 1e-4 | 37 / 43, p = 0.58 | 29 / 69, p = 7e-5 | 17 / 11, p = 0.34 |

Wrong answers alone, and how many of them have an arithmetic slip ADA finds:

- qwen2.5:1.5b: 10 of 96
- qwen3:1.7b: 5 of 66

Replies that failed or never ended: 2

## What it shows

- **The tool's definition does the harm.** Against the model alone, `schema` is 15 / 40 (p = 0.001) for
  qwen2.5:1.5b and 14 / 29 (p = 0.032) for qwen3:1.7b, close to `tool`; `prompt` is 22 / 21 and 15 / 22.
- **`check` is judged against its own first reply.** ADA finds a wrong equation in 8 of qwen3:1.7b's first
  replies under `check` and in 12 of qwen2.5:1.5b's. By benchmarks/README.md, told which, qwen3:1.7b turns 2
  wrong answers right and none wrong, and qwen2.5:1.5b fixes none and turns one right answer wrong; so
  qwen3:1.7b's first replies under `check` get 160 right, and most of the gain over 155 alone is the second
  asking.
- **A run does not repeat exactly on another runner**: qwen3:1.7b alone gets 155 here and 161 in the run
  before, with the same settings.

## What to keep in mind

- "Wrong answers alone, and how many of them have an arithmetic slip ADA finds" takes the grade from the
  `none` reply and the slip from ADA's checks of the `check` condition's own first reply. In this run those
  are two separate askings, and their replies are not always the same.
- 2 of the 2,210 replies failed with an HTTP 500 from Ollama and count as wrong, both for qwen2.5:1.5b and on
  the same problems as in the run before: `test/algebra/2626.json` alone and
  `test/intermediate_algebra/776.json` under `check`.
- The checks are ADA's at `f2620ed`. benchmarks/README.md says that after #164 (LaTeX) and #166 (steps with
  letters), ADA finds a wrong equation in 17 of qwen3:1.7b's first replies and 26 of qwen2.5:1.5b's; those
  counts are not in these records.

## Before and after

- Before it: [2026-11-03-math500-tool](../2026-11-03-math500-tool/README.md), `none` and `tool` only.
- After it: [2026-11-03-math500-qwen3-thinking](../2026-11-03-math500-qwen3-thinking/README.md), qwen3:1.7b
  with thinking on, all 500 problems, with `check` starting from the model's own reply.
- Recorded in [benchmarks/README.md](../../../benchmarks/README.md) (MATH-500, "Results") by PR #167.
