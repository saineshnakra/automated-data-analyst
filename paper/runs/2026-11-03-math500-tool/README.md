# MATH-500, two models with and without `ada_compute` (2026-11-03)

**Question.** Does offering `ada_compute` help a small model on competition algebra?

**Answer in short.** No. On MATH-500's 221 Algebra and Intermediate Algebra problems, both models get fewer
right with the tool offered than alone: qwen2.5:1.5b 100 against 127, qwen3:1.7b 141 against 161. The losses
do not come from the tool's answers: qwen3:1.7b calls the tool on 4 problems, and with the tool offered
qwen2.5:1.5b often leaves out the `\boxed{}` answer.

## What ran

| | |
|---|---|
| Data | MATH-500's `test.jsonl` (sha256 `35dc4108…a06132`), its 221 Algebra and Intermediate Algebra problems. Answers are graded by Math-Verify 0.9.0 |
| Models | qwen2.5:1.5b and qwen3:1.7b (thinking off), from Ollama's library |
| Conditions | `none` (alone) and `tool` (`ada_compute` offered, with a system prompt that says what it does); see [benchmarks/README.md](../../../benchmarks/README.md) |
| Settings | Temperature 0, seed 0, context 8192 tokens, at most 3,072 tokens a reply, at most 8 rounds of tool calls, one reply per problem and condition |
| Where | GitHub Actions, run [41691859618](https://github.com/saineshnakra/automated-data-analyst/actions/runs/41691859618) of `benchmark-math.yml` on `main`, started by hand: 16 jobs on `ubuntu-latest` runners, CPU only, four shards per model and condition; started 18:14 UTC, done by 19:16 UTC |
| Code | Commit `8f4c150` |

[config.json](config.json) has these settings in full. The records in [results.jsonl.gz](results.jsonl.gz)
are math500.py's, one per problem, condition and model, with the data set added: 884 records, with the
model's whole reply and its tool calls.

## Results

Made with `python paper/analysis.py paper/runs/2026-11-03-math500-tool`. The p-values are exact McNemar tests
against the model alone.

### MATH-500

Problems right, on the problems every condition finished:

| Model | Problems | none | tool |
|---|---|---|---|
| qwen2.5:1.5b | 221 | 127 | 100 |
| qwen3:1.7b | 221 | 161 | 141 |

Against the model alone: right only with the condition / right only alone, and p:

| Model | tool |
|---|---|
| qwen2.5:1.5b | 13 / 40, p = 3e-4 |
| qwen3:1.7b | 13 / 33, p = 0.0045 |
| All | 26 / 73, p = 2e-6 |

Replies that failed or never ended: 2

## What it shows

- **Offering the tool costs both models answers**: right only with the tool / right only alone is 13 / 40
  for qwen2.5:1.5b (p = 3e-4) and 13 / 33 for qwen3:1.7b (p = 0.0045).
- **qwen3:1.7b calls the tool on 4 problems** and gets all 4 right; none of the 46 problems where its two
  answers differ has a call. Its replies are shorter with the tool offered: a mean of 1,654 characters
  against 2,185 alone.
- **qwen2.5:1.5b calls it on 24 problems.** With the tool offered it puts an answer in `\boxed{}` in 124
  replies against 197 alone, and 18 of its 40 losses are replies with no boxed answer.

## What to keep in mind

- One reply per problem and condition at temperature 0 with seed 0, and a run does not repeat exactly on
  another runner: the next run gets 155 for qwen3:1.7b alone, against 161 here.
- 2 of the 884 replies failed with an HTTP 500 from Ollama (qwen2.5:1.5b alone, on
  `test/algebra/2626.json` and `test/intermediate_algebra/776.json`) and count as wrong.
- Only the two algebra subjects are asked, not all of MATH-500.

## Before and after

- Before it: [2026-11-01-gsm-three-models-tool](../2026-11-01-gsm-three-models-tool/README.md), where
  `ada_compute` did not help three small models on GSM8K and GSM-Hard and checking their arithmetic did.
- After it: [2026-11-03-math500-five-conditions](../2026-11-03-math500-five-conditions/README.md), which
  splits the tool into its prompt and its definition and adds `check`.
- Recorded in [benchmarks/README.md](../../../benchmarks/README.md) (MATH-500, "Results") by PR #160.
