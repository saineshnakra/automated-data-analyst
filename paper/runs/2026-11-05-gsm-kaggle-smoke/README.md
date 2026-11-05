# GSM8K on a Kaggle GPU, a smoke test (2026-11-05)

**Question.** Does the benchmark run on a Kaggle GPU, with shards side by side against one Ollama server?

**Answer in short.** Yes. qwen2.5:0.5b answers the first 20 GSM8K problems alone and under `solve` in four
shards side by side, with no failed replies, and the kernel's log ends 212 seconds after it starts. The model
gets 3 of 20 right alone and none under `solve`, where 19 of its replies have no "Answer:" line.

## What ran

| | |
|---|---|
| Data | The first 20 problems of GSM8K's `test.jsonl` (sha256 `3730d312…c39d14`), matched exactly |
| Models | qwen2.5:0.5b, from Ollama's library |
| Conditions | `none` (alone) and `solve` (the model writes the problem as equations, ADA solves them), with the first form of the `solve` prompt: the instruction alone, with one example inside it, as in [2026-11-05-gsm-solve-pilot](../2026-11-05-gsm-solve-pilot/README.md) |
| Settings | Temperature 0, seed 0, context 8192 tokens, at most 2048 tokens a reply, one reply per problem and condition |
| Where | Kaggle kernel `saineshnakra/ada-benchmark-smoke`; its version is not recorded. A GPU session where `nvidia-smi` lists two NVIDIA T4 GPUs, with one Ollama server (`OLLAMA_NUM_PARALLEL=4`) and four shards (`--shard 0/4` to `3/4`) side by side. The exact start time is not recorded: the last request in Ollama's log is at 02:42:58 UTC, about 3.5 minutes after the kernel started, so it started about 02:39 UTC |
| Code | Commit `3f9fa00`, which adds `--shard` to the code of the solve pilot (`f8e4be8`) and changes nothing else in how a problem is asked |

[config.json](config.json) has these settings in full. The records in [results.jsonl.gz](results.jsonl.gz)
are gsm8k.py's, one per problem and condition, with the data set and model added from the results files'
names: 40 records, with the model's whole reply, its equations under `solve` and what ADA read from them.

## Results

Made with `python paper/analysis.py paper/runs/2026-11-05-gsm-kaggle-smoke`. "ADA's answer, else the model's"
takes ADA's value when ADA solved the model's equations to an answer, and the model's own answer under `none`
when it did not. The p-value is an exact McNemar test against the model alone.

### GSM8K

Problems right, on the problems every condition finished:

| Model | Problems | none | solve | ADA's answer, else the model's |
|---|---|---|---|---|
| qwen2.5:0.5b | 20 | 3 | 0 | 3 |

Against the model alone: right only with the condition / right only alone, and p:

| Model | solve |
|---|---|
| qwen2.5:0.5b | 0 / 3, p = 0.25 |

ADA's statuses under solve:

- qwen2.5:0.5b: no answer line 19, refused 1

Replies that failed or never ended: 0

## What it shows

- **The Kaggle set-up works**: four shards of gsm8k.py against one Ollama server, 40 replies, none failed.
- **qwen2.5:0.5b does not follow the first `solve` prompt here either**: 19 of its 20 replies have no
  "Answer:" line, as in the solve pilot, where none of its 60 had one.

## What to keep in mind

- A smoke test, not a measurement: one model, 20 problems.
- Replies depend on the machine. On these 20 problems, with the same code for the model alone, temperature 0
  and seed 0, the solve pilot on GitHub's CPU runners gets 6 right alone and none of its 20 replies is the
  same word for word as here. The later Kaggle runs get 8 (the ladder pilot, commit `06a98f7`) and 5 (the
  full ladder, commit `c4b7e37`).
- benchmarks/README.md and the pull requests do not report this run.

## Before and after

- Before it: [2026-11-05-gsm-solve-pilot](../2026-11-05-gsm-solve-pilot/README.md), the same conditions on
  GitHub's runners.
- After it: [2026-11-05-gsm-ladder-pilot](../2026-11-05-gsm-ladder-pilot/README.md), fifteen models on the same
  Kaggle set-up.
- Not recorded in benchmarks/README.md or in a pull request.
