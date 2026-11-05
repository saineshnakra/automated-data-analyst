# GSM8K and GSM-Hard, fifteen models from 135M to 4B, a pilot (2026-11-05)

**Question.** Which small models can use the `solve` condition with worked examples, and where does ADA help?

**Answer in short.** On 30 problems of each data set: with small numbers (GSM8K), `solve` gets fewer right
than the model alone for every one of the 15 models (20 / 115 pooled, p = 2e-17). With large numbers
(GSM-Hard), it gets more for some models from 1.5B up, such as qwen2.5:1.5b (12 against 6 alone) and
llama3.2:3b (16 against 12), but 30 problems are too few to show that for one model. The three models under
0.5B get at most 2 of 30 right in any condition. qwen3:4b reasons aloud in its reply even with thinking off,
and the code of this run reads the equations it tries on the way together with its final ones; PR #178
changes that.

## What ran

| | |
|---|---|
| Data | The first 30 problems of GSM8K's `test.jsonl` (sha256 `3730d312…c39d14`), matched exactly, and of GSM-Hard's `gsmhardv2.jsonl` (sha256 `2032b018…e9ddef`), matched to within 0.001. GSM-Hard's problem *i* is GSM8K's problem *i* with larger numbers |
| Models | smollm2:135m, gemma3:270m, smollm2:360m, qwen2.5:0.5b, qwen3:0.6b, gemma3:1b, llama3.2:1b, qwen2.5:1.5b, qwen3:1.7b, smollm2:1.7b, llama3.2:3b, qwen2.5:3b, phi4-mini, gemma3:4b and qwen3:4b, from Ollama's library; thinking off |
| Conditions | `none` (alone), `check` (the `none` reply, checked by ADA) and `solve` (the model writes the problem as equations and ADA solves them), with four worked examples before the question as earlier turns of the chat; see [benchmarks/README.md](../../../benchmarks/README.md) |
| Settings | Temperature 0, seed 0, context 8192 tokens, at most 2048 tokens a reply, one reply per problem and condition |
| Where | Kaggle kernel `saineshnakra/ada-benchmark-ladder`, version 1: a GPU session where `nvidia-smi` lists two NVIDIA T4 GPUs, with one Ollama server (`OLLAMA_NUM_PARALLEL=4`); the models one after another, each with four shards per data set side by side. The exact times are not recorded: from the kernel's log it started about 03:07 UTC and the last request in Ollama's log is at 04:20:49 UTC. qwen3:4b took 1,813 seconds, the other models 76 to 255 each |
| Code | Commit `06a98f7` |
| Graded | By the code at `06a98f7`, not again. PR #178, later, reads the equations after a reply's last line of prose first; on these replies it takes qwen3:4b under `solve` from 16 to 22 on GSM8K and from 11 to 21 on GSM-Hard, and by PR #178 changes no other model's count of right answers |

[config.json](config.json) has these settings in full. The records in [results.jsonl.gz](results.jsonl.gz)
are gsm8k.py's, one per problem, condition and model, with the data set and model added from the results
files' names: 2,700 records, with the model's whole reply, its equations under `solve` and what ADA read from
them.

## Results

Made with `python paper/analysis.py paper/runs/2026-11-05-gsm-ladder-pilot`. "ADA's answer, else the model's"
takes ADA's value when ADA solved the model's equations to an answer, and the model's own answer under `none`
when it did not. The p-values are exact McNemar tests against the model alone.

### GSM8K

Problems right, on the problems every condition finished:

| Model | Problems | none | check | solve | ADA's answer, else the model's |
|---|---|---|---|---|---|
| gemma3:1b | 30 | 6 | 7 | 3 | 6 |
| gemma3:270m | 30 | 2 | 2 | 0 | 2 |
| gemma3:4b | 30 | 26 | 26 | 15 | 18 |
| llama3.2:1b | 30 | 12 | 13 | 7 | 9 |
| llama3.2:3b | 30 | 23 | 24 | 17 | 18 |
| phi4-mini | 30 | 27 | 27 | 16 | 23 |
| qwen2.5:0.5b | 30 | 12 | 12 | 1 | 9 |
| qwen2.5:1.5b | 30 | 16 | 16 | 13 | 17 |
| qwen2.5:3b | 30 | 22 | 22 | 16 | 18 |
| qwen3:0.6b | 30 | 11 | 11 | 3 | 7 |
| qwen3:1.7b | 30 | 22 | 22 | 14 | 16 |
| qwen3:4b | 30 | 29 | 29 | 16 | 29 |
| smollm2:1.7b | 30 | 11 | 11 | 5 | 9 |
| smollm2:135m | 30 | 1 | 1 | 0 | 0 |
| smollm2:360m | 30 | 1 | 1 | 0 | 0 |

Against the model alone: right only with the condition / right only alone, and p:

| Model | check | solve |
|---|---|---|
| gemma3:1b | 1 / 0, p = 1 | 2 / 5, p = 0.45 |
| gemma3:270m | 0 / 0, p = 1 | 0 / 2, p = 0.5 |
| gemma3:4b | 0 / 0, p = 1 | 0 / 11, p = 1e-3 |
| llama3.2:1b | 1 / 0, p = 1 | 4 / 9, p = 0.27 |
| llama3.2:3b | 1 / 0, p = 1 | 2 / 8, p = 0.11 |
| phi4-mini | 0 / 0, p = 1 | 1 / 12, p = 0.0034 |
| qwen2.5:0.5b | 0 / 0, p = 1 | 1 / 12, p = 0.0034 |
| qwen2.5:1.5b | 0 / 0, p = 1 | 5 / 8, p = 0.58 |
| qwen2.5:3b | 0 / 0, p = 1 | 1 / 7, p = 0.07 |
| qwen3:0.6b | 0 / 0, p = 1 | 1 / 9, p = 0.021 |
| qwen3:1.7b | 0 / 0, p = 1 | 1 / 9, p = 0.021 |
| qwen3:4b | 0 / 0, p = 1 | 0 / 13, p = 2e-4 |
| smollm2:1.7b | 0 / 0, p = 1 | 2 / 8, p = 0.11 |
| smollm2:135m | 0 / 0, p = 1 | 0 / 1, p = 1 |
| smollm2:360m | 0 / 0, p = 1 | 0 / 1, p = 1 |
| All | 3 / 0, p = 0.25 | 20 / 115, p = 2e-17 |

ADA's statuses under solve:

- gemma3:1b: number given 14, no answer line 8, refused 6, no value 2
- gemma3:270m: no answer line 23, number given 3, no value 3, failed 1
- gemma3:4b: solved 26, no value 3, refused 1
- llama3.2:1b: solved 23, no value 5, refused 1, number given 1
- llama3.2:3b: solved 27, no value 3
- phi4-mini: solved 22, no value 6, refused 2
- qwen2.5:0.5b: solved 12, no value 8, number given 4, no answer line 3, refused 3
- qwen2.5:1.5b: solved 24, no value 3, refused 2, no answer line 1
- qwen2.5:3b: solved 27, no value 3
- qwen3:0.6b: solved 13, no value 11, number given 6
- qwen3:1.7b: solved 26, refused 3, no value 1
- qwen3:4b: solved 17, refused 9, no answer line 4
- smollm2:1.7b: solved 19, no value 8, refused 3
- smollm2:135m: number given 13, no answer line 7, no value 6, refused 3, failed 1
- smollm2:360m: no value 11, no answer line 9, number given 5, refused 4, solved 1

Wrong answers alone, and how many of them have an arithmetic slip ADA finds:

- gemma3:1b: 3 of 24
- gemma3:270m: 4 of 28
- gemma3:4b: 0 of 4
- llama3.2:1b: 3 of 18
- llama3.2:3b: 2 of 7
- phi4-mini: 0 of 3
- qwen2.5:0.5b: 1 of 18
- qwen2.5:1.5b: 0 of 14
- qwen2.5:3b: 0 of 8
- qwen3:0.6b: 1 of 19
- qwen3:1.7b: 0 of 8
- qwen3:4b: 0 of 1
- smollm2:1.7b: 1 of 19
- smollm2:135m: 11 of 29
- smollm2:360m: 7 of 29

Replies that failed or never ended: 2

### GSM-Hard

Problems right, on the problems every condition finished:

| Model | Problems | none | check | solve | ADA's answer, else the model's |
|---|---|---|---|---|---|
| gemma3:1b | 30 | 7 | 7 | 4 | 6 |
| gemma3:270m | 30 | 2 | 2 | 0 | 2 |
| gemma3:4b | 30 | 11 | 12 | 9 | 13 |
| llama3.2:1b | 30 | 6 | 6 | 6 | 9 |
| llama3.2:3b | 30 | 12 | 13 | 16 | 16 |
| phi4-mini | 30 | 12 | 14 | 14 | 17 |
| qwen2.5:0.5b | 30 | 4 | 4 | 3 | 5 |
| qwen2.5:1.5b | 30 | 6 | 6 | 12 | 12 |
| qwen2.5:3b | 30 | 13 | 15 | 14 | 15 |
| qwen3:0.6b | 30 | 8 | 9 | 1 | 7 |
| qwen3:1.7b | 30 | 16 | 18 | 10 | 11 |
| qwen3:4b | 30 | 13 | 13 | 11 | 16 |
| smollm2:1.7b | 30 | 6 | 6 | 6 | 9 |
| smollm2:135m | 30 | 2 | 2 | 0 | 2 |
| smollm2:360m | 30 | 1 | 1 | 0 | 1 |

Against the model alone: right only with the condition / right only alone, and p:

| Model | check | solve |
|---|---|---|
| gemma3:1b | 0 / 0, p = 1 | 2 / 5, p = 0.45 |
| gemma3:270m | 0 / 0, p = 1 | 0 / 2, p = 0.5 |
| gemma3:4b | 1 / 0, p = 1 | 6 / 8, p = 0.79 |
| llama3.2:1b | 0 / 0, p = 1 | 5 / 5, p = 1 |
| llama3.2:3b | 1 / 0, p = 1 | 7 / 3, p = 0.34 |
| phi4-mini | 2 / 0, p = 0.5 | 7 / 5, p = 0.77 |
| qwen2.5:0.5b | 0 / 0, p = 1 | 1 / 2, p = 1 |
| qwen2.5:1.5b | 0 / 0, p = 1 | 8 / 2, p = 0.11 |
| qwen2.5:3b | 2 / 0, p = 0.5 | 6 / 5, p = 1 |
| qwen3:0.6b | 1 / 0, p = 1 | 1 / 8, p = 0.039 |
| qwen3:1.7b | 2 / 0, p = 0.5 | 4 / 10, p = 0.18 |
| qwen3:4b | 0 / 0, p = 1 | 3 / 5, p = 0.73 |
| smollm2:1.7b | 0 / 0, p = 1 | 3 / 3, p = 1 |
| smollm2:135m | 0 / 0, p = 1 | 0 / 2, p = 0.5 |
| smollm2:360m | 0 / 0, p = 1 | 0 / 1, p = 1 |
| All | 9 / 0, p = 0.0039 | 53 / 66, p = 0.27 |

ADA's statuses under solve:

- gemma3:1b: number given 19, no answer line 6, no value 3, refused 2
- gemma3:270m: no answer line 23, number given 2, no value 2, refused 1, solved 1, failed 1
- gemma3:4b: solved 21, no value 9
- llama3.2:1b: solved 25, no value 4, refused 1
- llama3.2:3b: solved 29, no answer line 1
- phi4-mini: solved 23, no value 5, refused 2
- qwen2.5:0.5b: solved 18, no value 8, no answer line 3, refused 1
- qwen2.5:1.5b: solved 26, no value 2, refused 2
- qwen2.5:3b: solved 28, no value 1, refused 1
- qwen3:0.6b: no value 18, solved 9, number given 2, refused 1
- qwen3:1.7b: solved 27, refused 2, no value 1
- qwen3:4b: solved 12, refused 12, no answer line 5, no value 1
- smollm2:1.7b: solved 20, no value 5, no answer line 2, refused 2, number given 1
- smollm2:135m: number given 15, no answer line 9, solved 2, no value 2, refused 1, failed 1
- smollm2:360m: number given 8, no answer line 8, no value 7, solved 5, failed 1, refused 1

Wrong answers alone, and how many of them have an arithmetic slip ADA finds:

- gemma3:1b: 9 of 23
- gemma3:270m: 15 of 28
- gemma3:4b: 5 of 19
- llama3.2:1b: 10 of 24
- llama3.2:3b: 8 of 18
- phi4-mini: 7 of 18
- qwen2.5:0.5b: 12 of 26
- qwen2.5:1.5b: 6 of 24
- qwen2.5:3b: 7 of 17
- qwen3:0.6b: 8 of 22
- qwen3:1.7b: 3 of 14
- qwen3:4b: 0 of 17
- smollm2:1.7b: 7 of 24
- smollm2:135m: 16 of 28
- smollm2:360m: 18 of 29

Replies that failed or never ended: 8

### The same problems with larger numbers

Of the GSM8K problems a model gets right alone, how many GSM-Hard versions it gets right:

| Model | GSM8K right alone | GSM-Hard, none | GSM-Hard, check | GSM-Hard, solve |
|---|---|---|---|---|
| gemma3:1b | 6 | 2 | 2 | 1 |
| gemma3:270m | 2 | 0 | 0 | 0 |
| gemma3:4b | 26 | 10 | 11 | 5 |
| llama3.2:1b | 12 | 4 | 4 | 1 |
| llama3.2:3b | 23 | 11 | 11 | 13 |
| phi4-mini | 27 | 12 | 13 | 12 |
| qwen2.5:0.5b | 12 | 1 | 1 | 1 |
| qwen2.5:1.5b | 16 | 3 | 3 | 6 |
| qwen2.5:3b | 22 | 9 | 11 | 10 |
| qwen3:0.6b | 11 | 5 | 5 | 0 |
| qwen3:1.7b | 22 | 13 | 14 | 8 |
| qwen3:4b | 29 | 13 | 13 | 10 |
| smollm2:1.7b | 11 | 1 | 1 | 0 |
| smollm2:135m | 1 | 0 | 0 | 0 |
| smollm2:360m | 1 | 0 | 0 | 0 |

## What it shows

- **With small numbers, `solve` does worse than the model alone for every model**, 20 / 115 pooled on GSM8K.
- **With large numbers, it is level overall** (53 / 66 on GSM-Hard, p = 0.27), better for qwen2.5:1.5b (8 /
  2), llama3.2:3b (7 / 3) and phi4-mini (7 / 5) and worse for qwen3:0.6b (1 / 8) and qwen3:1.7b (4 / 10).
- **Checking never costs an answer**: 3 / 0 on GSM8K and 9 / 0 on GSM-Hard (p = 0.0039).
- **Below 0.5B the models do nothing useful.** smollm2:135m, gemma3:270m and smollm2:360m get 1 or 2 of 30
  right alone, and under `solve` ADA solves the equations of 1 of their 90 GSM8K replies and 8 of their 90
  GSM-Hard ones.
- **qwen3:4b thinks aloud with thinking off.** It gets 29 of 30 GSM8K problems right alone, but ADA refuses 9
  of its 30 GSM8K replies under `solve` and 12 of its 30 GSM-Hard ones, as graded here.

## What to keep in mind

- 30 problems and one reply each: a difference of a few problems for one model is within chance.
- 10 of the 2,700 replies failed with an HTTP 500 from Ollama and count as wrong: 2 on GSM8K, both under
  `solve`, and 8 on GSM-Hard (4 alone, 1 checked, 3 under `solve`), all for models of 1B and under.
- qwen3:4b here is Ollama's `qwen3:4b`, a model with a thinking mode; the full ladder uses
  `qwen3:4b-instruct` instead.
- The three GSM-Hard problems with answers within 0.001 of 0, graded again in the full ladder, are not among
  these 30.

## Before and after

- Before it: [2026-11-05-gsm-kaggle-smoke](../2026-11-05-gsm-kaggle-smoke/README.md), one model on the same
  Kaggle set-up.
- After it: [2026-11-05-gsm-ladder](../2026-11-05-gsm-ladder/README.md), thirteen models on 300 problems, with
  PR #178's reading rules.
- Recorded in the description of PR #178 (its re-scored qwen3:4b counts); benchmarks/README.md does not
  report this run.
