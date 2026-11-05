# GSM8K and GSM-Hard, thirteen models (2026-11-05)

**Question.** Which small models does ADA help on grade-school word problems, and how?

**Answer in short.** With large numbers (GSM-Hard), a model that writes the problem as equations for ADA to
solve does far better than alone, from about 1.5B parameters up: llama3.2:3b gets 195 of 300 right that
way, against 86 alone. Under `solve` a model scores about the same with small and large numbers. With small
numbers (GSM8K), `solve` does worse than the model alone for every model, because a small model's mistakes
there are in reading the problem, not in the arithmetic. Checking the model's own arithmetic never costs
more than a few answers and helps with large numbers. The worked examples on their own do not help.

## What ran

| | |
|---|---|
| Data | The first 300 problems of GSM8K's `test.jsonl` (sha256 `3730d312…c39d14`), matched exactly, and of GSM-Hard's `gsmhardv2.jsonl` (sha256 `2032b018…e9ddef`), matched to within 0.001. GSM-Hard's problem *i* is GSM8K's problem *i* with larger numbers |
| Models | qwen2.5:0.5b, qwen3:0.6b, gemma3:1b, llama3.2:1b, qwen2.5:1.5b, qwen2-math:1.5b, qwen3:1.7b (thinking off), smollm2:1.7b, llama3.2:3b, qwen2.5:3b, phi4-mini, gemma3:4b and qwen3:4b-instruct, from Ollama's library |
| Conditions | `none` (alone), `shots` (after four worked examples), `check` (alone, then ADA checks every equation it wrote and the model answers again when one is wrong), `solve` (the model writes equations, ADA solves them); see [benchmarks/README.md](../../../benchmarks/README.md) |
| Settings | Temperature 0, seed 0, context 8192 tokens, at most 2048 tokens a reply, one reply per problem and condition |
| Where | Kaggle kernel `saineshnakra/ada-benchmark-ladder-full`, version 1: two T4 GPUs with one Ollama server each, six shards per data set ([kaggle_ladder.py](../../../benchmarks/kaggle_ladder.py)); started 04:48 UTC, done by 10:17 UTC, the longest GPU busy 4.5 hours |
| Code | Commit `c4b7e37` (the same files as `f37d804` on main) |
| Graded again | With commit `afbe233`, which writes ADA's solve answers in digits only. The run wrote 7/594080 as `1.17829248586e-05` and read it back as 1.178…, so 19 answers on three GSM-Hard problems whose answers are within 0.001 of 0 were marked wrong; [results.jsonl.gz](results.jsonl.gz) holds the corrected grades |

[config.json](config.json) has these settings in full. The records in [results.jsonl.gz](results.jsonl.gz)
are gsm8k.py's, one per problem, condition and model, with the data set and model added: 31,200 records,
with the model's whole reply, its equations under `solve` and what ADA read from them.

## Results

Made with `python paper/analysis.py paper/runs/2026-11-05-gsm-ladder`. "ADA's answer, else the model's"
takes ADA's value when ADA solved the model's equations to an answer, and the model's own answer under `none`
when it did not. The p-values are exact McNemar tests against the model alone.

### GSM8K

Problems right, on the problems every condition finished:

| Model | Problems | none | shots | check | solve | ADA's answer, else the model's |
|---|---|---|---|---|---|---|
| gemma3:1b | 300 | 133 | 131 | 133 | 71 | 113 |
| gemma3:4b | 300 | 256 | 248 | 256 | 149 | 202 |
| llama3.2:1b | 300 | 151 | 129 | 156 | 76 | 108 |
| llama3.2:3b | 300 | 242 | 229 | 243 | 200 | 211 |
| phi4-mini | 300 | 265 | 240 | 268 | 180 | 236 |
| qwen2-math:1.5b | 300 | 248 | 229 | 249 | 122 | 220 |
| qwen2.5:0.5b | 300 | 120 | 104 | 121 | 48 | 84 |
| qwen2.5:1.5b | 300 | 186 | 165 | 187 | 133 | 157 |
| qwen2.5:3b | 300 | 256 | 240 | 256 | 193 | 226 |
| qwen3:0.6b | 300 | 143 | 129 | 143 | 36 | 93 |
| qwen3:1.7b | 300 | 240 | 207 | 245 | 125 | 162 |
| qwen3:4b-instruct | 300 | 282 | 278 | 282 | 235 | 257 |
| smollm2:1.7b | 300 | 131 | 134 | 136 | 82 | 119 |

Against the model alone: right only with the condition / right only alone, and p:

| Model | shots | check | solve |
|---|---|---|---|
| gemma3:1b | 42 / 44, p = 0.91 | 0 / 0, p = 1 | 25 / 87, p = 3e-9 |
| gemma3:4b | 17 / 25, p = 0.28 | 0 / 0, p = 1 | 6 / 113, p = 1e-26 |
| llama3.2:1b | 31 / 53, p = 0.021 | 5 / 0, p = 0.062 | 29 / 104, p = 4e-11 |
| llama3.2:3b | 21 / 34, p = 0.1 | 1 / 0, p = 1 | 22 / 64, p = 7e-6 |
| phi4-mini | 10 / 35, p = 2e-4 | 3 / 0, p = 0.25 | 9 / 94, p = 5e-19 |
| qwen2-math:1.5b | 16 / 35, p = 0.011 | 1 / 0, p = 1 | 7 / 133, p = 3e-31 |
| qwen2.5:0.5b | 31 / 47, p = 0.089 | 1 / 0, p = 1 | 13 / 85, p = 4e-14 |
| qwen2.5:1.5b | 27 / 48, p = 0.02 | 1 / 0, p = 1 | 25 / 78, p = 2e-7 |
| qwen2.5:3b | 16 / 32, p = 0.029 | 0 / 0, p = 1 | 16 / 79, p = 3e-11 |
| qwen3:0.6b | 31 / 45, p = 0.14 | 0 / 0, p = 1 | 7 / 114, p = 5e-26 |
| qwen3:1.7b | 14 / 47, p = 3e-5 | 5 / 0, p = 0.062 | 8 / 123, p = 1e-27 |
| qwen3:4b-instruct | 6 / 10, p = 0.45 | 0 / 0, p = 1 | 7 / 54, p = 4e-10 |
| smollm2:1.7b | 43 / 40, p = 0.83 | 5 / 0, p = 0.062 | 28 / 77, p = 2e-6 |
| All | 305 / 495, p = 2e-11 | 22 / 0, p = 5e-7 | 202 / 1205, p = 5e-174 |

ADA's statuses under solve:

- gemma3:1b: number given 175, no answer line 61, refused 48, no value 13, solved 3
- gemma3:4b: solved 238, no value 58, refused 4
- llama3.2:1b: solved 236, no value 44, number given 11, refused 5, no answer line 4
- llama3.2:3b: solved 283, no value 12, refused 3, no answer line 2
- phi4-mini: solved 232, no value 52, refused 9, no answer line 3, number given 2, failed 2
- qwen2-math:1.5b: solved 155, no value 104, number given 21, no answer line 14, refused 6
- qwen2.5:0.5b: solved 165, no value 61, number given 49, no answer line 13, refused 12
- qwen2.5:1.5b: solved 246, no value 42, refused 8, no answer line 3, number given 1
- qwen2.5:3b: solved 263, no value 27, refused 10
- qwen3:0.6b: solved 114, no value 108, number given 64, refused 11, example repeated 2, no answer line 1
- qwen3:1.7b: solved 254, no value 35, refused 10, number given 1
- qwen3:4b-instruct: solved 275, no value 20, refused 5
- smollm2:1.7b: solved 203, no value 71, refused 10, number given 9, no answer line 7

Wrong answers alone, and how many of them have an arithmetic slip ADA finds:

- gemma3:1b: 14 of 167
- gemma3:4b: 1 of 44
- llama3.2:1b: 20 of 149
- llama3.2:3b: 4 of 58
- phi4-mini: 2 of 35
- qwen2-math:1.5b: 2 of 52
- qwen2.5:0.5b: 15 of 180
- qwen2.5:1.5b: 6 of 114
- qwen2.5:3b: 3 of 44
- qwen3:0.6b: 16 of 157
- qwen3:1.7b: 9 of 60
- qwen3:4b-instruct: 0 of 18
- smollm2:1.7b: 21 of 169

Replies that failed or never ended: 6

### GSM-Hard

Problems right, on the problems every condition finished:

| Model | Problems | none | shots | check | solve | ADA's answer, else the model's |
|---|---|---|---|---|---|---|
| gemma3:1b | 300 | 64 | 64 | 67 | 31 | 52 |
| gemma3:4b | 300 | 152 | 143 | 165 | 149 | 174 |
| llama3.2:1b | 300 | 43 | 38 | 51 | 85 | 93 |
| llama3.2:3b | 300 | 86 | 85 | 110 | 195 | 200 |
| phi4-mini | 300 | 123 | 106 | 137 | 184 | 205 |
| qwen2-math:1.5b | 300 | 156 | 136 | 162 | 107 | 171 |
| qwen2.5:0.5b | 300 | 48 | 41 | 51 | 47 | 62 |
| qwen2.5:1.5b | 300 | 99 | 95 | 111 | 134 | 149 |
| qwen2.5:3b | 300 | 127 | 122 | 149 | 190 | 201 |
| qwen3:0.6b | 300 | 81 | 76 | 88 | 35 | 76 |
| qwen3:1.7b | 300 | 148 | 129 | 162 | 138 | 153 |
| qwen3:4b-instruct | 300 | 190 | 191 | 190 | 232 | 243 |
| smollm2:1.7b | 300 | 54 | 62 | 68 | 70 | 88 |

Against the model alone: right only with the condition / right only alone, and p:

| Model | shots | check | solve |
|---|---|---|---|
| gemma3:1b | 26 / 26, p = 1 | 4 / 1, p = 0.38 | 13 / 46, p = 2e-5 |
| gemma3:4b | 17 / 26, p = 0.22 | 13 / 0, p = 2e-4 | 57 / 60, p = 0.85 |
| llama3.2:1b | 15 / 20, p = 0.5 | 8 / 0, p = 0.0078 | 70 / 28, p = 3e-5 |
| llama3.2:3b | 16 / 17, p = 1 | 24 / 0, p = 1e-7 | 130 / 21, p = 2e-20 |
| phi4-mini | 16 / 33, p = 0.021 | 14 / 0, p = 1e-4 | 93 / 32, p = 4e-8 |
| qwen2-math:1.5b | 13 / 33, p = 0.0045 | 6 / 0, p = 0.031 | 44 / 93, p = 3e-5 |
| qwen2.5:0.5b | 15 / 22, p = 0.32 | 3 / 0, p = 0.25 | 34 / 35, p = 1 |
| qwen2.5:1.5b | 28 / 32, p = 0.7 | 12 / 0, p = 5e-4 | 77 / 42, p = 0.0017 |
| qwen2.5:3b | 25 / 30, p = 0.59 | 22 / 0, p = 5e-7 | 93 / 30, p = 1e-8 |
| qwen3:0.6b | 25 / 30, p = 0.59 | 7 / 0, p = 0.016 | 19 / 65, p = 5e-7 |
| qwen3:1.7b | 17 / 36, p = 0.013 | 14 / 0, p = 1e-4 | 52 / 62, p = 0.4 |
| qwen3:4b-instruct | 14 / 13, p = 1 | 2 / 2, p = 1 | 68 / 26, p = 2e-5 |
| smollm2:1.7b | 25 / 17, p = 0.28 | 16 / 2, p = 0.0013 | 50 / 34, p = 0.1 |
| All | 252 / 335, p = 7e-4 | 145 / 5, p = 9e-37 | 800 / 574, p = 1e-9 |

ADA's statuses under solve:

- gemma3:1b: number given 172, no answer line 79, refused 27, no value 19, solved 3
- gemma3:4b: solved 241, no value 52, refused 7
- llama3.2:1b: solved 256, no value 29, number given 6, no answer line 6, refused 3
- llama3.2:3b: solved 280, no value 14, refused 5, no answer line 1
- phi4-mini: solved 241, no value 52, refused 3, no answer line 2, failed 2
- qwen2-math:1.5b: solved 169, no value 103, no answer line 10, refused 10, number given 7, failed 1
- qwen2.5:0.5b: solved 168, no value 68, number given 51, refused 6, no answer line 6, failed 1
- qwen2.5:1.5b: solved 253, no value 32, refused 12, no answer line 2, number given 1
- qwen2.5:3b: solved 275, no value 18, refused 7
- qwen3:0.6b: solved 120, no value 115, number given 51, refused 10, example repeated 3, no answer line 1
- qwen3:1.7b: solved 270, no value 23, refused 7
- qwen3:4b-instruct: solved 282, no value 13, refused 5
- smollm2:1.7b: solved 211, no value 64, number given 12, refused 7, no answer line 6

Wrong answers alone, and how many of them have an arithmetic slip ADA finds:

- gemma3:1b: 110 of 236
- gemma3:4b: 58 of 148
- llama3.2:1b: 127 of 257
- llama3.2:3b: 100 of 214
- phi4-mini: 66 of 177
- qwen2-math:1.5b: 40 of 144
- qwen2.5:0.5b: 99 of 252
- qwen2.5:1.5b: 68 of 201
- qwen2.5:3b: 63 of 173
- qwen3:0.6b: 72 of 219
- qwen3:1.7b: 50 of 152
- qwen3:4b-instruct: 13 of 110
- smollm2:1.7b: 100 of 246

Replies that failed or never ended: 24

### The same problems with larger numbers

Of the GSM8K problems a model gets right alone, how many GSM-Hard versions it gets right:

| Model | GSM8K right alone | GSM-Hard, none | GSM-Hard, shots | GSM-Hard, check | GSM-Hard, solve |
|---|---|---|---|---|---|
| gemma3:1b | 133 | 28 | 22 | 29 | 14 |
| gemma3:4b | 256 | 129 | 122 | 140 | 126 |
| llama3.2:1b | 151 | 21 | 15 | 24 | 35 |
| llama3.2:3b | 242 | 70 | 70 | 87 | 161 |
| phi4-mini | 265 | 110 | 95 | 124 | 164 |
| qwen2-math:1.5b | 248 | 134 | 110 | 136 | 92 |
| qwen2.5:0.5b | 120 | 18 | 19 | 19 | 18 |
| qwen2.5:1.5b | 186 | 62 | 56 | 66 | 78 |
| qwen2.5:3b | 256 | 107 | 105 | 126 | 163 |
| qwen3:0.6b | 143 | 40 | 37 | 43 | 17 |
| qwen3:1.7b | 240 | 119 | 104 | 129 | 110 |
| qwen3:4b-instruct | 282 | 178 | 178 | 178 | 218 |
| smollm2:1.7b | 131 | 25 | 25 | 31 | 27 |


## What it shows

- **With ADA solving, the size of the numbers stops mattering much.** Under `solve`, llama3.2:3b scores 200
  and 195, phi4-mini 180 and 184, qwen3:4b-instruct 235 and 232 on GSM8K and GSM-Hard. Alone, the same
  models score 242 and 86, 265 and 123, 282 and 190.
- **With large numbers, `solve` helps most models from 1B up**, at p < 0.01 for llama3.2:1b, qwen2.5:1.5b,
  llama3.2:3b, qwen2.5:3b, phi4-mini and qwen3:4b-instruct. "ADA's answer, else the model's" is the best
  of the five for 10 of the 13 models on GSM-Hard.
- **With small numbers, `solve` hurts every model.** Of the wrong answers alone, 113 of 1,247 have a slip
  ADA finds on GSM8K and 966 of 2,529 on GSM-Hard.
- **Checking** turned 167 wrong answers right and 5 right answers wrong over the 26 runs.
- **The worked examples alone do not help**: `shots` against alone is 305 / 495 on GSM8K and 252 / 335 on
  GSM-Hard.
- **Below about 1B, models do not write equations**: gemma3:1b answers with a number of its own on 175 of
  300 GSM8K problems and without an Answer line on 61.

## What to keep in mind

- One reply per problem and condition at temperature 0. Pooled differences and those with p < 0.001 for one
  model are far beyond chance; a few problems for one model are not.
- 30 of the 31,200 replies failed with an HTTP 500 from Ollama and count as wrong (12 alone, 9 after
  examples, 3 checked, 6 under solve).
- Three GSM-Hard problems have answers within 0.001 of 0, which any small enough number matches; they
  count the same way under every condition.
- The full reading rules for `solve` are in [benchmarks/README.md](../../../benchmarks/README.md); a reply ADA
  cannot read has no answer under `solve`.

## Before and after

- Before it: [2026-11-05-gsm-ladder-pilot](../2026-11-05-gsm-ladder-pilot/README.md), 15 models on 30
  problems, which found the reading problems fixed in PR #178 and showed the models under 0.5B do nothing
  useful.
- Recorded in [benchmarks/README.md](../../../benchmarks/README.md) by PR #179.
