# GSM8K and GSM-Hard, the combining rule tested on new problems (2026-11-05)

**Question.** Does the rule for combining the model's own answer with ADA's, chosen on problems 0 to 299,
hold on problems 300 to 599? And does letting the model decide between the two answers do better?

**Answer in short.** On GSM-Hard the rule holds clearly: pooled over the 13 models, `combine` gets 1,924 of
3,900 right, against 1,405 for the model alone, 1,542 checked and 1,529 under `solve`. On GSM8K it gets
2,618, more than the model alone (2,606) but fewer than `check` (2,634), so one of the three tests set
beforehand fails, by 16 answers. Letting the model decide (`judge`) does worse than the rule: on GSM8K it
loses more answers than it gains against the model alone.

## What ran

| | |
|---|---|
| Data | GSM8K's `test.jsonl` (sha256 `3730d312…c39d14`) and GSM-Hard's `gsmhardv2.jsonl` (sha256 `2032b018…e9ddef`), problems 300 to 599 of each (`--problems 600 --skip 300`), which no rule was chosen on. GSM8K is matched exactly, GSM-Hard to within 0.001 |
| Models | The 13 of [2026-11-05-gsm-ladder](../2026-11-05-gsm-ladder/README.md), from qwen2.5:0.5b to qwen3:4b-instruct |
| Conditions | `none`, `check`, `solve`, and the two ways of settling a difference between the model's answer and ADA's: `combine` (the rule: ADA's when ADA found a wrong equation in the model's answer or the question has a number of five or more digits, the model's otherwise) and `judge` (the model is shown ADA's equations and value beside its own answer and answers again). Both take the answer both give when they agree, and `check`'s when ADA has none; see [benchmarks/README.md](../../../benchmarks/README.md) |
| Settings | Temperature 0, seed 0, context 8192 tokens, at most 2048 tokens a reply, one reply per problem and condition |
| Where | Kaggle kernel `saineshnakra/ada-benchmark-combine`, version 1: two T4 GPUs with one Ollama server each, six shards per data set ([kaggle_ladder.py](../../../benchmarks/kaggle_ladder.py)); started 11:22 UTC, done by 15:43 UTC |
| Code | Commit `f5fe3de` on branch `combine-the-answers` |
| Chosen in | [analyses/2026-11-05-combine-rules](../../analyses/2026-11-05-combine-rules/README.md), written before this run |

[config.json](config.json) has these settings in full. [results.jsonl.gz](results.jsonl.gz) holds 39,000
records, with each `combine` and `judge` record's `decided_by`: why it took the answer it took.

## Results

Made with `python paper/analysis.py paper/runs/2026-11-05-gsm-combine`.

### GSM8K, problems 300 to 599

Problems right, on the problems every condition finished:

| Model | Problems | none | check | solve | combine | judge | ADA's answer, else the model's |
|---|---|---|---|---|---|---|---|
| gemma3:1b | 300 | 134 | 135 | 77 | 136 | 139 | 119 |
| gemma3:4b | 300 | 242 | 244 | 157 | 241 | 237 | 200 |
| llama3.2:1b | 300 | 141 | 145 | 71 | 146 | 111 | 95 |
| llama3.2:3b | 300 | 232 | 235 | 179 | 233 | 200 | 192 |
| phi4-mini | 300 | 253 | 258 | 176 | 256 | 248 | 229 |
| qwen2-math:1.5b | 300 | 246 | 246 | 120 | 243 | 245 | 213 |
| qwen2.5:0.5b | 300 | 116 | 117 | 48 | 117 | 115 | 87 |
| qwen2.5:1.5b | 300 | 194 | 195 | 125 | 194 | 185 | 155 |
| qwen2.5:3b | 300 | 243 | 247 | 187 | 246 | 234 | 212 |
| qwen3:0.6b | 300 | 154 | 155 | 38 | 152 | 152 | 114 |
| qwen3:1.7b | 300 | 243 | 244 | 138 | 242 | 233 | 164 |
| qwen3:4b-instruct | 300 | 275 | 276 | 245 | 276 | 278 | 258 |
| smollm2:1.7b | 300 | 133 | 137 | 73 | 136 | 125 | 111 |

Against the model alone: right only with the condition / right only alone, and p:

| Model | check | solve | combine | judge |
|---|---|---|---|---|
| gemma3:1b | 1 / 0, p = 1 | 19 / 76, p = 3e-9 | 2 / 0, p = 0.5 | 13 / 8, p = 0.38 |
| gemma3:4b | 2 / 0, p = 0.5 | 11 / 96, p = 4e-18 | 1 / 2, p = 1 | 16 / 21, p = 0.51 |
| llama3.2:1b | 4 / 0, p = 0.12 | 21 / 91, p = 1e-11 | 6 / 1, p = 0.12 | 17 / 47, p = 2e-4 |
| llama3.2:3b | 3 / 0, p = 0.25 | 19 / 72, p = 2e-8 | 3 / 2, p = 1 | 18 / 50, p = 1e-4 |
| phi4-mini | 5 / 0, p = 0.062 | 13 / 90, p = 2e-15 | 3 / 0, p = 0.25 | 9 / 14, p = 0.4 |
| qwen2-math:1.5b | 0 / 0, p = 1 | 8 / 134, p = 1e-30 | 0 / 3, p = 0.25 | 4 / 5, p = 1 |
| qwen2.5:0.5b | 1 / 0, p = 1 | 19 / 87, p = 1e-11 | 2 / 1, p = 1 | 2 / 3, p = 1 |
| qwen2.5:1.5b | 1 / 0, p = 1 | 27 / 96, p = 3e-10 | 2 / 2, p = 1 | 25 / 34, p = 0.3 |
| qwen2.5:3b | 4 / 0, p = 0.12 | 14 / 70, p = 4e-10 | 4 / 1, p = 0.38 | 10 / 19, p = 0.14 |
| qwen3:0.6b | 1 / 0, p = 1 | 2 / 118, p = 1e-32 | 0 / 2, p = 0.5 | 0 / 2, p = 0.5 |
| qwen3:1.7b | 1 / 0, p = 1 | 9 / 114, p = 3e-24 | 3 / 4, p = 1 | 7 / 17, p = 0.064 |
| qwen3:4b-instruct | 1 / 0, p = 1 | 12 / 42, p = 5e-5 | 2 / 1, p = 1 | 5 / 2, p = 0.45 |
| smollm2:1.7b | 4 / 0, p = 0.12 | 27 / 87, p = 2e-8 | 5 / 2, p = 0.45 | 11 / 19, p = 0.2 |
| All | 28 / 0, p = 7e-9 | 201 / 1173, p = 4e-167 | 33 / 21, p = 0.13 | 137 / 241, p = 1e-7 |

ADA's statuses under solve:

- gemma3:1b: number given 188, no answer line 48, refused 40, no value 20, solved 4
- gemma3:4b: solved 242, no value 52, refused 6
- llama3.2:1b: solved 236, no value 42, number given 13, no answer line 7, refused 2
- llama3.2:3b: solved 281, no value 12, refused 7
- phi4-mini: solved 228, no value 57, refused 11, number given 4
- qwen2-math:1.5b: solved 163, no value 91, number given 20, no answer line 17, refused 9
- qwen2.5:0.5b: solved 148, no value 75, number given 50, no answer line 17, refused 10
- qwen2.5:1.5b: solved 244, no value 30, refused 19, no answer line 5, number given 2
- qwen2.5:3b: solved 266, no value 25, refused 9
- qwen3:0.6b: no value 133, solved 80, number given 74, refused 11, example repeated 1, no answer line 1
- qwen3:1.7b: solved 267, no value 21, refused 12
- qwen3:4b-instruct: solved 285, no value 12, refused 3
- smollm2:1.7b: solved 200, no value 62, number given 18, refused 13, no answer line 7

Wrong answers alone, and how many of them have an arithmetic slip ADA finds:

- gemma3:1b: 7 of 166
- gemma3:4b: 2 of 58
- llama3.2:1b: 14 of 159
- llama3.2:3b: 5 of 68
- phi4-mini: 4 of 47
- qwen2-math:1.5b: 2 of 54
- qwen2.5:0.5b: 15 of 184
- qwen2.5:1.5b: 6 of 106
- qwen2.5:3b: 6 of 57
- qwen3:0.6b: 11 of 146
- qwen3:1.7b: 4 of 57
- qwen3:4b-instruct: 2 of 25
- smollm2:1.7b: 12 of 167

Replies that failed or never ended: 6

### GSM-Hard, problems 300 to 599

Problems right, on the problems every condition finished:

| Model | Problems | none | check | solve | combine | judge | ADA's answer, else the model's |
|---|---|---|---|---|---|---|---|
| gemma3:1b | 300 | 64 | 67 | 28 | 61 | 70 | 57 |
| gemma3:4b | 300 | 151 | 164 | 133 | 178 | 162 | 159 |
| llama3.2:1b | 300 | 55 | 61 | 71 | 98 | 56 | 84 |
| llama3.2:3b | 300 | 90 | 116 | 179 | 198 | 121 | 186 |
| phi4-mini | 300 | 118 | 139 | 177 | 199 | 146 | 193 |
| qwen2-math:1.5b | 300 | 156 | 166 | 107 | 176 | 160 | 169 |
| qwen2.5:0.5b | 300 | 45 | 50 | 50 | 70 | 49 | 63 |
| qwen2.5:1.5b | 300 | 86 | 94 | 138 | 162 | 117 | 151 |
| qwen2.5:3b | 300 | 146 | 160 | 191 | 206 | 166 | 204 |
| qwen3:0.6b | 300 | 84 | 87 | 26 | 77 | 87 | 64 |
| qwen3:1.7b | 300 | 155 | 171 | 132 | 162 | 173 | 145 |
| qwen3:4b-instruct | 300 | 197 | 199 | 229 | 239 | 207 | 241 |
| smollm2:1.7b | 300 | 58 | 68 | 68 | 98 | 60 | 87 |

Against the model alone: right only with the condition / right only alone, and p:

| Model | check | solve | combine | judge |
|---|---|---|---|---|
| gemma3:1b | 3 / 0, p = 0.25 | 8 / 44, p = 4e-7 | 8 / 11, p = 0.65 | 11 / 5, p = 0.21 |
| gemma3:4b | 17 / 4, p = 0.0072 | 52 / 70, p = 0.12 | 54 / 27, p = 0.0036 | 37 / 26, p = 0.21 |
| llama3.2:1b | 8 / 2, p = 0.11 | 48 / 32, p = 0.093 | 46 / 3, p = 7e-11 | 11 / 10, p = 1 |
| llama3.2:3b | 26 / 0, p = 3e-8 | 115 / 26, p = 1e-14 | 112 / 4, p = 2e-28 | 49 / 18, p = 2e-4 |
| phi4-mini | 21 / 0, p = 1e-6 | 86 / 27, p = 2e-8 | 87 / 6, p = 2e-19 | 33 / 5, p = 4e-6 |
| qwen2-math:1.5b | 10 / 0, p = 0.002 | 33 / 82, p = 6e-6 | 32 / 12, p = 0.0037 | 7 / 3, p = 0.34 |
| qwen2.5:0.5b | 5 / 0, p = 0.062 | 34 / 29, p = 0.61 | 32 / 7, p = 7e-5 | 6 / 2, p = 0.29 |
| qwen2.5:1.5b | 9 / 1, p = 0.021 | 81 / 29, p = 7e-7 | 80 / 4, p = 2e-19 | 41 / 10, p = 1e-5 |
| qwen2.5:3b | 14 / 0, p = 1e-4 | 75 / 30, p = 1e-5 | 72 / 12, p = 1e-11 | 25 / 5, p = 3e-4 |
| qwen3:0.6b | 3 / 0, p = 0.25 | 12 / 70, p = 4e-11 | 13 / 20, p = 0.3 | 3 / 0, p = 0.25 |
| qwen3:1.7b | 16 / 0, p = 3e-5 | 46 / 69, p = 0.04 | 44 / 37, p = 0.51 | 26 / 8, p = 0.0029 |
| qwen3:4b-instruct | 3 / 1, p = 0.62 | 54 / 22, p = 3e-4 | 51 / 9, p = 3e-8 | 12 / 2, p = 0.013 |
| smollm2:1.7b | 11 / 1, p = 0.0063 | 54 / 44, p = 0.36 | 49 / 9, p = 9e-8 | 13 / 11, p = 0.84 |
| All | 146 / 9, p = 5e-33 | 698 / 574, p = 6e-4 | 680 / 161, p = 1e-76 | 274 / 105, p = 2e-18 |

ADA's statuses under solve:

- gemma3:1b: number given 154, no answer line 79, refused 36, no value 26, solved 4, failed 1
- gemma3:4b: solved 255, no value 42, refused 3
- llama3.2:1b: solved 233, no value 40, no answer line 14, number given 7, refused 6
- llama3.2:3b: solved 284, no value 11, refused 5
- phi4-mini: solved 251, no value 42, refused 6, failed 1
- qwen2-math:1.5b: solved 172, no value 104, no answer line 12, refused 9, number given 3
- qwen2.5:0.5b: solved 173, no value 69, number given 44, no answer line 8, refused 5, failed 1
- qwen2.5:1.5b: solved 257, no value 23, refused 14, no answer line 4, number given 2
- qwen2.5:3b: solved 269, no value 21, refused 10
- qwen3:0.6b: solved 121, no value 119, number given 41, refused 16, example repeated 3
- qwen3:1.7b: solved 276, refused 12, no value 12
- qwen3:4b-instruct: solved 284, no value 12, refused 4
- smollm2:1.7b: solved 210, no value 58, refused 11, number given 11, no answer line 10

Wrong answers alone, and how many of them have an arithmetic slip ADA finds:

- gemma3:1b: 96 of 236
- gemma3:4b: 54 of 149
- llama3.2:1b: 136 of 245
- llama3.2:3b: 112 of 210
- phi4-mini: 70 of 182
- qwen2-math:1.5b: 45 of 144
- qwen2.5:0.5b: 99 of 255
- qwen2.5:1.5b: 48 of 214
- qwen2.5:3b: 60 of 154
- qwen3:0.6b: 72 of 216
- qwen3:1.7b: 49 of 145
- qwen3:4b-instruct: 11 of 103
- smollm2:1.7b: 100 of 242

Replies that failed or never ended: 17

### The same problems with larger numbers

Of the GSM8K problems a model gets right alone, how many GSM-Hard versions it gets right:

| Model | GSM8K right alone | GSM-Hard, none | GSM-Hard, check | GSM-Hard, solve | GSM-Hard, combine | GSM-Hard, judge |
|---|---|---|---|---|---|---|
| gemma3:1b | 134 | 23 | 25 | 5 | 23 | 23 |
| gemma3:4b | 242 | 122 | 134 | 106 | 142 | 128 |
| llama3.2:1b | 141 | 27 | 29 | 25 | 42 | 23 |
| llama3.2:3b | 232 | 75 | 97 | 140 | 157 | 98 |
| phi4-mini | 253 | 102 | 118 | 147 | 168 | 125 |
| qwen2-math:1.5b | 246 | 130 | 138 | 87 | 146 | 134 |
| qwen2.5:0.5b | 116 | 18 | 18 | 16 | 21 | 17 |
| qwen2.5:1.5b | 194 | 62 | 67 | 96 | 110 | 81 |
| qwen2.5:3b | 243 | 118 | 131 | 155 | 167 | 135 |
| qwen3:0.6b | 154 | 38 | 40 | 10 | 35 | 39 |
| qwen3:1.7b | 243 | 129 | 141 | 108 | 132 | 138 |
| qwen3:4b-instruct | 275 | 182 | 184 | 211 | 220 | 191 |
| smollm2:1.7b | 133 | 24 | 27 | 30 | 41 | 25 |


### Pooled over the 13 models

| Data | none | check | solve | combine | judge | ADA's answer, else the model's |
|---|---|---|---|---|---|---|
| GSM8K | 2,606 | 2,634 | 1,634 | 2,618 | 2,502 | 2,149 |
| GSM-Hard | 1,405 | 1,542 | 1,529 | 1,924 | 1,574 | 1,803 |

### What `combine` decided, and how often it was right

| Data | Decided by | Problems | combine right | alone right | check right |
|---|---|---|---|---|---|
| GSM8K | both agree | 1,525 | 1,433 | 1,433 | 1,434 |
| GSM8K | no answer from ADA | 887 | 519 | 515 | 519 |
| GSM8K | the model's answer | 1,373 | 637 | 637 | 641 |
| GSM8K | large numbers | 47 | 6 | 20 | 20 |
| GSM8K | a wrong equation in its answer | 68 | 23 | 1 | 20 |
| GSM-Hard | both agree | 934 | 831 | 831 | 824 |
| GSM-Hard | no answer from ADA | 849 | 293 | 274 | 293 |
| GSM-Hard | the model's answer | 279 | 140 | 140 | 142 |
| GSM-Hard | large numbers | 1,103 | 357 | 155 | 156 |
| GSM-Hard | a wrong equation in its answer | 735 | 303 | 5 | 127 |

## The tests set before the run

| Test | Result | Held? |
|---|---|---|
| GSM8K: `combine` at least as many right as `check` | 2,618 against 2,634 | No, 16 short |
| GSM8K: `combine` more right than the model alone | 2,618 against 2,606 (33 / 21, p = 0.13) | Yes, though not beyond chance |
| GSM-Hard: `combine` more right than `solve` and than `check` | 1,924 against 1,529 and 1,542 | Yes |
| Per model, `combine` not worse than alone at p < 0.05 on either data set | the smallest p for a loss is 0.25 (qwen2-math:1.5b on GSM8K, 0 / 3) | Yes |

## What it shows

- **The rule carries over to new problems where the numbers are large.** Against the model alone on
  GSM-Hard it is right on 680 more problems and wrong on 161 more (p = 1e-76), and it beats both `check`
  and `solve` for 10 of the 13 models. Each branch earns its place there: when ADA found a wrong equation in
  the model's answer, ADA's answer is right on 303 of 735 problems, the model's on 5 and the checked one on
  127; on the questions with large numbers, ADA's is right on 357 of 1,103 and the model's on 155.
- **With small numbers it costs little and gains little.** On GSM8K the rule mostly keeps the model's own
  answer. It falls short of `check` because of the large-numbers branch: GSM8K's few questions with a
  five-digit number (a house that costs $80,000) are not hard arithmetic, and there ADA's answer is right on
  6 of 47 and the model's on 20. This is seen after the test, so it is a lead for a later rule, not a change
  to this one.
- **Letting the model decide is worse than a fixed rule.** Shown ADA's equations and value beside its own
  answer, the model often switches to ADA's wrong answer on GSM8K (137 right only after deciding, 241 right
  only alone) and gains far less than the rule on GSM-Hard (1,574 against 1,924).
- **When the model's answer and ADA's agree, they are nearly always right**: 1,433 of 1,525 such problems
  on GSM8K (94%) and 831 of 934 on GSM-Hard (89%), against 67% and 36% for the model alone over all
  problems. Agreement is a strong sign that an answer can be trusted.

## What to keep in mind

- One reply per problem and condition at temperature 0. The pooled differences on GSM-Hard are far beyond
  chance; the 16-answer shortfall against `check` on GSM8K is not tested here as a difference in itself.
- `combine` asks for no more replies than `none`, `check` and `solve` already made; `judge` asks for one more
  reply on each problem where the two answers differ.
- `solve` here is the same condition as in [2026-11-05-gsm-ladder](../2026-11-05-gsm-ladder/README.md), at a
  later commit with the same reading rules, on different problems.

## Before and after

- Before it: [2026-11-05-gsm-ladder](../2026-11-05-gsm-ladder/README.md), whose problems 0 to 299 the rule
  was chosen on, in [analyses/2026-11-05-combine-rules](../../analyses/2026-11-05-combine-rules/README.md).
