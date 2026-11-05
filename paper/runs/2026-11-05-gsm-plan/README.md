# GSM8K and GSM-Hard, two ways to write better equations (2026-11-05)

**Question.** Under `solve`, the 1.5B to 4B models often write equations ADA can find no one value for, or
equations that are not the problem's. Do more of their answers come out right when they first work out in
words how to solve the problem (`plan`), or when ADA tells them which quantities have no value and they
write the equations again (`retry`)? And does either help the rule that combines the model's own answer
with ADA's?

**Answer in short.** `retry` gets more right than `solve`: pooled over the 7 models, 1,227 against 1,193 on
GSM8K and 1,259 against 1,207 on GSM-Hard. Most of that comes from asking again: of the problems asked again
after the same first reply `solve` got, 28 of 194 are right afterwards on GSM8K and 41 of 171 on GSM-Hard. Two
replies to the same message also differ, on about a third of the problems in this run, but those differences
go about as often one way as the other. `plan` helps the two qwen3 models and hurts llama3.2:3b and
qwen2.5:3b; pooled, it is no better than `solve` on GSM8K and worse on GSM-Hard. Inside the combining rule
neither changes much: on GSM-Hard, 1,390 right with `retry`'s answers and 1,374 with `plan`'s, against 1,385
with `solve`'s.

## What ran

| | |
|---|---|
| Data | GSM8K's `test.jsonl` (sha256 `3730d312…c39d14`) and GSM-Hard's `gsmhardv2.jsonl` (sha256 `2032b018…e9ddef`), problems 0 to 299 of each (`--problems 300`). GSM8K is matched exactly, GSM-Hard to within 0.001 |
| Models | 7 of the 13 of [2026-11-05-gsm-ladder](../2026-11-05-gsm-ladder/README.md), from 1.5B up: qwen2.5:1.5b, qwen3:1.7b, llama3.2:3b, qwen2.5:3b, phi4-mini, gemma3:4b and qwen3:4b-instruct |
| Conditions | `solve` (the model writes the problem as equations and ADA solves them), `plan` (as `solve`, but the model first works out in words which quantities the problem gives, which it asks for and how they are related; the four worked examples are reasoned out the same way) and `retry` (as `solve`, but when ADA finds no one value, the model is told why, such as the quantities with no value, and writes the equations again, once); see [benchmarks/README.md](../../../benchmarks/README.md) |
| Settings | Temperature 0, seed 0, context 8192 tokens, at most 2048 tokens a reply, one reply per problem and condition (two under `retry` when it asks again) |
| Where | Kaggle kernel `saineshnakra/ada-benchmark-plan`, version 1: two T4 GPUs with one Ollama server each, six shards per model and data set ([kaggle_ladder.py](../../../benchmarks/kaggle_ladder.py)); started 15:44 UTC and ran for 73 minutes |
| Code | Commit `9ad2943` on branch `plan-then-equations` |

[config.json](config.json) has these settings in full. [results.jsonl.gz](results.jsonl.gz) holds 12,600
records. A `retry` record that asked again keeps the first reply and what ADA made of it
(`first_formal`, `first_ada`) and what the model was told (`feedback`).

## Results

Made with `python paper/analysis.py paper/runs/2026-11-05-gsm-plan`. The run did not ask the model alone,
so each condition is set against `solve`.

### GSM8K, problems 0 to 299

Problems right, on the problems every condition finished:

| Model | Problems | solve | plan | retry |
|---|---|---|---|---|
| gemma3:4b | 300 | 152 | 162 | 154 |
| llama3.2:3b | 300 | 201 | 159 | 198 |
| phi4-mini | 300 | 176 | 164 | 185 |
| qwen2.5:1.5b | 300 | 133 | 123 | 136 |
| qwen2.5:3b | 300 | 187 | 154 | 191 |
| qwen3:1.7b | 300 | 121 | 187 | 129 |
| qwen3:4b-instruct | 300 | 223 | 259 | 234 |

Against solve: right only with the condition / right only under solve, and p:

| Model | plan | retry |
|---|---|---|
| gemma3:4b | 46 / 36, p = 0.32 | 13 / 11, p = 0.84 |
| llama3.2:3b | 26 / 68, p = 2e-5 | 6 / 9, p = 0.61 |
| phi4-mini | 39 / 51, p = 0.25 | 18 / 9, p = 0.12 |
| qwen2.5:1.5b | 39 / 49, p = 0.34 | 15 / 12, p = 0.7 |
| qwen2.5:3b | 37 / 70, p = 0.0018 | 17 / 13, p = 0.58 |
| qwen3:1.7b | 90 / 24, p = 4e-10 | 20 / 12, p = 0.22 |
| qwen3:4b-instruct | 45 / 9, p = 7e-7 | 14 / 3, p = 0.013 |
| All | 322 / 307, p = 0.58 | 103 / 69, p = 0.012 |

ADA's statuses under solve:

- gemma3:4b: solved 235, no value 59, refused 6
- llama3.2:3b: solved 278, no value 15, refused 4, no answer line 3
- phi4-mini: solved 228, no value 58, refused 9, no answer line 3, number given 2
- qwen2.5:1.5b: solved 247, no value 40, refused 9, no answer line 3, number given 1
- qwen2.5:3b: solved 260, no value 29, refused 11
- qwen3:1.7b: solved 243, no value 45, refused 11, number given 1
- qwen3:4b-instruct: solved 266, no value 26, refused 8

ADA's statuses under plan:

- gemma3:4b: solved 212, no value 51, refused 19, number given 18
- llama3.2:3b: solved 254, no value 31, refused 9, number given 4, no answer line 2
- phi4-mini: solved 175, no value 77, no answer line 20, number given 20, refused 8
- qwen2.5:1.5b: solved 239, no value 30, no answer line 15, number given 9, refused 7
- qwen2.5:3b: solved 174, no value 98, refused 19, number given 9
- qwen3:1.7b: solved 152, number given 106, no value 25, refused 17
- qwen3:4b-instruct: solved 286, no value 12, number given 2

ADA's statuses under retry:

- gemma3:4b: solved 244, no value 49, refused 7
- llama3.2:3b: solved 289, no value 8, refused 3
- phi4-mini: solved 241, no value 54, no answer line 2, refused 2, number given 1
- qwen2.5:1.5b: solved 251, no value 37, refused 4, example repeated 3, no answer line 3, number given 2
- qwen2.5:3b: solved 279, no value 19, refused 2
- qwen3:1.7b: solved 273, no value 22, refused 2, example repeated 2, number given 1
- qwen3:4b-instruct: solved 278, no value 18, refused 4

retry against solve. retry's first message is solve's, so where its first reply is solve's reply, the problems asked again show what asking again does; where it is not, the two replies differ only as two replies to the same message can:

| Model | Asked again | First reply as under solve, asked again | Right after asking again | First reply not as under solve | Right only under retry / only under solve |
|---|---|---|---|---|---|
| gemma3:4b | 64 | 39 | 3 | 89 | 10 / 11 |
| llama3.2:3b | 20 | 12 | 1 | 67 | 5 / 9 |
| phi4-mini | 72 | 42 | 8 | 93 | 10 / 9 |
| qwen2.5:1.5b | 55 | 36 | 2 | 94 | 13 / 12 |
| qwen2.5:3b | 38 | 16 | 5 | 129 | 12 / 13 |
| qwen3:1.7b | 46 | 26 | 3 | 139 | 17 / 12 |
| qwen3:4b-instruct | 30 | 23 | 6 | 65 | 8 / 3 |
| All | 325 | 194 | 28 | 676 | 75 / 69 |

Replies that failed or never ended: 0

### GSM-Hard, problems 0 to 299

Problems right, on the problems every condition finished:

| Model | Problems | solve | plan | retry |
|---|---|---|---|---|
| gemma3:4b | 300 | 149 | 167 | 153 |
| llama3.2:3b | 300 | 191 | 154 | 207 |
| phi4-mini | 300 | 182 | 172 | 192 |
| qwen2.5:1.5b | 300 | 134 | 124 | 143 |
| qwen2.5:3b | 300 | 191 | 138 | 191 |
| qwen3:1.7b | 300 | 138 | 160 | 138 |
| qwen3:4b-instruct | 300 | 222 | 239 | 235 |

Against solve: right only with the condition / right only under solve, and p:

| Model | plan | retry |
|---|---|---|
| gemma3:4b | 49 / 31, p = 0.057 | 15 / 11, p = 0.56 |
| llama3.2:3b | 18 / 55, p = 2e-5 | 19 / 3, p = 9e-4 |
| phi4-mini | 37 / 47, p = 0.33 | 16 / 6, p = 0.052 |
| qwen2.5:1.5b | 37 / 47, p = 0.33 | 18 / 9, p = 0.12 |
| qwen2.5:3b | 28 / 81, p = 4e-7 | 18 / 18, p = 1 |
| qwen3:1.7b | 56 / 34, p = 0.026 | 19 / 19, p = 1 |
| qwen3:4b-instruct | 37 / 20, p = 0.033 | 15 / 2, p = 0.0023 |
| All | 262 / 315, p = 0.03 | 120 / 68, p = 2e-4 |

ADA's statuses under solve:

- gemma3:4b: solved 239, no value 53, refused 8
- llama3.2:3b: solved 278, no value 15, refused 5, no answer line 2
- phi4-mini: solved 237, no value 55, refused 5, no answer line 2, number given 1
- qwen2.5:1.5b: solved 249, no value 35, refused 12, no answer line 3, number given 1
- qwen2.5:3b: solved 276, no value 18, refused 6
- qwen3:1.7b: solved 267, no value 21, refused 11, number given 1
- qwen3:4b-instruct: solved 276, no value 17, refused 7

ADA's statuses under plan:

- gemma3:4b: solved 232, no value 47, refused 15, number given 5, no answer line 1
- llama3.2:3b: solved 261, no value 30, refused 6, number given 2, no answer line 1
- phi4-mini: solved 208, no value 70, no answer line 11, number given 8, refused 3
- qwen2.5:1.5b: solved 237, no value 38, refused 11, no answer line 8, number given 6
- qwen2.5:3b: solved 181, no value 99, refused 13, number given 5, no answer line 2
- qwen3:1.7b: solved 222, number given 40, no value 26, refused 12
- qwen3:4b-instruct: solved 274, no value 18, refused 5, number given 2, no answer line 1

ADA's statuses under retry:

- gemma3:4b: solved 257, no value 36, refused 7
- llama3.2:3b: solved 293, no value 3, no answer line 2, refused 2
- phi4-mini: solved 256, no value 40, number given 2, no answer line 1, refused 1
- qwen2.5:1.5b: solved 270, no value 23, refused 3, number given 2, example repeated 2
- qwen2.5:3b: solved 283, no value 12, refused 4, number given 1
- qwen3:1.7b: solved 281, no value 14, refused 4, number given 1
- qwen3:4b-instruct: solved 288, no value 7, refused 4, example repeated 1

retry against solve. retry's first message is solve's, so where its first reply is solve's reply, the problems asked again show what asking again does; where it is not, the two replies differ only as two replies to the same message can:

| Model | Asked again | First reply as under solve, asked again | Right after asking again | First reply not as under solve | Right only under retry / only under solve |
|---|---|---|---|---|---|
| gemma3:4b | 61 | 43 | 10 | 83 | 5 / 11 |
| llama3.2:3b | 18 | 16 | 9 | 59 | 10 / 3 |
| phi4-mini | 56 | 34 | 6 | 90 | 10 / 6 |
| qwen2.5:1.5b | 41 | 28 | 5 | 98 | 13 / 9 |
| qwen2.5:3b | 27 | 11 | 1 | 117 | 17 / 18 |
| qwen3:1.7b | 36 | 22 | 5 | 123 | 14 / 19 |
| qwen3:4b-instruct | 25 | 17 | 5 | 61 | 10 / 2 |
| All | 264 | 171 | 41 | 631 | 79 / 68 |

Replies that failed or never ended: 0

### Inside the combining rule

`benchmarks/gsm8k.py`'s `combine` ("check or large numbers") with ADA's answers from each of this run's
conditions in turn, and the model's own answer and `check` from
[2026-11-05-gsm-ladder](../2026-11-05-gsm-ladder/README.md), which asked the same 7 models the same problems.
Made with `python paper/combine_rules.py paper/runs/2026-11-05-gsm-ladder --gsm8k test.jsonl --gsmhard
gsmhardv2.jsonl --ada paper/runs/2026-11-05-gsm-plan:CONDITION`, pooled over the 7 models:

| Data | none | check | ADA's answers from | Right under it | combine | either right |
|---|---|---|---|---|---|---|
| GSM8K | 1,727 | 1,737 | solve | 1,193 | 1,733 | 1,822 |
| GSM8K | | | plan | 1,208 | 1,736 | 1,816 |
| GSM8K | | | retry | 1,227 | 1,733 | 1,820 |
| GSM-Hard | 925 | 1,024 | solve | 1,207 | 1,385 | 1,492 |
| GSM-Hard | | | plan | 1,154 | 1,374 | 1,438 |
| GSM-Hard | | | retry | 1,259 | 1,390 | 1,520 |

## What it shows

- **Asking again fixes some of the equations ADA can't solve, and loses none of them.** When `retry`'s
  first reply is the one `solve` got and ADA finds no one value, asking again makes 28 of 194 such problems
  right on GSM8K (14%) and 41 of 171 on GSM-Hard (24%). On these problems `solve` has no answer, so asking
  again can only gain.
- **Two replies to the same message differ often.** `retry`'s first message is `solve`'s, at the same
  temperature and seed, yet its first reply is another reply on 676 of 2,100 GSM8K problems and 631 of
  2,100 GSM-Hard ones. On those, `retry` is right where `solve` is not on 75 and 79, and the other way on
  69 and 68: about as many each way, as when two replies differ by chance. For one model the difference this
  makes is as large as 8 answers. The McNemar tests above allow for it. Why the replies differ is not tested
  here. The six shards of a model ask one Ollama server side by side, and a reply computed alongside other
  requests can come out differently.
- **Working it out in words first suits some models and not others.** Under `plan`, qwen3:1.7b gets 187
  right on GSM8K against 121 under `solve` (90 / 24, p = 4e-10), and qwen3:4b-instruct 259 against 223. But
  llama3.2:3b falls from 201 to 159 and qwen2.5:3b from 187 to 154, and on GSM-Hard both fall too, from 191
  to 154 and from 191 to 138. On GSM-Hard under `plan`, qwen2.5:3b leaves 99 problems with no value against
  18 under `solve`, and qwen3:1.7b works 40 answers out itself instead of writing equations.
- **Neither helps the combining rule.** The rule takes ADA's answer only when ADA found a wrong equation in
  the model's own answer or the question has large numbers. Better equations on the other problems change
  little: with `retry`'s answers in place of `solve`'s it gets 0 more right on GSM8K and 5 more on GSM-Hard,
  with `plan`'s 3 more and 11 fewer.

## What to keep in mind

- These are the problems the two ways were tried on. What was chosen from them, and what a test on new
  problems should show, is in
  [analyses/2026-11-05-plan-and-retry](../../analyses/2026-11-05-plan-and-retry/README.md), written before
  that test.
- `solve` here is the same condition as in [2026-11-05-gsm-ladder](../2026-11-05-gsm-ladder/README.md), on
  the same problems, a few commits later; ADA's answers are now written in digits only. For these 7 models
  `solve` gets 1,193 right on GSM8K here and 1,215 there, and 1,207 and 1,222 on GSM-Hard: two runs of one
  condition differ by about as much as two replies to the same message do.
- `retry` asks for a second reply on the problems it asks again: 325 of 2,100 on GSM8K and 264 of 2,100 on
  GSM-Hard.

## Before and after

- Before it: [2026-11-05-gsm-ladder](../2026-11-05-gsm-ladder/README.md), whose `solve` statuses showed how
  often these models write equations with no one value, and
  [2026-11-05-gsm-combine](../2026-11-05-gsm-combine/README.md), which tested the combining rule.
- After it: [2026-11-05-gsm-plan-check](../2026-11-05-gsm-plan-check/README.md), the same conditions on
  problems 300 to 599, where all five tests set from this run held.
