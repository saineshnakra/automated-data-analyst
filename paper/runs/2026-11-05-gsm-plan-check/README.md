# GSM8K and GSM-Hard, plan and retry tested on new problems (2026-11-05)

**Question.** On problems 0 to 299, writing the equations again when ADA finds no value (`retry`) got more
right than `solve`, and working the problem out in words first (`plan`) helped some models and hurt
others. Do both hold on problems 300 to 599, which neither was tried on?

**Answer in short.** Yes: all five tests set beforehand in
[analyses/2026-11-05-plan-and-retry](../../analyses/2026-11-05-plan-and-retry/README.md) hold. `retry` gets
1,215 right against 1,185 for `solve` on GSM8K and 1,216 against 1,175 on GSM-Hard, pooled over the 7
models. Asked again after the first reply `solve` got, 30 of 170 problems come out right on GSM8K and 27 of
148 on GSM-Hard. `plan` helps the two qwen3 models and hurts llama3.2:3b and qwen2.5:3b on GSM8K, as
before. And inside the combining rule, `retry`'s answers change nothing that matters: 2 fewer right on each
data set.

## What ran

| | |
|---|---|
| Data | GSM8K's `test.jsonl` (sha256 `3730d312…c39d14`) and GSM-Hard's `gsmhardv2.jsonl` (sha256 `2032b018…e9ddef`), problems 300 to 599 of each (`--problems 600 --skip 300`). GSM8K is matched exactly, GSM-Hard to within 0.001 |
| Models | The 7 of [2026-11-05-gsm-plan](../2026-11-05-gsm-plan/README.md): qwen2.5:1.5b, qwen3:1.7b, llama3.2:3b, qwen2.5:3b, phi4-mini, gemma3:4b and qwen3:4b-instruct |
| Conditions | `solve`, `plan` and `retry`, as in [2026-11-05-gsm-plan](../2026-11-05-gsm-plan/README.md); see [benchmarks/README.md](../../../benchmarks/README.md) |
| Settings | Temperature 0, seed 0, context 8192 tokens, at most 2048 tokens a reply, one reply per problem and condition (two under `retry` when it asks again) |
| Where | Kaggle kernel `saineshnakra/ada-benchmark-plan-check`, version 1: two T4 GPUs with one Ollama server each, six shards per model and data set ([kaggle_ladder.py](../../../benchmarks/kaggle_ladder.py)); started 18:26 UTC and ran for 72 minutes. One shard stopped early and was run again in kernel `saineshnakra/ada-benchmark-plan-check-rerun`, version 1, started 19:51 UTC, 7 minutes |
| Code | Commit `9ad2943` on branch `plan-then-equations`, the trial's code; the shard run again at commit `23eb2c9` on `main` |
| Tested against | [analyses/2026-11-05-plan-and-retry](../../analyses/2026-11-05-plan-and-retry/README.md), written before this run |

[config.json](config.json) has these settings in full. [results.jsonl.gz](results.jsonl.gz) holds 12,600
records.

### The shard run again

phi4-mini's GSM-Hard shard 2 stopped at problem 548, 8 problems short: one of its replies made SymPy's
parser raise `IndexError`, which the trial's code did not catch.
[#182](https://github.com/saineshnakra/automated-data-analyst/pull/182) catches it, so the shard was run again
from its start at a commit of `main` that has both #182 and the conditions as they are here
([#183](https://github.com/saineshnakra/automated-data-analyst/pull/183)). Its 150 records replace the 126 of
the shard that stopped. Apart from replies like that one, which now have no value, the code reads every
reply as the trial's did.

## Results

Made with `python paper/analysis.py paper/runs/2026-11-05-gsm-plan-check`. The run did not ask the model
alone, so each condition is set against `solve`.

### GSM8K, problems 300 to 599

Problems right, on the problems every condition finished:

| Model | Problems | solve | plan | retry |
|---|---|---|---|---|
| gemma3:4b | 300 | 158 | 183 | 162 |
| llama3.2:3b | 300 | 178 | 163 | 175 |
| phi4-mini | 300 | 178 | 135 | 186 |
| qwen2.5:1.5b | 300 | 129 | 124 | 133 |
| qwen2.5:3b | 300 | 181 | 151 | 187 |
| qwen3:1.7b | 300 | 130 | 166 | 136 |
| qwen3:4b-instruct | 300 | 231 | 250 | 236 |

Against solve: right only with the condition / right only under solve, and p:

| Model | plan | retry |
|---|---|---|
| gemma3:4b | 53 / 28, p = 0.0073 | 10 / 6, p = 0.45 |
| llama3.2:3b | 26 / 41, p = 0.086 | 7 / 10, p = 0.63 |
| phi4-mini | 31 / 74, p = 3e-5 | 18 / 10, p = 0.18 |
| qwen2.5:1.5b | 35 / 40, p = 0.64 | 14 / 10, p = 0.54 |
| qwen2.5:3b | 39 / 69, p = 0.005 | 15 / 9, p = 0.31 |
| qwen3:1.7b | 69 / 33, p = 5e-4 | 18 / 12, p = 0.36 |
| qwen3:4b-instruct | 34 / 15, p = 0.0094 | 9 / 4, p = 0.27 |
| All | 287 / 300, p = 0.62 | 91 / 61, p = 0.018 |

ADA's statuses under solve:

- gemma3:4b: solved 246, no value 48, refused 6
- llama3.2:3b: solved 277, no value 16, refused 7
- phi4-mini: solved 224, no value 58, refused 12, number given 5, no answer line 1
- qwen2.5:1.5b: solved 245, no value 29, refused 16, no answer line 6, number given 4
- qwen2.5:3b: solved 269, no value 24, refused 5, no answer line 2
- qwen3:1.7b: solved 258, no value 26, refused 15, number given 1
- qwen3:4b-instruct: solved 283, no value 13, refused 4

ADA's statuses under plan:

- gemma3:4b: solved 231, no value 45, number given 14, refused 10
- llama3.2:3b: solved 256, no value 28, refused 10, number given 5, no answer line 1
- phi4-mini: solved 162, no value 109, no answer line 19, number given 8, refused 2
- qwen2.5:1.5b: solved 235, no value 33, number given 13, no answer line 10, refused 9
- qwen2.5:3b: solved 179, no value 100, refused 13, number given 8
- qwen3:1.7b: solved 138, number given 113, no value 26, refused 23
- qwen3:4b-instruct: solved 279, no value 18, refused 3

ADA's statuses under retry:

- gemma3:4b: solved 255, no value 42, refused 3
- llama3.2:3b: solved 293, refused 4, no value 2, no answer line 1
- phi4-mini: solved 240, no value 50, number given 6, no answer line 2, example repeated 1, refused 1
- qwen2.5:1.5b: solved 260, no value 22, refused 10, number given 4, example repeated 2, no answer line 2
- qwen2.5:3b: solved 278, no value 17, refused 4, no answer line 1
- qwen3:1.7b: solved 279, no value 13, refused 8
- qwen3:4b-instruct: solved 291, no value 5, refused 4

retry against solve. retry's first message is solve's, so where its first reply is solve's reply, the problems asked again show what asking again does; where it is not, the two replies differ only as two replies to the same message can:

| Model | Asked again | First reply as under solve, asked again | Right after asking again | First reply not as under solve | Right only under retry / only under solve |
|---|---|---|---|---|---|
| gemma3:4b | 54 | 42 | 3 | 63 | 7 / 6 |
| llama3.2:3b | 22 | 20 | 3 | 56 | 4 / 10 |
| phi4-mini | 67 | 39 | 9 | 99 | 9 / 10 |
| qwen2.5:1.5b | 45 | 28 | 3 | 119 | 11 / 10 |
| qwen2.5:3b | 36 | 18 | 4 | 128 | 11 / 9 |
| qwen3:1.7b | 35 | 10 | 3 | 148 | 15 / 12 |
| qwen3:4b-instruct | 18 | 13 | 5 | 71 | 4 / 4 |
| All | 277 | 170 | 30 | 684 | 61 / 61 |

Replies that failed or never ended: 0

### GSM-Hard, problems 300 to 599

Problems right, on the problems every condition finished:

| Model | Problems | solve | plan | retry |
|---|---|---|---|---|
| gemma3:4b | 300 | 128 | 150 | 143 |
| llama3.2:3b | 300 | 178 | 153 | 177 |
| phi4-mini | 300 | 180 | 137 | 186 |
| qwen2.5:1.5b | 300 | 136 | 125 | 139 |
| qwen2.5:3b | 300 | 189 | 119 | 202 |
| qwen3:1.7b | 300 | 138 | 150 | 135 |
| qwen3:4b-instruct | 300 | 226 | 233 | 234 |

Against solve: right only with the condition / right only under solve, and p:

| Model | plan | retry |
|---|---|---|
| gemma3:4b | 56 / 34, p = 0.026 | 17 / 2, p = 7e-4 |
| llama3.2:3b | 26 / 51, p = 0.0059 | 10 / 11, p = 1 |
| phi4-mini | 26 / 69, p = 1e-5 | 14 / 8, p = 0.29 |
| qwen2.5:1.5b | 36 / 47, p = 0.27 | 10 / 7, p = 0.63 |
| qwen2.5:3b | 24 / 94, p = 6e-11 | 19 / 6, p = 0.015 |
| qwen3:1.7b | 57 / 45, p = 0.28 | 15 / 18, p = 0.73 |
| qwen3:4b-instruct | 25 / 18, p = 0.36 | 9 / 1, p = 0.021 |
| All | 250 / 358, p = 1e-5 | 94 / 53, p = 9e-4 |

ADA's statuses under solve:

- gemma3:4b: solved 253, no value 43, refused 4
- llama3.2:3b: solved 282, no value 11, refused 6, no answer line 1
- phi4-mini: solved 246, no value 47, refused 7
- qwen2.5:1.5b: solved 257, no value 23, refused 15, no answer line 4, number given 1
- qwen2.5:3b: solved 270, no value 18, refused 12
- qwen3:1.7b: solved 278, no value 11, refused 10, number given 1
- qwen3:4b-instruct: solved 282, no value 13, refused 5

ADA's statuses under plan:

- gemma3:4b: solved 228, no value 42, refused 18, number given 11, no answer line 1
- llama3.2:3b: solved 255, no value 31, refused 7, number given 5, no answer line 2
- phi4-mini: solved 183, no value 88, number given 14, no answer line 9, refused 6
- qwen2.5:1.5b: solved 241, no value 35, refused 15, number given 5, no answer line 4
- qwen2.5:3b: solved 162, no value 111, refused 24, number given 3
- qwen3:1.7b: solved 204, number given 39, no value 39, refused 18
- qwen3:4b-instruct: solved 277, no value 19, refused 3, no answer line 1

ADA's statuses under retry:

- gemma3:4b: solved 269, no value 28, refused 3
- llama3.2:3b: solved 289, no value 5, refused 4, no answer line 2
- phi4-mini: solved 257, no value 37, no answer line 3, refused 3
- qwen2.5:1.5b: solved 262, no value 21, refused 9, example repeated 5, no answer line 2, number given 1
- qwen2.5:3b: solved 286, no value 8, refused 5, no answer line 1
- qwen3:1.7b: solved 289, refused 6, no value 5
- qwen3:4b-instruct: solved 292, no value 4, refused 3, example repeated 1

retry against solve. retry's first message is solve's, so where its first reply is solve's reply, the problems asked again show what asking again does; where it is not, the two replies differ only as two replies to the same message can:

| Model | Asked again | First reply as under solve, asked again | Right after asking again | First reply not as under solve | Right only under retry / only under solve |
|---|---|---|---|---|---|
| gemma3:4b | 44 | 31 | 6 | 80 | 11 / 2 |
| llama3.2:3b | 17 | 12 | 2 | 69 | 8 / 11 |
| phi4-mini | 54 | 33 | 6 | 91 | 8 / 8 |
| qwen2.5:1.5b | 45 | 29 | 3 | 87 | 7 / 7 |
| qwen2.5:3b | 23 | 16 | 2 | 109 | 17 / 6 |
| qwen3:1.7b | 25 | 14 | 3 | 138 | 12 / 18 |
| qwen3:4b-instruct | 17 | 13 | 5 | 60 | 4 / 1 |
| All | 225 | 148 | 27 | 634 | 67 / 53 |

Replies that failed or never ended: 0

### Inside the combining rule

`combine` ("check or large numbers") with ADA's answers from each of this run's conditions in turn, and the
model's own answer and `check` from [2026-11-05-gsm-combine](../2026-11-05-gsm-combine/README.md), which
asked the same models the same problems. Made with `python paper/combine_rules.py
paper/runs/2026-11-05-gsm-combine --gsm8k test.jsonl --gsmhard gsmhardv2.jsonl --ada
paper/runs/2026-11-05-gsm-plan-check:CONDITION`, pooled over the 7 models:

| Data | none | check | ADA's answers from | Right under it | combine | either right |
|---|---|---|---|---|---|---|
| GSM8K | 1,682 | 1,699 | solve | 1,185 | 1,688 | 1,784 |
| GSM8K | | | plan | 1,172 | 1,685 | 1,783 |
| GSM8K | | | retry | 1,215 | 1,686 | 1,777 |
| GSM-Hard | 943 | 1,043 | solve | 1,175 | 1,347 | 1,454 |
| GSM-Hard | | | plan | 1,067 | 1,335 | 1,400 |
| GSM-Hard | | | retry | 1,216 | 1,345 | 1,467 |

## The tests set before the run

| Test | Result | Held? |
|---|---|---|
| `retry` gets more right than `solve` | GSM8K 1,215 against 1,185 (91 / 61, p = 0.018); GSM-Hard 1,216 against 1,175 (94 / 53, p = 9e-4) | Yes |
| Asking again fixes equations ADA can't solve | of the problems asked again after the first reply `solve` got, 30 of 170 right afterwards on GSM8K (18%) and 27 of 148 on GSM-Hard (18%) | Yes |
| `plan` helps the qwen3 models | GSM8K: qwen3:1.7b 166 against 130 (69 / 33, p = 5e-4), qwen3:4b-instruct 250 against 231 (34 / 15, p = 0.0094) | Yes |
| `plan` hurts llama3.2:3b and qwen2.5:3b | GSM8K: llama3.2:3b 163 against 178 (26 / 41, p = 0.086), qwen2.5:3b 151 against 181 (39 / 69, p = 0.005) | Yes |
| Neither helps the combining rule | with `retry`'s answers, `combine` gets 1,686 against 1,688 with `solve`'s on GSM8K and 1,345 against 1,347 on GSM-Hard | Yes |

## What it shows

- **Telling the model why ADA found no value, and asking once more, is a small, real gain for `solve`.**
  On both sets of problems and both data sets, it turns 14% to 24% of the problems it asks again into right
  answers, and `retry` beats `solve` by 30 to 52 answers in 2,100 each time. It costs a second reply on 11%
  to 16% of problems.
- **Whether working it out in words first helps depends on the model, the same way twice.** The two qwen3
  models gain and llama3.2:3b and qwen2.5:3b lose on both sets of problems. gemma3:4b gains on both too
  (183 against 158 here, 162 against 152 before), and phi4-mini loses on both (135 against 178 here).
  Pooled over the 7 models `plan` is no better than `solve` on GSM8K and worse on GSM-Hard, both times.
- **Better equations don't make the combining rule better.** Of the problems `retry` gets right and `solve`
  does not, the model alone already gets 82 of 91 right on GSM8K and 50 of 94 on GSM-Hard, so the rule
  gains nothing there. With `retry`'s answers it gets 2 fewer right on each data set.
- **Two replies to the same message differ as often as before.** `retry`'s first reply is not `solve`'s on
  684 of 2,100 GSM8K problems and 634 of 2,100 GSM-Hard ones, and those go as often one way as the other (61
  / 61 and 67 / 53).

## What to keep in mind

- One reply per problem and condition at temperature 0. Per model, most of `retry`'s gains are within
  chance; the pooled ones are not.
- The tests named models for `plan` because the trial showed them; gemma3:4b and phi4-mini were not named,
  so what this run shows for them is a lead, not a test.

## Before and after

- Before it: [2026-11-05-gsm-plan](../2026-11-05-gsm-plan/README.md), the trial on problems 0 to 299, and
  [2026-11-05-gsm-combine](../2026-11-05-gsm-combine/README.md), whose `none` and `check` for these problems
  the combining rule uses here.
