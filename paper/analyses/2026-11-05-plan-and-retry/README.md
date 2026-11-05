# Choosing a way to write better equations (2026-11-05)

**Question.** Under `solve`, the 1.5B to 4B models often write equations ADA can find no one value for.
Which of two changes gives more right answers: the model working out in words how to solve the problem
before it writes the equations (`plan`), or ADA telling it which quantities have no value and the model
writing the equations again (`retry`)?

This page records what was chosen and what a test on new problems should show, **before** that test runs.

## What was tried

[2026-11-05-gsm-plan](../../runs/2026-11-05-gsm-plan/README.md) asked 7 models (qwen2.5:1.5b, qwen3:1.7b,
llama3.2:3b, qwen2.5:3b, phi4-mini, gemma3:4b and qwen3:4b-instruct) GSM8K and GSM-Hard problems 0 to 299
under `solve`, `plan` and `retry`. Pooled over the 7 models:

| Data | solve | plan | retry | retry against solve | Asked again after solve's own first reply, right afterwards |
|---|---|---|---|---|---|
| GSM8K | 1,193 | 1,208 | 1,227 | 103 / 69, p = 0.012 | 28 of 194 |
| GSM-Hard | 1,207 | 1,154 | 1,259 | 120 / 68, p = 2e-4 | 41 of 171 |

`plan` against `solve`, per model, on GSM8K: qwen3:1.7b 90 / 24 and qwen3:4b-instruct 45 / 9, but
llama3.2:3b 26 / 68 and qwen2.5:3b 37 / 70. GSM-Hard goes the same way for these four.

With each condition's answers as ADA's in the combining rule (`combine`, "check or large numbers"), and
the model's own answer and `check` from [2026-11-05-gsm-ladder](../../runs/2026-11-05-gsm-ladder/README.md):

| Data | with solve | with plan | with retry |
|---|---|---|---|
| GSM8K | 1,733 | 1,736 | 1,733 |
| GSM-Hard | 1,385 | 1,374 | 1,390 |

## What was chosen

- **`retry` is the better way for ADA's answer alone.** It gets more right than `solve` on both data sets,
  most of the gain comes from the problems it asks again, and it asks for a second reply on only 15% of
  GSM8K problems and 13% of GSM-Hard ones.
- **`plan` is not chosen for every model.** It helps the two qwen3 models and hurts llama3.2:3b and
  qwen2.5:3b. Whether that holds on new problems is tested below, but no rule for picking models is set on
  these four.
- **The combining rule keeps `solve`.** Neither change helps it, so it is not worth a second reply there.

## The test on new problems

GSM8K and GSM-Hard problems 300 to 599, which neither way was tried on (`--problems 600 --skip 300`). The
same 7 models, `solve`, `plan` and `retry`, at the same commit (`9ad2943`) and settings as the trial. The
model's own answer and `check` for these problems come from
[2026-11-05-gsm-combine](../../runs/2026-11-05-gsm-combine/README.md). Pooled over the 7 models, unless a
test names a model:

| Test | Holds when |
|---|---|
| `retry` gets more right than `solve` | on GSM8K and on GSM-Hard, each |
| Asking again fixes equations ADA can't solve | of the problems asked again after the first reply `solve` got, at least 10% are right afterwards, on each data set |
| `plan` helps the qwen3 models | on GSM8K, `plan` gets more right than `solve` for qwen3:1.7b and for qwen3:4b-instruct |
| `plan` hurts llama3.2:3b and qwen2.5:3b | on GSM8K, `plan` gets fewer right than `solve` for each of the two |
| Neither helps the combining rule | with `retry`'s answers in place of `solve`'s, `combine` gets within 15 answers of its total with the same run's `solve` answers, on each data set |

The tables will be made with `python paper/analysis.py` on the new run and `python paper/combine_rules.py
paper/runs/2026-11-05-gsm-combine --gsm8k test.jsonl --gsmhard gsmhardv2.jsonl --ada NEW_RUN:CONDITION`.

## How it came out

On problems 300 to 599 ([2026-11-05-gsm-plan-check](../../runs/2026-11-05-gsm-plan-check/README.md)), all
five tests held:

| Test | Result | Held? |
|---|---|---|
| `retry` gets more right than `solve` | GSM8K 1,215 against 1,185 (91 / 61, p = 0.018); GSM-Hard 1,216 against 1,175 (94 / 53, p = 9e-4) | Yes |
| Asking again fixes equations ADA can't solve | 30 of 170 right afterwards on GSM8K (18%), 27 of 148 on GSM-Hard (18%) | Yes |
| `plan` helps the qwen3 models | GSM8K: qwen3:1.7b 166 against 130, qwen3:4b-instruct 250 against 231 | Yes |
| `plan` hurts llama3.2:3b and qwen2.5:3b | GSM8K: 163 against 178, and 151 against 181 | Yes |
| Neither helps the combining rule | with `retry`'s answers, `combine` gets 1,686 against 1,688 on GSM8K and 1,345 against 1,347 on GSM-Hard | Yes |

One shard of the run stopped on a reply SymPy's parser could not read, and was run again with the fix of
[#182](https://github.com/saineshnakra/automated-data-analyst/pull/182); its run page says how.
