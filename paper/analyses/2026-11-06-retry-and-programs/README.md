# Asking again on numbers, and programs, tested on new problems (2026-11-06)

**Question.** Two changes to the GSM8K benchmark came out of the runs so far
([#186](https://github.com/saineshnakra/automated-data-analyst/pull/186)): `retry` now also asks again when
the model's Answer line gives a number it worked out instead of a quantity of its equations, and a `program`
condition has the model write a Python program that is run, the usual way of giving a model exact arithmetic
without ADA. Do they help, and how does `solve` compare with `program`? And does the combining rule hold on a
third set of problems?

This page records what the test should show, **before** the run.

## The run

GSM8K and GSM-Hard problems 600 to 1318 (`--problems 1319 --skip 600`), 719 of each, which no run has asked
before. The 13 models of [2026-11-05-gsm-ladder](../../runs/2026-11-05-gsm-ladder/README.md), from
qwen2.5:0.5b to qwen3:4b-instruct, under `none`, `check`, `solve`, `retry`, `program` and `combine`, at the
commit `03ecd4f` of `main`, which has #186, with the settings of the earlier runs (temperature 0, seed 0,
context 8192 tokens, at most 2048 tokens a reply). On Kaggle, in four kernels: each data set, with the models
split in two groups (qwen3:4b-instruct, gemma3:4b, phi4-mini, qwen2.5:3b, llama3.2:3b and smollm2:1.7b; and
the other 7).

## The tests

Pooled over the 13 models unless a test says otherwise:

| Test | Holds when |
|---|---|
| `retry` gets more right than `solve` | on GSM8K and on GSM-Hard, each |
| Asking again after a number loses nothing | of the problems `retry` asked again because the first reply gave a number, at least as many are right after asking again as the first number was right on, on each data set |
| The combining rule holds on a third set of problems | on GSM-Hard, `combine` gets more right than `none`, `check` and `solve` each; on GSM8K, `combine` is not worse than `none` at p < 0.05 |
| `retry`'s answers don't hurt the combining rule | with `retry`'s answers in place of `solve`'s, `combine` gets at least 60 fewer right on neither data set |

`program` against `solve` is reported both ways, per data set and per model, with no direction set beforehand:
no run so far has asked these models for programs.

## How it will be scored

`python paper/analysis.py` on the run's folder, and `python paper/combine_rules.py RUN --gsm8k test.jsonl
--gsmhard gsmhardv2.jsonl --ada RUN:retry` for the last test. The second test reads each `retry` record that
asked again after a "number given" first reply, and compares the first number (`first_ada`'s value) and the
final answer with the data set's answer.
