# Paper: what helps small models at math, and where ADA helps

This folder keeps every benchmark run made for the paper, so each number in it can be traced to the
replies it came from and worked out again.

**The question.** A small language model (0.5 to 4 billion parameters) gets many grade-school and
competition math problems wrong. Which of its mistakes does an exact solver fix, and how should the model
and the solver share the work? ADA is the solver: it checks the arithmetic in a model's answer, or solves
the equations a model writes, exactly, with SymPy underneath.

## Runs

Each run has a folder in [runs/](runs/), named by the date it started:

| Run | What it is |
|---|---|
| [2026-11-01-gsm-three-models-first-checker](runs/2026-11-01-gsm-three-models-first-checker/README.md) | GSM8K and GSM-Hard, three models, with the first version of ADA's checker, which flagged right steps as wrong |
| [2026-11-01-gsm-three-models-tool](runs/2026-11-01-gsm-three-models-tool/README.md) | GSM8K and GSM-Hard, three models (1B to 1.7B), 200 problems: alone, with `ada_compute` offered, and checked |
| [2026-11-03-math500-tool](runs/2026-11-03-math500-tool/README.md) | MATH-500 Algebra and Intermediate Algebra, two models with and without `ada_compute` |
| [2026-11-03-math500-five-conditions](runs/2026-11-03-math500-five-conditions/README.md) | MATH-500, the tool's system prompt and its definition tried apart, and the check |
| [2026-11-03-math500-qwen3-thinking](runs/2026-11-03-math500-qwen3-thinking/README.md) | MATH-500, all 500 problems, qwen3:1.7b with thinking on, alone and checked |
| [2026-11-04-math500-qwen3-votes](runs/2026-11-04-math500-qwen3-votes/README.md) | MATH-500, qwen3:1.7b with thinking, votes of several samples on the problems the first reply left open |
| [2026-11-04-math500-qwen3-longer-replies](runs/2026-11-04-math500-qwen3-longer-replies/README.md) | MATH-500, qwen3:1.7b, first replies with no answer asked again with 30,720 tokens |
| [2026-11-04-math500-qwen3-longer-replies-rest](runs/2026-11-04-math500-qwen3-longer-replies-rest/README.md) | MATH-500, qwen3:1.7b, the rest of those first replies asked again |
| [2026-11-05-gsm-solve-pilot](runs/2026-11-05-gsm-solve-pilot/README.md) | GSM8K and GSM-Hard, 30 problems: the first try of `solve`, before worked examples |
| [2026-11-05-gsm-kaggle-smoke](runs/2026-11-05-gsm-kaggle-smoke/README.md) | GSM8K on a Kaggle GPU, 20 problems: a test of the Kaggle runner |
| [2026-11-05-gsm-ladder-pilot](runs/2026-11-05-gsm-ladder-pilot/README.md) | GSM8K and GSM-Hard, 15 models from 135M to 4B, 30 problems: alone, checked and `solve` |
| [2026-11-05-gsm-ladder](runs/2026-11-05-gsm-ladder/README.md) | GSM8K and GSM-Hard, 13 models from 0.5B to 4B, 300 problems: alone, after worked examples, checked and `solve` |
| [2026-11-05-gsm-combine](runs/2026-11-05-gsm-combine/README.md) | GSM8K and GSM-Hard problems 300 to 599, 13 models: the combining rule chosen on problems 0 to 299, and the model deciding instead |
| [2026-11-05-gsm-plan](runs/2026-11-05-gsm-plan/README.md) | GSM8K and GSM-Hard, 7 models from 1.5B to 4B, 300 problems: `solve`, working the problem out in words first (`plan`), and writing the equations again when ADA finds no value (`retry`) |

A run that picks up an earlier run's replies keeps them in its own folder, so it reproduces its own
numbers; its page says how many records are its own. Counting several folders together counts those
records more than once.

## Analyses

Work done on runs already made, such as choosing a rule before testing it on new problems, is in
[analyses/](analyses/):

| Analysis | What it is |
|---|---|
| [2026-11-05-combine-rules](analyses/2026-11-05-combine-rules/README.md) | How to combine the model's answer with ADA's, chosen on the ladder's problems 0 to 299 and set down before it was tested on problems 300 to 599 |
| [2026-11-05-plan-and-retry](analyses/2026-11-05-plan-and-retry/README.md) | Which way of writing better equations to keep, chosen on problems 0 to 299, and the tests for problems 300 to 599 |

## What a run folder holds

| File | What it is |
|---|---|
| `README.md` | The question, what ran (data and its sha256, models, conditions, settings, where, commit), the results, what they show, what to keep in mind, and the runs before and after it |
| `config.json` | The same settings, to be read by a program |
| `results.jsonl.gz` | Every record the benchmark wrote: one line per problem, condition, model and sample, with the model's whole reply, its equations and what ADA read from them, and whether the answer is right. Sorted, and the same byte for byte each time it is made |

## Working the numbers out again

```bash
python paper/analysis.py paper/runs/2026-11-05-gsm-ladder        # accuracy, McNemar tests, ADA's statuses
python paper/combine_rules.py paper/runs/2026-11-05-gsm-ladder \
    --gsm8k test.jsonl --gsmhard gsmhardv2.jsonl                 # the ways of combining the two answers
python paper/combine_rules.py paper/runs/2026-11-05-gsm-ladder \
    --gsm8k test.jsonl --gsmhard gsmhardv2.jsonl \
    --ada paper/runs/2026-11-05-gsm-plan:retry                   # the same, with retry's answers as ADA's
```

A new run is added with `python paper/collect.py <folder of results files> paper/runs/<run>` and a README
and config.json written like the others. The benchmarks themselves, and how each condition works, are in
[benchmarks/](../benchmarks/README.md).

## How the runs are made

- The models run through Ollama at temperature 0 with seed 0, one reply per problem and condition unless a
  run says otherwise, so a run repeats closely but not always word for word. On Kaggle, where several
  shards ask one server at once, two replies to the same message differ on about a third of GSM8K problems
  ([2026-11-05-gsm-plan](runs/2026-11-05-gsm-plan/README.md)); those differences go about as often one way
  as the other.
- Answers are graded by the benchmark: GSM8K exactly, GSM-Hard to within 0.001, MATH-500 with Math-Verify.
- A rule or setting chosen by looking at results is tested on problems it was not chosen on, and the
  choice is written down in [analyses/](analyses/) before the test runs.
- Where a run's grading was later found wrong, the run's page says what changed and its results file holds
  the corrected grades.
