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
| [2026-11-05-gsm-ladder](runs/2026-11-05-gsm-ladder/README.md) | 13 models from 0.5B to 4B on GSM8K and GSM-Hard, 300 problems each: alone, after worked examples, checked, and writing equations for ADA to solve |

## Analyses

Work done on runs already made, such as choosing a rule before testing it on new problems, is in
[analyses/](analyses/):

| Analysis | What it is |
|---|---|
| [2026-11-05-combine-rules](analyses/2026-11-05-combine-rules/README.md) | How to combine the model's answer with ADA's, chosen on the ladder's problems 0 to 299 and set down before it was tested on problems 300 to 599 |

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
```

A new run is added with `python paper/collect.py <folder of results files> paper/runs/<run>` and a README
and config.json written like the others. The benchmarks themselves, and how each condition works, are in
[benchmarks/](../benchmarks/README.md).

## How the runs are made

- The models run through Ollama at temperature 0 with seed 0, one reply per problem and condition unless a
  run says otherwise, so a run repeats closely but not always word for word.
- Answers are graded by the benchmark: GSM8K exactly, GSM-Hard to within 0.001, MATH-500 with Math-Verify.
- A rule or setting chosen by looking at results is tested on problems it was not chosen on, and the
  choice is written down in [analyses/](analyses/) before the test runs.
- Where a run's grading was later found wrong, the run's page says what changed and its results file holds
  the corrected grades.
