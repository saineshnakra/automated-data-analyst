---
name: ada
description: Compute and check numbers from CSV or Excel tables with ADA (the automated-data-analyst Python package). It computes every number in pandas, SciPy and statsmodels with the calculation and rows behind it, checks a number stated about a table (verified, or unverified with a reason), maps a folder of related files and how they join, or refuses with a reason instead of guessing. Use when someone wants a number from a data file, wants a number in a report or an answer checked against the data, wants to know what is in a file or folder, compare groups, test a relationship or a trend,.
license: MIT
compatibility: Needs Python 3.11 or newer and `pip install automated-data-analyst` 0.7.0 or later.
metadata:
  homepage: https://github.com/saineshnakra/automated-data-analyst
  docs: https://github.com/saineshnakra/automated-data-analyst/blob/main/docs/README.md
---

# ADA: data analysis with every number computed

ADA computes numbers from CSV and Excel tables and checks numbers stated
about them. Every number comes from pandas, SciPy or statsmodels, with the
calculation and the rows behind it; when ADA can't compute something honestly
it refuses with the reason.

Let ADA do the arithmetic. Report the numbers it computed, with the test or
calculation it names, rather than computing or estimating them yourself; when
it refuses, pass the reason on. That way everything you report can be checked
against ADA's output and the data.

Pick the mode from what the person wants:

- **A number, or a check of one** ("what is the average delay?", "is it true
  that 32% survived?", or any number you are about to write about a data
  file): "Computing and checking numbers" below.
- **A folder of related files**: `ada scan data/` first and ask its
  questions, then `ada ask` and `ada claim` on the folder or a file.
- **Just what is in a file**: `ada analyze data.csv`, or in Python
  `ada.analyze("data.csv")`. See "First look" at the end.

```bash
pip install automated-data-analyst
ada --version
```

A paper's tables, figures and methods draft from a study, or a prediction for
a Kaggle-style folder, are in ADA Lab (`pip install ada-lab`; see
https://github.com/saineshnakra/ada-lab).

## Arithmetic on numbers you already have

Do not add, multiply, divide or take percentages in your head. Write the
expression and let ADA compute it exactly:

```bash
ada compute "0.15 * (1200 - 300)"      # 135, with each step
ada compute "1/3 + 1/6" --json         # value 0.5, fraction 1/2, exact: true
ada compute "x + y = 10; x - y = 2"    # x = 6, y = 4, with each step and a check
```

Fractions stay exact, each step is shown, and the conventions used are
listed (-2^2 is -4). An expression that reads two ways (`6/2(1+2)`, `17 % 5`,
`log(8)`) is refused with both readings: write the parentheses or the
function you mean. Report the value with its steps, and say "approximately"
when `exact` is false.

To solve equations, write them with one-letter unknowns and give them to the
same command: it solves linear equations and systems exactly, rules out values
that divide by zero, and ends with a check that puts the answer back. "No
solution" and "every value works" are answers, not errors.

## Computing and checking numbers

Compute with `ada ask`, and check every number you state about the data with
`ada claim`, including numbers you computed yourself or read in a report.

```bash
ada ask flights.csv "average dep_delay by origin" --json
ada claim flights.csv "The average departure delay was about 13 minutes."
ada claim flights.csv --question "average dep_delay" --value 12.64 --json
ada claim data/ "The average order_total is 154.10."     # a folder: checked against the file it fits
```

`ada claim` prints `verified` or `unverified` with one of six reasons, and
exits 0 (verified), 1 (the data give a different number, or a check on the
analysis failed), 2 (the file could not be read) or 3 (unverified for another
reason). With `--json` it gives the claim's interval ("about 13" is 12 to 14),
the plan that ran, the calculation, the rows read, used and left out with a
reason for each, the file's sha256, the assumptions, the warnings, and a
`clarification`: the one question that would let the check go ahead.

When you report numbers:

- Say which numbers ADA verified, and give each with its calculation and the
  rows it used ("12.64, mean(dep_delay) over the 328,521 flights with a
  departure delay recorded").
- Say which are unverified and why, in ADA's words. Unverified is not false:
  only `failed_check` with `agreement: "mismatch"` means the data give a
  different number.
- Never call a whole answer verified because one number in it was. A check
  covers one number, as read; it says nothing about the wording around it.
- When the reason is `missing_context` or `ambiguous_intent`, ask the person
  ADA's `clarification` (for example "Which column is meant: dep_delay or
  arr_delay?") instead of picking a reading yourself.
- Pass on the assumptions and warnings that matter: rows left out, a scale
  or unit that was not given, a correlation that runs the other way within
  groups, repeated rows from one subject.
- Read `decisions` on every answer and check: the choices ADA took, such as
  leaving World and region rows out of an average. Each has a `question`,
  its `options` and ADA's `default`. When the person's goal settles one
  ("compare countries" leaves World out), pass that option with
  `--decide KEY=OPTION`, such as `--decide aggregate_rows=keep`, and say you
  did. When it does not, ask the person the decision's `question`. With no
  one to ask, keep the default and say which it was. (`--decide` needs a
  current ADA; upgrade if it is not recognised.)

Start with `ada scan` on the file or the folder (`ada scan data.csv`, `ada
scan data/`). It lists each file, the keys that join them (with the rows a
join leaves unmatched or repeats) and the questions to ask before an
analysis: what the person wants to find out, what a row stands for, units,
repeated measurements, which links to trust, and the decisions ADA would
otherwise take by default (leaving World and region rows out, which links to
join on). Ask the person those questions; propose answers from what they
already told you. `ada scan PATH --template answers.yml` writes the questions
with ADA's proposed answers: put in what the person said and pass `--answers
answers.yml` to every `ada ask` and `ada claim`. With no one to ask, keep
ADA's defaults and say that you did.

A question or claim about a folder (`ada ask data/ "..."`, `ada claim data/
"..."`) is answered from the file it fits, or from two files joined along a
link ADA is sure of, and the answer says which join it used. Confirm a link
in the answers file, or pass `--decide join_links=all`, only when the person
agrees the link is right.

## First look at any table

To see what is in a file:

```bash
ada analyze data.csv           # findings, each with the test and arithmetic
ada analyze data.csv --json    # machine-readable
ada schema data.csv            # detected columns and what was cleaned
ada ask data.csv "average pH by site"
```

```python
import ada

report = ada.analyze("data.csv")        # a path, a DataFrame or raw bytes
print(report)
report.quality()                        # data-check issues with their rows
report.evidence                         # findings, each with .statement and .calculation
report.to_dict()                        # JSON-safe
```

When ADA can't answer it returns a `Refusal` (falsy, with `.code` and
`.message`) instead of a number. Pass the reason on rather than working
around it. The first look guesses what matters from the file alone: ask the
person what a row is and what they want to find out before you lean on it.
