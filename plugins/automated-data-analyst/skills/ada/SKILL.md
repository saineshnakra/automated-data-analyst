---
name: ada
description: Compute and check numbers from CSV or Excel tables with ADA (the automated-data-analyst Python package). It computes every number in pandas, SciPy and statsmodels with the calculation and rows behind it, checks a number stated about a table (verified, or unverified with a reason), maps a folder of related files and how they join, or refuses with a reason instead of guessing. Use when someone wants a number from a data file, wants a number in a report or an answer checked against the data, wants to know what is in a file or folder, compare groups, test a relationship or a trend, or make a paper's tables, figures and methods draft from research data with a rebuild that can be verified.
license: MIT
compatibility: Needs Python 3.11 or newer and `pip install automated-data-analyst` 0.6.0 or later (0.5.0 for the study workflow only).
metadata:
  homepage: https://github.com/saineshnakra/automated-data-analyst
  docs: https://github.com/saineshnakra/automated-data-analyst/blob/main/docs/research.md
---

# ADA: data analysis with every number computed

ADA turns a table of measurements and a plain `study.yml` into what a paper
needs: a data audit, a data dictionary, a record of every exclusion,
descriptive and results tables (CSV, Word, LaTeX), figures with their source
data and captions, a methods draft, and a package another researcher can
rebuild it all from. The numbers come from pandas, SciPy and statsmodels.

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
- **A paper or report from a study** (groups, subjects, time points, planned
  analyses): the study workflow below.
- **Just what is in a file**: `ada analyze data.csv`, or in Python
  `ada.analyze("data.csv")`. See "First look" at the end.

```bash
pip install automated-data-analyst
ada --version
```

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

## The study workflow

Run every command in the folder that holds the study's data.

### 1. Ask what only the researcher knows

Look at the file first (columns, a few rows, counts), then ask about anything
you can't tell from the data. Do not guess these and treat the guess as fact:

- What one row is (one mouse at one week, one well, one patient visit).
- Which column identifies a participant, animal, specimen, site or plate, and
  whether each one has several rows. Check this yourself: if a subject-like
  column has far fewer distinct values than there are rows, the rows are not
  independent and the design is repeated or clustered.
- The outcomes, the groups or conditions, the time variable and covariates.
- Which analyses were planned before anyone saw the data. Everything else is
  exploratory.
- What may be shared, and which columns must never leave (names, dates of
  birth, operator).

### 2. Write study.yml

```bash
ada init data/measurements.csv          # asks each question in a terminal
ada init data/measurements.csv --yes    # accepts ADA's guesses without asking
```

`ada init` reads the file and writes `study.yml` with its guesses. In an
agent session without a person at the terminal, use `--yes`, then edit
`study.yml` with the researcher's answers. `--yes` marks every analysis
exploratory, because only the researcher can say what was planned, and writes
`design.confirmed: false`, which the methods draft reports as an unconfirmed
design. Read the whole file before building: check the design, the subject or
cluster column, each variable's role and type, and the proposed analyses.
Set `confirmed: true` only after the researcher has confirmed the design.

The parts that matter most:

```yaml
data:
  path: data/assay.csv
  sha256: 3f2a…               # ADA stops if the file changes
  observation: one well
  missing_codes: [-99]

design:
  type: repeated              # independent | paired | repeated | clustered
  confirmed: true             # only once the researcher has confirmed it
  subject: Mouse              # clustered designs name a `cluster` instead
  time: Week

variables:                    # every column the design, processing or analyses name
  Mouse:
    role: identifier
    type: text
  Week:
    role: time
    type: numeric
    unit: weeks
  Weight (g):
    role: outcome             # outcome | group | time | covariate | identifier | other
    type: numeric             # numeric | categorical | binary | date | text
    unit: g
    range: [10, 60]           # checked in the audit, never enforced
  Diet:
    role: group
    type: categorical         # a group must be categorical or binary, even if coded 1/2/3
    levels: [Chow, High fat]  # the first is the reference

processing:                   # run in order; each step needs a reason
  - exclude:
      ids: [M07]
    reason: M07 was given the wrong diet in weeks 2 and 3 (feeding log, entry 12).
  - exclude:
      where: {column: Weight (g), below: 0}
    reason: Negative weights are scale errors.

decisions:
  multiple_comparisons: holm  # holm | bonferroni | fdr_bh | none
  alpha: 0.05
  confidence: 0.95

analyses:
  - id: weight-gain
    question: Do mice on the high-fat diet gain weight faster?
    outcome: Weight (g)
    change: Week              # or compare: a group, or relate: a numeric predictor
    by: Diet                  # with change: the time × group interaction
    covariates: []
    planned: true             # only when the researcher says it was planned
    family: primary           # p-values are adjusted within a family

sharing:
  data: processed             # processed | raw | demo | none
  withhold: [Operator]
```

Exclusion conditions: `below`, `above`, `at_most`, `at_least`, `equals`,
`not_equals`, `in`, `not_in`, `missing`. Other steps: `transform` (`log`,
`log10`, `log2`, `sqrt`, `zscore`, `center`, `scale`, …) into a new column, and
`recode` with a `map`.

### 3. Build

```bash
ada build              # or ada build --strict: exit 1 if any analysis was refused
```

This writes `paper-assets/` and prints how many analyses were requested,
fitted and refused, with the reason for each refusal. Then:

- Read the printed summary and `paper-assets/README.md`, which lists every
  table and figure with its caption.
- Read `paper-assets/audit/data-audit.md` and tell the researcher about each
  item to check.
- Look for analyses that were refused or fitted with warnings (the methods
  draft names them, and `supplement/models/` has each model's output). Report
  a refusal and its reason as it is.

### 4. Verify

```bash
ada verify
```

It rebuilds from `study.yml` and the data alone in a scratch folder and
compares every file, number by number. Exit code 0 means everything matches.
Run it before sharing results and in CI.

## How ADA picks the test

The design decides it; ADA does not search for a significant result.

| Design | `compare:` a group | `relate:` a number | `change:` time |
|---|---|---|---|
| independent | Welch's t-test, or Welch's ANOVA with Games–Howell contrasts; regression with covariates | regression slope with Pearson's r and Spearman's ρ | not fitted |
| paired | paired t-test | mixed model | mixed model |
| repeated | linear mixed model, random intercept per subject | mixed model | mixed model over time |
| clustered | linear mixed model, random intercept per cluster | mixed model | not fitted |

Binary outcomes get the risk difference, odds ratio and χ² or Fisher's exact
test, or logistic regression. `test:` asks for another test the design allows
(`mann-whitney`, `kruskal`, `wilcoxon`, `student-t`, `anova`). In a change
analysis, `time_as: categorical` or `time_as: linear` sets how time enters.
Crossover trials, nested random effects, survival, ordinal outcomes and
binary outcomes measured repeatedly are refused, not approximated.

## Rules to keep

- Never choose the design, the planned analyses or an exclusion from the
  results. An exclusion's reason must not depend on the outcome.
- Don't add analyses until one is significant. Anything added after seeing
  the data stays `planned: false`.
- Don't edit files in `paper-assets/` or the data file by hand. Change
  `study.yml` and rebuild; `ada verify` flags edited files.
- If the data file changes on purpose, put its new sha256 in `data.sha256`
  (the error message prints it). Never remove the check to get past it.
- If ADA refuses a design, say so and offer a `custom:` script for the
  researcher's own analysis instead of switching to a design that fits.
- Quote numbers from `paper-assets/tables/` (CSV files hold full precision)
  with the test named in the table, and say whether the analysis was planned.

## Your own tables and figures

List a Python file under `custom:` in study.yml. Functions take the processed
data and return a pandas table or a Matplotlib figure; ADA saves them in the
study's formats and records them in `provenance.json`.

```python
import matplotlib.pyplot as plt
from ada import research

@research.table("adjusted-model", title="Viability adjusted for passage")
def adjusted(data, study):
    return data.groupby("Treatment")["Viability (%)"].describe().reset_index()

@research.figure("dose-curve", title="Dose response", caption="Points are wells.")
def dose_curve(data, study):
    fig, ax = plt.subplots(figsize=research.figure_size(study))
    ax.scatter(data["Dose"], data["Viability (%)"])
    return fig, data[["Dose", "Viability (%)"]]
```

## Restricted data

```bash
ada preview    # every file, whether it may be shared, and why
ada share      # copies only what may be shared into share/
```

`sharing.data: demo` writes synthetic data with the same columns so others
can run the reproduction without the real data.

## What the build writes

| Folder | Contents |
|---|---|
| `audit/` | values outside their range, duplicates, whether the data fit the design |
| `dictionary/` | each column's meaning, unit, type, allowed and observed values |
| `processing/` | every step from the original file, rows before and after, the excluded rows |
| `tables/` | descriptive and results tables as CSV, Word and LaTeX |
| `figures/` | PDF, SVG, PNG or TIFF at a journal's size, `<id>.source.csv`, captions |
| `methods/` | methods draft and results draft, written from what was run |
| `supplement/` | exploratory analyses, model output, checks |
| `reproduce/` | study.yml, pinned versions, input checksums, instructions |
| `repository/` | README, CITATION.cff, .zenodo.json, availability statements |
| `provenance.json`, `CHANGES.md` | where each number came from; what changed at each build |

`output.journal` (`plos`, `nature`, `cell`, `science`, `elife`) sets figure
sizes, resolution and fonts.

## First look at any table

For a file with no study behind it, or to see what is in one before writing
`study.yml`:

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
around it. The first look guesses what matters from the file alone, so for a
paper use the study workflow, where the researcher states the design.
