# ADA: Automated Data Analyst

[![CI](https://github.com/saineshnakra/automated-data-analyst/actions/workflows/ci.yml/badge.svg)](https://github.com/saineshnakra/automated-data-analyst/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/automated-data-analyst?color=635bff)](https://pypi.org/project/automated-data-analyst/)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-20a779.svg)](https://github.com/saineshnakra/automated-data-analyst/blob/main/LICENSE)
[![Agent skill: Claude Code and Codex](https://img.shields.io/badge/agent%20skill-Claude%20Code%20%C2%B7%20Codex-0e0f13)](#use-it-from-claude-code-or-codex)

ADA computes numbers from data files and checks numbers stated about them.
Every number comes from pandas, SciPy or statsmodels, with the calculation and
the rows behind it. A number someone states about a table (in a report, or in
an AI assistant's answer) is verified against the data, or reported unverified
with the reason and the one question that would settle it. When something
can't be computed honestly, ADA says why instead of guessing.

It is a Python package and a command line, for people and for coding agents
such as Claude Code and Codex that need numbers they can back up.

```bash
pip install automated-data-analyst     # Python 3.11 or newer
```

## Check a number

```console
$ ada claim flights.csv "The average departure delay was about 13 minutes."
verified: The average departure delay was about 13 minutes.
  12.63907026 (328,521 rows) from mean(dep_delay) lies in 12 to 14, which about 13 minutes stands for.
  calc: mean(dep_delay)
  assumes: Read the claim's words as dep_delay, the only column the question fits.
  assumes: Left out 8,255 rows with dep_delay missing.
  assumes: Read "about" as 12 to 14.

$ ada claim flights.csv "The average delay was 13 minutes."
unverified: The average delay was 13 minutes.
  The claim's words fit more than one column, and they give different answers: dep_delay gives 12.63907026; arr_delay gives 6.895376757.
  to check it: Which column is meant: dep_delay or arr_delay?
```

(`flights.csv` is from [nycflights13](https://github.com/tidyverse/nycflights13).)

The claim's number stands for the interval its writing implies: "12.64" is
12.635 to 12.645, "about 13" is 12 to 14, "more than 30%" is above 30. A unit
in the column's name is converted (`body_mass_g` against "5.08 kg"); a scale
the data don't state is asked about, not assumed.

| Status | Reason | Meaning |
|---|---|---|
| `verified` | | The computed number lies in the claim's interval and no check on the analysis failed |
| `unverified` | `failed_check` | The data give a different number (`agreement: "mismatch"`), or a check on the analysis failed |
| `unverified` | `missing_context` | The claim leaves out something the check needs: a column, a group, a unit, a date column |
| `unverified` | `ambiguous_intent` | The words fit several calculations with different answers; each is listed |
| `unverified` | `unsupported_analysis` | ADA can't compute what the claim describes, or the claim holds several numbers |
| `unverified` | `conflicting_evidence` | The claim's unit measures something other than the column's |
| `unverified` | `inaccessible_data` | The data could not be read |

**Verified means verified as read**: this number, from these rows, by this
calculation, with the assumptions listed. It says nothing about the words
around the number, and a check of one number does not verify a whole answer.
**Unverified is not false**: only `failed_check` with a mismatch says the data
give a different number.

`--json` prints the claim, its interval, the plan that ran, the calculation,
the rows read, used and left out with a reason for each, the file's sha256, the
assumptions, the warnings, the checks run and the clarifying question. The
exit code is 0 verified, 1 failed check, 2 unreadable, 3 unverified for
another reason. In Python: `ada.check_claim("flights.csv", "...")`.
See [Checking claims](docs/reference/claims.md).

## Compute an answer

```bash
ada ask flights.csv "average dep_delay by origin"          # the answer and its calculation
ada ask flights.csv "average dep_delay by origin" --json   # the same as data
```

The JSON holds the full-precision numbers (`value`, `values` per group,
`figures` such as counts, shares and their denominators, p-values and
intervals), how the question was read, the calculation (`mean(dep_delay) by
origin`), and the evidence: the source file and its sha256, the rows read and
used (328,521), and each reason rows were left out (8,255 with `dep_delay`
missing). A question ADA can't answer comes back refused, with the reason.

## A folder of files

```console
$ ada scan nycflights13/
nycflights13/: 5 files, 6 proposed links
  flights.csv: 336,776 rows x 20 columns, one row per flight
  weather.csv: 26,115 rows x 16 columns, one row per origin, time_hour
  ...
  link flights.carrier -> airlines.carrier: many-to-one, 100.0% of keys found, 0 rows unmatched
  link flights.dest -> airports.faa: many-to-one, 96.2% of keys found, 7,602 rows unmatched (confirm)
  link flights.(origin, time_hour) -> weather.(origin, time_hour): many-to-one, 99.4% of keys found, 1,556 rows unmatched (confirm)
  ...
  12 questions to answer:
    [rows] What does one row of flights.csv stand for?
    [units] What unit is humid in weather.csv recorded in?
    [links] Join flights.dest to airports.faa (many-to-one)? the names differ; the link rests on the values alone; 4 flights.dest values (7,602 rows) have no match in airports.csv.
```

`ada scan` profiles every table, proposes the keys that join them from their
values (coverage both ways, cardinality, unmatched rows, how many rows a join
would repeat), reads a codebook's units, and groups what a person should
confirm. `--template answers.yml` writes the questions with ADA's proposed
answers; `--answers answers.yml` reads the confirmed ones back. ADA proposes
joins but does not run them. A claim about a folder is checked against the
file it fits. See [Reading a folder of tables](docs/reference/folder.md).

## What ADA checks before it answers

- **Rows left out**, each with its reason and count: missing values, values
  the data checks flag, exact repeats of other rows.
- **The unit of analysis.** When several rows come from one subject, a
  comparison or correlation counts each subject once ("does Reaction increase
  with Days" over 18 people measured on 10 days gives the average of their 18
  slopes, not a correlation over 180 rows). An "average order value" totals
  each order first.
- **Groups that reverse a pooled figure.** Bill length and depth fall together
  over all penguins (r = -0.24) and rise together within each species; the
  answer says so.
- **Thin data.** Too few rows or units for a test gives a reason, not a number.
- **The data themselves**: impossible values, missing-value codes, caps,
  copied rows and columns, before any figure is computed.

## Use it from Claude Code or Codex

```bash
pip install automated-data-analyst
ada skill               # .claude/skills/ada (Claude Code) and .agents/skills/ada (Codex)
ada skill --user        # the same in your home folder, for every project
ada skill --check       # after upgrading: is the installed skill current?
```

Or as a plugin: `/plugin marketplace add saineshnakra/automated-data-analyst`
then `/plugin install automated-data-analyst@automated-data-analyst` in Claude
Code; `codex plugin marketplace add saineshnakra/automated-data-analyst` then
`codex plugin add automated-data-analyst@automated-data-analyst` in Codex.

The [skill](ada/skills/ada/SKILL.md) tells the assistant to let ADA do the
arithmetic, check every number it states with `ada claim`, say which numbers
are verified (with their calculation) and which are not and why, ask you ADA's
clarifying question instead of choosing a reading, and run `ada scan` before
combining files. This does not make the assistant's other statements correct;
it means each number it reports has been checked, or is marked as not checked.

## Paper assets for a study

For research data with a design (groups, subjects, time points, planned
analyses), ADA builds what a paper needs from a `study.yml` the researcher
confirms: a data audit, a data dictionary, a record of every exclusion,
results tables in Word, LaTeX and CSV, figures with their source data and
captions, a methods draft, and a package another researcher can rebuild it
from. `ada verify` rebuilds everything in a scratch folder and checks every
number still matches. This is a separate check from `ada claim`: it checks
that results reproduce, not that a stated number is right.

```bash
ada demo mice && cd mice           # an example study to try first
ada init data/measurements.csv     # asks about the study; writes study.yml
ada build                          # writes paper-assets/
ada verify                         # rebuilds and compares every number
```

See [Paper assets for a study](docs/research.md), or the
[example study in the browser](https://automated-data-analyst.streamlit.app/example-study).

## A first look, and the web app

`ada analyze data.csv` (or `ada.analyze("data.csv")` in Python) reads a table
with no study behind it: the data checks, which groups differ, what relates
to what, changes over time, each finding with its test and arithmetic.
`ada check` watches a measure for unusual periods in scheduled jobs. The
Streamlit app (`pip install "automated-data-analyst[app]"`, then `ada app`)
does the same in a browser; no API key is needed and the data stay on your
machine ([Privacy](docs/privacy.md)).

## What it doesn't do yet

- **Joins.** `ada scan` proposes joins; a number that needs two files (a
  player's name in one, his home runs in another) is not computed.
- **Aggregate rows.** A World or Total row among countries is not set apart,
  so a total over every row can double count.
- **Many phrasings.** A claim is read with ADA's question parser: column names
  in other words ("life expectancy" for `lifeExp`), slopes ("rises by 10.5
  per day"), and claims with several numbers come back unverified.
- **Some files**: metadata lines above the header, frequency-weighted tables
  (one row per combination with a count column).

## Progress

Measured on 46 cases from 13 public datasets, each expected value computed
separately with pandas, SciPy and statsmodels; dev cases were used to find
problems, holdout cases only to score them
([details](results/verification/README.md)):

| | ADA 0.5.0 | Now |
|---|---|---|
| Dev cases | 4/26 | 20/26 |
| Holdout cases | 3/20 | 5/20 |
| Wrong verdicts on claims (verified when not supported, or failed when true) | | 0/20 |
| True claims verified | 0/14 | 3/14 |

Unseen data still defeats the reader more often than not; the misses come
back unverified with a reason rather than wrong. The first look was also
tested on [60 research datasets](results/README.md).

## Documentation

| Page | Contents |
|---|---|
| [Checking claims](docs/reference/claims.md) | How a claim is read, its interval, the reasons, the JSON record |
| [Reading a folder of tables](docs/reference/folder.md) | File profiles, proposed links, questions and answers |
| [Ask ADA](docs/reference/ask-ada.md) | How a question becomes a pandas calculation |
| [Paper assets for a study](docs/research.md) | `ada init`, `ada build`, `ada verify` and the study.yml format |
| [All documentation](docs/README.md) | Concepts, architecture, one reference page per step |
| [Credits](docs/credits.md) | The libraries behind each calculation, and how to cite them |

## Contributing

```bash
pip install -e ".[app,ai,dev]"
ruff check .
python -m unittest discover -s tests
```

A new rule needs a test, including how it behaves on thin data. See
[CONTRIBUTING](CONTRIBUTING.md) and the [Development](docs/development.md)
page; issues labelled good first issue are a place to start.

## Citing

See [CITATION.cff](CITATION.cff), and [Credits](docs/credits.md) for the
methods and libraries to cite alongside ADA.

## License

MIT. See [LICENSE](LICENSE).
