# ADA: Automated Data Analyst

[![CI](https://github.com/saineshnakra/automated-data-analyst/actions/workflows/ci.yml/badge.svg)](https://github.com/saineshnakra/automated-data-analyst/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/automated-data-analyst?color=635bff)](https://pypi.org/project/automated-data-analyst/)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-20a779.svg)](https://github.com/saineshnakra/automated-data-analyst/blob/main/LICENSE)
[![Stars](https://img.shields.io/github/stars/saineshnakra/automated-data-analyst?style=flat&logo=github&color=f5b731)](https://github.com/saineshnakra/automated-data-analyst/stargazers)
[![Agent skill: Claude Code and Codex](https://img.shields.io/badge/agent%20skill-Claude%20Code%20%C2%B7%20Codex-0e0f13)](#use-it-from-claude-code-or-codex)

<p align="center">
  <a href="https://github.com/saineshnakra/automated-data-analyst"><b>⭐ Star ADA</b></a> if it saves you an afternoon ·
  <a href="https://automated-data-analyst.streamlit.app/example-study">Try it in the browser</a> ·
  <a href="https://github.com/saineshnakra/automated-data-analyst/blob/main/docs/README.md">Docs</a> ·
  <a href="https://github.com/saineshnakra/automated-data-analyst/blob/main/CHANGELOG.md">What's new in 0.6.0</a>
  <br>
  <sub>Read this in <a href="https://readme-i18n.com/saineshnakra/automated-data-analyst?lang=zh">中文</a> · <a href="https://readme-i18n.com/saineshnakra/automated-data-analyst?lang=ja">日本語</a> · <a href="https://readme-i18n.com/saineshnakra/automated-data-analyst?lang=ko">한국어</a> · <a href="https://readme-i18n.com/saineshnakra/automated-data-analyst?lang=es">Español</a> · <a href="https://readme-i18n.com/saineshnakra/automated-data-analyst?lang=pt">Português</a> · <a href="https://readme-i18n.com/saineshnakra/automated-data-analyst?lang=de">Deutsch</a> · <a href="https://readme-i18n.com/saineshnakra/automated-data-analyst?lang=fr">Français</a> · <a href="https://readme-i18n.com/saineshnakra/automated-data-analyst?lang=ru">Русский</a></sub>
</p>

ADA computes numbers from data files and checks numbers stated about them.
Every number comes from pandas, SciPy or statsmodels, with the calculation and
the rows behind it. A stated number is verified against the data, or reported
unverified with the reason and the one question that would settle it. It is a
Python package and a command line, for people and for coding agents such as
Claude Code and Codex.

```bash
pip install automated-data-analyst     # Python 3.11 or newer
```

## Check a number

```console
$ ada claim flights.csv "The average departure delay was about 13 minutes."
verified: The average departure delay was about 13 minutes.
  12.63907026 (328,521 rows) from mean(dep_delay) lies in 12 to 14, which about 13 minutes stands for.
  calc: mean(dep_delay)
  assumes: Read "departure delay" as dep_delay.
  assumes: Left out 8,255 rows with dep_delay missing.
  assumes: Read "about" as 12 to 14.
  assumes: The unit "minutes" comes from the claim; the check compares numbers only.

$ ada claim flights.csv "The average delay was 13 minutes."
unverified: The average delay was 13 minutes.
  The claim's words fit more than one column, and they give different answers: dep_delay gives 12.63907026; arr_delay gives 6.895376757.
  to check it: Which column is meant: dep_delay or arr_delay?
```

`verified` means this number, from these rows, by this calculation, with the
assumptions listed. `unverified` is not false: only the reason `failed_check`
says the data give a different number. `--json` gives the whole record, and
`ada.check_claim(...)` does the same in Python. See
[Checking claims](docs/reference/claims.md) for the reasons, intervals and units.

## Compute an answer, or map a folder

```bash
ada ask flights.csv "average dep_delay by origin" --json   # the numbers, the calculation, the rows used
ada scan nycflights13/                                     # the files, the keys that join them, what to confirm
```

A question ADA can't answer comes back refused, with the reason. `ada scan`
proposes joins from the values and lists what a person should confirm; it
does not run them. See [Ask ADA](docs/reference/ask-ada.md) and
[Reading a folder of tables](docs/reference/folder.md).

Before it answers, ADA reports the rows it left out and why, counts each
subject once when several rows come from one, says when groups reverse a
pooled figure, adds up the count column of a table of counts, and gives a
reason instead of a number when the data are too thin.

## Use it from Claude Code or Codex

```bash
ada skill               # .claude/skills/ada (Claude Code) and .agents/skills/ada (Codex)
ada skill --user        # the same in your home folder, for every project
```

It also installs as a plugin ([how](plugins/automated-data-analyst/README.md)).
The [skill](ada/skills/ada/SKILL.md) has the assistant let ADA do the
arithmetic, check every number it states with `ada claim`, and ask you ADA's
question instead of choosing a reading.

## Paper assets for a study

```bash
ada demo mice && cd mice           # an example study to try first
ada init data/measurements.csv     # asks about the study; writes study.yml
ada build                          # writes paper-assets/
ada verify                         # rebuilds and compares every number
```

`ada build` makes a data audit, a data dictionary, a record of every
exclusion, results tables in Word, LaTeX and CSV, figures with captions and a
methods draft. See [Paper assets for a study](docs/research.md). `ada analyze`
gives a first look at any table, and `ada app` opens the same in a browser
(`pip install "automated-data-analyst[app]"`).

## What it doesn't do yet

Numbers that need two files joined, World or Total rows among countries, many
phrasings ("horsepower" for `hp`, slopes, several numbers in one claim), and
year columns read as dates. These come back unverified or refused with a
reason. See [Not covered yet](docs/reference/claims.md#not-covered-yet).

## Progress

73 cases on 20 public datasets, each expected value computed separately with
pandas, SciPy and statsmodels; holdout cases are only scored
([details](results/verification/README.md)).

| | ADA 0.5.0 | Now |
|---|---|---|
| Dev cases | 6/51 | 36/51 |
| Holdout cases | 3/22 | 12/22 |
| Wrong verdicts on claims | | 0/35 |
| True claims verified | 0/26 | 13/26 |
| False claims caught | 0/7 | 6/7 |

## More

[All documentation](docs/README.md) · [Contributing](CONTRIBUTING.md) ·
[Credits](docs/credits.md) · [Citing](CITATION.cff) · MIT [License](LICENSE)
