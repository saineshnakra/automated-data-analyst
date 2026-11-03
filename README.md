# ADA: the math arm for LLMs

<!-- mcp-name: io.github.saineshnakra/automated-data-analyst -->

[![CI](https://github.com/saineshnakra/automated-data-analyst/actions/workflows/ci.yml/badge.svg)](https://github.com/saineshnakra/automated-data-analyst/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/automated-data-analyst?color=635bff)](https://pypi.org/project/automated-data-analyst/)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-20a779.svg)](https://github.com/saineshnakra/automated-data-analyst/blob/main/LICENSE)
[![Stars](https://img.shields.io/github/stars/saineshnakra/automated-data-analyst?style=flat&logo=github&color=f5b731)](https://github.com/saineshnakra/automated-data-analyst/stargazers)
[![Agent skill: Claude Code and Codex](https://img.shields.io/badge/agent%20skill-Claude%20Code%20%C2%B7%20Codex-0e0f13)](#use-it-from-claude-code-or-codex)
[![Tool calling: Ollama, vLLM, llama.cpp, LM Studio](https://img.shields.io/badge/tool%20calling-Ollama%20%C2%B7%20vLLM%20%C2%B7%20llama.cpp%20%C2%B7%20LM%20Studio-0e0f13)](docs/llm-tools.md)
[![MCP server](https://img.shields.io/badge/MCP-server-0e0f13)](docs/llm-tools.md#apps-that-take-mcp-servers)

<p align="center">
  <a href="https://github.com/saineshnakra/automated-data-analyst"><b>⭐ Star ADA</b></a> if it saves you an afternoon ·
  <a href="https://automated-data-analyst.streamlit.app/">Try it in the browser</a> ·
  <a href="https://github.com/saineshnakra/automated-data-analyst/blob/main/docs/README.md">Docs</a> ·
  <a href="https://github.com/saineshnakra/automated-data-analyst/blob/main/CHANGELOG.md">What's new in 0.8.0</a>
  <br>
  <sub>Read this in <a href="https://readme-i18n.com/saineshnakra/automated-data-analyst?lang=zh">中文</a> · <a href="https://readme-i18n.com/saineshnakra/automated-data-analyst?lang=ja">日本語</a> · <a href="https://readme-i18n.com/saineshnakra/automated-data-analyst?lang=ko">한국어</a> · <a href="https://readme-i18n.com/saineshnakra/automated-data-analyst?lang=es">Español</a> · <a href="https://readme-i18n.com/saineshnakra/automated-data-analyst?lang=pt">Português</a> · <a href="https://readme-i18n.com/saineshnakra/automated-data-analyst?lang=de">Deutsch</a> · <a href="https://readme-i18n.com/saineshnakra/automated-data-analyst?lang=fr">Français</a> · <a href="https://readme-i18n.com/saineshnakra/automated-data-analyst?lang=ru">Русский</a></sub>
</p>

ADA (Automated Data Analyst) is the part of a language model that does the
math. Point it at CSV or Excel data and ask in plain words: it computes the
answer with pandas, SciPy and statsmodels and returns the number with its
calculation and the rows behind it, or refuses with the reason. It checks
numbers stated about the data too: verified, or unverified with the reason
and the one question that would settle it.

A model never has to do arithmetic in its head again. That helps Claude and
Codex, and it helps open-source models most: a small local model run with
Ollama, llama.cpp, vLLM or LM Studio only has to pick the tool and read the
answer. ADA is a Python package, a command line and a set of tool
definitions any model with function calling can use.

```bash
pip install automated-data-analyst     # Python 3.11 or newer
```

## Give a language model a math arm

**Chat with a local model.** With [Ollama](https://ollama.com) running:

```bash
ollama pull qwen3:1.7b       # any model with tool calling works
ada chat penguins.csv        # ask in plain words; the model asks ADA for every number
```

```console
$ ada chat penguins.csv
ADA chat: qwen3:1.7b at http://localhost:11434/v1, reading penguins.csv. Ask about the data; Ctrl-D or 'exit' leaves.
> Which species has the longest bills on average?
  ADA ada_ask(path="penguins.csv", question="average bill_length_mm by species")
    -> answered: Chinstrap has the highest average bill_length_mm of any species, at 48.83 (n = 68). The species groups do differ (Welch ANOVA, p < 0.001; η² = 0.71). Based on the 342 rows with bill_length_mm recorded, of 344 rows. Each: Chinstrap 48.83; Gentoo 47.50; Adelie 38.79.
The species with the longest average bill length is **Chinstrap**, with an average of **48.83 mm** (n = 68). This result is based on the mean calculation of `bill_length_mm` grouped by species, sorted in descending order. The data spans 342 rows, and the values are as follows:
- **Chinstrap**: 48.83 mm
- **Gentoo**: 47.50 mm
- **Adelie**: 38.79 mm

The p-value indicates significant differences between species (p < 0.001), confirming the observed trends.
```

The model, 1.4 GB on four processor cores with no GPU, picked the tool; ADA
computed every number and showed how. Each tool call and its result are
printed above the answer.

**Add ADA to an app that takes MCP servers**: LM Studio, Jan, Claude Desktop,
Cursor, VS Code, Cline and others. Put this in its `mcp.json`:

```json
{"mcpServers": {"ada": {"command": "uvx", "args": ["automated-data-analyst", "mcp"]}}}
```

For Open WebUI, run `ada mcp --http` and add `http://127.0.0.1:8000/mcp` as
an MCP (Streamable HTTP) server under External Tools.

**In your own code**, `ada.chat.chat(question, "penguins.csv", model=...)`
runs the whole loop, or give the model `tools.definitions()` and run its
calls with `tools.run`:

```python
from ada import tools

tools.definitions()                    # four tools for function calling: ask, check a claim, scan, compute
tools.run("ada_check_claim", {"path": "flights.csv", "claim": "The average departure delay was about 13 minutes."})
```

[ADA as the math tool of a language model](docs/llm-tools.md) has the setup
for each app, which local models work, a loop with the Anthropic API and
what comes back.

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

## Do the arithmetic exactly

```bash
ada compute "0.15 * (1200 - 300)"      # 135, with each step
ada compute "1/3 + 1/6"                # 0.5 = 1/2: fractions stay exact
ada compute "6/2(1+2)"                 # refused: it reads as 9 or as 1
ada compute "x + y = 10; x - y = 2"    # x = 6, y = 4, each step, and a check
```

`ada compute`, `ada.compute()` and the `ada_compute` tool work out an
expression with exact fractions and integers, write each step with the
values it combined, and state each convention they rely on (-2^2 is -4).
An expression that reads two ways is refused with both values, and so is
anything that isn't a real number. A model writes the expression and reads
the answer instead of carrying digits itself. See
[Exact arithmetic](docs/reference/compute.md).

Given an equation, the same command and tool solve it: linear equations and
systems in one-letter unknowns, with fractions kept exact, values that would
divide by zero ruled out, and the answer put back into each equation as the
last step. `ada.solve("d = r*t", unknowns="t")` rearranges a formula. See
[Algebra](docs/reference/algebra.md).

`ada.check_arithmetic(text)` checks each equation in a worked answer, so a
model can reason in its own words and be told each slip with the right
value. With three small local models on GSM8K and GSM-Hard, that turned 32
wrong answers right and one right answer wrong
([benchmarks](benchmarks/README.md)).

## Compute an answer, or map a folder

```bash
ada scan lahman/ --template answers.yml      # the files, how they join, the questions to ask first
ada ask flights.csv "average dep_delay by origin" --json   # the numbers, the calculation, the rows used
ada claim lahman/ "Barry Bonds hit 73 home runs in 2001." --answers answers.yml
```

`ada scan` asks what the analysis should find out, what a row is, and each
choice ADA would otherwise make by default, with its proposed answer; `ada
ask` and `ada claim` take the edited file. When an answer rests on a choice,
such as leaving World and region rows out of an average, it says which, and
`--decide` picks the other ([Decisions](docs/reference/decisions.md)). A
question or claim about a folder is answered from the file it fits, or from
two files joined along a link ADA is sure of. A question ADA can't answer
comes back refused, with the reason. See [Ask ADA](docs/reference/ask-ada.md)
and [Reading a folder of tables](docs/reference/folder.md).

Before it answers, ADA reports the rows it left out and why, counts each
subject once when several rows come from one, says when groups reverse a
pooled figure, adds up the count column of a table of counts, and gives a
reason instead of a number when the data are too thin.

`ada analyze data.csv` takes a first look at any table: problems in the data,
which groups differ, what relates to what and changes over time, each with
its calculation.

## Use it from Claude Code or Codex

```bash
ada skill               # .claude/skills/ada (Claude Code) and .agents/skills/ada (Codex)
ada skill --user        # the same in your home folder, for every project
```

It also installs as a plugin ([how](plugins/automated-data-analyst/README.md)).
The [skill](ada/skills/ada/SKILL.md) has the assistant let ADA do the
arithmetic, check every number it states with `ada claim`, and ask you ADA's
question instead of choosing a reading.

## What it doesn't do yet

Numbers that need three files joined, many phrasings ("horsepower" for `hp`,
slopes, several numbers in one claim), and some year columns read as numbers
rather than dates. These come back unverified or refused with a
reason. See [Not covered yet](docs/reference/claims.md#not-covered-yet).

## Progress

73 cases on 20 public datasets, each expected value computed separately with
pandas, SciPy and statsmodels; a fifth of each dataset's cases are holdout
cases, only scored ([details](results/verification/README.md)).

| | ADA 0.5.0 | Now |
|---|---|---|
| Dev cases | 9/58 | 50/58 |
| Holdout cases | 0/15 | 14/15 |
| Wrong verdicts on claims | | 0/35 |
| True claims verified | 0/25 | 23/25 |
| False claims caught | 0/7 | 6/7 |

Each change is also scored on rounds of new public data, six datasets at a
time from statistics offices, city portals and research projects, with claims
written as reports write them; the pull request gives the scores.

## ADA Lab

The research workflow (paper tables, figures and a methods draft from a
study.yml), the modelling mode (`ada model`) and the browser app are in
[ADA Lab](https://github.com/saineshnakra/ada-lab): `pip install ada-lab`.
With it installed, `ada init`, `ada build`, `ada model` and `ada app` run
from the `ada` command too.

## More

[All documentation](docs/README.md) · [Contributing](CONTRIBUTING.md) ·
[Credits](docs/credits.md) · [Citing](CITATION.cff) · MIT [License](LICENSE)
