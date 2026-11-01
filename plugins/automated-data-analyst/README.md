# Automated Data Analyst (ADA)

This plugin adds one skill, `ada`, that has the assistant analyze data with
[ADA](https://github.com/saineshnakra/automated-data-analyst), an open-source
Python package, instead of doing the arithmetic itself. ADA computes every
number with pandas, SciPy and statsmodels and prints the calculation behind
it, or refuses with a reason when a number can't be computed honestly. The
assistant reports ADA's numbers, each with its test or calculation, and
passes refusals on, so everything it reports can be checked against ADA's
output and the data.

## What it does

- **A check of any number.** `ada claim` checks a number stated about a table,
  in a report or in the assistant's own answer, and says verified, or
  unverified with the reason and the one question that would settle it. The
  assistant says which of its numbers are verified and which are not.
- **An answer computed from the data.** `ada ask` answers a question about a
  table or a folder with the number, its calculation and the rows used, or
  refuses with the reason.
- **A map of a folder.** `ada scan` lists the files, the keys that join them
  and the questions to settle before an analysis.
- **A first look at any table.** ADA reads a CSV or Excel file and reports
  what is in it: data-quality issues with their rows, group comparisons,
  relationships and trends, each with the test and the arithmetic.

## Requirements

A shell with Python 3.11 or newer. The skill installs ADA with:

```bash
pip install automated-data-analyst
```

## What it runs and sends

- The plugin contains no code: only the skill's instructions, its manifests,
  an icon, this README and the license.
- The skill has the assistant run `pip install automated-data-analyst`,
  which downloads ADA and its dependencies from PyPI, and then `ada`
  commands on files in your project.
- The `ada` commands the skill uses run on your machine and make no network
  calls. Your data is not sent anywhere by the plugin or by ADA.
- ADA writes a file only when a command is told to, such as the answers file
  `ada scan --template` writes.

## Install

In Claude Code:

```text
/plugin marketplace add saineshnakra/automated-data-analyst
/plugin install automated-data-analyst@automated-data-analyst
```

In Codex:

```bash
codex plugin marketplace add saineshnakra/automated-data-analyst
codex plugin add automated-data-analyst@automated-data-analyst
```

Without the plugin, `ada skill` writes the same skill into
`.claude/skills/ada/` and `.agents/skills/ada/` in the current project.

## Use

Ask in your own words, for example:

- "Use ADA to look at data/results.csv and tell me what is in it."
- "Check the numbers in my report against data/results.csv."
- "Check my data for problems before I analyze it."

## Links

- [Documentation](https://github.com/saineshnakra/automated-data-analyst/blob/main/docs/README.md)
- [Source code](https://github.com/saineshnakra/automated-data-analyst)
- [Issues](https://github.com/saineshnakra/automated-data-analyst/issues)
- [ADA on PyPI](https://pypi.org/project/automated-data-analyst/)

## License

MIT. See [LICENSE](LICENSE).
