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

- **A first look at any table.** ADA reads a CSV or Excel file and reports
  what is in it: data-quality issues with their rows, group comparisons,
  relationships and trends, each with the test and the arithmetic.
- **A paper's assets from a study.** The assistant asks the researcher what
  only they know (what one row is, which column identifies a subject, the
  design, which analyses were planned), writes a `study.yml`, and runs
  `ada build`. That makes a data audit, a data dictionary, a record of every
  exclusion, descriptive and results tables in CSV, Word and LaTeX, figures
  with their source data and captions, and a methods draft.
- **A check of the build.** `ada verify` rebuilds everything from
  `study.yml` and the data in a scratch folder and compares every number.

The design decides the test: Welch's t-test or ANOVA for independent groups,
a paired t-test, or a linear mixed model for repeated or clustered data.
Designs ADA doesn't support, such as crossover trials or survival outcomes,
are refused rather than approximated.

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
- ADA's command line runs on your machine and makes no network calls. Your
  data is not sent anywhere by the plugin or by ADA.
- ADA writes its output to `study.yml` and `paper-assets/` in the study
  folder, and `ada share` copies only the files a study allows to `share/`.

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
- "Set up a study from data/assay.csv and build the paper's tables and figures."
- "Check my data for problems before I analyze it."

## Links

- [Documentation for the study workflow](https://github.com/saineshnakra/automated-data-analyst/blob/main/docs/research.md)
- [Source code](https://github.com/saineshnakra/automated-data-analyst)
- [Issues](https://github.com/saineshnakra/automated-data-analyst/issues)
- [ADA on PyPI](https://pypi.org/project/automated-data-analyst/)

## License

MIT. See [LICENSE](LICENSE).
