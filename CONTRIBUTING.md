# Contributing to ADA

Contributions that improve accuracy, explainability, usability, privacy or
maintainability are welcome.

To find something to work on, see the
[good first issues](https://github.com/saineshnakra/automated-data-analyst/labels/good%20first%20issue)
(each has context, file pointers and acceptance criteria) or
[ROADMAP.md](ROADMAP.md). For a larger change, open a feature request first so
the scope is agreed before you start.

## Setup

Python 3.11 or later.

```bash
git clone https://github.com/saineshnakra/automated-data-analyst.git
cd automated-data-analyst
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
ada analyze -                      # a first look at a bundled sample
```

The `dev` extra adds Ruff, and openpyxl for the workbooks the tests write. The
package alone needs only `pip install -e .`. No API key is needed. The browser
app, the research workflow and the modelling mode are developed in
[ADA Lab](https://github.com/saineshnakra/ada-lab).

## Tests and lint

```bash
python -m unittest discover -s tests
ruff check .
```

Run one test file with
`python -m unittest discover -s tests -p "test_nlq.py"`.

Tests of `ada chat` use a fake model server on this computer, so the suite
needs no network access or API key. Ruff settings are in `pyproject.toml`
(line length 110; rule sets `E`, `F`, `I`, `UP`, `B`).

CI (`.github/workflows/ci.yml`) runs on every pull request and every push to
`main`: lint, the test suite and `compileall` on Python 3.11, 3.12 and 3.13,
then builds the wheel, installs it and checks that the `ada` command checks a
claim, answers a question, scans a folder and serves the tools over MCP.

## Before you change code

| Read | For |
|---|---|
| [docs/concepts.md](docs/concepts.md) | The terms: measure, segment, period, evidence, plan |
| [docs/how-it-works.md](docs/how-it-works.md) | The steps from a file to a report |
| [docs/architecture.md](docs/architecture.md) | Which module owns what |

Then read the [reference page](docs/README.md#reference) for the module you are
changing. Each ends with a "Changing this" section.

## Design rules

The reasoning is in
[docs/architecture.md](docs/architecture.md#design-rules).

1. Calculations go in the engine modules, not in the command line, the tools
   or the MCP server.
2. `ada/aggregation.py` is the only place that decides what a period is.
3. Formatting changes how a value looks, never what it is.
4. Evidence is calculated; recommendations are interpretation. Keep them
   separate, and do not claim cause.
5. Model-generated code is never executed: a model picks a tool and its
   arguments, and ADA runs its own calculations.
6. Every analytical rule has a test, including how it behaves on thin data.
7. Nothing needs an API key.

A new evidence card or recommendation must name the calculation behind it, and
if a heuristic can mislead, the limitation goes in the tests and in the text
the user sees.

## Pull requests

- Keep each pull request focused on one change. Split a larger change into a
  series that each pass CI on their own, for example the engine change with
  its tests first and the command-line change after.
- Avoid unrelated formatting changes.
- Do not add a dependency without a concrete need.
- Use synthetic data in tests and docs. Never commit customer or employer data,
  credentials or `.streamlit/secrets.toml`.

In the description, cover:

1. The user problem
2. Behaviour before and after
3. Any calculation, privacy boundary or external call affected
4. How you tested it
5. Example output for changes to what ADA prints

## Conduct and security

Participation is governed by [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). Report
vulnerabilities privately as described in [SECURITY.md](SECURITY.md).
