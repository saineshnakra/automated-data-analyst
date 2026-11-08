# Development

## Setup

```bash
git clone https://github.com/saineshnakra/automated-data-analyst.git
cd automated-data-analyst
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
ada analyze -                    # a first look at a bundled sample
```

Python 3.11 or later. No API key is needed. The browser app, the research
workflow and the modelling mode are developed in
[ADA Lab](https://github.com/saineshnakra/ada-lab); `app.py` here only runs
ADA Lab's app for the hosted demo.

## Checks

Run these while you work:

```bash
ruff check .
python -m unittest discover -s tests -v
python -m compileall -q app.py ada tools tests
```

Before a merge, run all the checks, on each Python version you have:

```bash
python tools/check.py python3.11 python3.12 python3.13
```

For each version it makes a fresh environment, installs `requirements-dev.txt` and runs the three commands
above. Then it builds the wheel, installs it without the app extras, and runs `ada compute`, `ada claim`,
`ada ask`, `ada scan` and `ada mcp` from it. It exits 1 at the first check that fails.

Ruff config lives in `pyproject.toml`: line length 110, rule sets `E`, `F`, `I`,
`UP`, `B`, with `E501` relaxed for `app.py`.

## Running one test file

```bash
python -m unittest discover -s tests -p "test_nlq.py" -v
```

## The test suite

| File | Covers |
|---|---|
| `test_analysis.py` | Cleaning, number and date parsing, non-finite values, profiling |
| `test_file_io.py` | Parsing, encodings, worksheet handling, rejections, CSV export |
| `test_schema.py` | Identifier names and date-role choice |
| `test_pipeline.py` | Preparation, role overrides, drill-down, the row limit |
| `test_aggregation.py` | Grain choice, rate averaging, empty and partial periods, segment totals |
| `test_timeseries.py` | Theil–Sen, calendar positions, robust scale, observed periods |
| `test_anomalies.py` | Detection, the calibrated threshold, blank periods |
| `test_forecasting.py` | Guards, seasonality, backtest, blank periods |
| `test_metrics.py` | Amounts, rates and levels, the aggregation override, and which direction is good |
| `test_formatting.py` | Number display, rate scale, changes in points and differences |
| `test_business_insights.py` | Evidence, KPIs, recommendations, rate shares, number formatting |
| `test_significance.py` | Correlation intervals and rank divergence |
| `test_nlq.py` | Question parsing, refused phrasings and plan execution |
| `test_autovis.py` | Chart-form choice and series folding |
| `test_public_api.py` | The `ada` package API and the `ada` command |
| `test_selection.py` | Chart selections as filters, their wording, and the slice-versus-whole comparison |
| `test_claims.py` | Reading a claim's number and interval, the six reasons, `ada claim` |
| `test_folder.py` | `ada scan`: file profiles, proposed links, the questions and the answers file |
| `test_tools.py` | Tool definitions, `run`, paths kept inside the root, small models' slips |
| `test_mcp.py` | The MCP server over standard input and output and over HTTP |
| `test_chat.py` | `ada chat` against a fake model server |
| `test_moved.py` | The commands and names that moved to ADA Lab |

Tests of `ada chat` use a fake model server on this computer, so the suite
needs no network and no API credits.

## CI

`.github/workflows/ci.yml` runs the same checks as `tools/check.py` on GitHub
Actions. It is started by hand from the Actions tab, not on every push,
since GitHub Actions does not run on this account; `tools/check.py` is run
before each merge instead. A change to the checks goes in both files.

## Releasing

`.github/workflows/release.yml` publishes to PyPI when a GitHub release is
published. To release version 0.8.0:

1. Set `version = "0.8.0"` in `pyproject.toml`, `__version__` in
   `ada/__init__.py` and `version` in `server.json` (twice), add the release to
   `CHANGELOG.md` and the version to `CITATION.cff`, and merge that to `main`.
2. Create a release on GitHub with the tag `v0.8.0`, and paste the
   release's section of `CHANGELOG.md` as its notes.

The workflow checks that the tag matches the version, runs the tests, builds
the sdist and the wheel, and uploads them. PyPI accepts the upload through
trusted publishing, so no token is stored anywhere: the PyPI project lists this
repository's `release.yml` and its `pypi` environment as its publisher. A
version number can be uploaded once, so a fix after a release needs a new
version.

After the upload, the workflow lists the new version in the official MCP
Registry from `server.json`, which apps such as VS Code read their server
lists from. The registry checks that the PyPI package is this server by the
`mcp-name` line at the top of `README.md`, and that the workflow runs in this
repository by GitHub's OIDC token, so no token is stored for it either.

## Conventions

**Where code goes.** See [Architecture](architecture.md). In short:
calculations in the engine modules, never in the command line, the tools or
the MCP server, which only wrap the engine.

**Data in tests.** Synthetic only. Never commit customer or employer data.
`ada/demo_data.py` is seeded, so it produces the same dataset every run.

**Every analytical rule gets a test**, including the case where it degrades:
short history, one segment, all zeros, missing dates, negative values.

**Every reported number carries its calculation.** If you add a number to
what ADA reports, add the string that explains how it was computed.

**Formatting never changes values.** See [Formatting](reference/formatting.md).

## Pull requests

Explain, in the body:

1. The user problem and why it matters
2. Behavior before and after
3. Which calculation or trust boundary the change touches
4. How you tested it
5. Screenshots for anything visible

Keep pull requests focused. Avoid unrelated formatting churn, and do not add a
dependency without a concrete product need.
