# Development

## Setup

```bash
git clone https://github.com/saineshnakra/automated-data-analyst.git
cd automated-data-analyst
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[app,ai,dev]"
streamlit run app.py
```

Python 3.11 or later. No API key is needed; the app opens with a built-in demo
dataset.

## Checks

Run these before opening a pull request; CI runs the same commands.

```bash
ruff check .
python -m unittest discover -s tests -v
python -m compileall -q app.py ada tools tests
```

Ruff config lives in `pyproject.toml`: line length 110, rule sets `E`, `F`, `I`,
`UP`, `B`, with `E501` relaxed for `app.py` and `ada/ui.py`.

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
| `test_ai_insights.py` | Typed output, plan validation and the privacy contract |
| `test_autovis.py` | Chart-form choice, series folding, the Explore frame |
| `test_public_api.py` | The `ada` package API and the `ada` command |
| `test_selection.py` | Chart selections as filters, their wording, and the slice-versus-whole comparison |
| `test_app.py` | Rendering smoke tests, and that the app survives without the AI layer |

Tests use fake clients for anything model-related, so the suite needs no network
and no API credits.

## CI

`.github/workflows/ci.yml` runs on every push to `main` and every pull request.
On Python 3.11, 3.12 and 3.13 it runs `ruff check .`, the unit suite,
`compileall` and an import of `app.py`. A second job builds the wheel, installs
it without the app extras and runs the `ada` command.

## Releasing

`.github/workflows/release.yml` publishes to PyPI when a GitHub release is
published. To release version 0.6.0:

1. Set `version = "0.6.0"` in `pyproject.toml` and `__version__` in
   `ada/__init__.py`, add the release to `CHANGELOG.md` and the version to
   `CITATION.cff`, and merge that to `main`.
2. Create a release on GitHub with the tag `v0.6.0`, and paste the
   release's section of `CHANGELOG.md` as its notes.

The workflow checks that the tag matches the version, runs the tests, builds
the sdist and the wheel, and uploads them. PyPI accepts the upload through
trusted publishing, so no token is stored anywhere: the PyPI project lists this
repository's `release.yml` and its `pypi` environment as its publisher. A
version number can be uploaded once, so a fix after a release needs a new
version.

## Conventions

**Where code goes.** See [Architecture](architecture.md). In short:
calculations in the engine modules, never in a Streamlit callback; presentation
in `ada/ui.py`; `app.py` orchestrates and holds session state.

**Data in tests.** Synthetic only. Never commit customer or employer data.
`ada/demo_data.py` is seeded, so it produces the same dataset every run.

**Every analytical rule gets a test**, including the case where it degrades:
short history, one segment, all zeros, missing dates, negative values.

**Every displayed number carries its calculation.** If you add something to the
screen, add the string that explains how it was computed.

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

## Optional: running with an API key

Not required for development. To exercise the AI layer, either enter a key in
the app's session-only sidebar field, or set it in the environment:

```toml
# .streamlit/secrets.toml (gitignored; never commit it)
OPENAI_API_KEY = "your-key"
```

Do not put an owner-funded key on a public deployment without authentication and
spending controls.
