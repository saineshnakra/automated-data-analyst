# Development

## Setup

```bash
git clone https://github.com/saineshnakra/automated-data-analyst.git
cd automated-data-analyst
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements-dev.txt
streamlit run app.py
```

Python **3.11+**. No API key needed — the app opens with a built-in demo
dataset.

## Checks

Run all three before opening a pull request. CI runs exactly these.

```bash
ruff check .
python -m unittest discover -s tests -v
python -m compileall -q aggregation.py analysis.py ai_insights.py anomalies.py \
  business_insights.py demo_data.py file_io.py forecasting.py formatting.py \
  nlq.py pipeline.py schema.py timeseries.py ui.py app.py tools tests
```

Ruff config lives in `pyproject.toml`: line length 110, rule sets `E`, `F`, `I`,
`UP`, `B`, with `E501` relaxed for `app.py` and `ui.py`.

## Running one test file

```bash
python -m unittest discover -s tests -p "test_nlq.py" -v
```

## The test suite

| File | Covers |
|---|---|
| `test_analysis.py` | Cleaning and profiling |
| `test_file_io.py` | Parsing, encodings, worksheet handling, rejections |
| `test_pipeline.py` | Preparation, role overrides, drill-down |
| `test_trend_series.py` | Grain choice, partial periods, zero-filling |
| `test_timeseries.py` | Theil–Sen, calendar positions, robust scale |
| `test_anomalies.py` | Detection and the calibrated threshold |
| `test_forecasting.py` | Guards, seasonality, backtest |
| `test_business_insights.py` | Evidence, KPIs, recommendations, formatting |
| `test_significance.py` | Correlation intervals and rank divergence |
| `test_nlq.py` | Question parsing and plan execution |
| `test_ai_insights.py` | Typed output and the privacy contract |
| `test_autovis.py` | Chart-form choice and series folding |
| `test_app.py` | Rendering smoke tests, and that the app survives without the AI layer |

Tests use fake clients for anything model-related, so the suite needs no network
and no API credits.

## CI

`.github/workflows/ci.yml` runs on every push to `main` and every pull request:
install, `ruff check .`, the full unit suite, then `compileall`. Python 3.11 on
Ubuntu.

## Conventions

**Where code goes** — see [Architecture](architecture.md). The short version:
calculations in the engine modules, never in a Streamlit callback; presentation
in `ui.py`; `app.py` orchestrates and holds session state.

**Data in tests** — synthetic only. Never commit customer or employer data.
`demo_data.py` is seeded, so it produces the same dataset every run.

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
# .streamlit/secrets.toml — already gitignored, never commit it
OPENAI_API_KEY = "your-key"
```

Do not put an owner-funded key on a public deployment without authentication and
spending controls.
