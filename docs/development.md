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

Python: tested on 3.11, 3.12 and 3.13 (the CI matrix); `requires-python` is
`>=3.11`. No API key needed — the app opens with a built-in demo dataset.

To reproduce CI's environment exactly rather than whatever resolves today, add
the pin file to the install line:

```bash
python -m pip install -r requirements-dev.txt -c constraints.txt
```

## Dependencies: ranges and the tested set

Two files describe the dependencies, and they answer different questions.

- `requirements.txt` states the **supported ranges** — what ADA is written
  against and expected to keep working with.
- `constraints.txt` states the **tested set** — the exact versions a clean
  `pip install -r requirements-dev.txt` resolved to when the file was last
  refreshed, and therefore the versions the suite is known to pass on. CI
  installs with `-c constraints.txt`, so a green run always refers to this set.

A constraints file only pins; it never adds a package. Installing without it is
supported, it is just not what CI checked.

Refresh it when a range in `requirements.txt` changes, or deliberately, to pick
up newer releases:

```bash
python -m venv /tmp/lockenv
/tmp/lockenv/bin/pip install -r requirements-dev.txt -q
/tmp/lockenv/bin/pip freeze > constraints.txt
```

Then put the header comment back, drop any editable or `file://` lines, run the
checks below against the new set, and commit the file in the same pull request
as the change that motivated it. The date and interpreter in the header say
when and where it was produced; keep them current.

## Checks

Run these before opening a pull request. CI runs exactly these.

```bash
ruff check .
python -m unittest discover -s tests -v
python -m compileall -q *.py tools tests
python -c "import app"
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
| `test_bug_bash.py` | Behaviours pinned after walking every customer journey |
| `test_app.py` | Rendering smoke tests, and that the app survives without the AI layer |

Tests use fake clients for anything model-related, so the suite needs no network
and no API credits.

## CI

`.github/workflows/ci.yml` runs on every push to `main` and every pull request:
install with `-c constraints.txt`, `ruff check .`, the full unit suite,
`compileall` over every top-level module, then `import app`. The whole job runs
once per interpreter in the matrix — Python 3.11, 3.12 and 3.13 — on Ubuntu.

### What a green CI run does and does not establish

A green run establishes that, on the pinned dependency set and on each of the
three interpreters, the code lints clean, every unit test passes, every module
compiles, and the app imports. Those are real guarantees and they catch most of
what breaks.

It does not measure whether the product answers human questions correctly. The
suite pins behaviours that were found and fixed — a parse, an aggregation, a
threshold — but no test asks a few hundred realistic business questions and
scores the answers against what an analyst would say. That semantic evaluation
set is a separate, open item: the [roadmap](../ROADMAP.md) lists an evaluation
corpus, and until
it exists, "CI is green" means "nothing we already check has regressed", not
"the answers are right".

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
