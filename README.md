# ADA — a dashboard that explains itself

[![CI](https://github.com/saineshnakra/automated-data-analyst/actions/workflows/ci.yml/badge.svg)](https://github.com/saineshnakra/automated-data-analyst/actions/workflows/ci.yml)
[![Streamlit](https://img.shields.io/badge/Streamlit-live_demo-ff4b4b?logo=streamlit&logoColor=white)](https://automated-data-analyst.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**One business file in. A decision-ready dashboard out.**

ADA is an automated business analyst for people who do not want to configure a BI tool. Upload a CSV or Excel workbook and ADA cleans the data, detects the likely metric, date, and business segment, builds a dashboard, explains what happened, and recommends what to investigate next.

**[Explore the live product](https://automated-data-analyst.streamlit.app/)**

## The product principle

Most analytics tools stop at charts. ADA separates the work into two explicit layers:

1. **What the data says** — traceable calculations such as period movement, segment contribution, concentration, relationships, exceptions, and completeness.
2. **What ADA recommends** — rule-based interpretation and prioritized next actions, each linked to its supporting evidence.

Calculations are never presented as causal proof, and recommendations are never disguised as facts.

## Zero-configuration workflow

```text
CSV or Excel upload
        │
        ▼
Automatic cleanup and schema detection
        │
        ├──► Executive brief
        ├──► Business KPI cards
        ├──► Auto-generated dashboard
        ├──► Evidence-backed recommendations
        └──► Cleaned data + downloadable report
```

ADA automatically looks for:

- A primary outcome such as revenue, sales, profit, cost, amount, or units
- A time field for week-over-week, month-over-month, or quarterly movement
- A useful segment such as product, category, channel, region, customer, or status
- Identifiers, data-quality problems, outliers, concentration, and numeric relationships

If the source schema is unusual, the **Tune detection** panel lets the user override the metric, date, and segment without rebuilding charts manually.

## Features

- CSV, XLSX, and XLSM uploads up to 25 MB
- Included synthetic operating-data demo
- Automatic column cleanup, type inference, duplicate removal, and cleaning audit
- Executive headline and plain-English briefing
- Four automatically selected business KPIs
- Trend, segment, distribution, and relationship visualizations
- Separate factual evidence and analyst-recommendation layers
- Transparent calculation shown under every evidence card
- Downloadable executive Markdown brief and cleaned CSV
- No API key, generated-code execution, or external model call

## Privacy and trust model

ADA processes the file inside the active Streamlit session. It does not send uploaded rows to an LLM API and does not require secrets or API credits.

Recommendations are deterministic interpretations of visible calculations. They are useful prompts for investigation, not guaranteed causal conclusions. Hosted infrastructure still handles request traffic and application memory, so users should only upload data they are authorized to process there.

See [SECURITY.md](SECURITY.md) for the complete security model.

## Code structure

- `app.py` — product interface and visualization layer
- `analysis.py` — conservative cleaning and data profiling
- `business_insights.py` — schema detection, business calculations, evidence, and recommendations
- `demo_data.py` — deterministic synthetic business dataset
- `file_io.py` — validated CSV and Excel parsing
- `tests/` — cleaning, detection, business-logic, report, and rendering tests

## Run locally

```bash
git clone https://github.com/saineshnakra/automated-data-analyst.git
cd automated-data-analyst
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
streamlit run app.py
```

No `.env` file is needed.

## Test

```bash
python -m pip install -r requirements-dev.txt
ruff check .
python -m unittest discover -s tests -v
```

GitHub Actions runs the same checks on every push and pull request.

## Deploy

Deploy `app.py` on Streamlit Community Cloud. The repository includes its theme, dependency, and upload-limit configuration. No secrets are required.

## Author

Built by [Sainesh Nakra](https://sainesh.com/).

## License

[MIT](LICENSE)
