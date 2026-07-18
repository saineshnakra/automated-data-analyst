# Automated Data Analyst

[![CI](https://github.com/saineshnakra/automated-data-analyst/actions/workflows/ci.yml/badge.svg)](https://github.com/saineshnakra/automated-data-analyst/actions/workflows/ci.yml)
[![Streamlit](https://img.shields.io/badge/Streamlit-live_demo-ff4b4b?logo=streamlit&logoColor=white)](https://automated-data-analyst.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A privacy-conscious CSV analysis app that produces a data-quality audit, factual observations, interactive Plotly visualizations, and downloadable results—without an API key or an LLM bill.

**[Try the live Streamlit app](https://automated-data-analyst.streamlit.app/)**

## Why this version is different

The original prototype used paid model calls to choose charts and summarize uploaded rows. It was costly, slow, difficult to reproduce, and included an unsafe generated-code execution path. This version removes that architecture entirely.

- No OpenAI or other model API
- No API credits or secrets required
- No generated Python execution
- No user rows sent to an external AI service
- Every observation is calculated directly from the dataset
- Conservative cleaning with a visible audit trail

## Features

- Upload a CSV up to 25 MB or explore the included demo
- Remove empty rows, empty columns, duplicate rows, and exported index columns
- Infer numeric and datetime values using conservative thresholds
- Profile types, missingness, cardinality, and example values
- Surface correlations, category concentration, missingness, time coverage, and possible outliers
- Explore histograms, category rankings, scatter plots, and a correlation heatmap
- Download the cleaned CSV and a Markdown analysis report
- Run locally or deploy to Streamlit Community Cloud

## Architecture

```text
CSV upload or demo
        │
        ▼
Conservative cleaning ──► cleaning audit
        │
        ▼
Deterministic profiling ─► observations + interactive charts
        │
        ▼
Cleaned CSV + Markdown report
```

`analysis.py` contains pure, testable data logic. `app.py` owns Streamlit presentation and Plotly chart selection. Keeping those responsibilities separate makes the results easier to verify and the interface easier to change.

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

GitHub Actions runs the same checks on pushes and pull requests.

## Deploy on Streamlit Community Cloud

1. Fork or push this repository to GitHub.
2. In Streamlit Community Cloud, create an app from the repository.
3. Select the deployment branch and set the entry point to `app.py`.
4. Deploy. There are no secrets to configure.

When a Community Cloud app is connected to a GitHub repository, pushes to its configured branch normally trigger an app update. If the current deployment points to another branch or file, update those settings or reboot the app from the Streamlit workspace.

## Privacy and limitations

Analysis happens in the Streamlit process and does not call an external AI API. A hosted Streamlit provider still handles the request and application memory, so only upload data you are authorized to process there.

Generated observations are exploratory, not causal conclusions. Correlations, IQR outliers, and inferred data types should be reviewed with domain context before making decisions.

See [SECURITY.md](SECURITY.md) for the security model and [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidance.

## Author

Built by [Sainesh Nakra](https://sainesh.com/).

## License

[MIT](LICENSE)
