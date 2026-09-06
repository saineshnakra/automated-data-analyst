# ADA — Automated Data Analyst

[![CI](https://github.com/saineshnakra/automated-data-analyst/actions/workflows/ci.yml/badge.svg)](https://github.com/saineshnakra/automated-data-analyst/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-20a779.svg)](https://github.com/saineshnakra/automated-data-analyst/blob/main/LICENSE)

**Upload a CSV or Excel file. Get a dashboard, an executive brief, anomaly
flags, a forecast, and answers to plain-English questions — with the calculation
shown under every number.**

[Live demo](https://automated-data-analyst.streamlit.app/) ·
[Documentation](https://github.com/saineshnakra/automated-data-analyst/tree/main/docs) ·
[Roadmap](https://github.com/saineshnakra/automated-data-analyst/blob/main/ROADMAP.md) ·
[Contributing](https://github.com/saineshnakra/automated-data-analyst/blob/main/CONTRIBUTING.md)

**Source:** [github.com/saineshnakra/automated-data-analyst](https://github.com/saineshnakra/automated-data-analyst)

![ADA turns CSV and Excel files into decision-ready business dashboards](https://raw.githubusercontent.com/saineshnakra/automated-data-analyst/main/assets/ada-social-preview.png)

## What it does

ADA reads your file, works out which column is the metric, which is the date,
and which is the segment, then builds the analysis around that.

- **Dashboard** — trend, segment breakdown, movement waterfall, segment × period heatmap
- **Ask ADA** — plain-English questions answered locally with pandas, calculation shown
- **Anomaly flags** — periods outside a calibrated band, sized so a stable series false-alarms about once in twenty analyses
- **Forecast** — a guarded baseline that refuses to run on thin history and reports when it was no better than assuming no change
- **Evidence and next steps** — every finding carries its calculation; recommendations are labelled as interpretation, never as cause
- **Downloads** — Markdown executive brief and cleaned CSV

Limits: 25 MB per file, 250,000 rows analyzed. Formats: `.csv`, `.xlsx`, `.xlsm`.

### Nothing to upload? Try a sample

Pick **Try a sample dataset** in the app, or download one from [`samples/`](https://github.com/saineshnakra/automated-data-analyst/tree/main/samples):

| Sample | What it shows |
|---|---|
| [SaaS subscriptions](https://github.com/saineshnakra/automated-data-analyst/blob/main/samples/saas-subscriptions.csv) | A real revenue drop the anomaly radar finds, and a forecast that beats no-change |
| [Support tickets](https://github.com/saineshnakra/automated-data-analyst/blob/main/samples/support-tickets.csv) | No revenue column, and a forecast honest enough to say it is useless |
| [Ecommerce orders](https://github.com/saineshnakra/automated-data-analyst/blob/main/samples/ecommerce-orders.csv) | Returns as negative rows, so totals cope with mixed signs |

All three are synthetic, so they carry no privacy or licensing baggage.

### Ask a business question. Get the number and its calculation.

![Ask ADA a plain-English question and receive a pandas-backed answer with its calculation](https://raw.githubusercontent.com/saineshnakra/automated-data-analyst/main/assets/readme/ask-ada.gif)

### Focus on one segment. The whole analysis regroups.

![Drill into one business segment and automatically regroup the dashboard by the next useful dimension](https://raw.githubusercontent.com/saineshnakra/automated-data-analyst/main/assets/readme/drilldown.gif)

<p align="center">
  <img src="https://raw.githubusercontent.com/saineshnakra/automated-data-analyst/main/assets/readme/anomaly-forecast.png" width="49%" alt="ADA dashboard showing anomaly markers, a guarded forecast, movement waterfall, and segment heatmap">
  <img src="https://raw.githubusercontent.com/saineshnakra/automated-data-analyst/main/assets/readme/evidence-ledger.png" width="49%" alt="ADA evidence ledger showing calculations, anomalies, concentration, correlation, and detected schema">
</p>

## Run it

```bash
git clone https://github.com/saineshnakra/automated-data-analyst.git
cd automated-data-analyst
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
streamlit run app.py
```

No API key required. The app opens with a built-in demo dataset.

## Does my data leave my machine?

**Running ADA yourself: no.** Cleaning, schema detection, every chart, and every
Ask ADA answer are computed locally with pandas, with no network call at all.

**Using the hosted demo: your file is uploaded to a Streamlit server**, because
that is what uploading a file to a website means. It is held in memory for the
session and never written to a database. If that matters for your data, run ADA
locally — it is four commands above and needs no key.

An optional AI layer adds two things when you supply a key: a query planner for
questions the rules cannot parse, and a strategic narrative. The planner shows
its proposed calculation and waits for your confirmation before ADA executes it
locally. **Neither ever receives your rows.** They receive column names, types,
and already-computed evidence — and because an evidence sentence names the
segment it is about, a segment label such as a customer or product name can
appear in it. Nothing else from a cell does. Model-generated code is never
executed.

Full details: [Privacy](https://github.com/saineshnakra/automated-data-analyst/blob/main/docs/privacy.md) · [SECURITY.md](https://github.com/saineshnakra/automated-data-analyst/blob/main/SECURITY.md)

## Documentation

| Page | What you get |
|---|---|
| [Concepts](https://github.com/saineshnakra/automated-data-analyst/blob/main/docs/concepts.md) | The words ADA uses: measure, segment, period, evidence, plan |
| [How it works](https://github.com/saineshnakra/automated-data-analyst/blob/main/docs/how-it-works.md) | Upload to dashboard, step by step |
| [Architecture](https://github.com/saineshnakra/automated-data-analyst/blob/main/docs/architecture.md) | Which file does what, and why |
| [Reference](https://github.com/saineshnakra/automated-data-analyst/blob/main/docs/README.md#reference) | One page per pipeline step |
| [Development](https://github.com/saineshnakra/automated-data-analyst/blob/main/docs/development.md) | Setup, tests, CI, conventions |
| [FAQ](https://github.com/saineshnakra/automated-data-analyst/blob/main/docs/faq.md) | Short answers to common questions |

For the design story behind the project, read
[I Built an AI Data Analyst That Tells You When It Hallucinates](https://medium.com/@saineshnakra/i-built-an-ai-data-analyst-that-tells-you-when-it-hallucinates-6051609c3f4a).

## Contributing

Good places to start: a new question shape for Ask ADA, a new deterministic
metric, schema-detection fixtures, chart accessibility, adversarial test data.

Read [CONTRIBUTING.md](https://github.com/saineshnakra/automated-data-analyst/blob/main/CONTRIBUTING.md), browse the
[good first issues](https://github.com/saineshnakra/automated-data-analyst/labels/good%20first%20issue),
or pick something from the [roadmap](https://github.com/saineshnakra/automated-data-analyst/blob/main/ROADMAP.md).

Every new recommendation needs a test and the calculation that supports it.

## Building on ADA?

A link back to this repo is appreciated. If you've shipped something with it,
[open an issue](https://github.com/saineshnakra/automated-data-analyst/issues/new) and I'll list it here.

ADA is MIT licensed, so you are free to use, change and ship it — commercially
too. The only thing the licence asks is that the copyright notice travels with
the code.

**Built with ADA**

- *Yours could be here.*

## License

[MIT](https://github.com/saineshnakra/automated-data-analyst/blob/main/LICENSE) · Copyright (c) 2024 Sainesh Nakra

Originally built at [github.com/saineshnakra/automated-data-analyst](https://github.com/saineshnakra/automated-data-analyst).
