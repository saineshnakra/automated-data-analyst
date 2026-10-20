# ADA: Automated Data Analyst

[![CI](https://github.com/saineshnakra/automated-data-analyst/actions/workflows/ci.yml/badge.svg)](https://github.com/saineshnakra/automated-data-analyst/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-20a779.svg)](https://github.com/saineshnakra/automated-data-analyst/blob/main/LICENSE)
[![Stars](https://img.shields.io/github/stars/saineshnakra/automated-data-analyst?style=flat&logo=github&color=f5b731)](https://github.com/saineshnakra/automated-data-analyst/stargazers)

ADA is a first pass at a CSV or Excel file you haven't looked at yet. It
detects the date, the measure and the grouping column, cleans what it can,
and builds the trend, a breakdown by group, anomaly flags and a baseline
forecast. Every number comes from pandas and is shown with the calculation
that produced it.

It works on sales exports, but it was also built for research data: lab
assays, field surveys, instrument logs. It sums only money and counts, averages
measurements such as temperature, pH or concentration instead of adding them
up, and reads `NA`, `nd` or `<LOD` in a numeric column as missing.

[**Try the web app**](https://automated-data-analyst.streamlit.app/) (no sign-up, no API key) or `pip install` it.

![Asking ADA a plain-English question and getting a pandas-backed answer with its calculation](https://raw.githubusercontent.com/saineshnakra/automated-data-analyst/main/assets/readme/ask-ada.gif)

## Quick start

```bash
pip install automated-data-analyst
```

```python
import ada

report = ada.analyze("lab-measurements.csv")   # a path, a DataFrame or raw bytes
print(report)
```

```text
lab-measurements.csv — 936 rows, 02 Sep 2024 to 23 Feb 2026
  date=Run Date  measure=Viability % (mean)  segment=Treatment

+0.1 pts latest movement; Control has the highest average, at 91.6%.
...
  91.6%  Highest average by treatment
      Control has the highest average Viability %, at 91.6%. Compound B has the lowest, at 84.8%.
      calc: mean Viability % per Treatment; averages are not split into shares
  1  Anomalous periods
      Jul 2025 is the sharpest anomaly: viability % reached 81.5%, below the expected 88.1%–90.1% range.
      1 period sits outside the trendline band.
      calc: Period values vs Theil–Sen trendline ± a band calibrated so that only 1 stable series in 20 raises a flag
  ...
```

That file ships with the package: `ada.sample("Lab Measurements")` loads it,
and `ada.samples()` lists all four bundled datasets.

From the command line:

```bash
ada analyze measurements.csv          # the same report
ada analyze measurements.csv --json   # as JSON
ada schema measurements.csv           # which column ADA used for what, and how it combines
ada ask orders.csv "which category made the most revenue"
```

## What it does

- **Cleans conservatively and says what it changed**: encodings (UTF-8,
  UTF-16, Latin-1), `;` and tab delimiters, a title line above the header,
  `$1,203.55` and `(48.10)` style numbers, day/month order in dates, missing-value
  markers, labels spelled differently only in case or accents (`No indica`,
  `NO INDICA`). Identical rows are counted, not dropped.
- **Detects column roles**: the date, the measure, the grouping column and an
  ID column, from names, values and cardinality. Numeric category codes (a 0/1
  flag, a status, a set dose) group rows instead of being added up; a column of
  unique row numbers is an ID; a running total such as a rain gauge's is never
  the measure. A long table (one column naming the parameter, one holding the
  result) is analyzed one parameter at a time, and a matrix of sample or
  channel columns is measured by its row mean. Override any role with
  `ada.analyze(df, measure="OD600")` or in the app.
- **Knows what can be added up**: money and counts are summed; rates
  (`Viability %`, `Churn Rate`), measured levels (`Water Temp (°C)`, `pH`,
  `Concentration (mg/L)`), per-unit prices and anything it doesn't recognise as
  money or a count (`CO2`, `mpg`) are averaged. Averages get no totals and no
  "share of total", and a change in a rate is reported in percentage points.
  When the name misleads, `aggregation="sum"` or `"mean"` (or **Combine rows
  by** in the app) says how the measure combines.
- **Prints readings in full**: `K` and `M` are kept for money and large
  counts, so a heart rate of 1,111 is not shown as `1.1K`; very small or very
  large readings use scientific notation (`3.624e+11`). A `$` needs both a
  money name and values that look like money, so an income band coded 1 to 15
  is not printed as dollars. Figures compared in one sentence never print
  alike, numeric groups are named with their column ("Status 2"), and an
  average resting on fewer than 30 rows says so.
- **Builds a timeline**: picks a period grain the data can fill, down to the
  hour, minute or second for a sensor log, drops a final period the data only
  partly covers, and notes both under the chart. A `Year` column (with a
  `Month` column, if there is one) counts as a date. Clock times with no date
  are read as times of day, never given today's date, and a column of elapsed
  seconds or minutes can be the time axis.
- **Compares like with like**: the latest change is measured on the groups
  that reported in both periods, a series with a yearly cycle is set against
  the same time of year, and a group's figure says when it covers only part
  of the dates. Logs stacked one after another in a file, and daily summaries
  appended below hourly readings, are found and kept apart.
- **Flags anomalies**: periods outside a band around a Theil-Sen trendline, or
  around the usual level for the time of year when the series has a yearly
  cycle. The band is calibrated by simulation so a stable series raises a false
  alarm in about one analysis in twenty.
- **Forecasts with a backtest**: a trendline plus month-of-year seasonality,
  scored on held-out periods against a no-change forecast. It refuses to run on
  too little history, and says plainly when the forecast did worse than
  assuming no change.
- **Answers plain-English questions** ("average pH by site", "mean signal by
  timepoint", "which month had the highest wind", "is pH correlated with
  temperature", "how many samples have pH below 5.5", "what share have
  diabetes", "average PM2.5 by hour") by turning them into a fixed query plan
  that pandas runs. Every column, value, grouping and condition the question
  names must reach the plan; otherwise the question is refused with the part
  ADA couldn't use, rather than answered approximately. A short name that fits
  one column ("strength" for `Concrete compressive strength (MPa)`) is used
  and the answer says so. Each answer states its filters, how many rows it
  used and whether the row limit left rows out.
- **Reports correlations with a 95% interval**, the sample size and a rank
  correlation check. In the brief it says that the pair was picked as the
  strongest of several; ask about a pair and it reports the same figures for
  that pair. Participant numbers, plate positions and near-duplicate pairs (a
  reading and its adjusted copy) are left out of the brief's pick.
- **Analyzes a slice you select**: draw a box or lasso on the Explore chart, or
  click some bars, and get the same analysis for those rows beside the whole.

The web app accepts files up to 25 MB. By default up to 1,000,000 rows are
analyzed (`row_limit=` changes that): the newest rows of a table with a date,
or a random sample of one without, and every answer says which.

<p align="center">
  <img src="https://raw.githubusercontent.com/saineshnakra/automated-data-analyst/main/assets/readme/anomaly-forecast.png" width="49%" alt="ADA dashboard for the lab sample: viability over time with an anomaly and a forecast, change per treatment, heatmap, distribution and scatter">
  <img src="https://raw.githubusercontent.com/saineshnakra/automated-data-analyst/main/assets/readme/evidence-ledger.png" width="49%" alt="ADA evidence ledger listing each finding with its calculation">
</p>

## What it doesn't do

It describes one table. It has no domain knowledge and is not a statistics
package: no hypothesis tests, no study design, no multiple-comparison
correction. The recommendations it prints are suggestions for what to look at
next, not causal claims. If you need a p-value for a specific comparison, ADA
is the step before that.

## Using it from Python

```python
report = ada.analyze(df, date="Survey Date", measure="Water Temp (°C)", dimension="Site")
ada.analyze(df, measure="Rainfall (mm)", aggregation="sum")   # monthly totals, not averages

report.profile()          # one row per column: type, missing, distinct values
report.evidence           # findings, each with .statement and .calculation
report.trend()            # the measure per period, with notes on adjustments
report.segments()         # the measure by group, as a DataFrame
report.anomalies          # flagged periods
report.forecast()         # forecast with .backtest
report.ask("average pH by site")
report.focus("North Marsh")   # the same analysis for one group
report.to_dict()          # JSON-safe, versioned by schema_version
report.to_markdown()      # the report the web app downloads
```

When ADA can't answer, it returns a `Refusal` instead of a number. A refusal is
falsy and has a `code` from `ada.REFUSAL_CODES`:

```python
forecast = report.forecast()
if not forecast:
    print(forecast.code, forecast.message)
    # thin_history A forecast needs more history than this; the table has 4 periods.
```

`report.anomalies` returns an empty tuple when it checked and found nothing,
and a `Refusal` when it couldn't check (no date column, too few periods). Both
are falsy, so check the type when the difference matters.

## Scheduled checks

`ada check` runs anomaly detection and reports through its exit code, so it can
sit in a cron job:

```bash
ada check saas-subscriptions.csv --frequency M
# anomaly:2025-04-01  162,605.02, below the expected 267,236.53
```

| Exit code | Meaning |
|---|---|
| `0` | checked, nothing flagged |
| `1` | at least one period flagged |
| `2` | the input could not be read |
| `3` | could not check: no date column or too few periods |

Pass `--frequency` (or `frequency=` in Python) for anything scheduled, so the
period grain doesn't change from one run to the next. The headline, the
findings, the anomalies and the forecast all use that grain. `--aggregation sum`
or `mean` fixes how the measure combines in the same way. A period with no rows
counts as zero for a summed measure and is skipped for an averaged one, so a
gap in a feed of readings is not flagged as a drop to zero.

## Sample datasets

All synthetic. Pick one in the web app or load it with `ada.sample(name)`.

| Sample | What it shows |
|---|---|
| [Lab measurements](https://github.com/saineshnakra/automated-data-analyst/blob/main/ada/samples/lab-measurements.csv) | A cell-culture assay by treatment: viability averaged while live-cell counts are summed, one declining arm, one bad plate |
| [SaaS subscriptions](https://github.com/saineshnakra/automated-data-analyst/blob/main/ada/samples/saas-subscriptions.csv) | A revenue drop in April 2025 that anomaly detection finds |
| [Support tickets](https://github.com/saineshnakra/automated-data-analyst/blob/main/ada/samples/support-tickets.csv) | No money column, so resolution hours are averaged rather than summed |
| [Ecommerce orders](https://github.com/saineshnakra/automated-data-analyst/blob/main/ada/samples/ecommerce-orders.csv) | Returns as negative rows, so totals have to handle mixed signs |

## Running the web app locally

```bash
git clone https://github.com/saineshnakra/automated-data-analyst.git
cd automated-data-analyst
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[app]"
streamlit run app.py
```

No API key is needed; the app opens with a demo dataset. `pip install -e
".[app,ai,dev]"` adds the optional AI layer and the linter. Tested on Python
3.11, 3.12 and 3.13.

## Does my data leave my machine?

**Running ADA yourself: no.** Cleaning, detection, charts and answers are
computed locally with pandas, with no network calls.

**The hosted web app:** your file is uploaded to the Streamlit server and held
in memory for the session. It is not written to a database. If that matters,
run it locally.

**The optional AI layer** (only with an API key) adds a query planner for
questions the rules can't parse and a written summary. It receives column
names, types and the computed findings, not rows. Findings can name a group
(for example a site or a product). A planned query is shown to you and runs
locally only after you approve it. Model-generated code is never executed.

Details: [Privacy](https://github.com/saineshnakra/automated-data-analyst/blob/main/docs/privacy.md) · [SECURITY.md](https://github.com/saineshnakra/automated-data-analyst/blob/main/SECURITY.md)

## Documentation

| Page | Contents |
|---|---|
| [Concepts](https://github.com/saineshnakra/automated-data-analyst/blob/main/docs/concepts.md) | Terms: measure, segment, period, rates and levels, evidence |
| [How it works](https://github.com/saineshnakra/automated-data-analyst/blob/main/docs/how-it-works.md) | The steps from upload to report |
| [Architecture](https://github.com/saineshnakra/automated-data-analyst/blob/main/docs/architecture.md) | Which module does what |
| [Reference](https://github.com/saineshnakra/automated-data-analyst/blob/main/docs/README.md#reference) | One page per step, with the exact rules and thresholds |
| [Development](https://github.com/saineshnakra/automated-data-analyst/blob/main/docs/development.md) | Setup, tests, CI |
| [FAQ](https://github.com/saineshnakra/automated-data-analyst/blob/main/docs/faq.md) | Short answers |

Background on why it refuses rather than guesses:
[I Built an AI Data Analyst That Tells You When It Hallucinates](https://medium.com/@saineshnakra/i-built-an-ai-data-analyst-that-tells-you-when-it-hallucinates-6051609c3f4a).

## Contributing

Bug reports with a sample file (synthetic is fine) are the most useful thing.
For code, see [CONTRIBUTING.md](https://github.com/saineshnakra/automated-data-analyst/blob/main/CONTRIBUTING.md),
the [good first issues](https://github.com/saineshnakra/automated-data-analyst/labels/good%20first%20issue)
and the [roadmap](https://github.com/saineshnakra/automated-data-analyst/blob/main/ROADMAP.md).
New analysis rules need a test, including how they behave on thin data.

If ADA is useful to you, a star helps other people find it.

## Citing

There is a `CITATION.cff` in the repository root; GitHub's **Cite this
repository** button turns it into BibTeX or APA.

## License

[MIT](https://github.com/saineshnakra/automated-data-analyst/blob/main/LICENSE) · Copyright (c) 2024 Sainesh Nakra
