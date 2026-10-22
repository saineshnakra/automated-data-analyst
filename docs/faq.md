# FAQ

**Does my data leave my machine?**
Not when you run ADA yourself. Every calculation runs in-process with pandas,
and without an API key ADA makes no network call. On the hosted demo, your
file is uploaded to a Streamlit server and held in memory. If you turn on the
AI features, the column schema and computed evidence are sent, but not your
rows; a segment name can appear inside an evidence sentence. See
[Privacy](privacy.md).

**Do I need an OpenAI API key?**
No. A key adds two optional features: a query planner for questions the rules
cannot parse, and a written AI summary of the evidence.

**What files can I analyze?**
CSV (comma, semicolon, tab or pipe separated), XLSX and XLSM, one worksheet at
a time. The app accepts files up to 25 MB and analyzes up to 1,000,000 rows. See
[Reading files](reference/reading-files.md).

**How is this different from pasting a CSV into a chatbot?**
ADA turns the question into an explicit query plan, runs it with pandas on the
machine running ADA, and prints the calculation under the answer. Your rows are
not sent to a model.

**ADA picked the wrong column as the measure.**
Open "Tune ADA's schema detection" in the app and set the date, measure or
segment yourself. From Python, pass `date=`, `measure=` or `dimension=` to
`ada.analyze`. If the wrong guess looks like a general rule failing rather than
something specific to your file, please open an issue. See
[Schema detection](reference/schema-detection.md).

**Why is there no forecast?**
A forecast needs at least 8 periods with data. Below that, ADA does not
produce one. See [Forecasting](reference/forecasting.md).

**Why did the last month disappear from the chart?**
The data probably stops part-way through it. A final period the data only
partly covers is excluded, and the note under the chart says so. See
[Periods](reference/periods.md).

**Why is nothing flagged as an anomaly?**
Either there are fewer than 8 periods, or no period is far enough from the
trendline. The band is calibrated so that a stable series raises a false alarm
in about one analysis in twenty, so most stable series show no flags. See
[Anomalies](reference/anomalies.md).

**Can I use it without the browser?**
Yes. `import ada; report = ada.analyze("orders.csv")` returns an object with
the headline, evidence, KPIs, trend, anomalies, forecast and `ask()`. The
`ada` command (`ada analyze`, `ada ask`, `ada check`) does the same from a
shell. When ADA cannot answer, it returns a `Refusal` with a code rather than a
number.

**Can I self-host it?**
Yes. It is a standard Streamlit app with no database:
`pip install -e ".[app]"` from a clone, then `streamlit run app.py`. It runs
on any host that runs Python 3.11 or later.

**Is ADA a PandasAI alternative?**
The use case overlaps; the approach differs. ADA computes answers with fixed
pandas calculations rather than model-generated code, and shows the
calculation with each answer. It is a Streamlit application and a Python
package, not a drop-in replacement for PandasAI's API.

**How do I add a question shape or a metric?**
Start with [Architecture](architecture.md#where-to-add-things), then read the
reference page for the module you are changing.
