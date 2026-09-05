# FAQ

**Does my data leave my machine?**
No. Cleaning, schema detection, every chart, every evidence card, and every Ask
ADA answer are computed locally with pandas. If you opt into the AI layer, only
column schema and computed evidence are sent — never rows or cell values. See
[Privacy](privacy.md).

**Do I need an OpenAI API key?**
No. ADA is a complete analyst without one. A key adds two things: a query-planner
fallback for questions the rules cannot parse, and an optional strategic
narrative.

**What files can I analyze?**
CSV (comma, semicolon, or tab separated), XLSX, and XLSM — including choosing a
worksheet from a multi-sheet workbook. Limits are 25 MB per file and 250,000
rows analyzed. See [Reading files](reference/reading-files.md).

**How is this different from pasting a CSV into a chatbot?**
A chatbot gives you fluent prose you cannot audit, and your rows become part of
a prompt. ADA turns your question into an explicit query plan, executes it with
pandas on your machine, and prints the calculation under the answer.

**ADA picked the wrong column as my main metric.**
Open "Tune ADA's schema detection" in the app and override the date, metric, or
segment. Nothing is rebuilt from scratch. If the guess was wrong in a way that
looks general rather than specific to your file, that is worth an issue — see
[Schema detection](reference/schema-detection.md).

**Why is there no forecast?**
It needs at least 8 periods of history. Below that, ADA returns nothing rather
than guessing. See [Forecasting](reference/forecasting.md).

**Why did the last month disappear from my chart?**
Your data probably stops part-way through it. A partly-covered final period is
excluded, because charted as-is it looks like a collapse. The note under the
chart says so. See [Periods](reference/periods.md).

**Why is nothing flagged as an anomaly?**
Either you have fewer than 8 periods, or nothing is far enough from the
trendline. The band is calibrated so a stable series raises a false alarm about
once in twenty analyses — quiet is the intended behavior. See
[Anomalies](reference/anomalies.md).

**Can I self-host it?**
Yes. It is a standard Streamlit app:
`pip install -r requirements.txt && streamlit run app.py`, or deploy to any host
that runs Python.

**Is ADA a PandasAI alternative?**
Related use case, different approach. ADA's core is deterministic pandas
calculations rather than model-generated code, and every answer shows its
calculation. It is a complete Streamlit application, not a drop-in library
replacement.

**How do I add a new question shape or metric?**
Start with [Architecture](architecture.md#where-to-add-things), then the
reference page for the module you are touching.
