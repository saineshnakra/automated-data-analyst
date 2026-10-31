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

**Can a local open-source model use ADA?**
Yes. `ada.tools.definitions()` gives ADA's tools in the format
OpenAI-compatible servers take, Ollama, llama.cpp's server, vLLM and LM
Studio among them, and `ada.tools.run(name, arguments)` runs the one the
model calls. The model picks the tool and reads the answer; ADA does the
arithmetic. See [ADA as the math tool of a language model](llm-tools.md).

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

**Can ADA make the tables and figures for my paper?**
Yes. `ada init data.csv` writes a `study.yml` describing the study, `ada build`
makes the data audit, tables in Word, LaTeX and CSV, figures with their source
data, and a methods draft, and `ada verify` checks a rebuild gives the same
numbers. See [Paper assets for a study](research.md).

**Can Claude Code or Codex use ADA?**
Yes. `ada skill` writes ADA's skill into `.claude/skills/ada/` for Claude Code
and `.agents/skills/ada/` for Codex. It tells the assistant to ask you for the
design and the planned analyses, let ADA compute the numbers, report them with
their calculations, and pass on any refusal. It also installs as a plugin: in
Claude Code, `/plugin marketplace add saineshnakra/automated-data-analyst` then
`/plugin install automated-data-analyst@automated-data-analyst`; in Codex,
`codex plugin marketplace add saineshnakra/automated-data-analyst` then
`codex plugin add automated-data-analyst@automated-data-analyst`.

**How do I check whether the installed skill matches my ADA version?**
Run `ada skill --check`, with the same `--for`, `--user` or `--dir` options
you used to install it. It checks every bundled skill file, reports which
files are missing or different, and writes nothing. It exits 0 when all
match, 1 when any are missing or different, and 2 for read errors, so you can
use it in CI after upgrading ADA. Invalid UTF-8 content counts as different.
`--check` refuses `--print` and `--force`. To update a missing installation,
run `ada skill`; review any local edits before using `ada skill --force` to
replace them.

**Does the skill stop an AI assistant from making up numbers?**
It takes the arithmetic away from the model: ADA computes every number with
pandas, SciPy and statsmodels and prints the calculation beside it, and the
skill tells the assistant to report those numbers and pass on refusals. That
doesn't make a model incapable of a mistake, but every number in its answer
can be checked against ADA's output and rebuilt with `ada verify`.

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
