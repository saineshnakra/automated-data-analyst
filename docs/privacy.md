# Privacy

This page describes what ADA does with your data. For reporting a vulnerability,
see [SECURITY.md](../SECURITY.md).

## The short version

**Where "local" means depends on where ADA is running.** Run it on your own
machine and nothing leaves it: cleaning, schema detection, every chart, every
evidence card and every Ask ADA answer are computed in-process with pandas, and
without an API key ADA makes no network call at all.

**On the hosted demo, your upload reaches a Streamlit server.** That is what
uploading to a website is. The parsed file is held in that server process's
memory — in a bounded cache of at most eight files for up to an hour, which
belongs to the process rather than to your browser session — and is never
written to disk or to a database. Run ADA locally if that matters for your
data.

**With an API key, two optional calls become available.** The query planner
receives only the question you typed and the column schema — names, types and
detected roles — and nothing that was computed from your values. The strategic
read receives the schema plus the already-computed evidence. **Neither sends
your rows.** An evidence sentence names the segment it describes, so in the
strategic read a segment label — a customer name, a product name — can travel
inside it. No other cell value does, in either call.

## What stays local, always

| | Where it runs |
|---|---|
| File parsing | Local |
| Cleaning and type inference | Local |
| Schema detection | Local |
| Every chart and every KPI | Local |
| Anomaly detection and forecasting | Local |
| Evidence and recommendations | Local |
| Ask ADA — parsing **and** execution | Local |

Every Ask ADA answer the rule parser can plan is computed with no model call at
all.

## What the optional AI calls receive

Both are opt-in, button-triggered, and require a key.

**Query planner** — runs only when the rule parser cannot read a question:

- the question you typed
- for each column: its name, its type (`numeric` / `datetime` / `category`), and
  its detected role

**Strategic read** — runs only when you press the button:

- the business context you typed, if any
- the detected schema
- the computed headline and summary
- the evidence cards, with their calculations
- the deterministic recommendations

Everything in the second list is already on your screen before the call is made.

## What is never sent

- Uploaded rows
- Your rows — not a sample, not a preview, not the first five
- Cell values, with one exception stated above: the segment labels an
  evidence sentence names, such as a customer or product name
- The question you type stays local unless the rule parser cannot read it and
  you have supplied a key; then the question itself goes to the planner
- Your file, its name, or its bytes

The payload builders (`build_query_schema`, `build_ai_payload` in
`ai_insights.py`) construct their output from DataFrame *structure* and computed
results. `tests/test_ai_insights.py` asserts on the exact payload, so the
guarantee is enforced by the test suite.

## Request settings

| Setting | Value |
|---|---|
| Prompt storage | Disabled (`store=False`) |
| Identity | A hashed anonymous session identifier, for abuse controls |
| Output cap | 500 tokens for plans, 1,400 for the narrative |
| Timeout | 25 seconds, 1 retry |
| Trigger | Button press only — nothing fires on page load |
| Caching | Per evidence payload, to avoid accidental repeat spend |

## Model output is not trusted

Model output is parsed into a typed Pydantic schema. A query plan is then
validated against your actual data: unknown columns are rejected, filter values
must match real values, and impossible intents are refused. Anything that fails
is dropped rather than repaired.

**Model-generated code is never executed.** The model picks parameters for a
fixed local executor.

See [The optional AI layer](reference/ai-layer.md) for the full validation list.

## Where your key lives

| Method | Scope |
|---|---|
| Sidebar field | Typed in the browser, posted to the server running ADA, and held in that session's memory — never written to disk. On the hosted demo, that server is Streamlit's |
| `OPENAI_API_KEY` env var | The deployment |
| `.streamlit/secrets.toml` | The deployment — gitignored, never commit it |

On a public deployment, prefer letting visitors bring their own key. An
owner-funded key on a public app needs authentication and spending controls.

## Self-hosting

ADA is a standard Streamlit app with no database. Nothing is written to disk.

Parsed uploads are held in a bounded in-memory cache — at most eight files, for
up to an hour — so that re-running the page does not re-parse the same file.
That cache belongs to the server process, not to your browser session, and it
is why "nothing is persisted" is worth stating precisely rather than loosely.
Restarting the app clears it.

Running ADA yourself means the deterministic path involves no third party at
all, and that cache lives on your own machine.
