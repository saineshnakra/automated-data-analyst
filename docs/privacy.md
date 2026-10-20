# Privacy

What ADA does with your data. To report a vulnerability, see
[SECURITY.md](../SECURITY.md).

## Summary

**Running ADA yourself.** Nothing leaves the machine. Cleaning, schema
detection, charts, evidence cards and Ask ADA answers are computed in-process
with pandas. Without an API key, ADA makes no network call.

**The hosted demo.** Your upload is sent to a Streamlit server, held in memory
for the session, and not written to a database. Run ADA locally if your data
should not leave your machine.

**With an API key.** Two optional features become available. Both send the
column schema and already-computed evidence, and neither sends your rows. An
evidence sentence names the segment it describes, so a segment label such as a
customer or product name can be included in it. No other cell value is sent.

## What always runs locally

| Step | Where it runs |
|---|---|
| File parsing | Local |
| Cleaning and type inference | Local |
| Schema detection | Local |
| Charts and KPIs | Local |
| Anomaly detection and forecasting | Local |
| Evidence and recommendations | Local |
| Ask ADA parsing and execution | Local |

A question the rule parser can plan is answered with no model call.

## What the optional AI features receive

Both need a key, and neither runs on page load.

**Query planner.** Runs only when the rule parser cannot read a question. It
receives:

- the question you typed
- for each column, its name, its type (`numeric`, `datetime` or `category`)
  and its detected role

The plan it returns is described to you in one sentence and runs locally only
after you approve it.

**AI summary.** Runs only when you press **Generate AI summary**. It receives:

- the business context you typed, if any
- the detected schema
- the computed headline and summary
- the evidence cards, with their calculations
- the deterministic recommendations

Everything in the second list is already on screen before the call.

## What is never sent

- Your rows, or any sample or preview of them
- Cell values, except the segment labels that evidence sentences name
- Your file, its name or its bytes
- The question you typed, unless the rule parser cannot read it and a key is
  set; then the question goes to the planner

The payload builders (`build_query_schema`, `build_planner_payload` and
`build_ai_payload` in `ada/ai_insights.py`) work from DataFrame structure and
computed results. `tests/test_ai_insights.py` puts distinctive values in
non-segment columns and asserts they appear in neither payload.

## Request settings

| Setting | Value |
|---|---|
| Prompt storage | Disabled (`store=False`) |
| Identity | A hashed, anonymous session identifier (`safety_identifier`) |
| Output cap | 500 tokens for plans, 1,400 for the narrative |
| Timeout | 25 seconds, 1 retry |
| Trigger | A question the rules cannot parse, or a button press |
| Caching | The narrative is cached per evidence payload for the session |

## Model output is validated

Model output is parsed into Pydantic models. A query plan is then checked
against the data: unknown columns are rejected, filter values must match
values in the column, and plans that cannot run as stated are refused. A plan
that fails a check is discarded, not repaired.

Model-generated code is never executed. The model chooses parameters for a
fixed local executor. See [The optional AI layer](reference/ai-layer.md) for
the full list of checks.

## Where the API key is kept

| Method | Scope |
|---|---|
| Sidebar field | Sent from the browser to the server running ADA and held in that session's memory; not written to disk. On the hosted demo, that server is Streamlit's. |
| `OPENAI_API_KEY` environment variable | The deployment |
| `.streamlit/secrets.toml` | The deployment. The file is gitignored; do not commit it. |

On a public deployment, let visitors supply their own key. A key paid for by
the deployment owner needs authentication and spending controls.

## Self-hosting

ADA has no database and writes nothing to disk.

Parsed uploads are kept in an in-memory cache (at most eight files, each for up
to an hour) so that a page rerun does not parse the same file again. The cache
belongs to the server process, not to a browser session, and a restart clears
it.

Self-hosted, the deterministic path involves no third party, and that cache is
on your own machine.
