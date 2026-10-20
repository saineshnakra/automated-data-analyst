# The optional AI layer

**Code:** `ada/ai_insights.py` · **Tests:** `tests/test_ai_insights.py`

The only module in ADA that makes a network call. Everything else works without
an API key. Install its dependencies with the `ai` extra
(`pip install -e ".[ai]"`).

## The two calls

| Call | When it runs | What it receives | What it returns |
|---|---|---|---|
| Query planner | When the rule parser cannot read a question and a key is set | Column names, types, roles, and the question | An `AIQueryPlan` |
| AI summary | When the user presses **Generate AI summary** | The computed brief: schema, evidence, recommendations | An `AINarrative` |

Neither call receives rows. The text of the computed evidence is sent, and an
evidence sentence names the segment it describes ("Northwind is the largest
customer"), so segment labels reach the model. Column names and types do too.
No other cell value is serialized. `tests/test_ai_insights.py` puts
distinctive values in non-segment columns and asserts they appear in neither
payload.

## Payloads

**Planner** (`build_planner_payload`): the question, plus one entry per column
with its name, type (`numeric`, `datetime` or `category`) and role (`primary
measure`, `date`, `primary dimension`, `identifier`, `dimension` or `other`).
`build_query_schema` builds the column list from the DataFrame's structure and
reads no cells.

**Narrative** (`build_ai_payload`): the business context the user typed, the
detected schema, the headline, the computed summary, the evidence cards and the
deterministic recommendations. All of it is already on screen.

## Request settings

| Setting | Value |
|---|---|
| Typed output | `responses.parse` into `AIQueryPlan` or `AINarrative` |
| Prompt storage | `store=False` on every request |
| Identity | `safety_identifier`: a SHA-256 hash of a random session token |
| Output cap | `max_output_tokens`: 500 for plans, 1,400 for the narrative |
| Time | 25 s timeout, 1 retry |
| Trigger | No call on page load; the narrative is cached per payload for the session |

Pydantic also bounds the summary (`AINarrative`): an executive summary of at
most 650 characters, an interpretation (`strategic_read`) of at most 900, one to three actions and
at most three watchouts.

## Plan validation

`_to_query_plan` discards a returned `AIQueryPlan` when:

- it sets `answerable: false` (the model is instructed to do this rather than
  guess)
- it names a measure or dimension column that does not exist
- it asks for a `trend` or `growth` intent with no date column, or with an
  aggregation other than `sum`
- it asks to `aggregate` with no measure
- it asks for a non-count aggregation of a non-numeric column
- it asks to `rank` or `breakdown` with no usable dimension
- its dimension is the measure, a date, an identifier, a column of fractional
  or nearly unique numbers, or a column that is more than 90% unique over at
  least 25 rows
- it sets a year or month with no date column
- any filter value does not match a value in its column

Filter values are matched against the column's values exactly, ignoring case
and surrounding spaces. A value the model made up matches nothing, and the
whole plan is discarded. Pydantic bounds the fields: `top_n` 1 to 50, `year`
1900 to 2100, `month` 1 to 12, at most 4 filters of up to 120 characters each.

Fields the chosen intent's executor would ignore (a dimension on an aggregate,
`top_n` outside `rank`, a grain outside `trend` and `growth`) are cleared.

## Approval

A plan that passes validation is not run straight away. `describe_query_plan`
turns it into one sentence, built per intent so that it describes the
calculation that will run, for example "Rank Product by total Revenue, showing
the 2 highest." The app shows that sentence with **Run calculation** and
**Reject plan** buttons. `execute_approved_ai_plan` runs the plan only when
`approved=True`, using the same `execute_plan` as a rule-parsed plan. The plan
carries `source="ai"`, so its origin stays visible.

Model-generated code is never executed. The model chooses parameters for a
fixed executor; it does not write the query.

## Models

| Preset | Model | Reasoning effort |
|---|---|---|
| Fast · Luna (default) | `gpt-5.6-luna` | low |
| Deep · Terra | `gpt-5.6-terra` | medium |

The query planner always uses `gpt-5.6-luna` with low effort
(`PLANNER_CONFIG`), since it runs inside the chat loop.

## Testing without a key

`generate_ai_narrative` and `plan_query_with_ai` take a `client` argument and
construct an `OpenAI` client only when it is `None`. The expected shape is a
`Protocol` (`_Client`, `_Responses`). Tests pass a fake client that records the
request, and check the payload directly:

```python
payload = build_ai_payload(brief, context="review")
self.assertNotIn("SECRET-NOTE", payload)   # a value from a non-segment column
```

## Changing this module

A change here needs a test showing that no cell values beyond the documented
segment labels enter a payload. The invariants:

1. Payloads are built from the schema and computed evidence only.
2. Output is parsed into a typed model.
3. Plans are validated against the data, and run only after approval.
4. Model output never becomes executable code.
5. Everything except these two features works with the key removed.
