# The optional AI layer

**Code:** `ai_insights.py` · **Tests:** `tests/test_ai_insights.py`

The only file in ADA allowed to make a network call. Everything here is
optional; ADA is a complete analyst with no API key.

## The two calls

| Call | When it runs | What it receives | What it returns |
|---|---|---|---|
| **Query planner** | Only when the rule parser cannot read a question | Column names, types, roles, and the question | A typed `AIQueryPlan` |
| **Strategic read** | Only when the user presses the button | The computed brief — schema, evidence, recommendations | A typed `AINarrative` |

Neither receives uploaded rows. Not a sample, not a preview, not the first five
rows. What does travel is the text of the computed evidence, and an evidence
sentence names the segment it is about — "Northwind is the largest customer" —
so segment labels reach the model. Column names and types do too. No other cell
value is serialized. `tests/test_ai_insights.py` pins this by putting a
distinctive value in a non-segment column and asserting it never appears in
either payload.

## What is actually sent

**Planner payload** (`build_planner_payload`): the question, plus one entry per
column giving its name, type (`numeric` / `datetime` / `category`), and role
(`primary measure`, `date`, `primary dimension`, `identifier`, `dimension`,
`other`). `build_query_schema` builds this from the DataFrame's structure
without reading a single cell.

**Narrative payload** (`build_ai_payload`): the business context the user typed,
the detected schema, the headline, the computed summary, the evidence cards, and
the deterministic recommendations. All of it already computed and already shown
on screen.

## The safety rails

| Rail | How |
|---|---|
| Typed output | Responses parsed into Pydantic models — `AIQueryPlan`, `AINarrative` |
| No stored prompts | `store=False` on every request |
| Anonymous identity | A hashed session identifier for abuse controls |
| Bounded output | `max_output_tokens` — 500 for plans, 1,400 for the narrative |
| Bounded time | 25s timeout, 1 retry |
| Button-triggered | Nothing fires on page load; results are cached per payload |

## Plans are validated, not trusted

A returned `AIQueryPlan` is rejected outright when:

- it sets `answerable: false` — the model is expected to refuse rather than guess
- it names a column that does not exist
- it asks for a trend or growth intent with no date column
- it asks to aggregate with no measure
- it asks to rank or break down with no available dimension
- any filter value does not match a real value in the column

Filter values are resolved by exact case-insensitive match against actual values
in the data. A value the model invented resolves to nothing, and the whole plan
is refused. Pydantic also bounds the fields: `top_n` 1–50, `year` 1900–2100,
`month` 1–12, at most 4 filters.

A surviving plan is marked `source="ai"`, executed by **the same local
`execute_plan`** as a rule-parsed one, and badged in the interface.

**Model-generated code is never executed.** The model chooses parameters for a
fixed executor; it does not write the query.

## Models

| Preset | Model | Reasoning |
|---|---|---|
| Fast · Luna (default) | `gpt-5.6-luna` | low |
| Deep · Terra | `gpt-5.6-terra` | medium |

The query planner always uses the fast preset — plan selection does not need
deep reasoning, and it sits in an interactive loop.

## Testing without a key

`generate_ai_narrative` and `plan_query_with_ai` take a `client` argument and
fall back to constructing a real `OpenAI` client only when it is `None`. The
expected shape is declared as a `Protocol`.

Tests pass a fake client and assert on the exact payload. That is how the
privacy contract is enforced by the test suite rather than by good intentions:

```python
def test_payload_contains_no_row_values():
    fake = FakeClient()
    generate_ai_narrative(brief, api_key="test", config=..., client=fake, ...)
    assert "some-customer-name" not in fake.last_input
```

## Changing this

Any change here needs a privacy test proving no cell values enter the payload.
The invariants are:

1. Payloads are built from schema and computed evidence only
2. Output is parsed into a typed schema
3. Plans are validated against the real data before they run
4. Model output never becomes executable code
5. Everything still works with the key removed
