# Ask ADA

**Code:** `ada/nlq.py` · **Tests:** `tests/test_nlq.py`

Answers plain-English questions about the loaded data and shows the
calculation behind each answer.

## Overview

The question becomes a `QueryPlan`, a small fixed structure, and pandas
executes the plan locally. No code is generated.

```
"Top 5 regions by revenue"
        ↓ parse_question
QueryPlan(intent="rank", measure="Revenue", dimension="Region",
          top_n=5, aggregation="sum", source="rules")
        ↓ execute_plan
QueryAnswer(answer="...", calculation="...", table=..., chart="bar")
```

## The plan

```python
@dataclass(frozen=True)
class QueryPlan:
    intent: Intent            # aggregate | count | rank | breakdown | trend | growth
    aggregation: Aggregation  # sum | mean | median | min | max | count
    measure: str | None
    dimension: str | None
    count_column: str | None
    top_n: int | None
    ascending: bool
    filters: tuple[ValueFilter, ...]
    year: int | None
    month: int | None
    grain: str | None         # D | W | M | Q | Y
    source: str               # "rules", or "ai" for the optional planner
```

## Intents

| Intent | Answers | Example |
|---|---|---|
| `aggregate` | One number | "Total revenue" |
| `count` | How many rows, or distinct values | "How many orders?" |
| `rank` | Top or bottom N groups | "Top 5 products by sales" |
| `breakdown` | The measure per group | "Revenue by region" |
| `trend` | The measure per period | "Monthly revenue trend" |
| `growth` | Latest period against the previous one | "Which region grew fastest?" |

## How the parser reads a question

1. Refuse phrasing a plan cannot represent: comparison operators (`>`, `<`,
   `=`, `≥`, `≤`, `≠`), thresholds ("greater than", "at least 10"), dates
   relative to today ("last 30 days", "this year", "year to date", "last
   December"), "or", and questions naming two or more months or years.
2. Match columns named in the question, including the name with a trailing
   `s` or `es` added or a trailing `s` removed. The longest matching name wins.
3. Refuse the question if any word is unrecognized (see below).
4. Detect a year (1900 to 2099) and a month name.
5. Detect a grain: daily, weekly, monthly, quarterly, yearly.
6. Detect value filters: values of segment columns (with at most 200 distinct
   values, and at least 3 characters long) named in the question, e.g. "West".
7. Detect an aggregation word (total, average, median, highest, lowest, ...),
   reading the question with the matched column names removed so that
   `Total Sales` does not read as "total".
8. Choose an intent, checking these signals in order:

| Signal | Result |
|---|---|
| "how many", "count" or "number of", without a growth word | `count`; `breakdown` by count if a segment and a grouping word are present |
| A growth word, and a date column exists | `growth`; ranked by segment when one is named, or by the detected segment when the question says "which", "fastest", "slowest" or a superlative |
| "top N", "bottom N", or a superlative, with a segment | `rank` |
| A grain, or "over time", "trend", "timeline", "history", "trajectory" | `trend` |
| A segment with "by", "per", "across", "breakdown", "split" or "each", or a segment with no measure named | `breakdown` |
| A measure with an aggregation word, or a measure named on its own | `aggregate` |

If none match, `parse_question` returns `None`. That triggers the AI planner
(when a key is set) or the suggestion list.

Without an aggregation word, the measure's default is used: a sum for money
and counts, a mean for anything else, or whatever the `aggregation` override
says for the active measure (see
[Concepts](../concepts.md#amounts-rates-and-levels)). A question about another
column uses that column's own default. A trend question whose aggregation word
differs from the default ("average revenue monthly") is refused, because the
trend executor always uses the default.

### Unrecognized words

Each word in the question must be one of:

- a word from a column name, or a value of a segment column
- part of the query grammar: aggregation, ranking, growth, time and grain
  words
- one of `QUERY_STOPWORDS`, a fixed list of function words ("the", "by",
  "how", "give", "were", ...)

Any other word (a column that does not exist, a typo, an unrelated request)
makes the parser return `None` instead of guessing. "Tell me a joke" and
"total sales by region" on a file with no `sales` column both return `None`.

The stopword list is maintained by hand, so a question using an unlisted
function word is refused until that word is added.

## Execution

`execute_plan` applies the filters and the year or month, then computes the
answer. It returns a `QueryAnswer`:

| Field | Contents |
|---|---|
| `answer` | The formatted answer |
| `calculation` | The operation that ran, in words |
| `table` | Supporting rows, when useful |
| `chart` | `"bar"`, `"line"` or `None` |

Trend and growth answers use the same `TrendSeries` as the dashboard, so the
timeline notes ([Periods](periods.md#timeline-adjustments)) are appended to the
calculation text.

Other behaviour:

- A breakdown shows at most 12 groups; a rank shows `top_n`.
- Sum and count groupings get a `Share %` column when every group has the same
  sign.
- Changes are given in the measure's own terms: percent for an amount or
  count, percentage points for a rate, the plain difference for a level.
- Growth by segment ranks amounts by percent change. It leaves out segments
  that started from zero and names them separately, since a growth rate from
  zero is undefined. For a rate or level, segments are ranked by the change in
  their average, and only segments with values in both periods are included.

## When the rules cannot parse

1. **The AI planner**, if an API key is set. It receives column names, types
   and roles, never rows, and returns the same `QueryPlan` shape. The app
   describes the plan in one sentence and runs it only after the user approves.
   See [The optional AI layer](ai-layer.md).
2. **Suggested questions** from `suggested_questions`: up to four starter
   questions built from the detected roles, each answerable by the rules.

From Python, `Analysis.ask()` returns a `Refusal` with code
`unsupported_phrasing` or `unrepresentable_question` instead of `None`.

## Edge cases

| Situation | Behavior |
|---|---|
| Empty question | `None` |
| No column named, but an aggregation word ("total") | Uses the detected measure |
| Growth question with no date column | Not treated as growth |
| Year or month asked, no date column | Answer says the scope is unavailable |
| No rows match the filters | Answer says there is nothing to calculate |
| Plan names a column missing from the data | `execute_plan` raises `ValueError`; `answer_question` returns `None` |
| "top 0" | `None` |
| More than 12 breakdown groups | First 12 shown |

## Adding a question shape

1. Extend `parse_question`.
2. Add the execution branch in `execute_plan`.
3. Extend `QueryPlan` only if the shape needs new state.
4. Test both parsing and execution.
5. Check that the `calculation` string describes what ran, and that
   `describe_query_plan` in `ada/ai_insights.py` still matches.

Comparisons ("West vs East"), shares ("what % of revenue is Enterprise") and
date ranges are on the [roadmap](../../ROADMAP.md).
