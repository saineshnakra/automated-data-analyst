# Ask ADA

**Code:** `nlq.py` · **Tests:** `tests/test_nlq.py`

Answers plain-English questions about the loaded data, and shows the calculation
behind every answer.

## The idea

ADA does not generate code and run it. Your question becomes a **`QueryPlan`** —
a small, fixed structure — and that plan is executed with pandas locally.

```
"Top 5 regions by revenue"
        ↓ parse_question
QueryPlan(intent="rank", measure="Revenue", dimension="Region",
          top_n=5, aggregation="sum", source="rules")
        ↓ execute_plan
QueryAnswer(answer="...", calculation="...", table=..., chart="bar")
```

The plan is inspectable and the calculation is printed under the answer. There
is no model-written code anywhere in this path.

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
    source: str               # "rules" or the AI planner
```

`source` is what the interface badges. A user can always see whether the rules
or a model produced the plan.

## The six intents

| Intent | Question it answers | Example |
|---|---|---|
| `aggregate` | One number | "Total revenue" |
| `count` | How many | "How many orders?" |
| `rank` | Best or worst N | "Top 5 products by sales" |
| `breakdown` | Split by segment | "Revenue by region" |
| `trend` | Over time | "Monthly revenue trend" |
| `growth` | What changed | "Which region grew fastest?" |

## How the parser decides

Roughly in this order:

1. **Match columns** mentioned in the question, allowing simple plural and
   singular variants
2. **Detect a time filter** — a year, a month name
3. **Detect a grain** — daily, weekly, monthly, quarterly, yearly
4. **Detect value filters** — segment values named in the question, e.g. "in the
   West"
5. **Detect an aggregation word** — total, average, median, highest, lowest, ...
6. Then choose an intent:

| Signal | Result |
|---|---|
| "how many", "count", "number of" | `count`, or `breakdown` with count if a segment is named |
| A growth word and a date column exists | `growth`, ranked if the question asks *which* |
| "top N" / "bottom N", or a superlative with a segment | `rank` |
| A grain, or "over time"/"trend"/"timeline"/"history" | `trend` |
| A segment plus "by"/"per"/"across"/"each" | `breakdown` |
| A measure with an aggregation word | `aggregate` |

If none match, `parse_question` returns `None` — and returning `None` is the
correct behavior, not a failure. It is what triggers the AI fallback or the
suggestion list, rather than ADA guessing.

The parser currently refuses questions containing words outside an allowlist
built from supported query grammar, column names, and low-cardinality dimension
values. Common function words live in `QUERY_STOPWORDS`. This bounds unsupported
input safely, but the natural-language allowlist is temporary: ordinary phrasing
can expose gaps, while schema and intent signals provide a finite basis for a
future refusal check.

## Execution

`execute_plan` applies filters, aggregates, and returns a `QueryAnswer`:

| Field | Contents |
|---|---|
| `answer` | The formatted answer |
| `calculation` | The exact operation, in words |
| `table` | Supporting rows, when useful |
| `chart` | `"bar"`, `"line"`, or nothing |

Trend and growth answers go through the same `TrendSeries` as the dashboard, so
the period adjustments ([Periods](periods.md#the-two-timeline-adjustments)) are
appended to the calculation text. A chat answer and the chart above it can never
quietly disagree about what a month is.

Breakdowns cap at 12 groups; filter candidate values cap at 200.

## When the rules cannot parse

Two fallbacks, in this order:

1. **The AI planner**, if an API key is configured. It receives column names,
   types, and roles — never rows — and returns the same `QueryPlan` shape, which
   goes through the same local executor. Unresolvable plans are refused rather
   than guessed. See [The optional AI layer](ai-layer.md).
2. **Suggested questions**, from `suggested_questions` — starter questions built
   from the detected roles, which the deterministic engine is guaranteed to be
   able to answer.

## Edge cases

| Situation | Behavior |
|---|---|
| Empty question | `None` |
| Question mentions no known column | Falls back to the detected measure |
| Growth question but no date column | Not treated as growth |
| Filter column missing from the data | `ValueError` — plans are validated before running |
| More than 12 breakdown groups | Truncated to 12 |

## Adding a question shape

1. Extend the parser in `parse_question`
2. Add the execution branch in `execute_plan`
3. Extend `QueryPlan` only if genuinely new state is needed
4. Test the parse **and** the execution — a plan that parses but executes wrong
   is worse than one that does not parse at all
5. Make sure the `calculation` string still describes what actually ran

Comparisons ("West vs East"), shares ("what % of revenue is Enterprise"), and
date ranges are on the [roadmap](../../ROADMAP.md).
