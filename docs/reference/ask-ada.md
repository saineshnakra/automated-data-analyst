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
    intent: Intent            # aggregate | count | rank | breakdown | trend | growth | correlation
    aggregation: Aggregation  # sum | mean | median | min | max | count
    measure: str | None
    dimension: str | None
    count_column: str | None
    partner: str | None       # the second column of a correlation
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
| `rank` | Top or bottom N groups, or the top period | "Top 5 products by sales", "Which month had the highest wind?" |
| `breakdown` | The measure per group | "Revenue by region", "Mean signal by timepoint" |
| `trend` | The measure per period | "Monthly revenue trend" |
| `growth` | Latest period against the previous one | "Which region grew fastest?" |
| `correlation` | How two numeric columns move together | "Is flipper length correlated with body mass?" |

## How the parser reads a question

1. Refuse phrasing a plan cannot represent: comparison operators (`>`, `<`,
   `=`, `≥`, `≤`, `≠`), thresholds ("greater than", "at least 10"), dates
   relative to today ("last 30 days", "this year", "year to date", "last
   December"), "or", and questions naming two or more months or years.
2. Refuse the question if any word is unrecognized (see below).
3. Match columns named in the question, including the name with a trailing
   `s` or `es` added or a trailing `s` removed. Longer names are matched
   first, so `Revenue Growth` is not also read as `Revenue`. A column also
   matches on its core name: the name without a trailing unit in brackets
   (`Concentration (mg/L)` matches "concentration") and without trailing unit
   words such as `mm`, `g`, `c` or `pct` (`body_mass_g` matches "body mass").
   `min` and `max` are never treated as units. A core name is not used when
   another column shares it or contains it, so with `Temp (°C)` and
   `Temp (°F)`, "temp" names neither. A core name that is also a query word
   (`Time (h)` gives "time") counts only after a grouping word: "by time".
4. Refuse the question if it uses a word that only the names of numeric
   columns contain and none of those columns was named in full. "Average body
   by island" on a file with `body_mass_g` is refused rather than answered
   with the default measure.
5. Detect a year (1900 to 2099) and a month name.
6. Detect a grain: daily, weekly, monthly, quarterly, yearly.
7. Detect value filters: values of segment columns (with at most 200 distinct
   values, and at least 3 characters long) named in the question, e.g. "West".
8. Pick the grouping column and the measure (see below).
9. Detect an aggregation word (total, average, median, highest, lowest, ...),
   reading the question with the matched column names removed so that
   `Total Sales` does not read as "total".
10. Choose an intent, checking these signals in order:

| Signal | Result |
|---|---|
| "correlation", "correlated", "relate", "related" or "relationship" | `correlation` of the two numeric columns named, in the order named |
| "how many", "count" or "number of", without a growth word | `count`; `breakdown` by count if a grouping column and a grouping word are present |
| A growth word, and a date column exists | `growth`; ranked by segment when one is named, or by the detected segment when the question says "which", "fastest", "slowest" or a superlative |
| "which" or "what" followed by day, week, month, quarter or year, with a superlative, and a date column exists | `rank` of the periods |
| "top N", "bottom N", or a superlative, with a grouping column | `rank` |
| A grain, "over time", "trend", "timeline", "history", "trajectory", or the date column named after "by" | `trend` |
| A grouping column with "by", "per", "across", "breakdown", "split" or "each", or a grouping column with no measure named | `breakdown` |
| A measure with an aggregation word, or a measure named on its own | `aggregate` |

If none match, `parse_question` returns `None`. That triggers the AI planner
(when a key is set) or the suggestion list.

Without an aggregation word, the measure's default is used: a sum for money
and counts, a mean for anything else, or whatever the `aggregation` override
says for the active measure (see
[Concepts](../concepts.md#amounts-rates-and-levels)). A question about another
column uses that column's own default. A trend question whose aggregation word
differs from the default ("average revenue monthly") is refused, because the
trend executor always uses the default. The same holds for ranking periods
("which month had the highest average revenue").

A correlation question is refused unless it names exactly two numeric columns.
It is also refused when it names a column to group by or a grain ("by
species", "per month"), since that asks for one correlation per group. A
column named only to filter on is fine: "correlation between flipper length
and body mass for Adelie".

### Grouping column and measure

The grouping column is usually a segment column. A column named right after a
grouping word ("by", "per", "across", "each") or a ranking word ("which",
"top N", "bottom N") can also group the rows:

- **A numeric column** groups the rows when no segment is named and it has at
  most 50 distinct values (`GROUPING_LIMIT`). In "mean signal by timepoint"
  the grouping is `timepoint` and the measure is `signal`. With no other
  numeric column named, the detected measure is used. After a ranking word the
  column is what gets ranked and a later "by" names the measure: in "top 3
  cylinders by mpg" the measure is `mpg`.
- **A text column** that schema detection left out of the segments because of
  its name, such as a `time` column holding "1 min" and "15 min", groups the
  rows when it has 2 to 50 distinct values.

These are refused rather than answered without the grouping:

- a numeric column with more than 50 distinct values named as the grouping
  while another numeric column is named ("average mpg by weight")
- two groupings ("mean signal by timepoint by region")
- a segment and the date column ("revenue by region by order date")

After a segment, the column named after "by" is the measure: "top 5 products
by revenue". With no grouping column, the measure is the numeric column with
the longest name.

The default aggregation is taken from the final measure, so a measure combines
the same way in every intent: "which origin has the highest mpg" and "mpg by
origin" both use `mpg`'s own aggregation.

### Counting

"How many" counts distinct values of an identifier or segment column it names
("how many orders" counts `Order ID`). A numeric column is counted the same way
when it holds whole numbers, its name is singular and the question names it in
the plural: "how many subjects" counts the distinct values of a `subject`
column numbered 1 to 20. A plural name such as `Units` holds a quantity, so
"how many units" returns the row count.

### Unrecognized words

Each word in the question must be one of:

- a word from a column name, or a value of a segment column
- part of the query grammar: aggregation, ranking, growth, correlation, time
  and grain words
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
| `chart` | `"bar"`, `"line"`, `"scatter"` or `None` |

Trend, growth and period-ranking answers use the same `TrendSeries` as the
dashboard, so the timeline notes ([Periods](periods.md#timeline-adjustments))
are appended to the calculation text.

Other behaviour:

- A breakdown shows at most 12 groups, largest first; a rank shows `top_n`.
- A breakdown by a numeric column lists its groups in their own order
  (timepoint 0, 1, 2, ...) rather than by value, up to 50 of them, and is drawn
  as a line. A ranking ("which timepoint has the highest signal") still sorts
  by value.
- Ranking periods ("which month had the highest wind") builds the trend at
  that grain with `build_trend` and names the highest or lowest period with its
  value. Periods left empty for a rate or level are skipped. "Which month had
  the most rows" ranks row counts.
- A correlation reports Pearson r with its 95% interval (Fisher transform), the
  number of complete pairs and the Spearman rank correlation, in the same
  wording as the brief's relationship card (see
  [Evidence](evidence.md#correlation)). The pair was named by the reader, so
  nothing is said about picking it from several; the answer ends by saying a
  correlation alone is not evidence of cause. Fewer than 12 complete pairs
  (`MIN_CORRELATION_PAIRS`) gives an answer saying so instead of an r. The
  table holds the pairs for a scatter, sampled down to 2,000 when there are
  more.
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

`rephrase_hint` adds a lead when it has one, shown above the suggestions and
appended to the refusal message:

| Refused question | Hint |
|---|---|
| Names part of a numeric column's name | "Did you mean body_mass_g?" (every column that word could mean) |
| Correlates a text column | "A correlation needs two numeric columns, and species is not numeric." |
| Groups by a numeric column with too many values | "weight has 351 distinct values, too many to group by (the limit is 50)." |

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
| A partial column name ("average body by island") | `None`, with a "Did you mean" hint |
| "Which year ..." with no date column | Not a period ranking; a numeric `year` column can still group if it has at most 50 values |
| Fewer than two periods with values | Answer says there is nothing to compare |
| A correlation over a column that does not vary | Answer says the correlation is undefined |

## Adding a question shape

1. Extend `parse_question`.
2. Add the execution branch in `execute_plan`.
3. Extend `QueryPlan` only if the shape needs new state.
4. Test both parsing and execution.
5. Check that the `calculation` string describes what ran, and that
   `describe_query_plan` in `ada/ai_insights.py` still matches.

Comparisons ("West vs East"), shares ("what % of revenue is Enterprise") and
date ranges are on the [roadmap](../../ROADMAP.md).
