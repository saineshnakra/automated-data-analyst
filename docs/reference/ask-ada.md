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

Every part of a question has to reach the plan. A column, a value, a grouping,
a condition, a "top N" or a word such as "which column", "of the year",
"within" or "distinct" the plan would leave out makes the parser refuse, and
`rephrase_hint` says which part and why. ADA never answers a simpler question
under the original wording.

## The plan

```python
@dataclass(frozen=True)
class QueryPlan:
    intent: Intent            # aggregate | count | rank | breakdown | trend | growth |
                              # correlation | compare | correlates | describe | columns |
                              # crosstab | missing | duplicates | outliers
    aggregation: Aggregation  # sum | mean | median | min | max | count | std | var | iqr |
                              # quantile | mode | share
    measure: str | None
    dimension: str | None
    count_column: str | None  # counts distinct values of this column instead of rows
    partner: str | None       # the second column of a correlation
    top_n: int | None
    ascending: bool
    filters: tuple[ValueFilter, ...]   # rows whose value is one of the listed values
    year: int | None
    month: int | None
    grain: str | None         # s | min | h | D | W | M | Q | Y
    source: str               # "rules", or "ai" for the optional planner
    conditions: tuple[Condition, ...]  # "pH below 5.5", "OBJECT is 0", "eCO2 missing"
    time_part: str | None     # "hour", "weekday", "month" or "quarter" of the year, or "year",
                              # of the date column in `dimension`
    notes: tuple[str, ...]    # how the question was read, said in the answer
    second_dimension: str | None  # the second grouping of a cross-table
    bins: int | None          # a continuous `dimension` split into this many equal-sized groups
    columns: tuple[str, ...]  # the columns a cross-column answer ranks or compares
    quantile: float | None    # 0.9 for "90th percentile"
```

A `Condition` is a column, an operator (`=`, `<`, `<=`, `>`, `>=` or
`missing`) and a number.

## Intents

| Intent | Answers | Example |
|---|---|---|
| `aggregate` | One number | "Total revenue", "maximum strength" |
| `count` | How many rows, or distinct values; with conditions, also their share | "How many orders?", "how many rows have negative yield" |
| `rank` | Top or bottom N groups, the row with the highest value, or the top period | "Top 5 products by sales", "Which taxonomy has the highest A101?" |
| `breakdown` | The measure per group, or per hour or weekday | "Revenue by region", "average PM2.5 by hour" |
| `trend` | The measure, or a row count, per period | "Monthly revenue trend", "how many readings per month" |
| `growth` | Latest period against the previous one | "Which region grew fastest?" |
| `correlation` | How two numeric columns move together | "Is flipper length correlated with body mass?" |
| `missing` | Missing values per column, or in one column | "Which columns have missing values?" |
| `compare` | Whether a reading differs between groups, with a test and an effect size | "Does pH differ by site?", "Is troponin higher in positive than negative?" |
| `correlates` | Every numeric column ranked by how strongly it relates to one | "What correlates with strength?", "what predicts yield?" |
| `describe` | n, mean, SD, median, quartiles, range and skew, overall or per group | "Distribution of ELOVL2_6", "describe pH by site" |
| `columns` | The numeric columns themselves, ranked by one aggregation | "Which column has the highest average?", "which gene has the highest mean?" |
| `crosstab` | Row counts, or a measure, for each pair of groups of two groupings | "Count by crop and year", "mean risk by antibiotics and probiotics" |
| `outliers` | Values outside 1.5×IQR and the values the data checks flag | "Outliers in HGB", "are there impossible values?" |
| `duplicates` | Rows that repeat an earlier row, or values found on several rows | "How many duplicate rows?" |

## Comparing groups, and what correlates with a column

Before the steps below, `_comparison_plan` reads these question shapes, with
their own small grammar (difference and direction words, "than", nouns such
as "patients" or "variables") on top of the usual one:

| Question | Plan |
|---|---|
| "does X differ by Y", "is X different between Y", "does X vary across Y", "compare X across Y", "is the difference in X between Y groups significant" | `compare` X across every group of Y |
| "is X higher in A than B", "is X lower for A than B" | `compare` X between groups A and B of one column, kept as a filter in the order named; the direction asked is stored as the plan's `aggregation` (`max` for higher, `min` for lower) |
| "is X higher in condition 2 than condition 1" | The same, for the numbered groups of a numeric column |
| "is X higher with Y", "is X lower without Y" | The same, for the rows where a 0/1 flag Y is set against those where it is not |
| "does X differ between high and low Y" | `compare` X between the halves of a continuous Y split at its median (`bins=2`) |
| "what correlates with X", "what is most related to X", "which variables are most correlated with X" | `correlates` of X |
| "what predicts X", "what drives X", "which factors affect X the most", "which marker best separates X" | `correlates` of X, from `_predictors_plan` |
| "does Y affect X", "what is the effect of Y on X", "the impact of Y on X" | X related to Y (see below) |
| "is there a trend in X with Y", "does X change with Y" | X related to Y, for a Y that is not the date |
| "which genes differ between AML and ALL" | `compare` of every column the word names, across one grouping |

`_predictors_plan` is the one place that decides how "what predicts X" is
answered. For now it ranks each numeric column against X one at a time, as
"what correlates with X" does, and says so; a joint model of the predictors
is called from there once one is available.

Two named columns are related the way their kinds allow (`_relation_plan`):
two quantities are correlated; a quantity and a set of category codes are
compared across the codes' groups; two sets of codes are cross-tabulated with
a chi-square test. The answer says when a correlation question became a
comparison: "Condition holds category codes, so CorrectTotal is compared
across its groups rather than correlated with it." Codes are text, 0/1 flags,
the numeric codes ADA declared (`roles.factors`) and whole numbers under an ID
or code name. A declared code whose name carries a unit ("Location_m", "Age
(day)") is a quantity set at a few levels and still correlates. This also
applies to "correlation between X and Y" read by the steps below.

The grouping column is the one named after "by", "across", "between",
"among" or "in" (a segment, or a numeric column with 2 to 50 values), or the
column whose values the question names. Values of other columns filter the
rows. Without a numeric column named, the detected measure is compared.
"compare X across Y" for a summed measure such as revenue asks for each
group's total, so it becomes a `breakdown`; words of difference still ask for
the test. A question naming two numeric columns ("is x related to y") stays a
`correlation`, and anything this grammar can't read goes on to the steps
below.

A `compare` answer is the brief's group comparison (see
[Comparing groups](compare.md)): the n, mean and median per group, Welch's
test with a Mann-Whitney check for two groups or Welch's ANOVA with a
Kruskal-Wallis check for more, an effect size, and "no clear difference"
when the test can't tell the groups apart. A skewed or capped reading is
compared by medians and ranks, and a repeating subject column counts each
subject once per group. A directed question gets its answer first: "Yes.",
"No, the other way round." or "Not clearly." The table has one row per group
and is drawn as a bar chart.

A `correlates` answer ranks every numeric reading against X as the brief's
relationship card does, by the larger of |Spearman r| and |Pearson r| without
the most influential record, and leaves out columns built from X or from the
same items, and copies of X. It names the columns it left out and why ("Left
out of the ranking for Even1: Even2 (the same measurement under another name
or unit)"), so the column that moves most with X is never dropped silently,
and counts only the columns it ranked. It states the strongest with the usual
correlation wording and checks, or says nothing reaches |r| = 0.3. The table
lists each column's Pearson r, 95% interval, Spearman r and n, drawn as a bar
chart.

Comparing every column a word names ("which genes differ") runs the brief's
difference scan (`compare.scan_differences`), with p-values Holm-adjusted over
every test, and names the columns that still differ clearly, or the largest
gap with its adjusted p-value when none does.

## How the parser reads a question

### 1. Phrasings no plan can hold

Refused before parsing: dates relative to today ("last 30 days", "this year",
"year to date", "last December"), "or", `!=` and `≠`, and questions naming two
or more months or years.

### 2. Every word is claimed

The question is normalized: lowercase, accents dropped ("orgánica" reads as
"organica"), a decimal point between digits kept ("5.5", "PM2.5"), and
comparison symbols read as words (`<` as "below", `>=` as "at least"). Then
each word is claimed, in this order, by:

1. **A column's full name**, including the name with a trailing `s` or `es`
   added or a trailing `s` removed, and the column's core name: without a
   trailing unit in brackets (`Concentration (mg/L)` matches "concentration")
   and without trailing unit words such as `mm`, `g`, `c` or `pct`
   (`body_mass_g` matches "body mass"). Longer names claim first, so
   `Revenue Growth` is not also read as `Revenue`. A core name shared with, or
   found inside, another column's name is not used, so with `Temp (°C)` and
   `Temp (°F)` "temp" names neither. A core name that is also a query word
   (`Time (h)` gives "time") counts only after a grouping word: "by time".
2. **A column name that starts with an aggregation word.** With columns
   `Wind Speed` and `Maximum Wind Speed`, "maximum wind speed" is the maximum
   of Wind Speed unless the question has another aggregation word ("average
   maximum wind speed"); the answer says how it was read. A column called
   just `MAX`, `mean` or `std` is the column when the question has another
   aggregation word, names it after "of", a grouping or a ranking word, types
   it exactly as the column is spelled and that spelling is not plain lower
   case ("MAX trend"), asks for a correlation ("correlation between mean and
   std"), or sets a condition on it ("std above 2"), or when "trend" follows.
   Otherwise it is the aggregation word.
3. **A part of the date**: "by hour", "by hour of day", "by weekday", "by day
   of week", "which hour", "which month of the year", "by quarter of the
   year", "by calendar month". The hour needs a date column that holds times
   of day. Months and quarters of the year pool every year.
4. **Conditions** written around a named column (see below).
5. **Values** of text columns, written in full (any case).
6. **Summary words**: "standard deviation", "interquartile range",
   "90th percentile", "upper quartile", "most common", "distribution of",
   "describe", "outliers", "unusual", "impossible", "duplicate", "distinct",
   "unique", and "which column" or "which variable", which rank the numeric
   readings themselves. "std", "sd", "variance" and "iqr" are aggregation
   words.
7. **Partial column names** (see below).
8. **Values** of long labels, by whole words: "Tartrazine" names "Tartrazine,
   FD&C Yellow 5 (US), E102 (EU)".
9. **Unit conditions**: "at 28 days" is `Age of testing (day)` = 28 when one
   numeric column has that unit in its name.
10. **Numbers**: the N of "top N" and "bottom N", a year (1900 to 2099), and a
    number that is a column's whole name ("correlation between 1001 and 1").
    In a table with no date column, a number naming a column is the column.
11. **An unnamed row label**: when the first column has no header
    (`Unnamed: 0`) and the notes read it as the row label, a noun that names
    nothing else right after "which", "top N" or a grouping word names it:
    "which probes have the highest mean". The answer says how it was read.
12. **Nouns for the rows**: a word from `ENTITY_NOUNS` ("patients", "birds",
    "readings") or a plural word right after "how many", "number of" or
    "share of".
13. **"missing"**, "null", "blank" or "empty" not attached to a column: a
    question about missing values.
14. **Column words** ("column", "variable") right after a column's name ("the
    pH column") or in a question about missing values, and **filter cues**
    ("within", "among") right before a filter or condition ("within CLASS 0").

The rest must be query grammar: aggregation, ranking, growth, correlation,
time, grain and share words, and `QUERY_STOPWORDS`, a fixed list of function
words ("the", "by", "how", "where", "tested", ...). Any other word refuses the
question, and each unbroken run of such words is quoted as typed: `ADA found
no column or value called "A1_1" in this table`. Words of comparison or
research ("differ", "affect") that no plan used get a reason of their own: they
ask to compare groups or relate columns, and ADA could not tell which columns.
When the row limit left rows out, the reason says so, since the value may be
in the rows ADA did not read.

Intent, grain and month words are read with every column name blanked, so a
word inside a name is never grammar: `average_weekly_activity` asks for no
weekly view and `Age of testing (day)` for no daily one.

### Conditions

| Written | Reads as |
|---|---|
| "pH below 5.5", "pH < 5.5", "yield under 0" | `pH < 5.5` |
| "at least 10", ">= 10", "at most", "more than", "greater than", "over", "above" | `>=`, `<=`, `>` |
| "pH between 5 and 6" | `pH >= 5` and `pH <= 6` |
| "negative yield", "yield is negative", "positive", "zero" | `< 0`, `> 0`, `= 0` |
| "when OBJECT is 0", "OBJECT = 0", "OBJECT equals 0" | `OBJECT = 0` |
| "for Status 2", "cars with 6 cyl" | `Status = 2`, for a column of codes holding that value |
| "how many have hypertension", "with diabetes", "are healthy", "received antibiotics", "without diabetes" | a 0/1 flag `= 1`, or `= 0` after "without" |
| "how many AQC1cm values are 0", "how many zeros in AQC1cm", "negative values of X" | `AQC1cm = 0`, `X < 0` |
| "missing eCO2", "eCO2 is missing" | `eCO2 missing` |
| "where Condition is Open", "Detector is ON", "Region West" | the value of that text column |

A column of codes is one listed among the segments (a status, a 0/1 flag) or
one with whole numbers and at most 50 values. "with diabetes" after a
correlation word ("correlated with diabetes") is a column to correlate, not a
condition. A threshold that names no column ("over 50") leaves its number
unclaimed, so the question is refused. A question made only of conditions
("revenue > 200", "West") counts the rows that meet them.

### Values

Values of text columns with up to 10,000 distinct values
(`MAX_FILTER_CANDIDATES`) can filter. A value written in full matches in any
case. A value of three letters or fewer must be written in its own case
("pH", "ON"), and a single letter also needs a cue before it ("for A"), so
"on" and a sentence-starting "A" stay words. A number held as text ("47") is
matched only after its column ("ID 47"). A value made only of query words
("Top", "Total") needs a cue such as "for", "in" or "is" before it, so "top 2
contours" ranks while "average Ca for Top" filters.

A long label matches by whole words when only one value holds them, or only
one starts with them. Words several labels share ("Yellow") are refused with
the candidates. A value found in several columns is refused with the columns,
unless it picks the same rows in each (a label and its definition), when the
first column is used.

### Partial column names

Words of a column's name that no other column shares name that column, and the
answer says so: "strength" names `Concrete compressive strength (MPa)`, "pH"
names `pH agua:suelo 2,5:1,0`, "sampling point" names
`sample.samplingPoint.label` if no other column has those words (camelCase is
split). A word that starts with a column's whole name word names it when the
name word is at least three letters and a third of the word: "conductivity"
names `Conduc`, "stations" names `STA`, "healthy" names `Crop_Health_Label`.

When several columns fit, the one suited to the word's place is used if it is
the only one: the identifier after "how many", a groupable column after "by"
or "which", a numeric column elsewhere. After "how many" the column named for
the noun is used (`_counted_column`): "how many drugs" counts `DRUG_NAME`
rather than the rows or the serial `DRUG_ID`, "how many crops" counts
`Current Crop` but not `Crop Rotation System`, and when `Plot Code` and `Plot
Name` count different numbers the code is used and the answer gives the
other's count. After "by" or "which", columns that pick the same groups of
rows (a code and its label) stand for each other, and the label is used.
Otherwise the question is refused with "Did you mean bill_length_mm or
flipper_length_mm?", listing at most five columns ("or 995 others"). A noun
for rows ("how many samples" in a table of `sample.*` columns) counts rows
instead.

"Which gene" naming three or more numeric columns (`Gene_1` to `Gene_1000`)
asks about the columns themselves: "which gene has the highest average" ranks
them, and "which genes differ between AML and ALL" compares each.

### 3. Choosing the intent

The parser detects a grain (hourly, by minute, daily, weekly, monthly,
quarterly, yearly; "hour" and "second" alone are not grain words, since "by
hour of day" asks for a profile and "second highest" a rank) and an
aggregation word (total, average, median, maximum, highest, longest, ...),
reading the question with the matched column names removed so that
`Total Sales` does not read as "total". Then it checks these signals in order:

| Signal | Result |
|---|---|
| A question about missing values | `missing`, for every column or the one named |
| "duplicate" | `duplicates`, of whole rows or of one column's values |
| "outliers", "unusual", "impossible" | `outliers`, in one column or across the numeric readings; "outlier spectra" in a matrix table judges each row by its row summary |
| "distribution of", "describe", "summary statistics" | `describe` of one measure, by a grouping when one follows "by" |
| "which column", "which gene" | `columns` |
| Two groupings named together ("by A and B", "by A by B") | `crosstab` |
| "most common" | `rank` of row counts for a grouping ("which Kind is most common"), or the `mode` of a number |
| "correlation", "correlated", "relate", "related" or "relationship" | `correlation` of the two numeric columns named, in the order named |
| "how many", "count", "number of", "share", "proportion", "percentage", "percent" or "fraction", without a growth word | `count`; a `breakdown` by count with a grouping word; a `trend` of row counts with a grain; a `share` within each group for "share of Result 1 by Gender" |
| A growth word, and a date column exists | `growth`; ranked by segment when one is named, or by the detected segment when the question says "which", "fastest", "slowest" or a superlative |
| "which" or "what" followed by day, week, month, quarter or year, with a superlative, and a date column exists | `rank` of the periods |
| "top N", "bottom N", or a superlative, with a grouping column | `rank` |
| A grain, "over time", "trend", "timeline", "history", "trajectory", or the date column named after "by" | `trend` |
| A grouping column with "by", "per", "across", "breakdown", "split" or "each", or a grouping column with no measure named | `breakdown`; a continuous grouping is split at its quartiles (`bins=4`) |
| A measure with an aggregation word or a superlative, or a measure named on its own | `aggregate` |
| Only conditions or values | `count` |

If none match, `parse_question` returns `None`. That triggers the AI planner
(when a key is set) or the suggestion list.

A segment with "by year", or with a part of the date ("average PM2.5 by
station and month of the year"), is a `crosstab` of the date part by the
segment. "Which year was the warmest" ranks the years of the measure, and the
answer says it read "warmest" as the highest value ("coldest" as the lowest).

"Share of Result 1 by Gender" is the share within each group: each Gender's
rows with Result = 1, out of its rows with Result recorded, with a chi-square
test of whether the shares differ. "Share of Antibiotics" for a 0/1 flag is
the share of rows where it is set.

Without an aggregation word, the measure's default is used: a sum for money
and counts, a mean for anything else, or whatever the `aggregation` override
says for the active measure (see
[Concepts](../concepts.md#amounts-rates-and-levels)). A question about another
column uses that column's own default. An explicit "total", "sum", "average",
"mean" or "median" is honoured in a ranking or breakdown too: "total Interval
Rain by station" sums. "Highest" asks for a ranking, so only the literal "min"
and "max" ask for a per-group extreme. In an aggregate, a superlative with no
aggregation word asks for the extreme: "the most snowfall" is the maximum.

A trend whose aggregation word differs from the default ("average revenue
monthly") is refused, because the trend executor always uses the default, and
the reason says how the periods combine. The same holds for ranking periods
("which month had the highest average revenue").

A spread ("standard deviation of X", "IQR of X by site") is a sample
statistic with n − 1, and a percentile ("90th percentile of X") uses pandas'
linear interpolation. Both work in aggregates, rankings and breakdowns, and
are refused in a trend.

A correlation question is refused unless it names exactly two numeric columns.
It is also refused when it names a column to group by or a grain ("by
species", "per month"), since that asks for one correlation per group. A
column named only to filter on is fine: "correlation between flipper length
and body mass for Adelie".

### Grouping column and measure

The grouping column is usually a segment column. A column named right after a
grouping word ("by", "per", "across", "each") or a ranking word ("which",
"top N", "bottom N") can also group the rows:

- **A text column with two or more values**, whatever it was detected as. A
  label on every row groups too: "which taxonomy has the highest A101" names
  the row, and "top 5 Taxonomy by A3" lists five rows.
- **A numeric column listed among the segments** (a 0/1 flag, a status code)
  groups like a text column.
- **Another numeric column** groups the rows when no segment is named and it
  has at most 50 distinct values (`GROUPING_LIMIT`), or holds whole-number
  codes that repeat (at most half as many values as rows, like a station code
  with 159 values over 119,040 rows). In "mean signal by timepoint" the
  grouping is `timepoint` and the measure is `signal`. With no other numeric
  column named, the detected measure is used. After a ranking word the column
  is what gets ranked and a later "by" names the measure: in "top 3 cylinders
  by mpg" the measure is `mpg`.
- **A continuous column** named after "by" ("average growth by temperature")
  is split into four groups of about equal size at its quartiles, labelled
  with their ranges ("Q1: 12 to 18.5"), and the answer says so.
- **The row label**, even when it is a number: "which OTU has the highest
  AQC1cm" names the row.

After a segment, the column named after "by" is the measure: "top 5 products
by revenue". With no grouping column, the measure is the numeric column with
the longest name. The default aggregation is taken from the final measure, so
a measure combines the same way in every intent.

### Counting

"How many" counts distinct values of a text column, identifier or segment it
names ("how many cities"). A numeric column is counted the same way when it
holds whole numbers, its name is singular and the question names it in the
plural right after "how many": "how many subjects" counts a `subject` column
numbered 1 to 20, and "how many stations" counts `STA`. A quantity
(`Units`) is totalled instead, and "how many" of a measurement (a
temperature) is refused.

A noun for the rows counts rows ("how many patients are positive"), or the
distinct values of an entity column when one repeats across rows: the
identifier, or the one column named with an ID word such as `ID`, `subject`
or `participant`. "Readings", "records", "samples" and other nouns in
`ROW_NOUNS` always count rows. The answer says which: "Counted the birds as
distinct ID values." or "Each row counts as one of the patients." "How many
participants per condition" counts distinct participants per condition.

A count with conditions or values gives its share of the rows: "18 of the
212,019 rows (under 0.1%) have Expected_Yield below 0." The base is the rows
where the filtered columns are recorded, and the answer says so when that
leaves rows out: "701 of the 4,569 rows with diabetes recorded (15.3%) have
diabetes = 1."

### 4. Nothing left out

After the plan is built, `_unused_part` checks it against the question. The
question is refused when the plan would leave out:

| Part | Reason given |
|---|---|
| A column named after "by" with one value | "Sample has a single value (O) in the 250,000 rows analysed (the row limit left out 2,858,545 others), so there is nothing to group by." |
| A grouping alongside a trend, or an average per one unit by another ("average X per transect by status") | "The question asks for two groupings, transect and protection_status; ADA answers one at a time." |
| A text column in the measure's place | "Precip holds text, not numbers, so it can't be averaged or totalled." |
| A second measure | "The answer can use one measure, and the question names B and Concentration." |
| A value of the grouping column | "Open is a value of Condition, which the answer groups by, so it can't also filter the rows." |
| "top N" with nothing to rank | `"top 5" needs a column to rank, as in "top 5 <group> by <measure>".` |
| A grain the plan doesn't use | "This table has no date column, so there is no month view." |
| A spread, a percentile, "distinct" or "which column" the plan doesn't use | "The answer would leave out the standard deviation the question asks for." |
| A running total the brief set aside, summed, averaged or correlated | "Total Rain keeps climbing like a meter reading ... Ask for its maximum, or use the column that records each reading's own amount." |

`rephrase_hint` returns this reason, or the problem found while reading (an
unknown word, a partial name or value several columns share). Otherwise it
names a text column the question asks to correlate. From Python,
`Analysis.ask()` returns a `Refusal` with code `unsupported_phrasing` or
`unrepresentable_question`, and the reason in its message.

## Execution

`execute_plan` applies the filters, the conditions and the year or month, then
computes the answer. It returns a `QueryAnswer`:

| Field | Contents |
|---|---|
| `answer` | The formatted answer |
| `calculation` | The operation that ran, in words |
| `table` | Supporting rows, when useful |
| `chart` | `"bar"`, `"line"`, `"scatter"` or `None` |

Every answer names its filters and conditions in the sentence ("for Dye Label
= Tartrazine, FD&C Yellow 5 (US), E102 (EU)"), ends with the plan's notes on
how the question was read, and says how many rows it used: "Based on 179 of
1,228 rows." or "Based on all 1,228 rows." An unfiltered aggregate says it in
its own sentence ("calculated from 1,228 rows") and a count states its rows.
Given `rows_not_analyzed` (`answer_question`, `execute_plan` and
`rephrase_hint` take it; `Analysis.ask` and the app pass the rows the row limit
left out), the rows are named for what they are: "Based on the 1,000,000
sampled rows (of 3,108,545)." for a table sampled at random, or "the newest"
rows for a dated one.

**A long table's focus.** When the brief follows one parameter of a long table
(pH, ammonia and temperature on separate rows), `Analysis.ask` and the app
pass every row as a `WholeTable`. A question that names another parameter
("average result for pH") is answered from its rows, and one that groups by
the parameter lists each one's value without ranking them, since they are in
different units. Any other answer ends: "This covers the 5,400 rows with
determinand.label = Ammonia(N), which the brief follows; name another
determinand.label to ask about it."

**Cautions.** An answer that combines a measure's values looks them up in the
data checks (`quality.scan`, run once per table). A maximum or minimum the
checks flag says so: "Caution: the data checks flag this value (999.9 in 8
rows, a missing-value code), so it is more likely an error or a code than a
reading." Any other answer says how many flagged values it uses, and a
grouped answer names a group the checks flag for the measure: "Caution: the
data checks flag CO for Ahmedabad: NO equals CO in 94% of Ahmedabad's rows."

**Running totals.** A column the notes set aside as a running total
(`schema.running_total_columns`) is never summed, averaged, binned or
correlated; its maximum and minimum are still answered.

Trend, growth and period-ranking answers use the same `TrendSeries` as the
dashboard, so the timeline notes ([Periods](periods.md#timeline-adjustments))
are appended to the calculation text.

Other behaviour:

- A breakdown with up to 20 groups shows them all, largest first. With more,
  the table shows the 10 highest and the 10 lowest (`EXTREME_GROUPS`) and the
  answer says how many groups there are. A rank shows `top_n`.
- A breakdown by a numeric column, an hour or a weekday lists its groups in
  their own order (timepoint 0, 1, 2...; Monday to Sunday), up to 50 of them,
  and is drawn as a line; beyond 50 it falls back to the highest and lowest.
  A ranking ("which timepoint has the highest signal") still sorts by value.
- A breakdown or ranking answer names the first group the way the brief does:
  "timepoint 5 has the highest average signal of any timepoint, at 10 (n =
  3)". A numeric group carries its column name (`formatting.group_label`),
  groups tied with the first are all named ("North and South tie for the
  highest average pH"), and an aggregate the column name already has is not
  repeated ("total rain", not "total total rain"). The table gives each
  group's `n`.
- An average, median or spread ranks only groups with at least five values
  (`MIN_RANK_VALUES`), and says how many it left out, or that every group is
  too small to rank firmly. A ranking leaves rows with no group out of first
  place.
- An average or median ranking adds one line from `compare.compare_groups` on
  whether the ranked groups really differ: "The Breed groups do not clearly
  differ (Welch ANOVA, p = 0.19; η² = 0.00), so this ranking may be chance."
  When the leading group's mean is more than twice its median, the answer says
  a few large values carry it and suggests the medians.
- A breakdown by a numeric column named like elapsed time (a gait cycle in 99
  steps, days since infection) is a profile: every value in order, however
  many there are, with the highest and the lowest named, instead of the ten
  highest and ten lowest.
- A trend says where the measure started and ended and between which periods,
  its Theil-Sen slope (per year, fitted on yearly averages of complete years
  once three or more are covered, so seasons can't tilt it; per decade too
  over 20 years) with its 95% interval and Mann-Kendall p-value
  (`timeseries.trend_test`), and its peak and low. An interval that spans
  zero is "no clear trend". Once three or more complete years are covered,
  the start and end it quotes are the first and last complete years'
  averages, so a seasonal series does not compare a summer with an autumn.
  It counts only the periods from the first to the last with a value, and
  says when the reading's own record starts after the table's ("pm25 is
  recorded only from 2021"). The slope is per period of the answer's own grain,
  so sparse months are not re-read as quarters. A series with one value
  throughout is stated once ("is 100.0% in every year") with no slope, peak or
  low. A first period the data starts
  part-way through is left out, as the last one is when it is still in
  progress, and the calculation says so. Period rankings leave it out too.
- `describe` gives n, mean, SD, median, quartiles, IQR, range and skewness,
  says whether the measure is roughly symmetric or skewed, and says when the
  mean is more than twice the median. By a grouping, the table has one row per
  group.
- `outliers` lists the values outside Q1 − 1.5×IQR and Q3 + 1.5×IQR, on a log
  scale for a right-skewed positive column as the brief judges its measure,
  with the values the data checks flag; a mostly-zero column, or one whose
  quartiles leave no spread, gets no list and says why. Across the readings it
  leads with the data checks and gives each column's count outside the fences.
- `duplicates` counts rows equal to an earlier row in every column, and again
  without the row identifier when that differs.
- `columns` ranks the columns by the aggregation asked for, with each one's n;
  the calculation notes this means something only when they share a unit.
- `crosstab` of counts gives the table with a chi-square test and Cramér's V;
  of a measure, each cell's value and n, naming the highest and lowest cells.
- When every group is one row (a label on each row), the answer names the row
  and its own value: "Hungatella hathewayi has the highest A101, at 6,803."
- A count of distinct values per group (`count_column`) gets no share column,
  since one entity can be in several groups.
- `missing` lists the columns with missing values, most first, with the count
  and share of rows, and the number of empty cells; for one column it gives
  that column's count.
- Ranking periods ("which month had the highest wind") builds the trend at
  that grain with `build_trend` and names the highest or lowest period with its
  value. Periods left empty for a rate or level are skipped. "Which month had
  the most rows" ranks row counts.
- A correlation reports Pearson r with its 95% interval (Fisher transform), the
  number of complete pairs and the Spearman rank correlation, in the same
  wording as the brief's relationship card (see
  [Evidence](evidence.md#relationships)), including whether it rests on one
  record or a few extreme ones. The pair was named by the reader, so
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
   and roles, never rows, and returns the same `QueryPlan` shape. Its filter
   values are matched with `matching_values`, the same matching as above. The
   app describes the plan in one sentence and runs it only after the user
   approves. See [The optional AI layer](ai-layer.md).
2. **Suggested questions** from `suggested_questions`: up to four starter
   questions built from the detected roles, each answerable by the rules.

The reason from `rephrase_hint` is shown above the suggestions and appended to
the refusal message.

## Edge cases

| Situation | Behavior |
|---|---|
| Empty question | `None` |
| No column named, but an aggregation word ("total") | Uses the detected measure |
| Growth question with no date column | Not treated as growth |
| Year or month asked, no date column | Answer says the scope is unavailable |
| No rows match the filters | Answer says there is nothing to calculate; a count says 0 of N rows |
| Plan names a column missing from the data | `execute_plan` raises `ValueError`; `answer_question` returns `None` |
| "top 0" | `None` |
| More than 20 breakdown groups | The 10 highest and the 10 lowest, with the number of groups |
| A breakdown by a continuous column | Four equal-sized groups at its quartiles, said in the answer |
| "average X per transect by status" | `None`: an average of per-transect totals is a two-step summary ADA does not build |
| A partial column name one column fits ("average body by island") | Uses `body_mass_g` and says so |
| A partial name several columns fit ("average length by island") | `None`, with "Did you mean bill_length_mm or flipper_length_mm?" |
| "Which year ..." with no date column | Not a period ranking; a numeric `year` column can still group if it has at most 50 values |
| Fewer than two periods with values | Answer says there is nothing to compare |
| A correlation over a column that does not vary | Answer says the correlation is undefined |
| "by hour" on dates without times | `None`, saying the dates hold no times of day |

## Adding a question shape

1. Make sure `_read_question` claims every word the shape uses; add
   function words to `QUERY_STOPWORDS` and query words to `QUERY_GRAMMAR`.
2. Extend `_plan_from_reading`, or add a helper beside `_count_plan`.
3. Add the execution branch in `_run_plan`.
4. Extend `QueryPlan` only if the shape needs new state. `_plan_columns` finds
   the columns any plan field names, so a new field needs no change there.
5. Test both parsing and execution, and that a question naming a part the
   shape leaves out is refused.
6. Check that the `calculation` string describes what ran, and that
   `describe_query_plan` in `ada/ai_insights.py` still matches.

Shares of a total ("what % of revenue is Enterprise"), date ranges and
two-step summaries ("average per transect by status") are on the
[roadmap](../../ROADMAP.md).
