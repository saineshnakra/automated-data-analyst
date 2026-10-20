# Evidence and recommendations

**Code:** `analyze_business` in `ada/business_insights.py`
**Tests:** `tests/test_business_insights.py`, `tests/test_significance.py`

Turns the calculated frames into the brief the user reads: a headline, KPIs,
evidence cards and prioritized next steps.

## Evidence and recommendations are separate

| | Evidence | Recommendation |
|---|---|---|
| What it is | Something ADA calculated | Something ADA suggests |
| Claim | A fact about the data | Interpretation, never a cause |
| Carries | The calculation | The evidence statement it rests on |
| Type | `Evidence` | `Recommendation` |

Evidence says what moved; recommendations say where to look.

## The output: `BusinessBrief`

| Field | Contents |
|---|---|
| `headline` | One sentence from the strongest available signals |
| `summary` | A short paragraph: the trend and group findings, or, without a trend, what the table seems to be (see below) |
| `roles` | The `ColumnRoles` used |
| `kpis` | Up to 4 `KPI`s |
| `evidence` | Up to 6 `Evidence` cards |
| `recommendations` | Up to 4 `Recommendation`s |

## Evidence kinds

Each kind has its own function, which returns nothing when the data does not
support it. The `trend`, `driver` and `anomaly` cards share one period grain:
the one passed as `analyze_business(..., frequency="M")`, or otherwise the one
chosen from the dates (see [Periods](periods.md#choosing-the-grain)). Cards are
assembled in this order:

| Kind | Reports | Skipped when |
|---|---|---|
| `trend` | Latest period-over-period change, compared with the series' usual movement | No date, fewer than 2 periods, or a percentage change from a previous period of zero |
| `driver` | The segment that moved the measure most between the last two periods | No date, measure or segment |
| `leader` | For a summed measure, the group with the largest share of the total ("Largest share by ..."), or with the most rows when there is no measure ("Most rows by ..."); for an averaged measure, the groups with the highest and lowest average ("Highest average by ..."); for mixed-sign values, the largest by size with no share | No segment, or a total of zero |
| `concentration` | Effective number of segments, and the top-three share | Shares are not meaningful (averaged measure or mixed signs), or the measure is neither money nor a count of things |
| `anomaly` | The most severe flagged period and how many were flagged | Under 8 periods, or nothing flagged |
| `relationship` | Strongest numeric correlation, with a confidence interval | See below |
| `outliers` | Count of measure values outside 1.5 × IQR | No measure, fewer than 12 values or 4 distinct values, IQR of zero, or no outliers |
| `quality` | Share of populated cells | No missing cells |

If more than six cards qualify, the first six are shown. A `quality` card that
falls past the sixth replaces the sixth, since it qualifies every other figure.

### Movement in context

The `trend` card gives the change in the measure's own terms: a percentage of
the previous period for an amount or count, percentage points for a rate, and
the plain difference for a level (a percentage of a temperature in °C means
nothing).

The previous and latest values are printed with enough precision to differ
(`format_values_apart` in `ada/formatting.py`), so a small change never reads
"from 3,004.3 to 3,004.3".

For an averaged measure, when either period's figure is the average of fewer
than 30 rows (`SMALL_SIDE_ROWS`), the card gives both row counts and says a
few records can swing it. The `driver` card does the same for its group: a
group whose latest average rests on 12 rows, against 180 the period before,
says so. A total is everything recorded, so a summed measure gets no such note.

It then compares the latest change with the series' own period-over-period
changes, measured the same way. It reports whether the change is within normal
variation, larger or quieter than usual, or unusually large or quiet, based on
its distance from the median change in robust-scale units (under 1, under 2,
2 or more). It needs at least five earlier changes
(`MIN_PERIODS_FOR_VOLATILITY`). When the last period ends short of typical
coverage but was not excluded, the card says part of the change may be
missing data.

The card's tone follows the measure's meaning: a rising cost or ticket count is
negative (`increase_is_welcome` in `ada/metrics.py`). A level whose name has no
word for which way is good, such as CO2 or a temperature, gets a neutral tone
(`MetricSpec.direction_known`), and so does its driver card.

### Offsetting movement

When the net change across segments is under 25% of the gross movement
(`OFFSETTING_NET_RATIO`), the `driver` card says the segments largely offset
each other and gives the driver's share of gross movement rather than of the
net. "Up 2% overall" can hide one region up 40% and another down 38%.

### Groups

Every group is named with `group_label`, so a numeric code reads as
"Status 2 has the highest average", never "2 has the highest average".

For an averaged measure the `leader` card names the highest and the lowest
group, printed with enough precision to differ. Groups whose averages are
equal are all named ("47 and 68 tie for the highest average"), and the KPI
tile and recommendations say "47 (tied with 1 other)". When a named group's
average rests on fewer than 30 rows, every named group is given with its row
count and the card says a few records can change the order.

### Concentration

Measured as effective segments, 1 ÷ the Herfindahl index (the sum of squared
shares), so it does not depend on how many groups exist. Below 3.0 effective
segments (`CONCENTRATED_EFFECTIVE_SEGMENTS`) the card is a warning. For an
all-negative measure, where effective segments are undefined, the card reports
the top-three share instead and warns at 70% or more.

The card appears only for money and counts of things (whole-numbered values),
since how concentrated revenue or orders are across products is a real
question, and how concentrated rainfall is across weather stations is not.

### Correlation

The strongest numeric pair is reported only when all of these hold:

| Requirement | Value |
|---|---|
| Absolute Pearson correlation | At least 0.45 (`MATERIAL_CORRELATION`) |
| Complete pairs | At least 12 (`MIN_CORRELATION_PAIRS`); a pair with fewer is not considered |
| Columns | Not identifiers, not experimental-design labels (`Replicate`, `Plate`, `Row`, `Column`, ...), not labels for the unit a row belongs to (`subject`, `participant`, `patient`, `animal`, `mouse`, `rat`, `donor`, `individual`, `specimen`, or an id-like word; `is_entity_name` in `ada/schema.py`), not named exactly as a time part (`year`, `month`, ...) |
| Not a near-duplicate | See below |

A participant numbered 1 to 20 with three rows each is not unique enough to
look like an identifier, but its number is a label, so it is left out by name
whatever its values. The same goes for a plate's `Column`, which correlates
with a dose only because the dilution series was laid out across the plate.

**Near-duplicates.** A pair with an absolute Pearson correlation of 0.99 or
more (`NEAR_DUPLICATE_CORRELATION`) is usually one quantity twice: a reading
and its adjusted copy, or a size in two directions that always match. Such a
pair is not chosen as the relationship. If any were set aside, the card ends
with one sentence naming up to three of them (`NEAR_DUPLICATES_NAMED`) and
saying they probably measure the same thing or one is derived from another.
If no other pair qualifies, there is no card. A pair whose rank correlation is
more than 0.25 below its Pearson figure is not a near-duplicate, since a few
extreme records produce that figure; it stays in and is reported with that
caveat.

The card gives a 95% confidence interval from the Fisher transform, and the
sample size. If the interval includes zero, the card says the sample cannot
rule out no relationship. If Spearman rank correlation differs from Pearson by
more than 0.25 (`RANK_DIVERGENCE`), the card says so: a stronger rank
correlation means a consistent but curved relationship, a weaker one means a
few extreme records produce the linear figure. It also notes that the pair was
picked as the strongest of several and that the interval is not adjusted for
that selection.

## KPIs

Added in this order, up to four:

1. Total measure (summed measures only)
2. Average measure
3. Latest movement (when there is a `trend` card)
4. The `leader` card's figure, labelled with its title ("Highest average by
   site", "Largest share by category") and naming the group

A label never repeats an aggregate the column name already has: the total of
`Total Rain` is labelled `Total Rain`, not `Total Total Rain`
(`with_aggregate` in `ada/formatting.py`).

Remaining slots are filled from: distinct identifiers, rows analyzed, data
completeness. An identifier with one row per value is a row key, not a count
of anything, so it gets no tile of its own; the rows tile says "One per
Reading ID" instead. A label already shown is skipped, so a file with little
structure gets fewer tiles rather than duplicates.

## Summary

With a trend, the summary repeats the trend and group statements. Without
one, the headline already carries the only finding, so the summary says what
the table seems to be: how many rows, what one row is when an identifier names
each row once, the date span, the measure and how it combines, the groups, and
which of a date and a grouping column is missing:

> 1,319 rows. Measure: Blood sugar, averaged across rows. Groups: the 2 values
> of Result. No date column was found.

When the roles list runner-up measures (`ColumnRoles.measure_alternatives`),
the summary ends by naming them: "Measure picked: realism (also plausible:
risky_decision and risk_perception). Change it with measure="risky_decision"
or in the app."

## Recommendations

`_recommendations` maps evidence to actions. Each action has a priority
(`Now`, `Next`, `Test`, `Watch` or `Fix`), and its rationale is the statement of
the evidence card it comes from.

| Evidence | Recommendation |
|---|---|
| `trend`, negative | Start with the driver segment, or find where the adverse change started |
| `trend`, positive or neutral | Check whether the change holds |
| `concentration`, warning | Check how much rests on the largest group, named |
| `leader`, averaged measure | Compare the highest and lowest group against the spread within each |
| `leader`, summed measure | Compare the largest group, named, with the others |
| `anomaly` | Explain the flagged periods |
| `relationship` | Test the relationship |
| `outliers` | Review the extreme records |
| `quality`, warning (5% or more missing) | Look into the missing values |

The driver is named only when it moved in the same direction as the total. With
no qualifying evidence, a single "Add a date or a grouping column"
recommendation explains which columns would enable the analysis.

Recommendations are suggestions of what to look at next, worded for any
table: they name only the splits the table has (no "time window" without a
date), and an extreme value is something to check, not presumed an error.

## The Markdown report

`build_business_report` renders the brief as a downloadable Markdown executive
brief, with the calculation under each card. `ada.Analysis.to_markdown()`
returns the same report, and `to_dict()` returns the brief as JSON-ready data.

## Changing this

A new evidence card needs:

1. A function returning `Evidence | None`, with `None` when the data cannot
   support it.
2. A `calculation` string a reader could reproduce by hand.
3. Tests for the case where it appears and the case where it is skipped.
4. Defined behaviour for no history, one segment, all zeros and negative
   values.

If a heuristic can produce a misleading result, add the case to the tests and
state the limitation in the card's text.
