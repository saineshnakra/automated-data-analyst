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
| `trend` | Latest period-over-period change on the groups present in both periods, compared with the series' usual movement or with the time of year | No date, fewer than 2 periods, or a percentage change from a previous period of zero |
| `driver` | The segment that moved the measure most between the last two periods | No date, measure or segment |
| `leader` | For a summed measure, the group with the largest share of the total ("Largest share by ..."), or with the most rows when there is no measure ("Most rows by ..."). For an averaged measure, a comparison of the groups with its test and effect size (see below). For mixed-sign values, the largest by size with no share | No segment, or a total of zero |
| `concentration` | Effective number of segments, and the top-three share | Shares are not meaningful (averaged measure or mixed signs), or the measure is neither money nor a count of things |
| `differences` | Which numeric readings differ between groups, largest first, after adjusting for the number of tests | No grouping, no reading, or a scan of fewer than 3 comparisons that found nothing |
| `anomaly` | The most severe flagged period and how many were flagged, against the band for its time of year when the series has a yearly pattern | Under 8 periods, or nothing flagged |
| `relationship` | What relates to the measure, then the strongest other pair, or a null result | Fewer than two numeric readings, or no pair with 12 complete rows |
| `outliers` | Count of measure values outside 1.5 × IQR | No measure, fewer than 12 values or 4 distinct values, IQR of zero, or no outliers |
| `quality` | Share of populated cells | No missing cells |

A `differences` card that found nothing comes after `outliers` rather than
after the segment cards, so a null result never pushes out a finding. If
more than six cards qualify, the first six are shown. A `quality` card that
falls past the sixth replaces the sixth, since it qualifies every other figure.

The `driver` and `leader` cards name a group of a numeric grouping with its
column (`Status 2`, not a bare `2`; see `formatting.group_label`).

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
coverage but was not excluded, the card calls it the "latest period" rather
than the "latest complete period" and says part of the change may be missing
data.

**Time of year.** At a monthly or quarterly grain, a series with a clear yearly
pattern (`yearly_pattern` in `ada/timeseries.py`; see
[Periods](periods.md#yearly-patterns)) is compared with the time of year
instead, since a winter-to-spring rise is not news in a series that rises every
spring. The card gives the range of the same move in earlier years (from the
previous month or quarter into the latest one), says whether the latest move
is within that range, and sets the latest value beside the same period a year
earlier: *"In the 6 earlier years on record, the move into Q2 ranged from
+8.63 to +16.40 (median +13.86), so this one is within the range usual for the
time of year. Against Q2 2022 (15.52), it is down 2.39."*

**Groups that come and go.** With a grouping column, the change is measured on
the groups that reported in both of the last two periods
(`matched_period_change` in `ada/aggregation.py`). Weather stations that start
reporting halfway through a record, or a site that stops, would otherwise
move a pooled average on their own. The card names the groups left out and
gives the pooled figures too: *"This compares the 88 station values that
reported in both periods; ... With every station included it went from 29.96
to 24.82."* The usual movement and the time-of-year comparison then use the
same groups' history. When no group reported in both periods, the pooled
change is kept and the card says it compares different groups.

The card's tone follows the measure's meaning: a rising cost or ticket count is
negative (`increase_is_welcome` in `ada/metrics.py`). A level whose name has no
word for which way is good, such as CO2 or a temperature, gets a neutral tone
(`MetricSpec.direction_known`), and so does its driver card.

### Drivers on the same groups

The `driver` card compares only segments with figures in both of the last two
periods (`segment_period_change`). A segment that starts or stops reporting
would read as rising from zero or falling to it, and as "100% of the
movement"; such segments are named as left out. When only one segment is in
both periods, the card says so instead of giving it a share.

### Ranked groups

The `leader` card ranks segments on the figures they have. A segment with no
values for the measure (a station with no rain gauge) has no figure rather
than zero: it is left out of the ranking and named. When a ranked segment's
figures cover less than 75% of the table's dates (`PARTIAL_COVERAGE_SHARE`),
the card gives its dates, since an average over two lockdown months is not
comparable with one over five years. `Analysis.segments()` lists every
segment's first and last date (`From`, `To`) when there is a date column.

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

### Group differences

A gap between two group averages says nothing without the spread and the
sample behind it, so for an averaged measure the `leader` card compares the
groups with `compare_groups` (see [Comparing groups](compare.md)): the n,
mean and median per group, Welch's test with a Mann-Whitney check for two
groups or Welch's ANOVA with a Kruskal-Wallis check for more, and an effect
size. A skewed or capped measure is compared by medians and ranks. When a
subject column repeats, each subject counts once per group.

| Result | Card |
|---|---|
| Clear difference | Title "Highest average by *group*" (or "Highest median by *group*" when medians are compared), the top group's mean or median as the value, and "X is higher in A than in B" or the highest and lowest groups, naming any tied with them |
| No clear difference, or a negligible one | Title "No clear difference by *group*", the effect size as the value, and the groups' figures; no group is called highest, no KPI names a top group, and the headline says there is no clear difference |

When too few groups have two values to test, the card falls back to naming the
highest and lowest average and says the gap can't be tested.

The `differences` card compares every numeric reading across the grouping
column and the other groupings with 2 to 12 levels (`MAX_SCAN_LEVELS`),
including numeric codes listed in `ColumnRoles.factors`, with
`scan_differences`. Readings are the numeric columns that are not identifiers,
codes, design labels, time parts, yes/no flags or numbers for a grouping's
levels (a `Group` 1 to 12 beside its `Gp` label). A grouping made of two others
(a cell code for contour and depth) is replaced by those two, and the subject
column is never a grouping.

- P-values are Holm-adjusted for every comparison run, and the calculation says
  how many.
- The card leads with the largest clear difference, names the next two with
  their effect sizes, and its value is the count of clear differences ("3 of 7").
- With nothing clear it says so and names the largest effect.
- With several groupings it adds the share of the measure's variance each one
  explains: "Of the variance in Ca, Depth explains 76%, Contour 5%".
- The measure across the grouping column is left to the `leader` card when the
  measure is averaged.

When the `leader` card finds no clear difference and the `differences` card
finds one, the headline and summary lead with the difference found.

### Relationships

The `relationship` card ranks what relates to the measure first, then the
strongest pair among the other columns, using `scan_relationships` (see
[Comparing groups](compare.md#relationships)):

1. Every numeric reading is correlated with the measure. The card states the
   strongest with its Pearson r, 95% interval by the Fisher transform, n and
   rank-correlation checks, and lists the next two with their Pearson and
   Spearman r and n.
2. It then states the strongest pair among the other columns.
3. A correlation reaches the card at |r| of 0.3 (`NOTABLE_CORRELATION`); below
   that the card says none of N columns reaches it and names the closest.
   Over 1,000 rows and 10 or more pairs, a table whose strongest correlation
   is under 0.05 is called worth a check of how its values were produced.
4. It always says how many pairs were scanned, and that the ones shown were
   picked from them, so they are leads to check rather than evidence of cause.
   When a picked correlation could arise by chance once every correlation
   tried is allowed for (a Bonferroni bound), it says so.

Pairs are ranked by the larger of |Spearman r| and |Pearson r without the
pair's most influential record|, so a single impossible record can neither
make nor hide the strongest pair. Pairs related by construction are left out
and listed in the calculation: a column and the columns it is built from (a
total and its items, a mean, a difference), a flag that only marks another
column's cap, same-stem names above |r| = 0.95, and near-duplicates
(|r| ≥ 0.99 with ranks that agree, such as a reading and its adjusted copy).
Near-duplicates are also named in the statement: up to three of them
(`NEAR_DUPLICATES_NAMED`), saying they probably measure the same thing or one
is derived from another.

Each pair the card states is checked:

| Check | Statement |
|---|---|
| The 95% interval includes zero | The sample cannot rule out no relationship |
| Dropping one record moves r by more than 0.2 | "It rests on one record: without the one where HCT is 3,715 and MCH is 3,117, r is -0.01" |
| Spearman r more than 0.25 above Pearson r (`RANK_DIVERGENCE`) | Extreme records hide the relationship, with r without them, when dropping records over 5 robust SDs out closes the gap; otherwise the relationship is consistent but not a straight line |
| Spearman r more than 0.25 below Pearson r | A few extreme records drive the linear figure, with r without the most extreme when that is known |
| The r within the groups of a grouping differs in sign, or by 0.3 or more | "Within each dye the relationship runs the other way: r = -0.97 once each dye's own means are removed", or that it is stronger or weaker within groups |

Participant numbers, plate positions and other labels are not readings, so
they never enter the scan. Wide tables are scanned on the measure and the 299
most variable other columns, and the calculation says so.

## KPIs

Added in this order, up to four:

1. Total measure (summed measures only)
2. Average measure
3. Latest movement (when there is a `trend` card)
4. The `leader` card's figure, labelled with its title ("Highest average by
   site", "Largest share by category") and naming the group; left out when the
   groups show no clear difference

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
| `leader`, averaged measure, clear difference | Find out what sets the top group apart |
| `leader`, averaged measure, no clear difference | Don't rank the groups on the measure |
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
