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
| `outliers` | Values outside 1.5 × IQR in every reading, judged on each one's shape and within the grouping that explains it; a shift when one reading's form one run in time; for a mostly-zero measure, its tail instead | No reading with more flagged values than chance gives, and no shift |
| `quality` | The data checks' leading issues with counts, and the share of populated cells | No issue and no missing cells |

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

### Outliers

The card counts values below Q1 − 1.5 × IQR or above Q3 + 1.5 × IQR in the
measure and every other numeric reading (up to 40, `OUTLIER_READINGS`; a
matrix table's columns are not counted one by one), each measured the way it
is shaped:

- A non-negative, right-skewed reading (concentrations, latencies, counts) is
  judged on log10 of its values, when that makes its bulk more symmetric
  (`prefers_log_scale` in `ada/quality.py`). On the raw scale its long tail
  would be called exceptional.
- Each reading is judged within the grouping that explains most of its
  variance (epsilon squared of 1% or more): the grouping column, another
  grouping or code, the subject column (a station), or a swept numeric setting
  (a frequency stepped evenly over up to 60 levels, each about equally common).
  Each group's values are judged against that group's own quartiles, so a
  group's level is not mistaken for outliers (arctic and tropical stations,
  resonant frequencies). A group with fewer than 12 values
  (`MIN_GROUP_FOR_FENCES`) or no spread uses the pooled quartiles.
- A count of subjects or samples is skipped, and so is a reading a quarter of
  whose values are one value: its IQR measures that pile, not the spread.
- A normal sample puts about 0.70% of its values outside the fences, so a
  reading's count is reported only when a normal sample of its size would reach
  it with under 1% chance (binomial, `OUTLIER_CHANCE`). One value in 104 is what
  chance gives and raises no card. The card leads with the measure when its
  count is beyond chance, and otherwise with the reading furthest beyond it;
  it names up to three more (`OUTLIERS_NAMED`), says when the measure's count is
  about what chance gives, and notes when a cap or floor holding 10% or more of
  the lead reading narrows its IQR.
- Cells the data checks flagged as impossible, codes, fill values, swapped or
  stuck are left out, and the calculation says how many.
- For a reading with more than 50,000 values, the scale, the quartiles and the
  choice of grouping come from a random sample of 50,000 of them
  (`FENCE_SAMPLE_ROWS`); the values are counted on every row.
- When half or more of the measure's values are 0 (`HEAVY_TAIL_ZEROS`), no
  outliers are listed. The card is titled **Heavy-tailed measure** and gives the
  share of zeros and the 99th percentile instead.
- With a date, each reading's flagged values are put in time order within each
  group, the measure first. When one unbroken run holds at least half of them
  and at least five (`SHIFT_RUN_SHARE`, `MIN_SHIFT_RUN`), the card is titled
  **Level shift**: that reading left its usual range at the run's first time and
  stayed out, to the end of the record or until its last time. A fire that
  raises every reading for ten minutes is an event, not a run of entry errors.

### Data checks

Before the cards are built, `ada/quality.py` scans every numeric column; see
[Data checks](quality.md). When it finds anything, the `quality` card is
titled **Data checks**, its value is the number of issues, and it names the
three issues about values (`QUALITY_ISSUES_SHOWN`) and the two about gaps
(`GAP_ISSUES_SHOWN`) that cover the most rows, each with its counts, followed by
how complete the table is. `Analysis.quality()` lists them
all. With no issue, the card is the earlier **Data completeness** card,
skipped when nothing is missing.

More comes from the same checks:

- **Check these first.** When an issue touches the measure, or the group the
  headline names, or covers 10% or more of the table as copied rows or
  simulated-looking data do (`CHECK_FIRST_SHARE`), or is any other issue about
  values covering a quarter of its column or more (`CHECK_FIRST_BROAD`, such as
  fill values in 61% of rows), the summary ends with up to two of them
  (`CHECK_FIRST_SHOWN`), those covering the most rows first.
- **Cells left out.** The relationship scan, the group comparisons and the
  outlier count run on a copy of the table with the cells the checks found
  can't be readings blanked (`statistics_frame` in `ada/quality.py`): impossible
  values, codes, fill values, swapped and stuck readings. Each of those cards
  ends its calculation with how many cells were left out.
- **A leader carried by flagged values.** For an averaged measure, the leading
  group's average is recomputed without the values the checks flagged
  (impossible values, codes, extreme values, survey codes). When that moves it
  by more than 5%, or hands the lead to another group, the leading-group card,
  and the headline built from it, say how many flagged values the average rests
  on, give the group's median, and name the group that would lead without them.

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

The groupings are taken design factors first: the grouping column, then text
labels, then two-level codes, then other codes by how few levels they have, so
a rating item read as a code comes last. At most six are scanned
(`MAX_SCAN_GROUPINGS`), and the calculation names the ones past the cap. Also
left out, and named:

| Left out | Why |
|---|---|
| A numeric column that repeats a part of the date column (`MO` is the month of `Date`) | It shows the season or the trend, not a factor |
| A grouping nested within one already scanned (concentrations 3 to 12 used only with the blue dyes) | Its differences repeat the other grouping's |
| A copy of a reading in other units (`MaxTemp` in °C beside `MAX` in °F) | The reading is compared once, as the measure when it is one |

Pairs related by construction are not compared:

| Pair | Example | Card |
|---|---|---|
| A grouping set by the reading | Farm size cut from acres; anaemia set by haemoglobin below 12 for women and 13.5 for men; a pollutant label naming the largest sub-index | The rule is stated in the card: "Result is set by Hemoglobin: Result 1 below 12.0 for Gender 0 and below 13.5 for Gender 1, on 100% of rows, so it is not compared with Hemoglobin or anything built from it" |
| A score and the items it is built from | A scale that is the mean of its rated items, read as codes | Named in the calculation |
| A reading fixed within each group | A station's battery voltage, a reef's coordinates | Named in the calculation as an attribute of the group |
| Coordinates and compass directions | Latitude, wind direction | Never compared |

The rules come from `grouping_rules` and the pairs built from one another from
the brief's screen of the table (`construction`), which the `relationship`
card uses too (see [Comparing groups](compare.md#pairs-related-by-construction)).
A grouping set by the measure also gets no `leader` card, so it never makes
the headline.

- P-values are Holm-adjusted for every comparison run, and the calculation says
  how many.
- The card leads with the largest clear difference, names the next two with
  their effect sizes, and its value is the count of clear differences ("3 of 7").
- With nothing clear it says so and names the largest effect.
- With several groupings it adds the share of the measure's variance each one
  explains: "Of the variance in Ca, Depth explains 76%, Contour 5%". Groupings
  set by the measure, or that it is built from, are left out of the shares.
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
total and its items, a mean, a difference, the largest of several readings, a
score standardised within each group), shares that add up to a fixed total, a
flag that only marks another column's cap, same-stem names above |r| = 0.95,
two map coordinates, and near-duplicates (|r| ≥ 0.99 with ranks that agree,
such as a reading and its adjusted copy). Near-duplicates are also named in
the statement: up to three of them (`NEAR_DUPLICATES_NAMED`), saying they
probably measure the same thing or one is derived from another. The columns
left out of the measure's own ranking are named with the reason ("Left out of
the ranking for Gas_Index: mq135, mq4, mq6 and 3 others (built from it, or it
from them)"). When every column is part of the measure, as with the row mean
ADA adds to a matrix table, the card says so and reports the strongest other
pair. Rows that break a block of shares, and a summary that reads 0 where
every part is blank (treated as missing), get a sentence of their own. Compass
directions never enter the scan.

Tables longer than 50,000 rows are screened on a fixed random sample of that
many rows, the same one the comparisons use; the pairs the card names are
computed on every row.

Each pair the card states is checked, and the sentence leads with the figure
that one record or a few extreme ones do not decide:

| Check | Statement |
|---|---|
| The 95% interval includes zero | The sample cannot rule out no relationship |
| Dropping one record moves r by more than 0.2, towards zero | "HCT and MCH move together only weakly once one record is set aside: Pearson r = 0.02 without it … It rests on one record: with the one where HCT is 3,715 and MCH is 3,117, r is 1.00" |
| Dropping one record moves r by more than 0.2, away from zero | "RBC and HGB move together once one record is set aside: Pearson r = 0.89 without it … One record hides a strong relationship: with the one where RBC is 90.80 and HGB is 12.60, r is only 0.46" |
| Spearman r more than 0.25 above Pearson r (`RANK_DIVERGENCE`), and dropping records over 5 robust SDs out closes the gap | Leads with r without those records; the extreme records hide the relationship |
| Spearman r more than 0.25 above Pearson r otherwise | Leads with the rank correlation: consistent but not a straight line |
| Spearman r more than 0.25 below Pearson r | Leads with the rank correlation and gives the cause: a few extreme records (with r without them), one unbroken stretch of records far beyond the rest (a phase of a run), or a pile-up at a column's floor or ceiling, which ties in rank |
| The r within the groups of a grouping differs in sign, or by 0.1 or more beyond sampling noise (`WITHIN_GROUP_GAP`) | "Within each block it is stronger: r = 0.91 in Block 1 and 0.94 in Block 2 (0.92 once each block's own means are removed)", or that it is weaker or runs the other way. The grouping column is always checked and reported when it differs; another grouping is added when it differs more. Groupings set by either column, or that it is built from, are not used |

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
| `outliers`, a level shift | Find what changed when the run began |
| `outliers`, a heavy-tailed measure | Compare medians, not means |
| `outliers` | Review the extreme records |
| `quality`, data checks | Check the flagged values first (placed first, since flagged values qualify every other figure) |
| `quality`, warning (5% or more missing), no other issue | Look into the missing values |

The driver is named only when it moved in the same direction as the total. With
no qualifying evidence, a single "Add a date or a grouping column"
recommendation explains which columns would enable the analysis.

Recommendations are suggestions of what to look at next, worded for any
table: they name only the splits the table has (no "time window" without a
date), and an extreme value is something to check, not presumed an error: the
outlier recommendation says values beyond 1.5 × IQR are often real but rare.

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
