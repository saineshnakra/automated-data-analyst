# Comparing groups and finding relationships

**Code:** `ada/compare.py`
**Tests:** `tests/test_compare.py`

Two questions come up with almost any table: does a reading differ between
groups, and what moves with it? This module answers both with the sample, the
test and the effect size behind the answer. The brief's `leader`,
`differences`, `rates`, `model` and `relationship` cards (see
[Evidence](evidence.md)) and the comparison questions in [Ask ADA](ask-ada.md)
all use it. The group tests are computed here for every column at once, with
the t, F and chi-squared distributions from `scipy.stats`, so a one-reading
comparison and a scan of hundreds of readings give the same numbers; they
match `scipy.stats.ttest_ind`, `f_oneway`, `mannwhitneyu` and `kruskal`. The
multiple-testing adjustments, the rate interval and the joint model come from
`statsmodels`, and the rate tests from `scipy.stats`. See
[Credits](../credits.md).

## Comparing a reading across groups

`compare_groups(dataframe, measure, grouping, entity=None, axis=None)` returns a
`GroupComparison`, or `None` when fewer than two groups have two or more
values.

| Groups | Reported |
|---|---|
| Two | n, mean and median per group; the difference in means with a Welch 95% interval; Welch's t-test; Hedges' g; checked with the Mann-Whitney test |
| Three or more | n, mean and median per group; Welch's ANOVA; eta squared (the between-group share of the total sum of squares); checked with Kruskal-Wallis; the highest and lowest group, with Hedges' g between them |

Groups run from the highest centre to the lowest. A group with fewer than two
values (`MIN_GROUP_SIZE`) is left out and named. A group with no spread at all
gets a variance floor far below anything measurable, so Welch's tests stay
defined.

### Small groups

| Rule | Why |
|---|---|
| A group with fewer than 3 values (`MIN_WELCH_GROUP`) switches the test on means to the pooled-variance one: Student's t-test, or the one-way ANOVA | Welch's test weights each group by its own variance, which two values estimate so poorly that a tight pair can carry a p of 0.0003 where the classical ANOVA gives 0.18 |
| Groups with fewer than 5 values (`MIN_NAMED_GROUP`) are never named as the highest or lowest | The highest and lowest are taken among groups with 5 or more values, and the smaller groups beyond them are listed as "Not ranked, with fewer than 5 values: CLL (mean −1.09, n = 2)" |
| A difference resting on a group of fewer than 5 is not called clear | The verdict is `thin` |

### Medians and ranks for skewed or capped readings

The comparison switches to medians and ranks when the reading is:

| Condition | Rule |
|---|---|
| Strongly skewed | Skewness of the values around their own group's mean of 2 or more (`SKEWED`), from at least 20 values (`SKEW_MIN_VALUES`) |
| Capped | At least 3 values, and at least 2% of them (`CAP_SHARE`), sit exactly at the maximum or the minimum, in a reading with 10 or more distinct values: a timer stopped at 3,600 s, an assay ceiling. A 1 to 7 rating is not called capped |

Then the Mann-Whitney test (two groups) or Kruskal-Wallis (more) leads, the
effect size is epsilon squared (H ÷ (n − 1)), and two groups also get the
share of cross-group pairs in which the higher group's value is the larger
("a value from positive is the larger in 89% of pairs"). Welch's test becomes
the check. The statement says why ranks were used.

### Which pairs differ

The test across three or more groups says only that some group differs. After
a clear difference in means across 3 to 12 groups of 5 or more values, every
pair is compared by Tukey's HSD, which holds the chance of any false pair at
5%, and the statement names the pairs it tells apart, or the few it does not:
"Tukey's HSD tells 5 of the 6 pairs apart; only 30-60 with 60-90 is not." A
pair differs when its Tukey-Kramer q is above the 95th percentile of the
studentized range (`scipy.stats.studentized_range`), the same decision
`scipy.stats.tukey_hsd` makes; the percentile is found once rather than a
p-value integrated for every pair, which took over a second for 12 groups.
The pairs are kept on the comparison as `pairs_apart`, largest q first, with
`pairs_tried`.

### Readings that are mostly zero

When half or more of a reading's values are 0 (`ZERO_HEAVY`), as for rain per
interval, every group's median is 0 and "medians from 0 to 0" says nothing.
Such a reading is named and ordered by its group means, and each group also
gets the share of its values that are not 0, compared with a chi-squared test,
or Fisher's exact test for two groups with an expected count below 5, and the
statement names the test used. The range of shares runs between groups large
enough to name, or between all groups when fewer than two are, so a lone
named group never stands at both ends. The statement gives both; the table
adds a `Share not 0` column. When the rank test leads but the group with the
higher mean is the one whose values rank lower (a few large values carry the
mean), the statement names the direction the rank test supports: "Crabs tends
to be larger in Nudibranchs 1 than in Nudibranchs 0: a value from Nudibranchs
1 is the larger in 88% of pairs (Mann-Whitney, p = 0.0059), though the mean is
higher in Nudibranchs 0". A check that disagrees with a thin difference is
named too.

### Ordered numeric levels

A grouping of three or more numeric levels whose name does not mark labels (a
speed, a depth, a dose; not `Status`, `Group`, `Plot` or `Block`) also gets a
trend test (`ordered_levels`): Spearman's rank correlation between the level
and the reading, with its p-value from the t approximation, which matches
`scipy.stats.spearmanr`. It needs 10 values (`MIN_TREND_VALUES`). The
statement says whether the reading rises or falls as the level increases
("Elongation falls as welding speed increases across its 5 levels"), or that
there is no clear trend. A trend uses the order of the levels, so it can be
clear when the unordered test is not.

Both rank tests use the normal approximation with ties corrected, so a scan
and a single comparison agree.

### The verdict

| Verdict | When | Wording |
|---|---|---|
| `clear` | p below 0.05, at least a small effect, both groups compared with 5 or more values, and the check agrees | "X is higher in A than in B", or "X differs across G" with the highest and lowest group |
| `negligible` | p below 0.05 but less than a small effect: Hedges' g under 0.2 for two groups compared by means, otherwise under 1% of the variance (`NEGLIGIBLE_EXPLAINED`) | "differs only by a negligible amount"; above 1,000 records (`LARGE_SAMPLE`) it adds that with this many records a trivial gap passes any test |
| `thin` | p below 0.05 but a group compared has fewer than 5 values | "looks higher", and too few values to call the difference clear |
| `disputed` | p below 0.05 but the check's p (adjusted by the same factor in a scan) is not | "shows no clear difference", naming both p-values and saying the two tests disagree |
| `none` | p of 0.05 or more | "shows no clear difference", with the interval or the range of the groups; never a ranking |

For two groups compared by means, p below 0.05 is the same as a 95% interval
for the difference that excludes zero. Effect sizes are named by the usual
cut-offs: Hedges' g of 0.2, 0.5 and 0.8 (small, medium, large) and a share of
variance of 0.01, 0.06 and 0.14. A card quotes one effect size in one unit:
when rows are reduced to one value per entity, so are the variance shares
beside the comparison.

### One value per subject

Repeated readings of one subject are not independent evidence. With `entity`
(a subject, patient or sample column), each entity contributes one value per
group: its mean, or its median for a reading that is skewed or capped. The
statement says "Each Bird counts once per group", and names each group's
figure for what it is: "the highest mean of per-Bird medians", not a plain
mean that a table of rows would show (`centre_word`). A reading copied onto
every row of an entity has one value per entity, not a summary of several:
"the highest mean over Visit values". When the same entity appears in more
than one group but not in all of them, the groups are not independent and the
statement says the p-value is only a guide.

`entity_column(dataframe, roles)` picks the entity, in this order:

| Entity | When |
|---|---|
| The sample the measure was read on (`reading_key`) | A column of labels (text, or whole-number codes named as a unit or a design label) that the measure never varies within, with at least 10 values (`KEY_MIN_LEVELS`) and a quarter or more of the rows repeating one (`KEY_REPEATED_SHARE`); the measure needs 10 distinct values. A site visit's conductivity copied onto each species row of the visit is one reading. The finest such column wins; a long table is screened on 5,000 rows first |
| The identifier, or a column named like one | `is_entity_name` in `ada/schema.py` (`Subject`, `Patient ID`, `subject#`), provided some value repeats and the values can label units: text, whole numbers with at most half as many distinct values as rows, or codes listed in runs of consecutive rows (`1.2` for mouse 2 of cage 1) |
| Unnamed codes listed in runs | At least 5 codes (`BLOCK_MIN_UNITS`), each filling one run of 4 or more rows on average, within which a numeric column climbs and starts again at the next run (`within_axis`): a time point or cycle counted per mouse or per test case |

A decimal reading named like a unit (`NO`, nitric oxide) is a measurement, and
so is the measure or a close alternative to it, except that runs with a
restarting clock overrule a close call on the measure.

`within_axis(dataframe, entity)` finds that clock: a numeric column that never
falls within 90% of the units (`AXIS_SHARE`), climbs on at least half its
steps, falls back at half the boundaries between units, and holds at most
half as many values as rows, so a row counter running down the file is not
one. A name like a time (`day`, `timept`, `cycle`) comes first.

`reading_key(dataframe, measure)`, `constant_within(dataframe, entity,
column)` and `one_row_per(dataframe, key)` are the parts: `one_row_per` keeps
each key's first row, under its own index label, with every column that never
varies within the key and the others left missing, so readings copied onto a
sample's rows are correlated once per sample.

### Groups followed for different lengths

With `axis` (each entity's clock), when each entity sits in one group and the
groups were followed over different stretches, a mean over all rows compares
different times: animals that died were measured to day 9 and survivors to
day 15, so survivors' late recovery would set the comparison. Each group's
typical span runs from the median of its entities' first clock values to the
median of their last; only rows inside the stretch every group shares are
compared, when cutting to it leaves out at least 10% of the rows
(`WINDOW_MIN_CUT`). The comparison keeps the stretch as `window` and each
group's median last value as `followed`, and the statement says "The groups
were followed for different lengths of day (the median mouse in Live to 15,
Dead to 9), so only day 0 to 9, the stretch they share, is compared."

### Groups that hold the same subjects

When at least 5 entities (`PAIRED_MIN_UNITS`), and 80% of them
(`PAIRED_SHARE`), appear in every group, as when each person walked in every
condition, the groups are compared within entities (`paired`), on one value
per entity in each group; the others are left out and counted
(`unpaired_left_out`):

| Groups | Test on means | Test on ranks | Effect |
|---|---|---|---|
| Two | Paired t-test (`scipy.stats.ttest_rel`), with the paired 95% interval for the mean within-entity difference | Wilcoxon's signed-rank test (`scipy.stats.wilcoxon`) | Partial eta squared |
| Three or more | Repeated-measures ANOVA, with partial eta squared (between-group over between-group plus error sums of squares) | Friedman's test (`scipy.stats.friedmanchisquare`) | Kendall's W on ranks, Friedman's statistic over n(k − 1) |

The repeated-measures F is computed for every column at once, with the figures
of `statsmodels.stats.anova.AnovaRM`, which fits one reading at a time. A
paired gap is judged small or not on the share of within-entity variation it
explains, since Hedges' g scales it by the spread between entities, which the
design sets aside. After a clear difference across three or more groups, the
pairs are told apart by paired t-tests, Holm-adjusted over the pairs, in place
of Tukey's HSD, which assumes independent groups. The statement says "Every
subject appears in every group, so each is compared with itself
(repeated-measures ANOVA)."

### Comparing within the levels of another grouping

`stratified_comparison(dataframe, measure, grouping, candidates, entity=None,
axis=None, prefer=None)` repeats the comparison within the levels of another
grouping that could make or hide it, and returns a `Stratified`, or None:

| Stratum | When |
|---|---|
| A confounder | Tied to the grouping, with Cramér's V of 0.5 or more (`STRATUM_ASSOCIATION`, from `scipy.stats.contingency.association`), and explaining at least 14% of the measure (`STRATUM_EXPLAINED`, one value per entity when there is one). Gravel soils show less seismic activity only because gravel occurs only at sites without landslides |
| A channel | Explaining half the measure or more on its own (`CHANNEL_EXPLAINED`), balanced or not: which joint an angle comes from |

Confounders come first, ranked by the product of their two links, with
`prefer` (the table's main grouping) winning among them. The stratum is
chosen on a fixed sample of 10,000 rows of a longer table
(`STRATUM_SCREEN_ROWS`), and the levels are compared on the scans' fixed
sample of 50,000 (`sampled` gives the table's rows when they were). Each level holding
two or more groups gets its own comparison; a level holding one group is
named (`single`). `retained(pooled)` is the gap within the levels as a share
of the pooled one, weighting each level by its size: the difference between
the same two groups' means or rates, or the share of variance explained for
more groups. `overturns(pooled)` is true when a clear pooled difference runs
the other way in a level, or shows clearly in none and keeps less than half
its size there (`RETAINED_SHARE`): a gap of the same size in levels too small
to confirm it still stands. `describe_stratified` writes the sentences,
saying when the gap shrinks below three quarters of its size
(`SHRUNK_SHARE`), and `stratified_lead` a headline: "The gap in Shaking by
Gravel comes from Landslide: within Landslide 0 it does not hold, ...".
`stratified_rates(dataframe, outcome, grouping, candidates, entity=None,
prefer=None)` does the same for a rate: the stratum must be tied to the
grouping and go with the outcome too (Cramér's V of 0.3 or more,
`STRATUM_OUTCOME`), as kidney disease is behind both hypertension and
anaemia. Two groupings tied at V of 0.99 or more are one split recorded twice
and are never strata.

`gain_within(frame, measure, predictors, grouping)` gives R² of the measure on
the grouping's levels alone and with the predictors added (statsmodels OLS):
the difference is what the predictors explain within the groups.

### Describing it

`describe_comparison` writes the statement, `comparison_calculation` the
calculation, `lead_sentence` a one-sentence headline for a clear difference,
`effect_clause` a short "Hedges' g = 0.50, p < 0.001", and `comparison_table`
one row per group (the centre, n, the other centre and the SD). P-values read
"p < 0.001", "p = 0.032" or "p > 0.99".

## Scanning every reading

`scan_differences(dataframe, readings, groupings, entity=None, axis=None, skip=())`
compares each reading across each grouping and returns a `DifferenceScan`.
Each reading is reduced to one value per entity as in `compare_groups`, and a
grouping whose groups hold the same entities is compared within entities by
the same tests, listed in `paired`.

- P-values are Holm-adjusted over every test run, so the chance of naming any
  difference that isn't there stays at 5%. `holm_adjust` gives the figures of
  `statsmodels.stats.multitest.multipletests(method="holm")`, which it does not
  call because that branch forces a full garbage collection on every call.
- Each reading against a grouping of ordered numeric levels also gets a trend
  test, counted among the tests.
- A difference is named when it is `clear` by the same rules as a single
  comparison, after the adjustment, or when its trend is clear. The named
  ones come largest first, ranked by eta squared, epsilon squared on ranks or
  ρ² for a trend, with those resting on under 20% of the table's rows
  (`WELL_COVERED`) listed after the rest. Each comparison carries its
  `coverage`, and its headline sentence says when few rows record both
  columns. The largest effect is kept even when nothing is clear, for a null
  result.
- A two-valued grouping that only marks where the reading hit its cap (a
  censoring flag set exactly on the rows at 3,600 s) is left out, since it
  separates the groups by definition.
- A reading with a single value in each group is another code for the
  grouping (a 0/1 flag behind a condition label) and is left out.
- A reading whose groups explain 99% or more of its variance
  (`ATTRIBUTE_EXPLAINED`) barely varies within a group: it describes the
  group (a station's battery voltage, a reef's depth) rather than its
  members, and is left out as an attribute (`attributes`).
- A grouping that holds the reading by definition is left out (`defined`,
  with the reason from `built_in_grouping(dataframe, reading, grouping)`): a
  group holding 10% or more of the values (`CONSTANT_GROUP_SHARE`) pinned at
  the reading's floor or ceiling while other groups vary, as days coded "no
  precipitation" hold an amount of 0; a grouping whose name extends the
  reading's (`RSKF` beside `RSK`) and explains half its ranks
  (`NAMED_GROUPING_SHARE`), or shares a name stem with it (`trophic_group`
  beside `trophic_level`) and explains 80% of them; a grouping named as a grade of an
  index (`AQI_Level`, a risk band; `INDEX_GRADE_WORDS`) that explains 80% of
  them (`INDEX_GROUPING_SHARE`); or one that explains all of them, one
  cluster of values per group.
- Coordinates and compass directions are never compared (`placed`): a
  coordinate places a row, and the mean of an angle means nothing.
- `skip` holds pairs reported elsewhere, such as the measure across the
  grouping column, which has its own card, and pairs related by construction.
- A table with more than 300 readings (`MAX_SCAN_COLUMNS`), such as a gene
  expression matrix, is a screen. Every reading is tested, up to 20 million
  cells (`SCREEN_CELLS`, then the most variable readings), and the p-values
  are Benjamini-Hochberg adjusted instead (`benjamini_hochberg`, by
  `statsmodels.stats.multitest.multipletests(method="fdr_bh")`), which keeps
  the expected share of false findings among those named at 5%. Holm's guard
  against any false finding would hide nearly every true difference in a
  screen. `passing` counts the tests with an unadjusted p below 0.05, to set
  beside the 5% that chance alone would give.
- Tables longer than 50,000 rows (`SCAN_ROWS`) are scanned on a fixed random
  sample of that many rows. A difference large enough to report shows in it.

`variance_explained(dataframe, measure, groupings, entity=None, axis=None)` gives the
share of the measure's variance each grouping explains on its own (eta
squared), with one value per entity in each group when `entity` is given. In a
soil survey, depth explains 76% of calcium and contour 5%. As in a comparison,
groups with fewer than two values are left out, so a grouping of mostly single
sites does not explain everything by construction.
`combined_groupings` finds a grouping that is the combination of two others
(each level of a cell code such as `T0` is exactly one contour and one depth);
it explains at least as much as either factor, which says nothing about which
of them matters. `nested_within(dataframe, finer, coarser)` asks whether the
levels of one grouping each occur with only some levels of another: half or
more of the cells expected to hold five or more rows are empty
(`NESTED_EMPTY_SHARE`), as when concentrations 3 to 12 were used only with the
blue dyes. Across such a grouping the two can't be told apart.
`item_attributes(dataframe, readings, items)` finds the readings that never
vary within each value of an item column (a species, a product) on 99% of
the items recorded twice or more (`ITEM_FIXED_SHARE`): a species' trophic
level describes the species, so comparing it record by record weighs each
species by how often it was recorded.
`date_part(values, dates)` says which part of the date a numeric column
repeats ("month", "year", "day" and so on) on 99% of rows: grouping by it
shows the season or the trend, not a factor.

### Groupings set by the readings

`grouping_rules(dataframe, readings, groupings, date=None)` returns a
`GroupingRule` for each grouping the readings set row by row. Comparing those
readings across it only restates how it was made, so the brief states the
rule instead.

| Kind | Rule | Statement |
|---|---|---|
| `bands` | The best cut points of one reading, with the groups ordered by median, put 99.5% of rows (`RULE_SHARE`) in their own group's band, and neighbouring bands meet: no more than one step of the reading, or 2% of its spread (`RULE_GAP`), lies between them | "Farm_Size is set by Farm_Size_Acres: Small up to 10, Medium 11 to 50 and Large from 51, on 100% of rows" |
| `cut-offs` | The same within each level of another grouping, with cut points that move with it | "Result is set by Hemoglobin: Result 1 below 12.0 for Gender 0 and below 13.5 for Gender 1, on 100% of rows" |
| `largest` | Each level names one of several columns, and that column is the largest on 90% of rows (`LARGEST_SHARE`) | "critical names the largest of pm25, pm10, so2, co, o3 and no2 on 98.5% of rows" |

The cut points come from a dynamic programme over the sorted values, one pass
per group; cuts fall only between distinct values, so tied values stay
together, and a value rounded onto a cut may sit on the wrong side. Groups that
merely differ a lot overlap, or leave a gap where no rows lie, so they are not
taken for bands. A band of fewer than five rows (`RULE_MIN_BAND`) can't show
where its edge lies and is not held to meeting its neighbours. A reading needs
ten distinct values (`RULE_MIN_VALUES`). A grouping whose levels are unbroken
stretches of rows or of time (the phases of a run, logs stacked one after
another) can separate a reading almost perfectly for other reasons, so it
must separate it on every row. Readings are screened on 1,000 rows
(`RULE_SCREEN_ROWS`) and a likely rule is then checked on every row, or on the
same sample of 50,000 rows the scan uses.

## Comparing rates

`compare_rates(dataframe, outcome, grouping, entity=None)` compares how often a yes/no
outcome is "yes" across the groups of another column and returns a
`RateComparison`, or `None`. The outcome is a 0/1 column, a boolean, or a
two-valued text column of yes and no words (`outcome_event`).

| Part | How |
|---|---|
| Each group | The rate and n |
| Test | Pearson's chi-squared test on the groups-by-outcome table (`scipy.stats.chi2_contingency`, no continuity correction); for two groups with an expected count below 5, Fisher's exact test (`scipy.stats.fisher_exact`) |
| Effect | Cramér's V; under 0.1 is negligible |
| Gap | The highest rate minus the lowest, among groups of 5 or more rows, with Newcombe's 95% interval (`statsmodels.stats.proportion.confint_proportions_2indep`) |

The verdicts are those of a group comparison. `describe_rates` writes "The
share of rows with Result 1 is 55.7% in Gender 1 (n = 740) against 30.5% in
Gender 0 (n = 681): a difference of 25.1 percentage points (95% CI 20.1 to
30.0)" and `rate_calculation` the calculation.

With `entity`, an outcome that is the same on every row of a unit within a
group (a cell line's methylation flag on each of its drug rows) is an
attribute of the unit, so each unit counts once and n counts units: "Counting
each COSMIC_ID once, the share with Methylation = Y is 97.2% in Gene
Expression = Y (n = 904) against 57.7% in Gene Expression = N (n = 26)",
where the rows gave 227,885 and 4,784. Groups of fewer than 5 units are not
named, as in a group comparison. An outcome that differs between a unit's rows
is kept by row and flagged as not independent (`clustered`).

## A joint model

`fit_joint_model(frame, measure, predictors, limit=6)` fits the measure on up
to `limit` predictors at once (`MAX_MODEL_TERMS`) by ordinary least squares and
returns a `JointModel`, or `None` with fewer than two usable predictors or
fewer than 10 distinct rows per predictor (`ROWS_PER_TERM`): a visit's
readings repeated on every species row of the visit are one row of evidence,
and a model with as many terms as distinct rows fits them exactly whatever
they hold. Ask uses it for "what predicts X".

- It uses the rows where the measure and every predictor are recorded, or a
  fixed random sample of 50,000 of them (`MODEL_ROWS`). `distinct` counts the
  distinct rows among them.
- Predictors are taken in the order given; one that does not vary, or that the
  earlier ones reproduce exactly, is dropped.
- Coefficients with 95% intervals and p-values, R², adjusted R² and the
  residual SD come from `statsmodels` OLS. Each `ModelTerm` also carries its
  standardised coefficient, the predictor's own Pearson r with the measure on
  the same rows, and its variance inflation factor from
  `statsmodels.stats.outliers_influence.variance_inflation_factor`.
- `collinear` names the predictors with a VIF above 5 (`MAX_VIF`): largely a
  mix of the others, so their separate coefficients are unstable.
- `reversed` names the predictors whose clear coefficient points the other way
  from their own correlation of 0.3 or more: the pairwise figure partly
  stands for the readings that travel with it.
- `clear` names the predictors whose coefficient stays clear of zero with the
  others held fixed.
- `exact` is true for an R² of 0.995 or more (`EXACT_FIT`), or a coefficient
  whose interval has no width: the measure is a fixed combination of its
  predictors (an annual mean and its seasons), an identity rather than a
  relationship.

A coefficient is an association with the others held fixed, not a cause.

## A logistic model

`fit_logistic_model(frame, outcome, predictors, limit=6)` fits a yes/no
outcome on up to `limit` predictors by logistic regression
(`statsmodels.discrete.discrete_model.Logit`) and returns a `LogisticModel`:
each predictor's odds ratio per standard deviation, with the others held
fixed, its 95% interval and p-value, the rows and the cases of the outcome,
and McFadden's pseudo R². It needs 10 cases of the rarer outcome per
predictor (`EVENTS_PER_TERM`), and returns `None` when the fit does not
converge or the predictors separate the outcomes completely or all but
completely (a log odds ratio per standard deviation beyond 5,
`SEPARATED_LOG_ODDS`): then there is no estimate to report.

## Relationships

`scan_relationships(dataframe, columns, measure=None, built=None)` ranks what
relates to the measure, then the strongest pairs among the other columns, and
returns a `RelationshipScan`.

`construction(dataframe, columns, measure=None, items=(), known=())` screens
the columns once for pairs related by construction and keeps the correlations
it found them from, so the brief's comparison and relationship cards share one
screen (`built`). `items` are further numeric columns, such as rating items
read as codes, that a score may be built from; they are tried as parts but
never as derived columns. `known` holds derivations the caller already knows,
such as the row mean ADA adds to a matrix table. Tables longer than 50,000
rows are screened on a fixed random sample of that many rows (`SCAN_ROWS`),
the same one the comparisons use. Compass directions (`is_angle`: a direction
word in the name and values from 0 to 360) are left out.

### Screening every pair at once

Pearson r and the number of complete rows for every pair come from a few
matrix products (`pairwise_correlations`), each pair over the rows where both
columns are recorded. Columns are standardised first so very large readings
keep their precision. Spearman r is the same calculation on ranks taken once
per column; with missing values that is a close screen rather than the exact
figure, and the pairs that are reported are recomputed exactly. A pair needs
12 complete rows (`MIN_CORRELATION_PAIRS`). Tables with more than 300 numeric
columns are scanned on the measure and the most variable others.

Pairs are ranked by the larger of |Spearman r| and |Pearson r without the
pair's most influential record|, so one stray record can neither make nor hide
a relationship. The p-value of each pair is for that figure.

### Pairs related by construction

These pairs are set aside and listed in the calculation:

| Kind | Rule |
|---|---|
| One quantity twice | \|r\| of 0.99 or more with ranks that agree, still 0.99 without the pair's most influential record, and either 90% of the values identical (`IDENTICAL_SHARE`) or one a linear function of the other on 99% of rows, to the columns' own rounding: a temperature in °C and in °F. These form one family; a log or square root of a column joins its family too. A taxon holding half of all reads can make two count columns look identical; without it they are not |
| Near-duplicates | \|r\| of 0.99 or more with ranks that agree, without being one quantity: two replicate samples, two counters that rise together. They are set aside from the ranking (`"near"`) but stay separate columns everywhere else |
| Statistics of one reading | Both summarise the same reading (`summary_families`): `pollutant_min`, `pollutant_avg` and `pollutant_max`, or a table's `mean`, `std` and quartiles of each row's values, or the same test over numbered windows (`OBS1_bilirubin_max`, `OBS2_bilirubin_min`). A family needs a mean or median, or a minimum and a maximum, and the minimum must stay below the maximum on 95% of rows |
| Built from the other | One column is built from others that include the second column (see below), or both are built from the same items (one's items contain the other's) |
| Shares of a total | Both belong to a block of columns that add up to a fixed total, such as phylum shares of 100 |
| Cap flags | A two-valued column set exactly on the rows where the other sits at its maximum or minimum |
| Same stem | The names share a word that says what was measured (not a unit, a bracketed note or a word such as `total` or `mean`), and \|Spearman r\| and \|Pearson r\| without the most influential record are both above 0.95 (`SHARED_STEM_CORRELATION`), as `MaxTemp` and `MeanTemp` |
| Coordinates | Both are map coordinates, which describe where the rows were taken |

Pairs within one family are reported once. `set_aside` lists the measure's
columns left out and why, so an answer can name them.

### Finding derived columns

`find_derivation(frame, column, candidates)` asks whether a column is a fixed
combination of a few others: a sum (`CorrectTotal = Correct1 + Correct2 +
Correct3 + Correct4`), a mean, a difference, or another fixed combination.

1. Candidates come strongest-correlated first, and are kept while at least 12
   rows, and 10% of the column's rows, hold all of them; a part recorded on
   only some rows still shows the identity where it is recorded.
2. A first fit with every candidate rules out most columns at once.
3. Otherwise parts are added one at a time, each the candidate closest to
   what the others leave unexplained, up to 8 (`MAX_PARTS`).
4. A fit is least squares refitted three times on the rows it matches best,
   or the same weights snapped to the nearest simple fraction (0.49 and 0.52
   become a half each). The simple formula is kept unless least squares
   matches more than 1% more rows (`SIMPLE_MARGIN`).
5. It counts when it reproduces the column on 90% of rows (`DERIVED_SHARE`)
   to within 1% of the column's spread, measured on its middle half
   (`DERIVED_TOLERANCE`), or within half the column's own step when it was
   rounded after it was computed, plus each part's step times its weight
   when the parts were rounded or truncated before they were combined (a mean
   of whole-number sensor readings), but never more than 10% of the spread.
   It must also explain at least 99% of the column's variance, or of the
   variance of the rows it matches when those span the column's range, so a
   few rows that break the formula (shares no longer adding up once one was
   clipped at zero) can't hide it.
6. A free combination, whose weights are not ±1 or 1/n, must hold on 99% of
   rows (`COMBINATION_SHARE`), since it has more freedom to fit.
7. A fit needs three distinct rows per term, so a few rows can't be matched
   by chance, however often they repeat.

A free combination that a rearrangement makes a plain sum or mean of the
others is read that way round: "70 = 2 × 60 − 50" is named "60 = mean of 50
and 70", the column filled in from its neighbours.

A fit whose weights are all −1 with a constant is a `complement`: the column
and its parts add up to a fixed total ("Diet_Silage, Diet_Grains and
Diet_Forage add up to 100"). The rows that break the total are counted on
every row and named in a note.

Further searches find columns built in other ways:

| Function | Finds |
|---|---|
| `find_product(frame, column, candidates)` | A product or ratio of powers of up to 3 columns (`MAX_PRODUCT_PARTS`), with powers from −3 to 3 and halves (`EXPONENTS`) and a constant factor: "body_mass_index = weight_kg ÷ height_cm² × 10,000". On logs a product is a sum, so it is `find_derivation` on the logs of positive columns. Tables of more than 100 positive readings (`PRODUCT_TARGETS`) are skipped |
| `find_transform(frame, column, source)` | A log10, natural log, log2, log of one plus, square root, square or reciprocal of another column (`TRANSFORMS`), checked for pairs whose ranks agree to 0.995 (`TRANSFORM_RANKS`): "lcond = log10(COND)" |
| Rates and lags | On rows in table order, a column that is a fixed multiple of another's row-to-row change on 90% of rows ("velocity = 100 × the row-to-row change in angle"; the rows where a new curve starts break it), and a combination that takes a part from the row before ("D-J-F = mean of JAN, FEB and DEC (row before)"). Tables of up to 60 readings (`ORDERED_TARGETS`) are searched, on their first 2,000 rows |
| Built from built columns | A sum or mean of columns that are built themselves: "metANN = mean of D-J-F, M-A-M, J-J-A and S-O-N" |
| Cut off at another | A column that never exceeds another on 99% of rows (`CAPPED_BELOW`) and equals it on at least 20% (`CAPPED_EQUAL`): a time to recurrence that equals the follow-up time wherever no recurrence was seen |

A column built from a built column is built from that column's parts too, so
an acceleration made from a velocity made from an angle relates to the angle
by construction.

Two summaries have no fixed formula:

| Function | Finds |
|---|---|
| `find_row_extreme(frame, column, candidates)` | A row maximum or minimum: never beyond any of its parts (bar 1% of rows), equal to one of them on 90% of rows, using the parts each row has. Each part must supply the value on at least two rows. A daily index reported as the worst of six sub-indices is one |
| `find_standardised(frame, column, candidates, groupings)` | A z-score: mean 0 and SD 1 overall (within 0.05), and within 0.005 in each group of a grouping, where chance can't produce it (`STANDARDISED_EXACT`). Any grouping made of whole groups passes too, so the finest one is named. The source is named when a candidate rescales to the column on 90% of rows |

A sum, mean or row maximum that reads 0 where every part is blank was filled
in, not measured. `Construction.masked` sets those zeros to missing, and the
derivation counts them (`blank_rows`).

An exact identity reads either way round (an item is the total minus the other
items), so a column is matched against the columns before it, where computed
columns are usually appended. A column named as a total or mean (`total`,
`sum`, `mean`, `average`, `overall`, ...) may also be the sum or mean of
columns after it. A total can be rebuilt from other totals, so each derivation
is then retraced to the columns that are not derived themselves, which names
the items it was made from.

### Checks on one correlation

| Helper | Returns |
|---|---|
| `most_influential(x, y)` | The record whose removal changes r most, and r without it, from one pass over the data. A change of more than 0.2 (`INFLUENCE_CHANGE`) means the relationship rests on that record |
| `without_extremes(x, y)` | r after dropping records more than 5 robust SDs out in either column (`EXTREME_Z`), when there are between 1 and 5% of them |
| `far_out(x, y)`, `longest_stretch(marked)` | Which records lie more than 5 robust SDs out, and the longest unbroken run of them: a phase of a run rather than a few stray values |
| `within_group_correlation(frame, x, y, grouping)` | r after removing each group's means, with each group's own r and rows and their range. Groups that differ in both columns can produce a pooled r of the opposite sign, or hide a stronger one |
| `repeated_correlation(frame, first, second, unit)` | The relation with the unit as the evidence when several rows come from each unit. When both columns change within units: each unit's least-squares slope of `second` on `first` (units with 3 or more rows), their mean with a t interval and a one-sample t-test across units (needs 5 units), how many are positive, and the repeated-measures r, the correlation of the values centred on each unit's means (Bakdash and Marusich 2017). When one column stays fixed within units: Pearson r over the units' means, one row per unit. `None` when no unit has two rows |
| `correlation_interval(r, n)` | The 95% interval by the Fisher transform |

## Changing this

Every rule here must make sense for any table: a sales export as much as an
assay. A new rule needs a test on a small synthetic frame in
`tests/test_compare.py` for the case where it fires and one where it doesn't.
