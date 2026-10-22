# Comparing groups and finding relationships

**Code:** `ada/compare.py`
**Tests:** `tests/test_compare.py`

Two questions come up with almost any table: does a reading differ between
groups, and what moves with it? This module answers both with the sample, the
test and the effect size behind the answer. The brief's `leader`,
`differences` and `relationship` cards (see [Evidence](evidence.md)) and the
comparison questions in [Ask ADA](ask-ada.md) all use it. scipy supplies the
t, F and chi-squared distributions; the statistics themselves are computed
here, so a one-reading comparison and a scan of hundreds of readings give the
same numbers.

## Comparing a reading across groups

`compare_groups(dataframe, measure, grouping, entity=None)` returns a
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

Both rank tests use the normal approximation with ties corrected, so a scan
and a single comparison agree.

### The verdict

| Verdict | When | Wording |
|---|---|---|
| `clear` | p below 0.05 and at least 1% of the variance explained | "X is higher in A than in B", or "X differs across G" with the highest and lowest group |
| `negligible` | p below 0.05 but under 1% of the variance explained (`NEGLIGIBLE_EXPLAINED`) | "differs only by a negligible amount"; with enough records a trivial gap passes any test |
| `none` | p of 0.05 or more | "shows no clear difference", with the interval or the range of the groups; never a ranking |

For two groups compared by means, p below 0.05 is the same as a 95% interval
for the difference that excludes zero. Effect sizes are named by the usual
cut-offs: Hedges' g of 0.2, 0.5 and 0.8 (small, medium, large) and a share of
variance of 0.01, 0.06 and 0.14.

### One value per subject

Repeated readings of one subject are not independent evidence. With `entity`
(a subject, patient or sample column), each entity contributes one value per
group: its mean, or its median for a reading that is skewed or capped. The
statement says "Each Bird counts once per group". When the same entity appears
in more than one group (each subject saw both objects), the groups are not
independent and the statement says the p-value is only a guide.

`entity_column(dataframe, roles)` picks the entity: the identifier, or a column
named like one (`is_entity_name` in `ada/schema.py`, such as `Subject` or
`Patient ID`), provided some value repeats.

### Describing it

`describe_comparison` writes the statement, `comparison_calculation` the
calculation, `lead_sentence` a one-sentence headline for a clear difference,
`effect_clause` a short "Hedges' g = 0.50, p < 0.001", and `comparison_table`
one row per group (the centre, n, the other centre and the SD). P-values read
"p < 0.001", "p = 0.032" or "p > 0.99".

## Scanning every reading

`scan_differences(dataframe, readings, groupings, entity=None, skip=())`
compares each reading across each grouping and returns a `DifferenceScan`.

- P-values are Holm-adjusted over every test run, so the chance of naming any
  difference that isn't there stays at 5%.
- A difference is named when its adjusted p is below 0.05 and it explains at
  least 1% of the variance. The named ones come largest first, ranked by eta
  squared, or epsilon squared on ranks. The largest effect is kept even when
  nothing is clear, for a null result.
- A two-valued grouping that only marks where the reading hit its cap (a
  censoring flag set exactly on the rows at 3,600 s) is left out, since it
  separates the groups by definition.
- A reading with a single value in each group is another code for the
  grouping (a 0/1 flag behind a condition label) and is left out.
- A reading whose groups explain 99% or more of its variance
  (`ATTRIBUTE_EXPLAINED`) barely varies within a group: it describes the
  group (a station's battery voltage, a reef's depth) rather than its
  members, and is left out as an attribute (`attributes`).
- Coordinates and compass directions are never compared (`placed`): a
  coordinate places a row, and the mean of an angle means nothing.
- `skip` holds pairs reported elsewhere, such as the measure across the
  grouping column, which has its own card, and pairs related by construction.
- Wide tables are scanned on their 300 most variable readings
  (`MAX_SCAN_COLUMNS`), and tables longer than 50,000 rows (`SCAN_ROWS`) on a
  fixed random sample of that many rows. A difference large enough to report
  shows in either.

`variance_explained(dataframe, measure, groupings)` gives the share of the
measure's variance each grouping explains on its own (eta squared). In a soil
survey, depth explains 76% of calcium and contour 5%.
`combined_groupings` finds a grouping that is the combination of two others
(each level of a cell code such as `T0` is exactly one contour and one depth);
it explains at least as much as either factor, which says nothing about which
of them matters. `nested_within(dataframe, finer, coarser)` asks whether the
levels of one grouping each occur with only some levels of another: half or
more of the cells expected to hold five or more rows are empty
(`NESTED_EMPTY_SHARE`), as when concentrations 3 to 12 were used only with the
blue dyes. Across such a grouping the two can't be told apart.
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
| Near-duplicates | \|r\| of 0.99 or more with ranks that agree, and still 0.99 without the pair's most influential record: one quantity twice, such as a temperature in °C and in °F. A taxon holding half of all reads can make two count columns look identical; without it they are not |
| Built from the other | One column is built from others that include the second column (see below), or both are built from the same items (one's items contain the other's) |
| Shares of a total | Both belong to a block of columns that add up to a fixed total, such as phylum shares of 100 |
| Cap flags | A two-valued column set exactly on the rows where the other sits at its maximum or minimum |
| Same stem | The names share a word that says what was measured (not a unit, a bracketed note or a word such as `total` or `mean`), and \|Spearman r\| and \|Pearson r\| without the most influential record are both above 0.95 (`SHARED_STEM_CORRELATION`), as `MaxTemp` and `MeanTemp` |
| Coordinates | Both are map coordinates, which describe where the rows were taken |

Pairs within one near-duplicate family are reported once. `set_aside` lists
the measure's columns left out and why, so an answer can name them.

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
7. A fit needs three rows per term, so a few rows can't be matched by chance.

A fit whose weights are all −1 with a constant is a `complement`: the column
and its parts add up to a fixed total ("Diet_Silage, Diet_Grains and
Diet_Forage add up to 100"). The rows that break the total are counted on
every row and named in a note.

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
| `correlation_interval(r, n)` | The 95% interval by the Fisher transform |

## Changing this

Every rule here must make sense for any table: a sales export as much as an
assay. A new rule needs a test on a small synthetic frame in
`tests/test_compare.py` for the case where it fires and one where it doesn't.
