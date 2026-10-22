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
- `skip` holds pairs reported elsewhere, such as the measure across the
  grouping column, which has its own card.
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
of them matters.

## Relationships

`scan_relationships(dataframe, columns, measure=None)` ranks what relates to
the measure, then the strongest pairs among the other columns, and returns a
`RelationshipScan`.

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
| Near-duplicates | \|r\| of 0.99 or more with ranks that agree: one quantity twice, such as a temperature in °C and in °F |
| Built from the other | One column is a fixed combination of others that include the second column, or both are built from the same items (one's items contain the other's) |
| Cap flags | A two-valued column set exactly on the rows where the other sits at its maximum or minimum |
| Same stem | The names share a word that says what was measured (not a unit, a bracketed note or a word such as `total` or `mean`) and \|r\| is above 0.95 (`SHARED_STEM_CORRELATION`), as `MaxTemp` and `MeanTemp` |

Pairs within one near-duplicate family are reported once.

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
4. A fit is least squares refitted on the rows it matches best, or the same
   weights snapped to the nearest simple fraction (0.49 and 0.52 become a half
   each), whichever matches more rows.
5. It counts when it reproduces the column on 90% of rows (`DERIVED_SHARE`)
   to within 1% of the column's spread, measured on its middle half
   (`DERIVED_TOLERANCE`), or within half the column's own step when it was
   rounded after it was computed (but never more than 10% of the spread), and
   explains at least 99% of its variance.
6. A fit needs three rows per term, so a few rows can't be matched by chance.

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
| `within_group_correlation(frame, x, y, grouping)` | r after removing each group's means, with the range of the per-group r. Groups that differ in both columns can produce a pooled r of the opposite sign |
| `correlation_interval(r, n)` | The 95% interval by the Fisher transform |

## Changing this

Every rule here must make sense for any table: a sales export as much as an
assay. A new rule needs a test on a small synthetic frame in
`tests/test_compare.py` for the case where it fires and one where it doesn't.
