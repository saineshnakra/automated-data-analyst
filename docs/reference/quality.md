# Data checks

**Code:** `scan` in `ada/quality.py`
**Tests:** `tests/test_quality.py`

Looks through every numeric column, not only the measure, for values that
should not drive a result: readings outside what the column's name allows,
codes standing in for missing readings, values piled up at a limit, filled-in
gaps, values far from the rest, copied columns, and gaps that follow a pattern.
Each issue says how many cells or rows it covers, gives examples and names the
rule that fired.

The checks change nothing, with one exception: cleaning sets clear
missing-value codes to missing in every numeric column (see
[Codes set to missing](#codes-set-to-missing)), and the checks list them as
such. Everything else is reported for the reader to decide on.

## Where the results appear

| Place | What it shows |
|---|---|
| `Analysis.quality()` | Every issue as a DataFrame: `Check`, `Columns`, `Rows`, `Share %`, `Detail`, `Rule`, `Examples`. Empty when nothing was flagged |
| `Analysis.to_dict()["data_checks"]` | The same rows, JSON-safe |
| The brief's **Data checks** card | The three leading issues about values and the two leading ones about gaps, with counts, and how complete the table is |
| The brief's summary | A "Check these first" line when an issue touches the measure or the group the headline names |
| The leading-group card | When the leading group's average rests on flagged values, a sentence saying so, with the group's median |
| The app's Data room | A **Data checks** table |

The card, the summary line and the leading-group sentence are described in
[Evidence and recommendations](evidence.md#data-checks).

## Which columns are checked

Numeric columns that hold readings. Left out:

- the identifier, and any column named as a key or a row label (`ID`, `No`,
  `Patient`) whose values are whole numbers; a decimal column under such a name,
  such as nitric oxide as `NO`, is checked
- experimental-design labels (`Replicate`, `Plate`, `Well`, ...)
- whole numbers unique on every row (a row counter)

A column of 15 or fewer whole numbers holds codes, unless its name bounds it (a
humidity, a percentage), and so does any column in `ColumnRoles.factors`. Code
columns get only the survey-code check.

## The checks

Issues are ordered by check, in this order, and within a check the measure's
come first.

| Check | Name in the table | Fires when |
|---|---|---|
| `impossible` | Impossible values | A value is outside what the name allows (see below), or a column that is otherwise above zero has a few negatives the size of its usual values (a sign slip) |
| `sentinel` | Sentinel codes | A code value stands alone far from the rest, or a value in the outer 1% occurs far more often than its neighbours |
| `group_level` | Group out of scale | One group's median of the measure is more than 5 times every other group's |
| `equal_columns` | Copied values | Two columns hold the same non-zero value in half or more of the rows, overall or within one group |
| `min_max` | Max below min | A max-named column is below its min-named sibling, or below the plain one |
| `extreme` | Extreme values | A value is more than 10 robust standard deviations from the median and cut off from the rest by a gap |
| `imputed` | Probable fill values | One non-round value holds 10% or more of a column |
| `cap` | Cap or floor | One value piles up at the maximum or the minimum |
| `survey_code` | Survey codes | Codebook values such as 7, 9, 77 or 99 sit above a small set of answer codes |
| `missing_by_group` | Missing by group | Some groups lack a column far more often than the rest |
| `missing_dependency` | Missing by value | A column's gaps fall where another column is high, or low |
| `missing_block` | Missing together | Two or more columns are missing on the same rows |
| `missing` | Mostly missing | A column is more than 20% missing |

### Impossible values

Bounds come from words in the column name:

| Name | Range |
|---|---|
| `%` as a unit (not after a number, as in the quantile label `25%`), `percent`, `pct`, `coverage`, `humidity`, or a lab abbreviation ending in a lower-case `p` (`LYMp`, `NEUTp`) | 0 to 100 |
| `pH` | 0 to 14 |
| A normalised-difference index (`NDVI`, `NDWI`, `NDMI`, `GNDVI`, `NDRE`, `NBR`) | -1 to 1 |
| Latitude, longitude | ±90, ±180 |
| `direction`, `heading`, `bearing`, `azimuth` | 0 to 360 |
| A size, an amount of matter or a count (`age`, `weight`, `height`, `yield`, `resolution`, `concentration`, `rainfall`, `speed`, `pressure`, ...), or a unit such as `mg/kg`, `cmol(+)/kg`, `cells/mL`, `ppm` | 0 and up |

A word that lets a value go below zero or past 100 (`change`, `growth`,
`difference`, `net`, `balance`, `profit`, `margin`, `temperature`,
`elevation`, `score`, ...) removes the bound. Money and sales quantities
(`revenue`, `amount`, `units`, `quantity`) have no lower bound, because refunds
and returns are negative.

When more than 20% of a column is outside its bound, the column is on another
scale than the name suggests, and nothing is flagged.

A column without a bound is checked for sign slips: at most 1% of its values
are negative, and every negative is at least as large as the 5th percentile of
the positive values, with nothing between it and zero. A tail that runs down
through zero (cold days in °F, a vegetation index over bare soil) is a real
range, not an error.

### Sentinel codes

A code is a run of nines (`99`, `999`, `999.9`, `-9999`), a repeated digit of
four or more (`1111`, `8888`) or a large negative power of ten (`-1000`,
`-100000`). It is flagged when it stands alone, with no other value within a
factor of two of it, and is either more than 10 robust standard deviations from
the median of the other values or negative in a column that is otherwise not.
`99` among bacterial counts that run past 1,000 is a count; `999.9` among wind
speeds under 20 is a code.

A converted blank shows up differently: a value in the outer 1% that occurs at
least 10 times as often as the two values on each side of it, and holds a fifth
of that tail. A blank Fahrenheit reading becomes 0 °F, which is -17.78 °C; in a
temperature column the issue says so. A column with more than two such values
has a texture instead (a counter that plateaus, readings rounded to quarters),
and none is flagged. A value rounder than its neighbours (5.8 among 5.79 and
5.81) is not counted.

The same code in several columns is one issue.

### Extreme values

Distance from the median is measured in robust standard deviations: the scaled
median absolute deviation, or the interquartile range divided by 1.349 when
more than half the values are identical. A value counts when it is more than
10 of them out (`EXTREME_Z`) and also cut off from the rest: walking outwards
from the bulk, a gap of at least 3 robust standard deviations (`GAP_SPREADS`)
opens before it, and it is at least twice as far from the median as the last
value before the gap. A continuous tail, such as arctic stations among tropical
ones or the brightest noons in a column of solar readings, is not cut off.

An essentially non-negative column is measured on a log scale of its positive
values when that makes its bulk more symmetric (concentrations, latencies,
counts); only high values count on that scale. When more than 2% of a column
qualifies, it is a second cluster or a heavy tail rather than stray errors,
and nothing is listed.

A fill value or a pile-up at a limit is left out when the median and spread
are measured, since it would sit in the bulk and shrink the spread.

Extreme values on the same rows in several columns are one issue: one record,
or one batch of records, is wrong or on another scale across all of them.

### Caps, floors and fill values

A cap or floor is one value at the maximum or minimum with at least 3 rows and
3 times the rows of each of its three nearest values, that either holds 3% or
more of the column or is its most frequent value. Zero is never a floor, and
neither is 1 in a column of whole numbers: counts start there.

A fill value is the most frequent value, holding 10% or more of the column and
written to more decimal places than nine in ten of the column's values (46.1526
among readings like 39.2). Fill values recurring on the same rows in several
columns are one issue; the detail says when the value is close to the mean of
the other values, as mean imputation leaves it.

### Copied values and max below min

Pairs are screened on an evenly spaced sample of 5,000 rows, overall and within
each group of the grouping column, and counted on the full table. A copy within
one group is reported only when the two columns rarely agree elsewhere (10% of
rows or fewer). With more than 100 columns (`PAIR_COLUMNS`), such as a spectrum
or a panel of samples, only the measure is paired with the others, since every
pair would cost more than the rest of the scan.

Siblings are found by name: `max` and `min`, `maximum` and `minimum`, `high`
and `low`, `upper` and `lower`. With no min-named sibling, the max-named column
is compared with the plain one (`Maximum Wind Speed` with `Wind Speed`).

### Groups

With three or more groups of at least five values, a group whose median of the
measure is more than 5 times the next highest median is flagged
(`GROUP_LEVEL_RATIO`). With two groups, one being far higher is as likely the
effect being studied, so nothing is flagged.

### Missing data

Columns 5% or more missing are compared for overlap; those whose missing rows
overlap by 90% or more (Jaccard) form a block. A block's columns are reported
once, as a block, and not again one by one.

For each group of the grouping column (2 to 50 groups of 10 rows or more), the
missing share of each gappy column is sorted, and the groups are split at the
widest drop. A drop of 50 points or more is reported: "Foster Weather Station
never records Wet Bulb Temperature, against 0% for the other values". A block a
group explains entirely is not reported as a block too.

For a column or block more than 20% missing that no group explains, each
numeric column recorded on at least half the rows is ranked, and the chance
that a value from a missing row outranks one from a recorded row is measured
(the area under the ROC curve) on the 5,000-row sample. Beyond 0.85 or below
0.15, the issue names the column and the two medians: "Acidity is missing
mostly where pH is high: median pH 6.19 where missing against 4.99 where
recorded".

## Codes set to missing

`blank_codes` runs during cleaning, before roles are detected or anything is
computed, on every numeric column except keys and layout numbers. It sets two
kinds of code to missing:

- a sentinel code that stands alone and is either negative in a column that
  is otherwise not, or at least ten times the 99th percentile of the other
  values: `-999` among pressures near 950, `999.9` among wind speeds below 20
- one negative value that three or more columns (`SHARED_CODE_COLUMNS`), each
  otherwise never negative, hold on the same rows: `-1` in every sensor channel
  of the rows where a read failed. A column of a few whole-number codes (`-1`,
  `0`, `1` for disagree, neutral and agree) is a scale, and is left alone

A cleaning note per column says how many cells and which value. The issues
`blank_codes` returns are kept on the cleaning report (`blanked_codes`), and
`scan(frame, roles, blanked=...)` ranks them with the rest, so
`Analysis.quality()` and the app's Data checks still list each code, with a
detail saying cleaning set those cells to missing. A value that only looks
like a code, such as a price of 9,999 among prices up to 3,000, stays, and so
does a converted blank found by its frequency (`-17.78` in a temperature
column), which the checks report for the reader to judge.

## Speed

Each column is sorted once and every check works on whole arrays; pairs and
missingness dependencies are screened on a sample. A table of 250,000 rows by
30 columns takes well under a second, and one of 1,100 rows by 7,900 columns
under two.

## Changing this

A new check needs a function returning `QualityIssue`s with a `detail` sentence,
a short `summary`, the `rule` that fired and, for a check on individual values,
the flagged rows in `cells`; an entry in `CHECKS`; and tests for a case where it
fires and a business table where it must not. A rule that depends on a column
name belongs in `name_bounds` or the word lists beside it, never in a check.
