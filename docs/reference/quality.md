# Data checks

**Code:** `scan` in `ada/quality.py`
**Tests:** `tests/test_quality.py`

Looks through every numeric column, not only the measure, for values that
should not drive a result: readings outside what the column's name or a
physical limit allows, codes standing in for missing readings, values piled up
at a limit, filled-in gaps, values far from the rest, second clusters, swapped
and copied columns, sensors stuck on one value, and gaps that follow a
pattern. It also looks at the table as a whole, for copied rows and for
columns so flat and unrelated that the data may be simulated. Each issue says
how many cells or rows it covers, gives examples and names the rule that
fired.

The checks change nothing in the table, with one exception: cleaning sets
clear missing-value codes to missing in every numeric column (see
[Codes set to missing](#codes-set-to-missing)), and the checks list them as
such. Everything else is reported for the reader to decide on. The brief does
compute relationships and group comparisons on a copy with the cells that
can't be readings blanked (see
[Cells left out of statistics](#cells-left-out-of-statistics)).

## Where the results appear

| Place | What it shows |
|---|---|
| `Analysis.quality()` | Every issue as a DataFrame: `Check`, `Columns`, `Rows`, `Share %`, `Detail`, `Rule`, `Examples`. Empty when nothing was flagged |
| `Analysis.to_dict()["data_checks"]` | The same rows, JSON-safe |
| The brief's **Data checks** card | The three issues about values and the two about gaps that cover the most rows, with counts, and how complete the table is |
| The brief's summary | A "Check these first" line when an issue touches the measure or the group the headline names, covers 10% or more of the table (copied rows, simulated-looking data), or covers a quarter or more of a column |
| The leading-group card | When the leading group's average rests on flagged values, a sentence saying so, with the group's median |
| The relationship, comparison and outlier cards | How many flagged cells were left out of their calculation |
| The app's Data room | A **Data checks** table |
| Ask ADA | "outliers in X" and "are there impossible values" list the flagged values; any answer that uses flagged values, or names a flagged group, adds a caution |

The card, the summary line and the leading-group sentence are described in
[Evidence and recommendations](evidence.md#data-checks).

## Which columns are checked

Numeric columns that hold readings. Left out:

- the identifier, and any column named as a key or a row label (`ID`, `No`,
  `Patient`) whose values are whole numbers; a decimal column under such a name,
  such as nitric oxide as `NO`, is checked
- experimental-design labels (`Replicate`, `Plate`, `Well`, ...)
- whole numbers unique on every row (a row counter)
- columns ADA adds itself, such as a matrix table's `Row mean (60 columns)`

A column of 15 or fewer whole numbers holds codes, unless its name bounds it (a
humidity, a percentage), and so does any column in `ColumnRoles.factors`. Code
columns get only the survey-code check.

## The checks

Issues about values come first, then issues about gaps. Within each, issues
are ranked by the rows (or cells) they cover, so a fill value in 61% of rows
comes before a single impossible value; ties follow the order below, the
measure's first.

| Check | Name in the table | Fires when |
|---|---|---|
| `generated` | Looks simulated | Most numeric columns are spread evenly between their extremes and no pair relates beyond chance |
| `duplicates` | Duplicate rows | 1% or more of the rows repeat an earlier row exactly, in a table with continuous readings |
| `swapped` | Swapped columns | On a block of consecutive rows, two columns hold values in each other's usual ranges |
| `impossible` | Impossible values | A value is outside what the name or a physical limit allows (see below), or a column that is otherwise above zero has a few negatives the size of its usual values (a sign slip) |
| `sentinel` | Sentinel codes | A code value stands alone far from the rest, or a value in the outer 1% occurs far more often than its neighbours |
| `group_level` | Group out of scale | One group's median of the measure is off from the others by about a unit conversion factor |
| `equal_columns` | Copied values | Two columns hold the same non-zero value in half or more of the rows, overall or within one group |
| `min_max` | Max below min | A max-named column is below its min-named sibling, or below the plain one |
| `cluster` | Second cluster | More than 2% of a column, and at least 10 values, lie cut off far from the rest |
| `extreme` | Extreme values | A value is more than 10 robust standard deviations from the median and cut off from the rest by a gap |
| `matrix_rows` | Rows off scale | In a matrix table, a row's overall level is far from every other row's |
| `imputed` | Probable fill values | One non-round value holds 10% or more of a column |
| `flatline` | Stuck readings | In time order, a reading sits at exactly zero for long stretches, or holds one value for a large share of the record |
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

When more than 20% of a column is outside a bound from its name, the column is
on another scale than the name suggests, and nothing is flagged.

A column without a bound is checked for sign slips: at most 1% of its values
are negative, and every negative is at least as large as the 5th percentile of
the positive values, with nothing between it and zero. A tail that runs down
through zero (cold days in °F, a vegetation index over bare soil) is a real
range, not an error.

#### Physical limits

A short table, `PHYSICAL_LIMITS`, gives the values a few common quantities can
take at all. The ranges are wide on purpose: a value outside one is an error,
not a rare case.

| Quantity | Names | Unit: range |
|---|---|---|
| Sodium | `Na`, `sodium` | mmol/L: 100 to 200 |
| Potassium | `K`, `potassium` | mmol/L: 1 to 10 |
| Chloride | `Cl`, `chloride` | mmol/L: 50 to 160 |
| Glucose | `glucose`, `GLU`, `FBS`, `RBS`, `sugar` | mg/dL: 10 to 3,000; mmol/L: 0.5 to 170 |
| Creatinine | `creatinine`, `creat`, `SCr` | mg/dL: 0.1 to 30; µmol/L: 10 to 2,700 |
| Albumin | `albumin`, `ALB` | g/dL: 1 to 7; g/L: 10 to 70 |
| Haemoglobin | `haemoglobin`, `hemoglobin`, `HGB`, `Hb` | g/dL: 3 to 25; g/L: 30 to 250 |
| Haematocrit | `haematocrit`, `HCT`, `PCV` | %: 10 to 75; L/L: 0.1 to 0.75 |
| Red cell count | `RBC`, `erythrocytes` | ×10¹²/L: 1 to 10 |
| Heart rate | `heart rate`, `pulse` | beats a minute: 20 to 300 |
| Body temperature | `body temperature`, `body temp` | °C: 25 to 45; °F: 77 to 113 |
| Wind speed | `wind speed`, `wind gust`, `WS` with a unit | m/s: up to 115; km/h: 415; mph: 260; knots: 225 |
| Air temperature | `air temperature`, `ambient temperature`, `outdoor temperature` | °C: -90 to 60; °F: -130 to 140 |

Words such as `serum`, `blood`, `level`, `max` or `mean` are set aside when the
name is matched. The unit comes from the header when it names one; otherwise
from the column's median, which tells the units apart (haemoglobin near 13 is in
g/dL, near 130 in g/L). A column whose median fits no unit, or whose header
names a unit the table doesn't list (`Sodium (mg)` in a food table), is left
alone. A wind speed or an air temperature with no unit in its header gets the
widest of its ranges. The clinical analytes count only in a table with at least
two of them whose medians fit, so a soil table's `K` is never read as
potassium. The rule in each issue names the unit and how it was found.

Since the median sets the unit, the misses are errors however many there are:
the 20% rule for name bounds does not apply.

### Sentinel codes

A code is a run of nines (`99`, `999`, `999.9`, `-9999`), a repeated digit of
four or more (`1111`, `8888`) or a large negative power of ten (`-1000`,
`-100000`). It is flagged when it stands alone, with no other value within a
factor of two of it, and is either more than 10 robust standard deviations from
the median of the other values or negative in a column that is otherwise not.
`99` among bacterial counts that run past 1,000 is a count; `999.9` among wind
speeds under 20 is a code.

A repeated digit that would be an ordinary value with one digit fewer is
called a likely entry error instead: a heart rate of 1,111 among readings up to
135 is as likely 111 typed with an extra 1. An age of 8,888 is a code.

A converted blank shows up differently: a value in the outer 1% that occurs at
least 10 times as often as the two values on each side of it, and holds a fifth
of that tail. A blank Fahrenheit reading becomes 0 °F, which is -17.78 °C; in a
temperature column the issue says so. A column with more than two such values
has a texture instead (a counter that plateaus, readings rounded to quarters),
and none is flagged. A value rounder than its neighbours (5.8 among 5.79 and
5.81) is not counted.

The same code in several columns is one issue. Cleaning uses the same rules
(`SENTINEL_VALUES`, `blank_codes`) to set clear codes to missing.

### Extreme values and second clusters

Distance from the median is measured in robust standard deviations: the scaled
median absolute deviation, or the interquartile range divided by 1.349 when
more than half the values are identical. A value counts when it is more than
10 of them out (`EXTREME_Z`) and also cut off from the rest: walking outwards
from the bulk, a gap of at least 3 robust standard deviations (`GAP_SPREADS`)
opens before it, and it is at least twice as far from the median as the last
value before the gap. A continuous tail, such as arctic stations among tropical
ones or the brightest noons in a column of solar readings, is not cut off.

An essentially non-negative column is measured on a log scale of its positive
values when the bulk is clearly right-skewed (a decile skewness of 0.1 or more,
`LOG_SKEW`) and more symmetric on a log scale, as concentrations, latencies and
counts are; only high values count on that scale. The deciles are used because
a few outliers can't move them. A column named as a temperature, a signed
quantity, a direction, a coordinate or a pH always keeps the linear scale (see
`prefers_log_scale`). A direction wraps around, so it gets no extreme check.

Up to 2% of a column (`EXTREME_SHARE`), or fewer than 10 values
(`CLUSTER_VALUES`), they are listed as extreme values. The wording calls them
likely entry or unit errors only when they lie 3 times or more beyond every
ordinary value (`ERROR_RATIO`); nearer ones are to be checked. Beyond that and
up to a quarter of the column (`CLUSTER_SHARE`), they form a second cluster: the
issue gives the rows, the range and the mean with and without them, and calls
it a second cluster or a possible sensor or unit fault. When the cluster is one
unbroken stretch of rows in time order, it says the unit, the scale or the
sensor probably changed for that stretch.

A fill value or a pile-up at a limit is left out when the median and spread
are measured, since it would sit in the bulk and shrink the spread.

Extreme values on the same rows in several columns are one issue: one record,
or one batch of records, is wrong or on another scale across all of them.

### Matrix tables

In a matrix table (`ColumnRoles.table_shape == "matrix"`: spectra, sample
panels), a bright emission line is part of a spectrum, so single cells are not
checked for extreme values. Each row is judged as a whole instead, by its
median across the matrix columns, with the same robust distance and gap rule.
Rows off scale are named by their label (the identifier) in the issue.

### Swapped columns

Rows where a column lies outside its own bulk (Tukey's 3 × IQR fences, from an
evenly spaced sample of 5,000 rows) are candidates. For each other column, a
candidate row counts as swapped when each column's value lies in the other's
usual range: the 1st to 99th percentiles of the other rows, widened by half
their width, with the two ranges apart. At least 3 swapped rows (`SWAP_ROWS`)
must fill 80% or more of the stretch they span, in time order when there is a
date. Wind speeds of 136 to 280 m/s on the same twelve days as wind directions
of 0.7 to 1.4 degrees are a swap, not two sets of extreme values, so impossible
or extreme values inside the block are not listed again.

### Caps, floors and fill values

A cap or floor is one value at the maximum or minimum with at least 3 rows and
3 times the rows of each of its three nearest values, that holds 3% or more of
the column or at least 5 rows. It must also stand out from every value inside
the range: twice as common as the most common of them (`CAP_STANDOUT`), or 10%
of the column or more (`CAP_CLEAR_SHARE`). A common recipe value at the end of
a range, matched by another value in the middle, is not a limit.

Not a cap or floor:

- zero, or 1 in a column of whole numbers: counts start there
- a count of subjects or samples (`n`, `sample_size`, `participants`), whose
  round values are a design, not a limit
- a value tied to a category: every row at the value falls in levels of one
  grouping column (3 levels or more) in which the column holds that value on
  95% of rows, as trophic level 2 is what every herbivore species has
- a value already read as a missing-value code (`-999` in 45% of rows)

A fill value is the most frequent value, holding 10% or more of the column and
written to more decimal places than nine in ten of the column's values (46.1526
among readings like 39.2). A simple fraction is not a fill value: 0.0166667 is
1/60, the mean of sixty counts of which one is 1. Only as many denominators are
tried as keep the chance of an accidental match under one in a thousand, so a
value written to four decimals is never explained away. Fill values recurring
on the same rows in several columns are one issue; the detail says when the
value is close to the mean of the other values, as mean imputation leaves it.

### Stuck readings

With a date, each reading is followed in time order, within each group of the
grouping column. Two patterns count:

- runs of exact zeros 24 readings or longer (`FLATLINE_RUN`), in a column that
  is otherwise positive and where zero holds less than half the values: a
  working sensor rarely reads exactly zero for that long
- one run of a single value holding 1% of the column or more (`FLATLINE_STALL`),
  and at least 240 readings

A column whose values repeat from one reading to the next on more than 20% of
rows anyway (`FLATLINE_REPEATS`), as coarse or slow readings do, is skipped,
and so are running totals (nine in ten changes upward), which hold still
between events, and values already reported as a cap or floor. Columns stuck
on exactly the same rows are one issue: the instrument behind them stopped.

### Copied values and max below min

Pairs are screened on an evenly spaced sample of 5,000 rows, overall and within
each group of the grouping column, and counted on the full table. A copy within
one group is reported only when the two columns rarely agree elsewhere (10% of
rows or fewer). With more than 100 columns (`PAIR_COLUMNS`), such as a spectrum
or a panel of samples, only the measure is paired with the others, since every
pair would cost more than the rest of the scan.

Not a copy:

- matches at a floor or a ceiling both columns share (two element contents both
  floored at 0.01)
- a column built as the row maximum or minimum of others: it never falls below
  (or rises above) them on 98% of rows and equals their maximum (or minimum) on
  95% (`ROW_EXTREME_SHARE`), as an index that is the largest of its
  sub-indices does

Siblings are found by name: `max` and `min`, `maximum` and `minimum`, `high`
and `low`, `upper` and `lower`. With no min-named sibling, the max-named column
is compared with the plain one (`Maximum Wind Speed` with `Wind Speed`). A pair
of spreads or counts (`sd`, `se`, `n`, `count`) is skipped, and so is a pair
whose two sides each have their own count column (`high_mp_n` beside
`high_mp_mean`): high and low then name two groups.

### Groups

With three or more groups of at least five values, a group is out of scale when
its median, compared with the typical median of the other groups, is within 10%
(`CONVERSION_TOLERANCE`) of a unit conversion factor (`CONVERSION_FACTORS`:
powers of ten, 1.8, 2.2046, 3.6, 7.5, 24, 25.4, 28.35, 60, 453.6, 3,600), and
clear of every other group by at least the square root of that factor. A group
that fills one stretch of time no other group shares, such as the burning phase
of a fire test, is an event, not a unit slip, and is left to the comparisons.
With two groups, one being far off is as likely the effect being studied, so
nothing is flagged.

### The whole table

**Duplicate rows.** Rows identical to an earlier row in every column, counted
when the table holds a continuous reading (decimals with many distinct values),
since such rows can't repeat by chance, and only among rows with one of those
readings recorded. Listed from 1% of the rows (`DUPLICATE_SHARE`), with how many
distinct rows remain; a table of codes and counts repeats naturally and is not
checked.

**Looks simulated.** Real measurements are rarely spread evenly between their
extremes, and readings taken together nearly always relate somewhere. Columns
with at least 100 values and 20 distinct ones are tested, up to 200 of them, on
distinct rows with every flagged cell left out. A column is flat when a
Kolmogorov-Smirnov test against a uniform spread over its range does not reject
it (p of 0.05 or more, `FLAT_P`) and its excess kurtosis is -0.6 or less
(`FLAT_KURTOSIS`; a uniform spread has -1.2). When more than half the columns,
and at least three, are flat, and the largest |r| between any two is below the
level chance reaches across that many pairs (Bonferroni, 1%, `FLAT_CHANCE`),
the issue says the table may be simulated and to check how it was produced.

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

## Cells left out of statistics

`statistics_mask(frame, issues)` marks the cells that can't be readings as
they stand: impossible values, missing-value codes and likely entry errors,
fill values, survey codes, values in swapped columns and stuck readings
(`MASKED_CHECKS`). Extreme values and second clusters may be real, so they are
not marked. `statistics_frame(frame, issues)` returns a copy with those cells
blank and a note such as "781 cells the data checks flagged (fill values) left
out", or the frame itself and an empty note when nothing is marked.

`analyze_business` builds that frame once and runs the relationship scan, the
group comparisons and the outlier count on it; each of those cards adds the
note to its calculation. With a wind-speed column stuck at one value for
18,000 rows, the wind pair's r is computed without the stuck rows.

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

Each column is sorted once and every check works on whole arrays; pairs,
missingness dependencies, swap ranges and row maxima are screened on a sample.
A table of 250,000 rows by 30 columns takes well under a second, and one of
1,100 rows by 7,900 columns under two.

## Libraries

The Kolmogorov-Smirnov distribution (`scipy.stats.kstwo`), the kurtosis
(`scipy.stats.kurtosis`) and the normal quantiles for the chance level of |r|
come from SciPy.

## Changing this

A new check needs a function returning `QualityIssue`s with a `detail` sentence,
a short `summary`, the `rule` that fired and, for a check on individual values,
the flagged rows in `cells`; an entry in `CHECKS`; and tests for a case where it
fires and a business table where it must not. A rule that depends on a column
name belongs in `name_bounds`, `PHYSICAL_LIMITS` or the word lists beside them,
never in a check.
