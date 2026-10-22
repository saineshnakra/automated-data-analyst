# Data checks

**Code:** `scan` in `ada/quality.py`
**Tests:** `tests/test_quality.py`

Looks through every numeric column, not only the measure, for values that
should not drive a result: readings outside what the column's name, its header
or a physical limit allows, codes standing in for missing readings, failed
records and rows of placeholders, values piled up at a limit or clipped at
both ends, filled-in gaps, values far from the rest, second clusters, swapped
and copied columns, parts above their whole, sensors stuck on one value, and
gaps that follow a pattern or a stretch of time. It also looks at the table as
a whole, for copied rows and for columns so flat and unrelated that the data
may be simulated, and describes what the analysis will skip or should treat
with care: constant columns, readings mostly 0, narrow piles. Each issue says
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
| The brief's **Data checks** card | The three issues about values and the two about gaps that cover the most rows, with counts, and how complete the table is; descriptions of the table come after the issues about values |
| The brief's summary | A "Check these first" line when an issue touches the measure or the group the headline names, covers 10% or more of the table (copied rows, simulated-looking data), or covers a quarter or more of a column; never a description of the table, and issues still in the data before codes cleaning set to missing |
| The leading-group card | When the leading group's average rests on flagged values, a sentence saying so, with the group's median |
| The relationship, comparison and outlier cards | How many flagged cells were left out of their calculation |
| The app's Data tab | A **Data checks** table |
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
columns get only the survey-code and rare-code checks.

When the brief covers one part of a stacked file (`ColumnRoles.table_shape ==
"stacked"`), the other tables' columns are empty on every row of it. They are
not gaps in this table, so `own_columns` drops them before the checks run, and
the completeness figure on the Data checks card counts only this table's
columns and says how many it set aside. In any other table a column empty
throughout is a gap and is reported.

## The checks

Issues about values come first, then descriptions of the table
(`INFO_CHECKS`: constant columns, readings mostly 0, single readings, bounded
pairs, narrow piles), then issues about gaps. Within each, issues are ranked by
the rows (or cells) they cover, so a fill value in 61% of rows comes before a
single impossible value; ties follow the order below, the measure's first.

| Check | Name in the table | Fires when |
|---|---|---|
| `generated` | Looks simulated | Most numeric columns are spread evenly between their extremes and no pair relates beyond chance, in the whole table or within every level of a label; or a few columns are like that beside real ones |
| `banded` | Group held in bands | One level of a two-level label never leaves bands the other crosses, and readings that move together elsewhere come apart in it |
| `duplicates` | Duplicate rows | 1% or more of the rows repeat an earlier row exactly, in a table with continuous readings |
| `swapped` | Swapped columns | On a block of consecutive rows, two columns hold values in each other's usual ranges |
| `impossible` | Impossible values | A value is outside what the name, the header or a physical limit allows (see below), or a column that is otherwise above zero has a few negatives the size of its usual values (a sign slip) |
| `implausible` | Implausible share | 5% or more of a column lies beyond a quantity's critical limits, though each value is possible |
| `sentinel` | Sentinel codes | A code value stands alone far from the rest, a value in the outer 1% occurs far more often than its neighbours, or a value repeated at the very edge stands apart from every other |
| `failed_record` | Failed records | Other columns hold one fixed value on the rows where a column holds a missing-value code |
| `zero_rows` | Rows of zeros | Nearly every reading on a row is exactly 0, in columns that are otherwise rarely 0 |
| `placeholder` | Not recorded | Three or more columns, one of them text, hold 0 or a dash on the same rows |
| `group_level` | Group out of scale | One group's median of the measure is off from the others by about a unit conversion factor |
| `equal_columns` | Copied values | Two columns hold the same non-zero value in half or more of the rows, overall or within one group |
| `min_max` | Max below min | A max-named column is below its min-named sibling, or below the plain one |
| `outside_range` | Average outside min to max | An average-named column lies below its min-named sibling or above its max-named one |
| `part_whole` | Part above whole | A part exceeds the whole its name pairs it with, on the few rows that break an otherwise steady relation |
| `cluster` | Second cluster | More than 2% of a column, and at least 10 values, lie cut off far from the rest |
| `extreme` | Extreme values | A value is more than 10 robust standard deviations from the median and cut off from the rest by a gap |
| `matrix_rows` | Rows off scale | In a matrix table, a row's overall level is far from every other row's |
| `imputed` | Probable fill values | One non-round value holds 10% or more of a column |
| `flatline` | Stuck readings | In time order, a reading sits at exactly zero for long stretches, or holds one value for a large share of the record |
| `cap` | Cap or floor | One value piles up at the maximum or the minimum, values crowd just under a round maximum, many columns share one limit, or a round value inside the range piles up in one group |
| `clipped` | Clipped at both ends | Piles of a similar small share sit at both the minimum and the maximum |
| `pile` | Narrow pile | A quarter or more of a column sits in a sliver of its range around one value, not on it exactly |
| `survey_code` | Survey codes | Codebook values such as 7, 9, 77 or 99 sit above an unbroken scale of answer codes |
| `rare_code` | Rare codes | A rare 0 sits below whole-number codes that run unbroken from 1 |
| `zeros` | Mostly zeros | A reading with decimals is exactly 0 in a third or more of its values |
| `single_reading` | Single readings | Two window statistics of one reading (max and min, first and last) are often equal: one reading in the window |
| `bounded` | Bounded by another column | One column never exceeds another but often falls short of it, as a time to recurrence and the follow-up time do |
| `constant` | Constant columns | A column holds one value on every row where it is recorded |
| `missing_by_group` | Missing by group | Some groups lack a column far more often than the rest |
| `missing_dependency` | Missing by value | A column's gaps fall where another column is high, or low |
| `missing_block` | Missing together | Two or more columns are missing on the same rows |
| `missing` | Mostly missing | A column is more than 20% missing |
| `missing_time` | Gaps in the log | A regular log holds far fewer readings than its interval and span imply |

### Impossible values

A range written in the header itself, such as `Humidity (40 - 100)` or
`[0 to 5]`, is the column's own statement of its valid values and comes
first. Otherwise bounds come from words in the column name:

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
and returns are negative. A stretch (`elongation`, `extension`, `expansion`,
`swelling`) written as a percentage can run past 100%, so it is bounded at 0
only.

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

| Quantity | Names | Unit: range | Critical limits |
|---|---|---|---|
| Sodium | `Na`, `sodium`, `sod` | mmol/L: 100 to 200 | 120 to 160 |
| Potassium | `K`, `potassium`, `pot` | mmol/L: 1 to 10 | 2.5 to 6.5 |
| Chloride | `Cl`, `chloride` | mmol/L: 50 to 160 | 80 to 120 |
| Calcium | `Ca`, `calcium`, `T Ca`, `total calcium` | mg/dL: 2 to 22; mmol/L: 0.5 to 5.5 | 6 to 13; 1.5 to 3.25 |
| Glucose | `glucose`, `GLU`, `FBS`, `RBS`, `BGR`, `sugar` | mg/dL: 10 to 3,000; mmol/L: 0.5 to 170 | |
| Creatinine | `creatinine`, `creat`, `SCr`, `SC` | mg/dL: 0.1 to 30; µmol/L: 10 to 2,700 | |
| Albumin | `albumin`, `ALB` | g/dL: 1 to 7; g/L: 10 to 70 | |
| Haemoglobin | `haemoglobin`, `hemoglobin`, `HGB`, `Hb`, `hemo` | g/dL: 3 to 25; g/L: 30 to 250 | |
| Haematocrit | `haematocrit`, `HCT`, `PCV` | %: 10 to 75; L/L: 0.1 to 0.75 | |
| Red cell count | `RBC`, `erythrocytes` | ×10¹²/L: 1 to 10 | |
| Heart rate | `heart rate`, `pulse` | beats a minute: 20 to 300 | |
| Body temperature | `body temperature`, `body temp` | °C: 25 to 45; °F: 77 to 113 | |
| Wind speed | `wind speed`, `wind gust`, `WS` with a unit | m/s: up to 115; km/h: 415; mph: 260; knots: 225 | |
| Air temperature | `air temperature`, `ambient temperature`, `outdoor temperature` | °C: -90 to 60; °F: -130 to 140 | |
| Water temperature | `water temperature`, `water temp` | °C: -3 to 45; °F: 27 to 113 | |
| Dissolved oxygen | `dissolved oxygen`, `DO` | mg/L: 0 to 25 | |
| pH of natural water | `pH` | 2 to 13 | |

Words such as `serum`, `blood`, `level`, `max` or `mean` are set aside when the
name is matched. The unit comes from the header when it names one; otherwise
from the column's median, which tells the units apart (haemoglobin near 13 is in
g/dL, near 130 in g/L). A column whose median fits no unit, or whose header
names a unit the table doesn't list (`Sodium (mg)` in a food table), is left
alone. A wind speed or an air temperature with no unit in its header gets the
widest of its ranges. A range written in the header wins over the table.

The blood analytes and the surface-water readings each form a panel, and a
member counts only when another member of its panel in the table fits its own
unit too, or, for water, a companion reading (salinity, turbidity,
conductivity, Secchi or water depth) is present. So a soil table's `K` is never
read as potassium, and a lab's pH of 1.2 is not judged as a lake's. The rule in
each issue names the unit and how it was found.

Since the median sets the unit, the misses are errors however many there are:
the 20% rule for name bounds does not apply.

**Implausible share.** A value inside the range but beyond a quantity's
critical limits can happen; 5% or more of a column beyond them
(`CRITICAL_SHARE`) can't, since critical values turn up in a few percent of
even very ill groups. A fifth of a table's sodium values above 160 mmol/L
points to errors, another unit or a mislabelled column; the values are listed
but not left out of statistics.

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

A converted blank can also sit at the very edge of a column, below the coldest
real reading: a value at the minimum or maximum in 10 rows or more
(`EDGE_CODE_VALUES`) but under a quarter of the column, 10 times as common as
any of its five nearest values, further from the nearest than a quarter of the
other values' IQR, with the values near that nearest one thin (a quarter of
their density between the quartiles, or less). A cap sits against the readings
it limits, which crowd up to it, so it is not taken for this. Zero is left out:
an instrument that is off reads 0 apart from its running values.

The same code in several columns is one issue. Cleaning uses the same rules
(`SENTINEL_VALUES`, `blank_codes`) to set clear codes to missing.

**Failed records.** On the rows where one column holds a missing-value code, a
logger that failed often writes its own fixed values in the other channels (a
sound level of 0, a battery of 2.5 V where temperature reads -25.6). A value
that fills 90% or more of the code's rows (`FAILED_RECORD_SHARE`, at least 10
rows) and turns up elsewhere on at most a tenth as many rows is listed with
the code; those cells are no more readings than the code is.

**Rows of zeros.** A row on which 75% or more (`ZERO_ROW_SHARE`) of at least
three readings are exactly 0, where each of those readings is 0 on under 5% of
its values (`ZERO_ROW_COLUMN_SHARE`), is a row of blanks written as zeros.

**Not recorded.** A feature table may write "-" for the name and 0 for the
formula, the score and the error of every feature it could not annotate. Three
or more columns, at least one of them text, holding 0 or a dash on rows that
agree by 98% or more (Jaccard, `PLACEHOLDER_OVERLAP`), covering 5% to 95% of
the table with ten or more rows of real values, form one issue: read as values,
those cells would make "score 0" a group.

**Rare codes.** A code set that starts at 1 (a five-point scale, categories 1
to 3) has no use for 0, so a 0 on under 2% of rows (`RARE_CODE_SHARE`), at most
a fifth as common as any other code, is named as a likely missing value and
left in place. A lone 7 or 8 is a survey code only above an unbroken scale of
answers (1 to 5): among histological types 1, 2, 4, 5 and 7, a 7 is one more
type.

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
Extreme values in more than 10 columns (`EXTREME_COLUMNS_LISTED`), as a wide
panel of one kind gives, are one issue too, naming the columns with the most.

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

A column of fewer than five values is checked only for one value holding half
of it or more at its minimum or maximum (a reporting limit, 0.001 on 94% of
rows); from 90% such a limit leaves the column almost no information, and the
issue says so. A round maximum (500 on an index whose scale stops there) is a
cap even when it holds under 3% of the column.

More shapes of a limit:

- **Clipped at both ends.** Piles at both the minimum and the maximum, each
  0.1% to 5% of the rows (`CLIP_SHARES`) and at least twice its neighbours,
  with the two shares within a factor of 2 (`CLIP_LIKENESS`), are one
  clipping, as winsorising at the 0.5th and 99.5th percentiles leaves. Columns
  clipped like that are one issue whatever their limits.
- **Squeezed under a maximum.** Values within 1% of the range below a round
  maximum, 2.5 times as many as a typical stretch of that width
  (`SQUEEZE_RATIO`) and 3% of the column or more, crowd against a limit
  without piling on it exactly (62 values from 79 to 80, only 5 of them at 80).
- **Shared limits.** A value that is the exact maximum or minimum of 10 or more
  columns (`SHARED_CLIP_COLUMNS`) and of 10% of the readings, as expression
  values clipped at -2 and 2 by a platform are, is one issue for all of them,
  and each column's other checks leave it out. Zero, and a limit the names
  give (100 for percentages), are left out.
- **A limit within one group.** A round value inside the range holding 5% or
  more of the column and three times any other value, with 80% or more of its
  rows in one level of a grouping (`GROUP_CAP_CONCENTRATION`), is a limit that
  group's tests stop at: compression tests stopped at 50% strain.

A fill value is the most frequent value, holding 10% or more of the column and
written to more decimal places than nine in ten of the column's values (46.1526
among readings like 39.2). A simple fraction is not a fill value: 0.0166667 is
1/60, the mean of sixty counts of which one is 1. Only as many denominators are
tried as keep the chance of an accidental match under one in a thousand, so a
value written to four decimals is never explained away. Fill values recurring
on the same rows in several columns are one issue; the detail says when the
value is close to the mean of the other values, as mean imputation leaves it.
A fill value whose rows all share one value of a key column (a text column
with ten values or more) is one sample's reading repeated on its rows, as a
survey listing every species found on a visit repeats the visit's sulphate,
and is not reported.

**Narrow pile.** A quarter or more of a column (`PILE_SHARE`) within 2% of its
range (`PILE_WIDTH`, the range between the 0.5th and 99.5th percentiles, on a
log scale for a right-skewed column), three times as many as the slivers of
that width on either side hold (`PILE_RATIO`), with a quarter of the other
values within ten such slivers of it (`PILE_REACH`) and no single value
holding most of them: soil pH entered near 7.0 unless measured otherwise.
Values only far from a pile, such as wind speeds of 140 among speeds under 1,
are outliers whatever the IQR, so a bulk with nothing near it is no pile. The
interquartile range then measures the pile, so the outlier card leaves the
column out and says so. A floor or cap inside the pile already explains it,
and no pile is reported then.

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

- agreement at a floor or a ceiling both columns share, or at zero: the two
  must also agree on half the other rows. Two element contents both floored at
  0.01 are not copies; two identical flag columns, most of whose rows sit at
  their lowest code, are
- a column built as the row maximum or minimum of others: it never falls below
  (or rises above) them on 98% of rows, equals their maximum (or minimum) on
  95% (`ROW_EXTREME_SHARE`), and takes that value from at least two of them,
  as an index that is the largest of its sub-indices does; a column equal to
  one of them almost everywhere is a copy of it
- the maximum and minimum of one reading, or one reading in two windows
  (`OBS1_glucose_max` and `OBS1_glucose_min`): where a single reading was
  taken, both are that value. All such pairs are one `single_reading` note
- a bounded pair: one column at or below the other on 99% of rows
  (`BOUNDED_SHARE`), strictly below on a tenth or more, and of the same order
  (a median ratio of a fifth or more), as a time to recurrence set to the
  follow-up time where there was none, or a discounted price that is the list
  price where no discount applied. It is a `bounded` note, not a copy

Siblings are found by name: `max` and `min`, `maximum` and `minimum`, `high`
and `low`, `upper` and `lower`. With no min-named sibling, the max-named column
is compared with the plain one (`Maximum Wind Speed` with `Wind Speed`). A pair
of spreads or counts (`sd`, `se`, `n`, `count`) is skipped, and so is a pair
whose two sides each have their own count column (`high_mp_n` beside
`high_mp_mean`): high and low then name two groups. An average-named sibling
(`PM avg`, `mean PM`, or the plain `PM` beside `PM max` and `PM min`) that lies
below the minimum or above the maximum on a row is an `outside_range` issue.

**Part above whole.** Two readings of one kind pair as part and whole when
their names end in the same word, the whole's other words are words such as
`total`, `gross` or `overall` (or, for a depth, name the whole water column:
`water`, `site`, `bottom`, `station`) and the part's are not: `SecchiDepth`
and `WaterDepth`, `Online Sales` and `Total Sales`. When the part stays at or
below the whole on 95% or more of the rows holding both (`PART_WHOLE_SHARE`),
the rows where it exceeds the whole are listed.

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

A table generated class by class looks real pooled, since the classes differ,
so the same test also runs within each level of a label (two to six levels of
at least 100 rows each): when in every level most readings are flat and no pair
relates beyond chance, the issue names the label. And when the table's other
readings relate as real ones do, two or more columns that are flat and relate
to nothing (an age, a weight and a farm size spread evenly between round
bounds) are named on their own.

**Group held in bands.** Controls filled in from reference ranges sit inside
the normal band of every test, which real controls do too, but readings that
move together in any real blood come apart in them. One level of a two-level
label counts when it stays within its own range on three or more readings
(`BAND_READINGS`) that the other level leaves on a quarter of its rows
(`BAND_OUTSIDE`), and a pair of those readings related at |r| ≥ 0.7 in the
other level is unrelated (|r| < 0.2) in this one. Not reported when the table
already looks simulated.

**Descriptions of the table.** These rank after the issues about values and
are never named first in the summary:

- `constant`: columns holding one value on every row where they are recorded
  (and recorded on 90% of rows or more). The analysis leaves them out; naming
  them keeps a never-detected compound, 0 in every sample, from vanishing
  without a word. Text columns count only in a flat table, since focusing on
  one part of a long or stacked table makes the column naming that part
  constant; dates and the measure are left out
- `zeros`: readings with decimals that are exactly 0 in a third or more of
  their values (`ZERO_HEAVY_SHARE`): absences, or values below a detection
  limit. Their Pearson correlations mostly contrast the zero rows with the
  rest, so the issue suggests rank correlations or zero against not zero. Rain
  and snow, whose zeros are plainly dry days, are left out
- `single_reading`, `bounded` and `pile`, described above

### Missing data

Columns 5% or more missing are compared for overlap; those whose missing rows
overlap by 90% or more (Jaccard) form a block. A block's columns are reported
once, as a block, and not again one by one.

With a date, a gap is first tested against time, in time order: when one
unbroken stretch holds 95% of the missing rows (`GAP_RUN_SHARE`), the column
was not recorded before or after a date, or for that stretch; when the missing
rows fall in at most six stretches (`GAP_STRETCHES`) averaging two rows or
more, they are gaps in the record rather than a subsample. Either is said in
the issue, and the gap is then not blamed on a group or another column.

**Gaps in the log.** A regular log (half its steps within a tenth of the
typical interval) that holds under 90% of the readings its interval and span
imply (`LOG_COVERAGE`), with a gap of at least a week and ten intervals, is
reported with the typical interval, the share of expected readings present
and the longest gap: "Readings come every 5 minutes (the typical interval), but
only 179,568 of the 309,600 expected ... are there (58.0%)". Totals and counts
per period then follow which stretches were logged.

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
fill values, survey codes, values in swapped columns, stuck readings, the
fixed values of failed records, rows of zeros and placeholder cells
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
missingness dependencies, swap ranges, row maxima and the tests for simulated
columns are screened on a sample.
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
