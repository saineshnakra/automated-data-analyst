# Schema detection

**Code:** `ada/schema.py`
**Tests:** `tests/test_schema.py`

Decides what the columns of an unfamiliar file mean. Everything downstream
depends on it, and the user can correct it in the app.

The output is a `ColumnRoles`:

| Field | What it holds |
|---|---|
| `measure`, `date`, `dimension`, `identifier` | The picked roles |
| `numeric` | Numeric columns that can be measured: quantities first, then category codes, labels and constant columns. Identifiers are left out |
| `dimensions` | Every column that can be the grouping, text or numeric |
| `factors` | Numeric columns that hold category codes, such as a 0/1 flag, a status code or a set dose. They are in `dimensions` too |
| `flags` | The factors holding only 0 and 1 |
| `measure_alternatives` | Up to three runners-up when the measure was a close call |
| `table_shape` | `"flat"`, `"long"`, `"stacked"` or `"matrix"` (see [Table shapes](#table-shapes)) |
| `focus` | For a long or stacked table, the value of `dimension` the brief covers |
| `notes` | What detection concluded, in plain sentences |
| `aggregation` | `None` unless the user says how the measure combines (see below) |
| `non_readings` | Numeric columns that place or describe a row rather than measure it (see [Not readings](#not-readings)) |

## Three signals

No single signal is reliable:

- **Name alone.** A column called `Value` could be anything.
- **Type alone.** `Order Number` is numeric, but its sum means nothing.
- **Distinct values alone.** A column with 8 distinct values might be
  `Region` or `Rating`.

So names, values and distinct-value counts are combined.

## Picking the date

Only datetime columns with at least two distinct values are considered
(after cleaning has inferred them, including a `Year` column it has turned
into dates; see [Cleaning](cleaning.md#dates-from-years)). A date that never
changes, such as a compilation stamp, places no row in time. They are ranked
by:

1. Name contains `date`, `time` or `created`.
2. The event that creates a row ahead of later events: a name with `order`,
   `transaction`, `sale`, `invoice`, `posting`, `created` or `booking` ranks
   above a neutral name, which ranks above `ship`, `delivery`, `delivered`,
   `due`, `updated`, `modified` or `closed`. So `Order Date` beats `Ship Date`.
3. Most non-missing values.
4. Column name, alphabetically, so column order never decides.

## Picking the identifier

Three kinds of column key or label the rows, and the first found wins:

1. **A key name.** One of the name's words is `id`, `uuid`, `key`, `code`,
   `number`, `invoice` or `order` (as a whole word), the last word is not a
   measure keyword, at least 80% of the values are distinct, and the values are
   whole numbers or text. A `Reynolds Number` with decimals is a measurement.
2. **A row key by its values.** Unique whole numbers on every row, under a
   row-key name (`seqn`, `index`, `idx`, `row`, `record`, `serial`, `obs`,
   `unnamed`), or in the table's first column climbing in uneven steps, as
   respondent numbers with gaps do. Unique whole numbers alone are not enough:
   forty order totals can all differ, and a dose stepped evenly down the rows
   is unique too.
3. **A row label.** A text column with a different value on every row, such
   as a taxon or a file name. Sentences are unique too but label nothing, so
   values of more than six words do not count.

Identifiers are left out of `numeric`, so they are never the measure, never in
the strongest-relationship scan and never offered as something to correlate.
A key found by its values or a row label gets a note. The identifier also
provides the "Distinct X" KPI or, when it names each row once, says what one
row is ("One per Reading ID").

Which names are keys also decides what the reader and cleaner keep as text
(`is_identifier_name`, see [Cleaning](cleaning.md)). The last word decides,
and `no` counts only after another word or with a mark (`Order No`, `No.`,
`No #`): a bare `NO` is as likely nitric oxide. The same holds for
`is_entity_name`, which names the unit a row belongs to (`Subject`,
`Station`, `Patient ID`).

## Category codes

A numeric column is a factor, a category code rather than a quantity, when:

| Evidence | Rule |
|---|---|
| A 0/1 flag | Exactly the values 0 and 1, whatever the name, unless it is a count like its siblings (see below) |
| A label name | Whole numbers, each used twice on average, that can group rows (see [Picking the segment](#picking-the-segment)), under a label word (`status`, `condition`, `group`, `treatment`, `label`, `class`, `stage`, `phase`, `sex`, `gender`, `race`, `ethnicity`, `phenotype`, `genotype`, `type`, `category`, `diagnosis`, `arm`, `cohort`), a design label (see below), an entity (`subject`, `patient`, `station`, `site`, `plot`, a key word) or a name starting with `id` or `code` (`id_reef`). So 159 weather stations, each with hundreds of days, are a label |
| A short code | Two values, or up to 12 whole numbers each used about five times or more, such as race, education or an income band |
| A design level | Up to 15 levels, each used about five times or more, under a word such as `dose`, `concentration`, `age`, `day`, `time` or `frequency`, and varying within the groups of another column: a set concentration crossed with dyes, a curing age crossed with recipes |
| A swept setting | 3 to 100 evenly spaced levels, each used about equally often (the rarest at least half as often as the commonest, and five times or more), stepping by more than the values' own resolution: a frequency swept from 100 to 590 GHz in steps of 10, a transect read every 5 m. A reading rounded to whole percent steps by its resolution and is not used equally at every level |

A name that marks a count or an amount keeps a short code a quantity: a count
word anywhere (`count`, `number`, `num`, `qty`, `quantity`, `amount`, `total`),
a plural last word (`Cylinders`, `Visits`), or a last word such as `size`,
`children`, `score`, `rating`, `price` or an outcome word. A 0/1 flag stays a
flag even then.

Rating items stay quantities too, however few their values. A column of whole
numbers is an item when it shares a numbered stem with another column
(`RDB1` … `RDB3`, `RP_1` … `RP_3`, `Realism_1` and `Realism_2`; headerless
`Unnamed: 3` columns are numbered by position and are no family), when it sits
on a rating scale shared by at least two other columns (whole numbers from 0
or 1 up to at most 10, using five points or more, such as 1 to 7), or when it
is on such a scale and named as a rating (`rating`, `likert`, `scale`,
`satisfaction`, `agreement`, `likelihood`). A lone code from 1 to 6, such as
marital status, is none of these.

Counts are read alike across their block. A 0/1 column named as a count
(`Nudibranchs`, `Seastars`) is a count, not a flag, when it sits beside other
count-named columns of whole numbers of at least zero, one of which passes 1
(`Crabs`), or is missing on exactly the same rows as such a column.
`Antibiotics` beside `Probiotics`, both 0/1, stay flags.

Short codes, design levels and swept settings are weaker evidence, so they
are dropped when every numeric column would otherwise be a factor: a table of
marks out of 10 and nothing else measures the marks.

A part of a date (`Month`, `MO`, `YR`, `DA`) beside a date column is neither a
factor nor a grouping, since it only repeats the timeline. A name counts as a
part of a date only when the values fit it (`is_time_part`): months 1 to 12 or
month names, days 1 to 31, hours 0 to 24, years 1000 to 2999 (or two-digit
years under a bare `Year` or `YR`). `Age_months` holding 12 to 66 is an age, and
`Age_years` or `Years` holding 1 to 80 are durations. `is_year_name` and
`is_month_name` take the values for the same check.

Factors are listed in `dimensions`, can be the grouping, are never the
measure, and are left out of totals and the strongest-relationship scan. A note
names them and gives each 0/1 flag's share of 1s, since a flag's average is a
share. `metrics.metric_for` treats a flag as a rate.

## Table shapes

### Stacked tables

A file can stack several tables, each with its own columns, beside a column
naming which table each row belongs to (`source_table`): foods and drinks in
one block, blood markers with their p-values in another. Picking a measure
from one block and a grouping from another compares nothing. A text column
with 2 to 20 values, recorded on nearly every row, names stacked tables when
its largest table and another each own at least two columns the other lacks
(recorded on half its rows or more, and on at most 2% of the other's), and
those columns make up at least a third of the file's columns. A table whose
rows merely leave a few columns blank, such as a refund amount filled only on
returned orders, is not stacked.

The table is then `"stacked"`: roles are detected on the largest table alone,
that table is the `focus` of the naming column, which becomes the grouping,
and a note names the other tables and their sizes. `ada.analyze` applies the
focus as for a long table, and the largest table's own grouping takes over.

### Long tables

When one text column names what each row measures and one numeric column holds
the result, averaging the result would mix ammonia in mg/l with conductivity
in µS/cm. A column qualifies as the parameter when its last word (after `label`,
`name` and similar) is `determinand`, `parameter`, `analyte`, `variable`,
`test`, `component`, `measure`, `measurement`, `indicator`, `characteristic`,
`metric` or `property`, and either a unit column (`unit`, `units`, `uom`) holds
several units or the result's typical size differs tenfold or more between
parameters. A unit column holding several units, with no parameter column, is
used as the parameter.

The table is then `"long"`: the parameter column is the grouping, the most
common parameter recorded in one unit is the `focus`, and a note names it and
the next most common ones. `ada.analyze` applies the focus, so the brief, the
answers and the source name cover that parameter only; naming a `dimension`
skips it. The grouping chosen within the focus rows comes first in
`dimensions`, so it takes over once the focus is applied. In the app the
drill-down starts on the focus and offers no "all values" option.

### Matrices

A table that is mostly numbered numeric columns of one kind (at least 20, and
at least three quarters of the numeric columns), such as sixty sample columns
`A3` … `B164` or 7,932 spectral channels named `1` … `7932`, has no single
measure. So does a table of counts whose columns have no numbered names, such
as 26 samples of read counts named `AQC1cm`, `CC1` … `Even3` beside an OTU id:
at least 10 columns of whole numbers of at least zero, three quarters of the
numeric columns or more, with average sizes within a factor of 1,000 of one
another, and an identifier naming each row. It is `"matrix"`: `measure` is
`None`, and a note says what the table appears to be, suggests reshaping it to
long form, and names prefix groups (`A*` and `B*`) that may be groups to
compare. For counts it also says how far the columns' totals differ and to put
them on a common scale before comparing them. `pipeline.prepare_analysis`
then adds each row's mean across the matrix columns, named
`Row mean (60 columns)`, and uses it as the measure (`schema.add_row_summary`).

### Summary statistics

A table whose columns are named `mean`, `min` and `max` with `std`, `sd`,
`median` or a quartile (`25%`, `50%`, `75%`), as the output of a summary per
probe or gene, already summarises a set of values on each row. `mean` is the
measure, and a note says the other statistics move with it by construction and,
when `count` is the same on every row, how many values each row summarises.

## Picking the measure

Numeric columns that are not identifiers, factors, matrix columns, running
totals or [non-readings](#not-readings) are candidates. A constant column is
never picked, and nor is one holding a different whole number on every row in
order, as a key or a date stored as a number (`MESS_DATUM` 19540601 …) does,
unless its name marks a quantity. With no candidate left, ADA counts rows. The
candidates are scored:

| Signal | Effect |
|---|---|
| Name matches a measure keyword | +4 to +14 for the last word; two thirds of the weight for a keyword elsewhere in the name |
| Names a statistic of another quantity (`p_value`, `pvalue`, `beta`, `coef`, `std`, `stderr`, `significant`, `t_stat`, `z_score`, `ci_low`, `low_mp_sd`, ...) | −6, and no keyword weight: a p-value is no `value` |
| Recorded on under half the rows the best-covered candidate has | −3, or −6 under a quarter |
| Looks like an identifier | −20 |
| Places a row rather than measures it: whole numbers named as a time part (`year`, `month`, `day`, `mo`, `yr`, ...), a design label or an entity, or any coordinate (`latitude`, `lon`, `easting`, `northing`, `gps`, ...) | −15 |
| Last word is an input to a mix or a dose (`cement`, `water`, `slag`, `ash`, `aggregate`, `admixture`, `plasticizer`, `binder`, `sand`, `dose`, ...) | −4 |
| A check that a study worked (`realism`, `attention`, `manipulation`, or last word `check`) | −6 |
| The one column whose unit differs from a unit at least three others share (a strength in MPa among ingredients in kg/m³) | +4 |
| Records the amounts of a running total that was set aside (`Interval Rain` for `Total Rain`) | +10 |
| One value in more than 80% of rows (nearly constant or mostly zero) | −6 |
| Any word is a condition usually recorded beside the outcome (`temp`, `temperature`, `humidity`, `wind`, `air`) or the size of a subject or plot (`weight`, `size`, `area`, `acres`, `height`, `age`) | −3, but only when another column is a plausible outcome |

When the grouping its name ranks first names an outcome class (its last word,
before `label` or `name`, is `result`, `outcome`, `diagnosis`, `class`,
`label`, `disease`, `case`, `condition` or `status`, with 2 to 20 values), ties
go first to the reading that best separates its groups: the most variance in
the reading's ranks that the classes explain, adjusted for their number. A
heart study's troponin beats CK-MB across positive and negative results, and a
blood count's haemoglobin beats MCV across anaemia diagnoses.

Other ties are broken by how much information the column holds: the entropy
of its values, counting missing cells as one more value, so many distinct,
evenly used values beat a column that is mostly one value or mostly missing.
Then fractional values, the share of non-missing values, and the column name
alphabetically. Two exports of one table with the columns in a different order
therefore detect the same measure.

When the runners-up score within three points of the pick, up to three of
them are kept in `measure_alternatives`, so the reader can be told the choice
was a judgement; a runner-up that is one value in more than 80% of rows is not
offered. They are cleared when the user picks the measure.

Keyword weights (`MEASURE_KEYWORDS`): `revenue` 14, `mrr` 14, `arr` 14,
`sales` 14, `gmv` 14, `turnover` 14, `profit` 13, `margin` 12, `amount` 11,
`value` 10, `income` 10, `spend` 9, `total` 9, `cost` 8, `expense` 8,
`price` 7, `fee` 7, `fare` 7, `quantity` 6, `units` 6, `orders` 6, `volume` 6,
`balance` 6, `count` 6, `abundance` 6, `yield` 6, `concentration` 5, the
outcome words `strength`, `response`, `latency`, `result`, `outcome`, `growth`,
`activity`, `viability`, `survival`, `decision`, `choice`, `risk` and
`diversity` 5, and `score` and `index` 4.
`MEASURE_KEYWORDS` in `ada/schema.py` is the authoritative list.

**Scoring a name.** The last word decides: `Product Category` is a category,
`Product Container` is a container. A keyword found anywhere else in the name,
even inside a longer word, scores two thirds of its weight (`MODIFIER_WEIGHT`),
so it can still win when nothing better is present. Before scoring, a trailing
annotation in parentheses or brackets is removed (`Revenue (USD)` is scored on
"revenue"), camelCase is split, so `netRevenue` scores like `net_revenue`, and
dots separate words, so `determinand.unit.label` ends in "label".

The covariate penalty is light and conditional: in a survey with `count` and
`air_temp_c`, or an assay with `Concentration (mg/L)` and `Temp (°C)`, the
outcome wins, but in a table of weather readings a temperature still beats a
snowfall column that is 98% zeros.

### Running totals

A column that climbs like a gauge or a meter, such as a weather station's
`Total Rain` or an odometer, is never the measure: its total or average over a
period means nothing. Within each group of the grouping, in date order, it
must rise many times for each fall, and its falls must be resets to a tenth of
the last value or less. A name such as `Cumulative Cases` (`cumulative`,
`cum`, `running`, `accumulated`, `ytd`, `odometer`) is enough without a reset.
Rainfall that falls back to zero after each shower rises and falls as often,
so it does not qualify, and neither does a day-of-month column. A sibling
column recording each reading's own amount (`Interval Rain`, `Hourly ...`,
`... Delta`) is preferred as the measure, and a note explains both.
`running_total_columns` reads the set-aside columns back from the notes, so
Ask ADA refuses to sum, average or correlate them.

## Picking the segment

A text column can be a segment when it is not the date, not an identifier,
not a quality flag (see [Not readings](#not-readings)), not named like a date
(`date`, `time`, `timestamp`, `created` or `updated` as a word), and:

- has at least 2 distinct values, and
- has distinct values in at most 65% of its non-missing rows, and
- has at most 100 distinct values, or else at least 3 rows in most (half or
  more) of its groups

The second rule excludes near-unique text such as notes, email addresses and
names, which would produce one bar per row. The third lets 101 districts of 3
to 20 soil samples each, or 286 drugs screened on hundreds of cell lines each,
group a table, while names that each appear once or twice do not. Factors
qualify too.

Qualifying columns are scored:

| Signal | Effect |
|---|---|
| Name matches a dimension keyword | +4 to +12 (two thirds for a keyword before the last word) |
| Last word is an experimental-design label (`Row`, `Plate`, `Run`, ...) or one subject each (`ID`, `Subject`, `Patient`, ...) | −15 |
| The largest group holds more than 90% of rows | −5 |
| More than 100 distinct values | −3 |
| Distinct count close to 10 | Preferred on a tie |

A grouping must share rows with the measure: at least 12 rows, or half the
measure's rows in a small table, holding both, in two groups or more. Groups
that never hold a value compare nothing. A time-part column (`Month`, `MO`,
`YR`, ...) is not picked when the table has a date, which already gives the
timeline.

When names cannot choose (the best score is shared), the grouping that
explains the most of the measure wins over the one with the handiest number
of groups, but only when it explains at least 5% of the variance in the
measure's ranks (adjusted for its number of groups) and at least 2 points more:
a swept frequency that explains 70% of a terahertz signal beats the materials
that explain 1%, while two groupings that explain nothing leave the choice to
the number of groups. A grouping that only splits another tied one's groups
further, as specimens split materials, is left out of that comparison.

Two groupings can split the rows the same way. When one is recorded on at
least 5 points more of the rows and splits the rows where both are recorded
into the same groups, it replaces the picked one (a complete `TCGA_DESC` for
a cancer type left blank on a fifth of the rows). Text labels replace a numeric
code that splits the rows the same way (`Gp` T0 … D6 for `Group` 1 … 12). A
note says so in both cases, and another names any other pairs of groupings
that split the rows identically (`side` and `protection_status`), since
comparing by either gives the same result.

Keyword weights (`DIMENSION_KEYWORDS`): `product` 12, `category` 12,
`segment` 12, `region` 11, `channel` 10, `country` 10, `treatment` 10,
`diagnosis` 10, `result` 10, `outcome` 10, `state` 9, `city` 9, `condition` 9,
`group` 9, `class` 9, `label` 9, `phenotype` 9, `genotype` 9, `arm` 9,
`cohort` 9, `customer` 8, `client` 8, `team` 7, `department` 7, `station` 7,
`site` 7, `status` 6, `stage` 6, `phase` 6, `type` 5, `sex` 5, `gender` 5,
`race` 4, `ethnicity` 4.

About ten groups reads well on a chart; two is thin, and sixty is too many
bars. A design label such as a plate's `Row` (A to H) groups wells by where
they sat, so it is picked only when no other column qualifies; `Sample` or
`Treatment` beside it wins. A heart study's `Result` (positive or negative)
beats its `Gender` code, and a smoke detector that is off in 98% of rows gives
way to the experiment's phase.

## Not readings

Some numeric columns place or describe a row rather than measure it. They are
listed in `ColumnRoles.non_readings`, stay in `numeric` (last) so questions can
name them, and are never the measure, a factor or a reading in the comparison
and relationship scans (`business_insights.numeric_readings` leaves them out).
A note names them, except parts of a date.

| Kind | Rule |
|---|---|
| Map coordinates | Any word `latitude`, `longitude`, `lat`, `lon`, `lng`, `easting`, `northing`, `gps`, `utm`, `coordinate(s)`, `coord(s)` |
| Compass directions | Last word `direction`, `dir`, `bearing`, `heading`, `azimuth`, `aspect` or `wd`, with values from 0 to 360 (`is_angle_column`): 359° sits next to 0°, so an average or a linear correlation means nothing |
| Quality flags | A word `qn`, `qc`, `qa` or `qf` (`QN_3`), a name starting with `quality` and ending in `flag` or `code`, or a name ending in `flag` after another column's name (`TMK_flag` beside `TMK`) (`is_quality_flag`). `Fraud_Flag` on its own is an outcome to compare. Text quality flags are not groupings either |
| Numbered rows | A different whole number on every row, in order, not named as a quantity: a key or a date stored as a number |
| Parts of a date | A time-part name with values that fit it (see [Category codes](#category-codes)) |

## Notes

`ColumnRoles.notes` explains what detection did that a reader would not
guess: identifiers found by their values, columns that are not readings,
category codes and flag shares, a summary-statistics table, running totals set
aside, groupings swapped for a twin or splitting the rows alike, a long or
stacked table's focus, and a matrix.
`pipeline.prepare_analysis` adds them to the cleaning notes, so the app, `ada
schema` and `to_dict()["cleaning_notes"]` list them too.

## Overriding

In the app, `pipeline.apply_role_selection` builds a new `ColumnRoles` from the
"Tune ADA's schema detection" dropdowns. `"None"` clears a role. Changing the
measure clears `measure_alternatives`, and changing the grouping clears the
`focus`. The dashboard re-renders without re-reading or re-cleaning the file.

In Python, pass `date=`, `measure=` or `dimension=` to `ada.analyze`. On the
command line, `ada schema FILE` prints the detected roles, and `--date`,
`--measure` and `--dimension` override them. A date override may name a
column of whole years (1000 to 2999), which is read as 1 January of each year.

How the measure combines is overridden the same way: **Combine rows by**
(Total or Average) in the app, `aggregation="sum"` or `"mean"` in
`ada.analyze`, and `--aggregation` on the command line. It is stored on
`ColumnRoles.aggregation`, and every module reads it through
`metrics.metric_for`, which applies it only to the active measure. `ada schema`
prints the measure with how it combines, for example `Resolution Hours
(mean)`.

## Edge cases

| Situation | Behavior |
|---|---|
| No datetime column and no year column | `date` is `None`; trend, anomalies and forecast are skipped |
| No numeric column, or only constant ones | `measure` is `None`; ADA counts records instead |
| Every numeric column a 0/1 flag or a labelled code | `measure` is `None`; ADA counts records by the grouping |
| Every text column is near-unique | `dimension` is `None` unless a factor qualifies |
| Several plausible measures | Highest score, then most information, wins; the runners-up are in `measure_alternatives` |

## Changing this

A weight change affects every file, so add a synthetic fixture that fails
before the change and passes after it. Keep fixtures synthetic; never commit
real customer data.
