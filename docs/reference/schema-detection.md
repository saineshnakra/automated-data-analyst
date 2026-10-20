# Schema detection

**Code:** `ada/schema.py`
**Tests:** `tests/test_schema.py`

Decides what the columns of an unfamiliar file mean. Everything downstream
depends on it, and the user can correct it in the app.

The output is a `ColumnRoles`: `measure`, `date`, `dimension` and
`identifier`, plus the full `numeric` and `dimensions` lists used for the
override dropdowns. Its `aggregation` is `None` unless the user says how the
measure combines (see below).

## Three signals

No single signal is reliable:

- **Name alone.** A column called `Value` could be anything.
- **Type alone.** `Order Number` is numeric, but its sum means nothing.
- **Distinct values alone.** A column with 8 distinct values might be
  `Region` or `Rating`.

So names, types and distinct-value counts are combined.

## Picking the date

Only datetime columns are considered (after cleaning has inferred them,
including a `Year` column it has turned into dates; see
[Cleaning](cleaning.md#dates-from-years)). They are ranked by:

1. Name contains `date`, `time` or `created`.
2. The event that creates a row ahead of later events: a name with `order`,
   `transaction`, `sale`, `invoice`, `posting`, `created` or `booking` ranks
   above a neutral name, which ranks above `ship`, `delivery`, `delivered`,
   `due`, `updated`, `modified` or `closed`. So `Order Date` beats `Ship Date`.
3. Most non-missing values.
4. Column name, alphabetically, so column order never decides.

## Picking the measure

Only numeric columns are considered. Each gets a score:

| Signal | Effect |
|---|---|
| Name matches a measure keyword | +4 to +14 for the last word; two thirds of the weight for a keyword elsewhere in the name |
| Looks like an identifier | −20 |
| Name is, or ends with, a time part (`year`, `month`, `week`, `day`, `hour`, `minute`, `quarter`) | −15 |
| Last word is an experimental-design label (`replicate`, `rep`, `batch`, `plate`, `well`, `run`, `trial`, `block`, `lane`, `row`, `column`, `col`, or `position` after one of these) | −15 |
| Any word is a coordinate (`latitude`, `longitude`, `lat`, `lon`, `lng`) | −15 |
| Any word is a condition usually recorded beside the outcome (`temp`, `temperature`, `humidity`, `wind`, `air`) | −3 |

Ties are broken by: whether the column has fractional values (amounts do;
postal codes and counters do not), then the share of non-missing values, then
the column name alphabetically. Two exports of one table with the columns in a
different order therefore detect the same measure.

Keyword weights (`MEASURE_KEYWORDS`): `revenue` 14, `mrr` 14, `arr` 14,
`sales` 14, `gmv` 14, `turnover` 14, `profit` 13, `margin` 12, `amount` 11,
`value` 10, `income` 10, `spend` 9, `total` 9, `cost` 8, `expense` 8,
`price` 7, `fee` 7, `fare` 7, `quantity` 6, `units` 6, `orders` 6, `volume` 6,
`balance` 6, `count` 6, `abundance` 6, `yield` 6, `concentration` 5, `score` 4.
`MEASURE_KEYWORDS` in `ada/schema.py` is the authoritative list.

**Scoring a name.** The last word decides: `Product Category` is a category,
`Product Container` is a container. A keyword found anywhere else in the name,
even inside a longer word, scores two thirds of its weight (`MODIFIER_WEIGHT`),
so it can still win when nothing better is present. Before scoring, a trailing annotation in parentheses or
brackets is removed (`Revenue (USD)` is scored on "revenue"), and camelCase is
split, so `netRevenue` scores like `net_revenue`.

The penalties prevent `Invoice ID` from outscoring `Revenue` because both are
numeric, and a `Month`, `Replicate` or `Latitude` column from becoming the
measure. The covariate penalty is light: in a survey with `count` and
`air_temp_c`, or an assay with `Concentration (mg/L)` and `Temp (°C)`, the
outcome wins even when the temperature has fewer gaps or more decimals, but a
temperature is still picked when it is the only thing measured. The identifier
check also looks at the last word: `Invoice Amount` ends in a measure keyword,
so it is an amount and keeps its score, while `Invoice Number` does not.

## Picking the segment

A column can be a segment when it is not the date, not numeric, not named like
a date (`date`, `time`, `timestamp`, `created` or `updated` as a word), and:

- has at least 2 distinct values, and
- has at most 100 distinct values, and
- has distinct values in at most 65% of its non-missing rows

The last rule excludes near-unique text such as notes, email addresses and
names, which would produce one bar per row.

Qualifying columns are scored:

| Signal | Effect |
|---|---|
| Name matches a dimension keyword | +5 to +12 (two thirds for a keyword before the last word) |
| Last word is an experimental-design label (`Row`, `Plate`, `Run`, ...) | −15 |
| Distinct count close to 10 | Preferred on a tie |

Keyword weights (`DIMENSION_KEYWORDS`): `product` 12, `category` 12,
`segment` 12, `region` 11, `channel` 10, `country` 10, `state` 9, `city` 9,
`customer` 8, `client` 8, `team` 7, `department` 7, `status` 6, `type` 5.

About ten groups reads well on a chart; two is thin, and sixty is too many
bars. A design label such as a plate's `Row` (A to H) groups wells by where
they sat, so it is picked only when no other column qualifies; `Sample` or
`Treatment` beside it wins.

## Picking the identifier

A column looks like an identifier when all of these hold:

- one of its name's words is `id`, `uuid`, `key`, `code`, `number`, `invoice`
  or `order`, matched as a whole word (so `Wide` is not an `id`)
- its last word is not a measure keyword
- at least 80% of its non-missing values are distinct
- it is not a datetime column

The first such column in the file wins. The role keeps identifiers out of the
measure slot and provides the "Distinct X" KPI, or, when it names each row
once, says what one row is ("One per Reading ID").

## Overriding

In the app, `pipeline.apply_role_selection` builds a new `ColumnRoles` from the
"Tune ADA's schema detection" dropdowns. `"None"` clears a role. The dashboard
re-renders without re-reading or re-cleaning the file.

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
| No numeric column | `measure` is `None`; ADA counts records instead |
| Every text column is near-unique | `dimension` is `None`; no segment analysis |
| Several plausible measures | Highest score wins; the user can override |

## Changing this

A weight change affects every file, so add a synthetic fixture that fails
before the change and passes after it. Keep fixtures synthetic; never commit
real customer data.
