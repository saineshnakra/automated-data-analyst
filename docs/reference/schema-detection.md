# Schema detection

**Code:** `schema.py` · **Tests:** `tests/test_analysis.py`, fixtures throughout

Works out what the columns of an unfamiliar file mean. Everything downstream
depends on getting this roughly right, and the user can correct it in the app.

The output is a `ColumnRoles`: `measure`, `date`, `dimension`, `identifier`,
plus the full lists `numeric` and `dimensions` for the override dropdowns.

## Why it takes three signals

No single signal is reliable:

- **Name alone** — a column called `Value` could be anything.
- **Type alone** — `Order Number` is numeric but summing it is nonsense.
- **Cardinality alone** — a column with 8 distinct values might be `Region` or
  might be `Rating`.

So names, types, and distinct-value counts are combined.

## Picking the date

Considers only real datetime columns (after cleaning has done its inference).
Ranks them by:

1. Name contains `date`, `time`, or `created` — this wins
2. Then: most non-missing values

## Picking the measure

Considers only numeric columns. Each gets a score:

| Signal | Effect |
|---|---|
| Name matches a measure keyword | +4 to +14 |
| Looks like an identifier | −20 |
| Name is a time part (`year`, `month`, `week`, `day`, `hour`, `minute`, `quarter`) | −15 |

Ties break on the proportion of non-missing values.

**Keyword weights** (`MEASURE_KEYWORDS`): `revenue` 14, `sales` 14, `gmv` 14,
`profit` 13, `margin` 12, `amount` 11, `value` 10, `income` 10, `spend` 9,
`cost` 8, `expense` 8, `price` 7, `quantity` 6, `units` 6, `orders` 6,
`volume` 6, `balance` 6, `score` 4.

The two penalties are what stop the obvious failures: `Invoice ID` outscoring
`Revenue` because both are numeric, and a `Month` column being treated as the
thing to measure.

## Picking the segment

A column qualifies as a possible segment when it is not the date, not numeric,
and:

- has **at least 2** distinct values, and
- has **at most 100** distinct values, and
- distinct values are **at most 65%** of non-missing values

That last rule removes near-unique text columns — free-text notes, email
addresses, names. Grouping by them produces one bar per row.

Qualifying columns are then scored:

| Signal | Effect |
|---|---|
| Name matches a dimension keyword | +5 to +12 |
| Distinct count close to 10 | Preferred |

**Keyword weights** (`DIMENSION_KEYWORDS`): `product` 12, `category` 12,
`segment` 12, `region` 11, `channel` 10, `country` 10, `state` 9, `city` 9,
`customer` 8, `client` 8, `team` 7, `department` 7, `status` 6, `type` 5.

Around ten groups is preferred because that is what reads well on a chart. Two
groups is thin, sixty is a wall of bars.

## Picking the identifier

A column looks like an identifier when **both** hold:

- one of its name's words is `id`, `uuid`, `key`, `code`, `number`, `invoice`,
  or `order` — matched as a whole word, so `Wide` is not an `id`
- **at least 80%** of its non-missing values are distinct

The first such column wins. The role is mostly used to keep identifiers out of
the measure slot, and to power the "Distinct X" KPI.

## Overriding

`pipeline.apply_role_selection` builds a new `ColumnRoles` from the dropdown
values in the app's "Tune ADA's schema detection" panel. `"None"` clears a role.
The dashboard re-renders; nothing is rebuilt from scratch.

`schema.override_roles` does the same thing for programmatic callers.

## Edge cases

| Situation | Behavior |
|---|---|
| No datetime column | `date` is `None`; trend, anomalies, and forecast are skipped |
| No numeric column | `measure` is `None`; ADA falls back to record-count framing |
| Every text column is near-unique | `dimension` is `None`; no segment analysis |
| Several plausible measures | Highest keyword score wins; the user can override |

## Changing this

Adjusting a weight changes behavior across the whole product, so add a synthetic
fixture that fails before the change and passes after. Keep fixtures synthetic —
never commit real customer data.
