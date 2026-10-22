# Formatting

**Code:** `ada/formatting.py` · **Tests:** `tests/test_formatting.py`, `tests/test_business_insights.py`

Decides how numbers and dates are written. Kept separate from the
calculations.

## The rule

Formatting changes how a value looks, never what it is. The module holds no
state, and every surface that displays a number (the dashboard, evidence
cards, Ask ADA answers, the Markdown report) calls the same functions, so a
figure reads the same everywhere.

## `format_number(value, column=None, *, compact=True, column_values=None, extra_decimals=0)`

`column_values` is the whole column the figure came from. It settles questions
that one value cannot: whether a percentage column stores `0.25` or `25`,
whether a column named like a rate holds money, whether a currency-named
column holds money at all, and whether a column holds whole counts.

| Input | Output |
|---|---|
| `1_250_000, "Revenue"` | `$1.2M` |
| `1.5e12, "Revenue"` | `$1.5T` |
| `-1_200, "Expense Amount"` | `-$1.2K` |
| `2_500, "Revenue EUR"` | `€2.5K` |
| `8.11, "income"`, values coded 1 to 15 | `8.11` |
| `12_000, "Units"` | `12.0K` |
| `1_250, "Orders"` | `1,250` |
| `1_111, "Heart rate"` | `1,111` |
| `1_944.66, "Looking Time"` | `1,944.7` |
| `55_034.3, "result"` | `55,034` |
| `3.624e11, "1001"` | `3.624e+11` |
| `45, "Units"` | `45` |
| `45.5, "Score"` | `45.50` |
| `0.0012, "Absorbance"` | `0.0012` |
| `0.000123, "Concentration"` | `1.23e-04` |
| `0.33, "Profit Margin %"` | `33.0%` |
| `float("nan")` | an em dash (U+2014) |

In order:

1. A non-finite value becomes an em dash.
2. A percentage column (see below) is printed with one decimal and `%`.
3. Money (see below) gets its symbol before the number and the sign before the
   symbol. With `compact`, amounts from 1,000 up are scaled to `K`, `M`, `B` or
   `T` with one decimal; from 10^15 up, scientific notation is used.
4. With `compact`, a count from 10,000 up (`COMPACT_COUNT_FROM`) is scaled the
   same way: a whole number from a column whose name marks a count (`Units`,
   `Orders`; see `is_additive_name` in `ada/metrics.py`) and whose values are
   all whole, or a row count with no column.
5. Anything else keeps its digits, to about five significant figures. Below
   0.001 or from 10^9 up it is written in scientific notation with four
   significant figures (`SCIENTIFIC_BELOW`, `SCIENTIFIC_FROM`).
6. Whole numbers print without decimals.
7. Values below 1 print with three significant figures, so `0.0012` does not
   become `0.00`.
8. Values print with two decimals below 1,000, one below 10,000 and none above,
   with thousands separators.

So `K`, `M`, `B` and `T` appear only on money and large counts. A reading such
as a heart rate of 1,111, a looking time of 1,944.7 seconds or a spectral
intensity of 3.624e+11 is never shortened to `1.1K`, `1.9K` or `362.4B`.

`extra_decimals` adds decimal places (or significant figures, in scientific
notation) to the steps that print them. It is used by `format_values_apart`.

## `format_values_apart(*values, column=None, column_values=None)`

Formats figures quoted together: the two ends of "from X to Y", a value and
the range it fell outside, or the highest and lowest group average. At the
usual precision 3,004.25 and 3,004.21 both print as `3,004.3`, and two revenue
figures of $2,700,100 and $2,700,400 both print as `$2.7M`, which would hide
the difference the sentence reports.

When any two different values print alike, all of them are written out in
full with the same precision (`3,004.25` and `3,004.21`), adding up to four
decimals (`MAX_EXTRA_DECIMALS`) until they differ. Equal values keep the usual
rendering, and so do values that differ only past the fourth extra decimal.

## Money

`is_money(column, column_values=None)` is true when the name is a currency
name (below) and the values look like amounts. A currency-named column whose
values are all whole numbers from 0 to 15 (`MAX_BAND_CODE`) holds coded bands,
such as an income bracket, and is printed as a plain number: its average is
`8.11`, not `$8.11`. Without `column_values` the name decides.

## Currency detection

`is_currency` checks whether any whole word of the normalized column name is
one of:

`revenue`, `mrr`, `arr`, `turnover`, `sales`, `gmv`, `profit`, `amount`,
`income`, `spend`, `cost`, `expense`, `price`, `balance`

or a currency code (`usd`, `eur`, `gbp`). `currency_symbol` picks `€` or `£`
when the name contains that code or symbol, and `$` otherwise. `is_money`
(above) adds the check on the values before a symbol is printed.

## Percentages

`is_percentage` is true for a name containing `%` or the word `rate`,
`margin` or `ratio`.

A name can carry both signals: `Profit Margin %` contains a currency word and
a percentage one. A `%` sign, or a percentage word as the last word, makes it a
percentage, so `Profit Margin` is a margin and `Margin Amount` is an amount.
Matching is on whole words, so `rate` does not match `Corporate Revenue` and
`eur` does not match `Europe Sales`.

The scale is decided once per column from `column_values`, not per value: if
every value is between 0 and 1, the column stores fractions (`0.25` is 25%);
otherwise it stores points (`25` is 25%). If any value in a rate-named column
exceeds `MAX_PLAUSIBLE_PERCENTAGE` (1,000), the column is printed as plain
numbers instead of `1250000.0%`.

`format_rate_change` writes a change in a rate as percentage points
("20.0 percentage points"), since 20% to 40% is 20 points but a 100% relative
increase.

## `normalized_name(name)`

Lowercases, splits camelCase (`netRevenue` becomes `net revenue`, `customerID`
becomes `customer id`, `pH` is left whole), turns `_` and `-` into spaces, and
collapses whitespace. `ada/schema.py` and `ada/metrics.py` use it too, so
column names are compared the same way everywhere: `Total_Revenue` and
`total revenue` are the same name.

## Names in sentences

`sentence_case(name)` lowercases the ordinary capitalised words of a column
name for use mid-sentence: `Water Temp (°C)` becomes `water temp (°C)`. A word
is left as written when it has other capitals (`SKU`, `pH`, `CO`), a digit
(`NO2`, `PM2.5`, `Co2`), or three letters or fewer (`MAX_SYMBOL_LETTERS`),
since short capitalised words are more often symbols and codes (`Ca`, `Na`,
`Gp`, `Hgb`) than ordinary words.

`group_label(column, value)` names a group: a numeric code with its column
(`Status 2`), a text value as it is. Every sentence that names a group uses it.

`with_aggregate(word, column)` puts an aggregate word before a column name
unless the name already says it: `Total` and `Total Rain` give `Total Rain`,
`Average` and `Mean Temp` give `Mean Temp`.

`list_names(names, limit=3)` joins names for a sentence: `A and B`,
`A, B and C`, `A, B, C and 2 others`. Rankings use it to name tied groups.

## `format_period(period, grain)`

| Grain | Output |
|---|---|
| `Q` | `Q3 2026` |
| `W`, `D` | `05 Sep 2026` |
| `Y` | `2026` |
| `h`, `min` | `05 Sep 2026 14:00` |
| `s` | `05 Sep 2026 14:00:05` |
| any other | `Sep 2026` |

### Times with no calendar date

A column of clock times with no date, or of elapsed time, is placed on
`UNDATED_DAY` (1 January 1900, the date `strptime` gives a bare time) so it
sorts and groups like any timestamp. `is_undated` recognises these, and
labels on them never show that date: `15:47`, `15:47:03`, `day 2 00:15` once
a log passes midnight, `day 3` at a daily grain. `format_moment` writes one
end of a date range the same way (`05 Mar 2024`, or `15:06:28`). Real
readings from 1900 at a grain finer than a day would be labelled as undated
too.

## Known gaps

- A rate or level column gets no total KPI, because its values do not add up.
  Its average still appears.

## Changing this

Any change here:

- must not rescale stored data; only the printed string changes
- must be consistent within a column, whatever the sign or size of a value
- must not change how existing ambiguous columns are formatted
- needs tests for negatives, zero, whole numbers and the compact boundaries
