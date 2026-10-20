# Formatting

**Code:** `ada/formatting.py` · **Tests:** `tests/test_formatting.py`, `tests/test_business_insights.py`

Decides how numbers and dates are written. Kept separate from the
calculations.

## The rule

Formatting changes how a value looks, never what it is. The module holds no
state, and every surface that displays a number (the dashboard, evidence
cards, Ask ADA answers, the Markdown report) calls the same functions, so a
figure reads the same everywhere.

## `format_number(value, column=None, *, compact=True, column_values=None)`

`column_values` is the whole column the figure came from. It settles questions
that one value cannot: whether a percentage column stores `0.25` or `25`, and
whether a column named like a rate holds money.

| Input | Output |
|---|---|
| `1_250_000, "Revenue"` | `$1.2M` |
| `1.5e12, "Revenue"` | `$1.5T` |
| `-1_200, "Expense Amount"` | `-$1.2K` |
| `2_500, "Revenue EUR"` | `€2.5K` |
| `12_000, "Units"` | `12.0K` |
| `1_250, "Orders"` | `1.2K` |
| `45, "Units"` | `45` |
| `45.5, "Score"` | `45.50` |
| `0.0012, "Absorbance"` | `0.0012` |
| `0.33, "Profit Margin %"` | `33.0%` |
| `float("nan")` | an em dash (U+2014) |

In order:

1. A non-finite value becomes an em dash.
2. A percentage column (see below) is printed with one decimal and `%`.
3. A currency column gets its symbol before the number and the sign before the
   symbol.
4. With `compact`, values from 1,000 up are scaled to `K`, `M`, `B` or `T` with
   one decimal; from 10^15 up, scientific notation is used.
5. Whole numbers in other columns print without decimals.
6. Values below 1 print with three significant figures, so `0.0012` does not
   become `0.00`.
7. Everything else prints with two decimals and thousands separators.

## Currency detection

`is_currency` checks whether any whole word of the normalized column name is
one of:

`revenue`, `mrr`, `arr`, `turnover`, `sales`, `gmv`, `profit`, `amount`,
`income`, `spend`, `cost`, `expense`, `price`, `balance`

or a currency code (`usd`, `eur`, `gbp`). `currency_symbol` picks `€` or `£`
when the name contains that code or symbol, and `$` otherwise.

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

## `format_period(period, grain)`

| Grain | Output |
|---|---|
| `Q` | `Q3 2026` |
| `W`, `D` | `05 Sep 2026` |
| `Y` | `2026` |
| any other | `Sep 2026` |

## Known gaps

- A rate or level column gets no total KPI, because its values do not add up.
  Its average still appears.

## Changing this

Any change here:

- must not rescale stored data; only the printed string changes
- must be consistent within a column, whatever the sign or size of a value
- must not change how existing ambiguous columns are formatted
- needs tests for negatives, zero, whole numbers and the compact boundaries
