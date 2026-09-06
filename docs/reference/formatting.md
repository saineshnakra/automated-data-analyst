# Formatting

**Code:** `formatting.py` · **Tests:** in `tests/test_business_insights.py`

Decides how numbers and dates are written down. Deliberately kept apart from the
calculations.

## The rule

**Formatting changes how a value looks. It never changes what the value is.**

Changing a display should never be able to change a result. That is why this
module is separate from the analysis modules, and why it holds no state.

Every module that displays a number — the dashboard, the evidence cards, the
chat answers, the Markdown report — calls the same two functions, so a figure
reads identically everywhere it appears.

## `format_number(value, column=None, *, compact=True, column_values=None)`

`column_values` is the whole column the figure came from. It is what settles
questions that cannot be answered one value at a time — whether a percentage
column stores `0.25` or `25`, and whether a column named like a rate is really
holding money.

| Input | Output |
|---|---|
| `1_250_000, "Revenue"` | `$1.2M` |
| `1.5e12, "Revenue"` | `$1.5T` |
| `-1_200, "Expense Amount"` | `-$1.2K` |
| `12_000, "Units"` | `12.0K` |
| `1_250, "Orders"` | `1.2K` |
| `45, "Units"` | `45` |
| `45.5, "Score"` | `45.50` |
| `0.33, "Profit Margin %"` | `33.0%` |
| `float("nan")` | `—` |

What it does:

1. Non-finite values become an em dash
2. Currency columns get a `$` prefix
3. When `compact`, values are scaled to `K`, `M`, or `B` at 1,000 / 1,000,000 /
   1,000,000,000
4. Whole numbers in non-currency columns print without decimals
5. Everything else prints with 2 decimals and thousands separators

## Currency detection

`is_currency` checks whether the normalized column name contains any of:

`revenue`, `sales`, `gmv`, `profit`, `amount`, `income`, `spend`, `cost`,
`expense`, `price`, `balance`

## `normalized_name(name)`

Lowercases, turns `_` and `-` into spaces, and collapses runs of whitespace.
Used here and in `schema.py`, so both compare column names the same way. It
compares meaning rather than punctuation: `Total_Revenue` and `total revenue`
are the same column name.

## `format_period(period, grain)`

| Grain | Output |
|---|---|
| `Q` | `Q3 2026` |
| `W`, `D` | `05 Sep 2026` |
| `Y` | `2026` |
| anything else | `Sep 2026` |

Periods are named the way a reader would say them out loud.

## Percentages and currencies

Both are implemented, and both were harder than they looked.

**Which reading wins.** A name can carry both signals: `Profit Margin %` holds
a currency word and a percentage one. An explicit `%`, or a percentage word in
the **head** position, settles it — so `Profit Margin` is a margin and
`Margin Amount` is an amount. Matching is on whole words, so `rate` does not
match `Corporate Revenue` and `eur` does not match `Europe Sales`.

**0–1 versus 0–100** is settled **once per column** from `column_values`, never
per value. Deciding per value rendered one column as `551.3%` on one row and
`1.4K` on the next. The same column-level rule catches a column named like a
ratio that is really holding money: above `MAX_PLAUSIBLE_PERCENTAGE` the whole
column falls through to plain numbers rather than printing `1250000.0%`.

**Sign placement.** A debt is `-$1.2M`, not `$-1.2M` — the minus belongs to the
amount, not the currency.

## Known gaps

- A rate column gets no *total* KPI, because adding percentages up means
  nothing. Its average still appears.

## Changing this

Whatever is added here:

- must not rescale stored data — only the printed string changes
- must be consistent within a column, whatever the sign or magnitude of a value
- must keep ambiguous columns behaving as they do today
- needs tests for negatives, zero, whole numbers, and the compact boundaries
