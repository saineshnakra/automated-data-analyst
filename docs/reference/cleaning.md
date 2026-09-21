# Cleaning

**Code:** `analysis.py` → `clean_dataframe` · **Tests:** `tests/test_analysis.py`

Tidies the data before analysis, and counts every change it makes.

The guiding rule: **conservative and visible**. ADA would rather leave a messy
column alone than silently reinterpret it. Everything it does change is reported
in the cleaning audit table.

## Steps, in order

| # | Step | Detail |
|---|---|---|
| 1 | Normalize and de-duplicate column names | Whitespace collapsed; a blank name becomes `column_<position>`; a repeated name gets `_2`, `_3`, and so on |
| 2 | Drop fully empty columns | Every value missing |
| 3 | Drop fully empty rows | Every value missing |
| 4 | Drop exported index columns | See below |
| 5 | Trim text | Leading/trailing whitespace removed; empty strings become missing |
| 6 | Infer dates | See below |
| 7 | Infer numbers | See below |
| 8 | Count duplicate rows | Exact duplicates across all columns are **counted and kept** |

If nothing analyzable survives, `clean_dataframe` raises `ValueError` rather
than returning an empty frame.

## Why duplicate rows are kept

Two sales of the same item, for the same amount, on the same day are an
ordinary Tuesday at a till — not a defect. Deleting them removes real revenue
from every number on the page, silently, and on a point-of-sale file that can
be half the total. So they are counted, reported in the cleaning audit as
"Identical rows kept", and left alone.

`clean_dataframe(frame, drop_duplicates=True)` opts in, for a caller that knows
its rows carry a key.

## Dates that could be read two ways

`01/03/2024` is 1 March in most of the world and 3 January in the United States.
When some value in the column settles it — anything above 12 in the first
position — that reading wins outright. When nothing settles it, month-first is
assumed **and the assumption is reported**, because a year of monthly figures
collapsing into twelve days of January is the one error no later step can
detect.

## Timezone offsets

The policy is **per-row wall clock, always**. The offset text on each value
(`+01:00`, `+0100`, `+01`, `Z`) is removed before parsing, so every row keeps
the local time it was written in, and one note says that offsets were
dropped. A column that arrives already timezone-aware is localised to naive
the same way.

Why not convert to UTC? Because every downstream calculation compares a date
against a period boundary built without a timezone, so UTC moves rows east of
Greenwich into the previous calendar day. And why never *sometimes* UTC? An
earlier version kept wall clocks when every row shared one offset and fell
back to UTC once the offsets were mixed -- so appending a single summer row
to a winter file moved every existing row an hour, some of them into the
previous month. What a row means must not depend on the rows after it.

## The exported-index rule

Saving a DataFrame to CSV with the index included produces a stray first column
named `Unnamed: 0` holding `0, 1, 2, 3...`. It is not data.

A column is dropped only when **both** are true:

- its name starts with `unnamed`, and
- its values are exactly `0, 1, 2, ... n-1` with no gaps and nothing missing

Both conditions are needed. A column of sequential numbers with a real name
might be a genuine counter.

## Type inference thresholds

Inference happens per column, and only on text columns.

**Dates** — a column named `date`, `time`, `timestamp`, `created` or `updated`
is converted when **at least 80%** of non-missing values parse. A column with
any other name is also tried, because dates arrive in columns called `Month`,
`Period` and `FY`: a 50-value sample has to parse first, and then **at least
95%** of the column. Numbers are tried before that second pass, so a column of
bare years stays numeric instead of becoming the 1st of January in each.

**Numbers** — converted when **at least 95%** of non-missing values parse.
The denominator is counted *after* trimming, so a whitespace-only cell is a
blank and not a value that failed. Whatever plain parsing cannot read is
retried with the business-number grammar below. The values that need it are
the large ones, so leaving them out biases every total downwards.

Once a column is converted, every non-missing cell that still could not be
read is **counted** (`numeric_cells_unreadable`) and a note names the column
and the count. `inf` and `-inf` -- which pandas reads as numbers -- are
treated as missing, counted separately (`non_finite_cells`), and noted.

Why the difference in thresholds? Dates arrive in mixed formats and a stricter
bar would reject real date columns. Numbers are unambiguous, so a column that is
5% unparseable is probably not really numeric.

### The business-number grammar

A value is read as a number when it is, in this order: at most one sign, at
most one currency symbol (`$ € £ ¥ ₹`), and at most one pair of accounting
parentheses, in any order (`-$100`, `$-100`, `($100)`, `-(100)` are all a
hundred owed), followed by digits with grouping marks. A slash or a letter
anywhere means it is not a number, so a date cannot become a very large
integer and `1e3` is left to plain parsing, which reads it.

The digits must satisfy the grammar, or the value is refused and counted:

- Marks between digit groups are `,` `.` space and `'`. Each mark sits between
  two digits; a doubled mark, or a mark at either end, is malformed.
- When both `,` and `.` appear, the one that comes last is the decimal mark
  and the other is grouping: `1,234.50` and `1.234,50` are both 1234.5.
- With one mark appearing once: `1,000` is a thousand and `1234,50` is a
  decimal (one or two digits after a comma); a single `.` is always a decimal,
  as `pd.to_numeric` reads it, so `1.234` is 1.234. A space or an apostrophe
  is only ever grouping.
- Grouping marks must delimit groups of **exactly three digits** after the
  first group, and there is at most one decimal mark. `1,2,3`, `1.2.3`,
  `12 34` and `1,000.000,50` are refused; the old parser read them as 123,
  123, 1234 and something else again.
- Parentheses must balance: `(100` and `100)` are refused.

When the formatted reader contributed values, the column is a float whatever
the plain values were, and a whole number at or past 2^53 has lost its last
digits. A note says so, in the same words as the overflow-widening note.

### Columns that are keys

A column whose name says it holds a key is never converted -- neither to a
number nor, in the unnamed date pass, to a date. `Customer ID` full of
`2024-01-01` values is a batch of ids, not the file's date. The test is
`schema.is_identifier_name`: the **last word** (or the whole name) is one of
`id`, `ids`, `code`, `codes`, `zip`, `postal`, `phone`, `sku`, `number`, `no`,
`key`, `uuid`, matched as a whole word. `Order No` and `Account Number` are
keys; `Paid Amount`, `Grid Value` and `Valid Amount` are not, although each
contains the letters `id`. The same function decides which columns
`file_io` reads as text, so an account number written `00042` is `00042` from
upload to chart, and nothing the reader preserved is undone here.

Why protect `zip` and `phone`? Because `01234` is a postcode, and converting it
to `1234` destroys it.

## The cleaning report

`CleaningReport` records original and final row/column counts plus a count for
every operation: duplicates removed, empty rows removed, empty columns removed,
index columns removed, text columns trimmed, numeric columns inferred, datetime
columns inferred, date cells that could not be read (`unparsed_date_cells`),
numeric cells coerced to missing (`numeric_cells_unreadable`), and infinite
values made missing (`non_finite_cells`). Each of the last three also gets a
note naming the column.

`pipeline.cleaning_audit_frame` turns it into the table shown in the app. Nothing
is changed without a number appearing here.

## Edge cases

| Situation | Behavior |
|---|---|
| Empty input | `ValueError` — "does not contain any rows and columns to analyze" |
| Everything dropped by cleaning | `ValueError` — "No analyzable data remained" |
| Column that is 90% numeric | Left as text — below the 95% bar |
| `Order Date` that is 85% parseable | Converted — above the 80% bar |
| `["$100", "$200", "  "]` | Converted — the blank is not counted against the bar |
| Column named `Customer ID` holding numbers | Left as text — its name says key |
| Column named `Customer ID` holding `2024-01-01` values | Left as text — never the date |
| Column named `Paid Amount` holding `$1,000` | Converted — `id` inside a word is not a word |
| `$-100` among `$200` values | −100, sign after the currency is read |
| `(100` among `200` values | Missing, counted in `numeric_cells_unreadable`, noted |
| `inf` among numbers | Missing, counted in `non_finite_cells`, noted |
| Nineteen `9007199254740993` and one `$1` | Float column; note says the last digits are approximate |
| `+01:00` rows, then a `+02:00` row appended | Existing rows unchanged — every row keeps its wall clock |

## Changing this

New cleaning steps must be counted in `CleaningReport` and surfaced in
`cleaning_audit_frame`. A step that cannot be explained in one line of the audit
table is too clever for this module.
