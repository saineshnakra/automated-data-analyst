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

Timezone offsets are dropped at this point, keeping the wall clock the file was
written in. Converting to UTC first would move rows east of Greenwich into the
previous calendar day, changing which period they belong to.

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
Whatever plain parsing cannot read is retried allowing the punctuation a
finance export writes: thousands separators, a currency symbol, and
parentheses for a negative. `$1,203.55` and `(48.10)` are numbers; the values
that need this are the large ones, so leaving them out biases every total
downwards. A value holding a slash is never read this way, so a date cannot
become a very large integer. Skipped entirely when the name contains `id`,
`code`, `zip`, `postal`, or `phone`.

Why the difference in thresholds? Dates arrive in mixed formats and a stricter
bar would reject real date columns. Numbers are unambiguous, so a column that is
5% unparseable is probably not really numeric.

Why protect `zip` and `phone`? Because `01234` is a postcode, and converting it
to `1234` destroys it.

## The cleaning report

`CleaningReport` records original and final row/column counts plus a count for
every operation: duplicates removed, empty rows removed, empty columns removed,
index columns removed, text columns trimmed, numeric columns inferred, datetime
columns inferred.

`pipeline.cleaning_audit_frame` turns it into the table shown in the app. Nothing
is changed without a number appearing here.

## Edge cases

| Situation | Behavior |
|---|---|
| Empty input | `ValueError` — "does not contain any rows and columns to analyze" |
| Everything dropped by cleaning | `ValueError` — "No analyzable data remained" |
| Column that is 90% numeric | Left as text — below the 95% bar |
| `Order Date` that is 85% parseable | Converted — above the 80% bar |
| Column named `Customer ID` holding numbers | Left as text — protected token |

## Changing this

New cleaning steps must be counted in `CleaningReport` and surfaced in
`cleaning_audit_frame`. A step that cannot be explained in one line of the audit
table is too clever for this module.
