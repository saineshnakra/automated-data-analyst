# Cleaning

**Code:** `clean_dataframe` in `ada/analysis.py`
**Tests:** `tests/test_analysis.py`

Tidies the data before analysis and counts each change. A column ADA is unsure
about is left as it is, and every change is reported in the cleaning audit
table or as a note.

## Steps, in order

| # | Step | Detail |
|---|---|---|
| 1 | Normalize and de-duplicate column names | Whitespace collapsed; a blank name becomes `column_<position>`; a repeated name gets `_2`, `_3` and so on |
| 2 | Drop empty columns | Every value missing or whitespace |
| 3 | Drop empty rows | Every value missing or whitespace |
| 4 | Drop exported index columns | See below |
| 5 | Drop timezones | Timezone-aware columns keep their wall-clock time |
| 6 | Trim text | Leading and trailing whitespace removed; empty strings become missing |
| 7 | Infer dates in date-named columns | See below |
| 8 | Infer numbers | See below |
| 9 | Infer dates in other columns | See below |
| 10 | Build dates from years | See below |
| 11 | Widen overflowing integers | An integer column whose total would overflow int64 becomes float64 |
| 12 | Replace infinities | `inf` and `-inf` in float columns become missing |
| 13 | Count identical rows | Counted and kept |

If nothing is left, `clean_dataframe` raises `ValueError` rather than
returning an empty frame.

## Identical rows are kept

Two sales of the same item, for the same amount, on the same day are normal in
transaction data. Deleting them would remove real revenue from every total. So
identical rows are counted, reported as "Identical rows kept", and left in.
`clean_dataframe(frame, drop_duplicates=True)` removes them, for a caller whose
rows carry a key.

## Dates that can be read two ways

`01/03/2024` is 1 March in most of the world and 3 January in the United
States. When a column has values in this form (separated by `/` or `-`), ADA
parses it both ways and uses the reading that parses more values, so a column
containing `25/12/2024` is read day-first. When both readings parse equally
but give different dates, month-first is used and a note says so, suggesting
ISO dates or a renamed column. Dot-separated dates (`13.01.2022`) are left to
pandas' default parsing.

Timezone offsets are dropped, keeping the wall-clock time written in the file.
Converting to UTC first would move rows east of UTC into the previous day, and
so into a different period. A column that mixes offsets (for example across a
daylight-saving change) is converted to UTC, since it has no single wall clock.

## Exported index columns

Saving a DataFrame to CSV with its index produces a first column named
`Unnamed: 0` holding row numbers. A column is dropped only when both hold:

- its name starts with `unnamed` (any case), and
- its values are whole numbers with nothing missing, starting at 0 or 1,
  strictly increasing, and dense (the last value minus the first is less than
  twice the number of rows); gaps left by removed blank rows are allowed

A sequential column with a real name is kept, since it may be a genuine
counter.

## Type inference

Inference runs only on text columns.

**Dates in date-named columns.** A column whose name contains `date`, `time`,
`timestamp`, `created` or `updated` is converted when at least 80% of its
non-missing values parse. Values that do not parse are counted and reported.

**Numbers.** A column is converted when at least 95% of its non-missing values
parse. Values plain parsing misses are retried as formatted numbers: thousands
separators (`,`, `.`, space or apostrophe), a decimal comma, a leading
currency symbol (`$`, `€`, `£`, `¥`, `₹`), a sign, and parentheses for a
negative. `$1,203.55`, `(48.10)` and `1 234,50` are all numbers. A value with
any other character, such as a slash, is not read this way, so a date cannot
become a large integer.

Cells holding a missing-value marker (`MISSING_MARKERS`: `NA`, `n/a`, `nd`,
`null`, `-`, `#DIV/0!`, `<LOD`, `below detection limit` and similar, or a
censored value such as `<0.1`) do not count against the 95%. If the column
converts, they become missing and a note lists them.

A column whose name ends in a key word (`id`, `code`, `zip`, `postcode`,
`phone`, `number`, `no`, `#` and others; `is_identifier_name` in
`ada/schema.py`) is never converted to numbers, and neither is a column with a
value that has a leading zero (`02134`, `007`).

**Dates in other columns.** Dates also arrive in columns called `Month`,
`Period` or `FY`. For a column with any other name, the first 50 values must
parse as dates at a 95% rate, and then 95% of the whole column. Numbers are
tried first, so a column of bare years stays numeric here; the next step
decides whether it is a timeline.

Dates use a lower bar because real date columns often mix formats. A column
that is 5% unparseable as numbers is probably not numeric.

`01234` is a postcode, and converting it to `1234` would change it. The reader
also keeps such columns as text from the start: see
[Reading files](reading-files.md).

## Dates from years

Annual data often has a `year` column of plain numbers and no date. When no
column is a date after inference, a numeric column becomes one if all of these
hold:

- its name ends in `year`, `years` or `yr` (`Year`, `survey_yr`)
- every value is a whole number from 1800 to 2100 (`DETECTED_YEARS`)
- it has at least two distinct values

Each year becomes 1 January of that year, and a note says so. If a column
named `month`, `months` or `mon` holds month names (`January`), abbreviations
(`Jan`) or the numbers 1 to 12 for every row with a year, the two are combined
into one column of monthly dates, named after both (`year-month`), and the
month column is dropped. When several columns qualify, the one whose name is
only the year word wins, then the first alphabetically.

The name and the range both matter. `model_year` holding 70 to 82 is not a
year, and a count that happens to fall between 1800 and 2100 is not a year
unless its name says so.

A column passed as the date (`ada.analyze(df, date="Season")`) is converted
whatever its name, as long as every value is a whole number from 1000 to 2999
(`REQUESTED_YEARS`). It is used alone, without a month column. Years outside
what a timestamp can hold (about 1678 to 2261) become missing and are reported
as unreadable dates.

## The cleaning report

`CleaningReport` records the original and final row and column counts, and a
count for each operation: empty rows, empty columns and index columns removed,
text columns whose values trimming changed, numeric and datetime columns
inferred, identical rows found and removed, unreadable date values and
infinite values. Its `notes` hold the sentences shown above the dashboard
(date-order assumptions, timezones, unreadable dates, dates built from years,
missing-value markers, widened columns, infinities, identical rows).

`pipeline.cleaning_audit_frame` turns the counts into the table shown in the
app.

## Edge cases

| Situation | Behavior |
|---|---|
| Empty input | `ValueError`: "The CSV does not contain any rows and columns to analyze." |
| Everything dropped by cleaning | `ValueError`: "No analyzable data remained after removing empty rows and columns." |
| Column that is 90% numeric | Left as text (below 95%) |
| `Order Date` that is 85% parseable | Converted (above 80%), with the unreadable count reported |
| `Customer ID` holding numbers | Left as text |
| `Year` holding 1880 to 2023, and no date column | Converted to 1 January of each year, with a note |
| `year` and `month` (`January` ...), and no date column | Combined into a monthly `year-month` column |
| `model_year` holding 70 to 82 | Left as numbers |
| A cell reading `inf` | Treated as missing, with a note |

## Changing this

A new cleaning step must be counted in `CleaningReport` and shown in
`cleaning_audit_frame` or the notes. If a step cannot be explained in one line
of the audit table, it does not belong in this module.
