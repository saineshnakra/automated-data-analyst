# Cleaning

**Code:** `clean_dataframe` in `ada/analysis.py`
**Tests:** `tests/test_analysis.py`

Tidies the data before analysis and counts each change. A column ADA is unsure
about is left as it is, and every change is reported in the cleaning audit
table or as a note.

## Steps, in order

| # | Step | Detail |
|---|---|---|
| 1 | Normalize and de-duplicate column names | Whitespace collapsed; a blank name becomes `column_<position>`; a repeated name gets `_2`, `_3` and so on. A note names each repeated header, and says whether the two columns differ |
| 2 | Drop empty columns | Every value missing or whitespace; a note names them |
| 3 | Drop empty rows | Every value missing or whitespace |
| 4 | Drop exported index columns | See below |
| 5 | Drop timezones | Timezone-aware columns keep their wall-clock time |
| 6 | Trim text | Leading and trailing whitespace removed; empty strings become missing |
| 7 | Read times of day | See [Times with no date](#times-with-no-date) |
| 8 | Infer dates in date-named columns | See below |
| 9 | Infer numbers | See below |
| 10 | Infer dates in other columns | See below |
| 11 | Join a date column and a time column | See [Times with no date](#times-with-no-date) |
| 12 | Read elapsed time | See [Elapsed time](#elapsed-time) |
| 13 | Read "not stated" as missing in text columns | See below |
| 14 | Build dates from years | See below |
| 15 | Merge label spellings | See below |
| 16 | Widen overflowing integers | An integer column whose total would overflow int64 becomes float64 |
| 17 | Replace infinities | `inf` and `-inf` in float columns become missing |
| 18 | Count identical rows | Counted and kept |

If nothing is left, `clean_dataframe` raises `ValueError` rather than
returning an empty frame.

Steps 6 to 10 work on each distinct value of a text column once and spread
the result back over its rows, since a site, a status or a date repeats many
times. Before a column is parsed in full as numbers or as times of day, 200 of
its cells drawn at random are tried. A column needs 95% of its cells to
convert, so one where fewer than half of the drawn cells read as numbers (or
times) is not parsed in full. On a file of 700,000 rows this takes cleaning
from about 8 seconds to under 3.

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

A column may mix ISO dates with and without a time of day
(`2015-01-01 01:00:00` and later `2019-04-02`). pandas infers one format from
the first value, so values in the other ISO form are parsed again on their
own rather than reported as unreadable.

Timezone offsets are dropped, keeping the wall-clock time written in the file.
Converting to UTC first would move rows east of UTC into the previous day, and
so into a different period. A column that mixes offsets (for example across a
daylight-saving change) is converted to UTC, since it has no single wall clock.

## Times with no date

A column of clock times with no date (`15:47:03`, `3:05 PM`, or a spreadsheet
time cell) is read as times of day. pandas would put today's date on each, so
a log recorded on an unknown day would claim to be from today; ADA never
attaches a date. The times are placed on `UNDATED_DAY` (see
[Formatting](formatting.md#times-with-no-calendar-date)), every label shows
the time alone, and a note says the column has no calendar date and what
span it covers. At least 80% of a date-named column's values, or 95% of any
other column's, must be bare times.

A log in time order that passes midnight (`23:59:59` then `00:00:00`) moves
on to the next day, so its order survives; the note counts the midnights. A
column whose times step back more than occasionally is not in time order, and
its times all stay on one day.

A bare time inside a column of full dates is not a date at all, so it is
counted as unreadable rather than given today's date.

When a table has one column of bare times and one column of plain dates (a
`Date` and a `Time`), the two are combined into the date column, which then
holds the date and time of each row, and the time column is dropped.

## Elapsed time

A numeric column of time since the start of a recording can be the time axis.
It is read as elapsed time when its name is made only of time words (`time`,
`elapsed`, `t`, with `since`, `start`, `from` or `run`) and one unit (`ms`,
`s`, `sec`, `min`, `h`, `hr`, `hours` and similar): `Time (s)`,
`elapsed_min`, `t [ms]`. On its own, ADA uses such a column only when the
table has no date column, the name states a unit, and the values never go
below zero or fall from one row to the next. A column passed as the date
(`ada.analyze(df, date="t")`) needs only the name, and counts seconds if the
name gives no unit. `Time to failure (h)` is never a time axis.

The values become times from `00:00:00` on `UNDATED_DAY`, so a reading 95
seconds in is labelled `00:01:35`, and a note says so.

## Rows at a coarser grain

After cleaning, `pipeline.prepare_analysis` checks the date column for rows
written at a different grain. When a log with times of day ends (or starts)
with a run of rows that give only a date, the run is set aside if it holds at
least 24 rows and 1% of the table, spans at least 3 days, and mostly falls
within the dates of the timed rows. Such rows are usually daily summaries
appended below hourly readings: mixed in, they would count those days twice
and blend two grains in every average. A note gives their number and dates.
A reading at midnight that keeps the log's usual step stays with the log.

## Logs stacked in one file

The same check looks for a file that holds several logs one after another
with no column saying which. A log in time order that goes back to before the
earliest tenth of the readings so far starts a new log. Small steps back (a
clock corrected by minutes, a stretch sent twice) do not count, and neither
does a block shorter than 100 rows or 1% of the table. A file with more than
12 such blocks, or with steps back on more than 1% of its rows, is left as it
is.

When no column already names the logs (a site or sensor whose value changes
wherever time starts again, and within which time never starts again), a
`Block` column labels each one (`Block 1`, `Block 2`), and a note gives each
block's rows and dates and how many timestamps appear in more than one. With
no other grouping column, `Block` becomes the grouping, so each log is
compared as its own series rather than pooled.

A log whose timestamps nearly all repeat the same number of times, with
different readings and nothing to tell the repeats apart, gets a note that it
may hold that many series recorded side by side.

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
`null`, `-`, `#DIV/0!`, `<LOD`, `below detection limit` and similar; a
"not stated" phrase from `NOT_STATED_MARKERS`, below; or a censored value such
as `<0.1` or `<3,87`) do not count against the 95%. If the column converts,
they become missing and a note lists them.

Values past a detection limit get a note of their own. The limit is parsed
like any number, so `<3,87` is read as below 3.87, and the note gives the
limits and the share of the column: "7,239 cells in Phosphorus (15.5%) were
below a detection limit (<3.87) and read as missing". When 10% or more of a
column is censored (`CENSORED_SHARE`), the note adds that the mean of the rest
is biased upward (downward for values above a limit), since the values left
out are the smallest (or largest) ones.

**A lone token.** Weather records write `T` for a trace of rain, and instrument
logs write `M` or `*`. When the only text left in a column after markers is one
token of up to three characters with no digits, repeated, and it is no more
than half the column, it is read as a marker too: the column converts, the
token's cells become missing, and a note counts them. For `T` the note adds
that if it marks a trace, the true values are near zero and the mean of the
rest runs high. The word `trace` (or `tr`) is clearly a trace, so it is read
as 0, with a note. A longer word among numbers (`Control` among doses) is a
real level, and the column stays text.

**Cells that could not be read.** Up to 5% of a column may fail to parse and
the column still converts; those cells become missing, and a note names them
with examples, so nothing disappears unseen.

A column whose name ends in a key word (`id`, `code`, `zip`, `postcode`,
`phone`, `number`, `no`, `#`, `notation` and others; `is_identifier_name` in
`ada/schema.py`) is not converted when its values are whole numbers, since
they are keys; decimals under such a name are measurements and are converted.
`no` counts only after another word or with a mark (`Order No`, `No.`), so a
column of nitric oxide readings named `NO` is numeric. A column with a value
that has a leading zero (`02134`, `007`) is never converted.

**Not stated, in text columns.** A phrase for "not stated" is a missing
value, not a group: left as a label, `NO INDICA` and `No indica` would become
two groups of their own and could lead the brief. After inference, cells in
text columns matching `NOT_STATED_MARKERS` (`no indica`, `no indicado`,
`sin dato`, `nicht angegeben`, `non renseigné`, `not stated`, `not specified`,
`N/A`, `ND`, `ERROR: #N/A` and similar, ignoring case and spacing) become
missing, and a note lists them. `ND` is left alone in a column of two-letter
codes, where it is North Dakota. The same list counts as markers in numeric
columns.

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

## Label spellings

Text labels that differ only in case, accents or spacing are one label:
`NO INDICA`, `No indica` and `no Indica` would otherwise split one group into
three in every breakdown. In each text column left after inference, such
spellings are written the way the most common one is, and a note gives the
count and an example. Key columns (see above) keep their spelling, since
`ab12` and `AB12` may be two accounts, and a column with more than 20,000
distinct values is free text and is left alone.

## The cleaning report

`CleaningReport` records the original and final row and column counts, and a
count for each operation: empty rows, empty columns and index columns removed,
text columns whose values trimming changed, numeric and datetime columns
inferred, identical rows found and removed, unreadable date values and
infinite values. Its `notes` hold the sentences shown above the dashboard
(repeated headers, dropped empty columns, date-order assumptions, timezones,
unreadable dates, dates built from years, missing-value markers, values past a
detection limit, lone tokens, unreadable numbers, "not stated" phrases, merged
label spellings, widened columns, infinities, identical rows).
`pipeline.prepare_analysis` adds what schema detection concluded about the
columns (see [Schema detection](schema-detection.md#notes)).

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
| `NO` or `Sample Code` holding decimals | Converted to numbers |
| `Yes`, `YES` and `yes` in one column | Merged into the most common spelling, with a note |
| `Year` holding 1880 to 2023, and no date column | Converted to 1 January of each year, with a note |
| `Time` holding `15:06:28` to `15:56:01` | Times of day with no calendar date, with a note |
| `Date` and `Time` columns | Combined into `Date`, with a note |
| `Time (s)` counting up from 0, and no date column | Elapsed time from `00:00:00`, with a note |
| `year` and `month` (`January` ...), and no date column | Combined into a monthly `year-month` column |
| `model_year` holding 70 to 82 | Left as numbers |
| A cell reading `inf` | Treated as missing, with a note |
| `Precip` holding numbers and 14% `T` | Converted; the `T` cells become missing, with a note |
| `P mg/kg` holding `<3,87` in 15% of rows | Converted; those cells become missing, with a note on the limit and the biased mean |
| `Drainage` holding `No indica` and `ERROR: #N/A` | Those cells become missing, with a note |
| A CSV header repeated, read as `diabetes` and `diabetes.1` | Both kept, with a note saying on how many rows they differ |

## Changing this

A new cleaning step must be counted in `CleaningReport` and shown in
`cleaning_audit_frame` or the notes. If a step cannot be explained in one line
of the audit table, it does not belong in this module.
