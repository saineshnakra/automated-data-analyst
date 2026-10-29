# Cleaning

**Code:** `clean_dataframe` in `ada/analysis.py`
**Tests:** `tests/test_analysis.py`

Tidies the data before analysis and counts each change. A column ADA is unsure
about is left as it is, and every change is reported in the cleaning audit
table or as a note.

## Steps, in order

| # | Step | Detail |
|---|---|---|
| 1 | Read a second header row | See [Two-row headers](#two-row-headers) |
| 2 | Normalize and de-duplicate column names | Whitespace collapsed; a blank name becomes `column_<position>`; a repeated name gets `_2`, `_3` and so on. A note names each repeated header, and says whether the two columns differ |
| 3 | Drop empty columns | Every value missing or whitespace; a note names them |
| 4 | Drop empty rows | Every value missing or whitespace |
| 4a | Leave out notes below the table | Up to 10 last rows that each fill one cell with words not seen above it, one of them starting like a note ("Note:", "Source:", "Last updated"), in a table of 4 columns and 20 rows or more; a note names the first |
| 5 | Drop exported index columns | See below |
| 6 | Drop timezones | Timezone-aware columns keep their wall-clock time |
| 7 | Trim text | Leading and trailing whitespace removed; empty strings become missing |
| 8 | Read times of day | See [Times with no date](#times-with-no-date) |
| 9 | Infer dates in date-named columns | See below |
| 10 | Infer numbers | See below |
| 11 | Infer dates in other columns | See below |
| 12 | Set missing-value codes to missing | See [Missing-value codes](#missing-value-codes) |
| 12a | Join one-hot columns | See [One-hot columns](#one-hot-columns) |
| 13 | Read whole numbers written as dates | See [Dates written as whole numbers](#dates-written-as-whole-numbers) |
| 14 | Join a date column and a time column | See [Times with no date](#times-with-no-date) |
| 15 | Read elapsed time | See [Elapsed time](#elapsed-time) |
| 16 | Read "not stated", `ND` and placeholder labels as missing in text columns | See below |
| 17 | Restore ranges a spreadsheet read as dates | See [Ranges read as dates](#ranges-read-as-dates) |
| 18 | Merge label spellings and abbreviations | See [Label spellings](#label-spellings) |
| 18a | Read a column for each year as a row for each year | See [A column for each year](#a-column-for-each-year) |
| 19 | Build dates from years | See below |
| 20 | Widen overflowing integers | An integer column whose total would overflow int64 becomes float64 |
| 21 | Replace infinities | `inf` and `-inf` in float columns become missing |
| 22 | Count identical rows | Counted and kept |

If nothing is left, `clean_dataframe` raises `ValueError` rather than
returning an empty frame.

Each text column is factorised once, when empty rows are found, and every
later step on it (trimming, markers, numbers, dates, labels) works on its
distinct values, with their text normalised once, and spreads the result back
over the rows. A site, a status or a date repeats many times, so a table of
242,000 rows with 13 text columns cleans in under a second. Before a column is
parsed in full as numbers or as times of day, 200 of its cells drawn at random
are tried. A column needs 95% of its cells to convert, so one where fewer than
half of the drawn cells read as numbers (or times) is not parsed in full.

## Two-row headers

A spreadsheet often puts a label over a run of columns, with the real names
in the row below: `Weld Output Response` over `UTS (MPa)` and `% Elongation`.
Read with one header row, the label's blank cells become `Unnamed: 6` and the
names become a row of text in numeric columns, which would then count as data
and leave a missing cell in every column.

The first row is read as a second header row when the header has at least one
blank cell (`Unnamed: N`) and one label, and the first row holds text in at
least two columns whose values below it read as numbers, never a number in
such a column, and text under every blank header cell with data below it. A
column under a group label takes its name from the second row; a label over a
single column is joined with the text below it (`Temperature` over `(°C)`
becomes `Temperature (°C)`); a column with nothing in the second row keeps its
label. Names the second row repeats are prefixed with their group label. The
row is dropped, and a note names the group labels.

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

## Quarters

A text column of quarters as statistics offices write them ("2026Q3",
INE's "2026QIII" and "2026 T3", "Q3 2026") is read as the first day of each
quarter, with a note, before any number is read from it; read as numbers,
the four quarters of a year would all be 2026. A month written as Statistics
Norway and Eurostat write it ("1993M01", "2025M1", "2025-M12") is read as the
first day of that month the same way.

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
12 such blocks, or with more steps back than 24 and 1% of its rows, is left as
it is: Formula 1's races, listed season by season from 2009 back to 1950, step
back 65 times in 1,058 rows and are one table out of date order, not 8 logs.
So is a file with fewer than 20 distinct times: three years shared by 344 rows
sorted by species are not the times of a log.

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

Inference runs only on text columns, except for
[dates written as whole numbers](#dates-written-as-whole-numbers) and
[dates from years](#dates-from-years).

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

The decimal mark is decided once per column, from all its values. A lone
comma is ambiguous: `1,234` may be a thousand. When some values can only be
using commas as decimal points (`-2,95`, `0,307`, `306,27883`, `1.234,5`) and
none uses them to group thousands (`1,234.5`, `1,234,567`), every comma in
the column is read as a decimal point, so `-0,307` is -0.307 beside `-2,95`
rather than -307. In such a column `1.234` groups a thousand. A note names
the columns where this changed how a value would otherwise be read.

Cells holding a missing-value marker (`MISSING_MARKERS`: `NA`, `n/a`, `nd`,
`null`, `unknown`, `-`, `#DIV/0!`, `<LOD`, `below detection limit` and
similar; a "not stated" phrase from `NOT_STATED_MARKERS`, below; or a censored
value such as `<0.1` or `<3,87`) do not count against the 95%. If the column
converts, they become missing and a note lists them.

Values past a detection limit get a note of their own, one for each side. The
limit is parsed like any number, so `<3,87` is read as below 3.87, and the
note gives the limits and the share of the column: "7,239 cells in Phosphorus
(15.5%) were below a detection limit (<3.87) and read as missing". When 10% or
more of a column is censored (`CENSORED_SHARE`), the note adds that the mean
of the rest is biased upward (downward for values above a limit), since the
values left out are the smallest (or largest) ones.

A `>X` limit is not always an instrument's. When the limits vary and, on 80% or
more of at least five such rows (`REACHED_SHARE`, `REACHED_ROWS`), X is another
numeric column's value on the same row (within 10%), a note says so: water
clarity written `>3.5` at a site 3.5 m deep means the disk was still seen on
the bottom, so the reading reached the depth and its true value is at least
that.

In a column of readings, `ND` (or `n.d.`) means not detected, so it counts as
below a detection limit, like `<LOD`. The exception is a column that writes
values below its limit with a number (`<0.4`): beside those, `ND` more likely
means the value was not determined, so its cells get a plain note saying so.

**Yes and no among 0s and 1s.** In a column whose numbers are only 0 and 1,
`true`, `false`, `yes`, `no`, `y`, `n`, `t` and `f` (any case) are read as 1
and 0 rather than set to missing, and a note says so.

**Readings written with a spread or a unit.** A column of cells such as
`5.66 ± 1.69 items/10 g` or `0.43 items/individual` holds readings, even under
a key name such as `particle_number`. When at least half of its readable
cells are a number, optionally followed by `±` and a spread, and a unit
starting with a letter, and at least half of those carry a unit or a spread,
each cell is read as the number it starts with (the mean before the `±`).
Ranges (`0.87 – 7.86 items/g`) and numbers in scientific notation
(`4.5 *10^(6) items`) are not single readings, so they become missing and a
note lists them; `ND` and `<LOQ` are read as below a detection limit. When
every reading is in one unit, the note names it. When the units differ, a new
text column `<name> unit` holds each row's unit, so the schema treats the
table as long by unit and never combines readings in different units (see
[Schema detection](schema-detection.md#long-tables)). A column with more than
12 units holds codes that start with a number (`37F-PM-1014-03`), not
readings, and stays text. So does a column of whole numbers glued to one of
three or more letters (`1A`, `6C`, `7D`, the codes of DC's Advisory
Neighborhood Commissions): those are places, not readings in units A, C and D.

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
`sin dato`, `sin datos`, `nicht angegeben`, `keine Daten`, `non renseigné`,
`pas de données`, `nessun dato`, `sem dados`, `tidak ada data`,
`geen gegevens`, `no data`, `not stated`, `not specified`, `N/A`,
`ERROR: #N/A` and similar, ignoring case and spacing) become missing, and a
note lists them. The same list counts as markers in numeric columns.

Placeholder labels (`PLACEHOLDER_LABELS`: `Unknown`, `unclassified`,
`not classified`, `UNABLE TO CLASSIFY`) stand for a value nobody could give.
Kept as labels, they become a group that mixes every kind of row, so they
become missing too, with a note saying what they stand for.

`ND` in a text column becomes missing with a note: when most of the column's
other cells start with a number (readings written with their units, such as
`5.66 ± 1.69 items/g`), the note says it means not detected; otherwise that it
usually means not detected or not determined. `ND` is left alone in a column
of two-letter codes, where it is North Dakota.

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

## Missing-value codes

Instruments and exports write a code in place of a missing reading: `-999`
among pressures near 950 hPa, `999.9` among wind speeds below 20. Left in, a
code drives every average, correlation and comparison on its column, and a
count of missing values misses it. After inference, `quality.blank_codes` sets
clear codes to missing in every numeric column, before roles are detected or
any figure is computed. Two kinds qualify:

- a code value (a run of nines, a repeated digit, a large negative power of
  ten) standing alone, far from the rest, and either negative in a column that
  is otherwise not or ten times the 99th percentile of the other values; see
  [Data checks](quality.md#codes-set-to-missing)
- one negative value that three or more columns, each otherwise never
  negative and holding readings rather than a few whole-number codes, hold on
  the same rows, as a logger writes `-1` in every channel when a read fails

In a text column, `NA` or `na` is missing when half or more of the column's
other values are words (`male`, `female`, `NA`). Among codes (values of three
characters or fewer, or in capitals up to five letters: `EU`, `APAC`, `NA`;
`Yes`, `No`, `NA`) it is kept as a code of its own, and `Na`, the symbol for
sodium, is never matched. A column holding only `NA` is left as it is.
A dash, `?` or `.` in a column whose other values are one yes and one no
(`Y`, `N` and `-` in Seattle's Frisk Flag) is a missing answer, so the
column is a yes/no column; among other labels it stays a label.

A total label among numbers ("ALL" beside the single years of age 0 to 115,
"All ages"), on two rows or more, keeps the column as labels: it is the
level for all the others in a published table of counts, not a blank.

Key columns and layout numbers are left alone. Each column gets a note with
its count ("11,177 cells in PM held -999, a code for a missing reading far from
every other value (median 955.6), and were set to missing"), and the data
checks still list each code, saying it was set to missing. The cells count as
missing everywhere afterwards: in the completeness figure, the data
dictionary and "how many rows have missing PM".

## Dates written as whole numbers

Weather and finance exports often write a date as a whole number:
`19540601` (YYYYMMDD) or `202403` (YYYYMM). A numeric column becomes dates
when every value is a valid date in one of these layouts, with years from 1800
to 2100, and either:

- its name has a date word (`date`, `datum`, `fecha`, `data`, `day`,
  `period`, `yyyymmdd` and similar) and it has two or more values, or
- the table has no other date (times of day with no date do not count) and it
  has three or more values

A column passed as the date (`ada.analyze(df, date="MESS_DATUM")`) is read
whatever its name or the rest of the table, with any year a timestamp can
hold. A YYYYMM value becomes the first of its month. A note gives the layout
and the span. Codes are set to missing first, so a `-999` in such a column
does not stop it being read.

## Ranges read as dates

A spreadsheet reads a range such as `10-30` as a date and shows it as
`Oct-30` (or `30-Oct`). In a text column made mostly of numeric ranges, with
at most 200 labels, such a value is matched to its likely original: the
month's number and the other number, low end first. When that range shares an
end with another range in the column (`0-10`, `10-30`, `30-60`), the values
are restored to it and a note says so. Otherwise they are left as written, and
the note names the range they may have been.

## A column for each year

A table that gives each year in a column of its own, named by the year alone
or the year and the same words before or after it ("1969 Deaths" to "2024
Deaths", "Number 2000" to "Number 2024", "1960" to "2023"), holding numbers, with at least five such columns making up half the
table or more, is read with a row for each year: the year in a new column,
Year, and the figure in one column named for those words (Deaths, or Value
when the names are years alone). A note names the columns. Year then becomes
the date, as below. A table that already has a date is left as it is.

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

With a day column too (`day`, `da`, `dy`), and then an hour, minute and second
column, each used only if the one before it was and only if it holds a whole
number in range on every row with a year, the parts become one timestamp at
the finest grain given, named after them (`year-month-day-hour`). The parts
are kept as columns of their own, and the note says what was combined, to
which grain, and that no time zone is attached. A date no calendar has (31
February, hour 24) is left missing rather than rolled into the next day, and
the note gives the number of such rows and the first one as written.

When no column is named for years, a column of whole years named for a date
or a period holds them: StatCan's `REF_DATE`, Eurostat's `TIME_PERIOD`, and the
words for time or year of statistics offices, such as Statistics Denmark's
`TID`, Sweden's `år` and Finland's `Vuosi`. English "time" is not one, since
race times of 1,800 to 2,100 seconds are no years.

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

Two more kinds of spelling are merged, conservatively, in columns of at most
50 labels (`_respelt_labels`). A single letter is merged into the one word in
the column it starts (`M` into `male`, `F` into `female`); it is left alone
when several words start with it, or when the only match is a code in capitals
(`A` beside `AB`). And a word of five letters or more one letter away from
another, by a letter added, dropped or two neighbours swapped (`Hipanic` and
`Hispanic`), is merged into the other when it is at most a fifth as common, as
a typo is; labels with digits or other marks (`Type A`, `Type B`) never count.
A spelling on ten rows or more that never sits beside the same value of
another column of 50 labels or fewer as the commoner one is a label of its
own, not a typo: `country` beside `Norge` only and `county` beside the
counties stay apart. A note names each merge.

## Single years beside spans

A text column named for years or a time period that holds single years and
spans of years (`2019`, `2023 - 25`, `2019/20`) gets a new column, named
after it with "year" added, holding the single years and nothing for the
spans: a span is no one year. The new column then gives the table its
dates. A column of spans alone gets none. A note says how many rows hold a
span.

## One-hot columns

A category is sometimes split into one 0/1 column per level, the way models
are fed: `Soil_Type_Gravel`, `Soil_Type_Sand`, `Soil_Type_Silt`. Columns of
only 0 and 1 that share a name prefix (up to a `_`, `.`, `:`, `-` or space)
and never mark two levels on one row are joined back into one text column
named for the prefix (`Soil_Type`), holding each row's level. Rows where all
of them are 0 hold the level left out, shown as `other`. Two columns qualify
only when every row has exactly one of them, and prefixes that start yes/no
questions (`is_`, `has_`, `was_`, ...) never do. A note names the columns
joined.

## The cleaning report

`CleaningReport` records the original and final row and column counts, and a
count for each operation: empty rows, empty columns and index columns removed,
text columns whose values trimming changed, numeric and datetime columns
inferred, identical rows found and removed, unreadable date values and
infinite values. Its `notes` hold the sentences shown above the dashboard
(a two-row header, repeated headers, dropped empty columns, date-order
assumptions, timezones, unreadable dates, missing-value codes set to missing,
dates read from whole numbers, dates built from years, missing-value markers,
values past a detection limit, lone tokens, unreadable numbers, "not stated"
phrases, `ND` and placeholder labels, ranges read as dates, merged label
spellings, widened columns, infinities, identical rows). Its `blanked_codes`
hold the data-check issues for the codes set to missing, which
`Analysis.quality()` and the app's data checks list beside the rest.
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
| `categori` holding `TIDAK ADA DATA`, or `Cancer Type` holding `UNABLE TO CLASSIFY` | Those cells become missing, with a note |
| `Lead` holding numbers and `ND` | Converted; `ND` counts as below a detection limit, with a note |
| `PM` and nine other columns holding `-999` | Those cells become missing, with a note per column, and the data checks list the code |
| `pm10`, `pm1_0` and `pm2_5` all `-1` on the same 4 rows, never otherwise negative | Those cells become missing, with a note per column |
| `MESS_DATUM` holding `19540601` to `20221231`, and no date column | Read as dates, with a note |
| `Depth` holding `0-10`, `Oct-30`, `30-60` | `Oct-30` restored to `10-30`, with a note |
| A header row of group labels over the real names | Names taken from the second row, which is dropped, with a note |
| A CSV header repeated, read as `diabetes` and `diabetes.1` | Both kept, with a note saying on how many rows they differ |

## Changing this

A new cleaning step must be counted in `CleaningReport` and shown in
`cleaning_audit_frame` or the notes. If a step cannot be explained in one line
of the audit table, it does not belong in this module.
