# Reading files

**Code:** `ada/file_io.py`; app limits in `app.py` · **Tests:** `tests/test_file_io.py`

Turns uploaded bytes into a DataFrame and rejects anything it cannot read. No
analysis happens here.

## Accepted formats

| Format | Extensions | Notes |
|---|---|---|
| CSV | `.csv` | Comma, semicolon, tab or pipe separated |
| Tab-separated text | `.tsv`, `.txt`, `.data`, `.dat` | Read as tab-separated when that splits the header, otherwise as a CSV (the UCI repository's `.data` files are comma-separated) |
| Excel | `.xlsx`, `.xlsm` | Read with `python-calamine`, several times faster than `openpyxl` on large sheets |

Any other extension is rejected with "ADA supports CSV, TSV, TXT, DATA, DAT, XLSX and XLSM files."
An empty file is rejected with "The uploaded file is empty."

## Limits

| Limit | Value | Where | Behaviour |
|---|---|---|---|
| File size | 25 MB | `MAX_UPLOAD_BYTES` in `app.py` | The app refuses the upload with a message |
| Rows analyzed | 1,000,000 | `MAX_ANALYSIS_ROWS` in `app.py`, `DEFAULT_ROW_LIMIT` in `ada/api.py` | A table with a date keeps its most recent rows; one without is sampled at random. The app, `print(report)` and every answer say which rows were analyzed |

The file-size limit is the app's; the library does not enforce it
(`ada.MAX_FILE_BYTES` records the value). Raise either limit when you run ADA
on your own hardware: `ada.analyze(..., row_limit=...)` in Python.

Cleaning, not the number of rows, sets how long a file takes: a 700,000-row
file is cleaned in under 3 seconds, and analyzing all of it rather than a
quarter adds about half a second. So the limit is high enough that a file the
app accepts is analyzed whole.

Above the limit, exports are usually written oldest first, so keeping the first
rows of a long file would analyze the oldest periods and forecast months the
file already contains. Rows are ranked by the detected date column. If the cut
splits the oldest kept period, that period is dropped too, unless that would
leave no rows. Cleaning runs before this cut, so the skipped count is the
number of cleaned rows left out.

A table with no date has no oldest rows, and its file order is often sorted by
a group, so its tail could hold one group only. It gets a random sample
instead (seed `SAMPLE_SEED`, so a rerun analyzes the same rows), kept in file
order and taken before cleaning so the rows left out are never cleaned. A note
names any grouping column with fewer values in the sample than in the file.
Whether a table has a date is judged by cleaning its first 2,000 rows.

`PreparedAnalysis.rows_note` and `Analysis.rows_note` say which rows were
analyzed ("a random sample of 1,000,000 of 3,108,545 rows", "the newest
1,000,000 of 1,500,000 rows, from 05 Mar 2024"). Every answer from
`Analysis.ask` ends with it, and `to_dict` reports it as `rows_note`.

## CSV parsing

1. Choose encodings. A UTF-16 byte-order mark selects `utf-16`; a UTF-8 one
   selects `utf-8-sig`. Otherwise `utf-8-sig`, `utf-8` and `latin-1` are tried
   in turn. `utf-8-sig` comes first so that a byte-order mark does not end up
   in the first column name.
2. Settle the separator and the header line on the opening lines of the file
   (the first 256 KB, `SNIFF_BYTES`, cut at a line end), then parse the whole
   file once with what they showed. A large file is never parsed once per
   separator tried. If the opening lines end inside a quoted field that spans
   lines, the whole file is used to decide instead.
3. Parse the opening lines with the comma separator.
4. If that gives exactly one column and the header line contains `;`, a tab or
   `|`, use that separator. Only the header line is checked, so a one-column
   file of sentences is not split on their semicolons. If the parse fails and
   the header line holds one of them more often than commas, it is tried too:
   Statistics Denmark separates by semicolons and writes "Tickets sold
   (1,000)", whose comma makes a comma read fail. A split that leaves every
   column but the first empty is a title line with a semicolon in it, not a
   header.
5. If there is still one column, or the parse fails on the first row wider
   than the header, look for lines above the header: a report's title, or a
   data portal's "Data Source" and "Last Updated Date" lines. The header is the
   first of the opening 11 lines (`MAX_LINES_ABOVE_HEADER` plus one) that has
   as many comma-separated fields as the five rows below it, with every line
   above it narrower; those lines are skipped. A World Bank indicator file
   has four such lines.

If every encoding fails: "The file could not be parsed as CSV."

Two parsing rules apply to CSV and Excel alike:

- A column whose name ends in a key word (`id`, `ids`, `code`, `codes`, `zip`,
  `postal`, `phone`, `sku`, `number`, `no`, `key`, `uuid`; see
  `is_identifier_name` in `ada/schema.py`) is read as text, so `00042` keeps
  its leading zeros.
- The text `NA` is kept as a value when reading, because in business data it
  is often a region code (North America). Cleaning then reads it as missing
  in a text column whose other values are words (`male`, `female`, `NA`); see
  [Cleaning](cleaning.md#missing-value-codes). pandas' other missing-value
  markers apply, and `\N`, the null MySQL and PostgreSQL write when they
  export a table, is read as missing too.

## Excel parsing

`list_excel_sheets` returns the worksheet names, or an empty list for a CSV.
When a workbook has more than one sheet, `app.py` shows a picker and passes the
choice through. With no choice, the first sheet is read, unless it holds only
notes: a first sheet with a cell of prose (over 80 characters) and fewer than
one number in twenty filled cells, as the Maddison Project's "Notes" sheet is.
Then the sheet with the most numbers is read ("Full data"), its name is kept
in the source's `sheet`, and a note says so.

A file with an Excel extension that is not a valid workbook raises "The file
is not a valid Excel workbook." rather than a raw `BadZipFile` or `KeyError`.

## Other functions

- `list_sample_datasets` returns the bundled files in `ada/samples/` keyed by
  display name ("SaaS Subscriptions").
- `safe_csv` writes the cleaned-data download. Text cells and headers that
  start with `=`, `+`, `-`, `@`, a tab or a carriage return get a leading
  apostrophe, so a spreadsheet shows them as text instead of running them as
  formulas.

## Edge cases

| Situation | Behavior |
|---|---|
| Empty file | `ValueError`: "The uploaded file is empty." |
| Multi-sheet workbook, no sheet chosen | First sheet; after a sheet of notes, the sheet with the most numbers |
| `.xlsx` that is not a workbook | `ValueError` with a readable message |
| CSV with a byte-order mark | Handled by `utf-8-sig` or `utf-16` |
| Semicolon, tab or pipe CSV | Detected on the re-parse |
| Title line above the header | Skipped |

## Changing this

Adding a format means adding its suffix to `SUPPORTED_SUFFIXES`, a branch in
`read_tabular_file`, and validation as strict as the existing formats. Every
error path raises `ValueError` with a message a non-technical user can act on;
the app shows these messages as they are.
