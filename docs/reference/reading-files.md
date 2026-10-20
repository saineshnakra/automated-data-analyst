# Reading files

**Code:** `ada/file_io.py`; app limits in `app.py` · **Tests:** `tests/test_file_io.py`

Turns uploaded bytes into a DataFrame and rejects anything it cannot read. No
analysis happens here.

## Accepted formats

| Format | Extensions | Notes |
|---|---|---|
| CSV | `.csv` | Comma, semicolon, tab or pipe separated |
| Excel | `.xlsx`, `.xlsm` | Read with `openpyxl` |

Any other extension is rejected with "ADA supports CSV, XLSX, and XLSM files."
An empty file is rejected with "The uploaded file is empty."

## Limits

| Limit | Value | Where | Behaviour |
|---|---|---|---|
| File size | 25 MB | `MAX_UPLOAD_BYTES` in `app.py` | The app refuses the upload with a message |
| Rows analyzed | 250,000 | `MAX_ANALYSIS_ROWS` in `app.py`, `DEFAULT_ROW_LIMIT` in `ada/api.py` | The most recent rows are kept; the app reports how many older rows were skipped and the date range analyzed |

The file-size limit is the app's; the library does not enforce it
(`ada.MAX_FILE_BYTES` records the value). Raise either limit when you run ADA
on your own hardware: `ada.analyze(..., row_limit=...)` in Python.

Exports are usually written oldest first, so keeping the first rows of a long
file would analyze the oldest periods and forecast months the file already
contains. Rows are ranked by the detected date column when there is one, and
taken from the end of the file when there is not. If the cut splits the oldest
kept period, that period is dropped too, unless that would leave no rows.
Cleaning runs before the cut, so the skipped count is the number of cleaned
rows left out.

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
   file of sentences is not split on their semicolons.
5. If there is still one column, and the first line has no comma while the
   second and third lines have the same number of commas, treat the first line
   as a report title and use the second as the header.

If every encoding fails: "The file could not be parsed as CSV."

Two parsing rules apply to CSV and Excel alike:

- A column whose name ends in a key word (`id`, `ids`, `code`, `codes`, `zip`,
  `postal`, `phone`, `sku`, `number`, `no`, `key`, `uuid`; see
  `is_identifier_name` in `ada/schema.py`) is read as text, so `00042` keeps
  its leading zeros.
- The text `NA` is kept as a value, not read as missing, because in business
  data it is often a region code (North America). pandas' other missing-value
  markers apply.

## Excel parsing

`list_excel_sheets` returns the worksheet names, or an empty list for a CSV.
When a workbook has more than one sheet, `app.py` shows a picker and passes the
choice through. With no choice, the first sheet is read.

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
| Multi-sheet workbook, no sheet chosen | First sheet |
| `.xlsx` that is not a workbook | `ValueError` with a readable message |
| CSV with a byte-order mark | Handled by `utf-8-sig` or `utf-16` |
| Semicolon, tab or pipe CSV | Detected on the re-parse |
| Title line above the header | Skipped |

## Changing this

Adding a format means adding its suffix to `SUPPORTED_SUFFIXES`, a branch in
`read_tabular_file`, and validation as strict as the existing formats. Every
error path raises `ValueError` with a message a non-technical user can act on;
the app shows these messages as they are.
