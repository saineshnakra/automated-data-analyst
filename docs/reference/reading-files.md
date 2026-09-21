# Reading files

**Code:** `file_io.py` · limits set in `app.py` · **Tests:** `tests/test_file_io.py`

Turns uploaded bytes into a DataFrame, and refuses anything it cannot read
safely. No analysis happens here.

## What is accepted

| Format | Extensions | Notes |
|---|---|---|
| CSV | `.csv` | Comma, semicolon, or tab separated |
| Excel | `.xlsx`, `.xlsm` | Read via `openpyxl` |

Anything else is rejected with "ADA supports CSV, XLSX, and XLSM files." An
empty file is rejected too.

## Limits

| Limit | Value | Where | What happens |
|---|---|---|---|
| File size | 25 MB | `MAX_UPLOAD_BYTES` in `app.py` | Upload refused with a message |
| Rows analyzed | 250,000 | `MAX_ANALYSIS_ROWS` in `app.py` | The **most recent** 250,000 kept, and the app says how many older rows were skipped and what date range it analyzed |

Both exist so a hosted deployment stays predictable. Change them if you run ADA
yourself on bigger machines.

Recency matters more than order of appearance. Exports are usually written
oldest-first, so keeping the head of a long file analyzes the periods nobody is
asking about — and then forecasts months the file already contains. Rows are
selected by the detected date column where there is one, and from the end of
the file where there is not. Cleaning runs before the cut, so the skipped count
is the real number of rows dropped.

## Columns kept as text

Both readers peek at the header first and read as **text** every column whose
name says it holds a key, so an account number written `00042` stays `00042`
instead of becoming `42`. The test is `schema.is_identifier_name`: the last
word (or the whole name) is one of `id`, `ids`, `code`, `codes`, `zip`,
`postal`, `phone`, `sku`, `number`, `no`, `key`, `uuid`, as a whole word.
`Customer ID`, `Account Number`, `SKU`, `Order No` and `ZIP` are kept;
`Paid Amount` is not, although it contains the letters `id`. Cleaning uses the
same function to decide what it leaves alone, so what is preserved here is
still there after `prepare_analysis`.

`NA` is read as a value, not as missing: in business data it is North America.
Every other default missing-value marker is kept.

## How CSV parsing works

1. Try encodings in order: `utf-8-sig`, then `utf-8`, then `latin-1`.
   `utf-8-sig` goes first so a Windows byte-order mark does not end up glued to
   the first column name.
2. Parse with the default comma separator.
3. If that produces exactly **one** column and the header line contains a `;`,
   a tab or a `|`, re-parse with that separator. The single-column result is
   the tell-tale sign of a semicolon-separated European export.
4. If the first line is a report title over a real header, use the second line
   as the header.

If every encoding fails, you get "The file could not be parsed as CSV."

### Rows wider than the header

`Region,Revenue` followed by `West,100,999` is refused: "Line 2 of the file
holds 3 values, but the header names 2 columns." pandas would otherwise read
it without complaint, taking the first column as an index and shifting every
other value one column left, so `Region` arrives full of revenue figures. The
reader spots the invented index (nothing here asks for one) and names the
first line whose width does not match. A trailing delimiter on every row is
the same mistake and gets the same message. A row wider than the header
further down the file was already a parse error and still is.

## How Excel parsing works

`list_excel_sheets` returns the worksheet names — an empty list for CSV. When a
workbook has more than one sheet, `app.py` shows a picker and passes the choice
through. With no choice, sheet index `0` is read.

A file with an Excel extension that is not really a workbook raises "The file is
not a valid Excel workbook" rather than a raw `BadZipFile`. A workbook that is
a zip but whose XML stops mid-element -- a truncated upload, a broken export --
raises "The workbook is damaged and could not be read" from both readers,
instead of the `ParseError` openpyxl produces. Nothing else is caught: an
unexpected exception is still a bug to look at, not a message to show.

### Formulas without saved results

A workbook written by a library (openpyxl among them) holds formulas with no
cached value; Excel computes on open, pandas does not. Such a column reads back
entirely missing, cleaning would drop it, and a different metric would be
chosen in silence. So when any column comes back all-missing the sheet is
opened once more with formulas visible; if that column's cells are formulas,
the file is refused with "The workbook contains formulas without saved
results (Revenue)" and the advice to re-save it from Excel or export values. A
column that is simply empty is not a formula, and is read as before.

## Edge cases

| Situation | Behavior |
|---|---|
| Empty file | `ValueError` — "The uploaded file is empty." |
| Multi-sheet workbook, no sheet chosen | First sheet |
| `.xlsx` that is not a zip | Clear error, not a traceback |
| `.xlsx` with truncated XML inside | "The workbook is damaged and could not be read" |
| Formula cells with no saved value | Refused, naming the column |
| CSV with a BOM | Handled by `utf-8-sig` |
| Semicolon CSV | Detected on the retry pass |
| A data row wider than the header | Refused, naming the line and both widths |
| `Customer ID` of `00042` | Read as text, zeros kept |
| `NA` in a region column | North America, not missing |

## Changing this

Adding a format means adding the suffix to `SUPPORTED_SUFFIXES`, adding a
branch in `read_tabular_file`, and keeping validation just as strict. Every
error path should raise `ValueError` with a message a non-technical user can
act on — those messages are shown verbatim in the app.
