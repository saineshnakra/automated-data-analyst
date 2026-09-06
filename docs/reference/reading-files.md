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

## How CSV parsing works

1. Try encodings in order: `utf-8-sig`, then `utf-8`, then `latin-1`.
   `utf-8-sig` goes first so a Windows byte-order mark does not end up glued to
   the first column name.
2. Parse with the default comma separator.
3. If that produces exactly **one** column and the first 4 KB contains a `;` or
   a tab, re-parse with separator sniffing. The single-column result is the
   tell-tale sign of a semicolon-separated European export.

If every encoding fails, you get "The file could not be parsed as CSV."

## How Excel parsing works

`list_excel_sheets` returns the worksheet names — an empty list for CSV. When a
workbook has more than one sheet, `app.py` shows a picker and passes the choice
through. With no choice, sheet index `0` is read.

A file with an Excel extension that is not really a workbook raises "The file is
not a valid Excel workbook" rather than a raw `BadZipFile`.

## Edge cases

| Situation | Behavior |
|---|---|
| Empty file | `ValueError` — "The uploaded file is empty." |
| Multi-sheet workbook, no sheet chosen | First sheet |
| `.xlsx` that is not a zip | Clear error, not a traceback |
| CSV with a BOM | Handled by `utf-8-sig` |
| Semicolon CSV | Detected on the retry pass |

## Changing this

Adding a format means adding the suffix to `SUPPORTED_SUFFIXES`, adding a
branch in `read_tabular_file`, and keeping validation just as strict. Every
error path should raise `ValueError` with a message a non-technical user can
act on — those messages are shown verbatim in the app.
