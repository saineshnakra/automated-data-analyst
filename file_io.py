"""Safe, testable parsing for ADA's supported business files."""

from __future__ import annotations

import csv
from io import BytesIO
from pathlib import Path
from xml.etree.ElementTree import ParseError
from zipfile import BadZipFile

import openpyxl
import pandas as pd
from openpyxl.utils.exceptions import InvalidFileException
from pandas.io.parsers.readers import STR_NA_VALUES

from schema import is_identifier_name

SUPPORTED_SUFFIXES = {".csv", ".xlsx", ".xlsm"}
SAMPLES_DIRECTORY = Path(__file__).parent / "samples"
# Title-casing a filename gets "Saas" and "Mrr" wrong; these keep their shape.
SAMPLE_ACRONYMS = {"Saas": "SaaS", "Mrr": "MRR", "Arr": "ARR", "Ar": "AR", "Kpi": "KPI"}
EXCEL_SUFFIXES = {".xlsx", ".xlsm"}


def list_sample_datasets() -> dict[str, Path]:
    """Bundled files a visitor can analyze without uploading anything."""
    if not SAMPLES_DIRECTORY.is_dir():
        return {}
    def label(stem: str) -> str:
        words = stem.replace("-", " ").title().split()
        return " ".join(SAMPLE_ACRONYMS.get(word, word) for word in words)

    return {label(path.stem): path for path in sorted(SAMPLES_DIRECTORY.glob("*.csv"))}


# A spreadsheet reads a cell starting with any of these as a formula, so a
# value carried out of an uploaded file can execute when the download is
# opened. Only text is affected; numbers are written by pandas as numbers.
FORMULA_LEADERS = ("=", "+", "-", "@", "\t", "\r")


def safe_csv(dataframe: pd.DataFrame) -> str:
    """Serialize to CSV without handing a spreadsheet a formula to run.

    Text cells that begin like a formula are prefixed with an apostrophe,
    which spreadsheets read as "this is text". The values are the user's own,
    but a downloaded file gets forwarded, and the person who opens it did not
    choose to run anything.
    """
    export = dataframe.copy()
    for column in export.select_dtypes(include=["object", "string"]).columns:
        values = export[column].astype("string")
        risky = values.str.startswith(FORMULA_LEADERS, na=False)
        export[column] = values.mask(risky, "'" + values.fillna(""))
    export.columns = [
        f"'{name}" if str(name).startswith(FORMULA_LEADERS) else name for name in export.columns
    ]
    return export.to_csv(index=False)


def _validate(contents: bytes, filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError("ADA supports CSV, XLSX, and XLSM files.")
    if not contents:
        raise ValueError("The uploaded file is empty.")
    return suffix


NOT_A_WORKBOOK = "The file is not a valid Excel workbook."
DAMAGED_WORKBOOK = (
    "The workbook is damaged and could not be read. Open it in Excel, save it "
    "again, and upload the new copy."
)


def list_excel_sheets(contents: bytes, filename: str) -> list[str]:
    """Worksheet names of a workbook; empty for CSV files."""
    suffix = _validate(contents, filename)
    if suffix not in EXCEL_SUFFIXES:
        return []
    try:
        with pd.ExcelFile(BytesIO(contents), engine="openpyxl") as workbook:
            return [str(name) for name in workbook.sheet_names]
    except (BadZipFile, KeyError) as error:
        # A zip that is not a workbook -- no [Content_Types].xml -- surfaces
        # from openpyxl as a KeyError, which the app did not catch.
        raise ValueError(NOT_A_WORKBOOK) from error
    except (ParseError, InvalidFileException) as error:
        # A workbook whose XML stops mid-element is a zip, so the check above
        # passes it, and the parser error it raises instead is not one the
        # app catches.
        raise ValueError(DAMAGED_WORKBOOK) from error


# pandas reads "NA" as missing by default. In business data it is North
# America, and a region code was being turned into a blank before cleaning
# ever saw it. Every other default marker is kept.
NA_VALUES = sorted(STR_NA_VALUES - {"NA"})


def _kept_as_text(columns) -> dict[str, type]:
    """Columns read as text because their name says they hold a key.

    An account number written 00042 must stay 00042. The cleaner consults
    the same test before it infers numbers, so nothing kept here is undone
    a step later.
    """
    return {column: str for column in columns if is_identifier_name(str(column))}


class RowWidthError(ValueError):
    """A data row holds more values than the header names.

    pandas reads such a file without complaint, taking the first column as an
    index and shifting every other value one column to the left, so a Region
    column arrives full of revenue figures. The header width is kept so the
    title-row rescue can tell this apart from a report title above the table.
    """

    def __init__(self, header_width: int, line_number: int, row_width: int) -> None:
        self.header_width = header_width
        super().__init__(
            f"Line {line_number} of the file holds {row_width} values, but the header names "
            f"{header_width} columns. Check that line for a stray delimiter, or the header "
            "for a missing column name."
        )


def _row_width_error(contents: bytes, encoding: str, header_width: int, **options) -> RowWidthError:
    delimiter = options.get("sep", ",")
    skipped = options.get("header", 0)
    text = contents.decode(encoding, errors="replace")
    for line_number, row in enumerate(csv.reader(text.splitlines(), delimiter=delimiter), start=1):
        if line_number > skipped + 1 and len(row) > header_width:
            return RowWidthError(header_width, line_number, len(row))
    # Nothing wider was found by counting, so pandas saw the mismatch in a
    # form the count does not -- name the first data line rather than none.
    return RowWidthError(header_width, skipped + 2, header_width + 1)


def _read_csv(contents: bytes, encoding: str, **options) -> pd.DataFrame:
    """read_csv with the header peeked first, so id columns stay text."""
    header = pd.read_csv(BytesIO(contents), encoding=encoding, nrows=0, **options)
    parsed = pd.read_csv(
        BytesIO(contents),
        encoding=encoding,
        dtype=_kept_as_text(header.columns),
        keep_default_na=False,
        na_values=NA_VALUES,
        low_memory=False,
        **options,
    )
    if not isinstance(parsed.index, pd.RangeIndex):
        # Nothing here asks for an index column, so any other index means
        # pandas built one out of rows wider than the header.
        raise _row_width_error(contents, encoding, len(header.columns), **options)
    return parsed


def _candidate_encodings(contents: bytes) -> tuple[str, ...]:
    """A byte-order mark settles the encoding; without one, guess in order.

    PowerShell's Export-Csv writes UTF-16 by default, and decoding that as
    Latin-1 turns every column name into mojibake and every row into blanks,
    which reaches the user as "no analyzable data" rather than as an error.
    """
    if contents[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return ("utf-16",)
    if contents[:3] == b"\xef\xbb\xbf":
        return ("utf-8-sig",)
    return ("utf-8-sig", "utf-8", "latin-1")


def _header_line(contents: bytes, encoding: str) -> str:
    head = contents.split(b"\n", 1)[0]
    return head.decode(encoding, errors="replace")


def _resplit_single_column(parsed: pd.DataFrame, contents: bytes, encoding: str) -> pd.DataFrame:
    """Re-read a semicolon- or tab-delimited file, but only when it is one.

    The separator has to appear in the header line. Sniffing the body instead
    splits an ordinary one-column file of sentences on the semicolons inside
    them, which destroys the data silently -- a worse outcome than the
    European CSV this rescue exists for.
    """
    if len(parsed.columns) != 1:
        return parsed
    header = _header_line(contents, encoding)
    for separator in (";", "\t", "|"):
        if separator not in header:
            continue
        candidate = _read_csv(contents, encoding, sep=separator)
        if len(candidate.columns) > 1:
            return candidate
    return parsed


def _skip_title_row(contents: bytes, encoding: str) -> pd.DataFrame | None:
    """Use the second line as the header when the first is a report title.

    Exported reports often carry a title above the table. Read as a header it
    yields one column over rows wider than it -- the very shape of a
    malformed file -- so this runs when that shape is seen, and the file is
    refused only when the title reading does not hold up either.
    """
    lines = contents.split(b"\n")[:3]
    if len(lines) < 3:
        return None
    title, header, first_row = (line.decode(encoding, errors="replace") for line in lines)
    # Only a comma is treated as the delimiter here. A one-column file of
    # sentences containing semicolons is a real file, and re-reading it as a
    # table would shred it -- the mistake this rescue is meant to prevent.
    if "," in title or not header.count(",") or header.count(",") != first_row.count(","):
        return None
    candidate = _read_csv(contents, encoding, header=1)
    return candidate if len(candidate.columns) > 1 else None


def _parse_csv(contents: bytes, encoding: str) -> pd.DataFrame:
    try:
        parsed = _read_csv(contents, encoding)
    except RowWidthError as error:
        rescued = _skip_title_row(contents, encoding) if error.header_width == 1 else None
        if rescued is None:
            raise
        return rescued
    return _resplit_single_column(parsed, contents, encoding)


def _formula_columns_without_values(
    contents: bytes, sheet: str | int, blank: list[tuple[int, str]]
) -> list[str]:
    """Which of the all-missing columns are formulas with no saved result.

    openpyxl writes formulas without evaluating them, and a workbook saved
    that way reads back as blanks. Cleaning would then drop the column and
    pick a different metric without a word. Only columns that came back
    entirely missing are looked at, so an ordinary workbook never pays for
    the second open.
    """
    book = openpyxl.load_workbook(BytesIO(contents), data_only=False, read_only=True)
    try:
        worksheet = book[sheet] if isinstance(sheet, str) else book.worksheets[sheet]
        flagged = []
        for position, column in blank:
            cells = worksheet.iter_rows(min_row=2, min_col=position, max_col=position)
            if any(cell.data_type == "f" for row in cells for cell in row):
                flagged.append(column)
        return flagged
    finally:
        book.close()


def _read_excel(contents: bytes, sheet: str | int) -> pd.DataFrame:
    header = pd.read_excel(BytesIO(contents), engine="openpyxl", sheet_name=sheet, nrows=0)
    parsed = pd.read_excel(
        BytesIO(contents),
        engine="openpyxl",
        sheet_name=sheet,
        dtype=_kept_as_text(header.columns),
        keep_default_na=False,
        na_values=NA_VALUES,
    )
    blank = [
        (position, str(column))
        for position, column in enumerate(parsed.columns, start=1)
        if parsed[column].isna().all()
    ]
    unevaluated = _formula_columns_without_values(contents, sheet, blank) if blank else []
    if unevaluated:
        raise ValueError(
            f"The workbook contains formulas without saved results ({', '.join(unevaluated)}), "
            "so their values cannot be read. Open it in Excel and save it again, or export "
            "the values, and upload the new copy."
        )
    return parsed


def read_tabular_file(
    contents: bytes,
    filename: str,
    sheet_name: str | None = None,
) -> pd.DataFrame:
    """Read CSV bytes, or the chosen (default: first) worksheet of a workbook."""
    suffix = _validate(contents, filename)
    if suffix in EXCEL_SUFFIXES:
        try:
            return _read_excel(contents, sheet_name if sheet_name is not None else 0)
        except (BadZipFile, KeyError) as error:
            raise ValueError(NOT_A_WORKBOOK) from error
        except (ParseError, InvalidFileException) as error:
            raise ValueError(DAMAGED_WORKBOOK) from error

    parse_errors: list[Exception] = []
    for encoding in _candidate_encodings(contents):
        try:
            return _parse_csv(contents, encoding)
        except (UnicodeDecodeError, pd.errors.ParserError) as error:
            parse_errors.append(error)
    raise ValueError("The file could not be parsed as CSV.") from parse_errors[-1]
