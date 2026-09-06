"""Safe, testable parsing for ADA's supported business files."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from zipfile import BadZipFile

import pandas as pd

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


def list_excel_sheets(contents: bytes, filename: str) -> list[str]:
    """Worksheet names of a workbook; empty for CSV files."""
    suffix = _validate(contents, filename)
    if suffix not in EXCEL_SUFFIXES:
        return []
    try:
        with pd.ExcelFile(BytesIO(contents), engine="openpyxl") as workbook:
            return [str(name) for name in workbook.sheet_names]
    except BadZipFile as error:
        raise ValueError("The file is not a valid Excel workbook.") from error


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
        candidate = pd.read_csv(BytesIO(contents), encoding=encoding, sep=separator, low_memory=False)
        if len(candidate.columns) > 1:
            return candidate
    return parsed


def _skip_title_row(parsed: pd.DataFrame, contents: bytes, encoding: str) -> pd.DataFrame:
    """Use the second line as the header when the first is a report title.

    Exported reports often carry a title above the table. Read as a header it
    yields one column, and every real column is silently discarded.
    """
    if len(parsed.columns) > 1 or parsed.empty:
        return parsed
    lines = contents.split(b"\n")[:3]
    if len(lines) < 3:
        return parsed
    title, header, first_row = (line.decode(encoding, errors="replace") for line in lines)
    # Only a comma is treated as the delimiter here. A one-column file of
    # sentences containing semicolons is a real file, and re-reading it as a
    # table would shred it -- the mistake this rescue is meant to prevent.
    if "," in title or not header.count(",") or header.count(",") != first_row.count(","):
        return parsed
    candidate = pd.read_csv(BytesIO(contents), encoding=encoding, header=1, low_memory=False)
    return candidate if len(candidate.columns) > 1 else parsed


def read_tabular_file(
    contents: bytes,
    filename: str,
    sheet_name: str | None = None,
) -> pd.DataFrame:
    """Read CSV bytes, or the chosen (default: first) worksheet of a workbook."""
    suffix = _validate(contents, filename)
    if suffix in EXCEL_SUFFIXES:
        try:
            return pd.read_excel(
                BytesIO(contents),
                engine="openpyxl",
                sheet_name=sheet_name if sheet_name is not None else 0,
            )
        except BadZipFile as error:
            raise ValueError("The file is not a valid Excel workbook.") from error

    parse_errors: list[Exception] = []
    for encoding in _candidate_encodings(contents):
        try:
            parsed = pd.read_csv(BytesIO(contents), encoding=encoding, low_memory=False)
            parsed = _resplit_single_column(parsed, contents, encoding)
            return _skip_title_row(parsed, contents, encoding)
        except (UnicodeDecodeError, pd.errors.ParserError) as error:
            parse_errors.append(error)
    raise ValueError("The file could not be parsed as CSV.") from parse_errors[-1]
