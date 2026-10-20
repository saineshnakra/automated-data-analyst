"""Read uploaded CSV and Excel files, list bundled samples, and export formula-safe CSV."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from zipfile import BadZipFile

import pandas as pd
from pandas.io.parsers.readers import STR_NA_VALUES

from .schema import is_identifier_name

SUPPORTED_SUFFIXES = {".csv", ".xlsx", ".xlsm"}
SAMPLES_DIRECTORY = Path(__file__).parent / "samples"
# Title-casing a filename gets "Saas" and "Mrr" wrong; these keep their shape.
SAMPLE_ACRONYMS = {"Saas": "SaaS", "Mrr": "MRR", "Arr": "ARR", "Ar": "AR", "Kpi": "KPI"}
EXCEL_SUFFIXES = {".xlsx", ".xlsm"}


def list_sample_datasets() -> dict[str, Path]:
    """Return bundled sample CSVs keyed by display label, e.g. "SaaS Subscriptions"."""
    if not SAMPLES_DIRECTORY.is_dir():
        return {}
    def label(stem: str) -> str:
        words = stem.replace("-", " ").title().split()
        return " ".join(SAMPLE_ACRONYMS.get(word, word) for word in words)

    return {label(path.stem): path for path in sorted(SAMPLES_DIRECTORY.glob("*.csv"))}


# Spreadsheets treat a cell starting with any of these as a formula (CSV injection).
# Only text is escaped; pandas writes numeric columns as plain numbers.
FORMULA_LEADERS = ("=", "+", "-", "@", "\t", "\r")


def safe_csv(dataframe: pd.DataFrame) -> str:
    """Serialize to CSV with formula-like text cells and headers escaped.

    Each is prefixed with an apostrophe, which spreadsheets display as literal text.
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
    """Return a workbook's worksheet names, or an empty list for CSV files."""
    suffix = _validate(contents, filename)
    if suffix not in EXCEL_SUFFIXES:
        return []
    try:
        with pd.ExcelFile(BytesIO(contents), engine="openpyxl") as workbook:
            return [str(name) for name in workbook.sheet_names]
    except (BadZipFile, KeyError) as error:
        # openpyxl raises KeyError for a zip without [Content_Types].xml.
        raise ValueError("The file is not a valid Excel workbook.") from error


# pandas' default missing-value markers except "NA", which in business data is
# usually a region code (North America).
NA_VALUES = sorted(STR_NA_VALUES - {"NA"})
def _kept_as_text(columns) -> dict[str, type]:
    """Read key columns (see ``schema.is_identifier_name``) as text, so "00042" keeps its zeros."""
    return {column: str for column in columns if is_identifier_name(str(column))}


def _read_csv(contents: bytes, encoding: str, **options) -> pd.DataFrame:
    """Run read_csv after peeking at the header, so key columns stay text."""
    header = pd.read_csv(BytesIO(contents), encoding=encoding, nrows=0, **options)
    return pd.read_csv(
        BytesIO(contents),
        encoding=encoding,
        dtype=_kept_as_text(header.columns),
        keep_default_na=False,
        na_values=NA_VALUES,
        low_memory=False,
        **options,
    )


def _candidate_encodings(contents: bytes) -> tuple[str, ...]:
    """Return the encodings to try: the one a byte-order mark names, else a fallback order.

    UTF-16 (common from Windows tools such as PowerShell) decodes as Latin-1 without
    error, into garbled headers and blank rows, so the BOM has to be checked first.
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
    """Re-read a one-column parse with ";", tab or "|" when that separator is in the header.

    Only the header line is checked; sniffing the body would split a one-column file
    of sentences on their semicolons.
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


def _skip_title_row(parsed: pd.DataFrame, contents: bytes, encoding: str) -> pd.DataFrame:
    """Use the second line as the header when the first is a report title.

    Exported reports often put a title above the table; read as the header, it
    yields a single column in place of the real ones.
    """
    if len(parsed.columns) > 1 or parsed.empty:
        return parsed
    # Decode before splitting: in UTF-16 the newline is two bytes.
    lines = contents.decode(encoding, errors="replace").lstrip("\ufeff").split("\n")[:3]
    if len(lines) < 3:
        return parsed
    title, header, first_row = lines
    # Only commas count as delimiters here, so a one-column file of sentences
    # containing semicolons is left as it is.
    if "," in title or not header.count(",") or header.count(",") != first_row.count(","):
        return parsed
    candidate = _read_csv(contents, encoding, header=1)
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
            sheet = sheet_name if sheet_name is not None else 0
            header = pd.read_excel(BytesIO(contents), engine="openpyxl", sheet_name=sheet, nrows=0)
            return pd.read_excel(
                BytesIO(contents),
                engine="openpyxl",
                sheet_name=sheet,
                dtype=_kept_as_text(header.columns),
                keep_default_na=False,
                na_values=NA_VALUES,
            )
        except (BadZipFile, KeyError) as error:
            raise ValueError("The file is not a valid Excel workbook.") from error

    parse_errors: list[Exception] = []
    for encoding in _candidate_encodings(contents):
        try:
            parsed = _read_csv(contents, encoding)
            parsed = _resplit_single_column(parsed, contents, encoding)
            return _skip_title_row(parsed, contents, encoding)
        except (UnicodeDecodeError, pd.errors.ParserError) as error:
            parse_errors.append(error)
    raise ValueError("The file could not be parsed as CSV.") from parse_errors[-1]
