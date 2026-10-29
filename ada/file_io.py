"""Read uploaded CSV, TSV, text and Excel files, list bundled samples, and export formula-safe CSV."""

from __future__ import annotations

import csv
from io import BytesIO
from pathlib import Path

import pandas as pd
from pandas.io.parsers.readers import STR_NA_VALUES
from python_calamine import CalamineError

from .numeric import to_numbers
from .schema import is_identifier_name

SUPPORTED_SUFFIXES = {".csv", ".tsv", ".txt", ".data", ".dat", ".xlsx", ".xlsm"}
SAMPLES_DIRECTORY = Path(__file__).parent / "samples"
# Title-casing a filename gets "Saas" and "Mrr" wrong; these keep their shape.
SAMPLE_ACRONYMS = {"Saas": "SaaS", "Mrr": "MRR", "Arr": "ARR", "Ar": "AR", "Kpi": "KPI"}
EXCEL_SUFFIXES = {".xlsx", ".xlsm"}
# python-calamine reads a workbook several times faster than openpyxl: a 178,000-row
# sheet in 4 s rather than 20.
EXCEL_ENGINE = "calamine"
# Instrument and analysis software writes tab-separated text under these names, and data
# archives such as the UCI repository write comma-separated text as .data.
TAB_SUFFIXES = {".tsv", ".txt", ".data", ".dat"}


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
        raise ValueError("ADA supports CSV, TSV, TXT, DATA, DAT, XLSX and XLSM files.")
    if not contents:
        raise ValueError("The uploaded file is empty.")
    return suffix


def list_excel_sheets(contents: bytes, filename: str) -> list[str]:
    """Return a workbook's worksheet names, or an empty list for CSV files."""
    suffix = _validate(contents, filename)
    if suffix not in EXCEL_SUFFIXES:
        return []
    try:
        with pd.ExcelFile(BytesIO(contents), engine=EXCEL_ENGINE) as workbook:
            return [str(name) for name in workbook.sheet_names]
    except CalamineError as error:
        raise ValueError("The file is not a valid Excel workbook.") from error


# pandas' default missing-value markers except "NA", which in business data is
# usually a region code (North America), plus "\N", the null that MySQL and
# PostgreSQL write when they export a table.
NA_VALUES = sorted(STR_NA_VALUES - {"NA"} | {"\\N"})
def _kept_as_text(columns) -> dict[str, type]:
    """Read key columns (see ``schema.is_identifier_name``) as text, so "00042" keeps its zeros."""
    return {column: str for column in columns if is_identifier_name(str(column), columns)}


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


def _header_separator(contents: bytes, encoding: str, beyond_commas: bool = False) -> dict:
    """Pick ";", tab or "|" when that separator is in the header and splits the file into columns.

    Only the header line is checked; sniffing the body would split a one-column file
    of sentences on their semicolons. With ``beyond_commas`` the header must hold
    the separator more often than commas. A split that leaves every column but the
    first empty is a title line with a semicolon above comma-separated rows, not a
    header. Returns the read_csv options, empty for a comma.
    """
    header = _header_line(contents, encoding)
    for separator in (";", "\t", "|"):
        if separator not in header or beyond_commas and header.count(separator) <= header.count(","):
            continue
        try:
            candidate = _read_csv(contents, encoding, sep=separator)
        except pd.errors.ParserError:
            continue
        if len(candidate.columns) > 1 and candidate.iloc[:, 1:].notna().any().any():
            return {"sep": separator}
    return {}


def _resplit_options(parsed: pd.DataFrame, contents: bytes, encoding: str) -> dict:
    """Pick ";", tab or "|" for a one-column parse when that separator is in the header."""
    return _header_separator(contents, encoding) if len(parsed.columns) == 1 else {}


# Lines above the header that an export may add (a title, a source, a date), at most.
MAX_LINES_ABOVE_HEADER = 10
# Rows below a candidate header that must be as wide as it.
HEADER_CHECK_ROWS = 5


def _lines_above_header(contents: bytes, encoding: str) -> dict:
    """Skip the lines a report or data portal writes above the header.

    A report puts a title there; a portal adds lines such as "Data Source" and
    "Last Updated Date". Read as the header, they give one column, or fail on
    the first wider row. The header is the first line as wide as the rows
    below it, with every line above it narrower. Returns the read_csv options,
    empty when the first line is the header.
    """
    # Decode before splitting: in UTF-16 the newline is two bytes.
    lines = contents.decode(encoding, errors="replace").lstrip("\ufeff").splitlines()
    lines = lines[: MAX_LINES_ABOVE_HEADER + HEADER_CHECK_ROWS + 1]
    # Only commas count as delimiters here, so a one-column file of sentences
    # containing semicolons is left as it is.
    widths = [len(row) if any(cell.strip() for cell in row) else 0 for row in csv.reader(lines)]
    for start in range(1, min(MAX_LINES_ABOVE_HEADER, len(widths) - 1) + 1):
        width = widths[start]
        below = [row for row in widths[start + 1 : start + 1 + HEADER_CHECK_ROWS] if row]
        if width < 2 or not below or any(row != width for row in below):
            continue
        return {"skiprows": start} if max(widths[:start]) < width else {}
    return {}


# Bytes read to settle the separator and header line before the whole file is parsed.
SNIFF_BYTES = 256 * 1024


def _sample(contents: bytes, encoding: str) -> bytes:
    """The opening lines of the file, cut at a line end, or the whole file when it is short.

    UTF-16 is never cut, since its newline is two bytes and a cut could split one.
    """
    if len(contents) <= SNIFF_BYTES or encoding == "utf-16":
        return contents
    end = contents.rfind(b"\n", 0, SNIFF_BYTES)
    return contents[: end + 1] if end > 0 else contents


def _parse_csv(contents: bytes, encoding: str, first: dict | None = None) -> pd.DataFrame:
    """Settle the separator and header line from the opening lines, then parse once.

    Deciding on a sample means a large file is parsed once rather than once per
    separator tried. A sample that does not parse (a quoted field cut at its
    line break) falls back to the whole file. ``first`` holds read_csv options
    to try before that, such as a tab for a .tsv file; they are kept when they
    split the header into more than one column.
    """
    sample = _sample(contents, encoding)
    if first:
        try:
            parsed = _read_csv(sample, encoding, **first)
        except pd.errors.ParserError:
            parsed = None
        if parsed is not None and len(parsed.columns) > 1:
            return parsed if sample is contents else _read_csv(contents, encoding, **first)
    try:
        parsed = _read_csv(sample, encoding)
    except pd.errors.ParserError:
        # Lines above the header fail on the first row wider than them, and so
        # does a file split by semicolons whose values hold commas ("Tickets sold (1,000)").
        options = (_header_separator(sample, encoding, beyond_commas=True)
                   or _lines_above_header(sample, encoding))
        if options:
            return _read_csv(contents, encoding, **options)
        if sample is contents:
            raise
        sample = contents
        parsed = _read_csv(sample, encoding)
    options = _resplit_options(parsed, sample, encoding)
    if not options and len(parsed.columns) == 1 and not parsed.empty:
        options = _lines_above_header(sample, encoding)
    if not options and sample is contents:
        return parsed
    return _read_csv(contents, encoding, **options)


# A cell of text longer than this is prose (a citation note, a list of sources), not a value.
PROSE_LENGTH = 80
# A sheet holds only notes when fewer than one filled cell in this many is a number.
NOTE_NUMBER_SHARE = 0.05


def _numbers_in(sheet: pd.DataFrame) -> int:
    """How many cells of a sheet read without a header hold a number."""
    return sum(int(to_numbers(sheet[column]).notna().sum()) for column in sheet.columns)


def _data_sheet(contents: bytes) -> str | None:
    """The worksheet to read when the first one holds only notes, or None to read the first.

    The Maddison Project's workbook opens on "Notes", a citation note, and
    keeps its data in "Full data". A first sheet with prose in it and hardly a
    number is notes; the sheet with the most numbers is read instead.
    """
    with pd.ExcelFile(BytesIO(contents), engine=EXCEL_ENGINE) as workbook:
        names = [str(name) for name in workbook.sheet_names]
        if len(names) < 2:
            return None
        first = workbook.parse(names[0], header=None)
        filled = int(first.notna().sum().sum())
        prose = first.map(lambda value: isinstance(value, str) and len(value) > PROSE_LENGTH).to_numpy().any()
        if not filled or not bool(prose) or _numbers_in(first) >= NOTE_NUMBER_SHARE * filled:
            return None
        counts = {name: _numbers_in(workbook.parse(name, header=None)) for name in names[1:]}
    best = max(counts, key=lambda name: counts[name])
    return best if counts[best] else None


def read_tabular_file(
    contents: bytes,
    filename: str,
    sheet_name: str | None = None,
) -> pd.DataFrame:
    """Read CSV, TSV or text bytes, or the chosen (default: first) worksheet of a workbook.

    A .tsv, .txt, .data or .dat file is read as tab-separated when that splits
    its header, and otherwise as a CSV. When no sheet is chosen and the first
    holds only notes, the sheet with the most numbers is read, and its name is
    kept in the frame's ``attrs["sheet"]``.
    """
    suffix = _validate(contents, filename)
    if suffix in EXCEL_SUFFIXES:
        try:
            found = _data_sheet(contents) if sheet_name is None else None
            sheet = sheet_name if sheet_name is not None else found or 0
            header = pd.read_excel(BytesIO(contents), engine=EXCEL_ENGINE, sheet_name=sheet, nrows=0)
            frame = pd.read_excel(
                BytesIO(contents),
                engine=EXCEL_ENGINE,
                sheet_name=sheet,
                dtype=_kept_as_text(header.columns),
                keep_default_na=False,
                na_values=NA_VALUES,
            )
        except CalamineError as error:
            raise ValueError("The file is not a valid Excel workbook.") from error
        if found:
            frame.attrs["sheet"] = found
        return frame

    parse_errors: list[Exception] = []
    for encoding in _candidate_encodings(contents):
        try:
            return _parse_csv(contents, encoding, {"sep": "\t"} if suffix in TAB_SUFFIXES else None)
        except (UnicodeDecodeError, pd.errors.ParserError) as error:
            parse_errors.append(error)
    raise ValueError("The file could not be parsed as CSV.") from parse_errors[-1]
