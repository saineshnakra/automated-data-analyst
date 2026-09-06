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
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            parsed = pd.read_csv(BytesIO(contents), encoding=encoding, low_memory=False)
            sample = contents[:4096]
            if len(parsed.columns) == 1 and any(separator in sample for separator in (b";", b"\t")):
                parsed = pd.read_csv(
                    BytesIO(contents),
                    encoding=encoding,
                    sep=None,
                    engine="python",
                )
            return parsed
        except (UnicodeDecodeError, pd.errors.ParserError) as error:
            parse_errors.append(error)
    raise ValueError("The file could not be parsed as CSV.") from parse_errors[-1]
