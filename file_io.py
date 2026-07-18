"""Safe, testable parsing for ADA's supported business files."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd

SUPPORTED_SUFFIXES = {".csv", ".xlsx", ".xlsm"}


def read_tabular_file(contents: bytes, filename: str) -> pd.DataFrame:
    """Read CSV or Excel bytes and return the first available table."""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError("ADA supports CSV, XLSX, and XLSM files.")
    if not contents:
        raise ValueError("The uploaded file is empty.")
    if suffix in {".xlsx", ".xlsm"}:
        return pd.read_excel(BytesIO(contents), engine="openpyxl")

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
