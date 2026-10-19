"""ADA: exploratory analysis for any table, with the arithmetic under every number.

    import ada
    report = ada.analyze("orders.csv")
    print(report)

Importing this package pulls in pandas and numpy and nothing else. Streamlit,
plotly and the optional AI layer live behind extras and are never imported
unless you reach for them.
"""

from .api import (
    DEFAULT_ROW_LIMIT,
    MAX_FILE_BYTES,
    REFUSAL_CODES,
    SCHEMA_VERSION,
    AdaError,
    Analysis,
    NothingToAnalyze,
    Refusal,
    UnreadableData,
    analyze,
    read_file,
    sample,
    samples,
)

__version__ = "0.2.0"

__all__ = [
    "DEFAULT_ROW_LIMIT",
    "MAX_FILE_BYTES",
    "REFUSAL_CODES",
    "SCHEMA_VERSION",
    "AdaError",
    "Analysis",
    "NothingToAnalyze",
    "Refusal",
    "UnreadableData",
    "__version__",
    "analyze",
    "read_file",
    "sample",
    "samples",
]
