"""ADA: exploratory analysis for CSV and Excel tables, with the calculation behind every number.

    import ada
    report = ada.analyze("measurements.csv")
    print(report)

The core depends on pandas, numpy, scipy, statsmodels, ruptures and
python-calamine. Streamlit, plotly and the optional AI layer are extras and are
imported only when used.
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
