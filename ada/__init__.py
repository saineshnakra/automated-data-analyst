"""ADA: paper assets for a study, and a first look at any table.

    from ada import research
    research.build("study.yml")         # the same as `ada build`

    import ada
    report = ada.analyze("measurements.csv")
    print(report)

The core depends on pandas, numpy, scipy, statsmodels, ruptures and
python-calamine. ``ada.research`` brings in Matplotlib, python-docx and PyYAML
only when it is imported, so ``import ada`` stays quick. Streamlit, plotly and
the optional AI layer are extras and are imported only when used.
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
from .claims import ClaimCheck, check_claim
from .decisions import Decision, UnknownDecision
from .folder import FolderScan, scan
from .sources import Source

__version__ = "0.6.0"

__all__ = [
    "DEFAULT_ROW_LIMIT",
    "MAX_FILE_BYTES",
    "REFUSAL_CODES",
    "SCHEMA_VERSION",
    "AdaError",
    "Analysis",
    "ClaimCheck",
    "Decision",
    "FolderScan",
    "NothingToAnalyze",
    "Refusal",
    "Source",
    "UnknownDecision",
    "UnreadableData",
    "__version__",
    "analyze",
    "check_claim",
    "read_file",
    "sample",
    "samples",
    "scan",
]
