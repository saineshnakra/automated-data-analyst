"""ADA: compute and check numbers from CSV and Excel data, for people and language models.

    import ada
    ada.compute("0.1 + 0.2")            # exactly 0.3, with its steps
    ada.solve("x + y = 10; x - y = 2")  # x = 6, y = 4, with each step and a check
    answer = ada.analyze("measurements.csv")
    check = ada.check_claim("flights.csv", "The average dep_delay was 12.64 minutes.")

    from ada import tools
    tools.definitions()                 # tool definitions for function calling
    tools.run("ada_ask", {"path": "orders.csv", "question": "total revenue by region"})

The core depends on pandas, numpy, scipy, statsmodels, ruptures, python-calamine
and PyYAML. The research workflow, the modelling mode and the browser app are in
ADA Lab (``pip install ada-lab``); ``ada.model`` and ``ada.research`` reach it
when it is installed.
"""

from . import tools
from .algebra import Simplified, Solution, Substitution, check_solution, simplify, solve
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
from .compute import Checked, Computation, check_arithmetic, compute
from .decisions import Decision, UnknownDecision
from .folder import FolderScan, scan
from .joins import ask_folder
from .reading import Reading, check_reading
from .sources import Source

__version__ = "0.8.0"

__all__ = [
    "DEFAULT_ROW_LIMIT",
    "MAX_FILE_BYTES",
    "REFUSAL_CODES",
    "SCHEMA_VERSION",
    "AdaError",
    "Analysis",
    "Checked",
    "ClaimCheck",
    "Computation",
    "Decision",
    "FolderScan",
    "NothingToAnalyze",
    "Reading",
    "Refusal",
    "Simplified",
    "Solution",
    "Source",
    "Substitution",
    "UnknownDecision",
    "UnreadableData",
    "__version__",
    "analyze",
    "ask_folder",
    "check_arithmetic",
    "check_claim",
    "check_reading",
    "check_solution",
    "compute",
    "read_file",
    "sample",
    "samples",
    "scan",
    "simplify",
    "solve",
    "tools",
]


#: Names that moved to ADA Lab, with the module there that holds each.
MOVED = {"model": "ada_lab", "ModelReport": "ada_lab", "research": "ada_lab.research"}


def __getattr__(name: str):
    """Reach ``ada.model``, ``ada.ModelReport`` and ``ada.research`` in ADA Lab, or say how to install it."""
    if name not in MOVED:
        raise AttributeError(f"module 'ada' has no attribute {name!r}")
    import importlib  # noqa: PLC0415

    try:
        module = importlib.import_module(MOVED[name])
    except ImportError as error:
        raise ImportError(
            f"ada.{name} is in ADA Lab now: pip install ada-lab, then use ada_lab.{name}."
        ) from error
    return module if name == "research" else getattr(module, name)
