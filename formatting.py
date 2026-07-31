"""How ADA writes numbers and periods down.

Kept apart from the calculations so that changing how a figure is displayed
can never change what it is, and so both the analysis modules and the query
engine reach for the same rendering rather than growing their own.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CURRENCY_TOKENS = (
    "revenue",
    "sales",
    "gmv",
    "profit",
    "amount",
    "income",
    "spend",
    "cost",
    "expense",
    "price",
    "balance",
)


def normalized_name(name: str) -> str:
    """Column names compared on meaning rather than punctuation."""
    return " ".join(name.lower().replace("_", " ").replace("-", " ").split())


def is_currency(column: str | None) -> bool:
    return bool(column and any(token in normalized_name(column) for token in CURRENCY_TOKENS))


def format_number(value: float, column: str | None = None, *, compact: bool = True) -> str:
    """Format a metric according to likely business meaning."""
    if not np.isfinite(value):
        return "—"

    absolute = abs(value)
    prefix = "$" if is_currency(column) else ""
    suffix = ""
    scaled = value
    if compact and absolute >= 1_000_000_000:
        scaled, suffix = value / 1_000_000_000, "B"
    elif compact and absolute >= 1_000_000:
        scaled, suffix = value / 1_000_000, "M"
    elif compact and absolute >= 1_000:
        scaled, suffix = value / 1_000, "K"

    if suffix:
        return f"{prefix}{scaled:,.1f}{suffix}"
    if float(value).is_integer() and not is_currency(column):
        return f"{int(value):,}"
    return f"{prefix}{value:,.2f}"


def format_period(period: pd.Timestamp, grain: str) -> str:
    """Name a period the way a reader would say it out loud."""
    if grain == "Q":
        return f"Q{period.quarter} {period.year}"
    if grain in ("W", "D"):
        return period.strftime("%d %b %Y")
    if grain == "Y":
        return period.strftime("%Y")
    return period.strftime("%b %Y")
