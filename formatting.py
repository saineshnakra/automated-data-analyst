"""How ADA writes numbers and periods down.

Kept apart from the calculations so that changing how a figure is displayed
can never change what it is, and so both the analysis modules and the query
engine reach for the same rendering rather than growing their own.
"""

from __future__ import annotations

import re

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

CURRENCY_CODES = {
    "eur": "€",
    "gbp": "£",
    "usd": "$",
}

PERCENTAGE_TOKENS = {"rate", "margin", "ratio"}

# Above this magnitude a value is not a percentage, whatever the column is
# called. "Gross Margin" holding 1,250,000 is money that happens to be named
# like a ratio, and reading it as 1250000.0% is worse than leaving it plain.
MAX_PLAUSIBLE_PERCENTAGE = 1_000.0


def normalized_name(name: str) -> str:
    """Column names compared on meaning rather than punctuation."""
    return " ".join(name.lower().replace("_", " ").replace("-", " ").split())


def _name_tokens(name: str) -> set[str]:
    """Return whole-word tokens from a column name."""
    return set(re.findall(r"[a-z0-9]+", normalized_name(name)))


def is_currency(column: str | None) -> bool:
    """Return whether the column name represents currency."""
    if not column:
        return False

    tokens = _name_tokens(column)
    return bool(tokens & set(CURRENCY_TOKENS)) or bool(tokens & set(CURRENCY_CODES))


def is_percentage(column: str | None) -> bool:
    """Return whether the column name represents a percentage."""
    if not column:
        return False

    name = normalized_name(column)
    tokens = _name_tokens(column)

    return "%" in name or bool(tokens & PERCENTAGE_TOKENS)


def currency_symbol(column: str | None) -> str:
    """Return the currency symbol indicated by the column name."""
    if not column:
        return ""

    name = normalized_name(column)
    tokens = _name_tokens(column)

    for code, symbol in CURRENCY_CODES.items():
        if code in tokens or symbol in name:
            return symbol

    if is_currency(column):
        return "$"

    return ""


def _percentage_uses_fraction_scale(
    value: float,
    column_values: pd.Series | None,
) -> bool:
    """Determine the percentage convention once for the available column."""
    if column_values is None:
        return 0 <= value <= 1

    values = pd.to_numeric(column_values, errors="coerce").dropna()
    if values.empty:
        return False

    # A column is treated as fractions only when every observed value is
    # within [0, 1]. Negative or >1 values therefore use percentage points.
    return bool(values.min() >= 0 and values.max() <= 1)


def format_number(
    value: float,
    column: str | None = None,
    *,
    compact: bool = True,
    column_values: pd.Series | None = None,
) -> str:
    """Format a metric according to likely business meaning."""
    if not np.isfinite(value):
        return "—"

    # Currency semantics always take precedence over percentage semantics.
    if is_currency(column):
        absolute = abs(value)
        prefix = currency_symbol(column)
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

        return f"{prefix}{value:,.2f}"

    if is_percentage(column):
        if _percentage_uses_fraction_scale(value, column_values):
            return f"{value * 100:.1f}%"
        if abs(value) <= MAX_PLAUSIBLE_PERCENTAGE:
            return f"{value:.1f}%"
        # Falls through: a ratio-named column holding a currency-sized number
        # is reported as a plain number rather than an absurd percentage.

    absolute = abs(value)
    scaled = value
    suffix = ""

    if compact and absolute >= 1_000_000_000:
        scaled, suffix = value / 1_000_000_000, "B"
    elif compact and absolute >= 1_000_000:
        scaled, suffix = value / 1_000_000, "M"
    elif compact and absolute >= 1_000:
        scaled, suffix = value / 1_000, "K"

    if suffix:
        return f"{scaled:,.1f}{suffix}"

    if float(value).is_integer():
        return f"{int(value):,}"

    return f"{value:,.2f}"


def format_period(period: pd.Timestamp, grain: str) -> str:
    """Name a period the way a reader would say it out loud."""
    if grain == "Q":
        return f"Q{period.quarter} {period.year}"
    if grain in ("W", "D"):
        return period.strftime("%d %b %Y")
    if grain == "Y":
        return period.strftime("%Y")
    return period.strftime("%b %Y")