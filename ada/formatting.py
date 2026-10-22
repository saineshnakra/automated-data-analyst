"""Display formatting for numbers, rates and periods.

Kept separate from the calculations so a display change cannot alter a value,
and so the analysis modules and the query engine share one renderer.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

CURRENCY_TOKENS = (
    "revenue",
    "mrr",
    "arr",
    "turnover",
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

# Above this magnitude a value is not rendered as a percentage, whatever the
# column name: "Gross Margin" holding 1,250,000 is money and prints as a plain number.
MAX_PLAUSIBLE_PERCENTAGE = 1_000.0


# Split camelCase so "netRevenue" normalizes like "net_revenue" and "customerID"
# like "customer id", without splitting short units such as "pH".
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z][a-z])|(?<=[a-z])(?=[A-Z]{2,})")


def normalized_name(name: str) -> str:
    """Lowercase a column name, splitting camelCase, underscores and hyphens into spaces."""
    spaced = _CAMEL_BOUNDARY.sub(" ", str(name))
    return " ".join(spaced.lower().replace("_", " ").replace("-", " ").split())


def _name_tokens(name: str) -> set[str]:
    """Return whole-word tokens from a column name."""
    return set(re.findall(r"[a-z0-9]+", normalized_name(name)))


def is_currency(column: str | None) -> bool:
    """Return whether the column name represents currency."""
    if not column:
        return False

    tokens = _name_tokens(column)
    return bool(tokens & set(CURRENCY_TOKENS)) or bool(tokens & set(CURRENCY_CODES))


# A currency-named column whose values are all whole numbers from 0 to this
# holds coded bands (income bracket 1 to 15), not amounts of money.
MAX_BAND_CODE = 15


def _finite_values(column_values: pd.Series) -> pd.Series:
    values = pd.to_numeric(column_values, errors="coerce").dropna()
    return values[np.isfinite(values)]


def _all_whole(values: pd.Series) -> bool:
    return bool((values == np.floor(values)).all())


def is_money(column: str | None, column_values: pd.Series | None = None) -> bool:
    """Whether a column holds money: a currency name, and values that look like amounts.

    The name alone is not enough. "income" coded 1 to 15 for income brackets
    is a category, and printing its average as "$8.11" would invent a sum of
    money. Without the column values the name decides.
    """
    if not is_currency(column):
        return False
    if column_values is None:
        return True
    values = _finite_values(column_values)
    if values.empty:
        return True
    coded = _all_whole(values) and values.min() >= 0 and values.max() <= MAX_BAND_CODE
    return not coded


def is_percentage(column: str | None) -> bool:
    """Return whether the column name represents a percentage."""
    if not column:
        return False

    name = normalized_name(column)
    tokens = _name_tokens(column)

    return "%" in name or bool(tokens & PERCENTAGE_TOKENS)


def percentage_outranks_currency(column: str | None) -> bool:
    """Whether a name with both currency and percentage signals reads as a percentage.

    True for a "%" sign or a percentage word as the last word: "Profit Margin"
    is a margin, "Margin Amount" is an amount.
    """
    if not column:
        return False
    name = normalized_name(column)
    if "%" in name:
        return True
    words = name.split()
    return bool(words) and words[-1] in PERCENTAGE_TOKENS


def rate_scale(column_values: pd.Series | None, value: float = 0.0) -> str:
    """Return "fraction" if the rate column stores 25% as 0.25, or "points" if as 25.

    Without column values, ``value`` decides: 0 to 1 reads as a fraction.
    """
    return "fraction" if _percentage_uses_fraction_scale(value, column_values) else "points"


def format_rate_change(delta: float, column_values: pd.Series | None) -> str:
    """Write a rate change as unsigned percentage points: "20.0 percentage points".

    20% to 40% is 20 points but 100% relative, so the unit is always spelt out.
    Returns "an unmeasurable amount" for a non-finite delta.
    """
    if not np.isfinite(delta):
        return "an unmeasurable amount"
    points = delta * 100 if rate_scale(column_values, delta) == "fraction" else delta
    return f"{abs(points):.1f} percentage points"


def format_change(
    previous: float,
    current: float,
    column: str | None,
    unit: str,
    column_values: pd.Series | None = None,
    *,
    signed: bool = False,
) -> str | None:
    """Size of the move from ``previous`` to ``current`` in the measure's own terms.

    Amounts and counts change by a percentage of the previous value, rates by
    percentage points, and levels (temperature, pH) by the plain difference.
    Returns None when the change is undefined: a non-finite value, or a
    percentage change from zero.
    """
    if not (np.isfinite(previous) and np.isfinite(current)):
        return None
    delta = current - previous
    sign = ("-" if delta < 0 else "+") if signed else ""
    if unit == "rate":
        return sign + format_rate_change(delta, column_values)
    if unit == "level":
        return sign + format_number(abs(delta), column, column_values=column_values)
    if previous == 0:
        return None
    return sign + format_percentage(abs(delta) / abs(previous) * 100)


def _column_reads_as_percentage(value: float, column_values: pd.Series | None) -> bool:
    """Whether values are plausible percentages, judged on the whole column when given.

    Judging per value would render one column as "551.3%" on one row and "1.4K"
    on the next.
    """
    if column_values is not None:
        values = pd.to_numeric(column_values, errors="coerce").dropna()
        if not values.empty:
            return bool(values.abs().max() <= MAX_PLAUSIBLE_PERCENTAGE)
    return abs(value) <= MAX_PLAUSIBLE_PERCENTAGE


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
    """Whether rates are stored as fractions, i.e. every value is within [0, 1].

    Without column values the single ``value`` decides; an all-NaN column reads
    as points.
    """
    if column_values is None:
        return 0 <= value <= 1

    values = pd.to_numeric(column_values, errors="coerce").dropna()
    if values.empty:
        return False

    return bool(values.min() >= 0 and values.max() <= 1)


# From 1e15 up, _compact_magnitude uses scientific notation ("3.6e+20") rather
# than "360,000,000,000.0B".
SCALES = ((1_000_000_000_000_000, "e15"), (1_000_000_000_000, "T"),
          (1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K"))

# Counts are shortened ("12.0K") only from this size up. Smaller counts, and
# readings of any size, keep their digits: "1.1K" would hide that a heart rate
# of 1,111 is impossible.
COMPACT_COUNT_FROM = 10_000

# Outside this range a figure that is not money is written in scientific
# notation: "3.624e+11" for a spectral intensity, "4.2e-05" for a concentration.
SCIENTIFIC_BELOW = 0.001
SCIENTIFIC_FROM = 1e9

# Significant digits of a figure in scientific notation.
SCIENTIFIC_DIGITS = 4


def _compact_magnitude(absolute: float, *, compact: bool) -> str:
    """Render a positive magnitude on the largest scale that fits ("1.2M"), or "".

    Returns "" below 1,000 or when ``compact`` is false.
    """
    if not compact:
        return ""
    if absolute >= 1_000_000_000_000_000:
        return f"{absolute:.3g}"
    for threshold, suffix in SCALES[1:]:
        if absolute >= threshold:
            return f"{absolute / threshold:,.1f}{suffix}"
    return ""


def _counts_things(value: float, column: str | None, column_values: pd.Series | None) -> bool:
    """Whether a figure is a count, or a mean or median of one: from a count-named column of whole numbers.

    A row count has no column. A reading such as a pressure in pascals can be
    a whole number too, so the name has to say it counts something. The column
    decides when its values are known, so the median of a count column prints
    like its other figures even when it falls between two whole numbers.
    """
    if column:
        # Imported here because metrics builds on this module.
        from .metrics import is_additive_name

        if not is_additive_name(column):
            return False
        if column_values is not None:
            values = _finite_values(column_values)
            return values.empty or _all_whole(values)
    return float(value).is_integer()


def _scientific(value: float, digits: int) -> str:
    """Scientific notation with up to ``digits`` significant digits: "3.624e+11", "4.2e-05"."""
    mantissa, exponent = f"{value:.{max(digits, 1) - 1}e}".split("e")
    if "." in mantissa:
        mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}e{exponent}"


def _plain_number(value: float, extra: int) -> str:
    """A figure that is neither money nor compacted, to about five significant digits.

    Whole numbers print in full, values from 1 to 1,000 with two decimals, up to
    10,000 with one, and beyond that as whole numbers; very small and very large
    ones in scientific notation.
    """
    absolute = abs(value)
    if absolute and (absolute < SCIENTIFIC_BELOW or absolute >= SCIENTIFIC_FROM):
        return _scientific(value, SCIENTIFIC_DIGITS + extra)
    if float(value).is_integer():
        return f"{int(value):,}"
    # Three significant figures below 1, so 0.0012 doesn't print as 0.00.
    if absolute < 1:
        return f"{value:.{3 + extra}g}"
    decimals = 2 if absolute < 1_000 else 1 if absolute < COMPACT_COUNT_FROM else 0
    return f"{value:,.{decimals + extra}f}"


def format_number(
    value: float,
    column: str | None = None,
    *,
    compact: bool = True,
    column_values: pd.Series | None = None,
    extra_decimals: int = 0,
) -> str:
    """Format a value as a percentage, money, a count or a plain number.

    The column name and, when given, ``column_values`` decide which. Only
    money and large counts are compacted ("$1.2K", "12.0K"); ``compact=False``
    writes them out in full. ``extra_decimals`` adds digits to a figure that is
    not compacted (see ``format_values_apart``). Non-finite values render as an
    em dash.
    """
    if not np.isfinite(value):
        return "—"
    extra = max(int(extra_decimals), 0)

    if is_percentage(column) and percentage_outranks_currency(column):
        if _percentage_uses_fraction_scale(value, column_values):
            return f"{value * 100:.{1 + extra}f}%"
        if _column_reads_as_percentage(value, column_values):
            return f"{value:.{1 + extra}f}%"

    # Otherwise money takes precedence over percentage.
    if is_money(column, column_values):
        sign = "-" if value < 0 else ""
        prefix = currency_symbol(column)
        body = _compact_magnitude(abs(value), compact=compact)
        # Sign before the symbol: -$1.2M, not $-1.2M.
        return f"{sign}{prefix}{body}" if body else f"{sign}{prefix}{abs(value):,.{2 + extra}f}"

    if is_percentage(column):
        if _percentage_uses_fraction_scale(value, column_values):
            return f"{value * 100:.{1 + extra}f}%"
        if _column_reads_as_percentage(value, column_values):
            return f"{value:.{1 + extra}f}%"
        # Fall through: a ratio-named column above MAX_PLAUSIBLE_PERCENTAGE
        # renders as a plain number.

    if compact and abs(value) >= COMPACT_COUNT_FROM and _counts_things(value, column, column_values):
        return f"{'-' if value < 0 else ''}{_compact_magnitude(abs(value), compact=True)}"

    return _plain_number(value, extra)


# How many decimals format_values_apart may add before it gives up on telling values apart.
MAX_EXTRA_DECIMALS = 4


def _kept_apart(values: tuple[float, ...], rendered: tuple[str, ...]) -> bool:
    """Whether every two different values have different renderings."""
    seen: dict[str, float] = {}
    return all(seen.setdefault(text, value) == value for value, text in zip(values, rendered, strict=True))


def format_values_apart(
    *values: float,
    column: str | None = None,
    column_values: pd.Series | None = None,
) -> tuple[str, ...]:
    """Format figures quoted together so that different values never print alike.

    At the usual precision 3,004.25 and 3,004.21 both print as "3,004.3", and
    "from 3,004.3 to 3,004.3" hides the change the sentence reports. When any
    two collide, all are written out in full ("3,004.25" and "3,004.21"), with
    more decimals if needed, so they keep one precision. Equal values, and
    values within rounding noise of each other, keep the usual rendering.
    """
    usual = tuple(format_number(value, column, column_values=column_values) for value in values)
    if _kept_apart(values, usual):
        return usual
    for extra in range(MAX_EXTRA_DECIMALS + 1):
        full = tuple(
            format_number(value, column, compact=False, column_values=column_values, extra_decimals=extra)
            for value in values
        )
        if _kept_apart(values, full):
            return full
    return usual


# Capitalised words this short are more often symbols or codes ("Ca", "Na",
# "Gp", "Hgb") than ordinary words, so sentence_case leaves them as written.
MAX_SYMBOL_LETTERS = 3
# Short words common in column names that are ordinary words, not codes.
SHORT_ORDINARY_WORDS = frozenset({
    "age", "air", "day", "dry", "dye", "fat", "gas", "low", "max", "min", "net", "new",
    "old", "raw", "run", "sea", "set", "sex", "sum", "wet", "year",
})


def _is_ordinary_word(word: str) -> bool:
    """A capitalised word with nothing else in capitals or digits, long enough not to be a code."""
    letters = sum(character.isalpha() for character in word)
    return (
        (letters > MAX_SYMBOL_LETTERS or word.lower() in SHORT_ORDINARY_WORDS)
        and word[:1].isupper()
        and word[1:].islower()
        and not any(character.isdigit() for character in word)
    )


def sentence_case(name: str) -> str:
    """Lowercase the ordinary capitalised words of a column name for use mid-sentence.

    "Water Temp (°C)" becomes "water temp (°C)". Acronyms, units, chemical
    symbols and short codes such as "SKU", "OD600", "pH", "Ca", "NO2" and
    "PM2.5" are left as written.
    """
    return " ".join(word.lower() if _is_ordinary_word(word) else word for word in str(name).split())


# Words that already say how a column combines, so with_aggregate doesn't repeat them.
AGGREGATE_SYNONYMS = {
    "total": frozenset({"total", "totals", "sum"}),
    "average": frozenset({"average", "avg", "mean"}),
    "maximum": frozenset({"maximum", "max"}),
    "minimum": frozenset({"minimum", "min"}),
}


def with_aggregate(word: str, column: str) -> str:
    """Put an aggregate word before a column name unless the name already says it.

    "Total" and "Total Rain" make "Total Rain", not "Total Total Rain", and
    "Average" and "Mean Temp" make "Mean Temp". A different aggregate is kept:
    "Average Total Rain" is the average of a total.
    """
    key = word.lower()
    synonyms = AGGREGATE_SYNONYMS.get(key, frozenset({key}))
    if synonyms & set(normalized_name(column).split()):
        return str(column)
    return f"{word} {column}"


def list_names(names: list[str], limit: int = 3) -> str:
    """Join names for a sentence: "A", "A and B", "A, B and C", "A, B, C and 2 others"."""
    if len(names) > limit:
        rest = len(names) - limit
        return f"{', '.join(names[:limit])} and {rest} other{'s' if rest != 1 else ''}"
    if len(names) <= 1:
        return "".join(names)
    return f"{', '.join(names[:-1])} and {names[-1]}"


def group_label(column: str | None, value: object) -> str:
    """Name a group for a sentence: text values as they are, numeric codes with their column.

    A group of a numeric column reads as "Status 2" rather than a bare "2",
    which on its own could be mistaken for a quantity.
    """
    if isinstance(value, bool | np.bool_):
        value = int(value)
    if isinstance(value, int | float | np.integer | np.floating) and column:
        number = float(value)
        shown = f"{int(number)}" if number.is_integer() else f"{number:g}"
        return f"{column} {shown}"
    return str(value)


def format_percentage(value: float, *, signed: bool = False) -> str:
    """Format a percentage, or an em dash when the value is NaN or infinite.

    Percentages come from division, so non-finite inputs are expected; an
    f-string would print them as "nan%".
    """
    if not np.isfinite(value):
        return "—"
    return f"{value:+.1f}%" if signed else f"{value:.1f}%"


#: The day that times with no calendar date are placed on, so a clock reading
#: or an elapsed time sorts and groups like any timestamp. It is the date strptime
#: gives a bare time; labels on it show the time alone, never this date.
UNDATED_DAY = pd.Timestamp("1900-01-01")
# How far past UNDATED_DAY a clock or elapsed-time axis may run and still read as undated.
UNDATED_SPAN = pd.Timedelta(days=366)
CLOCK_FORMATS = {"h": "%H:%M", "min": "%H:%M", "s": "%H:%M:%S"}


def is_undated(moment) -> bool:
    """Whether a timestamp is a time with no calendar date (see ``UNDATED_DAY``)."""
    if moment is None or pd.isna(moment):
        return False
    stamp = pd.Timestamp(moment)
    return UNDATED_DAY <= stamp < UNDATED_DAY + UNDATED_SPAN


def _undated_label(period: pd.Timestamp, grain: str) -> str:
    """Label a time with no calendar date: "15:47", "day 2 00:15", "day 3" or "week 2"."""
    day = (period.normalize() - UNDATED_DAY).days + 1
    if grain in CLOCK_FORMATS:
        clock = period.strftime(CLOCK_FORMATS[grain])
        return clock if day == 1 else f"day {day} {clock}"
    if grain == "D":
        return f"day {day}"
    return f"week {(day - 1) // 7 + 1}"


def format_period(period: pd.Timestamp, grain: str) -> str:
    """Label a period for its grain: "Q3 2024", "05 Mar 2024", "2024", "Mar 2024" or "05 Mar 2024 14:00".

    A time with no calendar date is labelled by its clock time alone ("15:47").
    """
    if is_undated(period):
        return _undated_label(pd.Timestamp(period), grain)
    if grain in CLOCK_FORMATS:
        return period.strftime(f"%d %b %Y {CLOCK_FORMATS[grain]}")
    if grain == "Q":
        return f"Q{period.quarter} {period.year}"
    if grain in ("W", "D"):
        return period.strftime("%d %b %Y")
    if grain == "Y":
        return period.strftime("%Y")
    return period.strftime("%b %Y")


def format_moment(moment: pd.Timestamp) -> str:
    """Write one timestamp for a date range: "05 Mar 2024", or "15:06:28" with no calendar date."""
    stamp = pd.Timestamp(moment)
    if is_undated(stamp):
        return _undated_label(stamp, "s")
    return stamp.strftime("%d %b %Y")