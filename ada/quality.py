"""Check every numeric column for values that should not drive a result.

Each check reports what it found, how many cells or rows it covers, a few
examples and the rule that fired. Nothing here changes the data except
``mask_sentinels``, which the pipeline applies to the measure alone: an
impossible reading, a fill value or a copied column is something to look at,
not something to delete unseen.

Every rule is general: bounds come from words in the column name (a percentage
can't pass 100, a yield can't be negative, pH runs 0 to 14) or from the column's
own distribution (a value repeated far more often than its neighbours, a value
ten robust standard deviations from the median). Each column is sorted once and
every check works on whole arrays, so a scan of 250,000 rows by 30 columns
takes a fraction of a second.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from .formatting import group_label, is_currency, normalized_name
from .schema import ColumnRoles, is_design_name, is_entity_name, is_identifier_name
from .timeseries import MAD_SCALE

#: Robust z-score (distance from the median in scaled-MAD units) above which a
#: value is called extreme. Ten is far beyond anything a normal or log-normal
#: column produces by chance, so what passes it is usually a slipped decimal,
#: a unit mix-up or a code.
EXTREME_Z = 10.0
#: Width, in robust standard deviations, of the empty stretch that must cut an
#: extreme value off from the rest. Neighbouring values in a continuous tail lie
#: far closer together.
GAP_SPREADS = 3.0
#: Above this share of a column, values beyond EXTREME_Z are a second cluster or
#: a heavy tail rather than stray errors, and are not listed as extreme.
EXTREME_SHARE = 0.02
#: Share of a column's values at its maximum or minimum that reads as a cap or floor.
CAP_SHARE = 0.03
#: Share of a column's values held by one non-round value that reads as filled-in gaps.
IMPUTED_SHARE = 0.10
#: How many times more often than its neighbours a value in the tail must occur
#: to read as a missing-value code.
SPIKE_RATIO = 10
#: A group whose median is this many times every other group's is out of scale.
GROUP_LEVEL_RATIO = 5.0
#: Share of rows in which two columns hold the same non-zero value that reads as a copy.
EQUAL_SHARE = 0.5
#: Columns missing more than this share are listed.
MISSING_SHARE = 0.20
#: Columns whose missing rows overlap this much (Jaccard) are missing together.
BLOCK_OVERLAP = 0.9
#: How far from 0.5 the chance that a missing row's value outranks a present
#: row's must be for missingness to depend on that column.
DEPENDENCY_SEPARATION = 0.35
#: Drop in missing share, in percentage points, that splits groups lacking a column from the rest.
GROUP_MISSING_GAP = 50.0
#: Integer columns with at most this many distinct values hold codes, not quantities.
CODED_LEVELS = 15
#: Rows sampled when screening column pairs and missingness dependencies.
SAMPLE_ROWS = 5_000
#: Above this many columns, only pairs with the measure are compared for copies:
#: every pair of a wide matrix (a spectrum, a panel of samples) costs more than the rest of the scan.
PAIR_COLUMNS = 100

#: Check names, in the order issues are ranked: wrong values before gaps.
CHECKS = {
    "impossible": "Impossible values",
    "sentinel": "Sentinel codes",
    "group_level": "Group out of scale",
    "equal_columns": "Copied values",
    "min_max": "Max below min",
    "extreme": "Extreme values",
    "imputed": "Probable fill values",
    "cap": "Cap or floor",
    "survey_code": "Survey codes",
    "missing_by_group": "Missing by group",
    "missing_dependency": "Missing by value",
    "missing_block": "Missing together",
    "missing": "Mostly missing",
}
#: Checks that flag individual cells, whose rows ``flagged_index`` returns.
VALUE_CHECKS = frozenset({"impossible", "sentinel", "extreme", "survey_code"})

# Words that make a quantity non-negative: a size, an amount of matter, a count.
# Money and sales quantities are left out, since refunds and returns are negative.
NON_NEGATIVE_WORDS = frozenset({
    "age", "weight", "mass", "height", "length", "width", "depth", "diameter", "radius",
    "area", "volume", "distance", "duration", "size", "yield", "coverage", "resolution",
    "concentration", "conc", "density", "abundance", "population", "count", "counts",
    "rain", "rainfall", "precipitation", "precip", "snow", "snowfall", "speed", "humidity",
    "moisture", "salinity", "conductivity", "turbidity", "pressure", "bmi", "pulse",
})
# Words that let a value fall below zero or pass 100 whatever else the name says.
SIGNED_WORDS = frozenset({
    "change", "changes", "diff", "difference", "delta", "net", "balance", "growth", "gain",
    "loss", "return", "returns", "profit", "margin", "residual", "error", "bias", "anomaly",
    "deviation", "offset", "z", "zscore", "correlation", "temperature", "temp", "elevation",
    "altitude", "relative", "yoy", "mom", "variance", "increase", "decrease", "score",
})
PERCENT_WORDS = frozenset({"percent", "percentage", "pct", "perc", "coverage", "humidity"})
# Normalised-difference indices, bounded by construction.
UNIT_INDEX_WORDS = frozenset({"ndvi", "ndwi", "ndmi", "gndvi", "ndre", "nbr"})
DIRECTION_WORDS = frozenset({"direction", "heading", "bearing", "azimuth"})
# Business quantities that may be negative (returns, refunds, adjustments), so a
# few negatives among them are not a sign of error.
BUSINESS_WORDS = frozenset({
    "quantity", "qty", "units", "orders", "amount", "total", "value", "sales", "volume",
})
# Lab exports name percentage columns with a trailing lower-case p: "LYMp", "NEUTp".
_PERCENT_SUFFIX = re.compile(r"(?<![a-z])[A-Z]{2,}p$")
# Amount-of-matter units: mg/kg, cmol(+)/kg, cells/mL, dS/m, ppm.
_AMOUNT_UNIT = re.compile(
    r"\b(?:[mµun]?g|kg|[mµun]?mol|cmol|meq|cfu|cells|iu|[dmµ]?s)\s*(?:\(\+\))?\s*/\s*\w|\bpp[mbt]\b",
    re.IGNORECASE,
)
_MAX_WORDS = {
    "max": "min", "maximum": "minimum", "high": "low", "highest": "lowest", "upper": "lower",
}

# Values that stand for "no reading" in numeric exports: runs of nines (-999,
# 999.9, 9999), a repeated digit (1111, 8888) and large negative powers of ten.
_NINES = [float("9" * width) for width in range(2, 8)] + [
    float("9" * whole + "." + "9" * fraction) for whole in range(2, 6) for fraction in (1, 2)
]
_REPEATED_DIGITS = [float(str(digit) * width) for digit in range(1, 9) for width in range(4, 8)]
SENTINEL_VALUES = np.array(sorted({
    *_NINES, *(-value for value in _NINES), *_REPEATED_DIGITS,
    *(-(10.0**power) for power in range(3, 10)),
}))
_SMALLEST_CODE = float(np.min(np.abs(SENTINEL_VALUES)))
# "Refused" and "don't know" in survey codebooks.
SURVEY_CODES = frozenset({7, 8, 9, 77, 88, 97, 98, 99, 777, 888, 997, 998, 999, 7777, 8888, 9999})
# 0 °F, which a blank Fahrenheit reading becomes when converted to Celsius.
_ZERO_FAHRENHEIT_C = -17.7778


@dataclass(frozen=True)
class QualityIssue:
    """One finding of the data checks: what it covers, why, and the rule that fired."""

    check: str
    columns: tuple[str, ...]
    #: Cells (for a value check) or rows the issue covers.
    rows: int
    #: ``rows`` as a percentage of the column's non-missing values, or of all rows.
    share: float
    #: One plain sentence for a reader.
    detail: str
    #: A short phrase for a list: "HCT: 3 extreme values (316 to 3,715)".
    summary: str
    rule: str
    examples: tuple[str, ...] = ()
    #: The value, cap or group the issue is about, when there is one.
    group: object | None = None
    #: Row labels of the flagged cells, by column, for checks on individual values.
    cells: dict = field(default_factory=dict, repr=False, compare=False)

    @property
    def column(self) -> str:
        return ", ".join(self.columns)

    @property
    def label(self) -> str:
        return CHECKS.get(self.check, self.check)


def show_value(value: float) -> str:
    """Write a flagged value exactly: "3,715", "46.1526", "-100,000".

    Compact forms such as "3.7K" would hide the very digits a reader needs to
    recognise a code or a slipped decimal point.
    """
    number = float(value)
    if number.is_integer() and abs(number) < 1e15:
        return f"{int(number):,}"
    if abs(number) < 1e-3:
        return f"{number:.3g}"
    shown = f"{number:,.6g}"
    return f"{number:,.2f}" if "e" in shown else shown


def _listed(items: list[str], limit: int = 4) -> str:
    """Join names in a sentence: "a, b and c", or "a, b, c, d and 3 more"."""
    if len(items) > limit:
        items = [*items[:limit], f"{len(items) - limit} more"]
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"


def _plural(count: int, singular: str, plural: str | None = None) -> str:
    return f"{count:,} {singular if count == 1 else (plural or singular + 's')}"


def _percent(share: float) -> str:
    return f"{share:.1f}%" if share < 10 else f"{share:.0f}%"


def _name_words(column: str) -> set[str]:
    return set(re.findall(r"[^\W_]+", normalized_name(column)))


def _is_percentage_name(column: str, words: set[str]) -> bool:
    if words & SIGNED_WORDS:
        return False
    return (
        "%" in column
        or bool(words & PERCENT_WORDS)
        or bool(_PERCENT_SUFFIX.search(column.strip()))
    )


def name_bounds(column: str) -> tuple[float, float, str] | None:
    """The range a column's name says its values must lie in, with the rule's wording.

    Returns None when the name implies no bound.
    """
    words = _name_words(column)
    if "ph" in words:
        return 0.0, 14.0, "pH runs from 0 to 14"
    if words & UNIT_INDEX_WORDS:
        return -1.0, 1.0, "a normalised-difference index runs from -1 to 1"
    if words & {"latitude", "lat"}:
        return -90.0, 90.0, "latitude runs from -90 to 90"
    if words & {"longitude", "lon", "lng"}:
        return -180.0, 180.0, "longitude runs from -180 to 180"
    if words & DIRECTION_WORDS:
        return 0.0, 360.0, "a direction runs from 0 to 360 degrees"
    if _is_percentage_name(column, words):
        return 0.0, 100.0, "a percentage runs from 0 to 100"
    if words & SIGNED_WORDS:
        return None
    if words & NON_NEGATIVE_WORDS or _AMOUNT_UNIT.search(column):
        return 0.0, float("inf"), "this quantity can't be negative"
    return None


def _may_go_negative(column: str) -> bool:
    """Whether a few negatives are ordinary here: money, sales quantities, signed readings."""
    return bool(_name_words(column) & (SIGNED_WORDS | BUSINESS_WORDS)) or is_currency(column)


def _is_code(values: np.ndarray) -> np.ndarray:
    """True for values in SENTINEL_VALUES, to four decimals."""
    found = np.zeros(values.shape, dtype=bool)
    large = np.abs(values) >= _SMALLEST_CODE - 0.01
    if large.any():
        rounded = np.round(values[large], 4)
        positions = np.clip(np.searchsorted(SENTINEL_VALUES, rounded), 0, SENTINEL_VALUES.size - 1)
        found[large] = SENTINEL_VALUES[positions] == rounded
    return found


@dataclass
class _Column:
    """One numeric column's finite values, sorted once and shared by every check."""

    name: str
    #: Row labels of the non-missing values, in table order.
    index: pd.Index
    array: np.ndarray
    ordered: np.ndarray
    uniques: np.ndarray
    counts: np.ndarray

    @classmethod
    def of(cls, name: str, series: pd.Series) -> _Column:
        array = series.to_numpy(dtype="float64", na_value=np.nan)
        present = np.isfinite(array)
        index = series.index
        if not present.all():
            array, index = array[present], index[present]
        ordered = np.sort(array)
        if ordered.size:
            starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
            uniques = ordered[starts]
            counts = np.diff(np.r_[starts, ordered.size])
        else:
            uniques = counts = np.array([])
        return cls(name, index, array, ordered, uniques, counts)

    @property
    def size(self) -> int:
        return int(self.array.size)

    @property
    def whole(self) -> bool:
        return bool(np.all(self.uniques == np.round(self.uniques)))

    def cells(self, mask: np.ndarray) -> pd.Series:
        """The values where ``mask`` holds, labelled by their rows."""
        return pd.Series(self.array[mask], index=self.index[mask])

    def quantile(self, share: float) -> float:
        return float(self.ordered[int(round(share * (self.size - 1)))])


def _sorted_median(ordered: np.ndarray) -> float:
    middle = ordered.size // 2
    if ordered.size % 2:
        return float(ordered[middle])
    return float(ordered[middle - 1] + ordered[middle]) / 2


def _robust_spread(ordered: np.ndarray) -> tuple[float, float]:
    """Median and robust standard deviation of sorted values.

    The spread is the scaled MAD, or IQR / 1.349 when more than half the values
    are identical and the MAD is zero; it is 0 when both are.
    """
    if ordered.size == 0:
        return 0.0, 0.0
    centre = _sorted_median(ordered)
    deviations = np.abs(ordered - centre)
    middle = deviations.size // 2
    spread = float(np.partition(deviations, middle)[middle]) * MAD_SCALE
    if spread == 0:
        low = ordered[int(round(0.25 * (ordered.size - 1)))]
        high = ordered[int(round(0.75 * (ordered.size - 1)))]
        spread = float(high - low) / 1.349
    return centre, spread


def _decile_skew(ordered: np.ndarray) -> float:
    """Skewness from the deciles of sorted values: 0 when symmetric, near 1 for a long right tail."""
    low, middle, high = (ordered[int(round(share * (ordered.size - 1)))] for share in (0.1, 0.5, 0.9))
    return float((high + low - 2 * middle) / (high - low)) if high > low else 0.0


def prefers_log_scale(ordered: np.ndarray) -> bool:
    """Whether sorted values are better judged on a log scale of their positive part.

    True for essentially non-negative values whose bulk is more symmetric on a
    log scale, as concentrations, latencies and counts are: a long but ordinary
    right tail then stays within reach. A symmetric reading keeps the linear
    scale, on which a low value is not exaggerated.
    """
    positive = ordered[ordered > 0]
    if positive.size < 10 or ordered[int(0.01 * ordered.size)] < 0:
        return False
    skew = _decile_skew(ordered)
    return skew > 0 and abs(_decile_skew(np.log10(positive))) < skew


def _decimals(values: np.ndarray) -> np.ndarray:
    """Decimal places each value is written to, up to 8 (46.1526 has 4, 39.2 has 1)."""
    places = np.full(values.shape, 8)
    for digits in range(7, -1, -1):
        scaled = values * 10.0**digits
        exact = np.abs(scaled - np.round(scaled)) <= 1e-6 * np.maximum(1.0, np.abs(scaled))
        places = np.where(exact, digits, places)
    return places


@dataclass(frozen=True)
class _Scale:
    """Where a column's bulk sits and how wide it is, on a linear or log scale."""

    centre: float
    spread: float
    log: bool

    @classmethod
    def of(cls, ordered: np.ndarray) -> _Scale:
        if prefers_log_scale(ordered):
            return cls(*_robust_spread(np.log10(ordered[ordered > 0])), True)
        return cls(*_robust_spread(ordered), False)

    @property
    def median(self) -> float:
        return 10**self.centre if self.log else self.centre

    def far(self, values: np.ndarray) -> np.ndarray:
        """True for values more than EXTREME_Z spreads from the centre.

        On a log scale only values above the centre count: a small positive
        value of a skewed quantity is ordinary, and zeros and negatives are
        judged by the bounds checks instead.
        """
        if self.spread == 0:
            return np.zeros(values.shape, dtype=bool)
        if not self.log:
            return np.abs(values - self.centre) > EXTREME_Z * self.spread
        return values > 10 ** (self.centre + EXTREME_Z * self.spread)

    def detached(self, uniques: np.ndarray) -> np.ndarray:
        """True for distinct values cut off from the bulk by an empty stretch.

        Walking outwards from the bulk, a value counts once a gap has opened
        between it and the bulk that is GAP_SPREADS spreads wide and leaves it
        at least twice as far from the centre as the last value before the gap:
        316 after 81 among haematocrits near 40 is cut off, while a cold tail
        running down one degree at a time, or the brightest noons in a column of
        solar readings, is not.
        """
        cut = np.zeros(uniques.shape, dtype=bool)
        if self.spread == 0 or uniques.size < 2:
            return cut
        scaled = np.log10(np.where(uniques > 0, uniques, np.nan)) if self.log else uniques
        distance = np.abs(scaled - self.centre)
        nearer, further = np.minimum(distance[1:], distance[:-1]), np.maximum(distance[1:], distance[:-1])
        gaps = (np.diff(scaled) >= GAP_SPREADS * self.spread) & (further >= 2 * nearer)
        above = scaled[1:] > self.centre
        cut[1:] |= (np.cumsum(gaps & above) > 0) & above
        if not self.log:
            below = scaled[:-1] < self.centre
            cut[:-1] |= (np.cumsum((gaps & below)[::-1])[::-1] > 0) & below
        return cut


def _issue(
    check: str, column: _Column, flagged: pd.Series, detail: str, summary: str, rule: str, **extra
) -> QualityIssue:
    """A value issue over the cells in ``flagged``, a slice of the column's values."""
    return QualityIssue(
        check=check,
        columns=(column.name,),
        rows=int(flagged.size),
        share=flagged.size / max(column.size, 1) * 100,
        detail=detail,
        summary=summary,
        rule=rule,
        examples=tuple(show_value(value) for value in pd.unique(flagged.to_numpy())[:5]),
        cells={column.name: flagged.index},
        **extra,
    )


def _range_of(values: pd.Series) -> str:
    low, high = float(values.min()), float(values.max())
    return show_value(low) if low == high else f"{show_value(low)} to {show_value(high)}"


def _impossible(column: _Column, coded: np.ndarray) -> tuple[QualityIssue | None, np.ndarray]:
    """Values outside what the name allows, or sign slips in a column otherwise never negative.

    Cells already read as a missing-value code (``coded``) are left to that check.
    """
    nothing = np.zeros(column.size, dtype=bool)
    bounds = name_bounds(column.name)
    if bounds is not None:
        low, high, wording = bounds
        outside = ((column.array < low) | (column.array > high)) & ~coded
        # Mostly outside means another scale than the name suggests, not errors.
        if not outside.any() or outside.mean() > 0.2:
            return None, nothing
        flagged = column.cells(outside)
        if high == float("inf"):
            what = _plural(flagged.size, "negative value")
        elif flagged.min() >= low:
            what = f"{_plural(flagged.size, 'value')} above {show_value(high)}"
        elif flagged.max() <= high:
            what = f"{_plural(flagged.size, 'value')} below {show_value(low)}"
        else:
            what = f"{_plural(flagged.size, 'value')} outside {show_value(low)} to {show_value(high)}"
        detail = f"{column.name} has {what} ({_range_of(flagged)}), though {wording}."
        summary = f"{column.name}: {what} ({_range_of(flagged)})"
        return _issue("impossible", column, flagged, detail, summary, wording), outside
    negative = (column.array < 0) & ~coded
    count = int(negative.sum())
    if (
        count == 0
        or column.size < 30
        or count > max(1, 0.01 * column.size)
        or _may_go_negative(column.name)
        or column.uniques.size < 10
    ):
        return None, nothing
    # A sign slip: every negative is as large as the usual positive values, with
    # nothing between it and zero. A tail running down through zero (cold days
    # in °F, a vegetation index over bare soil) is a real range instead.
    positive = column.ordered[column.ordered > 0]
    if positive.size == 0 or -float(column.array[negative].max()) < float(np.percentile(positive, 5)):
        return None, nothing
    flagged = column.cells(negative)
    rule = "negative values the size of the usual positive ones, in a column otherwise never negative"
    detail = (
        f"{column.name} has {_plural(count, 'negative value')} ({_range_of(flagged)}) while every "
        "other value is above zero and of the same size, so the minus sign is probably an entry error."
    )
    summary = f"{column.name}: {_plural(count, 'negative value')} ({_range_of(flagged)})"
    return _issue("impossible", column, flagged, detail, summary, rule), negative


def _isolated(value: float, others: np.ndarray) -> bool:
    """Whether no other value lies within a factor of two of ``value``.

    A code stands alone: 999.9 among wind speeds below 20. The same number in
    the middle of a populated range (11,111 among readings up to 39,000) is a
    reading that happens to look like one.
    """
    low, high = sorted((value / 2, value * 2))
    return int(np.searchsorted(others, high, side="right") - np.searchsorted(others, low)) == 0


def _standing_codes(column: _Column, ordered: np.ndarray) -> tuple[list[float], float]:
    """Code values that stand alone far from the rest of ``ordered``, and the rest's median.

    Far is judged on the linear scale, since a code is a number typed in place
    of a reading and not part of the column's spread: 999.9 among wind speeds
    near 3 is far whatever the tail of the speeds looks like. A negative code
    in a column that is otherwise not negative is far by that alone.
    """
    is_code = _is_code(column.uniques)
    if not is_code.any():
        return [], 0.0
    codes = column.uniques[is_code]
    others = column.uniques[~is_code]
    rest = ordered[~np.isin(ordered, codes)]
    if rest.size == 0:
        return [], 0.0
    centre, spread = _robust_spread(rest)
    non_negative = (rest < 0).mean() <= 0.01
    standing = [
        float(value)
        for value in codes
        if _isolated(float(value), others)
        and ((spread > 0 and abs(value - centre) > EXTREME_Z * spread) or (value < 0 and non_negative))
    ]
    return standing, centre


def _sentinels(column: _Column, ordered: np.ndarray) -> list[QualityIssue]:
    """Code values standing alone far from the rest (-999, 999.9, -100000)."""
    issues = []
    standing, centre = _standing_codes(column, ordered)
    for value in standing:
        flagged = column.cells(column.array == value)
        detail = (
            f"{column.name} holds {show_value(value)} in {_plural(flagged.size, 'row')}, far from its "
            f"other values (median {show_value(centre)}): a code for a missing reading, "
            "not a measurement."
        )
        summary = (
            f"{column.name}: {show_value(value)} in {_plural(flagged.size, 'row')}, a missing-value code"
        )
        rule = "a code value (a run of nines, a repeated digit, a large negative power of ten) standing alone"
        issues.append(_issue("sentinel", column, flagged, detail, summary, rule, group=value))
    return issues


def _spikes(column: _Column, taken: np.ndarray) -> list[QualityIssue]:
    """A value in the outer 1% repeated SPIKE_RATIO times more often than its neighbours.

    Holding a fifth or more of that tail, it reads as a converted blank (0 °F
    becoming -17.78 °C). A column with several such values has a texture
    instead: a counter that plateaus, readings rounded to quarters.
    """
    if column.uniques.size < 10 or column.size < 100 or column.counts.max() < 10:
        return []
    counts = column.counts.astype(float)
    # The busiest of the two nearest distinct values on each side.
    padded = np.pad(counts, 2)
    neighbours = np.max(np.vstack([padded[:-4], padded[1:-3], padded[3:-1], padded[4:]]), axis=0)
    inner = np.zeros(counts.size, dtype=bool)
    inner[1:-1] = True  # the minimum and maximum belong to the cap check
    outer = (column.uniques < column.quantile(0.05)) | (column.uniques > column.quantile(0.95))
    spiking = inner & outer & (counts >= 10) & (counts >= SPIKE_RATIO * np.maximum(neighbours, 1))
    spiking &= column.uniques != 0
    if not spiking.any() or spiking.sum() > 2:
        return []
    low, high = column.quantile(0.01), column.quantile(0.99)
    tails = {True: int((column.array <= low).sum()), False: int((column.array >= high).sum())}
    places = np.pad(_decimals(column.uniques), 2)
    issues = []
    for position in np.flatnonzero(spiking):
        value = float(column.uniques[position])
        # 5.8 among 5.79 and 5.81 is common because it is rounder, not because it is a code.
        rounder = places[position + 2] < max(places[position : position + 5])
        if low < value < high or rounder or counts[position] < 0.2 * max(tails[value <= low], 1):
            continue
        cells = (column.array == value) & ~taken
        if not cells.any():
            continue
        flagged = column.cells(cells)
        times = counts[position] / max(neighbours[position], 1)
        reason = ""
        if abs(value - _ZERO_FAHRENHEIT_C) < 0.01 and _name_words(column.name) & {"temp", "temperature"}:
            reason = "; it is 0 °F in °C, so probably a blank Fahrenheit reading converted"
        detail = (
            f"{column.name} holds {show_value(value)} in {_plural(flagged.size, 'row')}, about "
            f"{times:.0f}× as often as the values around it{reason}. A value standing out like this "
            "is usually a missing-value code."
        )
        summary = (
            f"{column.name}: {show_value(value)} in {_plural(flagged.size, 'row')}, "
            "probably a missing-value code"
        )
        rule = f"a value in the outer 1% occurring at least {SPIKE_RATIO}× as often as its neighbours"
        issues.append(_issue("sentinel", column, flagged, detail, summary, rule, group=value))
    return issues


def _caps(column: _Column) -> list[QualityIssue]:
    """One value piled up at the maximum or minimum: an instrument limit, a time-out, a top code."""
    if column.uniques.size < 5:
        return []
    issues = []
    busiest = int(column.counts.max())
    # The three nearest distinct values on each side: a limit piles up on one
    # value, while small counts thin out smoothly from their minimum.
    nearest = {"maximum": column.counts[-4:-1], "minimum": column.counts[1:4]}
    for side, position in (("maximum", -1), ("minimum", 0)):
        value = float(column.uniques[position])
        count = int(column.counts[position])
        if value == 0 or count < 3 or count < 3 * int(nearest[side].max()):
            continue
        # Counts start at 0 or 1, so piling up there is their shape, not a floor.
        if side == "minimum" and value == 1 and column.whole:
            continue
        # A share no continuous reading reaches by chance, or the most frequent value.
        if not (count / column.size >= CAP_SHARE or (count == busiest and count >= 5)):
            continue
        flagged = column.cells(column.array == value)
        kind, bias = ("cap", "down") if side == "maximum" else ("floor", "up")
        share = _percent(count / column.size * 100)
        detail = (
            f"{column.name} sits exactly at its {side}, {show_value(value)}, in "
            f"{_plural(count, 'row')} ({share}), which looks like a {kind} (a limit of the "
            "instrument or the protocol); true values there may lie beyond it, which biases the "
            f"mean {bias}."
        )
        summary = f"{column.name}: {_plural(count, 'row')} ({share}) piled at its {side}, {show_value(value)}"
        rule = f"one value at the {side} in {CAP_SHARE:.0%} or more of rows, or the most frequent value"
        issues.append(_issue("cap", column, flagged, detail, summary, rule, group=(side, value)))
    return issues


def _imputed(column: _Column) -> tuple[QualityIssue | None, np.ndarray]:
    """A non-round value repeated in a large share of rows, as mean imputation leaves behind."""
    nothing = np.zeros(column.size, dtype=bool)
    if column.size < 20 or column.uniques.size < 5:
        return None, nothing
    position = int(np.argmax(column.counts))
    count = int(column.counts[position])
    if count / column.size < IMPUTED_SHARE:
        return None, nothing
    value = float(column.uniques[position])
    places = _decimals(column.uniques)
    # Non-round: written to more decimals than nine in ten of the column's values.
    if places[position] <= np.percentile(places, 90) or places[position] < 2:
        return None, nothing
    cells = column.array == value
    flagged = column.cells(cells)
    others = column.array[~cells]
    mean_note = ""
    if others.size and abs(float(np.mean(others)) - value) <= 0.1 * float(np.std(others)):
        mean_note = f", close to the mean of the other values ({show_value(float(np.mean(others)))})"
    share = _percent(count / column.size * 100)
    detail = (
        f"{column.name} holds {show_value(value)} in {_plural(count, 'row')} ({share}){mean_note}. "
        "A non-round value this common is usually a filled-in gap, so these rows are probably "
        "missing values reported as complete."
    )
    summary = (
        f"{column.name}: {show_value(value)} in {_plural(count, 'row')} ({share}), probably filled-in gaps"
    )
    rule = f"one non-round value in {IMPUTED_SHARE:.0%} or more of rows"
    return _issue("imputed", column, flagged, detail, summary, rule, group=value), cells


def _extremes(column: _Column, scale: _Scale, taken: np.ndarray) -> QualityIssue | None:
    """Values more than EXTREME_Z robust standard deviations from the median, cut off
    from the rest of the column by a gap."""
    if column.size < 12 or column.uniques.size < 5:
        return None
    distant = scale.far(column.uniques)
    if not distant.any():
        return None
    candidates = column.uniques[distant & scale.detached(column.uniques)]
    if not candidates.size:
        return None
    far = np.isin(column.array, candidates) & ~taken
    if not far.any() or far.mean() > EXTREME_SHARE:
        return None
    flagged = column.cells(far)
    on_log = " on a log scale" if scale.log else ""
    detail = (
        f"{column.name} has {_plural(flagged.size, 'value')} more than {EXTREME_Z:.0f} robust "
        f"standard deviations from its median of {show_value(scale.median)}{on_log} "
        f"({_range_of(flagged)}), cut off from the other values; values this far out are often "
        "entry or unit errors, and they move means and correlations."
    )
    summary = f"{column.name}: {_plural(flagged.size, 'extreme value')} ({_range_of(flagged)})"
    rule = (
        f"robust z-score (median and scaled MAD{on_log}) above {EXTREME_Z:.0f}, cut off from the "
        f"rest by a gap of {GAP_SPREADS:.0f} or more"
    )
    return _issue("extreme", column, flagged, detail, summary, rule)


def _survey_codes(column: _Column) -> QualityIssue | None:
    """Codebook values (7, 9, 77, 99) sitting above a small set of answer codes."""
    if not column.whole:
        return None
    levels = [int(value) for value in column.uniques]
    codes = [level for level in levels if level in SURVEY_CODES]
    answers = [level for level in levels if level not in SURVEY_CODES]
    if not codes or len(answers) < 2 or min(codes) <= max(answers) + 1:
        return None
    cells = np.isin(column.array, codes)
    if cells.mean() > 0.1:
        return None
    flagged = column.cells(cells)
    shown = _listed([str(code) for code in codes])
    detail = (
        f"{column.name} holds {shown} in {_plural(flagged.size, 'row')}, well above its other "
        f"codes ({_listed([str(level) for level in answers])}); in survey data these usually mean "
        "refused or don't know, and they are kept as values here."
    )
    summary = f"{column.name}: {shown} in {_plural(flagged.size, 'row')}, probably refused or don't know"
    rule = "a 7/9/77/99-style code above the rest of a small set of integer codes"
    return _issue("survey_code", column, flagged, detail, summary, rule)


def _is_coded(column: _Column, factors: tuple[str, ...]) -> bool:
    """Whether a column holds category codes: a named factor, or a few whole numbers.

    A name that bounds the values (a humidity, a percentage) marks a reading even
    when it is recorded in whole numbers.
    """
    if column.name in factors:
        return True
    few = column.uniques.size <= CODED_LEVELS and column.whole
    return few and name_bounds(column.name) is None


def _value_issues(column: _Column, factors: tuple[str, ...]) -> list[QualityIssue]:
    if column.size == 0:
        return []
    if _is_coded(column, factors):
        found = _survey_codes(column)
        return [found] if found else []
    caps = _caps(column)
    imputed, filled = _imputed(column)
    # A fill value or a pile-up at a limit would sit in the bulk and shrink the
    # spread, so the scale comes from the other values.
    piled = [issue.group[1] for issue in caps] + ([imputed.group] if imputed else [])
    if piled:
        filled = filled | np.isin(column.array, piled)
    ordered = column.ordered[~np.isin(column.ordered, piled)] if piled else column.ordered
    scale = _Scale.of(ordered)
    # A code stands for a missing reading before it is a negative or out-of-bounds value.
    nothing = np.zeros(column.size, dtype=bool)
    codes = _sentinels(column, ordered) + _spikes(column, nothing)
    coded = np.zeros(column.size, dtype=bool)
    for found in codes:
        coded |= column.index.isin(found.cells[column.name])
    impossible, taken = _impossible(column, coded)
    issues: list[QualityIssue] = [impossible] if impossible else []
    taken |= coded
    issues.extend(codes)
    issues.extend(caps)
    if imputed:
        issues.append(imputed)
    extreme = _extremes(column, scale, taken | filled)
    if extreme:
        issues.append(extreme)
    return issues


def _merge_alike(issues: list[QualityIssue]) -> list[QualityIssue]:
    """Fold the same code, cap or fill pattern across columns into one issue.

    "MaxTemp, MinTemp and MeanTemp hold -17.78" reads better than three issues,
    and a fill value, or extreme values, recurring on the same rows in several
    columns are one finding: one bad record, or one batch of them.
    """
    merged: list[QualityIssue] = []
    groups: dict[tuple, list[QualityIssue]] = {}
    for issue in issues:
        if issue.check in ("sentinel", "cap"):
            groups.setdefault((issue.check, issue.group), []).append(issue)
        elif issue.check in ("imputed", "extreme"):
            rows = issue.cells[issue.columns[0]]
            groups.setdefault((issue.check, len(rows), rows[0], rows[-1]), []).append(issue)
        else:
            merged.append(issue)
    for (check, *_), alike in groups.items():
        merged.append(alike[0] if len(alike) == 1 else _combined(check, alike))
    return merged


def _combined(check: str, alike: list[QualityIssue]) -> QualityIssue:
    columns = tuple(issue.columns[0] for issue in alike)
    named = _listed(list(columns))
    first = alike[0]
    cells = sum(issue.rows for issue in alike)
    rows, share = cells, max(issue.share for issue in alike)
    if check == "extreme":
        rows, share = first.rows, first.share
        detail = (
            f"{named} have values more than {EXTREME_Z:.0f} robust standard deviations from their "
            f"medians on the same {_plural(rows, 'row')}, cut off from the other values; one record or "
            "one batch is probably wrong, or measured on another scale, across all of them."
        )
        summary = f"{len(columns)} columns: extreme values on the same {_plural(rows, 'row')} ({named})"
    elif check == "imputed":
        values = _listed([f"{issue.column} {issue.examples[0]}" for issue in alike], limit=6)
        rows, share = first.rows, first.share
        detail = (
            f"The same {_plural(rows, 'row')} ({_percent(share)}) hold one repeated non-round value "
            f"in {len(columns)} columns ({values}): probably gaps filled with each column's mean, so "
            "these rows are missing values reported as complete."
        )
        summary = (
            f"{_plural(rows, 'row')} ({_percent(share)}) hold a fill value in {len(columns)} "
            f"columns ({named})"
        )
    elif check == "cap":
        side, value = first.group
        kind = "cap" if side == "maximum" else "floor"
        detail = (
            f"{named} sit exactly at their {side}, {show_value(value)}, in {_plural(cells, 'value')} "
            f"in all, which looks like a shared {kind}; true values there may lie beyond it."
        )
        summary = f"{named}: {_plural(cells, 'value')} piled at their {side}, {show_value(value)}"
    else:
        each = _listed([f"{issue.rows:,}" for issue in alike])
        reason = "; it is 0 °F in °C, so probably blank Fahrenheit readings" if "0 °F" in first.detail else ""
        detail = (
            f"{named} hold {first.examples[0]} in {each} rows, standing far out from their other "
            f"values{reason}: a missing-value code, not a measurement."
        )
        summary = f"{named}: {first.examples[0]} in {_plural(cells, 'row')}, a missing-value code"
    return QualityIssue(
        check=check,
        columns=columns,
        rows=rows,
        share=share,
        detail=detail,
        summary=summary,
        rule=first.rule,
        examples=first.examples,
        group=first.group,
        cells={name: index for issue in alike for name, index in issue.cells.items()},
    )


def _sampled(frame: pd.DataFrame) -> pd.DataFrame:
    """Evenly spaced rows, so screening cost does not grow with the table."""
    if len(frame) <= SAMPLE_ROWS:
        return frame
    step = len(frame) / SAMPLE_ROWS
    return frame.iloc[(np.arange(SAMPLE_ROWS) * step).astype(int)]


def _equal_columns(
    frame: pd.DataFrame, columns: list[str], dimension: str | None, measure: str | None
) -> list[QualityIssue]:
    """Pairs of columns holding the same non-zero value in most rows, overall or in one group.

    Pairs are screened on a sample, overall and per group, so only likely
    copies are counted on the full table. With more than PAIR_COLUMNS columns
    only the measure is paired with the others.
    """
    anchors = len(columns) - 1
    if len(columns) > PAIR_COLUMNS:
        if measure not in columns:
            return []
        columns = [measure, *(column for column in columns if column != measure)]
        anchors = 1
    if len(columns) < 2:
        return []
    sampled = _sampled(frame)
    values = sampled[columns].to_numpy(dtype="float64")
    usable = ~np.isnan(values) & (values != 0)
    codes, levels = pd.factorize(sampled[dimension]) if dimension in frame.columns else (None, [])
    if codes is not None and 0 < len(levels) <= 100:
        groups = np.zeros((len(sampled), len(levels)), dtype="float64")
        groups[np.flatnonzero(codes >= 0), codes[codes >= 0]] = 1.0
    else:
        groups = np.ones((len(sampled), 1), dtype="float64")
    issues = []
    for first in range(anchors):
        rest = slice(first + 1, None)
        anchor = usable[:, first : first + 1]
        both = (anchor & ~np.isnan(values[:, rest])).astype("float64")
        equal = ((values[:, first : first + 1] == values[:, rest]) & anchor).astype("float64")
        present = groups.T @ both
        with np.errstate(divide="ignore", invalid="ignore"):
            shares = np.where(present >= 10, (groups.T @ equal) / present, 0.0)
        for offset in np.flatnonzero(shares.max(axis=0) >= 0.8 * EQUAL_SHARE):
            found = _equal_pair(frame, columns[first], columns[first + 1 + int(offset)], dimension)
            if found:
                issues.append(found)
    return issues


def _equal_pair(frame: pd.DataFrame, first: str, second: str, dimension: str | None) -> QualityIssue | None:
    a, b = frame[first], frame[second]
    both = a.notna() & b.notna()
    if both.sum() < 20:
        return None
    equal = both & a.eq(b) & a.ne(0)
    share = float(equal.sum() / both.sum())
    rule = f"two columns holding the same non-zero value in {EQUAL_SHARE:.0%} or more of rows"
    if share >= EQUAL_SHARE:
        count = int(equal.sum())
        detail = (
            f"{first} equals {second} in {_plural(count, 'row')} ({_percent(share * 100)} of rows "
            "with both); check whether one was copied into the other."
        )
        summary = f"{first} equals {second} in {_percent(share * 100)} of rows"
        return QualityIssue("equal_columns", (first, second), count, share * 100, detail, summary, rule)
    if not dimension or dimension not in frame.columns:
        return None
    rows = both.groupby(frame[dimension]).sum()
    shares = equal.groupby(frame[dimension]).sum() / rows.where(rows >= 20)
    if shares.dropna().empty or shares.max() < EQUAL_SHARE:
        return None
    group = shares.idxmax()
    inside = frame[dimension] == group
    elsewhere = float(equal[~inside].sum() / max(int(both[~inside].sum()), 1))
    if elsewhere > 0.1:
        return None
    label = group_label(dimension, group)
    count = int(equal[inside].sum())
    within = _percent(float(shares.max()) * 100)
    detail = (
        f"In {label}, {first} equals {second} in {_plural(count, 'row')} ({within} of its rows, "
        f"against {elsewhere * 100:.1f}% elsewhere), which suggests one was copied into the other there."
    )
    summary = f"{first} equals {second} in {within} of {label}'s rows"
    return QualityIssue("equal_columns", (first, second), count, float(shares.max() * 100), detail,
                        summary, rule + ", within one group", group=group)


def _sibling(words: list[str]) -> list[str] | None:
    for position, word in enumerate(words):
        if word in _MAX_WORDS:
            return [*words[:position], _MAX_WORDS[word], *words[position + 1 :]]
    return None


def _min_max(frame: pd.DataFrame, columns: list[str]) -> list[QualityIssue]:
    """Rows where a max-named column falls below its min-named sibling, or its plain one."""
    by_words = {tuple(normalized_name(column).split()): column for column in columns}
    issues = []
    for words, high in by_words.items():
        low_words = _sibling(list(words))
        if low_words is None:
            continue
        plain = tuple(word for word in words if word not in _MAX_WORDS)
        low = by_words.get(tuple(low_words)) or by_words.get(plain)
        if not low or low == high:
            continue
        below = frame[high] < frame[low]
        count = int(below.sum())
        if not count:
            continue
        pairs = frame.loc[below, [high, low]].head(3).to_numpy()
        shown = tuple(f"{show_value(x)} < {show_value(y)}" for x, y in pairs)
        both = max(int((frame[high].notna() & frame[low].notna()).sum()), 1)
        detail = (
            f"{high} is below {low} in {_plural(count, 'row')} (for example {shown[0]}), which can't "
            "happen; one of the two is wrong on those rows."
        )
        summary = f"{high} below {low} in {_plural(count, 'row')}"
        rule = "a max-named column below its min-named or plain sibling"
        issues.append(QualityIssue("min_max", (high, low), count, count / both * 100, detail, summary,
                                   rule, shown))
    return issues


def _group_levels(
    frame: pd.DataFrame, column: str, dimension: str, grouping: tuple[np.ndarray, pd.Index]
) -> QualityIssue | None:
    """A group whose median is more than GROUP_LEVEL_RATIO times every other group's.

    Needs three or more groups of five values: with two, one being far higher
    is as likely the effect being studied as a unit or recording problem.
    """
    codes, levels = grouping
    grouped = frame[column].groupby(codes)
    sizes = grouped.count().drop(-1, errors="ignore")
    medians = grouped.median().drop(-1, errors="ignore")[sizes >= 5]
    if len(medians) < 3 or (medians <= 0).any():
        return None
    ordered = medians.sort_values(ascending=False)
    top, runner = float(ordered.iloc[0]), float(ordered.iloc[1])
    if top <= GROUP_LEVEL_RATIO * runner:
        return None
    group = levels[ordered.index[0]]
    label = group_label(dimension, group)
    ratio = top / runner
    detail = (
        f"{label}'s median {column} ({show_value(top)}) is {ratio:.1f}× the highest median of any "
        f"other {dimension} ({show_value(runner)}); check its units and recording before comparing "
        "it with the rest."
    )
    summary = f"{label}'s median {column} is {ratio:.1f}× any other {dimension}'s"
    rule = f"one group's median more than {GROUP_LEVEL_RATIO:.0f}× every other group's"
    rows = int(sizes[ordered.index[0]])
    return QualityIssue("group_level", (column,), rows, rows / max(len(frame), 1) * 100, detail,
                        summary, rule, group=group)


def _missing_issues(
    frame: pd.DataFrame,
    numeric: list[str],
    dimension: str | None,
    grouping: tuple[np.ndarray, pd.Index] | None,
) -> list[QualityIssue]:
    """Columns mostly missing, columns missing together, and gaps that follow a value or a group."""
    rows = len(frame)
    if rows == 0:
        return []
    missing = frame.isna()
    shares = missing.mean()
    gappy = [column for column in frame.columns if 0.05 <= shares[column] < 1.0]
    blocks = _blocks(missing, gappy)
    in_block = {column for block in blocks for column in block}
    issues: list[QualityIssue] = []
    for block in blocks:
        count = int(missing[block].all(axis=1).sum())
        share = count / rows * 100
        detail = (
            f"{_listed(block)} are missing on the same {_plural(count, 'row')} ({_percent(share)}), "
            "which points to a subsample, a separate test or an instrument some rows lack."
        )
        summary = f"{len(block)} columns missing on the same {_plural(count, 'row')} ({_listed(block, 3)})"
        rule = f"columns whose missing rows overlap by {BLOCK_OVERLAP:.0%} or more"
        issues.append(QualityIssue("missing_block", tuple(block), count, share, detail, summary, rule))
    for column in frame.columns:
        if shares[column] > MISSING_SHARE and column not in in_block:
            count = int(missing[column].sum())
            share = _percent(shares[column] * 100)
            detail = f"{column} is missing in {_plural(count, 'row')} ({share})."
            issues.append(QualityIssue("missing", (column,), count, shares[column] * 100, detail,
                                       f"{column} {share} missing", f"more than {MISSING_SHARE:.0%} missing"))
    members = {block[0]: block for block in blocks}
    by_group: list[QualityIssue] = []
    if grouping is not None:
        targets = [block[0] for block in blocks] + [
            column for column in gappy if column not in in_block and column != dimension
        ]
        by_group = _missing_by_group(missing, targets, members, dimension, grouping)
    # Gaps a group explains (a station without the sensor) need no second explanation:
    # not by another column, and not as a block when the group holds all its rows.
    explained = {issue.columns[0] for issue in by_group}
    whole = {issue.columns for issue in by_group if issue.rows >= 0.95 * missing[issue.columns[0]].sum()}
    issues = [issue for issue in issues if not (issue.check == "missing_block" and issue.columns in whole)]
    targets = [block[0] for block in blocks] + [
        column for column in gappy if column not in in_block and shares[column] > MISSING_SHARE
    ]
    issues.extend(by_group)
    issues.extend(_missing_dependencies(
        frame, [target for target in targets if target not in explained], members, numeric
    ))
    return issues


def _blocks(missing: pd.DataFrame, columns: list[str]) -> list[list[str]]:
    """Groups of two or more columns whose missing rows overlap by BLOCK_OVERLAP (Jaccard)."""
    if len(columns) < 2:
        return []
    matrix = missing[columns].to_numpy(dtype="float32")
    together = matrix.T @ matrix
    counts = np.diag(together)
    overlap = together / np.maximum(counts[:, None] + counts[None, :] - together, 1)
    parent = list(range(len(columns)))

    def root(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for first, second in zip(*np.nonzero(np.triu(overlap >= BLOCK_OVERLAP, k=1)), strict=True):
        parent[root(int(first))] = root(int(second))
    grouped: dict[int, list[str]] = {}
    for position, column in enumerate(columns):
        grouped.setdefault(root(position), []).append(column)
    return [block for block in grouped.values() if len(block) >= 2]


def _missing_dependencies(
    frame: pd.DataFrame, targets: list[str], members: dict[str, list[str]], numeric: list[str]
) -> list[QualityIssue]:
    """For each gappy column, the numeric column whose values best separate its missing rows.

    The separation is the chance that a value from a missing row outranks one
    from a present row (the area under the ROC curve), screened on a sample.
    Only columns recorded on at least half the rows are candidates.
    """
    if not targets or not numeric:
        return []
    sample = _sampled(frame[list(dict.fromkeys([*targets, *numeric]))])
    numeric = [column for column in numeric if sample[column].notna().mean() >= 0.5]
    if not numeric:
        return []
    ranks = sample[numeric].rank()
    present = ranks.notna().to_numpy(dtype="float64")
    gaps = sample[targets].isna().to_numpy(dtype="float64")
    missing_with_value = gaps.T @ present
    rank_sums = gaps.T @ ranks.fillna(0).to_numpy(dtype="float64")
    totals = present.sum(axis=0)
    issues = []
    for row, target in enumerate(targets):
        block = members.get(target, [target])
        n_missing = missing_with_value[row]
        n_present = totals - n_missing
        with np.errstate(divide="ignore", invalid="ignore"):
            separation = (rank_sums[row] - n_missing * (n_missing + 1) / 2) / (n_missing * n_present)
        usable = (n_missing >= 10) & (n_present >= 10) & ~np.isin(numeric, block)
        separation = np.where(usable, separation, 0.5)
        best = int(np.argmax(np.abs(separation - 0.5)))
        if abs(separation[best] - 0.5) < DEPENDENCY_SEPARATION:
            continue
        driver = numeric[best]
        gap = frame[target].isna()
        when_missing = show_value(float(frame.loc[gap, driver].median()))
        when_present = show_value(float(frame.loc[~gap, driver].median()))
        direction = "high" if separation[best] > 0.5 else "low"
        named = _listed(block)
        verb = "are" if len(block) > 1 else "is"
        detail = (
            f"{named} {verb} missing mostly where {driver} is {direction}: median {driver} "
            f"{when_missing} where missing against {when_present} where recorded, so the recorded "
            "values cover only part of the range."
        )
        summary = f"{named} missing mostly where {driver} is {direction}"
        rule = (
            "missing rows separated by another column's values (rank separation beyond "
            f"{0.5 + DEPENDENCY_SEPARATION:.2f})"
        )
        issues.append(QualityIssue("missing_dependency", tuple(block), int(gap.sum()), gap.mean() * 100,
                                   detail, summary, rule, examples=(driver,)))
    return issues


def _missing_by_group(
    missing: pd.DataFrame,
    targets: list[str],
    members: dict[str, list[str]],
    dimension: str,
    grouping: tuple[np.ndarray, pd.Index],
) -> list[QualityIssue]:
    """Columns that some groups lack far more often than the rest.

    The groups are split at the widest drop in missing share, and reported when
    that drop is GROUP_MISSING_GAP points or more: a station without a sensor,
    a site where one test was never run.
    """
    codes, levels = grouping
    if not targets or not 2 <= len(levels) <= 50:
        return []
    labelled = codes >= 0
    sizes = np.bincount(codes[labelled], minlength=len(levels))
    gaps = missing[targets].to_numpy(dtype="float64")[labelled]
    counted = np.column_stack([
        np.bincount(codes[labelled], weights=gaps[:, position], minlength=len(levels))
        for position in range(len(targets))
    ])
    kept = sizes >= 10
    if kept.sum() < 2:
        return []
    shares = pd.DataFrame(counted[kept] / sizes[kept, None] * 100, index=levels[kept], columns=targets)
    issues = []
    for target in targets:
        ordered = shares[target].sort_values(ascending=False, kind="stable")
        drops = -np.diff(ordered.to_numpy())
        split = int(np.argmax(drops))
        if drops[split] < GROUP_MISSING_GAP:
            continue
        lacking, rest = ordered.iloc[: split + 1], ordered.iloc[split + 1 :]
        labels = _listed(sorted(group_label(dimension, group) for group in lacking.index))
        block = members.get(target, [target])
        named = _listed(block)
        single = len(lacking) == 1
        if lacking.min() >= 99.5:
            what = f"{labels} {'never records' if single else 'never record'} {named}"
        else:
            what = f"{labels} {'lacks' if single else 'lack'} {named} in {_span(lacking)} of rows"
        rest_named = f"the other {dimension} values" if len(rest) > 1 else f"the remaining {dimension} value"
        detail = f"{what}, against {_span(rest)} for {rest_named}."
        inside = np.isin(codes, levels.get_indexer(lacking.index))
        rows = int(missing[target].to_numpy()[inside].sum())
        rule = f"missing share dropping by {GROUP_MISSING_GAP:.0f} points or more between groups"
        issues.append(QualityIssue("missing_by_group", tuple(block), rows, float(lacking.max()), detail,
                                   what, rule, group=tuple(lacking.index)))
    return issues


def _span(shares: pd.Series) -> str:
    """A range of percentages, or one when they all round alike: "0%", "12% to 40%"."""
    low, high = f"{shares.min():.0f}%", f"{shares.max():.0f}%"
    return low if low == high else f"{low} to {high}"


def _profiles(frame: pd.DataFrame, roles: ColumnRoles | None) -> dict[str, _Column]:
    """Sorted values of each numeric column that holds measurements.

    Identifiers, row labels, layout numbers and row counters (whole numbers
    unique on every row) are left out: none of them is a reading to check.
    """
    identifier = roles.identifier if roles else None
    profiles = {}
    for column in frame.columns:
        series = frame[column]
        if not is_numeric_dtype(series) or is_bool_dtype(series) or column == identifier:
            continue
        name = str(column)
        if is_design_name(name):
            continue
        profile = _Column.of(name, series)
        if profile.size >= 20 and profile.uniques.size == profile.size and profile.whole:
            continue
        # A key or label name ("Patient", "No") holds keys only when its values are
        # whole numbers; decimal readings under such a name (nitric oxide as "NO") are checked.
        if (is_identifier_name(name) or is_entity_name(name)) and profile.whole:
            continue
        profiles[column] = profile
    return profiles


def scan(frame: pd.DataFrame, roles: ColumnRoles | None = None) -> tuple[QualityIssue, ...]:
    """Run every check and return the issues, the most consequential first.

    Issues are ordered by check (``CHECKS``), the measure's first within each.
    ``roles`` names the measure, the grouping column and the columns that hold
    codes; without it, codes are judged from the values alone and no group
    checks run.
    """
    profiles = _profiles(frame, roles)
    factors = tuple(roles.factors) if roles else ()
    measure = roles.measure if roles else None
    dimension = roles.dimension if roles else None
    issues: list[QualityIssue] = []
    for profile in profiles.values():
        issues.extend(_value_issues(profile, factors))
    issues = _merge_alike(issues)
    quantities = [column for column, profile in profiles.items() if not _is_coded(profile, factors)]
    wide = [column for column in quantities if profiles[column].uniques.size >= 10]
    issues.extend(_equal_columns(frame, wide, dimension, measure))
    issues.extend(_min_max(frame, quantities))
    grouping = pd.factorize(frame[dimension]) if dimension in frame.columns else None
    if measure in quantities and grouping is not None:
        found = _group_levels(frame, measure, dimension, grouping)
        if found:
            issues.append(found)
    issues.extend(_missing_issues(frame, quantities, dimension, grouping))
    order = {check: position for position, check in enumerate(CHECKS)}
    return tuple(sorted(
        issues,
        key=lambda issue: (order.get(issue.check, len(order)), measure not in issue.columns, -issue.rows),
    ))


def quality_frame(issues: tuple[QualityIssue, ...]) -> pd.DataFrame:
    """The issues as a table: check, columns, rows, share, detail, rule and examples."""
    columns = ["Check", "Columns", "Rows", "Share %", "Detail", "Rule", "Examples"]
    return pd.DataFrame(
        [
            [issue.label, issue.column, issue.rows, round(issue.share, 2), issue.detail, issue.rule,
             ", ".join(issue.examples)]
            for issue in issues
        ],
        columns=columns,
    )


def flagged_index(issues: tuple[QualityIssue, ...], column: str) -> pd.Index:
    """Row labels of the cells in ``column`` that a value check flagged."""
    flagged = pd.Index([])
    for issue in issues:
        if issue.check in VALUE_CHECKS and column in issue.cells:
            flagged = flagged.union(issue.cells[column])
    return flagged


def mask_sentinels(frame: pd.DataFrame, column: str | None) -> tuple[pd.DataFrame, str | None]:
    """Set clear sentinel codes in ``column`` to missing, returning the frame and a note.

    Only an unmistakable code goes: a run of nines, a repeated digit or a large
    negative power of ten, standing alone, at least EXTREME_Z robust standard
    deviations from the median, and either negative in a column that is
    otherwise not or ten times the 99th percentile of the rest. Everything else
    stays and is reported by ``scan``. Returns the frame unchanged, and None,
    when nothing qualifies.
    """
    if not column or column not in frame.columns:
        return frame, None
    series = frame[column]
    if not is_numeric_dtype(series) or is_bool_dtype(series):
        return frame, None
    profile = _Column.of(column, series)
    if profile.size < 12:
        return frame, None
    standing, _ = _standing_codes(profile, profile.ordered)
    if not standing:
        return frame, None
    rest = profile.ordered[~np.isin(profile.ordered, standing)]
    ceiling = float(np.percentile(np.abs(rest), 99))
    clear = [
        value
        for value in standing
        if (value < 0 and (rest < 0).mean() <= 0.01) or abs(value) >= 10 * max(ceiling, 1e-12)
    ]
    if not clear:
        return frame, None
    cells = series.isin(clear)
    count = int(cells.sum())
    masked = frame.copy()
    masked[column] = series.mask(cells)
    noun = "cells" if count > 1 else "cell"
    note = (
        f"{count:,} {noun} in {column} holding {_listed([show_value(value) for value in clear])}, a code "
        "for a missing reading far from every other value, were treated as missing."
    )
    return masked, note
