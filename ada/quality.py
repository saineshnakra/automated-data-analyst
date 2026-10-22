"""Check every numeric column for values that should not drive a result.

Each check reports what it found, how many cells or rows it covers, a few
examples and the rule that fired. Nothing here changes the data except
``blank_codes``, which cleaning applies to every numeric column: a code typed
in place of a reading (-999, 999.9) is not a value at all, while an
impossible reading, a fill value or a copied column is something to look at,
not something to delete unseen. ``statistics_frame`` gives the brief a copy
with the cells that can't be readings blanked, so relationships and group
comparisons are not computed on them.

Every rule is general: bounds come from words in the column name (a percentage
can't pass 100, a yield can't be negative, pH runs 0 to 14), from a short table
of physical and clinical limits (``PHYSICAL_LIMITS``), or from the column's own
distribution (a value repeated far more often than its neighbours, a value ten
robust standard deviations from the median). Each column is sorted once and
every check works on whole arrays, so a scan of 250,000 rows by 30 columns
takes a fraction of a second.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype
from scipy import stats

from .formatting import group_label, has_percent_unit, is_currency, normalized_name
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
#: Above this share of a column, values beyond EXTREME_Z are a second cluster
#: rather than stray errors, and are reported as one.
EXTREME_SHARE = 0.02
#: Above this share, values cut off from the bulk are part of the column's shape
#: (two populations of comparable size), not a cluster to point out.
CLUSTER_SHARE = 0.25
#: Fewest values a second cluster holds; fewer are listed as extreme values.
CLUSTER_VALUES = 10
#: Flagged values at least this many times the nearest ordinary value (or that
#: far below it) read as entry or unit errors; nearer ones only need checking.
ERROR_RATIO = 3.0
#: Share of a column's values at its maximum or minimum that reads as a cap or floor.
CAP_SHARE = 0.03
#: How many times as often as any value inside the range a cap or floor must
#: occur: a limit stands out, while a common recipe value at the end of a
#: range is matched by others in the middle.
CAP_STANDOUT = 2
#: A pile at the maximum or minimum holding this share of a column is a limit
#: even when another value inside the range is as common.
CAP_CLEAR_SHARE = 0.10
#: A column of fewer than five values is checked for one value holding this
#: share of it at its minimum or maximum: a reporting limit. From
#: ALMOST_CONSTANT_SHARE such a limit leaves any column almost no information.
NEAR_CONSTANT_SHARE = 0.5
ALMOST_CONSTANT_SHARE = 0.9
#: Shares of a column piled at each end that read as clipping at both (as
#: winsorising at the 0.5th and 99.5th percentiles leaves), and how far apart
#: the two shares may be.
CLIP_SHARES = (0.001, 0.05)
CLIP_LIKENESS = 2.0
#: Values bunched just under a round maximum: how many stretches of 1% of the
#: range below it make the typical count, and how many times that the top
#: stretch must hold.
SQUEEZE_WINDOWS = 20
SQUEEZE_RATIO = 2.5
#: Columns sharing one exact maximum or minimum that read as clipped together
#: (expression values capped at -2 and 2 by a platform), and the share of the
#: readings they must make up.
SHARED_CLIP_COLUMNS = 10
SHARED_CLIP_SHARE = 0.1
#: A value at the very bottom or top repeated at least this often, standing
#: apart from every other value, reads as a code.
EDGE_CODE_VALUES = 10
#: Above this many columns with extreme values of their own, they are listed as one issue.
EXTREME_COLUMNS_LISTED = 10
#: A round value inside a column's range holding this share of it, with this
#: share of its rows in one group, is a limit that group's measurements stop at.
GROUP_CAP_SHARE = 0.05
GROUP_CAP_CONCENTRATION = 0.8
#: A missing-value code on at least FAILED_RECORD_ROWS rows marks a failed
#: record when another column holds one value on FAILED_RECORD_SHARE of them.
FAILED_RECORD_ROWS = 10
FAILED_RECORD_SHARE = 0.9
#: Rows of zeros: readings each 0 on under ZERO_ROW_COLUMN_SHARE of their
#: values, ZERO_ROW_SHARE of them 0 on the same row.
ZERO_ROW_COLUMN_SHARE = 0.05
ZERO_ROW_SHARE = 0.75
#: Cells that stand for "not recorded" when several columns hold them on the same rows.
PLACEHOLDER_TOKENS = frozenset({"0", "-", "--", "—", "–"})
PLACEHOLDER_OVERLAP = 0.98
#: A reading exactly 0 in this share of rows is mostly below detection.
ZERO_HEAVY_SHARE = 1 / 3
#: Share of rows on which a part must stay at or below its whole for the rows past it to stand out.
PART_WHOLE_SHARE = 0.95
#: Share of rows on which one column must stay at or below another for the two
#: to read as one bounded by the other (a time censored at follow-up).
BOUNDED_SHARE = 0.99
# Readings whose zeros are dry days, not values below a limit.
DRY_ZERO_WORDS = frozenset({"rain", "rainfall", "precipitation", "precip", "snow", "snowfall"})
#: Share of a column's values held by one non-round value that reads as filled-in gaps.
IMPUTED_SHARE = 0.10
#: How many times more often than its neighbours a value in the tail must occur
#: to read as a missing-value code.
SPIKE_RATIO = 10
#: Factors one group's values are off by when they were recorded in other units:
#: powers of ten and common conversions (°C steps to °F, kg to lb, m/s to km/h,
#: inches to millimetres, ounces to grams, minutes, hours and days).
CONVERSION_FACTORS = (
    1.8, 2.2046, 3.6, 7.5, 24.0, 25.4, 28.35, 60.0, 453.6, 3600.0,
    10.0, 100.0, 1000.0, 1e4, 1e5, 1e6,
)
#: How close, as a ratio, a group's level must come to a conversion factor.
CONVERSION_TOLERANCE = 1.1
#: Factors below this are within the range real groups differ by, so a group
#: must match one to within SMALL_FACTOR_TOLERANCE, every other group's median
#: must lie within SMALL_FACTOR_SPREAD of the rest, and the group must clear
#: them all by nearly the whole factor.
SMALL_FACTOR = 5.0
SMALL_FACTOR_TOLERANCE = 1.03
SMALL_FACTOR_SPREAD = 1.25
#: Share of rows in which two columns hold the same non-zero value that reads as a copy.
EQUAL_SHARE = 0.5
#: Share of the rows where a column is present on which it must equal the row
#: maximum (or minimum) of other columns to be read as built from them.
ROW_EXTREME_SHARE = 0.95
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
#: Columns that must hold the same negative value on the same rows for it to read
#: as a code for no reading, as a logger writes -1 in every channel when a read fails.
SHARED_CODE_COLUMNS = 3
#: Share of rows repeating an earlier row exactly that is listed. Only rows
#: holding a continuous reading count, since those can't repeat by chance; a
#: table of codes and counts repeats naturally.
DUPLICATE_SHARE = 0.01
#: Fewest values, and distinct values, a column needs before its shape is
#: tested for flatness.
FLAT_MIN_VALUES = 100
FLAT_MIN_LEVELS = 20
#: A flat column passes a Kolmogorov-Smirnov test against a uniform spread
#: over its own range at this level, and has excess kurtosis at most
#: FLAT_KURTOSIS (a uniform spread has -1.2, a bell curve 0).
FLAT_P = 0.05
FLAT_KURTOSIS = -0.6
#: Columns compared when testing whether any pair correlates beyond chance,
#: and the chance of the largest |r| among them passing the level by luck.
FLAT_PAIR_COLUMNS = 200
FLAT_CHANCE = 0.01
#: Readings in a row at exactly zero that read as a sensor that was off.
FLATLINE_RUN = 24
#: Above this share of consecutive repeats outside long runs, a column repeats
#: by its resolution (whole numbers, a slow signal) and runs say nothing.
FLATLINE_REPEATS = 0.2
#: Zeros holding this share of a column are its shape (rain, counts), not a stuck sensor.
FLATLINE_VALUE_SHARE = 0.5
#: Share of a column one run of a single non-zero value must hold to read as a stall.
FLATLINE_STALL = 0.01
#: A block of swapped rows must be at least this long.
SWAP_ROWS = 3
#: Share of a column beyond a quantity's critical limits (sodium above 160)
#: that no real group of patients shows.
CRITICAL_SHARE = 0.05
#: A 0 below codes running from 1 is named when it holds under this share of
#: rows and is at most 1/RARE_CODE_RATIO as common as any other code.
RARE_CODE_SHARE = 0.02
RARE_CODE_RATIO = 5

#: Check names, by kind. Within each kind issues are ranked by the rows they
#: cover; ties follow this order.
CHECKS = {
    "generated": "Looks simulated",
    "duplicates": "Duplicate rows",
    "swapped": "Swapped columns",
    "impossible": "Impossible values",
    "implausible": "Implausible share",
    "sentinel": "Sentinel codes",
    "failed_record": "Failed records",
    "zero_rows": "Rows of zeros",
    "placeholder": "Not recorded",
    "group_level": "Group out of scale",
    "equal_columns": "Copied values",
    "min_max": "Max below min",
    "outside_range": "Average outside min to max",
    "part_whole": "Part above whole",
    "cluster": "Second cluster",
    "extreme": "Extreme values",
    "matrix_rows": "Rows off scale",
    "imputed": "Probable fill values",
    "flatline": "Stuck readings",
    "cap": "Cap or floor",
    "clipped": "Clipped at both ends",
    "survey_code": "Survey codes",
    "rare_code": "Rare codes",
    "zeros": "Mostly zeros",
    "single_reading": "Single readings",
    "bounded": "Bounded by another column",
    "constant": "Constant columns",
    "missing_by_group": "Missing by group",
    "missing_dependency": "Missing by value",
    "missing_block": "Missing together",
    "missing": "Mostly missing",
}
#: Checks that flag individual cells, whose rows ``flagged_index`` returns.
VALUE_CHECKS = frozenset({
    "impossible", "implausible", "sentinel", "extreme", "cluster", "survey_code", "swapped", "rare_code",
    "failed_record", "zero_rows", "placeholder",
})
#: Checks whose cells can't be readings as they stand (an impossible value, a
#: code, a filled-in gap, a value in the wrong column, a stuck sensor, the rest
#: of a failed record, a row of zeros, a placeholder), and which
#: ``statistics_mask`` blanks for computing relationships and comparisons.
#: Extreme values and clusters may be real, so they stay.
MASKED_CHECKS = frozenset({
    "impossible", "sentinel", "imputed", "survey_code", "swapped", "flatline", "failed_record", "zero_rows",
    "placeholder",
})
#: Checks about the whole table, named first when they cover enough of it.
TABLE_CHECKS = frozenset({"generated", "duplicates"})
#: Checks that describe the table rather than flag values to doubt (a constant
#: column, a reading mostly 0, max equal to min where one reading was taken).
#: They rank after the issues about values and are never named first.
INFO_CHECKS = frozenset({"constant", "zeros", "single_reading", "bounded"})

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
    "altitude", "relative", "yoy", "mom", "variance", "increase", "decrease", "score", "strain",
})
PERCENT_WORDS = frozenset({"percent", "percentage", "pct", "perc", "coverage", "humidity"})
# Stretches written in percent, which can't be negative but pass 100 when a
# sample more than doubles its length: an elongation of 150% is a ductile metal.
STRETCH_WORDS = frozenset({"elongation", "extension", "expansion", "swelling"})
# A valid range written in a header: "Sound Level (dBA) (40 - 100)", "[0-14]", "(0 to 100%)".
_HEADER_RANGE = re.compile(
    r"[(\[]\s*(-?\d+(?:\.\d+)?)\s*(?:-|–|—|to)\s*(-?\d+(?:\.\d+)?)\s*%?\s*[)\]]", re.IGNORECASE
)
# Coordinates: where a row was taken, not a reading of it.
LOCATION_WORDS = frozenset({"latitude", "lat", "longitude", "lon", "lng", "easting", "northing"})
# Names whose values are never judged on a log scale (see prefers_log_scale).
_LINEAR_WORDS = SIGNED_WORDS | LOCATION_WORDS | {"ph", "direction", "heading", "bearing", "azimuth"}
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
# The middle of a reading's range, beside its min and max: "pollutant_avg".
MIDDLE_WORDS = frozenset({"avg", "average", "mean", "median", "mid"})
# Words that make a reading the whole its parts add up to or lie within:
# "Total Sales" beside "Online Sales"; for a depth, the water column itself.
WHOLE_WORDS = frozenset({"total", "overall", "whole", "gross", "full", "entire"})
WHOLE_DEPTH_WORDS = frozenset({"water", "bottom", "site", "station"})
# Words naming which reading of several in a window a column holds.
WINDOW_WORDS = frozenset({*_MAX_WORDS, *_MAX_WORDS.values(), "first", "last", "start", "end"})
# Statistics of a spread or a count: a high group's SD or n may well be below a low group's.
SPREAD_WORDS = frozenset({
    "sd", "se", "sem", "std", "stdev", "var", "variance", "iqr", "n", "count", "counts", "size",
})
# Words that make a column a count of subjects or samples, a design number
# whose round values are not an instrument's limit.
SAMPLE_WORDS = frozenset({
    "participants", "subjects", "respondents", "patients", "individuals", "samples", "replicates",
    "animals", "cases", "enrolled", "enrollment",
})
_COUNT_OF = frozenset({"n", "number", "num", "no", "count", "total"})
# Columns ADA adds itself, such as a matrix table's row mean.
_BUILT_NAME = re.compile(r"^Row mean \([\d,]+ columns\)$")


@dataclass(frozen=True)
class _Unit:
    """One unit a reading may be recorded in, with where its values can lie."""

    name: str
    #: Word sets that name this unit in a header ("mmol/L" reads as {"mmol", "l"}).
    spellings: tuple[frozenset, ...]
    #: Where a column's median falls in this unit; it tells g/L from g/dL.
    usual: tuple[float, float]
    #: The values a reading in this unit can take at all.
    limits: tuple[float, float]
    #: Values beyond these are seen only in critical illness, in a small share
    #: of any real group of patients, so a column with many of them is suspect
    #: even though each one alone is possible.
    critical: tuple[float, float] | None = None


@dataclass(frozen=True)
class _Limit:
    """A physical or clinical quantity recognised by name, and its limits by unit."""

    what: str
    #: Word sets that name it once units and words such as "serum" are set aside.
    names: tuple[frozenset, ...]
    units: tuple[_Unit, ...]
    #: Where the limits hold: "in a living patient", "at the Earth's surface".
    setting: str
    #: The kind of table the quantity belongs to ("blood", "water"). Such a
    #: quantity counts only in a table with at least one other of its kind, so
    #: a soil table's K is never read as potassium, nor a lab's pH as a lake's.
    panel: str | None = None
    #: A unit to assume when the header names none and the median can't tell.
    default: str | None = None


def _spelled(*spellings: str) -> tuple[frozenset, ...]:
    return tuple(frozenset(spelling.split()) for spelling in spellings)


def _unit(
    name: str,
    spellings: tuple[str, ...],
    usual: tuple[float, float],
    limits: tuple[float, float],
    critical: tuple[float, float] | None = None,
) -> _Unit:
    return _Unit(name, _spelled(*spellings), usual, limits, critical)


_UNBOUNDED = (-np.inf, np.inf)
_MMOL = ("mmol l", "meq l")
_CELSIUS = ("c", "degc", "celsius")
_FAHRENHEIT = ("f", "degf", "fahrenheit")
_PATIENT = "in a living patient"
_SURFACE_WATER = "in natural surface water"
#: Ranges a reading can take at all, kept wide so that a value outside one is
#: an error and not a rare case: clinical analytes by their standard names and
#: common short forms, in their usual units, as survivable in a living
#: patient; vital signs; wind speed at the Earth's surface; air temperature
#: within the records ever measured; and the temperature, dissolved oxygen and
#: pH of natural surface water. The unit is read from the header, or else from
#: the column's median (haemoglobin near 13 is in g/dL, near 130 in g/L); a
#: column whose median fits no unit is left alone. Electrolytes also carry the
#: critical limits laboratories report at once (sodium above 160).
PHYSICAL_LIMITS = (
    _Limit("sodium", _spelled("sodium", "na", "sod"), (
        _unit("mmol/L", _MMOL, (120, 160), (100, 200), (120, 160)),), _PATIENT, panel="blood"),
    _Limit("potassium", _spelled("potassium", "k", "pot"), (
        _unit("mmol/L", _MMOL, (2.5, 7), (1, 10), (2.5, 6.5)),), _PATIENT, panel="blood"),
    _Limit("chloride", _spelled("chloride", "cl"), (
        _unit("mmol/L", _MMOL, (85, 125), (50, 160), (80, 120)),), _PATIENT, panel="blood"),
    _Limit("calcium", _spelled("calcium", "ca", "t ca", "total calcium"), (
        _unit("mg/dL", ("mg dl",), (6, 14), (2, 22), (6, 13)),
        _unit("mmol/L", _MMOL, (1.5, 3.5), (0.5, 5.5), (1.5, 3.25))), _PATIENT, panel="blood"),
    _Limit("glucose", _spelled("glucose", "glu", "fbs", "rbs", "bgr", "sugar"), (
        _unit("mg/dL", ("mg dl",), (60, 400), (10, 3000)),
        _unit("mmol/L", ("mmol l",), (3, 22), (0.5, 170))), _PATIENT, panel="blood"),
    _Limit("creatinine", _spelled("creatinine", "creatinin", "creat", "scr", "sc"), (
        _unit("mg/dL", ("mg dl",), (0.4, 12), (0.1, 30)),
        _unit("µmol/L", ("umol l", "µmol l", "μmol l"), (35, 1000), (10, 2700))), _PATIENT, panel="blood"),
    _Limit("albumin", _spelled("albumin", "alb"), (
        _unit("g/dL", ("g dl",), (1.5, 6), (1, 7)),
        _unit("g/L", ("g l",), (15, 60), (10, 70))), _PATIENT, panel="blood"),
    _Limit("haemoglobin", _spelled("haemoglobin", "hemoglobin", "hgb", "hb", "hemo"), (
        _unit("g/dL", ("g dl",), (5, 20), (3, 25)),
        _unit("g/L", ("g l",), (50, 200), (30, 250))), _PATIENT, panel="blood"),
    _Limit("haematocrit", _spelled("haematocrit", "hematocrit", "hct", "pcv"), (
        _unit("%", ("pct", "percent"), (15, 65), (10, 75)),
        _unit("L/L", ("l l",), (0.15, 0.65), (0.1, 0.75))), _PATIENT, panel="blood"),
    _Limit("red cell count", _spelled("rbc", "erythrocytes", "red blood cells", "red cell count"), (
        _unit("×10¹²/L", ("x10 12 l", "10 12 l", "x10 6 ul", "10 6 ul", "m ul"), (2, 8), (1, 10)),),
        _PATIENT, panel="blood"),
    _Limit("heart rate", _spelled("heart rate", "heartrate", "pulse", "pulse rate"), (
        _unit("beats a minute", ("bpm",), (40, 150), (20, 300)),), _PATIENT),
    _Limit("body temperature", _spelled("body temperature", "body temp", "core temperature"), (
        _unit("°C", _CELSIUS, (34, 41), (25, 45)),
        _unit("°F", _FAHRENHEIT, (93, 106), (77, 113))), _PATIENT),
    _Limit("wind speed", _spelled("wind speed", "windspeed", "wind gust", "gust speed", "ws"), (
        _unit("m/s", ("m s", "mps", "ms"), _UNBOUNDED, (0, 115)),
        _unit("km/h", ("km h", "kmh", "kph", "kmph"), _UNBOUNDED, (0, 415)),
        _unit("mph", ("mph",), _UNBOUNDED, (0, 260)),
        _unit("knots", ("knots", "kt", "kts", "kn"), _UNBOUNDED, (0, 225))), "at the Earth's surface",
        default="km/h"),
    _Limit("air temperature", _spelled(
        "air temperature", "air temp", "ambient temperature", "ambient temp", "outdoor temperature",
        "outside temperature"), (
        _unit("°C", _CELSIUS, _UNBOUNDED, (-90, 60)),
        _unit("°F", _FAHRENHEIT, _UNBOUNDED, (-130, 140))), "on Earth",
        default="°F"),
    _Limit("water temperature", _spelled("water temperature", "water temp", "watertemp"), (
        _unit("°C", _CELSIUS, (0, 32), (-3, 45)),
        _unit("°F", _FAHRENHEIT, (32, 90), (27, 113))), _SURFACE_WATER, panel="water"),
    _Limit("dissolved oxygen", _spelled("dissolved oxygen", "do"), (
        _unit("mg/L", ("mg l",), (1, 16), (0, 25)),), _SURFACE_WATER, panel="water"),
    _Limit("pH", _spelled("ph"), (_unit("", (), (4, 10), (2, 13)),), _SURFACE_WATER, panel="water"),
)
#: Readings that mark a table as one of a panel's kind without a limit of
#: their own: salinity or turbidity beside a pH says the pH is a lake's.
PANEL_COMPANIONS = {
    "water": _spelled("salinity", "turbidity", "conductivity", "secchi depth", "water depth"),
}
# Header words set aside before a name is matched: the sample a value comes
# from and words for "level"; units are set aside by the table above.
_LIMIT_FILLER = frozenset({
    "serum", "blood", "plasma", "level", "levels", "value", "values", "conc", "test", "max", "maximum",
    "min", "minimum", "mean", "average", "avg", "median", "standard", "units",
})
_LIMIT_UNIT_WORDS = frozenset(
    word
    for limit in PHYSICAL_LIMITS
    for unit in limit.units
    for spelling in unit.spellings
    for word in spelling
) | {"degree", "degrees", "deg", "degreec", "degreef", "per", "x10", "10", "12", "6", "l", "dl", "ml", "ul",
     "m", "ppt", "psu", "ntu"}

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
        """The columns as a list, or "(whole rows)" for an issue about rows as a whole."""
        return ", ".join(self.columns) or "(whole rows)"

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
    """A percentage for a sentence: "0.4%", "12%", and "under 0.1%" rather than "0.0%"."""
    if 0 < share < 0.1:
        return "under 0.1%"
    return f"{share:.1f}%" if share < 10 else f"{share:.0f}%"


def _name_words(column: str) -> set[str]:
    return set(re.findall(r"[^\W_]+", normalized_name(column)))


def _is_percentage_name(column: str, words: set[str]) -> bool:
    if words & SIGNED_WORDS:
        return False
    return (
        has_percent_unit(column)
        or bool(words & PERCENT_WORDS)
        or bool(_PERCENT_SUFFIX.search(column.strip()))
    )


def _header_range(column: str) -> tuple[float, float] | None:
    """A range written in the header, such as "(40 - 100)", as numbers, or None."""
    found = _HEADER_RANGE.search(column)
    if not found:
        return None
    low, high = float(found.group(1)), float(found.group(2))
    return (low, high) if low < high else None


def name_bounds(column: str) -> tuple[float, float, str] | None:
    """The range a column's name says its values must lie in, with the rule's wording.

    A range written in the header itself ("(40 - 100)") comes first. Returns
    None when the name implies no bound.
    """
    stated = _header_range(column)
    if stated is not None:
        low, high = stated
        return low, high, f"the header gives its valid range as {low:g} to {high:g}"
    words = _name_words(column)
    if words & STRETCH_WORDS:
        return 0.0, float("inf"), "a stretch can't be negative, though it may pass 100%"
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


@dataclass(frozen=True)
class _Bounds:
    """The range a column's values must lie in, and the rule that says so."""

    low: float
    high: float
    rule: str
    #: Set when a physical limit gave the range; its unit came from the header
    #: or the median, so values outside it are wrong however many there are.
    limit: _Limit | None = None
    #: The limit's critical range in the column's unit, with the quantity named
    #: in that unit ("a sodium level in mmol/L"), when it has one.
    critical: tuple[float, float] | None = None
    quantity: str = ""


def _core_words(column: str) -> tuple[frozenset, set[str]]:
    """A header's words with units and filler such as "serum" set aside, and its unit words."""
    words = _name_words(column)
    if "%" in column:
        words.add("percent")
    units = words & _LIMIT_UNIT_WORDS
    return frozenset(words - units - _LIMIT_FILLER), units


def _limit_named(column: str) -> tuple[_Limit, set[str]] | None:
    """The physical quantity a column's name gives, with the header's unit words."""
    core, units = _core_words(column)
    for limit in PHYSICAL_LIMITS:
        if core in limit.names:
            # "WS" is a wind speed only beside a speed unit.
            if core == {"ws"} and not units:
                return None
            return limit, units
    return None


def _limit_unit(limit: _Limit, units: set[str], median: float) -> tuple[_Unit, str] | None:
    """The unit a column is in, and how it was found: "stated", "median" or "default"."""
    named = [unit for unit in limit.units if any(spelling <= units for spelling in unit.spellings)]
    if named:
        usual = named[0].usual
        return (named[0], "stated") if usual[0] <= median <= usual[1] else None
    if units - {"percent"} and limit.panel:
        # A unit the table doesn't list (mg in a food table): leave the column alone.
        return None
    fitting = [
        unit for unit in limit.units if unit.usual != _UNBOUNDED and unit.usual[0] <= median <= unit.usual[1]
    ]
    if len(fitting) == 1:
        return fitting[0], "median"
    if limit.default and not fitting:
        return next(unit for unit in limit.units if unit.name == limit.default), "default"
    return None


def physical_limits(profiles: dict) -> dict[str, _Bounds]:
    """Limits from ``PHYSICAL_LIMITS`` for the columns whose name and values fit one.

    ``profiles`` maps column names to anything with sorted values in
    ``ordered``. A quantity that belongs to a panel counts only when another
    of the panel's quantities in the table fits its own unit too, or a
    companion reading (``PANEL_COMPANIONS``) is present, which tells a blood
    panel from a soil or a food table, and a lake's pH from a lab's.
    """
    found: dict[str, tuple[_Limit, _Unit, str]] = {}
    members: dict[str, int] = {}
    for column, profile in profiles.items():
        core, _ = _core_words(str(column))
        for panel, companions in PANEL_COMPANIONS.items():
            if core in companions:
                members[panel] = members.get(panel, 0) + 1
        named = _limit_named(str(column))
        if named is None or profile.ordered.size == 0:
            continue
        limit, units = named
        chosen = _limit_unit(limit, units, _sorted_median(profile.ordered))
        if chosen is not None:
            found[column] = (limit, *chosen)
            if limit.panel:
                members[limit.panel] = members.get(limit.panel, 0) + 1
    bounds = {}
    for column, (limit, unit, how) in found.items():
        if limit.panel and members.get(limit.panel, 0) < 2:
            continue
        low, high = unit.limits
        quantity = _article(limit.what)
        if how == "default":
            quantity += f" in any common unit ({unit.name} gives the widest range)"
        elif unit.name:
            quantity += f" in {unit.name}" + (" (the unit its median points to)" if how == "median" else "")
        if low == 0:
            rule = f"{quantity} stays below {show_value(high)} {limit.setting}"
        else:
            rule = f"{quantity} lies between {show_value(low)} and {show_value(high)} {limit.setting}"
        bounds[column] = _Bounds(float(low), float(high), rule, limit, unit.critical, quantity)
    return bounds


def _article(what: str) -> str:
    """ "a sodium level", "an air temperature": the quantity as a phrase."""
    analytes = (
        "sodium", "potassium", "chloride", "calcium", "glucose", "creatinine", "albumin", "haemoglobin",
        "haematocrit",
    )
    phrase = f"{what} level" if what in analytes else what
    if what == "pH":
        return "the pH"
    return f"{'an' if phrase[0] in 'aeiou' else 'a'} {phrase}"


def column_bounds(column: str, limits: dict[str, _Bounds] | None = None) -> _Bounds | None:
    """The tightest range a column must lie in, from its name and from any physical limit.

    A range written in the header is the column's own statement and wins.
    """
    named = name_bounds(column)
    limit = (limits or {}).get(column)
    if named is None:
        return limit
    low, high, wording = named
    if limit is None or _header_range(column) is not None:
        return _Bounds(low, high, wording)
    # A wind speed can't be negative by name, and can't pass 115 m/s by physics.
    return replace(limit, low=max(low, limit.low), high=min(high, limit.high))


def is_location_name(column: str) -> bool:
    """Whether a column holds a coordinate (latitude, easting), whose far values are distant sites."""
    return bool(_name_words(column) & LOCATION_WORDS)


def counts_units(column: str) -> bool:
    """Whether a column counts subjects or samples (a sample size, n, participants)."""
    words = _name_words(column)
    if words & SAMPLE_WORDS or {"sample", "size"} <= words or "n" in words:
        return True
    return bool(words & _COUNT_OF) and bool(words & {"people", "persons", "sites", "studies"})


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


#: Decile skewness the bulk of a column needs before a log scale is considered:
#: a gamma distribution of shape 10 has about this much, a bell curve none.
LOG_SKEW = 0.1


def prefers_log_scale(ordered: np.ndarray, column: str | None = None) -> bool:
    """Whether sorted values are better judged on a log scale of their positive part.

    True for essentially non-negative values whose bulk (the deciles, which a
    few outliers can't move) is clearly right-skewed and more symmetric on a
    log scale, as concentrations, latencies and counts are: a long but ordinary
    right tail then stays within reach. A symmetric reading keeps the linear
    scale, on which a low value is not exaggerated, and so does any column
    named as a temperature, a signed quantity, a direction, a coordinate or a
    pH (already a logarithm): its zero is not the bottom of a ratio scale, so
    a ratio of its values means nothing.
    """
    if column is not None and _name_words(column) & _LINEAR_WORDS:
        return False
    positive = ordered[ordered > 0]
    if positive.size < 10 or ordered[int(0.01 * ordered.size)] < 0:
        return False
    skew = _decile_skew(ordered)
    return skew >= LOG_SKEW and abs(_decile_skew(np.log10(positive))) < skew


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
    def of(cls, ordered: np.ndarray, column: str | None = None) -> _Scale:
        if prefers_log_scale(ordered, column):
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

    def upper_cluster(self, uniques: np.ndarray, counts: np.ndarray) -> np.ndarray:
        """True for distinct values above a wide gap in a log-scaled tail, when CLUSTER_VALUES lie beyond it.

        A long right tail runs on unbroken for a while, so ``detached`` asks a
        gap to leave its values twice as far out as the tail before it, which a
        value 180 times the median beside a tail reaching 9 times it misses. A
        gap of at least a factor of two and GAP_SPREADS spreads with ten or more
        values beyond it is no chance spacing of a tail, which thins out one
        value at a time.
        """
        cut = np.zeros(uniques.shape, dtype=bool)
        if not self.log or self.spread == 0 or uniques.size < 2:
            return cut
        scaled = np.log10(np.where(uniques > 0, uniques, np.nan))
        gaps = np.diff(scaled) >= max(GAP_SPREADS * self.spread, np.log10(2))
        far = scaled[:-1] > self.centre + EXTREME_Z * self.spread
        beyond = np.cumsum(counts[::-1])[::-1][1:]
        opening = np.flatnonzero(gaps & far & (beyond >= CLUSTER_VALUES))
        if opening.size:
            cut[opening[0] + 1 :] = True
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


def _rounded(value: float) -> str:
    """A computed figure to four significant digits: "1,306", "85.25"."""
    return show_value(float(f"{float(value):.4g}"))


def _range_of(values: pd.Series) -> str:
    low, high = float(values.min()), float(values.max())
    return show_value(low) if low == high else f"{show_value(low)} to {show_value(high)}"


def _impossible(
    column: _Column, coded: np.ndarray, bounds: _Bounds | None
) -> tuple[QualityIssue | None, np.ndarray]:
    """Values outside what the name or a physical limit allows, or sign slips in a
    column otherwise never negative.

    Cells already read as a missing-value code (``coded``) are left to that check.
    """
    nothing = np.zeros(column.size, dtype=bool)
    if bounds is not None:
        low, high, wording = bounds.low, bounds.high, bounds.rule
        outside = ((column.array < low) | (column.array > high)) & ~coded
        # Mostly outside a name's bound means another scale than the name
        # suggests. A physical limit's unit was read from the values, so its
        # misses are errors however many there are.
        if not outside.any() or (bounds.limit is None and outside.mean() > 0.2):
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


def _mistyped(value: float, others: np.ndarray) -> float | None:
    """The reading a repeated-digit value would be with one digit fewer, when that is an ordinary value.

    1,111 among heart rates up to 135 is as likely 111 typed with an extra 1
    as a code; 8,888 among ages up to 90 is a code.
    """
    if value <= 0 or value not in _REPEATED_DIGITS or others.size == 0:
        return None
    shorter = float(str(int(value))[:-1])
    return shorter if float(others.min()) <= shorter <= float(others.max()) else None


def _sentinels(column: _Column, ordered: np.ndarray) -> list[QualityIssue]:
    """Code values standing alone far from the rest (-999, 999.9, -100000).

    A repeated digit that would be an ordinary reading with one digit fewer is
    called a likely entry error rather than a code.
    """
    issues = []
    standing, centre = _standing_codes(column, ordered)
    others = ordered[~np.isin(ordered, standing)]
    for value in standing:
        flagged = column.cells(column.array == value)
        rows = _plural(flagged.size, "row")
        shorter = _mistyped(value, others)
        if shorter is not None:
            detail = (
                f"{column.name} holds {show_value(value)} in {rows}, far beyond its other values "
                f"(median {show_value(centre)}, highest {show_value(float(others.max()))}). With one "
                f"digit fewer it would be an ordinary {show_value(shorter)}, so it is a likely entry "
                "error, or else a code for a missing reading; either way it is not a measurement."
            )
            summary = f"{column.name}: {show_value(value)} in {rows}, a likely entry error"
        else:
            detail = (
                f"{column.name} holds {show_value(value)} in {rows}, far from its other values "
                f"(median {show_value(centre)}): a code for a missing reading, not a measurement."
            )
            summary = f"{column.name}: {show_value(value)} in {rows}, a missing-value code"
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


def _edge_codes(column: _Column, taken: np.ndarray) -> list[QualityIssue]:
    """A value repeated at the very bottom or top of a column, standing apart from every other value.

    A cap sits against the readings it limits, which crowd up to it; a
    converted blank (0 °F becoming -17.78 °C, below the coldest real reading)
    sits beyond a thin tail. The value must hold EDGE_CODE_VALUES rows or more
    but under a quarter of the column, be SPIKE_RATIO times as common as any
    of the five nearest values, and lie further from the nearest of them than
    a quarter of the other values' interquartile range; and the other values
    within that distance of the nearest must be a quarter as dense, or less,
    as they are between their quartiles. Zero is left out: an instrument that
    is off reads 0 apart from its running values, and that is a reading.
    """
    if column.uniques.size < 10 or column.size < 50:
        return []
    issues = []
    for position, inner in ((0, slice(1, 6)), (-1, slice(-6, -1))):
        value = float(column.uniques[position])
        count = int(column.counts[position])
        if value == 0 or count < EDGE_CODE_VALUES or count >= 0.25 * column.size:
            continue
        if count < SPIKE_RATIO * int(column.counts[inner].max()):
            continue
        rest = column.ordered[column.ordered != value]
        first, third = (rest[int(round(share * (rest.size - 1)))] for share in (0.25, 0.75))
        neighbour = float(column.uniques[1 if position == 0 else -2])
        reach = (third - first) / 4
        if third <= first or abs(neighbour - value) <= reach:
            continue
        low, high = (neighbour, neighbour + reach) if position == 0 else (neighbour - reach, neighbour)
        beside = int(np.searchsorted(rest, high, side="right") - np.searchsorted(rest, low))
        if beside > 0.25 * (0.5 * rest.size / (third - first)) * reach:
            continue
        cells = (column.array == value) & ~taken
        if not cells.any():
            continue
        flagged = column.cells(cells)
        rows = _plural(flagged.size, "row")
        end = "bottom" if position == 0 else "top"
        reason = "a code for a missing reading, not a measurement"
        if abs(value - _ZERO_FAHRENHEIT_C) < 0.01 and _name_words(column.name) & {"temp", "temperature"}:
            reason = "it is 0 °F in °C, so probably blank Fahrenheit readings converted, not measurements"
        detail = (
            f"{column.name} holds {show_value(value)} in {rows}, at the very {end} of the column and apart "
            f"from every other value (the nearest is {show_value(neighbour)}): {reason}."
        )
        summary = f"{column.name}: {show_value(value)} in {rows}, standing apart, likely a missing-value code"
        rule = (
            f"a value at the minimum or maximum in {EDGE_CODE_VALUES} or more rows, {SPIKE_RATIO}× its "
            "nearest values and further from them than a quarter of the rest's IQR"
        )
        issues.append(_issue("sentinel", column, flagged, detail, summary, rule, group=value))
    return issues


def _round_value(value: float) -> bool:
    """Whether a value is round as limits are: one significant figure (80, 500, 0.001), or 25, 150, 7.5."""
    if value == 0 or not np.isfinite(value):
        return False
    leading = abs(value) / 10.0 ** np.floor(np.log10(abs(value)))
    return bool(np.isclose(leading, np.round(leading)) or np.isclose(leading % 1, 0.5))


def _caps(
    column: _Column,
    fill: float | None = None,
    tied: Callable[[str, float], bool] | None = None,
    shared: frozenset = frozenset(),
) -> list[QualityIssue]:
    """One value piled up at the maximum or minimum: an instrument limit, a time-out, a top code.

    The value must stand out from every value inside the range, not only from
    its neighbours, unless it holds CAP_CLEAR_SHARE of the column, or is a round
    maximum (500 on an index whose scale stops there): 594 kg/m3 of sand at the
    bottom of a range in which 755.8 is as common is a recipe, not a floor. A
    column of fewer than five values is checked only for one value holding
    most of it (a reporting limit, 0.001 on 94% of rows). A pile of a similar
    small share at both ends is one column clipped at both, reported as such.
    A count of subjects or samples is a design number, and a value that
    belongs to one category (every herbivore at trophic level 2) is that
    category's value; neither is a limit. ``fill`` is a filled-in value left
    out of the comparison, and ``shared`` holds limits reported across
    columns at once (see ``_shared_clips``).
    """
    if column.uniques.size < 2 or counts_units(column.name):
        return []
    few = column.uniques.size < 5
    inside = column.counts[1:-1]
    if fill is not None:
        inside = inside[column.uniques[1:-1] != fill]
    busiest_inside = int(inside.max()) if inside.size else 0
    # The three nearest distinct values on each side: a limit piles up on one
    # value, while small counts thin out smoothly from their minimum.
    nearest = {"maximum": column.counts[-4:-1], "minimum": column.counts[1:4]}
    piles: dict[str, tuple[float, int, bool]] = {}
    for side, position in (("maximum", -1), ("minimum", 0)):
        value = float(column.uniques[position])
        count = int(column.counts[position])
        neighbours = int(nearest[side].max())
        if value == 0 or count < 3 or count < 2 * neighbours or value in shared:
            continue
        # Counts start at 0 or 1, so piling up there is their shape, not a floor.
        if side == "minimum" and value == 1 and column.whole:
            continue
        if few and count / column.size < NEAR_CONSTANT_SHARE:
            continue
        if tied is not None and tied(column.name, value):
            continue
        # A round maximum hit a handful of times is a scale's ceiling (an index stopping at 500);
        # a large pile there is judged like any other.
        ceiling = side == "maximum" and abs(value) >= 1 and _round_value(value)
        ceiling = ceiling and count / column.size < CAP_SHARE
        clear = count >= CAP_STANDOUT * busiest_inside or count / column.size >= CAP_CLEAR_SHARE or ceiling
        # A share no continuous reading reaches by chance, or enough rows to stand out.
        enough = count / column.size >= CAP_SHARE or count >= 5
        piles[side] = (value, count, count >= 3 * neighbours and clear and enough)
    if len(piles) == 2 and _clipped_both(piles, column.size):
        return [_clipped_issue(column, piles["minimum"][0], piles["maximum"][0])]
    # Many values just under a maximum that only a few hit exactly are squeezed against it.
    squeezed = _squeezed(column)
    issues = [squeezed] if squeezed else []
    for side, (value, count, limit) in piles.items():
        if not limit or (side == "maximum" and squeezed):
            continue
        flagged = column.cells(column.array == value)
        kind, bias = ("cap", "down") if side == "maximum" else ("floor", "up")
        share = _percent(count / column.size * 100)
        if few or count / column.size >= ALMOST_CONSTANT_SHARE:
            detail = (
                f"{column.name} sits exactly at its {side}, {show_value(value)}, in {count:,} of its "
                f"{column.size:,} values ({share}): probably a reporting limit, so the column carries "
                "almost no information, and true values there may lie beyond it."
            )
        else:
            detail = (
                f"{column.name} sits exactly at its {side}, {show_value(value)}, in "
                f"{_plural(count, 'row')} ({share}), which looks like a {kind} (a limit of the "
                "instrument or the protocol); true values there may lie beyond it, which biases the "
                f"mean {bias}."
            )
        summary = f"{column.name}: {_plural(count, 'row')} ({share}) piled at its {side}, {show_value(value)}"
        rule = (
            f"one value at the {side} in {CAP_SHARE:.0%} or more of rows, or the most frequent value, "
            "or a round maximum"
        )
        issues.append(_issue("cap", column, flagged, detail, summary, rule, group=(side, value)))
    return issues


def _clipped_both(piles: dict[str, tuple[float, int, bool]], size: int) -> bool:
    """Whether piles at both ends are one clipping: each a small share, the two alike."""
    shares = [count / size for _, count, _ in piles.values()]
    counts = [count for _, count, _ in piles.values()]
    return (
        min(counts) >= 5
        and all(CLIP_SHARES[0] <= share <= CLIP_SHARES[1] for share in shares)
        and max(shares) <= CLIP_LIKENESS * min(shares)
    )


def _clipped_issue(column: _Column, low: float, high: float) -> QualityIssue:
    """A column piled at both its minimum and maximum in a similar small share: winsorised."""
    at_low, at_high = column.array == low, column.array == high
    flagged = column.cells(at_low | at_high)
    shares = f"{_percent(at_low.mean() * 100)} and {_percent(at_high.mean() * 100)}"
    lows, highs = _plural(int(at_low.sum()), "row"), _plural(int(at_high.sum()), "row")
    detail = (
        f"{column.name} sits exactly at its minimum, {show_value(low)}, in {lows} and at its maximum, "
        f"{show_value(high)}, in {highs} ({shares}): it looks clipped at both ends (winsorised), so true "
        "values lie beyond both limits."
    )
    summary = f"{column.name}: clipped at {show_value(low)} and {show_value(high)} ({shares} of rows)"
    rule = (
        f"piles at both the minimum and the maximum, each {CLIP_SHARES[0]:.1%} to {CLIP_SHARES[1]:.0%} of "
        f"rows and at least twice its neighbours, the two shares within a factor of {CLIP_LIKENESS:g}"
    )
    return _issue("clipped", column, flagged, detail, summary, rule, group=(low, high))


def _squeezed(column: _Column) -> QualityIssue | None:
    """Values bunched just under a round maximum without piling on it exactly.

    Readings squeezed against a limit (62 values from 79 to 80, only 5 of them
    exactly 80) crowd the top stretch of the range: the values within 1% of
    the range below the maximum must be SQUEEZE_RATIO times as many as in a
    typical stretch of that width below, and CAP_SHARE of the column.
    """
    if column.size < 100 or column.uniques.size < 20:
        return None
    high = float(column.uniques[-1])
    if high <= 0 or not _round_value(high):
        return None
    width = 0.01 * (high - column.quantile(0.01))
    if width <= 0:
        return None
    edges = high - width * np.arange(SQUEEZE_WINDOWS + 1, -1, -1)
    counts = np.histogram(column.ordered, bins=edges)[0]
    top, below = int(counts[-1]), float(np.median(counts[:-1]))
    exact = int(column.counts[-1])
    # Most of them must fall short of the maximum; a pile on it is a cap or nothing.
    if top / column.size < CAP_SHARE or top < SQUEEZE_RATIO * max(below, 1.0) or top < 2 * exact:
        return None
    cells = column.array >= high - width
    flagged = column.cells(cells)
    share = _percent(top / column.size * 100)
    detail = (
        f"{column.name} has {_plural(top, 'value')} ({share}) within {show_value(float(f'{width:.2g}'))} "
        f"of its maximum, {show_value(high)}, about {top / max(below, 1.0):.0f}× as many as in a stretch of "
        f"that width lower down, though only {exact:,} equal it exactly: values seem squeezed against a "
        f"limit at {show_value(high)}, so true values there may lie beyond it."
    )
    summary = f"{column.name}: {_plural(top, 'value')} ({share}) bunched just under {show_value(high)}"
    rule = (
        f"values within 1% of the range below a round maximum, {SQUEEZE_RATIO}× a typical stretch of that "
        f"width and {CAP_SHARE:.0%} or more of the column"
    )
    return _issue("cap", column, flagged, detail, summary, rule, group=("near maximum", high))


def _shared_clips(
    profiles: dict[str, _Column], columns: list[str]
) -> tuple[dict[str, frozenset], QualityIssue | None]:
    """Limits that many columns share as their exact maximum or minimum: values clipped together.

    Expression values capped at -2 and 2 by a platform put both numbers at the
    ends of hundreds of columns, a few cells each, so no single column piles up
    enough to look capped and its clipped values look extreme. A value that is
    the maximum (or minimum) of SHARED_CLIP_COLUMNS columns or more, and of
    SHARED_CLIP_SHARE of the readings, is reported once for them all, and each
    column's other checks leave it out. Zero, and a limit the names give (100
    for percentages), are where such columns end anyway.

    Returns the shared limits by column and the one issue, or None.
    """
    needed = max(SHARED_CLIP_COLUMNS, SHARED_CLIP_SHARE * len(columns))
    if len(columns) < needed:
        return {}, None
    limits: dict[str, set[float]] = {}
    for position in (-1, 0):
        ends = pd.Series({column: float(profiles[column].ordered[position]) for column in columns})
        counts = ends.value_counts()
        for value in counts.index[counts >= needed]:
            if value == 0:
                continue
            for column in ends.index[ends == value]:
                bound = name_bounds(column)
                if bound is None or value not in bound[:2]:
                    limits.setdefault(column, set()).add(float(value))
    if len(limits) < needed:
        return {}, None
    named = list(limits)
    values = sorted({value for found in limits.values() for value in found})
    cells = {column: profiles[column].index[np.isin(profiles[column].array, list(found))]
             for column, found in limits.items()}
    total = sum(len(rows) for rows in cells.values())
    shown = " and ".join(show_value(value) for value in values)
    detail = (
        f"{len(named):,} columns ({_listed(named)}) end exactly at {shown}, {total:,} cells in all: "
        f"the values look clipped at {shown} (by the instrument, the platform or the export), so true "
        "values there lie beyond them. The clipped cells are left out of each column's extreme-value check."
    )
    summary = f"{len(named):,} columns clipped at {shown} ({total:,} cells)"
    rule = (
        f"one value the exact maximum or minimum of {SHARED_CLIP_COLUMNS} or more columns and of "
        f"{SHARED_CLIP_SHARE:.0%} or more of the readings"
    )
    share = max(len(rows) / max(profiles[column].size, 1) * 100 for column, rows in cells.items())
    issue = QualityIssue("cap", tuple(named), total, share, detail, summary, rule,
                         tuple(show_value(value) for value in values), group=("shared", tuple(values)),
                         cells=cells)
    return {column: frozenset(found) for column, found in limits.items()}, issue


def _group_caps(
    frame: pd.DataFrame,
    profiles: dict[str, _Column],
    columns: list[str],
    groupings: dict[str, tuple[np.ndarray, pd.Index]],
) -> list[QualityIssue]:
    """A round value inside a column's range, piled up in one group: a limit that group's tests stop at.

    Compression tests stopped at 50% strain leave Elongation = 50 on a tenth
    of the rows, nearly all of them compression tests, though tension tests
    run past it. The value must be round, hold GROUP_CAP_SHARE of the column
    and three times as many rows as any other value, and GROUP_CAP_CONCENTRATION
    of its rows must fall in one level of a grouping that holds far less of
    the column. ``groupings`` maps grouping columns to their level codes and levels.
    """
    issues = []
    for column in columns:
        profile = profiles[column]
        if profile.uniques.size < 10 or profile.size < 50:
            continue
        order = np.argsort(profile.counts)
        position, runner_up = int(order[-1]), int(profile.counts[order[-2]])
        value, count = float(profile.uniques[position]), int(profile.counts[position])
        if (
            count / profile.size < GROUP_CAP_SHARE or count < 3 * runner_up or not _round_value(value)
            or position in (0, profile.uniques.size - 1)
        ):
            continue
        at = frame[column].to_numpy(dtype="float64", na_value=np.nan) == value
        present = frame[column].notna().to_numpy()
        for grouping, (codes, names) in groupings.items():
            if grouping == column:
                continue
            levels = np.bincount(codes[at & (codes >= 0)], minlength=1)
            top = int(np.argmax(levels))
            within = levels[top] / count
            base = float(((codes == top) & present).sum()) / max(int(present.sum()), 1)
            if within < GROUP_CAP_CONCENTRATION or within < 1.5 * base:
                continue
            label = group_label(grouping, names[top])
            flagged = profile.cells(profile.array == value)
            share = _percent(count / profile.size * 100)
            detail = (
                f"{column} holds exactly {show_value(value)} in {_plural(count, 'row')} ({share}), "
                f"{int(levels[top]):,} of them in {label}, and at least three times as often as any other "
                f"value: probably a limit at which those measurements stopped, so true values there may lie "
                "beyond it."
            )
            summary = f"{column}: {show_value(value)} in {_plural(count, 'row')} ({share}), mostly {label}"
            rule = (
                f"a round value inside the range in {GROUP_CAP_SHARE:.0%} or more of rows and 3× any other "
                f"value, with {GROUP_CAP_CONCENTRATION:.0%} or more of them in one group"
            )
            group = ("within", value, label)
            issues.append(_issue("cap", profile, flagged, detail, summary, rule, group=group))
            break
    return issues


def _simple_fraction(value: float, places: int) -> bool:
    """Whether a value written to ``places`` decimals is a fraction k/n with a small n.

    0.0166667 is 1/60, the mean of sixty counts of which one is 1: common in
    a table of counts and never a fill value. Only as many denominators are
    tried as keep the chance of a match by accident under one in a thousand,
    so a fill value written to four decimals, such as 46.1526, is never
    "explained" by some n. A value at full precision (8 or more decimals) must
    match to within float rounding.
    """
    if places >= 8:
        largest, tolerance = 1000, 1e-9
    else:
        largest, tolerance = int(np.sqrt(2e-3 * 10.0**places)), 0.5 * 10.0**-places
    if largest < 2:
        return False
    denominators = np.arange(2, largest + 1)
    scaled = abs(value) * denominators
    return bool(np.any(np.abs(scaled - np.round(scaled)) <= tolerance * denominators))


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
    if _simple_fraction(value, int(places[position])):
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


def _times_beyond(flagged: np.ndarray, rest: np.ndarray) -> np.ndarray:
    """How many times beyond the nearest ordinary value each flagged value lies.

    3,715 among haematocrits up to 81 is 46 times beyond; a vanadium content
    of 4% beside a next-highest of 2% is twice. A value across zero from the
    rest counts as infinitely far.
    """
    if rest.size == 0:
        return np.full(flagged.shape, np.inf)
    high, low = float(rest.max()), float(rest.min())
    nearest = np.where(flagged > high, high, low)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.maximum(flagged / nearest, nearest / flagged)
    return np.where((nearest == 0) | ((flagged > 0) != (nearest > 0)), np.inf, ratio)


def _extremes(column: _Column, scale: _Scale, taken: np.ndarray) -> QualityIssue | None:
    """Values more than EXTREME_Z robust standard deviations from the median, cut off
    from the rest of the column by a gap.

    Up to EXTREME_SHARE of the column, or fewer than CLUSTER_VALUES of them,
    they are listed as extreme values; up to CLUSTER_SHARE, as a second
    cluster with how far it moves the mean. The
    wording calls them likely errors only when they lie ERROR_RATIO times or
    more beyond every ordinary value.
    """
    if column.size < 12 or column.uniques.size < 5:
        return None
    distant = scale.far(column.uniques)
    if not distant.any():
        return None
    apart = scale.detached(column.uniques) | scale.upper_cluster(column.uniques, column.counts)
    candidates = column.uniques[distant & apart]
    if not candidates.size:
        return None
    far = np.isin(column.array, candidates) & ~taken
    if not far.any() or far.mean() > CLUSTER_SHARE:
        return None
    flagged = column.cells(far)
    rest = column.array[~np.isin(column.array, candidates)]
    on_log = " on a log scale" if scale.log else ""
    rule = (
        f"robust z-score (median and scaled MAD{on_log}) above {EXTREME_Z:.0f}, cut off from the "
        f"rest by a gap of {GAP_SPREADS:.0f} or more"
    )
    if far.mean() > EXTREME_SHARE and far.sum() >= CLUSTER_VALUES:
        share = _percent(far.mean() * 100)
        side = "above" if float(flagged.min()) > scale.median else "below"
        detail = (
            f"{column.name} has {_plural(flagged.size, 'value')} ({share}) from {_range_of(flagged)}, "
            f"cut off {side} the rest (median {show_value(scale.median)}{on_log}): a second cluster, "
            f"or a possible sensor or unit fault. They move its mean to {_rounded(column.array.mean())}, "
            f"against {_rounded(rest.mean())} without them."
        )
        summary = (
            f"{column.name}: {_plural(flagged.size, 'value')} ({share}) in a second cluster "
            f"({_range_of(flagged)})"
        )
        rule += f", in more than {EXTREME_SHARE:.0%} of rows"
        return _issue("cluster", column, flagged, detail, summary, rule)
    one = flagged.size == 1
    beyond = _times_beyond(flagged.to_numpy(), rest)
    if beyond.min() >= ERROR_RATIO:
        verdict = (
            f"{'It is' if one else 'They are'} many times beyond every other value, as entry or unit "
            f"errors often are, and {'it moves' if one else 'they move'} means and correlations."
        )
    elif beyond.max() >= ERROR_RATIO:
        farthest = float(flagged.to_numpy()[np.argmax(beyond)])
        verdict = (
            f"The farthest, {show_value(farthest)}, is many times beyond every other value, as entry or "
            "unit errors often are; check that the others are real, since they move means and correlations."
        )
    else:
        verdict = (
            f"Check that {'it is' if one else 'they are'} real: {'it moves' if one else 'they move'} "
            "means and correlations."
        )
    detail = (
        f"{column.name} has {_plural(flagged.size, 'value')} more than {EXTREME_Z:.0f} robust "
        f"standard deviations from its median of {show_value(scale.median)}{on_log} "
        f"({_range_of(flagged)}), cut off from the other values. {verdict}"
    )
    summary = f"{column.name}: {_plural(flagged.size, 'extreme value')} ({_range_of(flagged)})"
    return _issue("extreme", column, flagged, detail, summary, rule)


def _survey_codes(column: _Column) -> QualityIssue | None:
    """Codebook values (7, 9, 77, 99) sitting above a small set of answer codes.

    A lone 7 or 8 counts only above an unbroken scale of answers (1 to 5): in a
    clinical code set such as histological types 1, 2, 4, 5 and 7, a 7 is one
    more type, not a refusal.
    """
    if not column.whole:
        return None
    levels = [int(value) for value in column.uniques]
    answers = [level for level in levels if level not in SURVEY_CODES]
    scale = bool(answers) and answers == list(range(answers[0], answers[-1] + 1))
    codes = [level for level in levels if level in SURVEY_CODES and (level > 8 or scale)]
    answers = [level for level in levels if level not in codes]
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


def _rare_zero_code(column: _Column) -> QualityIssue | None:
    """A rare 0 below whole-number codes that run unbroken from 1.

    A code set that starts at 1 (thal 1 to 3, a five-point scale) has no use
    for 0, so a 0 in a handful of rows is more likely a missing value written
    as a number than a category. It is named for checking and left in place.
    """
    if not column.whole or column.uniques.size < 4 or column.size < 50 or column.uniques[0] != 0:
        return None
    others = column.uniques[1:]
    if not np.array_equal(others, np.arange(1, others.size + 1)):
        return None
    count = int(column.counts[0])
    if count / column.size >= RARE_CODE_SHARE or count * RARE_CODE_RATIO > int(column.counts[1:].min()):
        return None
    flagged = column.cells(column.array == 0)
    rows = _plural(count, "row")
    detail = (
        f"{column.name} holds 0 in {rows} ({_percent(count / column.size * 100)}), below its other "
        f"codes (1 to {show_value(others[-1])}), each of which is far more common: a code set that "
        "starts at 1 has no use for 0, so check whether it stands for a missing value."
    )
    summary = f"{column.name}: 0 in {rows}, below codes running from 1, possibly a missing value"
    rule = (
        f"a 0 in under {RARE_CODE_SHARE:.0%} of rows, at most 1/{RARE_CODE_RATIO} as common as any other "
        "code, below whole-number codes running unbroken from 1"
    )
    return _issue("rare_code", column, flagged, detail, summary, rule, group=0.0)


def _implausible(column: _Column, taken: np.ndarray, bounds: _Bounds | None) -> QualityIssue | None:
    """Many values beyond a quantity's critical limits, though each lies within what is possible.

    A sodium of 165 can happen; a fifth of a table's patients above 160 can't,
    since critical values turn up in a few percent of even very ill groups.
    Cells another check took (``taken``) are left out.
    """
    if bounds is None or bounds.critical is None or column.size < 50:
        return None
    low, high = bounds.critical
    beyond = ((column.array < low) | (column.array > high)) & ~taken
    count = int(beyond.sum())
    if count < 10 or count / column.size < CRITICAL_SHARE:
        return None
    flagged = column.cells(beyond)
    if flagged.min() > high:
        what = f"{_plural(count, 'value')} above {show_value(high)}"
    elif flagged.max() < low:
        what = f"{_plural(count, 'value')} below {show_value(low)}"
    else:
        what = f"{_plural(count, 'value')} outside {show_value(low)} to {show_value(high)}"
    share = _percent(count / column.size * 100)
    detail = (
        f"{column.name} has {what} ({share} of its values), beyond the critical limits of "
        f"{bounds.quantity}. Each is possible in a very ill patient, but they turn up in a few percent of "
        "even the sickest groups, so this many points to errors, another unit or a mislabelled column."
    )
    summary = f"{column.name}: {what} ({share}), beyond critical limits"
    rule = f"{CRITICAL_SHARE:.0%} or more of a column beyond a quantity's critical limits ({low:g}, {high:g})"
    return _issue("implausible", column, flagged, detail, summary, rule)


def _is_coded(column: _Column, factors: tuple[str, ...]) -> bool:
    """Whether a column holds category codes: a named factor, or a few whole numbers.

    A name that bounds the values (a humidity, a percentage) marks a reading even
    when it is recorded in whole numbers; a range written in the header ("Score
    (1-5)") does not, since a scale of codes states one too.
    """
    if column.name in factors:
        return True
    few = column.uniques.size <= CODED_LEVELS and column.whole
    return few and (name_bounds(column.name) is None or _header_range(column.name) is not None)


def _value_issues(
    column: _Column,
    factors: tuple[str, ...],
    bounds: _Bounds | None = None,
    tied: Callable[[str, float], bool] | None = None,
    single_cells: bool = True,
    shared: frozenset = frozenset(),
) -> list[QualityIssue]:
    """The checks on one column's values. ``single_cells`` is False for the
    columns of a matrix table, whose far values are judged by whole rows.
    ``shared`` holds the column's limits that ``_shared_clips`` reports across
    columns."""
    if column.size == 0:
        return []
    if _is_coded(column, factors):
        stated = column_bounds(column.name) if _header_range(column.name) else None
        outside = _impossible(column, np.zeros(column.size, dtype=bool), stated)[0] if stated else None
        return [found for found in (outside, _survey_codes(column), _rare_zero_code(column)) if found]
    imputed, filled = _imputed(column)
    caps = _caps(column, imputed.group if imputed else None, tied, shared)
    # A fill value or a pile-up at a limit would sit in the bulk and shrink the
    # spread, so the scale comes from the other values.
    piled = [*_pile_values(caps), *shared] + ([imputed.group] if imputed else [])
    if piled:
        filled = filled | np.isin(column.array, piled)
    ordered = column.ordered[~np.isin(column.ordered, piled)] if piled else column.ordered
    scale = _Scale.of(ordered, column.name)
    # A code stands for a missing reading before it is a negative or out-of-bounds value.
    nothing = np.zeros(column.size, dtype=bool)
    codes = _sentinels(column, ordered) + _spikes(column, nothing)
    coded = np.zeros(column.size, dtype=bool)
    for found in codes:
        coded |= column.index.isin(found.cells[column.name])
    if bounds is None:
        bounds = column_bounds(column.name)
    # A value no reading can take is impossible, whether or not it stands apart.
    edges = [
        found for found in _edge_codes(column, coded)
        if bounds is None or bounds.low <= found.group <= bounds.high
    ]
    for found in edges:
        coded |= column.index.isin(found.cells[column.name])
    codes += edges
    # A code piled at the minimum (-999 in 45% of rows) is a code, not a floor.
    caps = [issue for issue in caps if not set(_pile_values([issue])) & {found.group for found in codes}]
    impossible, taken = _impossible(column, coded, bounds)
    issues: list[QualityIssue] = [impossible] if impossible else []
    taken |= coded
    implausible = _implausible(column, taken, bounds)
    if implausible:
        issues.append(implausible)
    issues.extend(codes)
    issues.extend(caps)
    if imputed:
        issues.append(imputed)
    # A direction wraps around (359° sits beside 0°), so distance from its median means nothing.
    circular = bool(_name_words(column.name) & DIRECTION_WORDS)
    extreme = _extremes(column, scale, taken | filled) if single_cells and not circular else None
    if extreme:
        issues.append(extreme)
    return issues


def _pile_values(issues: list[QualityIssue]) -> list[float]:
    """The values cap and clipping issues pile at: a floor, a cap, or both ends of a clipped column.

    Values bunched under a maximum without piling on it give none.
    """
    values: list[float] = []
    for issue in issues:
        if issue.check == "clipped" and issue.group:
            values.extend(issue.group)
        elif issue.check == "cap" and issue.group[0] in ("maximum", "minimum"):
            values.append(issue.group[1])
    return values


def _merge_alike(issues: list[QualityIssue]) -> list[QualityIssue]:
    """Fold the same code, cap or fill pattern across columns into one issue.

    "MaxTemp, MinTemp and MeanTemp hold -17.78" reads better than three issues,
    and a fill value, or extreme values, recurring on the same rows in several
    columns are one finding: one bad record, or one batch of them. Columns
    clipped at both ends are one finding whatever their limits, since the same
    treatment was applied to each, and so are extreme values in more than
    EXTREME_COLUMNS_LISTED columns, as a wide panel of one kind gives.
    """
    merged: list[QualityIssue] = []
    groups: dict[tuple, list[QualityIssue]] = {}
    for issue in issues:
        if issue.check in ("sentinel", "cap"):
            groups.setdefault((issue.check, issue.group), []).append(issue)
        elif issue.check in ("imputed", "extreme", "cluster"):
            rows = issue.cells[issue.columns[0]]
            groups.setdefault((issue.check, len(rows), rows[0], rows[-1]), []).append(issue)
        elif issue.check == "clipped":
            groups.setdefault(("clipped",), []).append(issue)
        else:
            merged.append(issue)
    for (check, *_), alike in groups.items():
        merged.append(alike[0] if len(alike) == 1 else _combined(check, alike))
    extremes = [issue for issue in merged if issue.check == "extreme"]
    if len({column for issue in extremes for column in issue.columns}) > EXTREME_COLUMNS_LISTED:
        merged = [issue for issue in merged if issue.check != "extreme"] + [_many_extremes(extremes)]
    return merged


def _many_extremes(issues: list[QualityIssue]) -> QualityIssue:
    """Extreme values in many columns as one issue, as a wide panel of one kind gives."""
    columns = tuple(dict.fromkeys(column for issue in issues for column in issue.columns))
    cells_by_column = {name: index for issue in issues for name, index in issue.cells.items()}
    counts = sorted(len(index) for index in cells_by_column.values())
    cells = sum(counts)
    spread = f"{counts[0]:,}" if counts[0] == counts[-1] else f"{counts[0]:,} to {counts[-1]:,}"
    detail = (
        f"{len(columns):,} columns ({_listed(list(columns))}) each have values more than "
        f"{EXTREME_Z:.0f} robust standard deviations from their median, cut off from their other values: "
        f"{cells:,} cells in all, {spread} per column. With this many columns affected, the values are "
        "as likely the shape of those columns as entry errors; check them before relying on a finding "
        "about one of these columns."
    )
    summary = f"{len(columns):,} columns: {cells:,} extreme values ({spread} each)"
    return QualityIssue(
        check="extreme",
        columns=columns,
        rows=max(issue.rows for issue in issues),
        share=max(issue.share for issue in issues),
        detail=detail,
        summary=summary,
        rule=issues[0].rule,
        examples=issues[0].examples,
        cells=cells_by_column,
    )


def _combined(check: str, alike: list[QualityIssue]) -> QualityIssue:
    columns = tuple(issue.columns[0] for issue in alike)
    named = _listed(list(columns))
    first = alike[0]
    cells = sum(issue.rows for issue in alike)
    rows, share = cells, max(issue.share for issue in alike)
    if check == "extreme":
        rows, share = first.rows, first.share
        detail = (
            f"The columns {named} have values more than {EXTREME_Z:.0f} robust standard deviations "
            f"from their medians on the same {_plural(rows, 'row')}, cut off from the other values; "
            "one record or one batch is probably wrong, or measured on another scale, across all of them."
        )
        summary = f"Extreme values on the same {_plural(rows, 'row')} in columns {named}"
    elif check == "cluster":
        rows, share = first.rows, first.share
        detail = (
            f"The columns {named} have a second cluster of values on the same {_plural(rows, 'row')} "
            f"({_percent(share)}), cut off from their other values: one batch of records measured "
            "differently, or a sensor or unit fault across all of them."
        )
        summary = (
            f"A second cluster on the same {_plural(rows, 'row')} ({_percent(share)}) in columns {named}"
        )
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
    elif check == "clipped":
        shares = sorted(issue.share / 2 for issue in alike)
        spread = _percent(shares[0]) if _percent(shares[0]) == _percent(shares[-1]) else (
            f"{_percent(shares[0])} to {_percent(shares[-1])}"
        )
        rows, share = max(issue.rows for issue in alike), max(issue.share for issue in alike)
        detail = (
            f"{len(columns):,} columns ({named}) each sit exactly at their own minimum and maximum in about "
            f"{spread} of rows at each end, {cells:,} cells in all: they look clipped at both ends "
            "(winsorised), so true values lie beyond both limits in every one of them."
        )
        summary = f"{len(columns):,} columns clipped at both ends ({_listed(list(columns), 3)})"
    elif check == "cap":
        side, value = first.group
        kind = "cap" if side == "maximum" else "floor"
        detail = (
            f"{named} sit exactly at their {side}, {show_value(value)}, in {_plural(cells, 'value')} "
            f"in all, which looks like a shared {kind}; true values there may lie beyond it."
        )
        summary = f"{named}: {_plural(cells, 'value')} piled at their {side}, {show_value(value)}"
        if side == "near maximum":
            detail = (
                f"{named} bunch just under their maximum, {show_value(value)}, in {_plural(cells, 'value')} "
                "in all, as values squeezed against a shared limit do; true values there may lie beyond it."
            )
            summary = f"{named}: {_plural(cells, 'value')} bunched just under {show_value(value)}"
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
        # Columns clipped at both ends each have their own limits, so no one value stands for them.
        group=None if check == "clipped" else first.group,
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
    if not issues:
        return issues
    # A column built as the row maximum or minimum of others equals one of them on every row.
    built: dict[str, frozenset] = {}
    for column in {column for issue in issues for column in issue.columns}:
        built[column] = _row_extreme_of(frame, column, columns)
    issues = [
        issue
        for issue in issues
        if issue.columns[1] not in built[issue.columns[0]] and issue.columns[0] not in built[issue.columns[1]]
    ]
    # The highest and lowest reading in a window are equal wherever one reading was taken.
    windows = [issue for issue in issues if _same_reading(*issue.columns)]
    if windows:
        issues = [issue for issue in issues if issue not in windows] + [_single_readings(windows)]
    return issues


def _reading_stem(column: str) -> tuple[str, ...]:
    """A header's words without max, min, first, last or window numbers: OBS1_glucose_max is obs, glucose."""
    words = (re.sub(r"\d+", "", word) for word in normalized_name(column).split())
    return tuple(word for word in words if word and word not in WINDOW_WORDS)


def _same_reading(first: str, second: str) -> bool:
    """Whether two columns hold the max, min, first or last of one reading, or of it in two windows."""
    named = set(normalized_name(first).split()) | set(normalized_name(second).split())
    return bool(named & WINDOW_WORDS) and _reading_stem(first) == _reading_stem(second)


def _single_readings(issues: list[QualityIssue]) -> QualityIssue:
    """Pairs of window statistics of one reading that are often equal, as one note rather than copies."""
    leading = max(issues, key=lambda issue: issue.rows)
    shares = sorted(issue.share for issue in issues)
    spread = _percent(shares[0]) if len(issues) == 1 else f"{_percent(shares[0])} to {_percent(shares[-1])}"
    pairs = _plural(len(issues), "pair")
    detail = (
        f"In {pairs} of columns that hold the highest and lowest value of one reading, or the same reading "
        f"in two windows (such as {leading.columns[0]} and {leading.columns[1]}), the two are equal on "
        f"{spread} of rows. Where a single reading was taken in the window its maximum and minimum are that "
        "one value, so this is no sign of copying."
    )
    summary = f"{pairs} of window columns equal where one reading was taken"
    rule = "two columns of one reading that differ only by max, min, first, last or a window's number"
    columns = tuple(dict.fromkeys(column for issue in issues for column in issue.columns))
    return QualityIssue("single_reading", columns, leading.rows, leading.share, detail, summary, rule)


def _row_extreme_of(table: pd.DataFrame, column: str, columns: list[str]) -> frozenset:
    """The columns whose row maximum (or minimum) ``column`` holds, or an empty set.

    An air-quality index that is the largest of six sub-indices on 99% of days
    equals each of them wherever it is the largest; that is how it is built,
    not a copy. The parts are the other columns it never falls below (or
    rises above); it must equal their maximum (or minimum) on ROW_EXTREME_SHARE
    of the rows where both are present, and at least two parts must supply
    that value on 1% of rows or more: a column equal to one part almost
    everywhere is a copy of it. Judged on an evenly spaced sample.
    """
    frame = _sampled(table)[columns]
    others = [other for other in columns if other != column]
    if len(others) < 2:
        return frozenset()
    # Arrays rather than frames: a wide table checks thousands of columns here.
    target = frame[column].to_numpy(dtype="float64", na_value=np.nan)
    values = frame[others].to_numpy(dtype="float64", na_value=np.nan)
    present = ~np.isnan(values) & ~np.isnan(target)[:, None]
    counts = present.sum(axis=0)
    for below in (True, False):
        holds = values <= target[:, None] if below else values >= target[:, None]
        within = (holds | ~present).mean(axis=0)
        chosen = np.flatnonzero((within >= 0.98) & (counts >= 20))
        if chosen.size < 2:
            continue
        block = values[:, chosen]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            extreme = np.nanmax(block, axis=1) if below else np.nanmin(block, axis=1)
        both = ~np.isnan(target) & ~np.isnan(extreme)
        if both.sum() < 20 or float((target[both] == extreme[both]).mean()) < ROW_EXTREME_SHARE:
            continue
        supplying = (block[both] == target[both][:, None]).mean(axis=0)
        if (supplying >= 0.01).sum() >= 2 and supplying.max() < ROW_EXTREME_SHARE:
            return frozenset(others[index] for index in chosen)
    return frozenset()


def _shared_limits(a: pd.Series, b: pd.Series) -> list[float]:
    """Values that are the lowest non-zero value of both columns, or the highest of both.

    Two element contents both floored at 0.01, a detection limit, agree on
    every row below it without either being copied.
    """
    limits = []
    a_values, b_values = a[a.ne(0)].dropna(), b[b.ne(0)].dropna()
    if a_values.empty or b_values.empty:
        return limits
    for extreme in ("min", "max"):
        value = getattr(a_values, extreme)()
        if value == getattr(b_values, extreme)():
            limits.append(float(value))
    return limits


def _bounded_pair(a: pd.Series, b: pd.Series, first: str, second: str) -> tuple[str, str] | None:
    """The lower and upper column when one never exceeds the other but often falls short of it.

    A time to recurrence is set to the follow-up time where there was no
    recurrence, so it equals follow-up on part of the rows and is shorter on
    the rest, never longer; a discounted price is the list price where no
    discount applied. Such a pair is bounded, not copied. One must stay at or
    below the other on BOUNDED_SHARE of rows, be strictly below on a tenth,
    and there be of the same order (a median ratio of a fifth or more): a
    column copied into one on a ten-times larger scale lies below it too.
    """
    for low, high, names in ((a, b, (first, second)), (b, a, (second, first))):
        short = low < high
        if float((low <= high).mean()) < BOUNDED_SHARE or float(short.mean()) < 0.1:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = float(np.nanmedian((low[short] / high[short]).to_numpy(dtype="float64")))
        if ratio >= 0.2:
            return names
    return None


def _bounded_issue(low: str, high: str, count: int, share: float) -> QualityIssue:
    """A note that one column equals another where it reached its bound, rather than a copy."""
    detail = (
        f"{low} never exceeds {high} and equals it on {_plural(count, 'row')} ({_percent(share * 100)} of "
        f"rows with both): it looks recorded up to {high}, as a time to an event is set to the follow-up "
        "time where the event did not happen (censored), so this is no sign of copying."
    )
    summary = f"{low} reaches its bound, {high}, on {_percent(share * 100)} of rows"
    rule = (
        f"one column at or below another on {BOUNDED_SHARE:.0%} of rows, below it on a tenth or more, and "
        f"equal on {EQUAL_SHARE:.0%} or more"
    )
    return QualityIssue("bounded", (low, high), count, share * 100, detail, summary, rule)


def _equal_pair(frame: pd.DataFrame, first: str, second: str, dimension: str | None) -> QualityIssue | None:
    a, b = frame[first], frame[second]
    both = a.notna() & b.notna()
    if both.sum() < 20:
        return None
    equal = both & a.eq(b) & a.ne(0)
    share = float(equal.sum() / both.sum())
    # Agreement at a floor or a ceiling both share, or at zero, says nothing; the other rows decide.
    limits = _shared_limits(a, b)
    away = both & ~(a.eq(0) & b.eq(0))
    if limits:
        away &= ~(a.isin(limits) & b.isin(limits))
    agree = float((equal & away).sum() / away.sum()) if away.sum() >= 20 else 0.0
    rule = (
        f"two columns holding the same non-zero value in {EQUAL_SHARE:.0%} or more of rows, and of the "
        "rows away from any floor or ceiling the two share"
    )
    if share >= EQUAL_SHARE and (bounded := _bounded_pair(a[both], b[both], first, second)):
        return _bounded_issue(*bounded, int(equal.sum()), share)
    if share >= EQUAL_SHARE and agree >= EQUAL_SHARE:
        count = int(equal.sum())
        detail = (
            f"{first} equals {second} in {_plural(count, 'row')} ({_percent(share * 100)} of rows "
            "with both); check whether one was copied into the other."
        )
        summary = f"{first} equals {second} in {_percent(share * 100)} of rows"
        return QualityIssue("equal_columns", (first, second), count, share * 100, detail, summary, rule)
    if not dimension or dimension not in frame.columns:
        return None
    equal, both = equal & away, away
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
        f"against {_percent(elsewhere * 100) if elsewhere else '0%'} elsewhere), which suggests one "
        "was copied into the other there."
    )
    summary = f"{first} equals {second} in {within} of {label}'s rows"
    return QualityIssue("equal_columns", (first, second), count, float(shares.max() * 100), detail,
                        summary, rule + ", within one group", group=group)


def _sibling(words: list[str]) -> list[str] | None:
    for position, word in enumerate(words):
        if word in _MAX_WORDS:
            return [*words[:position], _MAX_WORDS[word], *words[position + 1 :]]
    return None


def _own_counts(words: tuple[str, ...], every: list[set[str]]) -> bool:
    """Whether a high/low column's side has its own count column (high_mp_n beside high_mp_mean).

    Then "high" and "low" name two groups, each with its own n, not the two
    ends of one quantity's range.
    """
    side = next(word for word in words if word in _MAX_WORDS)
    low_side = _MAX_WORDS[side]
    stem = set(words) - {side}
    return all(
        any(
            {end} <= names and names & SPREAD_WORDS and names - SPREAD_WORDS <= stem | {end}
            for names in every
        )
        for end in (side, low_side)
    )


def _min_max(frame: pd.DataFrame, columns: list[str]) -> list[QualityIssue]:
    """Rows where a max-named column falls below its min-named sibling, or its plain one, and rows
    where an average-named sibling of the two lies outside them.

    Spreads and counts are skipped (a high-exposure group's SD may well be
    below a low one's), and so are pairs whose two sides each have their own
    count column, which marks them as groups.
    """
    by_words = {tuple(normalized_name(column).split()): column for column in columns}
    every = [set(normalized_name(column).split()) for column in frame.columns]
    issues = []
    for words, high in by_words.items():
        low_words = _sibling(list(words))
        if low_words is None or set(words) & SPREAD_WORDS or _own_counts(words, every):
            continue
        plain = tuple(word for word in words if word not in _MAX_WORDS)
        low = by_words.get(tuple(low_words)) or by_words.get(plain)
        if not low or low == high:
            continue
        both = max(int((frame[high].notna() & frame[low].notna()).sum()), 1)
        below = frame[high] < frame[low]
        count = int(below.sum())
        if count:
            pairs = frame.loc[below, [high, low]].head(3).to_numpy()
            shown = tuple(f"{show_value(x)} < {show_value(y)}" for x, y in pairs)
            detail = (
                f"{high} is below {low} in {_plural(count, 'row')} (for example {shown[0]}), which can't "
                "happen; one of the two is wrong on those rows."
            )
            summary = f"{high} below {low} in {_plural(count, 'row')}"
            rule = "a max-named column below its min-named or plain sibling"
            issues.append(QualityIssue("min_max", (high, low), count, count / both * 100, detail, summary,
                                       rule, shown))
        middle = next((by_words[found] for found in _middles(words) if found in by_words), None)
        if middle and low == by_words.get(tuple(low_words)):
            found = _outside_range(frame, middle, low, high)
            if found:
                issues.append(found)
    return issues


def _middles(words: tuple[str, ...]) -> list[tuple[str, ...]]:
    """The names an average of a max-named column's reading would have: "pm avg", "mean pm", "pm"."""
    position = next(position for position, word in enumerate(words) if word in _MAX_WORDS)
    names = [(*words[:position], middle, *words[position + 1 :]) for middle in sorted(MIDDLE_WORDS)]
    return [*names, (*words[:position], *words[position + 1 :])]


def _outside_range(frame: pd.DataFrame, middle: str, low: str, high: str) -> QualityIssue | None:
    """Rows where an average lies outside the minimum and maximum of the same readings."""
    outside = (frame[middle] < frame[low]) | (frame[middle] > frame[high])
    count = int(outside.sum())
    if not count:
        return None
    rows = frame.loc[outside, [middle, low, high]].head(3).to_numpy()
    shown = tuple(f"{show_value(m)} outside {show_value(x)} to {show_value(y)}" for m, x, y in rows)
    both = max(int(frame[[middle, low, high]].notna().all(axis=1).sum()), 1)
    detail = (
        f"{middle} lies outside {low} to {high} in {_plural(count, 'row')} (for example {shown[0]}), "
        "which can't happen for an average of the same readings; one of the three is wrong on those rows."
    )
    summary = f"{middle} outside {low} to {high} in {_plural(count, 'row')}"
    rule = "an average-named column below its min-named sibling or above its max-named one"
    return QualityIssue("outside_range", (middle, low, high), count, count / both * 100, detail, summary,
                        rule, shown)


def _noun_words(column: str) -> list[str]:
    """A header's words in order, without anything in brackets: "SecchiDepth (m)" gives secchi, depth."""
    bare = re.sub(r"\([^)]*\)|\[[^\]]*\]", " ", normalized_name(column))
    return re.findall(r"[^\W_]+", bare)


def _parts_of_wholes(frame: pd.DataFrame, columns: list[str]) -> list[QualityIssue]:
    """Rows where a part exceeds the whole it belongs to, for pairs the names make plain.

    Two readings of the same kind ("SecchiDepth" and "WaterDepth", "Online Sales"
    and "Total Sales") pair as part and whole when they end in the same word,
    the whole's other words, if any, are WHOLE_WORDS (or, for a depth, name
    the whole water column), and the part's are not. The part must stay at or
    below the whole on PART_WHOLE_SHARE of the rows holding both, so the
    relation holds and the rows past it stand out.
    """
    named = {column: _noun_words(column) for column in columns}
    bounds_words = set(_MAX_WORDS) | set(_MAX_WORDS.values()) | MIDDLE_WORDS
    issues = []
    for whole, whole_words in named.items():
        if not whole_words:
            continue
        wholes = WHOLE_WORDS | WHOLE_DEPTH_WORDS if whole_words[-1] == "depth" else WHOLE_WORDS
        if set(whole_words[:-1]) - wholes:
            continue
        for part, part_words in named.items():
            qualifiers = set(part_words[:-1])
            if (
                part == whole or not part_words or part_words[-1] != whole_words[-1] or not qualifiers
                or qualifiers & (wholes | bounds_words)
            ):
                continue
            a, b = frame[part], frame[whole]
            both = a.notna() & b.notna()
            if both.sum() < 20:
                continue
            over = both & (a > b + 1e-9 * b.abs())
            count = int(over.sum())
            if not count or count > (1 - PART_WHOLE_SHARE) * both.sum():
                continue
            rows = frame.loc[over, [part, whole]].head(3).to_numpy()
            shown = tuple(f"{show_value(x)} > {show_value(y)}" for x, y in rows)
            within = f"{float((~over[both]).mean()) * 100:.1f}%"
            detail = (
                f"{part} is greater than {whole} in {_plural(count, 'row')} (for example {shown[0]}), "
                f"though it stays at or below it on {within} of the rows with both: a part can't exceed its "
                "whole, so one of the two is wrong on those rows."
            )
            summary = f"{part} above {whole} in {_plural(count, 'row')}"
            rule = (
                f"a part above the whole its name pairs it with, though at or below it on "
                f"{PART_WHOLE_SHARE:.0%} or more of rows"
            )
            issues.append(QualityIssue("part_whole", (part, whole), count, count / int(both.sum()) * 100,
                                       detail, summary, rule, shown))
    return issues


def _conversion(ratio: float, gap: float, spread: float) -> float | None:
    """The unit conversion factor a group's level is off by, if any.

    ``ratio`` is the group's median over the others' typical median, ``gap``
    over the nearest other median, and ``spread`` the largest over the
    smallest of the others' medians.
    """
    for factor in CONVERSION_FACTORS:
        if factor >= SMALL_FACTOR:
            if abs(np.log(ratio / factor)) <= np.log(CONVERSION_TOLERANCE) and gap >= np.sqrt(factor):
                return factor
        elif (
            abs(np.log(ratio / factor)) <= np.log(SMALL_FACTOR_TOLERANCE)
            and spread <= SMALL_FACTOR_SPREAD
            and gap >= factor / SMALL_FACTOR_SPREAD
        ):
            return factor
    return None


def _one_stretch(frame: pd.DataFrame, date: str | None, inside: np.ndarray) -> bool:
    """Whether a group's rows fill one stretch of time that no other group shares.

    A fire phase, a treatment period or a station's era is an event in time,
    so a level far from the rest during it is what happened, not a unit slip.
    """
    if not date or date not in frame.columns:
        return False
    times = frame[date]
    own = times[inside].dropna()
    if own.empty:
        return False
    within = (times >= own.min()) & (times <= own.max())
    others = int((within & ~inside).sum())
    return others <= 0.05 * int(within.sum())


def _group_levels(
    frame: pd.DataFrame,
    column: str,
    dimension: str,
    grouping: tuple[np.ndarray, pd.Index],
    date: str | None = None,
) -> QualityIssue | None:
    """A group whose median is off from the rest by about a unit conversion factor.

    The group's median, compared with the median of the other groups'
    medians, must lie within CONVERSION_TOLERANCE of a factor in
    CONVERSION_FACTORS (ten, a thousand) and clear every other group's median
    by at least the square root of that factor, so the others agree among
    themselves. A factor under SMALL_FACTOR (3.6 for m/s and km/h, 2.2 for kg
    and lb) is a ratio real groups often differ by, so it must match closely
    and the other groups must agree closely (see ``_conversion``). A group that fills one stretch of time is
    an event, not a unit slip, and is left to the comparisons. Needs three or
    more groups of five values: with two, one being far off is as likely the
    effect being studied.
    """
    codes, levels = grouping
    grouped = frame[column].groupby(codes)
    sizes = grouped.count().drop(-1, errors="ignore")
    medians = grouped.median().drop(-1, errors="ignore")[sizes >= 5]
    if len(medians) < 3 or (medians <= 0).any():
        return None
    ordered = medians.sort_values(ascending=False)
    for position, direction in ((0, "above"), (-1, "below")):
        odd = float(ordered.iloc[position])
        rest = ordered.drop(ordered.index[position])
        typical = float(rest.median())
        nearest = float(rest.max() if direction == "above" else rest.min())
        ratio = odd / typical if direction == "above" else typical / odd
        gap = odd / nearest if direction == "above" else nearest / odd
        factor = _conversion(ratio, gap, float(rest.max() / rest.min()))
        if factor is None:
            continue
        inside = codes == ordered.index[position]
        if _one_stretch(frame, date, inside):
            continue
        group = levels[ordered.index[position]]
        label = group_label(dimension, group)
        detail = (
            f"{label}'s median {column} ({show_value(odd)}) is {ratio:.3g}× {direction} the typical "
            f"median of the other {dimension} values ({show_value(typical)}), close to a unit "
            f"conversion factor of {show_value(factor)}; check whether {label} was recorded in "
            "other units before comparing it with the rest."
        )
        summary = (
            f"{label}'s median {column} is {ratio:.3g}× {direction} the other {dimension} values', "
            "near a unit factor"
        )
        rule = (
            f"one group's median within {CONVERSION_TOLERANCE - 1:.0%} of a unit conversion factor "
            f"({SMALL_FACTOR_TOLERANCE - 1:.0%} for factors under {SMALL_FACTOR:.0f}) times the other "
            "groups' typical median, clear of all of them, and not one stretch of time"
        )
        rows = int(sizes[ordered.index[position]])
        return QualityIssue("group_level", (column,), rows, rows / max(len(frame), 1) * 100, detail,
                            summary, rule, group=group)
    return None


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
    sample = _sampled(frame)[list(dict.fromkeys([*targets, *numeric]))]
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


def _keys_or_labels(name: str, values: np.ndarray) -> bool:
    """Whether a numeric column holds layout numbers or keys rather than readings.

    A key or label name ("Patient", "No") holds keys only when its values are
    whole numbers; decimal readings under such a name (nitric oxide as "NO") are checked.
    """
    if is_design_name(name):
        return True
    whole = bool(np.all(values == np.round(values)))
    return whole and (is_identifier_name(name) or is_entity_name(name))


def _time_order(frame: pd.DataFrame, date: str | None) -> np.ndarray:
    """Row positions in time order when there is a date, else in table order."""
    if date and date in frame.columns:
        return np.argsort(frame[date].to_numpy(), kind="stable")
    return np.arange(len(frame))


def _longest_run(positions: np.ndarray) -> tuple[int, int]:
    """Start and length of the longest stretch of consecutive integers in sorted ``positions``."""
    if positions.size == 0:
        return 0, 0
    breaks = np.flatnonzero(np.diff(positions) != 1)
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, positions.size - 1]
    longest = int(np.argmax(ends - starts))
    return int(starts[longest]), int(ends[longest] - starts[longest] + 1)


def _stretch(frame: pd.DataFrame, date: str | None, labels: pd.Index) -> str:
    """Where a block of rows sits: "from 20 May 2023 to 31 May 2023", or "rows 133 to 144"."""
    if date and date in frame.columns:
        times = frame.loc[labels, date].dropna()
        if not times.empty:
            first, last = pd.Timestamp(times.min()), pd.Timestamp(times.max())
            dated = first == first.normalize() and last == last.normalize()
            shown = "%d %b %Y" if dated else "%d %b %Y %H:%M"
            return f"from {first.strftime(shown)} to {last.strftime(shown)}"
    positions = np.sort(frame.index.get_indexer(labels))
    return f"rows {positions[0] + 1:,} to {positions[-1] + 1:,} of the table"


def _swapped(
    frame: pd.DataFrame, profiles: dict[str, _Column], columns: list[str], date: str | None
) -> list[QualityIssue]:
    """Blocks of consecutive rows in which two columns hold each other's values.

    Rows where column A lies outside its own bulk (Tukey's 3×IQR fences) are
    candidates. For each other column B, a candidate row counts as swapped
    when A's value lies in B's usual range and B's value in A's, the usual
    ranges being the 1st to 99th percentiles of the other rows, widened by half
    their width, and the two ranges not overlapping. The swapped rows, at
    least SWAP_ROWS of them, must fill 80% or more of the stretch they span,
    in time order when there is a date.
    """
    if len(columns) < 2 or len(columns) > PAIR_COLUMNS or len(frame) < 20:
        return []
    seeds: dict[int, np.ndarray] = {}
    for a, column in enumerate(columns):
        profile = profiles[column]
        first, third = profile.quantile(0.25), profile.quantile(0.75)
        out = (profile.array < first - 3 * (third - first)) | (profile.array > third + 3 * (third - first))
        if SWAP_ROWS <= int(out.sum()) <= CLUSTER_SHARE * len(frame):
            seeds[a] = np.flatnonzero(frame[column].notna().to_numpy())[out]
    if not seeds:
        return []
    arrays = [frame[column].to_numpy(dtype="float64", na_value=np.nan) for column in columns]
    # Ranges come from evenly spaced rows; blocks are found on every row.
    picked = np.arange(len(frame))
    if len(frame) > SAMPLE_ROWS:
        picked = np.unique((np.arange(SAMPLE_ROWS) * (len(frame) / SAMPLE_ROWS)).astype(int))
    sample = np.column_stack([array[picked] for array in arrays])
    rank = order = None
    issues: list[QualityIssue] = []
    seen: set[frozenset] = set()
    for a, rows in seeds.items():
        rest = sample[~np.isin(picked, rows)]

        def usual(column: int, rest: np.ndarray = rest) -> tuple[float, float]:
            low, high = np.nanpercentile(rest[:, column], [1, 99])
            return low - (high - low) / 2, high + (high - low) / 2

        low_a, high_a = usual(a)
        candidates = np.column_stack([array[rows] for array in arrays])
        # B's values on the candidate rows inside A's usual range, for every B at once.
        in_a = (candidates >= low_a) & (candidates <= high_a)
        for b in np.flatnonzero(in_a.sum(axis=0) >= SWAP_ROWS):
            if b == a or frozenset((a, b)) in seen:
                continue
            low_b, high_b = usual(b)
            if not (high_a < low_b or high_b < low_a):
                continue
            swapped = in_a[:, b] & (candidates[:, a] >= low_b) & (candidates[:, a] <= high_b)
            if swapped.sum() < SWAP_ROWS:
                continue
            if rank is None:
                order = _time_order(frame, date)
                rank = np.empty(len(frame), dtype=int)
                rank[order] = np.arange(len(frame))
            positions = np.sort(rank[rows[swapped]])
            if positions.size < SWAP_ROWS:
                continue
            # One block: the swapped rows fill at least 80% of the stretch they span.
            block = order[positions[0] : positions[-1] + 1]
            if positions.size < 0.8 * block.size:
                continue
            seen.add(frozenset((a, b)))
            length = int(block.size)
            labels = frame.index[block]
            names = (columns[a], columns[b])
            shown_a = _range_of(pd.Series(arrays[a][block]))
            shown_b = _range_of(pd.Series(arrays[b][block]))
            where = _stretch(frame, date, labels)
            detail = (
                f"On {_plural(length, 'consecutive row')} ({where}), {names[0]} holds values in "
                f"{names[1]}'s usual range ({shown_a}) and {names[1]} values in {names[0]}'s ({shown_b}): "
                "the two columns look swapped there, so neither is a reading of its own quantity on "
                "those rows."
            )
            summary = f"{names[0]} and {names[1]} swapped on {_plural(length, 'row')} ({where})"
            rule = (
                "a block of consecutive rows where each of two columns lies in the other's usual range "
                "(1st to 99th percentile of the other rows, widened by half), the ranges apart"
            )
            issues.append(QualityIssue(
                "swapped", names, length, length / len(frame) * 100, detail, summary, rule,
                examples=(f"{show_value(arrays[a][block[0]])} and {show_value(arrays[b][block[0]])}",),
                cells={names[0]: labels, names[1]: labels},
            ))
    return issues


def _without_swapped(issues: list[QualityIssue], swaps: list[QualityIssue]) -> list[QualityIssue]:
    """Drop value issues whose cells all lie in a swapped block, or another issue that explains
    them (a row of zeros): that issue says what is wrong with them."""
    if not swaps:
        return issues
    covered: dict[str, pd.Index] = {}
    for swap in swaps:
        for column, labels in swap.cells.items():
            covered[column] = covered[column].union(labels) if column in covered else labels
    kept = []
    for issue in issues:
        if issue.check in ("impossible", "extreme", "cluster", "sentinel") and issue.cells and all(
            column in covered and labels.isin(covered[column]).all() for column, labels in issue.cells.items()
        ):
            continue
        kept.append(issue)
    return kept


def _cluster_runs(issues: list[QualityIssue], frame: pd.DataFrame, date: str | None) -> list[QualityIssue]:
    """Say when a second cluster is one unbroken stretch of rows: a change of unit, scale or sensor then."""
    order = None
    reworded = []
    for issue in issues:
        if issue.check != "cluster" or len(issue.cells) != 1:
            reworded.append(issue)
            continue
        if order is None:
            order = np.empty(len(frame), dtype=int)
            order[_time_order(frame, date)] = np.arange(len(frame))
        labels = next(iter(issue.cells.values()))
        positions = np.sort(order[frame.index.get_indexer(labels)])
        _, length = _longest_run(positions)
        if length < max(SWAP_ROWS, 0.9 * positions.size):
            reworded.append(issue)
            continue
        where = _stretch(frame, date, labels)
        detail = issue.detail.replace(
            "a second cluster, or a possible sensor or unit fault",
            f"one unbroken stretch of {_plural(length, 'row')} ({where}), so the unit, the scale or the "
            "sensor probably changed for that stretch",
        )
        reworded.append(QualityIssue(
            issue.check, issue.columns, issue.rows, issue.share, detail,
            issue.summary.replace("in a second cluster", f"in one stretch, {where},"), issue.rule,
            issue.examples, issue.group, issue.cells,
        ))
    return reworded


def _groupings(frame: pd.DataFrame, roles: ColumnRoles | None) -> list[str]:
    """Columns that sort rows into categories: text columns and named codes with 3 or more levels."""
    named = set(roles.factors) | set(roles.dimensions) if roles else set()
    found = []
    for column in frame.columns:
        series = frame[column]
        if column in named or series.dtype == object or isinstance(series.dtype, pd.CategoricalDtype):
            levels = series.nunique(dropna=True)
            if 3 <= levels <= len(frame) / 3:
                found.append(column)
    return found


def _labels(
    frame: pd.DataFrame, roles: ColumnRoles | None, most: int = 20
) -> dict[str, tuple[np.ndarray, pd.Index]]:
    """Columns that split the rows into 2 to ``most`` labelled groups, with each row's level code.

    Text, category and true/false columns count, and so do the codes the roles
    name; the identifier does not.
    """
    named = set(roles.factors) | set(roles.dimensions) if roles else set()
    identifier = roles.identifier if roles else None
    found = {}
    for column in frame.columns:
        series = frame[column]
        if column == identifier:
            continue
        if (
            column in named or series.dtype == object or is_bool_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
        ):
            codes, levels = pd.factorize(series)
            if 2 <= len(levels) <= most:
                found[column] = (codes, levels)
    return found


def _tied_check(frame: pd.DataFrame, roles: ColumnRoles | None) -> Callable[[str, float], bool]:
    """A test of whether a piled-up value belongs to a category rather than a limit.

    It does when every row at the value falls in levels of one grouping in
    which the column holds that value on 95% or more of rows, those levels
    averaging three rows or more each: trophic level 2 is what every
    herbivore species has, not a floor of the method. The groupings are
    found the first time a pile-up is tested.
    """
    found: list[list[str]] = []

    def tied(column: str, value: float) -> bool:
        if not found:
            found.append(_groupings(frame, roles))
        values = frame[column]
        at = values == value
        rows = int(at.sum())
        for grouping in found[0]:
            if grouping == column:
                continue
            keys = frame[grouping]
            levels = keys[at].dropna().unique()
            if len(levels) == 0 or len(levels) > rows / 3 or keys[at].isna().mean() > 0.05:
                continue
            inside = keys.isin(levels) & values.notna()
            if float((values[inside] == value).mean()) >= 0.95:
                return True
        return False

    return tied


def _one_sample_fills(frame: pd.DataFrame, issues: list[QualityIssue]) -> list[QualityIssue]:
    """Drop fill values whose rows all belong to one sample: one reading repeated on its rows.

    A survey that lists every species found on a visit repeats the visit's
    sulphate on each species row, so 429.815 on 13 rows is one visit, not a
    gap filled 13 times. A key is a text column with at least ten values; the
    fill is dropped when every row holding it shares one of them.
    """
    keys = [
        column for column in frame.columns
        if frame[column].dtype == object and 10 <= frame[column].nunique() < len(frame)
    ] if any(issue.check == "imputed" for issue in issues) else []
    kept = []
    for issue in issues:
        if issue.check == "imputed" and any(
            all(frame.loc[labels, key].nunique(dropna=False) == 1 for labels in issue.cells.values())
            for key in keys
        ):
            continue
        kept.append(issue)
    return kept


def _failed_records(
    frame: pd.DataFrame, issues: list[QualityIssue], columns: list[str]
) -> list[QualityIssue]:
    """Values other columns hold on the rows where one column holds a missing-value code: one failed record.

    A logger that fails writes its code in one channel and its own fixed
    values in the others (a sound level of 0, a battery of 2.5 V on the rows
    where temperature reads -25.6). A value counts when it fills
    FAILED_RECORD_SHARE of the code's rows and turns up elsewhere on at most a
    tenth as many rows; such cells are no more readings than the code is.
    """
    found = []
    for issue in issues:
        if issue.check != "sentinel" or not issue.cells:
            continue
        rows = next(iter(issue.cells.values()))
        for labels in issue.cells.values():
            rows = rows.union(labels)
        if len(rows) < FAILED_RECORD_ROWS:
            continue
        inside = frame.index.isin(rows)
        held: dict[str, float] = {}
        cells: dict[str, pd.Index] = {}
        for column in columns:
            if column in issue.columns:
                continue
            values = frame[column].to_numpy(dtype="float64", na_value=np.nan)
            codes, levels = pd.factorize(values[inside])
            if not len(levels):
                continue
            top = int(np.argmax(np.bincount(codes[codes >= 0])))
            value = float(levels[top])
            at = values == value
            hits = int((at & inside).sum())
            if hits < FAILED_RECORD_SHARE * len(rows) or int((at & ~inside).sum()) > 0.1 * hits:
                continue
            held[column] = value
            cells[column] = frame.index[at & inside]
        if not held:
            continue
        source = _listed(list(issue.columns))
        code = issue.examples[0] if issue.examples else "a code"
        values = _listed([f"{column} {show_value(value)}" for column, value in held.items()])
        detail = (
            f"On the {_plural(len(rows), 'row')} where {source} holds {code} (a missing-value code), "
            f"{values}, values {'that column' if len(held) == 1 else 'those columns'} almost never "
            "hold elsewhere: the whole record failed there, so these cells are not readings either and are "
            "left out of relationships and comparisons."
        )
        summary = f"{_listed([*issue.columns, *held], 3)}: one failed record on {_plural(len(rows), 'row')}"
        rule = (
            f"one value filling {FAILED_RECORD_SHARE:.0%} or more of the rows a missing-value code is on, "
            "and elsewhere on at most a tenth as many rows"
        )
        found.append(QualityIssue(
            "failed_record", tuple(held), len(rows), len(rows) / max(len(frame), 1) * 100, detail, summary,
            rule, tuple(show_value(value) for value in held.values()), group=issue.group, cells=cells,
        ))
    return found


def _zero_rows(frame: pd.DataFrame, profiles: dict[str, _Column], columns: list[str]) -> QualityIssue | None:
    """Rows on which nearly every reading is exactly 0, in columns otherwise rarely 0.

    A mineral with no electrons and no density is a row of blanks written as
    zeros, not a measurement. The columns must each be 0 on under
    ZERO_ROW_COLUMN_SHARE of their values, at least three of them; a row
    counts when ZERO_ROW_SHARE of them are 0 on it.
    """
    rare = [
        column for column in columns
        if float(profiles[column].counts[profiles[column].uniques == 0].sum()) / profiles[column].size
        < ZERO_ROW_COLUMN_SHARE
    ]
    if len(rare) < 3:
        return None
    zeros = (frame[rare].to_numpy(dtype="float64", na_value=np.nan) == 0)
    hit = zeros.sum(axis=1) >= max(3, ZERO_ROW_SHARE * len(rare))
    count = int(hit.sum())
    if count < 2 or count > 0.25 * len(frame):
        return None
    labels = frame.index[hit]
    involved = [column for position, column in enumerate(rare) if zeros[hit, position].mean() >= 0.5]
    cells = {column: labels[zeros[hit, rare.index(column)]] for column in involved}
    named = _listed(involved)
    detail = (
        f"{_plural(count, 'row')} hold exactly 0 in nearly every reading ({named}), though each of these "
        "columns is otherwise rarely 0: probably missing values recorded as 0, so those cells are left "
        "out of relationships and comparisons."
    )
    summary = f"{_plural(count, 'row')} of zeros in {len(involved)} readings ({_listed(involved, 3)})"
    rule = (
        f"rows 0 in {ZERO_ROW_SHARE:.0%} or more of three or more readings that are each 0 on under "
        f"{ZERO_ROW_COLUMN_SHARE:.0%} of their values"
    )
    share = count / max(len(frame), 1) * 100
    return QualityIssue("zero_rows", tuple(involved), count, share, detail, summary, rule,
                        (f"{count:,} rows",), cells=cells)


def _placeholder_mask(series: pd.Series) -> np.ndarray | None:
    """True for cells holding a placeholder (0, "0", "-"), or None when the column holds none to test."""
    if is_bool_dtype(series):
        return None
    if is_numeric_dtype(series):
        return series.to_numpy(dtype="float64", na_value=np.nan) == 0
    if series.dtype != object:
        return None
    text = series.astype("string").str.strip()
    return text.isin(PLACEHOLDER_TOKENS).to_numpy(dtype=bool, na_value=False)


def _placeholder_blocks(frame: pd.DataFrame, identifier: str | None) -> list[QualityIssue]:
    """Columns holding 0 or "-" on exactly the same rows, and real values on the rest: not recorded there.

    A feature table writes "-" for the name and 0 for the formula, the score
    and the error of every feature it could not annotate. Those cells say
    nothing was recorded; read as values they make "score 0" a group and a
    finding. A block needs three or more columns, at least one of them text,
    whose placeholder rows agree (Jaccard PLACEHOLDER_OVERLAP) and cover 5% to
    95% of the table, with ten or more rows of real values.
    """
    rows = len(frame)
    masks, names, texts = [], [], []
    for column in frame.columns:
        if column == identifier:
            continue
        mask = _placeholder_mask(frame[column])
        if mask is None:
            continue
        held = int(mask.sum())
        if 0.05 * rows <= held <= 0.95 * rows and int(frame[column].notna().sum()) - held >= 10:
            masks.append(mask)
            names.append(column)
            texts.append(not is_numeric_dtype(frame[column]))
    if len(names) < 3 or not any(texts):
        return []
    matrix = np.column_stack(masks).astype("float32")
    together = matrix.T @ matrix
    counts = np.diag(together)
    overlap = together / np.maximum(counts[:, None] + counts[None, :] - together, 1)
    issues = []
    taken: set[int] = set()
    for seed in np.flatnonzero(texts):
        if seed in taken:
            continue
        members = [int(position) for position in np.flatnonzero(overlap[seed] >= PLACEHOLDER_OVERLAP)]
        members = [position for position in members if position not in taken]
        if len(members) < 3:
            continue
        taken.update(members)
        block = [names[position] for position in members]
        shared = np.logical_and.reduce([masks[position] for position in members])
        count = int(shared.sum())
        cells = {names[position]: frame.index[masks[position]] for position in members}
        detail = (
            f"{_listed(block)} hold 0 or a dash on the same {_plural(count, 'row')} "
            f"({_percent(count / rows * 100)}) and real values on the rest: on those rows nothing was "
            "recorded in them, so the zeros and dashes are left out of relationships and comparisons rather "
            "than read as values or a group."
        )
        on_rows = _plural(count, "row")
        summary = f"{len(block)} columns not recorded on the same {on_rows} ({_listed(block, 3)})"
        rule = (
            f"three or more columns, one of them text, holding 0 or a dash on rows that agree by "
            f"{PLACEHOLDER_OVERLAP:.0%} or more"
        )
        issues.append(QualityIssue("placeholder", tuple(block), count, count / rows * 100, detail, summary,
                                   rule, cells=cells))
    return issues


def _constant_columns(frame: pd.DataFrame, roles: ColumnRoles | None) -> QualityIssue | None:
    """Columns holding one value on every row where they are recorded.

    They can't differ or relate to anything, so the analysis leaves them out;
    saying so keeps a never-detected compound (0 in every sample) or a fill
    value from vanishing without a word. Text columns count only in a flat
    table, since focusing on one part of a long or stacked table makes the
    column naming that part constant. Dates are left to the timeline, and a
    measure that never changes to the brief, which reports it as it is.
    """
    flat = roles is None or roles.table_shape == "flat"
    measure = roles.measure if roles else None
    rows = len(frame)
    found: dict[str, object] = {}
    for column in frame.columns:
        series = frame[column]
        text = not is_numeric_dtype(series)
        if column == measure or is_datetime64_any_dtype(series) or (text and not flat):
            continue
        present = series.dropna()
        if len(present) < max(2, 0.9 * rows):
            continue
        first = present.iloc[0]
        if is_numeric_dtype(series) and not is_bool_dtype(series):
            values = present.to_numpy(dtype="float64")
            if np.all(values == values[0]):
                found[column] = float(values[0])
        elif (present == first).all():
            found[column] = first
    if not found:
        return None
    zeros = [column for column, value in found.items() if isinstance(value, float) and value == 0]
    others = {column: value for column, value in found.items() if column not in zeros}
    parts = []
    if zeros:
        parts.append(f"{_listed(zeros)} {'is' if len(zeros) == 1 else 'are'} 0 in every row")
    if others:
        shown = [f"{column} ({value:g})" if isinstance(value, float) else f"{column} ({value})"
                 for column, value in others.items()]
        verb = "holds one value" if len(others) == 1 else "each hold one value"
        parts.append(f"{_listed(shown)} {verb} in every row")
    detail = (
        f"{'; '.join(parts)}. A column that never changes can't differ between groups or relate to "
        "anything, so the analysis leaves it out: a 0 everywhere means never detected or never recorded, "
        "and one repeated reading may be a fill value or a detection limit."
    )
    noun = "column" if len(found) == 1 else "columns"
    summary = f"{len(found)} constant {noun} ({_listed(list(found), 3)})"
    rule = "one value on every row where the column is recorded, recorded on 90% of rows or more"
    return QualityIssue("constant", tuple(found), len(found), 100.0, detail, summary, rule)


def _zero_heavy(profiles: dict[str, _Column], columns: list[str]) -> QualityIssue | None:
    """Readings that are exactly 0 in a third or more of rows: absences, or values below a detection limit.

    Their Pearson correlations mostly contrast the zero rows with the rest, so
    rank correlations, or zero against not zero, say more. Whole numbers
    (counts) and rain, whose zeros are plainly dry days, are left out.
    """
    heavy = {}
    for column in columns:
        profile = profiles[column]
        zeros = float(profile.counts[profile.uniques == 0].sum())
        share = zeros / max(profile.size, 1)
        if not ZERO_HEAVY_SHARE <= share < 1 or profile.whole or profile.size - zeros < 10:
            continue
        if _name_words(column) & DRY_ZERO_WORDS:
            continue
        heavy[column] = share
    if not heavy:
        return None
    shown = _listed([f"{column} ({_percent(share * 100)})" for column, share in heavy.items()])
    detail = (
        f"{shown} {'is' if len(heavy) == 1 else 'are'} exactly 0 in a third or more of rows, either true "
        "absences or values below a detection limit: Pearson correlations with them mostly contrast the zero "
        "rows with the rest, so the rank correlation, or comparing zero with non-zero rows, is the better "
        "guide."
    )
    summary = f"{_listed(list(heavy), 3)}: exactly 0 in a third or more of rows"
    rule = f"exactly 0 in {ZERO_HEAVY_SHARE:.0%} or more of a reading's values, not whole numbers"
    return QualityIssue("zeros", tuple(heavy), len(heavy), max(heavy.values()) * 100, detail, summary, rule)


def _duplicates(frame: pd.DataFrame, profiles: dict[str, _Column]) -> QualityIssue | None:
    """Rows that repeat an earlier row exactly, in every column.

    Only tables with a continuous reading (decimals with many distinct
    values) are checked, since those rows can't repeat by chance, and only
    rows with such a reading recorded count: two blank records are not copies.
    Listed from DUPLICATE_SHARE of the rows.
    """
    rows = len(frame)
    readings = [
        column for column, profile in profiles.items()
        if not profile.whole and profile.uniques.size >= FLAT_MIN_LEVELS
    ]
    if rows < 20 or frame.shape[1] < 2 or not readings:
        return None
    # Whole rows can only repeat where the readings do, so those rows alone are hashed in full.
    key = np.zeros(rows, dtype=np.uint64)
    recorded = np.zeros(rows, dtype=bool)
    for column in readings:
        values = frame[column].to_numpy(dtype="float64", na_value=np.nan) + 0.0
        recorded |= ~np.isnan(values)
        key = key * np.uint64(1_000_003) ^ np.nan_to_num(values, nan=np.inf).view(np.uint64)
    candidates = pd.Series(key).duplicated(keep=False).to_numpy() & recorded
    if candidates.sum() < 2:
        return None
    hashed = pd.util.hash_pandas_object(frame[candidates], index=False)
    count = int(hashed.duplicated().sum())
    share = count / rows
    if not count or share < DUPLICATE_SHARE:
        return None
    times = int(hashed.value_counts().iloc[0])
    distinct = rows - count
    most = f"; one row appears {times:,} times" if times > 2 else ""
    detail = (
        f"{count:,} of {rows:,} rows ({_percent(share * 100)}) repeat an earlier row exactly in every "
        f"column, leaving {distinct:,} distinct rows{most}. Rows of continuous readings don't repeat by "
        "chance, so these are probably copies, and tests that count them as separate records are "
        "over-confident."
    )
    summary = (
        f"{count:,} rows ({_percent(share * 100)}) are exact copies of other rows ({distinct:,} distinct)"
    )
    rule = f"rows identical to an earlier row in every column, {DUPLICATE_SHARE:.0%} or more of the table"
    return QualityIssue("duplicates", (), count, share * 100, detail, summary, rule,
                        examples=(f"{distinct:,} distinct",))


def _flatness(column: _Column) -> tuple[float, float]:
    """Kolmogorov-Smirnov p-value against a uniform spread over the column's range, and excess kurtosis.

    The uniform range is widened by half the smallest step between values on
    each side, so whole numbers spread evenly over 1 to 50 count as flat.
    """
    if column.uniques.size < 2:
        return 0.0, 0.0
    step = float(np.min(np.diff(column.uniques)))
    low, width = float(column.uniques[0]) - step / 2, float(column.uniques[-1] - column.uniques[0]) + step
    above = np.cumsum(column.counts) / column.size
    below = above - column.counts / column.size
    expected_above = (column.uniques + step / 2 - low) / width
    expected_below = (column.uniques - step / 2 - low) / width
    distance = float(max(np.max(np.abs(above - expected_above)), np.max(np.abs(below - expected_below))))
    p_value = float(stats.kstwo.sf(distance, column.size))
    return p_value, float(stats.kurtosis(column.array))


def _quartile_share(ordered: np.ndarray) -> float:
    """The interquartile range of sorted values as a share of their 5th-to-95th percentile range.

    A uniform spread gives 0.56 and a bell curve 0.41, and a few outliers
    can't move it, so it screens columns cheaply before the full test.
    """
    if ordered.size < 20:
        return 0.0
    q05, q25, q75, q95 = (
        ordered[int(round(share * (ordered.size - 1)))] for share in (0.05, 0.25, 0.75, 0.95)
    )
    return float((q75 - q25) / (q95 - q05)) if q95 > q05 else 0.0


def _generated(
    frame: pd.DataFrame, profiles: dict[str, _Column], columns: list[str], flagged: dict[str, pd.Index]
) -> QualityIssue | None:
    """Say when the table may be simulated: most readings flat, and no pair related beyond chance.

    Real measurements are rarely spread evenly between their extremes, and
    readings taken together nearly always relate somewhere. The test runs on
    distinct rows with the cells other checks flagged left out, so copies and
    a few wrong values neither hide nor fake the pattern. A column is flat
    when a Kolmogorov-Smirnov test (scipy) does not reject a uniform spread
    over its range (p of FLAT_P or more) and its excess kurtosis is
    FLAT_KURTOSIS or less. The largest |r| between columns is compared with the
    largest chance gives across that many pairs (Bonferroni, FLAT_CHANCE).
    """
    available = [
        column for column in columns
        if profiles[column].size >= FLAT_MIN_VALUES and profiles[column].uniques.size >= FLAT_MIN_LEVELS
    ]
    eligible = available[:FLAT_PAIR_COLUMNS]
    if len(eligible) < 3:
        return None

    def kept(column: str) -> np.ndarray:
        profile = profiles[column]
        labels = flagged.get(column)
        return profile.ordered if labels is None else np.sort(profile.array[~profile.index.isin(labels)])

    screened = [column for column in eligible if _quartile_share(kept(column)) >= 0.48]
    if len(screened) < 3 or len(screened) <= len(eligible) / 2:
        return None
    table = frame[eligible].copy()
    for column, labels in flagged.items():
        if column in table.columns:
            table[column] = table[column].mask(table.index.isin(labels))
    table = table.drop_duplicates()
    flat = []
    for column in screened:
        profile = _Column.of(column, table[column])
        if profile.size < FLAT_MIN_VALUES or profile.uniques.size < FLAT_MIN_LEVELS:
            continue
        p_value, kurtosis = _flatness(profile)
        if p_value >= FLAT_P and kurtosis <= FLAT_KURTOSIS:
            flat.append(column)
    if len(flat) < 3 or len(flat) <= len(eligible) / 2:
        return None
    sample = _sampled(table)
    values = sample.to_numpy(dtype="float64")
    correlations = sample.corr().to_numpy() if np.isnan(values).any() else np.corrcoef(values, rowvar=False)
    pairs = len(eligible) * (len(eligible) - 1) // 2
    strongest = float(np.nanmax(np.abs(correlations[np.triu_indices(len(eligible), k=1)])))
    n = len(sample)
    if n < 30:
        return None
    chance = float(np.tanh(stats.norm.isf(FLAT_CHANCE / 2 / pairs) / np.sqrt(n - 3)))
    if strongest >= chance:
        return None
    named = _listed(flat)
    tested = f"{len(eligible)}"
    if len(available) > len(eligible):
        tested = f"the first {len(eligible)} of {len(available):,}"
    detail = (
        f"{len(flat)} of {tested} numeric columns ({named}) are spread evenly between their "
        "lowest and highest values, as random numbers are, and no two columns relate beyond what "
        f"chance gives (largest |r| = {strongest:.2f} across {pairs:,} pairs; chance alone reaches "
        f"{chance:.2f}). Real measurements are rarely flat and unrelated like this, so the table may "
        "be simulated: check how it was produced before reading anything into its findings."
    )
    summary = f"{len(flat)} of {len(eligible)} columns are flat and none relate: the table may be simulated"
    rule = (
        f"most columns uniform-like (Kolmogorov-Smirnov p ≥ {FLAT_P}, excess kurtosis ≤ "
        f"{FLAT_KURTOSIS}) and no pair beyond the Bonferroni {FLAT_CHANCE:.0%} chance level for |r|"
    )
    return QualityIssue("generated", tuple(flat), len(frame), 100.0, detail, summary, rule,
                        examples=(f"largest |r| {strongest:.2f}",))


def _flatlines(
    frame: pd.DataFrame,
    profiles: dict[str, _Column],
    columns: list[str],
    date: str | None,
    dimension: str | None,
    limits: dict[str, set[float]],
) -> list[QualityIssue]:
    """Readings stuck at zero, or on one value for a long stretch, in time order.

    Each group of the grouping column (a station, a city) is followed on its
    own. Two patterns count: runs of exact zeros FLATLINE_RUN readings or
    longer in a column that is otherwise positive and where zero is not common
    (FLATLINE_VALUE_SHARE), since a working sensor rarely reads exactly zero for
    that long; and a single run of any value holding FLATLINE_STALL of the
    column. Columns whose values often repeat anyway (FLATLINE_REPEATS), as
    coarse or slow readings do, running totals (which hold still between
    events) and values already reported as a cap or floor are left out.
    """
    if not date or date not in frame.columns or len(frame) < 2 * FLATLINE_RUN:
        return []
    keys = np.zeros(len(frame), dtype=int)
    if dimension in frame.columns:
        keys = pd.factorize(frame[dimension])[0]
    order = np.lexsort((frame[date].to_numpy(), keys))
    same_group = np.r_[False, keys[order][1:] == keys[order][:-1]]
    issues = []
    for column in columns:
        profile = profiles[column]
        if profile.uniques.size < FLAT_MIN_LEVELS or profile.size < 2 * FLATLINE_RUN:
            continue
        x = frame[column].to_numpy(dtype="float64")[order]
        present = ~np.isnan(x)
        repeat = np.r_[False, x[1:] == x[:-1]] & same_group & present
        starts = np.flatnonzero(~repeat)
        lengths = np.diff(np.r_[starts, x.size])
        held = x[starts]
        long = (lengths >= FLATLINE_RUN) & ~np.isnan(held)
        if not long.any():
            continue
        in_long = np.repeat(long, lengths)
        if float(repeat[~in_long].mean()) > FLATLINE_REPEATS:
            continue
        step = np.r_[np.nan, np.diff(x)][same_group & present]
        step = step[np.isfinite(step) & (step != 0)]
        if step.size and float((step > 0).mean()) >= 0.9:
            continue  # a running total
        zero_share = float(profile.counts[profile.uniques == 0].sum()) / profile.size
        zeros = long & (held == 0) & (zero_share < FLATLINE_VALUE_SHARE) & (profile.ordered[0] >= 0)
        stalls = long & (held != 0) & (lengths >= max(FLATLINE_STALL * profile.size, 10 * FLATLINE_RUN))
        stalls &= ~np.isin(held, list(limits.get(column, set())))
        stuck = np.flatnonzero(zeros | stalls)
        if not stuck.size:
            continue
        rows = int(lengths[stuck].sum())
        longest = stuck[np.argmax(lengths[stuck])]
        span = frame.index[order[starts[longest] : starts[longest] + lengths[longest]]]
        share = _percent(rows / profile.size * 100)
        if zeros.any() and not stalls.any():
            detail = (
                f"{column} reads exactly 0 for {FLATLINE_RUN} or more readings in a row in "
                f"{_plural(stuck.size, 'stretch', 'stretches')}, {_plural(rows, 'row')} ({share}) in all; "
                f"the longest runs {lengths[longest]:,} readings ({_stretch(frame, date, span)}). A working "
                "sensor rarely reads exactly zero for that long, so the instrument was probably off or below "
                "its limit, and these zeros are not measurements."
            )
            summary = f"{column}: exactly 0 for long stretches, {_plural(rows, 'row')} ({share})"
        else:
            detail = (
                f"{column} holds one unchanged value for long stretches, {_plural(rows, 'row')} ({share}) in "
                f"all; the longest holds {show_value(held[longest])} for {lengths[longest]:,} readings in a "
                f"row ({_stretch(frame, date, span)}). Its values otherwise change from one reading to the "
                "next, so the sensor probably stuck or stopped reporting there."
            )
            summary = f"{column}: stuck on one value for {_plural(rows, 'row')} ({share})"
        rule = (
            f"in time order, runs of {FLATLINE_RUN} or more exact zeros, or one run holding "
            f"{FLATLINE_STALL:.0%} of the column, in a column that otherwise repeats on at most "
            f"{FLATLINE_REPEATS:.0%} of rows"
        )
        cells = frame.index[np.concatenate([order[starts[k] : starts[k] + lengths[k]] for k in stuck])]
        issues.append(QualityIssue("flatline", (column,), rows, rows / profile.size * 100, detail, summary,
                                   rule, examples=(show_value(held[longest]),), cells={column: cells}))
    return _same_stretches(issues)


def _same_stretches(issues: list[QualityIssue]) -> list[QualityIssue]:
    """Fold stuck readings on exactly the same rows in several columns into one issue: one station down."""
    groups: dict[tuple, list[QualityIssue]] = {}
    for issue in issues:
        rows = next(iter(issue.cells.values()))
        groups.setdefault((len(rows), rows.min(), rows.max()), []).append(issue)
    merged = []
    for alike in groups.values():
        if len(alike) == 1:
            merged.append(alike[0])
            continue
        first = alike[0]
        columns = tuple(issue.columns[0] for issue in alike)
        named = _listed(list(columns))
        held = _listed([f"{issue.columns[0]} {issue.examples[0]}" for issue in alike])
        detail = (
            f"{named} stop changing on the same {_plural(first.rows, 'row')} ({_percent(first.share)}), "
            f"holding {held}: the instrument behind them probably stopped reporting there, so these "
            "rows are not measurements."
        )
        summary = f"{named} stuck on the same {_plural(first.rows, 'row')} ({_percent(first.share)})"
        merged.append(QualityIssue(
            "flatline", columns, first.rows, first.share, detail, summary, first.rule, first.examples,
            cells={name: index for issue in alike for name, index in issue.cells.items()},
        ))
    return merged


def _matrix_rows(frame: pd.DataFrame, columns: list[str], label: str | None) -> QualityIssue | None:
    """Rows of a matrix table whose overall level is far from the other rows'.

    A spectrum or a sample is judged as a whole, by its median across the
    matrix columns: a bright emission line is part of a spectrum, while a
    spectrum recorded on another scale is off in every channel.
    """
    if len(columns) < 2 or len(frame) < 12:
        return None
    medians = np.nanmedian(frame[columns].to_numpy(dtype="float64"), axis=1)
    levels = pd.Series(medians, index=frame.index).dropna()
    profile = _Column.of("row level", levels)
    if profile.uniques.size < 5:
        return None
    scale = _Scale.of(profile.ordered)
    distant = scale.far(profile.uniques) & scale.detached(profile.uniques)
    far = np.isin(profile.array, profile.uniques[distant])
    if not far.any() or far.mean() > EXTREME_SHARE:
        return None
    flagged = profile.cells(far)
    labels = frame.loc[flagged.index, label] if label in frame.columns else flagged.index
    names = [str(value) for value in labels]
    shown = _listed(names, limit=5)
    detail = (
        f"{_plural(flagged.size, 'row')} ({shown}) sit far from the others as a whole: the median across "
        f"the {len(columns):,} columns is {_range_of(flagged)} on {'it' if flagged.size == 1 else 'them'}, "
        f"against {show_value(scale.median)} for a typical row. Check whether they were measured on "
        "another scale or setting before comparing them with the rest."
    )
    summary = f"{_plural(flagged.size, 'row')} off scale as a whole ({_listed(names, limit=3)})"
    rule = (
        f"a row's median across the matrix columns more than {EXTREME_Z:.0f} robust standard deviations "
        "from the typical row's, cut off from the rest"
    )
    return QualityIssue("matrix_rows", (), int(flagged.size), flagged.size / len(frame) * 100, detail,
                        summary, rule, examples=tuple(names[:5]))


def _profiles(frame: pd.DataFrame, roles: ColumnRoles | None) -> dict[str, _Column]:
    """Sorted values of each numeric column that holds measurements.

    Identifiers, row labels, layout numbers, row counters (whole numbers
    unique on every row) and columns ADA added itself, such as a matrix
    table's row mean, are left out: none of them is a reading to check.
    """
    identifier = roles.identifier if roles else None
    profiles = {}
    for column in frame.columns:
        series = frame[column]
        if not is_numeric_dtype(series) or is_bool_dtype(series) or column == identifier:
            continue
        name = str(column)
        if is_design_name(name) or _BUILT_NAME.match(name):
            continue
        profile = _Column.of(name, series)
        # An empty column has no values to check; the gap checks report it.
        if profile.size == 0:
            continue
        if profile.size >= 20 and profile.uniques.size == profile.size and profile.whole:
            continue
        if _keys_or_labels(name, profile.uniques):
            continue
        profiles[column] = profile
    return profiles


def own_columns(frame: pd.DataFrame, roles: ColumnRoles | None) -> pd.DataFrame:
    """The frame without the columns that are empty throughout it, when it is one part of a stacked file.

    Once the brief covers one of several stacked tables, the other tables'
    columns are empty on every row; they are not gaps in this table, so the
    checks and the completeness figure leave them out. Any other table keeps
    every column, since a column empty throughout is a gap worth naming there.
    """
    if roles is None or roles.table_shape != "stacked":
        return frame
    recorded = frame.notna().any().to_numpy()
    return frame if recorded.all() else frame.loc[:, recorded]


def scan(
    frame: pd.DataFrame, roles: ColumnRoles | None = None, blanked: tuple[QualityIssue, ...] = ()
) -> tuple[QualityIssue, ...]:
    """Run every check and return the issues, the ones covering most rows first.

    Issues about values come before issues about gaps; within each, they are
    ranked by the rows (or cells) they cover, ties following ``CHECKS`` and
    the measure's first. ``roles`` names the measure, the date, the grouping
    column, the columns that hold codes and the table's shape; without it,
    codes are judged from the values alone and no group or time checks run.
    ``blanked`` holds the issues ``blank_codes`` returned when cleaning set
    codes to missing; they are no longer in ``frame``, so they are ranked with
    the rest here and the checks still list them.
    """
    frame = own_columns(frame, roles)
    profiles = _profiles(frame, roles)
    factors = tuple(roles.factors) if roles else ()
    measure = roles.measure if roles else None
    dimension = roles.dimension if roles else None
    date = roles.date if roles else None
    quantities = [column for column, profile in profiles.items() if not _is_coded(profile, factors)]
    matrix = set(quantities) if roles and roles.table_shape == "matrix" else set()
    limits = physical_limits({column: profiles[column] for column in quantities})
    tied = _tied_check(frame, roles)
    clips, clipped = _shared_clips(profiles, quantities)
    issues: list[QualityIssue] = [clipped] if clipped else []
    for column, profile in profiles.items():
        bounds = column_bounds(str(column), limits)
        issues.extend(_value_issues(
            profile, factors, bounds, tied, single_cells=column not in matrix,
            shared=clips.get(column, frozenset()),
        ))
    singles = [column for column in quantities if column not in matrix]
    issues.extend(_group_caps(frame, profiles, singles, _labels(frame, roles)))
    # Limits each column piles at, which a stuck sensor's long run must not be mistaken for.
    piled: dict[str, set[float]] = {column: set(found) for column, found in clips.items()}
    for issue in issues:
        if issue.check in ("cap", "clipped"):
            for column in issue.columns:
                piled.setdefault(column, set()).update(_pile_values([issue]))
    issues = [*_merge_alike(_one_sample_fills(frame, issues)), *blanked]
    swaps = _swapped(frame, profiles, [column for column in quantities if column not in matrix], date)
    # A row of zeros explains the extreme values on it, as a swap explains its values.
    zeros = _zero_rows(frame, profiles, singles)
    explaining = swaps + ([zeros] if zeros else [])
    issues = _cluster_runs(_without_swapped(issues, explaining), frame, date) + explaining
    issues.extend(_failed_records(frame, issues, singles))
    issues.extend(_placeholder_blocks(frame, roles.identifier if roles else None))
    wide = [column for column in quantities if profiles[column].uniques.size >= 10]
    issues.extend(_equal_columns(frame, wide, dimension, measure))
    issues.extend(_min_max(frame, quantities))
    issues.extend(_parts_of_wholes(frame, singles))
    grouping = pd.factorize(frame[dimension]) if dimension in frame.columns else None
    if measure in quantities and grouping is not None:
        found = _group_levels(frame, measure, dimension, grouping, date)
        if found:
            issues.append(found)
    if matrix:
        off = _matrix_rows(frame, sorted(matrix, key=list(frame.columns).index), roles.identifier)
        if off:
            issues.append(off)
    readings = [column for column in quantities if column not in matrix]
    issues.extend(_flatlines(frame, profiles, readings, date, dimension, piled))
    duplicated = _duplicates(frame, profiles)
    if duplicated:
        issues.append(duplicated)
    generated = _generated(frame, profiles, quantities, _flagged_cells(issues))
    if generated:
        issues.append(generated)
    for found in (_zero_heavy(profiles, singles), _constant_columns(frame, roles)):
        if found:
            issues.append(found)
    issues.extend(_missing_issues(frame, quantities, dimension, grouping))
    order = {check: position for position, check in enumerate(CHECKS)}
    return tuple(sorted(
        issues,
        key=lambda issue: (
            2 if issue.check.startswith("missing") else 1 if issue.check in INFO_CHECKS else 0,
            -issue.rows, order.get(issue.check, len(order)), measure not in issue.columns,
        ),
    ))


def _flagged_cells(issues: list[QualityIssue]) -> dict[str, pd.Index]:
    """Row labels of every cell a check on values flagged, by column."""
    cells: dict[str, pd.Index] = {}
    for issue in issues:
        if issue.check in VALUE_CHECKS | MASKED_CHECKS:
            for column, labels in issue.cells.items():
                cells[column] = cells[column].union(labels) if column in cells else labels
    return cells


def statistics_mask(frame: pd.DataFrame, issues: tuple[QualityIssue, ...]) -> pd.DataFrame:
    """True for each cell the checks found can't be a reading as it stands.

    That is an impossible value, a missing-value code or likely entry error, a
    filled-in gap, a survey code, a value in a swapped column, a stuck
    reading, the rest of a failed record, a row of zeros or a placeholder
    (``MASKED_CHECKS``). Extreme values and second clusters may be real, so
    they are not marked.
    """
    mask = pd.DataFrame(False, index=frame.index, columns=frame.columns)
    for issue in issues:
        if issue.check not in MASKED_CHECKS:
            continue
        for column, labels in issue.cells.items():
            if column in mask.columns:
                mask[column] |= frame.index.isin(labels)
    return mask


def statistics_frame(frame: pd.DataFrame, issues: tuple[QualityIssue, ...]) -> tuple[pd.DataFrame, str]:
    """The frame with the cells ``statistics_mask`` marks set to missing, and a note saying how many.

    Relationships and group comparisons are computed on it, so a code such
    as 999.9 or a column's filled-in mean doesn't drive a correlation. The
    frame comes back unchanged, with an empty note, when nothing is marked.
    """
    kinds = {issue.check for issue in issues if issue.check in MASKED_CHECKS and issue.cells}
    if not kinds:
        return frame, ""
    mask = statistics_mask(frame, issues)
    count = int(mask.to_numpy().sum())
    if not count:
        return frame, ""
    named = {
        "impossible": "impossible values", "sentinel": "missing-value codes", "imputed": "fill values",
        "survey_code": "survey codes", "swapped": "swapped values", "flatline": "stuck readings",
        "failed_record": "failed records", "zero_rows": "rows of zeros", "placeholder": "placeholders",
    }
    what = _listed([label for kind, label in named.items() if kind in kinds])
    cleaned = frame.copy()
    for column in mask.columns[mask.any().to_numpy()]:
        cleaned[column] = frame[column].mask(mask[column])
    noun = "cell" if count == 1 else "cells"
    return cleaned, f"{count:,} {noun} the data checks flagged ({what}) left out"


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


def _clear_codes(column: _Column) -> tuple[list[float], float]:
    """Standing codes that no reading could be, and the median of the other values.

    Only an unmistakable code qualifies: a run of nines, a repeated digit or a
    large negative power of ten, standing alone, at least EXTREME_Z robust
    standard deviations from the median, and either negative in a column that
    is otherwise not or ten times the 99th percentile of the rest.
    """
    if column.size < 12:
        return [], 0.0
    standing, centre = _standing_codes(column, column.ordered)
    if not standing:
        return [], centre
    rest = column.ordered[~np.isin(column.ordered, standing)]
    ceiling = float(np.percentile(np.abs(rest), 99))
    clear = [
        value
        for value in standing
        if (value < 0 and (rest < 0).mean() <= 0.01) or abs(value) >= 10 * max(ceiling, 1e-12)
    ]
    return clear, centre


def _shared_negatives(arrays: dict[str, np.ndarray]) -> dict[float, tuple[list[str], np.ndarray]]:
    """Negative values that SHARED_CODE_COLUMNS or more columns hold on the same rows.

    Each column must be otherwise never negative and hold readings rather than
    a few whole-number codes (-1, 0 and 1 for disagree, neutral and agree).
    Returns, for each value, the columns and the rows where at least
    SHARED_CODE_COLUMNS of them hold it.
    """
    candidates: dict[float, list[str]] = {}
    for name, array in arrays.items():
        present = array[np.isfinite(array)]
        negatives = pd.unique(present[present < 0])
        if negatives.size != 1:
            continue
        if bool(np.all(present == np.round(present))) and pd.unique(present).size <= CODED_LEVELS:
            continue
        candidates.setdefault(float(negatives[0]), []).append(name)
    found: dict[float, tuple[list[str], np.ndarray]] = {}
    for value, names in candidates.items():
        if len(names) < SHARED_CODE_COLUMNS:
            continue
        hits = np.column_stack([arrays[name] == value for name in names])
        shared = hits.sum(axis=1) >= SHARED_CODE_COLUMNS
        kept = [name for name, column in zip(names, hits.T, strict=True) if (column & shared).any()]
        if len(kept) >= SHARED_CODE_COLUMNS:
            found[value] = (kept, shared)
    return found


def _cells(count: int) -> str:
    return "1 cell" if count == 1 else f"{count:,} cells"


def _standing_issue(value: float, counts: dict[str, tuple[int, float, int]]) -> QualityIssue:
    """The data-check issue for one standing code set to missing in one or more columns.

    ``counts`` gives, by column, the cells set to missing, the median of the
    other values and how many values the column had.
    """
    columns = tuple(counts)
    cells = sum(count for count, _, _ in counts.values())
    share = max(count / max(size, 1) * 100 for count, _, size in counts.values())
    shown = show_value(value)
    if len(columns) == 1:
        count, centre, _ = counts[columns[0]]
        detail = (
            f"{columns[0]} held {shown} in {_plural(count, 'row')}, far from its other values (median "
            f"{show_value(centre)}): a code for a missing reading. Cleaning set those cells to missing."
        )
        summary = f"{columns[0]}: {shown} in {_plural(count, 'row')}, a missing-value code set to missing"
    else:
        sizes = sorted(count for count, _, _ in counts.values())
        spread = f"{sizes[0]:,}" if sizes[0] == sizes[-1] else f"{sizes[0]:,} to {sizes[-1]:,}"
        detail = (
            f"{_listed(list(columns))} held {shown} in {cells:,} cells ({spread} per column), far from "
            "their other values: a code for a missing reading. Cleaning set those cells to missing."
        )
        summary = f"{_listed(list(columns))}: {shown} in {_cells(cells)}, a missing-value code set to missing"
    rule = (
        "a code value (a run of nines, a repeated digit, a large negative power of ten) standing alone, "
        "negative in a column otherwise not or ten times the rest's 99th percentile; set to missing"
    )
    return QualityIssue("sentinel", columns, cells, share, detail, summary, rule, (shown,), group=value)


def _shared_issue(value: float, columns: list[str], cells: int, rows: int, total: int) -> QualityIssue:
    """The data-check issue for one negative value set to missing where several columns share it."""
    shown = show_value(value)
    named = _listed(columns)
    detail = (
        f"{named} held {shown} in {_cells(cells)} on {_plural(rows, 'row')}, each row holding it in "
        f"{SHARED_CODE_COLUMNS} or more of them, though none is otherwise negative: a code for no "
        "reading. Cleaning set those cells to missing."
    )
    summary = f"{named}: {shown} on {_plural(rows, 'row')} at once, a missing-value code set to missing"
    rule = (
        f"one negative value on the same rows in {SHARED_CODE_COLUMNS} or more columns that are "
        "otherwise never negative; set to missing"
    )
    return QualityIssue(
        "sentinel", tuple(columns), rows, rows / max(total, 1) * 100, detail, summary, rule, (shown,),
        group=value,
    )


def blank_codes(frame: pd.DataFrame) -> tuple[pd.DataFrame, tuple[QualityIssue, ...], tuple[str, ...]]:
    """Set clear missing-value codes to missing in every numeric column.

    Two kinds go. One is a standing code (see ``_clear_codes``): -999 among
    pressures near 950, 999.9 among wind speeds below 20. The other is one
    negative value that SHARED_CODE_COLUMNS or more otherwise non-negative
    columns hold on the same rows, as a logger writes -1 in every channel when
    a read fails. Left in, either drives every average and correlation it
    touches. Keys and layout numbers are left alone, and everything else stays
    and is reported by ``scan``.

    Returns the frame (the same object when nothing qualifies), the issues the
    data checks list for what was set to missing, and one note per column.
    """
    arrays: dict[str, np.ndarray] = {}
    for column in frame.columns:
        series = frame[column]
        if not is_numeric_dtype(series) or is_bool_dtype(series):
            continue
        array = series.to_numpy(dtype="float64", na_value=np.nan)
        present = array[np.isfinite(array)]
        if present.size and not _keys_or_labels(str(column), present):
            arrays[column] = array
    blank: dict[str, np.ndarray] = {}
    standing: dict[float, dict[str, tuple[int, float, int]]] = {}
    notes: list[str] = []
    for column, array in arrays.items():
        # Profiling sorts the column, so only a column holding a code value is profiled.
        if not _is_code(array[np.isfinite(array)]).any():
            continue
        profile = _Column.of(str(column), frame[column])
        clear, centre = _clear_codes(profile)
        if not clear:
            continue
        hits = np.isin(array, clear)
        blank[column] = hits
        for value in clear:
            standing.setdefault(value, {})[column] = (int((array == value).sum()), centre, profile.size)
        count = int(hits.sum())
        kind = "codes" if len(clear) > 1 else "a code"
        notes.append(
            f"{_cells(count)} in {column} held {' or '.join(show_value(value) for value in clear)}, {kind} "
            f"for a missing reading far from every other value (median {show_value(centre)}), and "
            f"{'were' if count > 1 else 'was'} set to missing."
        )
    issues = [_standing_issue(value, counts) for value, counts in standing.items()]
    left = {
        column: np.where(blank[column], np.nan, array) if column in blank else array
        for column, array in arrays.items()
    }
    for value, (columns, rows) in _shared_negatives(left).items():
        hits = {column: (left[column] == value) & rows for column in columns}
        for column in columns:
            others = [other for other in columns if other != column and (hits[other] & hits[column]).any()]
            named = _listed(others) if len(others) <= 3 else f"{len(others)} other columns"
            blank[column] = blank.get(column, np.zeros(len(frame), dtype=bool)) | hits[column]
            count = int(hits[column].sum())
            notes.append(
                f"{_cells(count)} in {column} held {show_value(value)} on rows where {named}, none of them "
                "otherwise negative, hold it too: a code for no reading, so "
                f"{'they were' if count > 1 else 'it was'} set to missing."
            )
        cells = sum(int(mask.sum()) for mask in hits.values())
        issues.append(_shared_issue(value, columns, cells, int(rows.sum()), len(frame)))
    if not blank:
        return frame, (), ()
    masked = frame.copy()
    for column, hits in blank.items():
        masked[column] = masked[column].mask(hits)
    return masked, tuple(issues), tuple(notes)
