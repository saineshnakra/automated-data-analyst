"""Elapsed time as an axis: profiles over a time-point column, and columns that form a series.

Many experiments record time as a number rather than a date: days since
infection, minutes after a stimulus, the percentage of a gait cycle, a visit
number. Such a column orders the rows, so the measure is shown per time point
in that order, with its peak and trough, instead of being ranked by value or
compared as if it were an outcome. Groups followed for different lengths of
time are compared over the time they share. Numbered columns in an
arithmetic progression (40, 50, ... 260 minutes) are a series laid out wide
and get the same treatment across the columns.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .compare import benjamini_hochberg
from .schema import ColumnRoles

# Words naming elapsed time or a point in a sequence, as the first word of a
# name ("timept", "Day", "hours_post_dose"). Abbreviations that also mean
# something else are left out: "min" is a minimum, "hr" a heart rate, "T" a total.
TIMEPOINT_WORDS = frozenset({
    "time", "timept", "timepoint", "day", "days", "hour", "hours", "minute", "minutes", "seconds", "week",
    "weeks", "visit", "cycle", "phase", "session", "dpi", "hpi", "elapsed",
})
# Words that make any time word in a name elapsed time: "days_since_infection", "hours after".
SINCE_WORDS = frozenset({"since", "post", "after", "elapsed"})
# A time-point axis repeats each value across units, so it has at most this many
# distinct values, at least MIN_TIMEPOINTS, and each value on several rows.
MAX_TIMEPOINTS = 400
MIN_TIMEPOINTS = 4
MIN_ROWS_PER_TIMEPOINT = 2
# Groups compared across the axis: few enough to name.
MAX_PROFILE_GROUPS = 6
# Numbered columns form a series when at least this many are evenly spaced.
MIN_SERIES_COLUMNS = 5
# Periods tried for a repeating pattern across a series, and the false-discovery
# rate for calling a row's fit clear.
PERIOD_CANDIDATES = 40
FALSE_DISCOVERY_RATE = 0.05
SERIES_ROWS_NAMED = 3


def _name_words(name: str) -> list[str]:
    spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", str(name))
    return re.findall(r"[a-z]+", spaced.casefold())


def is_timepoint_name(name: str) -> bool:
    """Whether a column name marks elapsed time or a point in a sequence ("timept", "Day", "hours_post").

    The time word must open the name, or come with a word such as "since":
    "drinks_day" is a rate and "welding speed (mm/min)" a speed.
    """
    words = _name_words(name)
    if not words:
        return False
    if words[0] in TIMEPOINT_WORDS or words[0].startswith("time"):
        return True
    return bool(set(words) & SINCE_WORDS) and bool(set(words) & TIMEPOINT_WORDS)


def timepoint_column(dataframe: pd.DataFrame, roles: ColumnRoles) -> str | None:
    """A numeric column of time points that orders the rows, when the table has no date.

    It is named like elapsed time, holds MIN_TIMEPOINTS to MAX_TIMEPOINTS
    non-negative values, and repeats each across rows, as when several
    subjects are measured on each day. Returns None with a date column, since
    a "day" beside a date is a part of it, and skips the columns the schema
    set aside as parts of a date held in their own columns (a day beside a
    year and a month column).
    """
    if roles.date:
        return None
    set_aside = set(getattr(roles, "non_readings", ()) or ())
    for column in dataframe.columns:
        if column == roles.measure or column in set_aside or not is_timepoint_name(column):
            continue
        values = dataframe[column]
        if not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
            continue
        present = values.dropna()
        distinct = present.nunique()
        if (
            MIN_TIMEPOINTS <= distinct <= MAX_TIMEPOINTS
            and float(present.min()) >= 0
            and len(present) >= MIN_ROWS_PER_TIMEPOINT * distinct
        ):
            return column
    return None


@dataclass(frozen=True)
class GroupReach:
    """How far each group was followed along the axis, and the groups compared over the shared stretch."""

    grouping: str
    reach: tuple[tuple[object, float, float], ...]
    shared: tuple[float, float]
    #: Each group's mean of the measure over the shared stretch, and over all its rows.
    shared_means: tuple[tuple[object, float], ...]
    pooled_means: tuple[tuple[object, float], ...]


@dataclass(frozen=True)
class Profile:
    """The measure's mean at each time point, in order."""

    axis: str
    measure: str
    means: pd.Series
    #: Kendall's tau of the means against time and its p-value.
    tau: float
    p_value: float
    reach: GroupReach | None = None


def time_profile(dataframe: pd.DataFrame, roles: ColumnRoles, axis: str) -> Profile | None:
    """The measure's profile over a time-point axis, and how far each group was followed.

    Returns None without a measure or with fewer than MIN_TIMEPOINTS time
    points holding a value.
    """
    if not roles.measure or roles.measure == axis:
        return None
    working = dataframe[[axis, roles.measure]].dropna()
    means = working.groupby(axis)[roles.measure].mean().sort_index()
    if len(means) < MIN_TIMEPOINTS:
        return None
    tau, p_value = stats.kendalltau(means.index.to_numpy(dtype=float), means.to_numpy(dtype=float))
    return Profile(
        axis=axis,
        measure=roles.measure,
        means=means,
        tau=float(tau) if np.isfinite(tau) else 0.0,
        p_value=float(p_value) if np.isfinite(p_value) else 1.0,
        reach=_group_reach(dataframe, roles, axis),
    )


def _group_reach(dataframe: pd.DataFrame, roles: ColumnRoles, axis: str) -> GroupReach | None:
    """Each group's first and last time point, when the groups were followed for different lengths."""
    grouping = roles.dimension
    if not grouping or grouping == axis:
        return None
    working = dataframe[[axis, roles.measure, grouping]].dropna()
    spans = working.groupby(grouping)[axis].agg(["min", "max"])
    same_reach = spans["max"].nunique() == 1 and spans["min"].nunique() == 1
    if not 2 <= len(spans) <= MAX_PROFILE_GROUPS or same_reach:
        return None
    low, high = float(spans["min"].max()), float(spans["max"].min())
    if low >= high:
        return None
    inside = working[working[axis].between(low, high)]
    shared = inside.groupby(grouping)[roles.measure].mean()
    pooled = working.groupby(grouping)[roles.measure].mean()
    return GroupReach(
        grouping=grouping,
        reach=tuple((group, float(row["min"]), float(row["max"])) for group, row in spans.iterrows()),
        shared=(low, high),
        shared_means=tuple((group, float(value)) for group, value in shared.items()),
        pooled_means=tuple((group, float(value)) for group, value in pooled.items()),
    )


def _as_number(name: object) -> float | None:
    try:
        return float(str(name).strip())
    except ValueError:
        return None


def series_columns(dataframe: pd.DataFrame) -> tuple[list[str], np.ndarray] | None:
    """Numeric columns named by evenly spaced numbers (40, 50, ... 260), in order, with their positions.

    At least MIN_SERIES_COLUMNS of them are needed. Returns None otherwise.
    """
    named = []
    for column in dataframe.columns:
        number = _as_number(column)
        if number is not None and pd.api.types.is_numeric_dtype(dataframe[column]):
            named.append((number, column))
    if len(named) < MIN_SERIES_COLUMNS:
        return None
    named.sort()
    positions = np.array([number for number, _ in named])
    steps = np.diff(positions)
    if steps.min() <= 0 or not np.allclose(steps, steps[0], rtol=0.01):
        return None
    return [column for _, column in named], positions


@dataclass(frozen=True)
class SeriesSummary:
    """A wide series: its mean profile, the rows that vary most, and the period that fits best."""

    columns: tuple[str, ...]
    positions: tuple[float, ...]
    profile: pd.Series
    #: Rows with the largest standard deviation across the series, as (label, SD).
    varied: tuple[tuple[str, float], ...]
    rows: int
    period: float | None = None
    median_fit: float = 0.0
    clear_rows: int = 0


def _row_labels(dataframe: pd.DataFrame, roles: ColumnRoles, columns: list[str]) -> pd.Series:
    """What to call each row: the identifier, a text column that names every row once, or its number."""
    if roles.identifier and roles.identifier in dataframe.columns:
        return dataframe[roles.identifier].astype(str)
    for column in dataframe.columns:
        values = dataframe[column]
        if column not in columns and not pd.api.types.is_numeric_dtype(values) and values.is_unique:
            return values.astype(str)
    return pd.Series([f"row {index + 1}" for index in range(len(dataframe))], index=dataframe.index)


def summarize_series(
    dataframe: pd.DataFrame, roles: ColumnRoles, columns: list[str], positions: np.ndarray
) -> SeriesSummary:
    """Summarise a series laid out across columns, respecting its order.

    The profile is each column's mean. Rows are ranked by their standard
    deviation across the series. A repeating pattern is looked for by least
    squares on a sine and cosine at PERIOD_CANDIDATES periods between two
    steps and the whole span, all rows at once. A row's fit counts as clear
    by its F test at a FALSE_DISCOVERY_RATE (Benjamini-Hochberg), and the
    period with the most clear rows is kept, then the one with the highest
    mean R², so a pattern in a minority of rows is not lost in the noise of
    the rest.
    """
    values = dataframe[columns].to_numpy(dtype=float)
    complete = np.isfinite(values).all(axis=1)
    labels = _row_labels(dataframe, roles, columns)
    profile = pd.Series(np.nanmean(values, axis=0), index=positions)
    spread = np.nanstd(values, axis=1, ddof=1)
    order = np.argsort(np.where(np.isfinite(spread), -spread, np.inf))[:SERIES_ROWS_NAMED]
    varied = tuple(
        (str(labels.iloc[index]), float(spread[index])) for index in order if np.isfinite(spread[index])
    )
    rows = values[complete]
    places = tuple(float(position) for position in positions)
    summary = SeriesSummary(tuple(columns), places, profile, varied, int(complete.sum()))
    count = len(positions)
    if len(rows) < 2 or count < 2 * 3:
        return summary
    step = float(positions[1] - positions[0])
    span = float(positions[-1] - positions[0])
    centred = rows - rows.mean(axis=1, keepdims=True)
    total = (centred**2).sum(axis=1)
    usable = total > 0
    if not usable.any():
        return summary
    best = None
    for period in np.linspace(2 * step, span, PERIOD_CANDIDATES):
        angle = 2 * np.pi * (positions - positions[0]) / period
        design = np.column_stack([np.sin(angle), np.cos(angle)])
        design = design - design.mean(axis=0)
        coefficients, *_ = np.linalg.lstsq(design, centred.T, rcond=None)
        fitted = (design @ coefficients).T
        residual = ((centred - fitted) ** 2).sum(axis=1)
        explained = (1 - residual / np.where(usable, total, 1))[usable]
        statistic = (explained / 2) / np.maximum((1 - explained) / (count - 3), 1e-12)
        clear = int((benjamini_hochberg(stats.f.sf(statistic, 2, count - 3)) < FALSE_DISCOVERY_RATE).sum())
        # The period most rows follow clearly, then the one that explains most on average.
        score = (clear, float(explained.mean()))
        if best is None or score > best[0]:
            best = (score, float(period), float(np.median(explained)))
    (clear, _), period, median = best
    return SeriesSummary(
        tuple(columns), places, profile, varied, int(complete.sum()),
        period=period, median_fit=median, clear_rows=clear,
    )
