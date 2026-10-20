"""A slice the reader picked on a chart, and how it compares to the whole.

A selection is a filter, not a finding. The reader drew a box, a lasso or
clicked some bars; ADA turns that gesture into a filter it can state in words,
analyzes the rows inside it with the same evidence machinery as everything
else, and puts every figure beside the same figure for the whole view. Nothing
here suggests ADA discovered a grouping -- the reader chose it.

Nothing here draws anything or knows about Streamlit. The chart's selection
event arrives as a plain mapping, so the translation from gesture to filter is
testable without a browser.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

from .aggregation import preferred_frequency
from .autovis import ChartSpec
from .formatting import format_number, format_percentage
from .metrics import resolve_metric
from .nlq import ValueFilter
from .schema import ColumnRoles

# Below this many records an average, or a share of a segment mix, is mostly
# noise: one unusual row moves it by a fifth. The count and the total are still
# arithmetic and are still reported; the statistics are not.
MIN_SLICE_RECORDS = 5

# Which gestures each chart form accepts. A heatmap cell, a table row and a
# single stat have no selection that maps back to rows honestly. A chart takes
# either clicks or a drawn area, not both: while a box or lasso tool is active
# the chart reports clicks as plain events, not selections.
SELECTION_MODES: dict[str, tuple[str, ...]] = {
    "scatter": ("box", "lasso"),
    "line": ("box",),
    "area": ("box",),
    "column": ("points",),
    "bar": ("points",),
    "histogram": ("box",),
}


@dataclass(frozen=True)
class RangeFilter:
    """Rows whose value in ``column`` lies between two bounds, inclusive."""

    column: str
    low: Any
    high: Any


@dataclass(frozen=True)
class RegionFilter:
    """Rows whose (x, y) point lies inside a lasso the reader drew."""

    x: str
    y: str
    vertices: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class Selection:
    """The filters a chart gesture stands for. Empty means nothing is selected."""

    values: tuple[ValueFilter, ...] = ()
    ranges: tuple[RangeFilter, ...] = ()
    region: RegionFilter | None = None
    method: str = ""  # box | lasso | click
    # Set when the drawn window was widened to whole periods, saying from what.
    adjustment: str = ""

    @property
    def is_empty(self) -> bool:
        return not (self.values or self.ranges or self.region)


@dataclass(frozen=True)
class SliceStat:
    """One figure for the slice, the same figure for the whole, and how it was computed."""

    label: str
    slice_value: str
    whole_value: str
    difference: str
    calculation: str


@dataclass(frozen=True)
class SliceComparison:
    description: str
    records: int
    whole_records: int
    stats: tuple[SliceStat, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def supports_statistics(self) -> bool:
        return self.records >= MIN_SLICE_RECORDS and self.records < self.whole_records


def selection_modes(spec: ChartSpec) -> tuple[str, ...]:
    """The gestures a chart of this form can turn into a filter."""
    return SELECTION_MODES.get(spec.form, ())


# --- from a chart event to a selection ---------------------------------------


def _latest(entries: Any) -> Mapping | None:
    """The most recent box or lasso. Plotly keeps earlier ones while shift is held."""
    if not entries:
        return None
    entry = entries[-1]
    return entry if isinstance(entry, Mapping) else None


def _bound(dataframe: pd.DataFrame, column: str, value: Any) -> Any:
    """Read one coordinate the way the column stores its values."""
    series = dataframe[column]
    if is_datetime64_any_dtype(series):
        stamp = pd.Timestamp(value)
        tz = getattr(series.dt, "tz", None)
        if tz is not None and stamp.tzinfo is None:
            stamp = stamp.tz_localize(tz)
        return stamp
    return float(value)


def _range(dataframe: pd.DataFrame, column: str | None, values: Sequence[Any]) -> RangeFilter | None:
    if not column or column not in dataframe.columns or len(values) < 2:
        return None
    try:
        bounds = [_bound(dataframe, column, value) for value in values]
    except (TypeError, ValueError):
        return None
    return RangeFilter(column=column, low=min(bounds), high=max(bounds))


GRAIN_NAMES = {"W": "weeks", "M": "months", "Q": "quarters", "Y": "years"}


def _whole_periods(dataframe: pd.DataFrame, window: RangeFilter) -> tuple[RangeFilter, str]:
    """Widen a time window to the whole periods the slice will be analyzed in.

    A window drawn from the 19th of one month to the 8th of another leaves both
    end months part-covered, and a month with a third of its days reads as a
    collapse to every trend and anomaly check downstream. The grain is the one
    ``aggregation`` picks for the rows inside the window, so the slice and its
    trend agree on what a period is. The widening is reported, never silent.
    """
    column = dataframe[window.column]
    tz = getattr(column.dt, "tz", None)
    naive = column.dt.tz_localize(None) if tz is not None else column
    low = window.low.tz_localize(None) if window.low.tzinfo else window.low
    high = window.high.tz_localize(None) if window.high.tzinfo else window.high
    start, end, grain = low, high, None
    for _ in range(len(GRAIN_NAMES)):
        inside = naive[(naive >= start) & (naive <= end)].dropna()
        if inside.empty:
            return window, ""
        chosen = preferred_frequency(inside)
        if chosen == grain or chosen not in GRAIN_NAMES:
            break
        grain = chosen
        # A coarser grain contains the finer one, so widening again can only
        # settle, and never shrinks what the reader drew.
        start = min(start, pd.Period(low, freq=grain).start_time)
        end = max(end, pd.Period(high, freq=grain).end_time)
    if grain is None or (start == low and end == high):
        return window, ""
    if tz is not None:
        start, end = start.tz_localize(tz), end.tz_localize(tz)
    widened = RangeFilter(window.column, start, end)
    values = dataframe[window.column]
    drawn = (
        f"{_format_bound(window.low, window.column, values)} to "
        f"{_format_bound(window.high, window.column, values)}"
    )
    note = (
        f"Widened to whole {GRAIN_NAMES[grain]} from the {drawn} you selected, so a part-covered "
        f"period at either end cannot read as a rise or a fall."
    )
    return widened, note


def _time_selection(dataframe: pd.DataFrame, window: RangeFilter, method: str) -> Selection:
    if is_datetime64_any_dtype(dataframe[window.column]):
        window, adjustment = _whole_periods(dataframe, window)
        return Selection(ranges=(window,), method=method, adjustment=adjustment)
    return Selection(ranges=(window,), method=method)


def _category_column(spec: ChartSpec) -> tuple[str | None, str]:
    """Which column holds the clicked labels, and which axis carries them."""
    if spec.form == "bar":  # horizontal: categories run down the y axis
        return spec.x, "y"
    return spec.x, "x"


def selection_from_event(
    event: Mapping[str, Any] | None, spec: ChartSpec, dataframe: pd.DataFrame
) -> Selection:
    """Translate a Plotly selection event into filters over the raw rows.

    The geometry the reader drew is used, not the indices of the plotted
    points: a trend or a bar chart plots aggregates, so a point index says
    nothing about which rows are behind it, while "dates from March to June"
    does, and it is also what the reader can check against the axis.
    """
    if not event or spec.form not in SELECTION_MODES:
        return Selection()
    points = list(event.get("points") or [])
    box = _latest(event.get("box"))
    lasso = _latest(event.get("lasso"))

    if spec.form == "scatter":
        if lasso and spec.x and spec.y:
            xs, ys = list(lasso.get("x") or []), list(lasso.get("y") or [])
            if len(xs) >= 3 and len(xs) == len(ys):
                try:
                    vertices = tuple((float(x), float(y)) for x, y in zip(xs, ys, strict=True))
                except (TypeError, ValueError):
                    vertices = ()
                if vertices:
                    return Selection(region=RegionFilter(spec.x, spec.y, vertices), method="lasso")
        if box:
            ranges = tuple(
                item
                for item in (
                    _range(dataframe, spec.x, box.get("x") or []),
                    _range(dataframe, spec.y, box.get("y") or []),
                )
                if item
            )
            if ranges:
                return Selection(ranges=ranges, method="box")
        return Selection()

    if spec.form in ("line", "area", "histogram"):
        # Only the horizontal extent maps to rows. The vertical extent of a box
        # drawn over a trend is a range of period totals, not of records.
        if box:
            window = _range(dataframe, spec.x, box.get("x") or [])
            if window:
                return _time_selection(dataframe, window, "box")
        if points and spec.form != "histogram":
            clicked = [point.get("x") for point in points if point.get("x") is not None]
            if len(clicked) == 1:
                clicked = clicked * 2
            window = _range(dataframe, spec.x, clicked)
            if window:
                return _time_selection(dataframe, window, "click")
        return Selection()

    # Bars: whatever bars the click or the box touched.
    column, axis = _category_column(spec)
    if not column or column not in dataframe.columns:
        return Selection()
    labels: list[str] = []
    for point in points:
        label = point.get(axis)
        if label is not None and str(label) not in labels:
            labels.append(str(label))
    present = set(dataframe[column].dropna().astype(str))
    labels = [label for label in labels if label in present]
    if not labels:
        return Selection()
    method = "box" if box else "click"
    return Selection(values=(ValueFilter(column=column, values=tuple(labels)),), method=method)


# --- applying it -------------------------------------------------------------


def _inside(xs: np.ndarray, ys: np.ndarray, vertices: Sequence[tuple[float, float]]) -> np.ndarray:
    """Even-odd ray casting: whether each point lies inside the polygon."""
    inside = np.zeros(len(xs), dtype=bool)
    count = len(vertices)
    for index in range(count):
        x1, y1 = vertices[index]
        x2, y2 = vertices[(index + 1) % count]
        if y1 == y2:
            continue
        crosses = (y1 > ys) != (y2 > ys)
        at_x = (x2 - x1) * (ys - y1) / (y2 - y1) + x1
        inside ^= crosses & (xs < at_x)
    return inside


def selection_mask(dataframe: pd.DataFrame, selection: Selection) -> pd.Series:
    """True for every row inside the selection. A row missing a selected value is outside it."""
    mask = pd.Series(True, index=dataframe.index)
    for value_filter in selection.values:
        if value_filter.column not in dataframe.columns:
            raise ValueError(f"Unknown selection column: {value_filter.column}")
        column = dataframe[value_filter.column]
        mask &= column.notna() & column.astype(str).isin(value_filter.values)
    for window in selection.ranges:
        if window.column not in dataframe.columns:
            raise ValueError(f"Unknown selection column: {window.column}")
        column = dataframe[window.column]
        mask &= column.notna() & (column >= window.low) & (column <= window.high)
    if selection.region is not None:
        region = selection.region
        for name in (region.x, region.y):
            if name not in dataframe.columns:
                raise ValueError(f"Unknown selection column: {name}")
        xs = pd.to_numeric(dataframe[region.x], errors="coerce").to_numpy(dtype=float)
        ys = pd.to_numeric(dataframe[region.y], errors="coerce").to_numpy(dtype=float)
        present = ~(np.isnan(xs) | np.isnan(ys))
        mask &= pd.Series(present & _inside(xs, ys, region.vertices), index=dataframe.index)
    return mask


def apply_selection(dataframe: pd.DataFrame, selection: Selection) -> pd.DataFrame:
    """The rows inside the selection. An empty selection returns the frame untouched."""
    if selection.is_empty:
        return dataframe
    return dataframe.loc[selection_mask(dataframe, selection)].reset_index(drop=True)


# --- saying it in words ------------------------------------------------------


def _format_bound(value: Any, column: str, values: pd.Series) -> str:
    if isinstance(value, pd.Timestamp):
        has_time = bool((values.dropna().dt.normalize() != values.dropna()).any()) if len(values) else False
        return value.strftime("%d %b %Y %H:%M" if has_time else "%d %b %Y")
    return format_number(float(value), column, compact=False, column_values=values)


def _join_labels(labels: Sequence[str], limit: int = 4) -> str:
    shown = list(labels[:limit])
    rest = len(labels) - len(shown)
    if rest:
        return f"{', '.join(shown)} or {rest} more"
    if len(shown) == 1:
        return shown[0]
    # "Region is West or South": a row has one value, so the values are alternatives.
    return f"{', '.join(shown[:-1])} or {shown[-1]}"


def describe_filters(selection: Selection, dataframe: pd.DataFrame) -> list[str]:
    """One clause per filter, in the column's own units, checkable against the axis."""
    clauses: list[str] = []
    for value_filter in selection.values:
        clauses.append(f"{value_filter.column} is {_join_labels(value_filter.values)}")
    for window in selection.ranges:
        values = dataframe[window.column] if window.column in dataframe.columns else pd.Series(dtype=float)
        low = _format_bound(window.low, window.column, values)
        high = _format_bound(window.high, window.column, values)
        clauses.append(
            f"{window.column} {low}" if low == high else f"{window.column} from {low} to {high}"
        )
    if selection.region is not None:
        region = selection.region
        xs = [vertex[0] for vertex in region.vertices]
        ys = [vertex[1] for vertex in region.vertices]
        x_values = dataframe[region.x] if region.x in dataframe.columns else None
        y_values = dataframe[region.y] if region.y in dataframe.columns else None
        clauses.append(
            f"inside the lasso you drew over {region.x} "
            f"{format_number(min(xs), region.x, compact=False, column_values=x_values)}–"
            f"{format_number(max(xs), region.x, compact=False, column_values=x_values)} and {region.y} "
            f"{format_number(min(ys), region.y, compact=False, column_values=y_values)}–"
            f"{format_number(max(ys), region.y, compact=False, column_values=y_values)}"
        )
    return clauses


def describe_selection(selection: Selection, whole: pd.DataFrame, part: pd.DataFrame) -> str:
    """State the slice in words, so a screenshot of it still explains itself.

    For example: 62 of 400 records you selected: Order Date from 1 Mar 2025 to 30 Jun 2025.
    """
    records = "record" if len(part) == 1 else "records"
    clauses = describe_filters(selection, whole)
    head = f"{len(part):,} of {len(whole):,} {records} you selected"
    return f"{head}: {'; '.join(clauses)}" if clauses else head


# --- the slice against the whole ---------------------------------------------


def _relative(part_value: float, whole_value: float) -> str:
    if not np.isfinite(part_value) or not np.isfinite(whole_value) or whole_value == 0:
        return "—"
    change = (part_value - whole_value) / abs(whole_value) * 100
    sign = "+" if change > 0 else ""
    return f"{sign}{change:.1f}% vs the whole"


def _points(part_value: float, whole_value: float) -> str:
    change = part_value - whole_value
    sign = "+" if change > 0 else ""
    return f"{sign}{change:.1f} pts vs the whole"


def compare_slice(
    whole: pd.DataFrame, selection: Selection, roles: ColumnRoles
) -> tuple[pd.DataFrame, SliceComparison]:
    """The selected rows, and every figure about them set against the whole.

    Slice statistics are never returned on their own: each one carries the
    matching figure for the whole view. A slice too small to support a
    statistic gets a note saying so instead of the statistic.
    """
    part = apply_selection(whole, selection)
    description = describe_selection(selection, whole, part)
    count, total_count = len(part), len(whole)
    notes: list[str] = [selection.adjustment] if selection.adjustment else []
    stats: list[SliceStat] = [
        SliceStat(
            label="Records",
            slice_value=f"{count:,}",
            whole_value=f"{total_count:,}",
            difference=f"{format_percentage(count / total_count * 100)} of the whole" if total_count else "—",
            calculation="count of rows inside the selection ÷ count of rows in the current view",
        )
    ]

    if count == 0:
        notes.append("No records fall inside the selection, so there is nothing to analyze.")
        return part, SliceComparison(description, count, total_count, tuple(stats), tuple(notes))
    if count == total_count:
        notes.append(
            "The selection covers every record in the view, so the slice is the whole and there is "
            "no difference to show."
        )
        return part, SliceComparison(description, count, total_count, tuple(stats), tuple(notes))

    measure = roles.measure if roles.measure and roles.measure in whole.columns else None
    if measure:
        spec = resolve_metric(measure)
        whole_values = pd.to_numeric(whole[measure], errors="coerce").dropna()
        part_values = pd.to_numeric(part[measure], errors="coerce").dropna()
        if spec.additive:
            part_total, whole_total = float(part_values.sum()), float(whole_values.sum())
            same_sign = (whole_values >= 0).all() or (whole_values <= 0).all()
            stats.append(
                SliceStat(
                    label=f"Total {measure}",
                    slice_value=format_number(part_total, measure, column_values=whole_values),
                    whole_value=format_number(whole_total, measure, column_values=whole_values),
                    difference=(
                        f"{format_percentage(part_total / whole_total * 100)} of the whole"
                        if whole_total and same_sign
                        else "—"
                    ),
                    calculation=f"sum of {measure} inside the selection ÷ sum of {measure} in the view",
                )
            )
        if len(part_values) >= MIN_SLICE_RECORDS:
            part_mean, whole_mean = float(part_values.mean()), float(whole_values.mean())
            stats.append(
                SliceStat(
                    label=f"Average {measure}",
                    slice_value=format_number(part_mean, measure, column_values=whole_values),
                    whole_value=format_number(whole_mean, measure, column_values=whole_values),
                    difference=_relative(part_mean, whole_mean),
                    calculation=(
                        f"mean of {measure} over the {len(part_values):,} selected records with a value, "
                        f"against the mean over all {len(whole_values):,}"
                    ),
                )
            )
        else:
            having = "record has" if len(part_values) == 1 else "records have"
            notes.append(
                f"Only {len(part_values):,} selected {having} "
                f"a {measure} value. An average needs at least {MIN_SLICE_RECORDS}, so none is reported."
            )

    dimension = roles.dimension if roles.dimension and roles.dimension in whole.columns else None
    if dimension:
        if count >= MIN_SLICE_RECORDS:
            part_mix = part[dimension].astype(str).where(part[dimension].notna()).value_counts(normalize=True)
            whole_mix = whole[dimension].astype(str).where(whole[dimension].notna()).value_counts(
                normalize=True
            )
            if not part_mix.empty:
                leader = str(part_mix.index[0])
                part_share = float(part_mix.iloc[0]) * 100
                whole_share = float(whole_mix.get(leader, 0.0)) * 100
                stats.append(
                    SliceStat(
                        label=f"Most common {dimension}",
                        slice_value=f"{leader} · {format_percentage(part_share)} of records",
                        whole_value=f"{leader} · {format_percentage(whole_share)} of records",
                        difference=_points(part_share, whole_share),
                        calculation=(
                            f"records with {dimension} = {leader} ÷ records, inside the selection and "
                            "in the view; rows with no label are left out of both"
                        ),
                    )
                )
        else:
            notes.append(
                f"{count:,} {'record is' if count == 1 else 'records are'} too few to compare the "
                f"{dimension} mix with the whole; at least {MIN_SLICE_RECORDS} are needed."
            )

    return part, SliceComparison(description, count, total_count, tuple(stats), tuple(notes))


def comparison_frame(comparison: SliceComparison) -> pd.DataFrame:
    """The comparison as a table: every slice figure beside the whole's."""
    return pd.DataFrame(
        [
            [stat.label, stat.slice_value, stat.whole_value, stat.difference, stat.calculation]
            for stat in comparison.stats
        ],
        columns=["Measure", "Your selection", "Whole view", "Difference", "Calculation"],
    )
