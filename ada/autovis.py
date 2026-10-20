"""Recommend a chart form for the columns a reader picked.

The form follows the task (compare magnitudes, show a trend, separate series,
relate two measures), and each spec carries a sentence explaining the choice.
Nothing is drawn here, so recommendations can be tested without a browser.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

# Series past MAX_SERIES fold into "Other" (extra hues are hard to tell apart);
# categories past MAX_BARS render as a table.
MAX_SERIES = 4
MAX_BARS = 25
# Every heatmap cell is sent to the browser; two high-cardinality columns can
# exceed Streamlit's websocket message limit.
MAX_HEATMAP_CELLS = 4_000
# Past this many categories, or labels longer than LONG_LABEL_CHARACTERS, bars go horizontal.
HORIZONTAL_BAR_THRESHOLD = 6
LONG_LABEL_CHARACTERS = 12

Form = str


@dataclass(frozen=True)
class ChartSpec:
    """What to draw, and the reasoning that picked it."""

    form: Form  # stat | line | area | bar | column | histogram | scatter | heatmap | table | none
    rationale: str
    x: str | None = None
    y: str | None = None
    color: str | None = None
    aggregation: str | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_chart(self) -> bool:
        return self.form not in ("stat", "table", "none")


def _kind(series: pd.Series) -> str:
    if is_datetime64_any_dtype(series):
        return "datetime"
    if is_numeric_dtype(series):
        return "numeric"
    return "category"


def _classify(dataframe: pd.DataFrame, columns: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {"datetime": [], "numeric": [], "category": []}
    for column in columns:
        if column in dataframe.columns:
            groups[_kind(dataframe[column])].append(column)
    return groups


def _distinct(dataframe: pd.DataFrame, column: str) -> int:
    return int(dataframe[column].nunique(dropna=True))


def _labels_are_long(dataframe: pd.DataFrame, column: str) -> bool:
    values = dataframe[column].dropna().astype(str)
    if values.empty:
        return False
    return bool(values.str.len().max() > LONG_LABEL_CHARACTERS)


def _bar_form(dataframe: pd.DataFrame, category: str) -> tuple[Form, str]:
    """Choose horizontal bars or vertical columns, with a phrase saying why.

    Many categories or long labels don't fit along a vertical layout's x axis.
    """
    if _distinct(dataframe, category) > HORIZONTAL_BAR_THRESHOLD or _labels_are_long(
        dataframe, category
    ):
        return "bar", "laid out horizontally so the category names stay readable"
    return "column", "as vertical columns, which read naturally for a handful of categories"


def recommend_chart(
    dataframe: pd.DataFrame,
    columns: list[str],
    *,
    aggregation: str = "sum",
) -> ChartSpec:
    """Pick the form that suits these columns, and say why."""
    columns = [column for column in columns if column in dataframe.columns]
    if not columns:
        return ChartSpec(form="none", rationale="Pick a column to chart.")

    groups = _classify(dataframe, columns)
    dates, numbers, categories = groups["datetime"], groups["numeric"], groups["category"]
    notes: list[str] = []

    if len(numbers) > 1:
        notes.append(
            f"Charting {numbers[0]} only. Two measures on one chart would need two "
            "scales, which makes their lines comparable by accident rather than by fact."
        )

    # A date with a measure is always a trend, whatever else was picked.
    if dates and numbers:
        date, measure = dates[0], numbers[0]
        if categories:
            segment = categories[0]
            distinct = _distinct(dataframe, segment)
            if distinct > MAX_SERIES:
                notes.append(
                    f"{segment} has {distinct} values; the {MAX_SERIES} largest are drawn "
                    "and the rest are grouped as Other, because more lines than that stop "
                    "being tellable apart."
                )
            return ChartSpec(
                form="line",
                x=date,
                y=measure,
                color=segment,
                aggregation=aggregation,
                rationale=(
                    f"A line per {segment}, because the reader has to tell the series "
                    f"apart over time rather than read one total."
                ),
                notes=tuple(notes),
            )
        return ChartSpec(
            form="area",
            x=date,
            y=measure,
            aggregation=aggregation,
            rationale=(
                f"{measure} over time as a single filled series, because there is one "
                "thing to follow and the shape of the movement is the point."
            ),
            notes=tuple(notes),
        )

    # Two categories and a measure form a grid: a heatmap, not grouped bars.
    if len(categories) >= 2 and numbers:
        rows, columns_axis, measure = categories[0], categories[1], numbers[0]
        cells = _distinct(dataframe, rows) * _distinct(dataframe, columns_axis)
        if cells > MAX_HEATMAP_CELLS:
            return ChartSpec(
                form="table",
                x=columns_axis,
                y=rows,
                color=measure,
                aggregation=aggregation,
                rationale=(
                    f"{rows} and {columns_axis} would make a {cells:,}-cell grid. Past about "
                    f"{MAX_HEATMAP_CELLS:,} cells a heatmap is unreadable and too large to "
                    "send, so the numbers are shown as a table instead."
                ),
                notes=tuple(notes),
            )
        return ChartSpec(
            form="heatmap",
            x=columns_axis,
            y=rows,
            color=measure,
            aggregation=aggregation,
            rationale=(
                f"{measure} across every {rows} and {columns_axis} pair, shaded light to "
                "dark, because the job is comparing magnitude across a grid."
            ),
            notes=tuple(notes),
        )

    if categories and numbers:
        category, measure = categories[0], numbers[0]
        distinct = _distinct(dataframe, category)
        if distinct > MAX_BARS:
            return ChartSpec(
                form="table",
                x=category,
                y=measure,
                aggregation=aggregation,
                rationale=(
                    f"{category} has {distinct} values. Past about {MAX_BARS} bars nobody "
                    "reads the chart, so the numbers are shown as a table instead."
                ),
                notes=tuple(notes),
            )
        form, layout = _bar_form(dataframe, category)
        return ChartSpec(
            form=form,
            x=category,
            y=measure,
            aggregation=aggregation,
            rationale=(
                f"{aggregation} of {measure} by {category}, {layout}. One hue, "
                "light to dark, because the job is magnitude rather than identity."
            ),
            notes=tuple(notes),
        )

    # Two measures and nothing to group by: show how they relate.
    if len(numbers) >= 2:
        return ChartSpec(
            form="scatter",
            x=numbers[0],
            y=numbers[1],
            rationale=(
                f"One point per record, {numbers[0]} against {numbers[1]}, because the "
                "question two measures ask together is whether they move together."
            ),
            notes=(),
        )

    if dates and not numbers:
        return ChartSpec(
            form="area",
            x=dates[0],
            y=None,
            aggregation="count",
            rationale=(
                "Records per period. With no measure chosen, how often something "
                "happened is the only honest thing to plot."
            ),
            notes=tuple(notes),
        )

    if numbers:
        measure = numbers[0]
        if len(dataframe[measure].dropna()) <= 1:
            return ChartSpec(
                form="stat",
                y=measure,
                rationale=(
                    "A single value is a number, not a chart. A one-bar bar chart "
                    "carries no more information than the figure itself."
                ),
            )
        return ChartSpec(
            form="histogram",
            x=measure,
            y=None,
            aggregation="count",
            rationale=(
                f"The distribution of {measure} in equal-width bins, because one measure "
                "on its own asks how its values are spread rather than how they compare. "
                "Counting each distinct value instead would draw one bar per row."
            ),
            notes=tuple(notes),
        )

    category = categories[0]
    distinct = _distinct(dataframe, category)
    if distinct > MAX_BARS:
        return ChartSpec(
            form="table",
            x=category,
            aggregation="count",
            rationale=(
                f"{category} has {distinct} values, too many to read as bars, so they "
                "are counted into a table."
            ),
        )
    form, layout = _bar_form(dataframe, category)
    return ChartSpec(
        form=form,
        x=category,
        y=None,
        aggregation="count",
        rationale=f"How many records fall in each {category}, {layout}.",
    )


def fold_small_series(
    frame: pd.DataFrame, column: str, value: str, limit: int = MAX_SERIES
) -> pd.DataFrame:
    """Keep the ``limit`` largest series by total ``value`` and fold the rest into "Other".

    Extra generated colours are hard to tell from ones already in use. The label
    becomes "Other (2)" and so on if the column already has an "Other".
    """
    if frame.empty or column not in frame.columns:
        return frame
    totals = (
        frame.groupby(column, dropna=True, observed=True)[value].sum().sort_values(ascending=False)
    )
    if len(totals) <= limit:
        return frame
    keep = set(totals.head(limit).index)
    folded = frame.copy()
    # Cast to object so a Categorical accepts the new label, and use a label the
    # column lacks so the tail never merges into a real "Other" segment.
    folded[column] = folded[column].astype(object)
    label = _fold_label(set(totals.index))
    folded[column] = folded[column].where(folded[column].isin(keep), label)
    return folded


def _fold_label(existing: set) -> str:
    """Return a label for the folded tail that no existing category uses."""
    label = "Other"
    suffix = 2
    present = {str(value) for value in existing}
    while label in present:
        label = f"Other ({suffix})"
        suffix += 1
    return label
