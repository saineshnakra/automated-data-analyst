"""Choosing the chart that fits the columns a reader picked.

The form is decided by what the data has to do -- compare magnitudes, show a
trend, tell series apart, relate two measures -- and not by what happens to be
available. Every decision is returned as a sentence alongside the spec, because
a chart nobody can question is the same problem as a number nobody can check.

Nothing here draws anything. It reports what should be drawn and why, so the
choice can be tested without a browser.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

# Past this many classes a legend stops helping and a table reads better.
MAX_SERIES = 4
MAX_BARS = 25
# A heatmap is a rectangle of cells and the browser is sent every one of them.
# Two identifier-ish columns produce millions, which exceeds Streamlit's own
# websocket message limit before it ever reaches a screen.
MAX_HEATMAP_CELLS = 4_000
# Beyond this a vertical axis cannot hold the labels.
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
    """A vertical axis cannot hold many categories, or long ones."""
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

    # Time on one axis is the strongest signal there is: the job is a trend.
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

    # Two categories and a measure form a grid, which a heatmap reads better
    # than any number of side-by-side bars.
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

    # Two measures and nothing to group by: the question is how they relate.
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
    """Keep the largest series and group the tail, rather than adding hues.

    A generated fifth or ninth colour is indistinguishable from one already in
    use, so the honest move is to stop drawing series, not to invent colours.
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
    # A Categorical rejects a label that is not already one of its categories,
    # and a column that genuinely contains "Other" would have the folded tail
    # silently added to a real segment. Both are avoided by choosing a label
    # the column does not already use and dropping the categorical dtype.
    folded[column] = folded[column].astype(object)
    label = _fold_label(set(totals.index))
    folded[column] = folded[column].where(folded[column].isin(keep), label)
    return folded


def _fold_label(existing: set) -> str:
    """A name for the folded tail that is not already a real category."""
    label = "Other"
    suffix = 2
    present = {str(value) for value in existing}
    while label in present:
        label = f"Other ({suffix})"
        suffix += 1
    return label
