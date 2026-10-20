"""Application-level orchestration for preparing an ADA analysis."""

from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd

from .aggregation import preferred_frequency
from .analysis import CleaningReport, clean_dataframe
from .business_insights import BusinessBrief, Evidence, analyze_business
from .metrics import metric_for
from .schema import ColumnRoles, add_row_summary, detect_roles
from .selection import Selection, SliceComparison, compare_slice


@dataclass(frozen=True)
class PreparedAnalysis:
    dataframe: pd.DataFrame
    cleaning_report: CleaningReport
    detected_roles: ColumnRoles
    truncated_rows: int
    analyzed_from: pd.Timestamp | None = None
    analyzed_to: pd.Timestamp | None = None

    def analyze(self, roles: ColumnRoles | None = None) -> BusinessBrief:
        return analyze_business(self.dataframe, roles or self.detected_roles)


def _most_recent(dataframe: pd.DataFrame, date_column: str | None, row_limit: int) -> pd.DataFrame:
    """Keep the newest ``row_limit`` rows, not the first ones.

    Exports are usually oldest-first. Rows are ranked by ``date_column`` when it
    exists; otherwise the tail of the file is kept.
    """
    if len(dataframe) <= row_limit:
        return dataframe
    if date_column and date_column in dataframe.columns:
        order = dataframe[date_column].rank(method="first", ascending=False, na_option="bottom")
        kept = dataframe.loc[order <= row_limit]
        # Drop the oldest period if the cut removed any of its rows; a half-present
        # first period would read as growth into the next.
        dates = kept[date_column].dropna()
        if not dates.empty:
            grain = preferred_frequency(dates)
            buckets = dataframe[date_column].dt.to_period(grain)
            oldest = dates.min().to_period(grain)
            if (buckets[~dataframe.index.isin(kept.index)] == oldest).any():
                whole = kept[buckets.loc[kept.index] != oldest]
                # Skip the drop when every kept row is in that period (a large
                # export from one busy week), since it would leave no rows.
                if not whole.empty:
                    kept = whole
        return kept
    return dataframe.tail(row_limit)


def prepare_analysis(
    raw_dataframe: pd.DataFrame, *, row_limit: int, date: str | None = None
) -> PreparedAnalysis:
    """Clean the data, keep the most recent slice of it, and detect its schema.

    ``date`` names a column the caller wants as the date, so cleaning can turn
    a column of years into dates before the schema is detected. A matrix table
    gains a per-row mean to measure (see ``schema.add_row_summary``). What
    detection concluded about the columns is added to the cleaning notes, so
    every surface that lists the notes explains the roles too.
    """
    dataframe, cleaning_report = clean_dataframe(raw_dataframe, date=date)
    roles = detect_roles(dataframe)

    available = len(dataframe)
    dataframe = _most_recent(dataframe, roles.date, row_limit).reset_index(drop=True)
    if len(dataframe) < available:
        # Re-detect so the numeric and dimension candidates describe the kept rows.
        roles = detect_roles(dataframe)
    dataframe, roles = add_row_summary(dataframe, roles)
    cleaning_report = replace(
        cleaning_report, notes=tuple(dict.fromkeys((*cleaning_report.notes, *roles.notes)))
    )

    dated = roles.date is not None and roles.date in dataframe.columns
    span = dataframe[roles.date].dropna() if dated else pd.Series(dtype="datetime64[ns]")
    return PreparedAnalysis(
        dataframe=dataframe,
        cleaning_report=cleaning_report,
        detected_roles=roles,
        truncated_rows=available - len(dataframe),
        analyzed_from=span.min() if not span.empty else None,
        analyzed_to=span.max() if not span.empty else None,
    )


def apply_role_selection(
    detected: ColumnRoles,
    *,
    date: str,
    measure: str,
    dimension: str,
    aggregation: str | None = None,
) -> ColumnRoles:
    chosen = None if dimension == "None" else dimension
    picked = None if measure == "None" else measure
    return replace(
        detected,
        date=None if date == "None" else date,
        measure=picked,
        dimension=chosen,
        aggregation=aggregation,
        # The runners-up were for the detected measure, and the focus is a value
        # of the detected grouping, so each lapses when its role is changed.
        measure_alternatives=detected.measure_alternatives if picked == detected.measure else (),
        focus=detected.focus if chosen == detected.dimension else None,
    )


def focus_options(dataframe: pd.DataFrame, roles: ColumnRoles, limit: int = 40) -> list[str]:
    """Values of the active segment, most common first, for drill-down."""
    if not roles.dimension or roles.dimension not in dataframe.columns:
        return []
    counts = dataframe[roles.dimension].value_counts(dropna=True)
    return [str(value) for value in counts.head(limit).index]


def apply_focus(
    dataframe: pd.DataFrame,
    roles: ColumnRoles,
    focus: str | None,
) -> tuple[pd.DataFrame, ColumnRoles]:
    """Drill into one segment value and regroup by the next useful dimension.

    The returned roles have no ``focus`` left to apply, since the rows already
    hold only that value.
    """
    if not focus or not roles.dimension:
        return dataframe, roles
    filtered = dataframe[dataframe[roles.dimension].astype(str) == focus]
    if filtered.empty:
        return dataframe, roles
    replacement = next(
        (
            column
            for column in roles.dimensions
            if column != roles.dimension
            and column in filtered.columns
            and filtered[column].nunique(dropna=True) >= 2
        ),
        None,
    )
    return filtered.reset_index(drop=True), replace(roles, dimension=replacement, focus=None)


@dataclass(frozen=True)
class SelectionAnalysis:
    dataframe: pd.DataFrame
    comparison: SliceComparison
    # Each piece of evidence about the slice, beside the same kind of evidence
    # for the whole view (None when the whole view's brief has none of it).
    evidence: tuple[tuple[Evidence, Evidence | None], ...] = ()


def analyze_selection(
    dataframe: pd.DataFrame,
    roles: ColumnRoles,
    selection: Selection,
    whole_brief: BusinessBrief | None = None,
) -> SelectionAnalysis:
    """Analyze a slice the reader picked on a chart, set against the view it came from.

    The slice goes through the same ``analyze_business`` as the whole view, and
    each finding is paired with the matching one for the whole, so no slice
    figure is ever shown alone. A slice too small for statistics, or one that
    is the whole view, is not analyzed rather than reported on thin evidence.
    """
    part, comparison = compare_slice(dataframe, selection, roles)
    if not comparison.supports_statistics:
        return SelectionAnalysis(dataframe=part, comparison=comparison)
    sliced = analyze_business(part, roles)
    whole = whole_brief or analyze_business(dataframe, roles)
    by_kind: dict[str, Evidence] = {}
    for item in whole.evidence:
        by_kind.setdefault(item.kind, item)
    pairs = tuple((item, by_kind.get(item.kind)) for item in sliced.evidence)
    return SelectionAnalysis(dataframe=part, comparison=comparison, evidence=pairs)


def cleaning_audit_frame(report: CleaningReport) -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["Empty rows removed", report.empty_rows_removed],
            ["Empty columns removed", report.empty_columns_removed],
            ["Exported index columns removed", report.index_columns_removed],
            ["Duplicate rows removed", report.duplicate_rows_removed],
            ["Identical rows kept", report.duplicate_rows_found - report.duplicate_rows_removed],
            ["Numeric columns inferred", report.numeric_columns_inferred],
            ["Datetime columns inferred", report.datetime_columns_inferred],
            ["Date values that could not be read", report.unparsed_date_cells],
        ],
        columns=["Operation", "Count"],
    )


def schema_frame(roles: ColumnRoles) -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["Primary metric", roles.measure or "Not detected"],
            # Total, Average, or Count when there is no metric and rows are counted.
            ["Combine rows by", metric_for(roles.measure, roles).combines_as.capitalize()],
            ["Grouping column", roles.dimension or "Not detected"],
            ["Date", roles.date or "Not detected"],
            ["Identifier", roles.identifier or "Not detected"],
        ],
        columns=["Role", "Column"],
    )
