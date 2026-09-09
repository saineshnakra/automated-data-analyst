"""Application-level orchestration for preparing an ADA analysis."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from aggregation import preferred_frequency
from analysis import CleaningReport, clean_dataframe
from business_insights import BusinessBrief, analyze_business
from schema import ColumnRoles, detect_roles


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
    """Keep the newest rows, not the first ones.

    Exports are usually written oldest-first, so taking the head of a long
    file analyzes the periods nobody is asking about and drops the ones they
    are -- and then forecasts months the file already contains. When a date
    column is available the newest rows are selected by date; otherwise the
    end of the file is the best available proxy for the recent end of it.
    """
    if len(dataframe) <= row_limit:
        return dataframe
    if date_column and date_column in dataframe.columns:
        order = dataframe[date_column].rank(method="first", ascending=False, na_option="bottom")
        kept = dataframe.loc[order <= row_limit]
        # A cut that lands inside a period leaves that period half-present,
        # and a half-present oldest period reads as growth into the next one.
        # It is dropped whenever any row of it was cut.
        dates = kept[date_column].dropna()
        if not dates.empty:
            grain = preferred_frequency(dates)
            buckets = dataframe[date_column].dt.to_period(grain)
            oldest = dates.min().to_period(grain)
            if (buckets[~dataframe.index.isin(kept.index)] == oldest).any():
                whole = kept[buckets.loc[kept.index] != oldest]
                # Only when something survives it. Every kept row sitting in
                # one truncated period is the ordinary shape of a big export
                # from a busy week: dropping it leaves ZERO rows, and the app
                # then reports an empty dataset and a $0.00 headline for a
                # file with a quarter of a million rows in it.
                if not whole.empty:
                    kept = whole
        return kept
    return dataframe.tail(row_limit)


def prepare_analysis(raw_dataframe: pd.DataFrame, *, row_limit: int) -> PreparedAnalysis:
    """Clean the data, keep the most recent slice of it, and detect its schema."""
    dataframe, cleaning_report = clean_dataframe(raw_dataframe)
    roles = detect_roles(dataframe)

    available = len(dataframe)
    dataframe = _most_recent(dataframe, roles.date, row_limit).reset_index(drop=True)
    if len(dataframe) < available:
        # Roles are detected on everything, but the numeric and dimension
        # candidates have to describe the rows actually being analyzed.
        roles = detect_roles(dataframe)

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
) -> ColumnRoles:
    return ColumnRoles(
        date=None if date == "None" else date,
        measure=None if measure == "None" else measure,
        dimension=None if dimension == "None" else dimension,
        identifier=detected.identifier,
        numeric=detected.numeric,
        dimensions=detected.dimensions,
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
    """Drill into one segment value and regroup by the next useful dimension."""
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
    focused_roles = ColumnRoles(
        date=roles.date,
        measure=roles.measure,
        dimension=replacement,
        identifier=roles.identifier,
        numeric=roles.numeric,
        dimensions=roles.dimensions,
    )
    return filtered.reset_index(drop=True), focused_roles


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
            ["Business segment", roles.dimension or "Not detected"],
            ["Date", roles.date or "Not detected"],
            ["Identifier", roles.identifier or "Not detected"],
        ],
        columns=["Role", "Column"],
    )
