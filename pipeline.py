"""Application-level orchestration for preparing an ADA analysis."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from analysis import CleaningReport, clean_dataframe
from business_insights import BusinessBrief, ColumnRoles, analyze_business, detect_roles


@dataclass(frozen=True)
class PreparedAnalysis:
    dataframe: pd.DataFrame
    cleaning_report: CleaningReport
    detected_roles: ColumnRoles
    truncated_rows: int

    def analyze(self, roles: ColumnRoles | None = None) -> BusinessBrief:
        return analyze_business(self.dataframe, roles or self.detected_roles)


def prepare_analysis(raw_dataframe: pd.DataFrame, *, row_limit: int) -> PreparedAnalysis:
    """Bound work, clean data, and detect its likely business schema."""
    original_rows = len(raw_dataframe)
    bounded = raw_dataframe.head(row_limit).copy() if original_rows > row_limit else raw_dataframe
    dataframe, cleaning_report = clean_dataframe(bounded)
    return PreparedAnalysis(
        dataframe=dataframe,
        cleaning_report=cleaning_report,
        detected_roles=detect_roles(dataframe),
        truncated_rows=max(original_rows - row_limit, 0),
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


def cleaning_audit_frame(report: CleaningReport) -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["Empty rows removed", report.empty_rows_removed],
            ["Empty columns removed", report.empty_columns_removed],
            ["Exported index columns removed", report.index_columns_removed],
            ["Duplicate rows removed", report.duplicate_rows_removed],
            ["Numeric columns inferred", report.numeric_columns_inferred],
            ["Datetime columns inferred", report.datetime_columns_inferred],
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
