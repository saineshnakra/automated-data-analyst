"""Deterministic, local analysis helpers for the Streamlit application."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype


@dataclass(frozen=True)
class CleaningReport:
    original_rows: int
    original_columns: int
    final_rows: int
    final_columns: int
    duplicate_rows_removed: int
    empty_rows_removed: int
    empty_columns_removed: int
    index_columns_removed: int
    trimmed_text_columns: int
    numeric_columns_inferred: int
    datetime_columns_inferred: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class Insight:
    title: str
    detail: str
    level: str = "info"


def _make_unique_columns(columns: pd.Index) -> list[str]:
    """Return readable, unique column names without changing their meaning."""
    counts: dict[str, int] = {}
    result: list[str] = []

    for position, raw_name in enumerate(columns, start=1):
        base = " ".join(str(raw_name).strip().split()) or f"column_{position}"
        counts[base] = counts.get(base, 0) + 1
        suffix = f"_{counts[base]}" if counts[base] > 1 else ""
        result.append(f"{base}{suffix}")

    return result


def _looks_like_exported_index(series: pd.Series, name: str) -> bool:
    if not name.lower().startswith("unnamed"):
        return False

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        return False

    return np.array_equal(numeric.to_numpy(), np.arange(len(series)))


def clean_dataframe(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Apply conservative, explainable cleaning and return an audit report."""
    if dataframe.empty or len(dataframe.columns) == 0:
        raise ValueError("The CSV does not contain any rows and columns to analyze.")

    cleaned = dataframe.copy()
    original_rows, original_columns = cleaned.shape
    cleaned.columns = _make_unique_columns(cleaned.columns)

    empty_columns = [column for column in cleaned.columns if cleaned[column].isna().all()]
    cleaned = cleaned.drop(columns=empty_columns)

    empty_row_mask = cleaned.isna().all(axis=1)
    empty_rows_removed = int(empty_row_mask.sum())
    cleaned = cleaned.loc[~empty_row_mask].copy()

    index_columns = [
        column for column in cleaned.columns if _looks_like_exported_index(cleaned[column], column)
    ]
    cleaned = cleaned.drop(columns=index_columns)

    trimmed_text_columns = 0
    numeric_columns_inferred = 0
    datetime_columns_inferred = 0

    protected_numeric_tokens = ("id", "code", "zip", "postal", "phone")
    date_tokens = ("date", "time", "timestamp", "created", "updated")

    for column in cleaned.select_dtypes(include=["object", "string"]).columns:
        series = cleaned[column]
        non_null_before = int(series.notna().sum())
        cleaned[column] = series.map(lambda value: value.strip() if isinstance(value, str) else value)
        cleaned[column] = cleaned[column].replace("", pd.NA)
        trimmed_text_columns += 1

        if non_null_before == 0:
            continue

        normalized_name = column.lower()
        if any(token in normalized_name for token in date_tokens):
            parsed_dates = pd.to_datetime(cleaned[column], errors="coerce")
            if parsed_dates.notna().sum() / non_null_before >= 0.8:
                cleaned[column] = parsed_dates
                datetime_columns_inferred += 1
                continue

        if not any(token in normalized_name for token in protected_numeric_tokens):
            parsed_numeric = pd.to_numeric(cleaned[column], errors="coerce")
            if parsed_numeric.notna().sum() / non_null_before >= 0.95:
                cleaned[column] = parsed_numeric
                numeric_columns_inferred += 1

    duplicate_rows = int(cleaned.duplicated().sum())
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    if cleaned.empty or len(cleaned.columns) == 0:
        raise ValueError("No analyzable data remained after removing empty rows and columns.")

    report = CleaningReport(
        original_rows=original_rows,
        original_columns=original_columns,
        final_rows=len(cleaned),
        final_columns=len(cleaned.columns),
        duplicate_rows_removed=duplicate_rows,
        empty_rows_removed=empty_rows_removed,
        empty_columns_removed=len(empty_columns),
        index_columns_removed=len(index_columns),
        trimmed_text_columns=trimmed_text_columns,
        numeric_columns_inferred=numeric_columns_inferred,
        datetime_columns_inferred=datetime_columns_inferred,
    )
    return cleaned, report


def column_profile(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Build a compact data dictionary suitable for display or export."""
    rows: list[dict[str, Any]] = []

    for column in dataframe.columns:
        series = dataframe[column]
        non_null = series.dropna()
        sample = "—" if non_null.empty else str(non_null.iloc[0])[:80]

        if is_datetime64_any_dtype(series):
            semantic_type = "datetime"
        elif is_numeric_dtype(series):
            semantic_type = "numeric"
        else:
            unique_ratio = series.nunique(dropna=True) / max(len(non_null), 1)
            semantic_type = "category" if series.nunique(dropna=True) <= 50 or unique_ratio <= 0.2 else "text"

        rows.append(
            {
                "Column": column,
                "Type": semantic_type,
                "Pandas dtype": str(series.dtype),
                "Non-null": int(series.notna().sum()),
                "Missing": int(series.isna().sum()),
                "Missing %": round(float(series.isna().mean() * 100), 2),
                "Unique": int(series.nunique(dropna=True)),
                "Example": sample,
            }
        )

    return pd.DataFrame(rows)


def generate_insights(dataframe: pd.DataFrame, limit: int = 6) -> list[Insight]:
    """Generate factual, reproducible observations without an external model."""
    insights: list[Insight] = []
    row_count = len(dataframe)

    missing = dataframe.isna().mean().sort_values(ascending=False)
    if not missing.empty and missing.iloc[0] > 0:
        column = str(missing.index[0])
        rate = float(missing.iloc[0] * 100)
        insights.append(
            Insight(
                "Missing data deserves attention",
                f"{column} has the highest missing-value rate at {rate:.1f}%.",
                "warning" if rate >= 20 else "info",
            )
        )

    numeric_columns = dataframe.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_columns) >= 2:
        correlations = dataframe[numeric_columns].corr()
        upper_triangle = correlations.where(np.triu(np.ones(correlations.shape), k=1).astype(bool))
        stacked = upper_triangle.stack().dropna()
        if not stacked.empty:
            pair = stacked.abs().idxmax()
            value = float(correlations.loc[pair[0], pair[1]])
            insights.append(
                Insight(
                    "Strongest numeric relationship",
                    f"{pair[0]} and {pair[1]} have a Pearson correlation of {value:.2f}. "
                    "Correlation does not imply causation.",
                )
            )

    outlier_candidates: list[tuple[str, int, float]] = []
    for column in numeric_columns:
        series = dataframe[column].dropna()
        if len(series) < 8 or series.nunique() < 3:
            continue
        first_quartile, third_quartile = series.quantile([0.25, 0.75])
        iqr = third_quartile - first_quartile
        if iqr == 0:
            continue
        count = int(((series < first_quartile - 1.5 * iqr) | (series > third_quartile + 1.5 * iqr)).sum())
        if count:
            outlier_candidates.append((column, count, count / len(series) * 100))

    if outlier_candidates:
        column, count, rate = max(outlier_candidates, key=lambda candidate: candidate[2])
        insights.append(
            Insight(
                "Potential outliers",
                f"{column} contains {count:,} values ({rate:.1f}%) outside the standard 1.5×IQR range.",
                "warning",
            )
        )

    non_numeric_columns = [column for column in dataframe.columns if column not in numeric_columns]
    dominance_candidates: list[tuple[str, str, float]] = []
    for column in non_numeric_columns:
        counts = dataframe[column].value_counts(normalize=True, dropna=True)
        if not counts.empty and dataframe[column].nunique(dropna=True) <= 100:
            dominance_candidates.append((column, str(counts.index[0]), float(counts.iloc[0] * 100)))

    if dominance_candidates:
        column, value, share = max(dominance_candidates, key=lambda candidate: candidate[2])
        insights.append(
            Insight(
                "Largest category share",
                f"{value} is the most common value in {column}, representing "
                f"{share:.1f}% of non-missing rows.",
            )
        )

    datetime_columns = dataframe.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    if datetime_columns:
        column = datetime_columns[0]
        values = dataframe[column].dropna()
        if not values.empty:
            insights.append(
                Insight(
                    "Time coverage",
                    f"{column} spans {values.min():%Y-%m-%d} through {values.max():%Y-%m-%d}.",
                )
            )

    if not insights:
        insights.append(
            Insight(
                "Dataset is ready to explore",
                f"The cleaned dataset contains {row_count:,} rows across "
                f"{len(dataframe.columns):,} columns with no immediate quality flags.",
                "success",
            )
        )

    return insights[:limit]


def build_markdown_report(
    dataframe: pd.DataFrame,
    cleaning_report: CleaningReport,
    insights: list[Insight],
    description: str = "",
) -> str:
    """Create a portable report containing only computed facts."""
    profile = column_profile(dataframe)
    lines = [
        "# Automated Data Analysis Report",
        "",
        description.strip() or "Local, deterministic exploratory data analysis.",
        "",
        "## Dataset overview",
        "",
        f"- Rows: {len(dataframe):,}",
        f"- Columns: {len(dataframe.columns):,}",
        f"- Missing cells: {int(dataframe.isna().sum().sum()):,}",
        f"- Duplicate rows removed: {cleaning_report.duplicate_rows_removed:,}",
        "",
        "## Key observations",
        "",
    ]
    lines.extend(f"- **{insight.title}:** {insight.detail}" for insight in insights)
    lines.extend(["", "## Data dictionary", ""])

    for row in profile.to_dict(orient="records"):
        lines.append(
            f"- **{row['Column']}** — {row['Type']}; {row['Missing %']:.2f}% missing; "
            f"{row['Unique']:,} unique values"
        )

    lines.extend(
        [
            "",
            "---",
            "Generated locally by Automated Data Analyst. No uploaded data was sent "
            "to an external AI service.",
        ]
    )
    return "\n".join(lines)
