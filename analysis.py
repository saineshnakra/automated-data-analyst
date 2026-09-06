"""Deterministic, local analysis helpers for the Streamlit application."""

from __future__ import annotations

import re
import warnings
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
    duplicate_rows_found: int = 0
    unparsed_date_cells: int = 0
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Insight:
    title: str
    detail: str
    level: str = "info"


def _speculative_dates(values: pd.Series) -> pd.Series:
    """Parse values as dates without complaining about the ones that are not.

    Every call here is a guess about a column whose format is unknown and
    which is usually not dates at all, so pandas' "could not infer format"
    warning is the expected case rather than something to report.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return pd.to_datetime(values, errors="coerce")


def _drop_timezone(series: pd.Series) -> pd.Series:
    """Return the same instants as naive local time.

    Every downstream calculation compares a date against a period boundary,
    and pandas builds those boundaries without a timezone. Converting to UTC
    first would move a row into the previous calendar day for anyone east of
    Greenwich, so the wall clock the file was written in is what is kept.
    """
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        return series.dt.tz_localize(None)
    return series


def _normalize_datetime_columns(frame: pd.DataFrame) -> int:
    """Strip timezones from every date column, whatever put them there."""
    changed = 0
    for column in frame.columns:
        series = frame[column]
        if isinstance(series.dtype, pd.DatetimeTZDtype):
            frame[column] = _drop_timezone(series)
            changed += 1
    return changed


# A business-formatted number: an optional sign or accounting parentheses, an
# optional currency symbol, then digits with grouping punctuation. Anything
# holding a slash or a letter is not this -- notably a date, which would
# otherwise survive as a very large integer.
_MONEY = re.compile(r"^[+-]?\(?\s*[-+]?\s*[$\u20ac\u00a3\u00a5\u20b9]?\s*\d[\d,.\s']*\)?$")


def _numeric_from_text(values: pd.Series) -> pd.Series:
    """Read business-formatted numbers: "$1,203.55", "(48.10)", "1 234,50".

    A thousands separator is punctuation, not data, and an amount in
    parentheses is how every accounting export writes a negative. Reading
    them as text loses the largest values in the file, which are exactly the
    ones with a separator in them.
    """
    text = values.astype("string").str.strip()
    money = text.str.match(_MONEY, na=False)
    text = text.where(money)
    negative = text.str.contains(r"\(", na=False)
    text = text.str.replace(r"[^\d,.]", "", regex=True)
    # "1.234,50" is European; "1,234.50" is not. Whichever separator comes
    # last is the decimal point.
    european = text.str.contains(r",\d{1,2}$", na=False) & ~text.str.contains(r"\.\d", na=False)
    text = text.mask(
        european, text.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    )
    text = text.mask(~european, text.str.replace(",", "", regex=False))
    parsed = pd.to_numeric(text, errors="coerce")
    return parsed.mask(negative & parsed.notna(), -parsed)


# A date written day-or-month first: "3/1/2024", "03-01-24". A year-leading
# value is ISO and cannot be read two ways.
_DAY_OR_MONTH_FIRST = re.compile(r"^\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}")


def _dated(values: pd.Series, *, dayfirst: bool) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return pd.to_datetime(values, errors="coerce", dayfirst=dayfirst)


def _read_dates(values: pd.Series) -> tuple[pd.Series, str]:
    """Parse a date column, and say so when the ordering had to be guessed.

    "01/03/2024" is 1 March in most of the world and 3 January in the United
    States. When some row in the column settles it -- a 13 or higher in the
    first position -- that reading wins outright. When nothing settles it,
    month-first is assumed and the assumption is reported, because silently
    turning a year of monthly figures into twelve days of January is the one
    outcome nobody can detect downstream.
    """
    month_first = _dated(values, dayfirst=False)
    # Only a value that leads with a day or a month can be read two ways.
    # 2024-03-01 is ISO and settled; 01/03/2024 is not.
    two_ways = values.astype("string").str.match(_DAY_OR_MONTH_FIRST, na=False)
    if not bool(two_ways.any()):
        return month_first, ""

    day_first = _dated(values, dayfirst=True)
    if day_first.notna().sum() > month_first.notna().sum():
        return day_first, "day-first"
    if month_first.notna().sum() > day_first.notna().sum():
        return month_first, ""
    disagree = two_ways & month_first.notna() & day_first.notna() & month_first.ne(day_first)
    return month_first, "ambiguous" if bool(disagree.any()) else ""


def _ordering_note(column: str, ordering: str) -> str:
    if ordering == "day-first":
        return f"{column} was read day-first (25/12/2024 is 25 December)."
    return (
        f"{column} could be read either way round; month-first was assumed "
        f"(01/03/2024 is 3 January). Rename the column or use ISO dates to be sure."
    )


def _make_unique_columns(columns: pd.Index) -> list[str]:
    """Return readable, unique column names without changing their meaning."""
    counts: dict[str, int] = {}
    taken: set[str] = set()
    result: list[str] = []

    for position, raw_name in enumerate(columns, start=1):
        base = " ".join(str(raw_name).strip().split()) or f"column_{position}"
        counts[base] = counts.get(base, 0) + 1
        candidate = base if counts[base] == 1 else f"{base}_{counts[base]}"
        # A file can already contain the name the suffix would produce, so
        # keep counting until the result is genuinely unused.
        while candidate in taken:
            counts[base] += 1
            candidate = f"{base}_{counts[base]}"
        taken.add(candidate)
        result.append(candidate)

    return result


def _looks_like_exported_index(series: pd.Series, name: str) -> bool:
    if not name.lower().startswith("unnamed"):
        return False

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        return False

    return np.array_equal(numeric.to_numpy(), np.arange(len(series)))


def clean_dataframe(
    dataframe: pd.DataFrame, *, drop_duplicates: bool = False
) -> tuple[pd.DataFrame, CleaningReport]:
    """Apply conservative, explainable cleaning and return an audit report.

    Identical rows are counted but kept. Two sales of the same item, for the
    same amount, on the same day are an ordinary Tuesday at a till, not a
    defect, and deleting them silently removes real revenue from every number
    on the page. Callers that know their rows carry a key can opt in.
    """
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
    datetime_columns_inferred = _normalize_datetime_columns(cleaned)
    unparsed_date_cells = 0
    notes: list[str] = []
    if datetime_columns_inferred:
        notes.append(
            "Timezone offsets were dropped; dates are read as the local time they were written in."
        )

    protected_numeric_tokens = ("id", "code", "zip", "postal", "phone")
    date_tokens = ("date", "time", "timestamp", "created", "updated")
    named_date_ratio, numeric_ratio, unnamed_date_ratio = 0.8, 0.95, 0.95
    date_sample_size = 50

    for column in cleaned.select_dtypes(include=["object", "string"]).columns:
        series = cleaned[column]
        non_null_before = int(series.notna().sum())
        cleaned[column] = series.map(lambda value: value.strip() if isinstance(value, str) else value)
        cleaned[column] = cleaned[column].replace("", pd.NA)
        trimmed_text_columns += 1

        if non_null_before == 0:
            continue

        normalized_name = column.lower()
        named_like_a_date = any(token in normalized_name for token in date_tokens)
        if named_like_a_date:
            parsed_dates, ordering = _read_dates(cleaned[column])
            if parsed_dates.notna().sum() / non_null_before >= named_date_ratio:
                unreadable = int(non_null_before - parsed_dates.notna().sum())
                if unreadable:
                    unparsed_date_cells += unreadable
                    notes.append(f"{unreadable} {column} values could not be read as dates.")
                if ordering:
                    notes.append(_ordering_note(column, ordering))
                cleaned[column] = _drop_timezone(parsed_dates)
                datetime_columns_inferred += 1
                continue

        if not any(token in normalized_name for token in protected_numeric_tokens):
            parsed_numeric = pd.to_numeric(cleaned[column], errors="coerce")
            if parsed_numeric.notna().sum() < non_null_before:
                # Whatever plain parsing could not read, try again allowing the
                # punctuation a finance export writes: grouping separators, a
                # currency symbol, parentheses for a negative. The values that
                # need it are the large ones, so leaving them out biases every
                # total downwards.
                formatted = _numeric_from_text(cleaned[column])
                if formatted.notna().sum() > parsed_numeric.notna().sum():
                    parsed_numeric = formatted
            if parsed_numeric.notna().sum() / non_null_before >= numeric_ratio:
                cleaned[column] = parsed_numeric
                numeric_columns_inferred += 1
                continue

        if not named_like_a_date:
            # A column holds dates whatever it happens to be called -- "Month",
            # "Period", "FY". Numbers were tried first, so a column of bare years
            # stays numeric instead of becoming the 1st of January in each of
            # them. A sample decides whether the full parse is worth attempting,
            # because most text columns are not dates and this runs over all of
            # them.
            sample = cleaned[column].dropna().head(date_sample_size)
            if len(sample) and _speculative_dates(sample).notna().mean() >= unnamed_date_ratio:
                parsed_dates, ordering = _read_dates(cleaned[column])
                if parsed_dates.notna().sum() / non_null_before >= unnamed_date_ratio:
                    if ordering:
                        notes.append(_ordering_note(column, ordering))
                    cleaned[column] = _drop_timezone(parsed_dates)
                    datetime_columns_inferred += 1

    duplicate_rows = int(cleaned.duplicated().sum())
    if drop_duplicates:
        cleaned = cleaned.drop_duplicates()
    cleaned = cleaned.reset_index(drop=True)
    if duplicate_rows and not drop_duplicates:
        notes.append(
            f"{duplicate_rows:,} identical rows were kept; repeat transactions are not "
            "assumed to be mistakes."
        )

    if cleaned.empty or len(cleaned.columns) == 0:
        raise ValueError("No analyzable data remained after removing empty rows and columns.")

    report = CleaningReport(
        original_rows=original_rows,
        original_columns=original_columns,
        final_rows=len(cleaned),
        final_columns=len(cleaned.columns),
        duplicate_rows_removed=duplicate_rows if drop_duplicates else 0,
        empty_rows_removed=empty_rows_removed,
        empty_columns_removed=len(empty_columns),
        index_columns_removed=len(index_columns),
        trimmed_text_columns=trimmed_text_columns,
        numeric_columns_inferred=numeric_columns_inferred,
        datetime_columns_inferred=datetime_columns_inferred,
        duplicate_rows_found=duplicate_rows,
        unparsed_date_cells=unparsed_date_cells,
        notes=tuple(dict.fromkeys(notes)),
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
