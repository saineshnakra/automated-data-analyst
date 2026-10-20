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
    non_finite_cells: int = 0
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Insight:
    title: str
    detail: str
    level: str = "info"


def _speculative_dates(values: pd.Series) -> pd.Series:
    """Parse values as dates, suppressing pandas' "could not infer format" warning.

    Callers probe columns that are usually not dates, so the warning is expected.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return pd.to_datetime(values, errors="coerce")


def _is_blank(series: pd.Series) -> pd.Series:
    """True for missing cells and for text that is only whitespace."""
    blank = series.isna()
    if series.dtype == object or str(series.dtype) == "string":
        blank = blank | series.astype("string").str.strip().eq("").fillna(False)
    return blank


def _drop_timezone(series: pd.Series) -> pd.Series:
    """Return the same instants as naive local time.

    Period boundaries are tz-naive. Converting to UTC first would shift rows east of
    UTC into the previous day, so the wall-clock time from the file is kept.
    """
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        return series.dt.tz_localize(None)
    return series


INT64_LIMIT = float(np.iinfo(np.int64).max)


def _widen_columns_that_would_overflow(frame: pd.DataFrame) -> int:
    """Cast integer columns to float64 when their absolute total would overflow int64.

    numpy sums int64 in int64, so an overflowing total wraps and can flip sign; float
    only loses the last few digits. Returns the number of columns widened.
    """
    widened = 0
    for column in frame.select_dtypes(include=["int64", "int32", "uint64"]).columns:
        magnitude = float(frame[column].astype("float64").abs().sum())
        if magnitude > INT64_LIMIT:
            frame[column] = frame[column].astype("float64")
            widened += 1
    return widened


def _normalize_datetime_columns(frame: pd.DataFrame) -> int:
    """Strip timezones from every tz-aware column; return how many changed."""
    changed = 0
    for column in frame.columns:
        series = frame[column]
        if isinstance(series.dtype, pd.DatetimeTZDtype):
            frame[column] = _drop_timezone(series)
            changed += 1
    return changed


_CURRENCY_CHARS = "$\u20ac\u00a3\u00a5\u20b9"
_SEPARATOR_CHARS = ",. '"
_DIGITS = "0123456789"


def _digit_groups(body: str) -> tuple[list[str], list[str]] | None:
    """Split "1,234.50" into its digit groups and the single marks between them.

    Returns None on a letter, a doubled mark, or a mark without a digit on both
    sides ("1,,000", ",5", "5.").
    """
    groups: list[str] = []
    marks: list[str] = []
    current = ""
    for char in body:
        if char in _DIGITS:
            current += char
        elif char in _SEPARATOR_CHARS and current:
            groups.append(current)
            marks.append(char)
            current = ""
        else:
            return None
    if not current:
        return None
    groups.append(current)
    return groups, marks


def _integer_and_fraction(body: str) -> tuple[str, str] | None:
    """Return the integer and fraction digits of a formatted number, or None if malformed.

    With several marks, the earlier ones must match and are grouping; a different last
    "," or "." is the decimal point. A single "," is a decimal only before one or two
    digits ("1,234" is a thousand); a single "." is always a decimal, as in pd.to_numeric.
    Spaces and apostrophes only group. Integer groups after the first need three digits.
    """
    split = _digit_groups(body)
    if split is None:
        return None
    groups, marks = split
    if not marks:
        return groups[0], ""
    last = marks[-1]
    if len(marks) == 1:
        if last == ",":
            decimal = len(groups[-1]) in (1, 2)
        else:
            decimal = last == "."
    else:
        grouping = set(marks[:-1])
        if len(grouping) != 1:
            return None
        if last in grouping:
            decimal = False
        elif last in ",.":
            decimal = True
        else:
            return None
    integer_groups, fraction = (groups[:-1], groups[-1]) if decimal else (groups, "")
    if any(len(group) != 3 for group in integer_groups[1:]):
        return None
    return "".join(integer_groups), fraction


def _read_formatted_number(text: str) -> float | None:
    """Parse one business-formatted number, or return None.

    A sign, a currency symbol and enclosing parentheses may lead in any order, each at
    most once ("$-100", "-$100", "($100)"). A minus or parentheses make the value
    negative. The remaining digits must pass _integer_and_fraction.
    """
    raw = text.strip()
    negative = False
    sign_seen = currency_seen = parentheses_seen = False
    while raw:
        if raw[0] in "+-":
            if sign_seen:
                return None
            # `or`, not `=`: parentheses may already have set negative ("(+100)").
            sign_seen = True
            negative = negative or raw[0] == "-"
            raw = raw[1:].lstrip()
        elif raw[0] in _CURRENCY_CHARS:
            if currency_seen:
                return None
            currency_seen = True
            raw = raw[1:].lstrip()
        elif raw[0] == "(":
            if parentheses_seen or not raw.endswith(")"):
                return None
            parentheses_seen, negative = True, True
            raw = raw[1:-1].strip()
        else:
            break
    if not raw or raw.endswith(")"):
        return None
    digits = _integer_and_fraction(raw)
    if digits is None:
        return None
    integer, fraction = digits
    value = float(f"{integer}.{fraction or '0'}")
    return -value if negative else value


def _numeric_from_text(values: pd.Series) -> pd.Series:
    """Parse business-formatted numbers cell by cell: "$1,203.55", "(48.10)", "1 234,50".

    Non-string and unreadable cells become NaN.
    """
    return values.map(
        lambda value: _read_formatted_number(value) if isinstance(value, str) else None
    ).astype("float64")


# Day-or-month-first dates such as "3/1/2024" or "03-01-24". Year-first (ISO)
# dates are unambiguous.
_DAY_OR_MONTH_FIRST = re.compile(r"^\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}")


def _dated(values: pd.Series, *, dayfirst: bool) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", (UserWarning, FutureWarning))
        parsed = pd.to_datetime(values, errors="coerce", dayfirst=dayfirst)
        if parsed.dtype == object:
            # Mixed UTC offsets (e.g. across a daylight-saving change) parse to
            # object dtype. There is no single wall clock to keep, so use UTC.
            parsed = pd.to_datetime(values, errors="coerce", dayfirst=dayfirst, utc=True)
            parsed = parsed.dt.tz_localize(None)
        return parsed


def _read_dates(values: pd.Series) -> tuple[pd.Series, str]:
    """Parse a date column, reporting the day/month order when it had to be decided.

    Returns the series and "day-first", "ambiguous" or "". The order that parses more
    values wins ("25/12/2024" only parses day-first). On a tie where the readings
    differ, month-first is used and "ambiguous" is returned so the caller can warn.
    """
    month_first = _dated(values, dayfirst=False)
    # Only day-or-month-first values (01/03/2024, not 2024-03-01) can be read two ways.
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
        # The file may already use a suffixed name ("Sales_2"), so keep counting.
        while candidate in taken:
            counts[base] += 1
            candidate = f"{base}_{counts[base]}"
        taken.add(candidate)
        result.append(candidate)

    return result


def _looks_like_exported_index(series: pd.Series, name: str) -> bool:
    """True for a row-number column left behind by to_csv(index=True).

    Only "Unnamed..." columns qualify, so a named counter is never dropped. Gaps are
    allowed because blank rows are removed before this runs; the values must still
    start at 0 or 1, ascend and stay dense.
    """
    if not name.lower().startswith("unnamed"):
        return False

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any() or not (numeric % 1 == 0).all():
        return False

    values = numeric.to_numpy()
    if values.size == 0 or values[0] not in (0, 1):
        return False
    ascending = bool(np.all(np.diff(values) > 0))
    # Dense enough to be row numbers rather than sparse identifiers.
    dense = values[-1] - values[0] < 2 * len(values)
    return ascending and dense


def _drop_non_finite(frame: pd.DataFrame) -> dict[str, int]:
    """Replace inf and -inf in float columns with NaN in place; return counts per column.

    pandas parses the text "inf" as a number, and one such cell makes every total,
    mean and chart axis on the column infinite.
    """
    counts: dict[str, int] = {}
    for column in frame.select_dtypes(include=["float64", "float32"]).columns:
        infinite = np.isinf(frame[column])
        if infinite.any():
            frame[column] = frame[column].mask(infinite)
            counts[column] = int(infinite.sum())
    return counts


def clean_dataframe(
    dataframe: pd.DataFrame, *, drop_duplicates: bool = False
) -> tuple[pd.DataFrame, CleaningReport]:
    """Clean the frame conservatively and return it with a report of what changed.

    Identical rows are counted but kept, because repeated records are common in
    real data. Pass ``drop_duplicates=True`` to remove them. Raises ValueError if
    the input, or what remains after cleaning, has no rows or columns.
    """
    if dataframe.empty or len(dataframe.columns) == 0:
        raise ValueError("The CSV does not contain any rows and columns to analyze.")

    cleaned = dataframe.copy()
    original_rows, original_columns = cleaned.shape
    cleaned.columns = _make_unique_columns(cleaned.columns)

    empty_columns = [column for column in cleaned.columns if _is_blank(cleaned[column]).all()]
    cleaned = cleaned.drop(columns=empty_columns)

    empty_row_mask = cleaned.apply(_is_blank).all(axis=1)
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
                # Retry cells plain parsing missed as formatted numbers ("$1,203.55",
                # "(48.10)"). Large values are the ones with separators, so skipping
                # them biases totals down. Values already read ("1e3") are kept.
                gaps = parsed_numeric.isna() & cleaned[column].notna()
                if gaps.any():
                    formatted = _numeric_from_text(cleaned[column].where(gaps))
                    parsed_numeric = parsed_numeric.astype("float64").fillna(formatted)
            if parsed_numeric.notna().sum() / non_null_before >= numeric_ratio:
                cleaned[column] = parsed_numeric
                numeric_columns_inferred += 1
                continue

        if not named_like_a_date:
            # Detect dates in columns without a date-like name ("Month", "FY").
            # Numbers are tried first, so a column of bare years stays numeric.
            # A sample is checked first because most text columns are not dates.
            sample = cleaned[column].dropna().head(date_sample_size)
            if len(sample) and _speculative_dates(sample).notna().mean() >= unnamed_date_ratio:
                parsed_dates, ordering = _read_dates(cleaned[column])
                if parsed_dates.notna().sum() / non_null_before >= unnamed_date_ratio:
                    if ordering:
                        notes.append(_ordering_note(column, ordering))
                    cleaned[column] = _drop_timezone(parsed_dates)
                    datetime_columns_inferred += 1

    # Runs after inference, which can create new int64 columns.
    if _widen_columns_that_would_overflow(cleaned):
        notes.append(
            "A column's values are large enough that their total would not fit in a whole "
            "number, so it is measured as a decimal; the last few digits are approximate."
        )

    non_finite = _drop_non_finite(cleaned)
    non_finite_cells = sum(non_finite.values())
    for column, count in non_finite.items():
        noun, verb = ("values", "were") if count > 1 else ("value", "was")
        notes.append(
            f"{count:,} infinite {noun} in {column} {verb} treated as missing; "
            "an infinity is not a quantity the totals can carry."
        )

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
        non_finite_cells=non_finite_cells,
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
