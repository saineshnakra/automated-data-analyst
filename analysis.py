"""Deterministic, local analysis helpers for the Streamlit application."""

from __future__ import annotations

import re
import warnings
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

from schema import is_identifier_name


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
    numeric_cells_unreadable: int = 0
    non_finite_cells: int = 0
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Insight:
    title: str
    detail: str
    level: str = "info"


# A UTC offset at the end of a timestamp: "+01:00", "+0100", "+01", "Z". It
# has to follow a clock time, so the "-01" in "2024-01" is never taken for one.
_TRAILING_OFFSET = re.compile(
    r"^(.*?\d{1,2}:\d{2}(?::\d{2}(?:[.,]\d+)?)?)\s*(?:[Zz]|[+-]\d{2}(?::?\d{2})?)$"
)
OFFSETS_DROPPED_NOTE = (
    "Timezone offsets were dropped; dates are read as the local time they were written in."
)


def _without_offsets(values: pd.Series) -> tuple[pd.Series, int]:
    """Remove the UTC offset from each timestamp before it is parsed.

    Every row keeps the wall clock it was written in. Parsing the offsets
    instead made the result depend on the rest of the column: a file whose
    rows all carried +01:00 kept its local times, but the moment one row
    with +02:00 was appended pandas fell back to UTC and every existing row
    moved an hour -- some of them into the previous month. A date must not
    change because of what came after it.
    """
    carried = values.map(lambda value: isinstance(value, str) and bool(_TRAILING_OFFSET.match(value)))
    dropped = int(carried.sum())
    if not dropped:
        return values, 0
    stripped = values.map(
        lambda value: _TRAILING_OFFSET.sub(r"\1", value) if isinstance(value, str) else value
    )
    return stripped, dropped


def _speculative_dates(values: pd.Series) -> pd.Series:
    """Parse values as dates without complaining about the ones that are not.

    Every call here is a guess about a column whose format is unknown and
    which is usually not dates at all, so pandas' "could not infer format"
    warning is the expected case rather than something to report.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return pd.to_datetime(_without_offsets(values)[0], errors="coerce")


def _is_blank(series: pd.Series) -> pd.Series:
    """Missing, or text that is nothing but whitespace -- an empty cell either way."""
    blank = series.isna()
    if series.dtype == object or str(series.dtype) == "string":
        blank = blank | series.astype("string").str.strip().eq("").fillna(False)
    return blank


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


INT64_LIMIT = float(np.iinfo(np.int64).max)
# The largest whole number a float holds exactly; above it, neighbours merge.
EXACT_FLOAT_LIMIT = float(2**53)


def _widen_columns_that_would_overflow(frame: pd.DataFrame) -> int:
    """Move an integer column to float when its own total will not fit.

    numpy sums int64 in int64, so a strictly positive ledger whose total
    passes 9.2e18 wraps round to a negative number and every figure built on
    it is nonsense. Float loses precision in the last few digits; wrapping
    loses the sign. Only columns that would actually overflow are touched, so
    ordinary integers keep their type.
    """
    widened = 0
    for column in frame.select_dtypes(include=["int64", "int32", "uint64"]).columns:
        magnitude = float(frame[column].astype("float64").abs().sum())
        if magnitude > INT64_LIMIT:
            frame[column] = frame[column].astype("float64")
            widened += 1
    return widened


def _normalize_datetime_columns(frame: pd.DataFrame) -> int:
    """Strip timezones from every date column, whatever put them there."""
    changed = 0
    for column in frame.columns:
        series = frame[column]
        if isinstance(series.dtype, pd.DatetimeTZDtype):
            frame[column] = _drop_timezone(series)
            changed += 1
    return changed


def _drop_non_finite(frame: pd.DataFrame) -> dict[str, int]:
    """Turn inf and -inf into missing, and say how many cells that was.

    pandas reads the text "inf" as a number, and one such cell then owns
    every total, mean and chart axis built on the column. It carries no
    quantity a business file could mean, so it is treated as a value that
    could not be read.
    """
    counts = {}
    for column in frame.select_dtypes(include=["float64", "float32"]).columns:
        infinite = np.isinf(frame[column])
        if infinite.any():
            frame[column] = frame[column].mask(infinite)
            counts[column] = int(infinite.sum())
    return counts


_CURRENCY_CHARS = "$\u20ac\u00a3\u00a5\u20b9"
_SEPARATOR_CHARS = ",. '"
_DIGITS = "0123456789"


def _digit_groups(body: str) -> tuple[list[str], list[str]] | None:
    """Split "1,234.50" into its digit groups and the single marks between them.

    Anything else -- a letter, a doubled mark, a mark with no digit on one
    side -- is not a number, and saying so is the whole point: "(100",
    "1,2,3" and "12 34" used to come out as 100, 123 and 1234.
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
    """Read the digits of a formatted number, deciding which mark is the decimal.

    When both "," and "." appear, the one that comes last is the decimal
    point and the other is grouping. When only one mark appears once, three
    digits after a comma is grouping ("1,000") and one or two is a decimal
    ("1234,50"); "1,234" alone reads as a thousand, which is what pandas and
    every US export mean by it, and a single "." is always a decimal, as
    pd.to_numeric reads it. A space or an apostrophe is only ever grouping.
    Grouping marks must then delimit groups of exactly three digits, so a
    stray mark is refused rather than read past.
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
    """Read one business-formatted number, or refuse it.

    The sign is settled first and never touched again: a leading minus, a
    leading plus, or accounting parentheses. An earlier version of this
    stripped "everything that is not a digit" and took the minus sign with it,
    so refunds became revenue. That is the one mistake this function must not
    be able to make again, whatever else it gets wrong.

    A sign, a currency symbol and a pair of parentheses may lead the number
    in any order -- "$-100" and "-$100" are both a hundred owed -- but each
    at most once, and an opening parenthesis must have its closing one. The
    digits then have to satisfy _integer_and_fraction.
    """
    raw = text.strip()
    negative = False
    sign_seen = currency_seen = parentheses_seen = False
    while raw:
        if raw[0] in "+-":
            if sign_seen:
                return None
            sign_seen, negative = True, raw[0] == "-"
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
    """Read business-formatted numbers cell by cell: "$1,203.55", "(48.10)", "1 234,50".

    A thousands separator is punctuation, not data, and an amount in
    parentheses is how every accounting export writes a negative. Reading
    them as text loses the largest values in the file, which are exactly the
    ones with a separator in them.
    """
    return values.map(
        lambda value: _read_formatted_number(value) if isinstance(value, str) else None
    ).astype("float64")


def _infer_numeric(values: pd.Series) -> tuple[pd.Series, bool]:
    """Plain parsing first, then the formatted reader for whatever it left.

    A value plain parsing already read ("1e3") is never replaced. The second
    result says whether the formatted reader contributed anything, because
    when it did the column is a float whatever the plain values were, and a
    whole number past 2**53 has by then lost its last digits.
    """
    parsed = pd.to_numeric(values, errors="coerce")
    gaps = parsed.isna() & values.notna()
    if not gaps.any():
        return parsed, False
    formatted = _numeric_from_text(values.where(gaps))
    if not formatted.notna().any():
        return parsed, False
    return parsed.astype("float64").fillna(formatted), True


# A date written day-or-month first: "3/1/2024", "03-01-24". A year-leading
# value is ISO and cannot be read two ways.
_DAY_OR_MONTH_FIRST = re.compile(r"^\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}")


def _dated(values: pd.Series, *, dayfirst: bool) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", (UserWarning, FutureWarning))
        parsed = pd.to_datetime(values, errors="coerce", dayfirst=dayfirst)
        if parsed.dtype == object:
            # Zones written in a form the offset stripper does not know, and
            # not all the same, come back as objects no downstream step can
            # use. Each keeps its own wall clock, the same policy as the
            # rest of the column, rather than being moved to UTC.
            parsed = pd.to_datetime(
                parsed.map(
                    lambda moment: moment.tz_localize(None)
                    if isinstance(moment, pd.Timestamp) and moment.tzinfo is not None
                    else moment
                ),
                errors="coerce",
            )
        return parsed


def _read_dates(values: pd.Series) -> tuple[pd.Series, str, int]:
    """Parse a date column, and say so when the ordering had to be guessed.

    The third result is how many values carried a UTC offset, which was
    removed before parsing so that each row keeps its own wall clock.

    "01/03/2024" is 1 March in most of the world and 3 January in the United
    States. When some row in the column settles it -- a 13 or higher in the
    first position -- that reading wins outright. When nothing settles it,
    month-first is assumed and the assumption is reported, because silently
    turning a year of monthly figures into twelve days of January is the one
    outcome nobody can detect downstream.
    """
    values, offsets = _without_offsets(values)
    month_first = _dated(values, dayfirst=False)
    # Only a value that leads with a day or a month can be read two ways.
    # 2024-03-01 is ISO and settled; 01/03/2024 is not.
    two_ways = values.astype("string").str.match(_DAY_OR_MONTH_FIRST, na=False)
    if not bool(two_ways.any()):
        return month_first, "", offsets

    day_first = _dated(values, dayfirst=True)
    if day_first.notna().sum() > month_first.notna().sum():
        return day_first, "day-first", offsets
    if month_first.notna().sum() > day_first.notna().sum():
        return month_first, "", offsets
    disagree = two_ways & month_first.notna() & day_first.notna() & month_first.ne(day_first)
    return month_first, "ambiguous" if bool(disagree.any()) else "", offsets


def _non_finite_note(column: str, count: int) -> str:
    return f"{count} {column} values were infinite and were left missing."


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
    """A stray row-number column left behind by someone's to_csv(index=True).

    Gaps are allowed. Blank rows are dropped before this runs, and each one
    takes a number out of the sequence -- a single blank line was enough to
    leave the row numbers in place and let them become the file's headline
    metric. The "unnamed" prefix is what makes this safe to be lenient about;
    a counter with a real name is never touched.
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
    # Still has to look like row numbers rather than sparse identifiers.
    dense = values[-1] - values[0] < 2 * len(values)
    return ascending and dense


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
    numeric_cells_unreadable = 0
    non_finite_cells = 0
    notes: list[str] = []

    if datetime_columns_inferred:
        notes.append(OFFSETS_DROPPED_NOTE)

    date_tokens = ("date", "time", "timestamp", "created", "updated")
    named_date_ratio, numeric_ratio, unnamed_date_ratio = 0.8, 0.95, 0.95
    date_sample_size = 50

    for column in cleaned.select_dtypes(include=["object", "string"]).columns:
        series = cleaned[column]
        cleaned[column] = series.map(lambda value: value.strip() if isinstance(value, str) else value)
        cleaned[column] = cleaned[column].replace("", pd.NA)
        trimmed_text_columns += 1

        # Counted after trimming. A whitespace-only cell is a blank, and
        # counting it as a value put a column of "$100", "$200", "  " under
        # the bar, so it stayed text.
        non_null = int(cleaned[column].notna().sum())
        if non_null == 0:
            continue

        normalized_name = column.lower()
        named_like_a_date = any(token in normalized_name for token in date_tokens)
        # A column named as a key holds neither a quantity to add up nor a
        # moment to bucket, whatever its values look like: "Customer ID"
        # full of 2024-01-01 is not the file's date. The reader kept these
        # as text by the same test, so what it preserved is not undone here.
        protected = is_identifier_name(column)
        if named_like_a_date:
            parsed_dates, ordering, offsets = _read_dates(cleaned[column])
            if parsed_dates.notna().sum() / non_null >= named_date_ratio:
                unreadable = int(non_null - parsed_dates.notna().sum())
                if unreadable:
                    unparsed_date_cells += unreadable
                    notes.append(f"{unreadable} {column} values could not be read as dates.")
                if ordering:
                    notes.append(_ordering_note(column, ordering))
                if offsets:
                    notes.append(OFFSETS_DROPPED_NOTE)
                cleaned[column] = _drop_timezone(parsed_dates)
                datetime_columns_inferred += 1
                continue

        if not protected:
            # Whatever plain parsing could not read is tried again allowing
            # the punctuation a finance export writes: grouping separators, a
            # currency symbol, parentheses for a negative. The values that
            # need it are the large ones, so leaving them out biases every
            # total downwards.
            parsed_numeric, formatted_used = _infer_numeric(cleaned[column])
            infinite = np.isinf(parsed_numeric)
            parsed_numeric = parsed_numeric.mask(infinite)
            readable = int(parsed_numeric.notna().sum())
            if readable / non_null >= numeric_ratio:
                unreadable = non_null - readable - int(infinite.sum())
                if unreadable:
                    numeric_cells_unreadable += unreadable
                    notes.append(
                        f"{unreadable} {column} values could not be read as numbers and were left missing."
                    )
                if infinite.any():
                    non_finite_cells += int(infinite.sum())
                    notes.append(_non_finite_note(column, int(infinite.sum())))
                if formatted_used and bool((parsed_numeric.abs() >= EXACT_FLOAT_LIMIT).any()):
                    notes.append(
                        f"{column} holds whole numbers that would not fit exactly in a decimal, "
                        "which reading its formatted values required; the last few digits are "
                        "approximate."
                    )
                cleaned[column] = parsed_numeric
                numeric_columns_inferred += 1
                continue

        if not named_like_a_date and not protected:
            # A column holds dates whatever it happens to be called -- "Month",
            # "Period", "FY". Numbers were tried first, so a column of bare years
            # stays numeric instead of becoming the 1st of January in each of
            # them. A sample decides whether the full parse is worth attempting,
            # because most text columns are not dates and this runs over all of
            # them.
            sample = cleaned[column].dropna().head(date_sample_size)
            if len(sample) and _speculative_dates(sample).notna().mean() >= unnamed_date_ratio:
                parsed_dates, ordering, offsets = _read_dates(cleaned[column])
                if parsed_dates.notna().sum() / non_null >= unnamed_date_ratio:
                    if ordering:
                        notes.append(_ordering_note(column, ordering))
                    if offsets:
                        notes.append(OFFSETS_DROPPED_NOTE)
                    cleaned[column] = _drop_timezone(parsed_dates)
                    datetime_columns_inferred += 1

    # A column the file already delivered as numbers can carry inf too;
    # read_csv reads the text "inf" as one before cleaning ever sees it.
    for column, count in _drop_non_finite(cleaned).items():
        non_finite_cells += count
        notes.append(_non_finite_note(column, count))

    # Inference can produce a new int64 column, so the overflow check runs
    # once everything that will be numeric already is.
    if _widen_columns_that_would_overflow(cleaned):
        notes.append(
            "A column's values are large enough that their total would not fit in a whole "
            "number, so it is measured as a decimal; the last few digits are approximate."
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
        numeric_cells_unreadable=numeric_cells_unreadable,
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
