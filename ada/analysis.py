"""Deterministic, local analysis helpers for the Streamlit application."""

from __future__ import annotations

import datetime as dt
import re
import unicodedata
import warnings
from dataclasses import asdict, dataclass
from functools import cached_property
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype

from .formatting import UNDATED_DAY, format_moment, normalized_name
from .numeric import to_numbers
from .quality import QualityIssue, blank_codes
from .schema import finer_part_name, is_identifier_name, is_month_name, is_year_name


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
    #: What the data checks list for the missing-value codes set to missing
    #: (see ``quality.blank_codes``); pass it to ``quality.scan`` as ``blanked``.
    blanked_codes: tuple[QualityIssue, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Insight:
    title: str
    detail: str
    level: str = "info"


# Text that stands for a missing reading in a numeric column: R's NA, "not
# detected", spreadsheet errors, and values below or above a detection limit.
MISSING_MARKERS = frozenset({
    "na", "n/a", "n.a.", "nan", "nd", "n.d.", "none", "null", "missing", "unknown", "-", "--", "—",
    "?", ".", "#n/a", "#value!", "#div/0!", "not detected", "bdl", "bql", "lod", "loq",
    "<lod", "<loq", "<dl", "below detection limit",
})
# Phrases for "not stated" or "no data" in the languages exports most often
# arrive in, and a spreadsheet's lookup error. They are missing values in a text
# column as much as in a numeric one, and left as labels they become a group of
# their own.
NOT_STATED_MARKERS = frozenset({
    "n/a", "#n/a", "error: #n/a", "not stated", "not specified", "not indicated", "not available",
    "no data", "no indica", "no indicado", "no indicada", "sin dato", "sin datos", "sin información",
    "sin informacion", "nicht angegeben", "keine angabe", "keine angaben", "keine daten",
    "non renseigné", "non renseigne", "pas de données", "pas de donnees", "aucune donnée",
    "aucune donnee", "nessun dato", "dato mancante", "dati mancanti", "sem dados", "sem informação",
    "sem informacao", "não informado", "nao informado", "tidak ada data", "data tidak tersedia",
    "geen gegevens", "brak danych", "ingen data",
})
# Labels a coder writes when no value could be given. Left as labels, they
# become a group that mixes every kind of row.
PLACEHOLDER_LABELS = frozenset({
    "unknown", "unclassified", "not classified", "unable to classify", "unclassifiable",
})
# "Not detected": a reading below what the method sees, written without a number.
NOT_DETECTED_MARKERS = frozenset({"nd", "n.d."})
# Censoring markers without a number: the reading was below what the method sees.
BELOW_LIMIT_MARKERS = frozenset({
    "<lod", "<loq", "<dl", "bdl", "bql", "below detection limit", "not detected",
}) | NOT_DETECTED_MARKERS
# "<0.5" or "<3,87": a reading below, or above, a detection limit.
_CENSORED_VALUE = re.compile(r"^[<>]\s*[\d.,]")
# A lone token that means a trace amount when it is the column's only text.
TRACE_TOKENS = frozenset({"t", "tr", "trace"})
# Cells drawn at random from a text column to rule out a type before parsing all
# of it. A column needs 95% of its cells to convert, so one where under half the
# drawn cells read as numbers (or times) cannot, and is not parsed in full.
PROBE_SIZE = 200
PROBE_RATIO = 0.5


def _normalized_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.lower().str.replace(r"\s+", " ", regex=True)


def _flags(matches: pd.Series) -> np.ndarray:
    """A True/False array from a match that may hold missing values."""
    return matches.fillna(False).to_numpy(dtype=bool)


class _Distinct:
    """A column's distinct values, and which of them each row holds.

    Text columns repeat a few values many times (a site, a status, a date), so
    every check on one (markers, numbers, dates, labels) runs once per distinct
    value, on text normalised once, and is spread back over the rows by
    position. That keeps cleaning a large file fast.
    """

    def __init__(self, index: pd.Index, codes: np.ndarray, values: pd.Series):
        self.index = index
        #: Position in ``values`` of each row's value; -1 where the row is missing.
        self.codes = codes
        self.values = values

    @classmethod
    def of(cls, series: pd.Series) -> _Distinct:
        codes, uniques = pd.factorize(series, use_na_sentinel=True)
        return cls(series.index, codes, pd.Series(np.asarray(uniques, dtype=object), dtype=object))

    @cached_property
    def text(self) -> pd.Series:
        """Each value stripped, lower-cased and with runs of spaces closed up."""
        return _normalized_text(self.values)

    @cached_property
    def counts(self) -> np.ndarray:
        """How many rows hold each value."""
        return np.bincount(self.codes[self.codes >= 0], minlength=len(self.values))

    @property
    def present(self) -> int:
        return int(self.counts.sum())

    @cached_property
    def cells(self) -> pd.Series:
        """The distinct values, then one missing value for the missing rows to map to, if any are.

        Without missing rows nothing is added, so whole numbers parse as whole numbers.
        """
        if not (self.codes < 0).any():
            return self.values
        return pd.concat([self.values, pd.Series([None], dtype=object)], ignore_index=True)

    def on_cells(self, flags: np.ndarray) -> np.ndarray:
        """One flag per distinct value, lined up with ``cells``."""
        return np.append(flags, False) if len(self.cells) > len(self.values) else flags

    def expand(self, result: pd.Series) -> pd.Series:
        """Spread a result worked out on ``cells`` over the rows."""
        return result.iloc[self.codes].set_axis(self.index)

    def spread(self, transform) -> pd.Series:
        """Run ``transform`` (a Series in, a Series out) on ``cells`` and spread it over the rows."""
        return self.expand(transform(self.cells))

    def rows(self, hits: np.ndarray) -> pd.Series:
        """True for the rows whose value ``hits`` marks (one flag per distinct value)."""
        flags = np.append(np.asarray(hits, dtype=bool), False)
        return pd.Series(flags[self.codes], index=self.index)

    def column(self) -> pd.Series:
        return self.spread(lambda cells: cells)

    def relabel(self, values: pd.Series) -> _Distinct:
        """The same rows holding ``values`` in place of the distinct values.

        Values that become equal merge, those that become missing leave their
        rows missing, and values no row holds are dropped.
        """
        inner, uniques = pd.factorize(values.where(self.counts > 0), use_na_sentinel=True)
        codes = np.append(inner, -1)[self.codes]
        return _Distinct(self.index, codes, pd.Series(np.asarray(uniques, dtype=object), dtype=object))

    def take(self, keep: np.ndarray) -> _Distinct:
        """The rows where ``keep`` is True."""
        return _Distinct(self.index[keep], self.codes[keep], self.values)

    def _at(self, positions: np.ndarray) -> pd.Series:
        return pd.Series(
            self.values.to_numpy()[self.codes[positions]], index=self.index[positions], dtype=object
        )

    def head(self, size: int) -> pd.Series:
        """The first ``size`` values present, in row order."""
        return self._at(np.flatnonzero(self.codes >= 0)[:size])

    def sample(self, size: int = PROBE_SIZE) -> pd.Series:
        """``size`` values present, drawn at random with a fixed seed, or all of them when fewer."""
        positions = np.flatnonzero(self.codes >= 0)
        if len(positions) > size:
            positions = positions[np.random.RandomState(0).choice(len(positions), size, replace=False)]
        return self._at(positions)


def _per_value(series: pd.Series, transform) -> pd.Series:
    """Run ``transform`` once per distinct value and spread the result back over the rows.

    ``transform`` takes and returns a Series; missing cells reach it as one missing value.
    """
    return _Distinct.of(series).spread(transform)


def _missing_markers(distinct: _Distinct) -> np.ndarray:
    """True for distinct values that are missing-value markers such as "NA", "nd", "no indica" or "<0.1"."""
    text = distinct.text
    return _flags(text.isin(MISSING_MARKERS | NOT_STATED_MARKERS) | text.str.match(_CENSORED_VALUE.pattern))


def _censored(distinct: _Distinct) -> tuple[np.ndarray, np.ndarray]:
    """Distinct values below a detection limit, and those above one ("<0.1", "BDL", "ND"; ">12,88")."""
    text = distinct.text
    below = text.str.match(r"^<\s*[\d.,]") | text.isin(BELOW_LIMIT_MARKERS)
    return _flags(below), _flags(text.str.match(r"^>\s*[\d.,]"))


def _limits(values: pd.Series) -> list[float]:
    """The detection limits written in censored cells, parsed like any number: "<3,87" is 3.87."""
    numbers = values.astype("string").str.strip().str.lstrip("<>").str.strip()
    parsed = {_read_formatted_number(number) for number in numbers.dropna().unique()}
    return sorted(value for value in parsed if value is not None)


def _censored_note(column: str, distinct: _Distinct, censored: np.ndarray, side: str, total: int) -> str:
    """Say how many cells were past a detection limit on one ``side``, and which way that biases the mean.

    ``censored`` marks distinct values; ``side`` is "below" or "above".
    """
    count = int(distinct.counts[censored].sum())
    limits = _limits(distinct.values[censored])
    shown = ""
    if limits:
        mark = "<" if side == "below" else ">"
        low, high = (f"{mark}{limit:g}" for limit in (limits[0], limits[-1]))
        shown = f" ({low})" if low == high else f" ({low} to {high})"
    share = count / max(total, 1) * 100
    portion = "under 0.1%" if share < 0.05 else f"{share:.1f}%"
    subject = f"{count:,} cells in {column} ({portion}) were" if count > 1 else f"1 cell in {column} was"
    note = f"{subject} {side} a detection limit{shown} and read as missing"
    if share >= CENSORED_SHARE * 100:
        direction = "upward" if side == "below" else "downward"
        note += f"; with this many left out, the mean of the rest is biased {direction}"
    return note + "."


#: Share of a column past a detection limit at which its note warns that the mean is biased.
CENSORED_SHARE = 0.10
#: A ">X" limit that matches another column on the same row (within
#: REACHED_TOLERANCE of it) on REACHED_SHARE of at least REACHED_ROWS rows
#: is that column's value: the reading reached it.
REACHED_ROWS = 5
REACHED_SHARE = 0.8
REACHED_TOLERANCE = 0.1


def _upper_limits(distinct: _Distinct, markers: np.ndarray) -> pd.Series:
    """The limit in each row written as above one (">3.5" gives 3.5), for those rows only."""
    above = _censored(distinct)[1] & markers
    limits = np.full(len(distinct.values) + 1, np.nan)
    for position in np.flatnonzero(above):
        read = _read_formatted_number(str(distinct.values.iloc[position]).strip().lstrip("<>").strip())
        limits[position] = np.nan if read is None else read
    return pd.Series(limits[distinct.codes], index=distinct.index).dropna()


def _reached_note(frame: pd.DataFrame, column: str, limits: pd.Series) -> str | None:
    """Say so when the limit in a column's ">X" cells is another column's value on the same row.

    Water clarity written ">3.5" at a site 3.5 m deep means the disk was still
    seen on the bottom: the limit is the depth, and the clarity is at least
    that. The limits must vary, so one shared limit matching a constant
    column is not taken for this.
    """
    if len(limits) < REACHED_ROWS or limits.nunique() < 3:
        return None
    best = None
    for other in frame.columns:
        series = frame[other]
        if other == column or is_bool_dtype(series) or not is_numeric_dtype(series):
            continue
        values = series.reindex(limits.index).astype("float64")
        held = values.notna().to_numpy() & (values > 0).to_numpy()
        if held.sum() < REACHED_ROWS:
            continue
        close = int(((limits[held] - values[held]).abs() <= REACHED_TOLERANCE * values[held]).sum())
        if close >= REACHED_SHARE * held.sum() and (best is None or close > best[1]):
            best = (other, close, int(held.sum()))
    if best is None:
        return None
    other, close, held = best
    return (
        f'On {close:,} of the {held:,} rows where {column} is written as ">X" and {other} is recorded, X is '
        f"{other} (within {REACHED_TOLERANCE:.0%}): the reading reached {other}, so its true value is at "
        "least that."
    )


def _lone_token(distinct: _Distinct, unread: np.ndarray) -> str | None:
    """The single text token left in an otherwise numeric column, when it reads as a marker.

    Weather records write "T" for a trace of rain and instrument logs write "M"
    or "*": one short token, repeated, with numbers everywhere else. A longer
    word ("Control" among doses) is a real level and is not returned. ``unread``
    marks the distinct values that did not read as numbers.
    """
    if not unread.any():
        return None
    tokens = distinct.text[unread].dropna().unique()
    if len(tokens) != 1:
        return None
    token = str(tokens[0])
    if token in TRACE_TOKENS or (len(token) <= 3 and not any(char.isdigit() for char in token)):
        return token
    return None


def _not_detected(distinct: _Distinct) -> np.ndarray:
    """True for distinct values that read "ND" or "n.d.".

    "ND" is left alone in a column of two-letter codes, where it is North Dakota.
    """
    text = distinct.text
    hits = _flags(text.isin(NOT_DETECTED_MARKERS))
    if hits.any() and text.str.fullmatch(r"[a-z]{2}").mean() >= 0.5:
        return np.zeros(len(text), dtype=bool)
    return hits


# A cell that starts with a number, as a reading written with its unit does ("5.66 ± 1.69 items/g").
_READING = re.compile(r"^[<>~≈±]?\s*\d")


def _holds_readings(distinct: _Distinct, markers: np.ndarray) -> bool:
    """Whether most cells other than ``markers`` start with a number: readings written as text."""
    counts = distinct.counts
    readable = int(counts[~markers].sum())
    readings = _flags(distinct.text.str.match(_READING.pattern)) & ~markers
    return readable > 0 and int(counts[readings].sum()) >= 0.5 * readable


def _marked(distinct: _Distinct, hits: np.ndarray) -> tuple[list[str], int]:
    """The marked values as written, in the order they first appear, and how many cells hold them."""
    shown = list(dict.fromkeys(str(value).strip() for value in distinct.values[hits]))
    return shown, int(distinct.counts[hits].sum())


def _unstated(column: str, distinct: _Distinct) -> tuple[np.ndarray, list[str]]:
    """Distinct values of a text column that stand for no value, with a note for each kind.

    "Not stated" phrases, "ND" (not detected) and placeholder labels such as
    "Unknown" or "UNABLE TO CLASSIFY" are missing values, not groups of their own.
    So is "NA" among words ("male", "female", "NA"), though not among codes, where
    it is a code of its own: a region (North America) or a country (Namibia).
    """
    text = distinct.text
    stated = _flags(text.isin(NOT_STATED_MARKERS))
    detected = _not_detected(distinct)
    placeholders = _flags(text.isin(PLACEHOLDER_LABELS))
    bare = _bare_na(distinct) & ~stated & ~detected & ~placeholders
    notes = []
    if bare.any():
        count = int(distinct.counts[bare].sum())
        noun, verb = ("cells", "were") if count > 1 else ("cell", "was")
        notes.append(
            f'{count:,} {noun} in {column} held "NA" and {verb} read as missing: the column\'s other '
            "values are words, not codes, so it is not a code such as NA for North America."
        )
    if stated.any():
        shown, count = _marked(distinct, stated)
        phrases = "This phrase means" if len(shown) == 1 else "These phrases mean"
        notes.append(f"{_missing_marker_note(column, shown, count)} {phrases} the value was not stated.")
    if detected.any():
        shown, count = _marked(distinct, detected)
        if _holds_readings(distinct, stated | detected | placeholders):
            meaning = "among readings it means not detected, below what the method can measure"
        else:
            meaning = "it usually means not detected or not determined"
        noun, verb = ("cells", "were") if count > 1 else ("cell", "was")
        notes.append(f'{count:,} {noun} in {column} held "{shown[0]}" and {verb} read as missing: {meaning}.')
    if placeholders.any():
        shown, count = _marked(distinct, placeholders)
        labels = "This label stands" if len(shown) == 1 else "These labels stand"
        notes.append(
            f"{_missing_marker_note(column, shown, count)} {labels} for a value nobody could give, "
            "not a group of its own."
        )
    return stated | detected | placeholders | bare, notes


# A label this short, or in capitals up to CODE_LENGTH long, is a code ("EU", "APAC", "No").
SHORT_LABEL = 3
CODE_LENGTH = 5


def _bare_na(distinct: _Distinct) -> np.ndarray:
    """Distinct values written "NA" or "na" in a column whose other values are mostly words.

    Among codes (half or more of the other values short or in capitals: "EU",
    "APAC", "No") "NA" is kept as a code. "Na", the symbol for sodium, is never matched.
    """
    written = distinct.values.astype(str).str.strip()
    bare = written.isin(("NA", "na")).to_numpy()
    others = written[~bare & distinct.values.notna().to_numpy()]
    if not bare.any() or others.empty:
        return np.zeros(len(written), dtype=bool)
    codes = (others.str.len() <= SHORT_LABEL) | others.str.fullmatch(rf"[A-Z]{{2,{CODE_LENGTH}}}")
    return bare if float(codes.mean()) < 0.5 else np.zeros(len(written), dtype=bool)


# "10-30" or "0.5 - 2": a numeric range written as text, with the mark between its ends.
_NUMERIC_RANGE = re.compile(r"^(\d+(?:\.\d+)?)(\s*[-–]\s*)(\d+(?:\.\d+)?)$")
# "Oct-30" or "30-Oct": how a spreadsheet shows "10-30" once it has read it as a date.
_MONTH_AND_NUMBER = re.compile(r"^(?:([A-Za-z]{3})-(\d{1,2})|(\d{1,2})-([A-Za-z]{3}))$")
# A column of ranges (depth bands, age bands) has few labels; one with more is not checked.
MAX_RANGE_LABELS = 200


def _mangled_ranges(column: str, distinct: _Distinct) -> tuple[pd.Series | None, str | None]:
    """Find numeric ranges ("10-30") that a spreadsheet turned into dates ("Oct-30").

    Only a column made mostly of numeric ranges is checked, and such a column
    has few labels. The likely original is the month's number and the other
    number, low end first. When that range shares an end with another in the
    column ("0-10", "10-30", "30-60"), the values are restored to it; otherwise
    they are only named in the note. Returns the restored values (or None) and
    the note (or None).
    """
    if len(distinct.values) > MAX_RANGE_LABELS:
        return None, None
    text = distinct.values.astype("string").str.strip()
    ranges = text.str.extract(_NUMERIC_RANGE)
    parts = text.str.extract(_MONTH_AND_NUMBER)
    month = parts[0].fillna(parts[3]).str.lower().map(MONTH_NUMBERS)
    number = to_numbers(parts[1].fillna(parts[2]))
    dated = month.notna() & number.notna()
    is_range = ranges[0].notna()
    if not dated.any() or is_range.sum() < 2 or is_range.sum() < 0.5 * (~dated).sum():
        return None, None
    ends = set(to_numbers(ranges[0]).dropna()) | set(
        to_numbers(ranges[2]).dropna()
    )
    mark = str(ranges[1][is_range].iloc[0])
    originals: dict[str, str] = {}
    chained = True
    for position in np.flatnonzero(dated.to_numpy()):
        low, high = sorted((int(month.iloc[position]), int(number.iloc[position])))
        originals[str(distinct.values.iloc[position])] = f"{low}{mark}{high}"
        chained &= low in ends or high in ends
    count = int(distinct.counts[dated.to_numpy()].sum())
    written = " or ".join(f'"{value}"' for value in originals)
    restored = " and ".join(f'"{value}"' for value in dict.fromkeys(originals.values()))
    one = len(set(originals.values())) == 1
    subject = f"{count:,} {column} values read" if count > 1 else f"1 {column} value reads"
    shows = (
        f"how a spreadsheet shows the range {restored} once it has read it as a date"
        if one
        else f"how a spreadsheet shows the ranges {restored} once it has read them as dates"
    )
    pronoun = "they were" if count > 1 else "it was"
    if not chained:
        return None, f"{subject} {written}, which may be {shows}; {pronoun} left as written."
    target = restored if one else "those ranges"
    note = f"{subject} {written}, which is {shows}; {pronoun} restored to {target}."
    return distinct.values.map(lambda value: originals.get(str(value), value)), note


_MANGLED_DUPLICATE = re.compile(r"^(.+)\.(\d+)$")


def _duplicate_header_notes(original: pd.Index, frame: pd.DataFrame) -> list[str]:
    """Name each repeated header and the name its copy now has, and say whether they differ.

    A reader renames a repeated "diabetes" to "diabetes.1"; ADA renames one in a
    DataFrame to "diabetes_2". Either way the two columns may hold different data.
    """
    notes = []
    renamed: list[tuple[str, str]] = []
    names = [str(name) for name in frame.columns]
    for name in names:
        match = _MANGLED_DUPLICATE.match(name)
        # Numbers as headers ("400" and "400.5" nm) are not a repeat.
        if match and match.group(1) in names and any(char.isalpha() for char in match.group(1)):
            renamed.append((match.group(1), name))
    seen: dict[str, int] = {}
    for position, raw_name in enumerate(original):
        base = " ".join(str(raw_name).strip().split()) or f"column_{position + 1}"
        seen[base] = seen.get(base, 0) + 1
        if seen[base] > 1:
            renamed.append((base, names[position]))
    for first, copy in renamed:
        a, b = frame[first], frame[copy]
        if not (is_numeric_dtype(a) and is_numeric_dtype(b)):
            a, b = a.astype("string"), b.astype("string")
        differ = int((a.ne(b) & ~(a.isna() & b.isna())).sum())
        rows = "row" if differ == 1 else "rows"
        same = "They hold the same values." if differ == 0 else f"They differ on {differ:,} {rows}."
        notes.append(f'The header "{first}" appears more than once; the repeat is named "{copy}". {same}')
    return notes


def _dropped_columns_note(columns: list[str]) -> str:
    shown = [str(column) for column in columns]
    if len(shown) > 8:
        shown = [*shown[:8], f"{len(shown) - 8} more"]
    listed = shown[0] if len(shown) == 1 else f"{', '.join(shown[:-1])} and {shown[-1]}"
    noun = "columns were" if len(columns) > 1 else "column was"
    return f"{len(columns)} empty {noun} dropped: {listed}."


def _speculative_dates(values: pd.Series) -> pd.Series:
    """Parse values as dates, suppressing pandas' "could not infer format" warning.

    Callers probe columns that are usually not dates, so the warning is expected.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return pd.to_datetime(values, errors="coerce")


def _text_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column, kind in frame.dtypes.items() if str(kind) in ("object", "string")]


def _blank_text(distinct: _Distinct) -> np.ndarray:
    """True for the rows of a text column that are missing or only whitespace."""
    empty = _flags(distinct.values.astype("string").str.strip().eq(""))
    return ~distinct.rows(~empty).to_numpy()


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


def _integer_and_fraction(body: str, decimal_comma: bool = False) -> tuple[str, str] | None:
    """Return the integer and fraction digits of a formatted number, or None if malformed.

    With several marks, the earlier ones must match and are grouping; a different last
    "," or "." is the decimal point. A single "," is a decimal before one or two digits,
    after a lone 0 ("0,307"), or anywhere in a column that writes its decimals with
    commas (``decimal_comma``); otherwise "1,234" is a thousand. A single "." is a
    decimal, as in pd.to_numeric, except before three digits in such a column, where
    "1.234" is a thousand. Spaces and apostrophes only group. Integer groups after the
    first need three digits.
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
            decimal = decimal_comma or len(groups[-1]) in (1, 2) or groups[0] == "0"
        else:
            # Where commas are the decimals, "1.234" groups a thousand.
            decimal = last == "." and not (decimal_comma and len(groups[-1]) == 3)
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


def _read_formatted_number(text: str, decimal_comma: bool = False) -> float | None:
    """Parse one business-formatted number, or return None.

    A sign, a currency symbol and enclosing parentheses may lead in any order, each at
    most once ("$-100", "-$100", "($100)"). A minus or parentheses make the value
    negative. The remaining digits must pass _integer_and_fraction, which reads a
    single comma as the decimal point when ``decimal_comma`` says the column does.
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
    digits = _integer_and_fraction(raw, decimal_comma)
    if digits is None:
        return None
    integer, fraction = digits
    value = float(f"{integer}.{fraction or '0'}")
    return -value if negative else value


def _numeric_from_text(values: pd.Series) -> pd.Series:
    """Parse business-formatted numbers cell by cell: "$1,203.55", "(48.10)", "1 234,50".

    Non-string and unreadable cells become NaN.
    """
    return _per_value(
        values,
        lambda distinct: distinct.map(
            lambda value: _read_formatted_number(value) if isinstance(value, str) else None
        ).astype("float64"),
    )


# Day-or-month-first dates such as "3/1/2024" or "03-01-24". Year-first (ISO)
# dates are unambiguous.
_DAY_OR_MONTH_FIRST = re.compile(r"^\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}")
_ISO_DATE = r"^\s*\d{4}-\d{1,2}-\d{1,2}"
# A time of day with no date: "15:47", "15:47:03.250", "3:05 PM".
_TIME_OF_DAY = re.compile(r"^\s*\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?\s*(?:[AaPp]\.?[Mm]\.?)?\s*$")
# A clock that goes back this far between rows has passed midnight.
MIDNIGHT_DROP = pd.Timedelta(hours=12)
# Share of backward steps a clock column may have and still be read as one log in time order.
CLOCK_ORDER_TOLERANCE = 0.01


def _time_only(values: pd.Series) -> pd.Series:
    """True for bare times of day ("15:47:03", "3:05 PM") and spreadsheet time cells."""
    text = values.astype("string").str.match(_TIME_OF_DAY.pattern, na=False).astype(bool)
    cells = values.map(lambda value: isinstance(value, dt.time))
    return text | cells.astype(bool)


def _clock_times(values: pd.Series) -> tuple[pd.Series, int]:
    """Place bare times of day on ``UNDATED_DAY`` rather than on any real date.

    A log in time order that passes midnight moves on to the next day, so its
    order survives. Returns the timestamps and the number of midnights passed.
    """

    def of_day(distinct: pd.Series) -> pd.Series:
        text = distinct.astype("string")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", (UserWarning, FutureWarning))
            parsed = pd.to_datetime(text, errors="coerce", format="%H:%M:%S")
            others = parsed.isna() & text.notna()
            if others.any():
                parsed[others] = pd.to_datetime(text[others], errors="coerce", format="mixed")
        # The parser fills in a date of its own; only the time of day is kept.
        return parsed - parsed.dt.normalize()

    times = _per_value(values, of_day)
    steps = times.dropna().diff().dropna()
    midnights = steps < -MIDNIGHT_DROP
    backward = (steps < pd.Timedelta(0)) & ~midnights
    if not (midnights.any() and backward.mean() <= CLOCK_ORDER_TOLERANCE):
        return UNDATED_DAY + times, 0
    days = midnights.reindex(times.index, fill_value=False).astype(int).cumsum()
    return UNDATED_DAY + pd.to_timedelta(days, unit="D") + times, int(midnights.sum())


def _clock_note(column: str, clock: pd.Series, midnights: int) -> str:
    readings = clock.dropna()
    span = f" ({format_moment(readings.min())} to {format_moment(readings.max())})" if len(readings) else ""
    crossing = ""
    if midnights:
        crossing = f", passing midnight {midnights} time{'s' if midnights > 1 else ''}"
    return (
        f"{column} holds times of day with no calendar date{span}{crossing}. They are read "
        "as times of day, and no date is attached to them."
    )


# Words that name an elapsed-time column ("Time (s)", "elapsed_min", "t [ms]"), and
# the units it may be counted in, in seconds.
ELAPSED_WORDS = frozenset({"time", "elapsed", "t"})
ELAPSED_QUALIFIERS = frozenset({"since", "start", "from", "run"})
TIME_UNITS = {
    "ms": 0.001, "msec": 0.001, "millisecond": 0.001, "milliseconds": 0.001,
    "s": 1.0, "sec": 1.0, "secs": 1.0, "second": 1.0, "seconds": 1.0,
    "min": 60.0, "mins": 60.0, "minute": 60.0, "minutes": 60.0,
    "h": 3600.0, "hr": 3600.0, "hrs": 3600.0, "hour": 3600.0, "hours": 3600.0,
}


UNIT_NAMES = {0.001: "milliseconds", 1.0: "seconds", 60.0: "minutes", 3600.0: "hours"}


def _date_and_time(frame: pd.DataFrame, clock_columns: list[str]) -> tuple[str, str] | None:
    """Pair a column of bare times with the one column of plain dates beside it.

    Logs often split a timestamp into a Date and a Time column; together they
    are the real time of each row. Returns (date column, time column) or None.
    """
    if len(clock_columns) != 1:
        return None
    dated = [
        column
        for column in frame.columns
        if column not in clock_columns and is_datetime64_any_dtype(frame[column])
    ]
    if len(dated) != 1:
        return None
    days = frame[dated[0]].dropna()
    if days.empty or not bool((days == days.dt.normalize()).all()):
        return None
    return dated[0], clock_columns[0]


def _elapsed_unit(name: str) -> tuple[bool, float | None]:
    """Whether the name is an elapsed-time column, and the unit it states in seconds.

    Every word must be a time word, a unit or a qualifier such as "since start",
    so "Time to failure (h)" or "Max T" is not an elapsed time.
    """
    words = re.findall(r"[a-z]+", normalized_name(name))
    units = [TIME_UNITS[word] for word in words if word in TIME_UNITS]
    rest = [word for word in words if word not in TIME_UNITS]
    named = bool(rest) and set(rest) <= ELAPSED_WORDS | ELAPSED_QUALIFIERS and bool(set(rest) & ELAPSED_WORDS)
    if not named or len(units) > 1:
        return False, None
    return True, units[0] if units else None


def _elapsed_column(frame: pd.DataFrame, requested: str | None) -> tuple[str, float] | None:
    """Choose a numeric column of elapsed time to use as the time axis, with its unit.

    A column the caller asked for as the date qualifies on its name alone, and
    counts seconds when the name states no unit. Otherwise one is found only when the
    table has no date column: an elapsed-time name with a stated unit, and
    values that never go below zero and never fall from one row to the next.
    """
    if requested in frame.columns:
        series = frame[requested]
        named, unit = _elapsed_unit(requested)
        if named and is_numeric_dtype(series) and not is_bool_dtype(series):
            if series.dropna().ge(0).all():
                return requested, unit or 1.0
        return None
    if any(is_datetime64_any_dtype(frame[column]) for column in frame.columns):
        return None
    for column in frame.columns:
        series = frame[column]
        if is_bool_dtype(series) or not is_numeric_dtype(series):
            continue
        named, unit = _elapsed_unit(column)
        values = series.dropna()
        if (
            named
            and unit is not None
            and values.nunique() >= 3
            and bool(values.ge(0).all())
            and bool((values.diff().dropna() >= 0).all())
        ):
            return column, unit
    return None


def _parse_dates(values: pd.Series, *, dayfirst: bool) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", (UserWarning, FutureWarning))
        parsed = pd.to_datetime(values, errors="coerce", dayfirst=dayfirst)
        if parsed.dtype == object:
            # Mixed UTC offsets (e.g. across a daylight-saving change) parse to
            # object dtype. There is no single wall clock to keep, so use UTC.
            parsed = pd.to_datetime(values, errors="coerce", dayfirst=dayfirst, utc=True)
            parsed = parsed.dt.tz_localize(None)
        # The format is inferred from the first value, so ISO dates with and
        # without a time ("2019-04-02" after "2015-01-01 01:00:00") need a second pass.
        unread = values[parsed.isna() & values.notna()]
        retry = unread[unread.astype("string").str.match(_ISO_DATE, na=False).astype(bool)]
        if len(retry):
            iso = pd.to_datetime(retry, errors="coerce", format="ISO8601")
            if not isinstance(iso.dtype, pd.DatetimeTZDtype) and iso.dtype != object:
                parsed = parsed.where(~parsed.index.isin(retry.index), iso)
        # A bare time of day ("15:47:03") is given today's date; it is never a date.
        today = values[parsed.dt.normalize() == pd.Timestamp.today().normalize()]
        return parsed.mask(_time_only(today).reindex(parsed.index, fill_value=False))


def _dated(distinct: _Distinct, *, dayfirst: bool) -> pd.Series:
    return distinct.spread(lambda cells: _parse_dates(cells, dayfirst=dayfirst))


def _read_dates(distinct: _Distinct) -> tuple[pd.Series, str]:
    """Parse a date column, reporting the day/month order when it had to be decided.

    Returns the series and "day-first", "ambiguous" or "". The order that parses more
    values wins ("25/12/2024" only parses day-first). On a tie where the readings
    differ, month-first is used and "ambiguous" is returned so the caller can warn.
    """
    month_first = _dated(distinct, dayfirst=False)
    # Only day-or-month-first values (01/03/2024, not 2024-03-01) can be read two ways.
    two_ways = distinct.rows(_flags(distinct.values.astype("string").str.match(_DAY_OR_MONTH_FIRST)))
    if not bool(two_ways.any()):
        return month_first, ""

    day_first = _dated(distinct, dayfirst=True)
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


# The name a reader gives a column whose header cell is blank: "Unnamed: 3".
_BLANK_HEADER = re.compile(r"^Unnamed: \d+(?:_level_\d+)?$")
# Rows below the header looked at to judge whether the first row holds names.
HEADER_PROBE_ROWS = 200


def _names_in_first_row(frame: pd.DataFrame) -> bool:
    """Whether the first row holds names over numbers, as a second header row does.

    It must be text in at least two columns whose values below it read as
    numbers, never a number in such a column, and text in every column with a
    blank header cell over data.
    """
    first = frame.iloc[0]
    below = frame.iloc[1 : HEADER_PROBE_ROWS + 1]
    labels = 0
    for position, name in enumerate(frame.columns):
        cell = first.iloc[position]
        named = isinstance(cell, str) and bool(cell.strip())
        values = below.iloc[:, position].dropna()
        if values.empty:
            continue
        if _BLANK_HEADER.match(str(name)) and not named:
            return False
        if to_numbers(values).notna().mean() < 0.9:
            continue
        if named and pd.isna(to_numbers(pd.Series([cell])).iloc[0]):
            labels += 1
        elif not pd.isna(cell):
            return False
    return labels >= 2


def _two_row_header(frame: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    """Name the columns from both header rows when the first holds labels over groups of them.

    A spreadsheet often puts a label over a run of columns ("Weld Output
    Response" over "UTS (MPa)" and "% Elongation") with the names in the row
    below. Read with one header row, the label's blank cells become "Unnamed: 6"
    and the names become a row of text in numeric columns, which then counts as
    data. Each column under a group label takes its name from the second row; a
    label over a single column is joined with the text below it ("Temperature"
    over "(°C)"). Returns the frame without that row, and a note, or the frame
    unchanged and None.
    """
    names = [str(name) for name in frame.columns]
    blank = [bool(_BLANK_HEADER.match(name)) for name in names]
    if len(frame) < 3 or not any(blank) or all(blank) or not _names_in_first_row(frame):
        return frame, None
    subs = [cell.strip() if isinstance(cell, str) and cell.strip() else None for cell in frame.iloc[0]]
    groups: list[str | None] = []
    for name, is_blank in zip(names, blank, strict=True):
        groups.append((groups[-1] if groups else None) if is_blank else name.strip())
    spanning = {group for group, is_blank in zip(groups, blank, strict=True) if is_blank and group}
    built = []
    for name, is_blank, group, sub in zip(names, blank, groups, subs, strict=True):
        if not is_blank and group not in spanning:
            built.append(f"{group} {sub}" if sub else str(group))
        else:
            built.append(sub or group or name)
    repeated = {name for name in built if built.count(name) > 1}
    built = [
        f"{group} {name}" if name in repeated and group and group != name else name
        for name, group in zip(built, groups, strict=True)
    ]
    promoted = frame.iloc[1:].reset_index(drop=True).set_axis(built, axis=1).infer_objects()
    labels = [f'"{group}"' for group in dict.fromkeys(groups) if group in spanning]
    if labels:
        listed = labels[0] if len(labels) == 1 else f"{', '.join(labels[:-1])} and {labels[-1]}"
        over = "labels a group" if len(labels) == 1 else "label groups"
        note = (
            f"The header has two rows: {listed} {over} of columns named in the row below, so the "
            "columns take their names from that row, which is not read as data."
        )
    else:
        note = "The header has two rows, so the columns are named from both, and the second is not data."
    return promoted, note


def _looks_like_exported_index(series: pd.Series, name: str) -> bool:
    """True for a row-number column left behind by to_csv(index=True).

    Only "Unnamed..." columns qualify, so a named counter is never dropped. Gaps are
    allowed because blank rows are removed before this runs; the values must still
    start at 0 or 1, ascend and stay dense.
    """
    if not name.lower().startswith("unnamed"):
        return False

    numeric = to_numbers(series)
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


# Years a column may hold to become dates. A column the caller names as the date
# may hold any four-digit year; one found on its own needs a year-like name and a
# plausible modern year, so a count that happens to look like a year is left alone.
REQUESTED_YEARS = (1000, 2999)
DETECTED_YEARS = (1800, 2100)

_MONTHS = (
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
)
MONTH_NUMBERS = {
    name: number for number, month in enumerate(_MONTHS, start=1) for name in (month, month[:3])
} | {"sept": 9}


def _whole_years(series: pd.Series, bounds: tuple[int, int]) -> bool:
    """True for a numeric column whose values are all whole numbers within ``bounds``."""
    if is_bool_dtype(series) or not is_numeric_dtype(series):
        return False
    values = series.dropna()
    return (
        not values.empty
        and bool((values % 1 == 0).all())
        and bool(values.between(*bounds).all())
    )


# Whole numbers that write a date: YYYYMMDD in eight digits, YYYYMM in six.
DIGIT_DATE_LAYOUTS = {8: ("%Y%m%d", "YYYYMMDD"), 6: ("%Y%m", "YYYYMM")}
# Words that name a date column, in the languages files most often arrive in.
DATE_NAME_WORDS = frozenset({
    "date", "datum", "fecha", "data", "dt", "day", "period", "yyyymmdd", "yyyymm", "ymd",
})
# Years a timestamp can hold, for a column the caller names as the date.
TIMESTAMP_YEARS = (1678, 2261)


def _digit_dates(series: pd.Series, bounds: tuple[int, int]) -> tuple[pd.Series, str] | None:
    """Read whole numbers written as YYYYMMDD or YYYYMM as dates, with the layout read.

    Every value must be a valid date whose year lies within ``bounds``; otherwise
    returns None, so a count or a code that only looks like a date stays a number.
    """
    if is_bool_dtype(series) or not is_numeric_dtype(series):
        return None
    values = series.dropna()
    if values.empty:
        return None
    low, high = float(values.min()), float(values.max())
    for digits, (layout, label) in DIGIT_DATE_LAYOUTS.items():
        if not 10 ** (digits - 1) <= low <= high < 10**digits:
            continue
        if not bool((values % 1 == 0).all()):
            return None
        distinct = pd.unique(values.astype("int64"))
        dates = pd.to_datetime(pd.Series(distinct).astype(str), format=layout, errors="coerce")
        if dates.isna().any() or not bool(dates.dt.year.between(*bounds).all()):
            return None
        lookup = pd.Series(dates.to_numpy(), index=distinct)
        read = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
        read[values.index] = lookup.reindex(values.astype("int64").to_numpy()).to_numpy()
        return read, label
    return None


def _digit_date_columns(
    frame: pd.DataFrame, requested: str | None, dated: bool
) -> list[tuple[str, pd.Series, str]]:
    """The whole-number columns to read as YYYYMMDD or YYYYMM dates, read.

    A column the caller names as the date qualifies on its values alone. One
    found on its own needs plausible modern years and either a date-like name
    and two or more values, or three or more values and no other date in the
    table (``dated`` is False).
    """
    found = []
    for column in frame.columns:
        series = frame[column]
        if is_bool_dtype(series) or not is_numeric_dtype(series):
            continue
        if column == requested:
            read = _digit_dates(series, TIMESTAMP_YEARS)
        else:
            named = bool(DATE_NAME_WORDS & set(normalized_name(column).split()))
            read = _digit_dates(series, DETECTED_YEARS) if named or not dated else None
            if read is not None and read[0].nunique() < (2 if named else 3):
                read = None
        if read is not None:
            found.append((column, *read))
    return found


def _month_numbers(series: pd.Series) -> pd.Series:
    """Months written as names ("March", "Mar") or as numbers 1-12; anything else is NaN."""
    text = series.astype("string").str.strip().str.lower().str.rstrip(".")
    named = text.map(MONTH_NUMBERS, na_action="ignore").astype("float64")
    numbered = to_numbers(text).astype("float64")
    return named.fillna(numbered.where(numbered.isin(range(1, 13))))


# The whole numbers a column of each part finer than a month may hold. An hour
# column may hold 24, which some loggers write for the end of a day; those rows
# get no timestamp rather than one moved into the next day.
FINER_PART_RANGES = {"day": (1, 31), "hour": (0, 24), "minute": (0, 59), "second": (0, 59)}


def _fills_part(values: pd.Series, bounds: tuple[int, int]) -> bool:
    """True when every value is present and a whole number within ``bounds``."""
    if values.empty or is_bool_dtype(values) or not is_numeric_dtype(values) or values.isna().any():
        return False
    return bool((values % 1 == 0).all()) and bool(values.between(*bounds).all())


def _year_columns(
    frame: pd.DataFrame, requested: str | None
) -> tuple[str, str | None, dict[str, str]] | None:
    """Choose the year column, and the month and finer columns to go with it, to build dates from.

    A column the caller asked for as the date is used alone. Otherwise one is
    found only when the table has no date column: a year-named column of whole
    years in DETECTED_YEARS with at least two distinct values. A month-named
    column readable for every row with a year is paired with it, so monthly
    rows get monthly dates. Then a day, an hour, a minute and a second, by
    name and each only after the one before it, join them when they hold a
    whole number in range on every row with a year, so hourly rows get hourly
    timestamps. The finer columns are returned by part, coarsest first.
    """
    if requested in frame.columns and _whole_years(frame[requested], REQUESTED_YEARS):
        return requested, None, {}
    if any(is_datetime64_any_dtype(frame[column]) for column in frame.columns):
        return None
    candidates = [
        column
        for column in frame.columns
        if is_year_name(column)
        and _whole_years(frame[column], DETECTED_YEARS)
        and frame[column].nunique() >= 2
    ]
    if not candidates:
        return None
    # A bare "Year" before "Model Year"; then the name, so column order never decides.
    year = min(candidates, key=lambda column: (len(normalized_name(column).split()), column))
    dated = frame[year].notna()
    month = next(
        (
            column
            for column in frame.columns
            if column != year
            and is_month_name(column)
            and _month_numbers(frame[column])[dated].notna().all()
        ),
        None,
    )
    finer: dict[str, str] = {}
    if month is None:
        return year, month, finer
    for part, bounds in FINER_PART_RANGES.items():
        column = next(
            (
                column
                for column in frame.columns
                if finer_part_name(column) == part and _fills_part(frame.loc[dated, column], bounds)
            ),
            None,
        )
        if column is None:
            break
        finer[part] = column
    return year, month, finer


def _built_dates(
    frame: pd.DataFrame, year: str, month: str | None = None, finer: dict[str, str] | None = None
) -> pd.Series:
    """Dates built from a year column and, when given, a month column and finer parts.

    A year alone gives 1 January of each year and a month the first of each
    month; finer parts add the day, hour, minute and second they hold. A
    combination no calendar has (31 February, or hour 24) becomes missing, as do
    years outside what a timestamp can hold (about 1678 to 2261).
    """
    parts = pd.DataFrame(
        {"year": frame[year], "month": _month_numbers(frame[month]) if month else 1, "day": 1},
        index=frame.index,
    )
    for part, column in (finer or {}).items():
        parts[part] = frame[column]
    dates = pd.to_datetime(parts, errors="coerce")
    # pandas would move hour 24 into the next day.
    return dates.mask(parts["hour"] == 24) if "hour" in parts else dates


def _add_timestamp(frame: pd.DataFrame, dates: pd.Series, parts: list[str], unreadable: int) -> list[str]:
    """Put ``dates``, built from the ``parts`` columns down to a day or finer, beside them; return notes.

    The new column is named after the parts ("year-month-day-hour") and placed
    where the year was. The parts stay: they are what the file holds, and a day
    or an hour is a grouping in its own right (the daily cycle).
    """
    name = "-".join(parts)
    while name in frame.columns:
        name += " (built)"
    frame.insert(frame.columns.get_loc(parts[0]), name, dates)
    listed = f"{', '.join(parts[:-1])} and {parts[-1]}"
    notes = [
        f"{listed} were combined into one timestamp column, {name}, to the {finer_part_name(parts[-1])}, "
        "with no time zone attached; the original columns are kept."
    ]
    if unreadable:
        first = frame.loc[frame[parts[0]].notna() & dates.isna(), parts].iloc[0]
        written = ", ".join(f"{column} {_as_written(first[column])}" for column in parts)
        notes.append(
            f"{unreadable:,} rows have no {name}, because their {listed} make no valid date and time "
            f"(such as 31 February or hour 24); the first is {written}."
        )
    return notes


def _as_written(value) -> str:
    """A whole number without its decimal point (2021, not 2021.0); anything else as it is."""
    if isinstance(value, float | np.floating) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _holds_decimals(distinct: _Distinct) -> bool:
    """True when the text reads as numbers, nearly all of them, and some have a fractional part.

    Missing-value markers ("NA", "<LOD") are left out of the count.
    """
    markers = _missing_markers(distinct)
    numbers = to_numbers(distinct.values.mask(markers))
    read = numbers.notna().to_numpy()
    counts = distinct.counts
    fractional = bool((numbers[read] % 1 != 0).any())
    return int(counts[read].sum()) >= 0.95 * max(int(counts[~markers].sum()), 1) and fractional


# Text columns with more distinct values than this are free text, not labels to merge.
MAX_LABELS_TO_MERGE = 20_000


def _label_key(value: str) -> str:
    """A label compared without case, accents or spacing: "No indica" and "NO INDICA" match."""
    decomposed = unicodedata.normalize("NFKD", value)
    plain = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(plain.casefold().split())


def _merge_label_variants(distinct: _Distinct, column: str) -> tuple[dict, str | None]:
    """Write labels that differ only in case, accents or spacing one way, the most common one.

    "NO INDICA", "No indica" and "no Indica" are one answer; left apart, they
    split one group into three in every breakdown. Returns the mapping from
    each variant to the spelling kept and a note, or an empty mapping and None.
    """
    # Most common first, and in order of first appearance among equals.
    counts = pd.Series(distinct.counts, index=distinct.values.to_numpy()).sort_values(
        ascending=False, kind="stable"
    )
    labels = [label for label, count in counts.items() if isinstance(label, str) and count]
    if not 2 <= len(labels) <= MAX_LABELS_TO_MERGE:
        return {}, None
    spellings: dict[str, list[str]] = {}
    for label in labels:
        # Most common first, so each group's first spelling is the one kept.
        spellings.setdefault(_label_key(label), []).append(label)
    merged = [group for group in spellings.values() if len(group) > 1]
    if not merged:
        return {}, None
    mapping = {variant: group[0] for group in merged for variant in group[1:]}
    changed = int(counts[list(mapping)].sum())
    largest = max(merged, key=lambda group: int(counts[group].sum()))
    shown = " and ".join(f'"{variant}"' for variant in largest[1:3])
    others = ""
    if len(merged) > 1:
        others = f", and {len(merged) - 1:,} other label{'s' if len(merged) > 2 else ''} likewise"
    note = (
        f"{changed:,} {column} values spelled differently only in case, accents or spacing were "
        f'merged: {shown} became "{largest[0]}"{others}.'
    )
    return mapping, note


# Number bodies, once signs, currency, brackets and spaces are set aside, that
# show how a column writes its decimals. A comma before other than three digits,
# or after a lone 0, is a decimal point, as is one after dot-grouped thousands;
# a comma before a dot, or several commas, groups thousands.
_COMMA_DECIMAL = re.compile(r"^(?:\d+,(?:\d{1,2}|\d{4,})|0,\d+|\d{1,3}(?:\.\d{3})+,\d+)$")
_COMMA_GROUPING = re.compile(r"^\d{1,3}(?:,\d{3})+\.\d+$|^\d{1,3}(?:,\d{3}){2,}$")
# Thousands grouped with dots and no decimals: "1.234" in a column of "999,5".
_DOT_THOUSANDS = re.compile(r"^[-+]?\d{1,3}(?:\.\d{3})+$")
# A comma read differently once the column's decimal mark is known: one before
# three digits or more ("-0,307", "306,27883").
_LONG_COMMA_FRACTION = re.compile(r"^[-+]?\d+,\d{3,}$")
# A dot that can only be a decimal point: one before other than three digits ("5.66").
_DOT_DECIMAL = re.compile(r"^\d+\.(?:\d{1,2}|\d{4,})$")


def _decimal_comma(text: pd.Series) -> bool:
    """Whether a column writes its decimal point as a comma, judged from all its values.

    One value is often ambiguous on its own: "0,307" or "-2,95" can only be
    decimals, but "1,234" may be a thousand. When some values show commas used
    as decimal points and none shows them grouping thousands ("1,234.5"), every
    comma in the column is a decimal point, so one column is never read on two
    scales.
    """
    bodies = text.dropna().astype(str).str.replace(r"[\s'$€£¥₹()+\-<>≤≥]", "", regex=True)
    commas = bodies[bodies.str.contains(",", regex=False)]
    if commas.empty:
        return False
    decimal_commas = int(commas.str.match(_COMMA_DECIMAL.pattern).sum())
    if not decimal_commas or commas.str.match(_COMMA_GROUPING.pattern).any():
        return False
    # A few comma decimals among many dot decimals ("< 3,80" in a column of "5.66")
    # are the exceptions: the column's decimal mark is the dot, and "6.207" is 6.207.
    return decimal_commas >= int(bodies.str.match(_DOT_DECIMAL.pattern).sum())


# Labels misspelt or abbreviated are merged only in columns of a few category
# labels, and only when the rarer spelling is at most this share of the other.
MAX_LABELS_TO_RESPELL = 50
RARE_SPELLING = 0.2
MIN_RESPELT_LETTERS = 5


def _one_letter_apart(first: str, second: str) -> bool:
    """Whether two words differ by a letter added or dropped, or two neighbours swapped: "Hipanic"."""
    if abs(len(first) - len(second)) == 1:
        shorter, longer = sorted((first, second), key=len)
        return any(longer[:index] + longer[index + 1:] == shorter for index in range(len(longer)))
    if len(first) == len(second):
        apart = [index for index in range(len(first)) if first[index] != second[index]]
        return (
            len(apart) == 2 and apart[1] == apart[0] + 1
            and first[apart[0]] == second[apart[1]] and first[apart[1]] == second[apart[0]]
        )
    return False


def _respelt_labels(distinct: _Distinct, column: str) -> tuple[dict, str | None]:
    """Merge labels that are one label written another way: "M" with "male", "Hipanic" with "Hispanic".

    Two cases, kept conservative. A single letter is the first letter of
    exactly one word in the column ("M" and "male", but not "A" beside "AB",
    which is a code too). And two words of five letters or more one letter
    apart (see ``_one_letter_apart``) where the rarer is at most RARE_SPELLING
    of the other, as a typo is; "Type A" and "Type B" differ in a letter that
    is not a typo, so letters other than a-z never count. The single letter
    gives way to the word, and the rarer spelling to the commoner. Returns
    the mapping from each spelling dropped to the one kept and a note.
    """
    counts = pd.Series(distinct.counts, index=distinct.values.to_numpy())
    counts = counts[[isinstance(label, str) and count > 0 for label, count in counts.items()]]
    if not 2 <= len(counts) <= MAX_LABELS_TO_RESPELL:
        return {}, None
    keys = {label: _label_key(label) for label in counts.index}
    mapping: dict[str, str] = {}
    words = [label for label, key in keys.items() if key.isalpha() and len(key) >= 2 and not label.isupper()]
    for letter in [label for label, key in keys.items() if len(key) == 1 and key.isalpha()]:
        matches = [word for word in words if keys[word][0] == keys[letter]]
        if len(matches) == 1:
            mapping[letter] = matches[0]
    spelt = [
        label for label, key in keys.items()
        if re.fullmatch(r"[a-z ]+", key) and len(key) >= MIN_RESPELT_LETTERS
    ]
    for rare in spelt:
        for common in spelt:
            if (
                rare != common and rare not in mapping and common not in mapping
                and counts[rare] <= RARE_SPELLING * counts[common]
                and _one_letter_apart(keys[rare], keys[common])
            ):
                mapping[rare] = common
                break
    if not mapping:
        return {}, None
    changed = int(counts[list(mapping)].sum())
    shown = ", ".join(f'"{old}" into "{new}"' for old, new in list(mapping.items())[:3])
    more = f", and {len(mapping) - 3:,} more" if len(mapping) > 3 else ""
    return mapping, (
        f"{changed:,} {column} values look like another label written a different way, so they were "
        f"merged: {shown}{more}."
    )


def _read_numbers(distinct: _Distinct) -> tuple[pd.Series, np.ndarray, bool]:
    """Parse the distinct values as numbers.

    Returns the numbers, worked out on ``distinct.cells`` for ``expand``, which
    distinct values were missing-value markers, and whether the column writes
    decimals with commas (see ``_decimal_comma``).
    """
    markers = _missing_markers(distinct)
    cells = distinct.cells
    marked = distinct.on_cells(markers)
    numbers = to_numbers(cells.mask(marked))
    comma = _decimal_comma(distinct.values.mask(markers))
    # Retry cells plain parsing missed as formatted numbers ("$1,203.55",
    # "(48.10)"). Large values are the ones with separators, so skipping
    # them biases totals down. Values already read ("1e3") are kept, except
    # dot-grouped thousands in a column whose decimals are commas.
    gaps = numbers.isna().to_numpy() & cells.notna().to_numpy() & ~marked
    if comma:
        gaps |= _flags(cells.astype("string").str.strip().str.match(_DOT_THOUSANDS.pattern)) & ~marked
    if gaps.any():
        formatted = cells.where(gaps).map(
            lambda value: _read_formatted_number(value, comma) if isinstance(value, str) else None
        )
        numbers = numbers.astype("float64").mask(gaps, formatted.astype("float64"))
    return numbers, markers, comma


def _numeric_share(values: pd.Series) -> float:
    """Share of the non-marker cells that read as numbers."""
    distinct = _Distinct.of(values)
    numbers, markers, _ = _read_numbers(distinct)
    counts = distinct.counts
    readable = int(counts[~markers].sum())
    read = numbers.notna().to_numpy()[: len(distinct.values)]
    return float(counts[read].sum() / readable) if readable else 0.0


def _missing_marker_note(column: str, examples: list[str], count: int) -> str:
    shown = ", ".join(f'"{value}"' for value in examples[:4]) + (", ..." if len(examples) > 4 else "")
    noun = "cells" if count > 1 else "cell"
    return f"{count:,} {noun} in {column} read as missing: {shown}."


def _marker_notes(
    column: str,
    distinct: _Distinct,
    markers: np.ndarray,
    token_cells: pd.Series,
    token: str | None,
    total: int,
) -> list[str]:
    """Notes for the markers read in a column that became numeric.

    Values past a detection limit get their own note, since leaving them out
    biases the mean; other markers and a lone token are listed as read.
    ``markers`` marks the distinct values read as markers. "ND" counts as below
    a limit, unless the column writes values below its limit as "<0.4": an
    "ND" beside those more likely means the value was not determined.
    """
    notes = []
    below, above = _censored(distinct)
    undetected = _flags(distinct.text.isin(NOT_DETECTED_MARKERS)) & markers
    limited = _flags(distinct.text.str.match(_CENSORED_VALUE.pattern)) & markers
    undetermined = undetected if limited.any() else np.zeros(len(undetected), dtype=bool)
    below = below & markers & ~undetermined
    above = above & markers
    plain = markers & ~below & ~above & ~undetermined
    if plain.any():
        notes.append(_missing_marker_note(column, *_marked(distinct, plain)))
    if undetermined.any():
        shown, count = _marked(distinct, undetermined)
        example = str(distinct.values[limited].iloc[0]).strip()
        notes.append(
            f'{_missing_marker_note(column, shown, count)} The column writes values past its limit as '
            f'"{example}", so "{shown[0]}" more likely means the value was not determined.'
        )
    for side, censored in (("below", below), ("above", above)):
        if censored.any():
            notes.append(_censored_note(column, distinct, censored, side, total))
    if token_cells.any():
        count = int(token_cells.sum())
        first = distinct.codes[np.flatnonzero(token_cells.to_numpy())[0]]
        shown = str(distinct.values.iloc[first]).strip()
        noun, verb = ("cells", "were") if count > 1 else ("cell", "was")
        if token in TRACE_TOKENS - {"t"}:
            notes.append(
                f'{count:,} {noun} in {column} held "{shown}" and {verb} read as 0: a trace is below the '
                "smallest amount measured."
            )
        else:
            note = (
                f'{count:,} {noun} in {column} held "{shown}", the only text in the column, and {verb} '
                "read as missing"
            )
            if token in TRACE_TOKENS:
                note += (
                    f'; if "{shown}" marks a trace, the true values are near zero and the mean of the '
                    "rest runs high"
                )
            notes.append(note + ".")
    return notes


# Yes/no words in a column otherwise holding 0 and 1, and the number each stands for.
YES_NO_WORDS = {"true": 1.0, "false": 0.0, "yes": 1.0, "no": 0.0, "y": 1.0, "n": 0.0, "t": 1.0, "f": 0.0}


def _yes_no_in_flags(
    column: str, distinct: _Distinct, numbers: pd.Series, markers: np.ndarray
) -> tuple[pd.Series, str | None]:
    """Read "true", "false", "yes" or "no" as 1 or 0 in a column whose numbers are only 0 and 1.

    Left unread they would become missing, though they say the same as the
    1s and 0s around them. ``numbers`` is worked out on ``distinct.cells``.
    Returns the numbers with the words read, and a note, or them unchanged and None.
    """
    count = len(distinct.values)
    read = numbers.iloc[:count]
    unread = read.isna().to_numpy() & ~markers & (distinct.counts > 0)
    if not unread.any():
        return numbers, None
    words = distinct.text[unread]
    known = read[~read.isna().to_numpy()].unique()
    if not len(known) or not set(known) <= {0, 1} or not words.isin(list(YES_NO_WORDS)).all():
        return numbers, None
    filled = numbers.copy()
    positions = np.flatnonzero(unread)
    filled.iloc[positions] = words.map(YES_NO_WORDS).to_numpy(dtype=float)
    shown, cells = _marked(distinct, unread)
    listed = ", ".join(f'"{word}"' for word in shown[:4])
    noun = "cells" if cells > 1 else "cell"
    return filled, (
        f"{cells:,} {noun} in {column} held {listed}, read as 1 for yes and 0 for no, since the column "
        "otherwise holds only 0 and 1."
    )


# "5.66 ± 1.69 items/10 g", "0.43 items/individual", "12 mg/L": a reading written
# with its spread, its unit or both. The unit starts with a letter, so "4.5
# *10^(6) items" is not read as 4.5, and may hold digits ("items/10 g") but no
# second number range.
_READING_WITH_UNIT = re.compile(
    r"^(?P<number>[-+]?\d+(?:[.,]\d+)?)\s*(?P<spread>(?:±|\+/-|\+-)\s*\d+(?:[.,]\d+)?)?\s*"
    r"(?P<unit>[A-Za-zµμ°%][^±–—]*)?$"
)
# Share of a column's readable cells that must read this way, and of those the
# share that must carry a unit or a spread, so plain numbers keep their own path.
READINGS_SHARE = 0.5
LABELLED_SHARE = 0.5
# Most units a column of readings is written in. More are codes that start with a number
# ("37F-PM-1014-03", a squirrel's ID), not readings in units.
MAX_READING_UNITS = 12


def _unit_text(unit: str) -> str:
    """A unit written one way: "items/ individual" and "items/individual" are one unit."""
    return re.sub(r"\s*/\s*", "/", " ".join(unit.split()))


def _readings_with_units(
    distinct: _Distinct, probe: pd.Series
) -> tuple[pd.Series, pd.Series, np.ndarray] | None:
    """Numbers and units of a text column of readings written with a spread or a unit, or None.

    Returns, on ``distinct.cells``, the number each cell starts with (the mean
    before a "±"), its unit, and which distinct values were missing-value
    markers ("ND", "<LOQ"). ``probe`` holds cells drawn from the column, which
    must show such readings before the whole column is parsed.
    """
    probe = probe.astype(str).str.strip().str.extract(_READING_WITH_UNIT.pattern)
    shown = probe["number"].notna() & (probe["unit"].notna() | probe["spread"].notna())
    if not shown.mean() >= PROBE_RATIO * READINGS_SHARE * LABELLED_SHARE:
        return None
    markers = _missing_markers(distinct)
    text = distinct.values.astype("string").str.strip()
    parts = text.str.extract(_READING_WITH_UNIT.pattern)
    matched = parts["number"].notna().to_numpy() & ~markers
    counts = distinct.counts
    readable = int(counts[~markers].sum())
    labelled = matched & (parts["unit"].notna() | parts["spread"].notna()).to_numpy()
    if not readable or counts[matched].sum() < READINGS_SHARE * readable:
        return None
    # Ordinals ("1st", "2nd", "3rd") are labels, not readings in a unit called "st"; "Crew" beside
    # them is another class, not a reading that failed to parse.
    if labelled.any() and text.str.fullmatch(r"\d+(?:st|nd|rd|th)", case=False).to_numpy()[labelled].all():
        return None
    if counts[labelled].sum() < LABELLED_SHARE * counts[matched].sum():
        return None
    numbers = parts["number"].where(matched).map(
        lambda value: _read_formatted_number(value) if isinstance(value, str) else None
    ).astype("float64")
    units = parts["unit"].where(matched).map(lambda unit: _unit_text(unit) if isinstance(unit, str) else None)
    if units.dropna().nunique() > MAX_READING_UNITS:
        return None
    cells = len(distinct.cells)
    if cells > len(distinct.values):
        numbers = pd.concat([numbers, pd.Series([np.nan])], ignore_index=True)
        units = pd.concat([units, pd.Series([None], dtype=object)], ignore_index=True)
    return numbers.reset_index(drop=True), units.reset_index(drop=True).astype(object), markers


def _readings_note(column: str, example: str, value: float, units: list[str], unit_column: str | None) -> str:
    note = (
        f'{column} holds readings written with a spread or a unit ("{example}"), so each was read as '
        f"the number it starts with ({value:g})"
    )
    if unit_column:
        shown = ", ".join(units[:3]) + (f" and {len(units) - 3:,} more" if len(units) > 3 else "")
        return note + (
            f". They are in {len(units):,} different units ({shown}), recorded in a new column, "
            f"{unit_column}, so readings in different units are never combined."
        )
    if units:
        return note + f". Every one is in {units[0]}."
    return note + "."


def _decimal_comma_note(columns: list[tuple[str, str, float]]) -> str:
    """Say which columns were read with commas as decimal points, with one value as written and as read."""
    names = [column for column, _, _ in columns]
    shown = ", ".join(names[:3]) + (f" and {len(names) - 3:,} more" if len(names) > 3 else "")
    _, written, read = columns[0]
    return (
        f'Commas mark the decimal point in {shown}, as other values there show, so every comma '
        f'in {"it" if len(names) == 1 else "them"} was read that way: "{written}" is {read:.10g}.'
    )


# Prefixes that start the names of separate yes/no questions, not the levels of one category.
QUESTION_PREFIXES = frozenset({"is", "has", "had", "was", "flag", "any", "ever", "did"})
# Where all of a one-hot set's columns are 0, the row belongs to the level left out.
ONE_HOT_REST = "other"


def _one_hot_sets(frame: pd.DataFrame) -> list[tuple[str, list[str]]]:
    """Sets of 0/1 columns sharing a name prefix that never mark two levels on one row.

    "Soil_Type_Gravel", "Soil_Type_Sand" and "Soil_Type_Silt" are soil type
    split into one column per level, the way models are fed; a row where all
    are 0 holds the level left out. Two columns qualify only when every row has
    exactly one of them, and prefixes that start yes/no questions ("is_",
    "has_") never do.
    """
    numeric = [
        column for column, kind in frame.dtypes.items() if is_numeric_dtype(kind) and not is_bool_dtype(kind)
    ]
    if len(numeric) < 2:
        return []
    # Columns running from 0 to 1, then only those holding nothing else and no gaps.
    lowest, highest = frame[numeric].min(), frame[numeric].max()
    spanning = [column for column in numeric if lowest[column] == 0 and highest[column] == 1]
    flags = [
        column for column in spanning
        if bool(((frame[column] == 0) | (frame[column] == 1)).all())
    ]
    prefixes: dict[str, list[str]] = {}
    for column in flags:
        name = str(column)
        for match in re.finditer(r"[_.:\- ]+", name):
            prefix, rest = name[: match.start()], name[match.end():]
            if len(prefix) >= 3 and rest and normalized_name(prefix).split()[0] not in QUESTION_PREFIXES:
                prefixes.setdefault(prefix, []).append(column)
    found: list[tuple[str, list[str]]] = []
    used: set[str] = set()
    # The longest prefix shared by a set names it ("Soil_Type", not "Soil").
    for prefix in sorted(prefixes, key=len, reverse=True):
        members = [column for column in prefixes[prefix] if column not in used]
        if len(members) < 2 or prefix in frame.columns:
            continue
        sums = frame[members].sum(axis=1)
        if sums.max() > 1 or (len(members) == 2 and sums.min() < 1):
            continue
        found.append((prefix, members))
        used.update(members)
    return found


def _join_one_hot(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Put each one-hot set (see ``_one_hot_sets``) back into one category column, with a note."""
    notes = []
    for prefix, members in _one_hot_sets(frame):
        levels = [str(column)[len(prefix):].lstrip("_.:- ") for column in members]
        chosen = frame[members].to_numpy().argmax(axis=1)
        rest = frame[members].sum(axis=1).to_numpy() == 0
        named = np.array(levels, dtype=object)[chosen]
        joined = pd.Series(np.where(rest, ONE_HOT_REST, named), index=frame.index)
        position = frame.columns.get_loc(members[0])
        frame = frame.drop(columns=members)
        frame.insert(position, prefix, joined)
        note = (
            f"{', '.join(map(str, members[:-1]))} and {members[-1]} each mark one level of a category "
            f"and never two on one row, so they were joined into one column, {prefix}"
        )
        if rest.any():
            note += (
                f"; the {int(rest.sum()):,} rows where all are 0 hold the level left out, shown as "
                f'"{ONE_HOT_REST}"'
            )
        notes.append(note + ".")
    return frame, notes


def _unread_note(column: str, unread: pd.Series) -> str:
    """Name the cells a numeric column could not read, which became missing."""
    examples = list(dict.fromkeys(unread.astype(str).str.strip()))
    shown = ", ".join(f'"{value}"' for value in examples[:4]) + (", ..." if len(examples) > 4 else "")
    if len(unread) == 1:
        return f"1 cell in {column} could not be read as a number and was treated as missing: {shown}."
    return (
        f"{len(unread):,} cells in {column} could not be read as numbers and were treated as "
        f"missing: {shown}."
    )


def clean_dataframe(
    dataframe: pd.DataFrame, *, drop_duplicates: bool = False, date: str | None = None
) -> tuple[pd.DataFrame, CleaningReport]:
    """Clean the frame conservatively and return it with a report of what changed.

    Identical rows are counted but kept, because repeated records are common in
    real data. Pass ``drop_duplicates=True`` to remove them. ``date`` names the
    column the caller wants as the date; if it holds whole years, they become
    1 January of each year, and whole numbers written as YYYYMMDD or YYYYMM
    become those dates. Raises ValueError if the input, or what remains after
    cleaning, has no rows or columns.
    """
    if dataframe.empty or len(dataframe.columns) == 0:
        raise ValueError("The CSV does not contain any rows and columns to analyze.")

    frame, header_note = _two_row_header(dataframe)
    cleaned = frame.copy()
    original_rows, original_columns = cleaned.shape
    cleaned.columns = _make_unique_columns(cleaned.columns)
    notes: list[str] = [header_note] if header_note else []
    notes.extend(_duplicate_header_notes(frame.columns, cleaned))

    # Each text column is factorised once here, and every later check on it
    # works from its distinct values.
    distincts = {column: _Distinct.of(cleaned[column]) for column in _text_columns(cleaned)}
    blanks = {
        column: _blank_text(distincts[column]) if column in distincts else cleaned[column].isna().to_numpy()
        for column in cleaned.columns
    }
    empty_columns = [column for column, blank in blanks.items() if blank.all()]
    # Each drop copies the whole frame, so only drop what is there.
    if empty_columns:
        cleaned = cleaned.drop(columns=empty_columns)
        notes.append(_dropped_columns_note(empty_columns))

    kept = [blank for column, blank in blanks.items() if column not in empty_columns]
    empty_row_mask = np.logical_and.reduce(kept) if kept else np.ones(len(cleaned), dtype=bool)
    empty_rows_removed = int(empty_row_mask.sum())
    if empty_rows_removed:
        cleaned = cleaned.loc[~empty_row_mask].copy()
        distincts = {column: distinct.take(~empty_row_mask) for column, distinct in distincts.items()}

    index_columns = [
        column for column in cleaned.columns if _looks_like_exported_index(cleaned[column], column)
    ]
    if index_columns:
        cleaned = cleaned.drop(columns=index_columns)

    trimmed_text_columns = 0
    numeric_columns_inferred = 0
    datetime_columns_inferred = _normalize_datetime_columns(cleaned)
    unparsed_date_cells = 0

    if datetime_columns_inferred:
        notes.append(
            "Timezone offsets were dropped; dates are read as the local time they were written in."
        )

    date_tokens = ("date", "time", "timestamp", "created", "updated")
    named_date_ratio, numeric_ratio, unnamed_date_ratio = 0.8, 0.95, 0.95
    date_sample_size = 50

    clock_columns: list[str] = []
    upper_limits: dict[str, pd.Series] = {}
    # Columns read with commas as decimal points, each with a value as written and as read.
    comma_columns: list[tuple[str, str, float]] = []
    # The text columns that stay text, for the checks after inference.
    still_text: dict[str, _Distinct] = {}
    for column in _text_columns(cleaned):
        distinct = distincts.get(column) or _Distinct.of(cleaned[column])
        non_null_before = distinct.present
        values = distinct.values
        padded = values.map(lambda value: isinstance(value, str) and value != value.strip())
        if padded.any():
            trimmed_text_columns += 1
        if padded.any() or values.eq("").any():
            distinct = distinct.relabel(
                values.map(lambda value: (value.strip() or pd.NA) if isinstance(value, str) else value)
            )
            cleaned[column] = distinct.column()
        still_text[column] = distinct

        if non_null_before == 0:
            continue

        named_like_a_date = any(token in column.lower() for token in date_tokens)
        clock_ratio = named_date_ratio if named_like_a_date else unnamed_date_ratio
        probe = distinct.sample()
        clock = distinct.spread(_time_only) if _time_only(probe).mean() >= PROBE_RATIO else None
        if clock is not None and clock.sum() / non_null_before >= clock_ratio:
            times, midnights = _clock_times(cleaned[column].where(clock))
            unreadable = int(non_null_before - times.notna().sum())
            if unreadable:
                unparsed_date_cells += unreadable
                notes.append(f"{unreadable} {column} values could not be read as times.")
            notes.append(_clock_note(column, times, midnights))
            cleaned[column] = times
            clock_columns.append(column)
            datetime_columns_inferred += 1
            del still_text[column]
            continue

        if named_like_a_date:
            parsed_dates, ordering = _read_dates(distinct)
            if parsed_dates.notna().sum() / non_null_before >= named_date_ratio:
                unreadable = int(non_null_before - parsed_dates.notna().sum())
                if unreadable:
                    unparsed_date_cells += unreadable
                    notes.append(f"{unreadable} {column} values could not be read as dates.")
                if ordering:
                    notes.append(_ordering_note(column, ordering))
                cleaned[column] = _drop_timezone(parsed_dates)
                datetime_columns_inferred += 1
                del still_text[column]
                continue

        # "02134" or "007" is a code: converting it would drop the leading zero.
        leading_zero = _flags(distinct.values.astype("string").str.match(r"^0\d")).any()
        # Whole numbers under a key name ("Order No") are keys; decimals under one
        # are measurements whatever the name says.
        keys = is_identifier_name(column) and not _holds_decimals(distinct)
        readings = None if leading_zero else _readings_with_units(distinct, probe)
        if readings is not None:
            numbers, units, markers = readings
            cleaned[column] = distinct.expand(numbers)
            numeric_columns_inferred += 1
            per_value = units.iloc[: len(distinct.values)]
            rows = pd.Series(distinct.counts).groupby(per_value.to_numpy()).sum().sort_values(ascending=False)
            found = [str(unit) for unit in rows.index]
            unit_column = f"{column} unit"
            if len(found) > 1 and unit_column not in cleaned.columns:
                cleaned.insert(cleaned.columns.get_loc(column) + 1, unit_column, distinct.expand(units))
            else:
                unit_column = None
            # An example with a spread shows best what was read.
            read = numbers.iloc[: len(distinct.values)].notna().to_numpy()
            spread = _flags(distinct.values.astype("string").str.contains("±", regex=False)) & read
            first = int(np.flatnonzero(spread if spread.any() else read)[0])
            example, value = str(distinct.values.iloc[first]).strip(), float(numbers.iloc[first])
            notes.append(_readings_note(column, example, value, found, unit_column))
            empty = distinct.rows(np.zeros(len(distinct.values), dtype=bool))
            notes.extend(_marker_notes(column, distinct, markers, empty, None, non_null_before))
            unread = distinct.rows(numbers.isna().to_numpy()[: len(distinct.values)] & ~markers)
            if unread.any():
                notes.append(_unread_note(column, distinct.column()[unread]))
            del still_text[column]
            continue
        if not keys and not leading_zero and _numeric_share(probe) >= PROBE_RATIO:
            numbers, markers, comma = _read_numbers(distinct)
            numbers, yes_no_note = _yes_no_in_flags(column, distinct, numbers, markers)
            if yes_no_note:
                notes.append(yes_no_note)
            parsed_numeric = distinct.expand(numbers)
            unread_values = numbers.isna().to_numpy()[: len(distinct.values)] & ~markers
            unread = distinct.rows(unread_values)
            # One short token repeated among numbers ("T" for a trace) is a marker,
            # as long as it is not most of the column.
            token = _lone_token(distinct, unread_values)
            token_cells = unread if token and unread.sum() <= 0.5 * non_null_before else unread & False
            marked = int(distinct.counts[markers].sum())
            readable = non_null_before - marked - int(token_cells.sum())
            if readable and parsed_numeric.notna().sum() / readable >= numeric_ratio:
                if token_cells.any() and token in TRACE_TOKENS - {"t"}:
                    parsed_numeric = parsed_numeric.mask(token_cells, 0.0)
                cleaned[column] = parsed_numeric
                numeric_columns_inferred += 1
                if comma:
                    long_fraction = _flags(distinct.text.str.match(_LONG_COMMA_FRACTION.pattern))
                    if long_fraction.any():
                        first = int(np.flatnonzero(long_fraction)[0])
                        written = str(distinct.values.iloc[first]).strip()
                        comma_columns.append((column, written, float(numbers.iloc[first])))
                notes.extend(_marker_notes(column, distinct, markers, token_cells, token, non_null_before))
                limits = _upper_limits(distinct, markers)
                if len(limits):
                    upper_limits[column] = limits
                skipped = unread & ~token_cells
                if skipped.any():
                    notes.append(_unread_note(column, distinct.column()[skipped]))
                del still_text[column]
                continue

        if not named_like_a_date:
            # Detect dates in columns without a date-like name ("Month", "FY").
            # Numbers are tried first, so bare years stay numbers here; only a
            # year-named column is turned into dates, after this loop.
            # A sample is checked first because most text columns are not dates.
            sample = distinct.head(date_sample_size)
            if len(sample) and _speculative_dates(sample).notna().mean() >= unnamed_date_ratio:
                parsed_dates, ordering = _read_dates(distinct)
                if parsed_dates.notna().sum() / non_null_before >= unnamed_date_ratio:
                    if ordering:
                        notes.append(_ordering_note(column, ordering))
                    cleaned[column] = _drop_timezone(parsed_dates)
                    datetime_columns_inferred += 1
                    del still_text[column]

    if comma_columns:
        notes.append(_decimal_comma_note(comma_columns))

    # Codes typed in place of a reading (-999, 999.9) would drive every figure
    # on their columns, so they are set to missing before anything is computed.
    cleaned, blanked_codes, blanked_notes = blank_codes(cleaned)
    notes.extend(blanked_notes)
    for column, limits in upper_limits.items():
        if note := _reached_note(cleaned, column, limits):
            notes.append(note)

    # 0/1 columns that each mark one level of a category are put back together.
    cleaned, one_hot_notes = _join_one_hot(cleaned)
    notes.extend(one_hot_notes)

    # Whole numbers written as dates (19540601, 202403), now that text columns
    # of them have become numbers and codes among them are gone.
    real_dates = any(
        is_datetime64_any_dtype(cleaned[column]) and column not in clock_columns for column in cleaned.columns
    )
    for column, read, layout in _digit_date_columns(cleaned, date, real_dates):
        cleaned[column] = read
        datetime_columns_inferred += 1
        first, last = (
            stamp.strftime("%b %Y") if layout == "YYYYMM" else format_moment(stamp)
            for stamp in (read.min(), read.max())
        )
        notes.append(
            f"{column} holds whole numbers written as {layout} ({first} to {last}), so it was read as dates."
        )

    paired = _date_and_time(cleaned, clock_columns)
    if paired:
        day, time = paired
        times = cleaned[time]
        cleaned[day] = cleaned[day] + (times - times.dt.normalize())
        cleaned = cleaned.drop(columns=[time])
        datetime_columns_inferred -= 1
        notes = [note for note in notes if not note.startswith(f"{time} holds times of day")]
        notes.append(f"{day} and {time} were combined into one column, {day}, holding the date and time.")

    # Runs after inference, so a column of elapsed seconds held as text is a number.
    elapsed = _elapsed_column(cleaned, date)
    if elapsed:
        column, unit = elapsed
        cleaned[column] = UNDATED_DAY + pd.to_timedelta(cleaned[column].astype("float64") * unit, unit="s")
        datetime_columns_inferred += 1
        notes.append(
            f"{column} counts {UNIT_NAMES[unit]} from the start, so it is used as a time axis "
            "starting at 00:00:00; no calendar date is attached to it."
        )

    # Text columns that stayed text: "not stated" phrases and placeholders are
    # missing values, not a group; ranges a spreadsheet read as dates are put
    # back; and labels spelled several ways are written one way. Keys keep their
    # case: "ab12" and "AB12" may be two accounts.
    for column, distinct in still_text.items():
        unstated, unstated_notes = _unstated(column, distinct)
        changed = bool(unstated.any())
        if changed:
            notes.extend(unstated_notes)
            distinct = distinct.relabel(distinct.values.mask(unstated))
        restored, range_note = _mangled_ranges(column, distinct)
        if range_note:
            notes.append(range_note)
        if restored is not None:
            distinct, changed = distinct.relabel(restored), True
        if not is_identifier_name(column):
            for merge in (_merge_label_variants, _respelt_labels):
                mapping, merge_note = merge(distinct, column)
                if mapping:
                    notes.append(merge_note)
                    distinct = distinct.relabel(distinct.values.replace(mapping))
                    changed = True
        if changed:
            cleaned[column] = distinct.column()

    # Runs after inference, so year columns held as text have become numbers.
    year_columns = _year_columns(cleaned, None if elapsed else date)
    if year_columns:
        year, month, finer = year_columns
        dates = _built_dates(cleaned, year, month, finer)
        if dates.notna().any():
            unreadable = int(cleaned[year].notna().sum() - dates.notna().sum())
            unparsed_date_cells += unreadable
            datetime_columns_inferred += 1
            if finer:
                notes.extend(_add_timestamp(cleaned, dates, [year, month, *finer.values()], unreadable))
            elif month:
                combined = f"{year}-{month}"
                combined = year if combined in cleaned.columns else combined
                cleaned[year] = dates
                cleaned = cleaned.drop(columns=[month]).rename(columns={year: combined})
                notes.append(
                    f"{year} and {month} were combined into one date column, {combined}, "
                    "on the first day of each month."
                )
            else:
                cleaned[year] = dates
                notes.append(f"{year} holds whole years, so it was read as dates on 1 January of each year.")
            if unreadable and not finer:
                notes.append(f"{unreadable} {year} values could not be read as dates.")

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
        notes.append(f"{count:,} infinite {noun} in {column} {verb} treated as missing.")

    duplicate_rows = int(cleaned.duplicated().sum())
    if drop_duplicates:
        cleaned = cleaned.drop_duplicates()
    cleaned = cleaned.reset_index(drop=True)
    if duplicate_rows and not drop_duplicates:
        notes.append(
            f"{duplicate_rows:,} identical rows were kept, since repeated records are often legitimate."
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
        blanked_codes=blanked_codes,
    )
    return cleaned, report


# A run of date-only rows after rows with times of day is read as summaries at a
# coarser grain when it is this long, covers this many days, and overlaps the timed rows.
MIN_SUMMARY_ROWS = 24
MIN_SUMMARY_SHARE = 0.01
MIN_SUMMARY_DAYS = 3


def _timed(dates: pd.Series) -> np.ndarray:
    """True for rows whose timestamp carries a time of day other than midnight."""
    return (dates.notna() & (dates != dates.dt.normalize())).to_numpy()


def set_aside_coarser_rows(frame: pd.DataFrame, date_column: str) -> tuple[pd.DataFrame, str]:
    """Leave out a run of date-only rows that follows (or precedes) rows with times of day.

    Daily averages appended below hourly readings repeat what those readings
    say. Mixed in, they would weight their days twice and blend two grains in
    every average, so they are set aside with a note. Returns the frame and the
    note, which is "" when nothing was set aside.
    """
    dates = frame[date_column]
    timed = _timed(dates)
    if not timed.any() or timed.mean() < 0.5:
        return frame, ""
    first, last = np.flatnonzero(timed)[[0, -1]]
    # A reading at midnight that keeps the rows' usual step belongs with them.
    values = dates.to_numpy().astype("int64")
    step = np.median(np.diff(values[timed][-1_000:]))
    while last + 1 < len(frame) and values[last + 1] - values[last] == step:
        last += 1
    while first > 0 and values[first] - values[first - 1] == step:
        first -= 1
    runs = {"last": slice(last + 1, len(frame)), "first": slice(0, first)}
    for where, rows in runs.items():
        run = dates.iloc[rows].dropna()
        if len(run) < max(MIN_SUMMARY_ROWS, MIN_SUMMARY_SHARE * len(frame)):
            continue
        days = run.dt.normalize()
        rest = dates[timed]
        overlapping = days.between(rest.min().normalize(), rest.max()).mean()
        if days.nunique() < MIN_SUMMARY_DAYS or overlapping < 0.5:
            continue
        kept = frame.drop(index=frame.index[rows]).reset_index(drop=True)
        note = (
            f"The {where} {rows.stop - rows.start:,} rows give {date_column} as a date with no "
            f"time of day ({format_moment(run.min())} to {format_moment(run.max())}), unlike the "
            "rows with times of day beside them. They look like summaries at a coarser grain, "
            "so they were set aside rather than mixed with the readings they summarize."
        )
        return kept, note
    return frame, ""


# A log that starts again part-way through the file goes back to before this
# share of the readings so far; a clock correction or a re-sent stretch does not.
RESTART_QUANTILE = 0.1
MIN_BLOCK_ROWS = 100
# A log has a time for nearly every reading. A file with fewer distinct times than this
# shares each time across many rows, as a table sorted by species with a year beside it does.
MIN_LOG_TIMES = MIN_BLOCK_ROWS // 5
MIN_BLOCK_SHARE = 0.01
MAX_BLOCKS = 12
# Columns with more distinct values than this are not checked as the series label.
MAX_LABEL_VALUES = 1_000


def _restarts(dates: np.ndarray) -> list[int]:
    """Positions where a log in time order starts again from an earlier time.

    Small steps back (a clock reset by minutes, a re-sent stretch) are not
    restarts: the time must fall back to the earliest tenth of the block so far.
    The block is in time order apart from such steps, so its earliest tenth
    ends about a tenth of the way through it.
    """
    backward = np.flatnonzero(np.diff(dates) < 0) + 1
    # Many steps back mean rows that are not in time order at all: Formula 1's races
    # listed season by season from 2009 back to 1950, then onwards from 2010.
    if len(backward) > max(2 * MAX_BLOCKS, 0.01 * len(dates)):
        return [0]
    starts = [0]
    for position in backward:
        start = starts[-1]
        if position - start < MIN_BLOCK_ROWS:
            continue
        if dates[position] <= dates[start + int((position - start) * RESTART_QUANTILE)]:
            starts.append(int(position))
    smallest = max(MIN_BLOCK_ROWS, MIN_BLOCK_SHARE * len(dates))
    ends = [*starts[1:], len(dates)]
    return [start for start, end in zip(starts, ends, strict=True) if start == 0 or end - start >= smallest]


def _labels_series(frame: pd.DataFrame, date_column: str, starts: np.ndarray) -> bool:
    """Whether some column already names each log (a file sorted by site, then by date).

    Such a column changes value wherever time starts again, and within each of
    its values time never starts again. Its groups must be large enough for a
    restart to show, so a day number does not pass for a log label.
    """
    present = frame[frame[date_column].notna()]
    for column in present.columns:
        series = present[column]
        if column == date_column or is_datetime64_any_dtype(series):
            continue
        if is_numeric_dtype(series) and bool((series.dropna() % 1 != 0).any()):
            continue  # a reading, not a label
        if not 2 <= series.nunique(dropna=True) <= MAX_LABEL_VALUES:
            continue
        values = series.to_numpy()
        if any(values[start] == values[start - 1] for start in starts if start > 0):
            continue
        if series.value_counts().median() < 2 * MIN_BLOCK_ROWS:
            continue
        groups = present[date_column].groupby(series, sort=False)
        if all(len(_restarts(dates.to_numpy().astype("int64"))) == 1 for _, dates in groups):
            return True
    return False


def _block_name(frame: pd.DataFrame) -> str:
    return next(name for name in ("Block", "File block", "Log block") if name not in frame.columns)


def separate_stacked_logs(
    frame: pd.DataFrame, date_column: str
) -> tuple[pd.DataFrame, str | None, str]:
    """Label the logs in a file that holds several, one after another.

    When time starts again part-way through a file and no column says which
    log a row came from, every pooled figure mixes the logs. Each stretch is
    labelled in a new Block column so it can be compared as its own series.
    Returns the frame, the new column's name (None when there is one log) and
    a note. A file whose timestamps mostly repeat a fixed number of times with
    nothing to tell the repeats apart gets a note only. Dates shared by many
    rows (three years across 344 rows sorted by species) are not a log's times,
    so they never start a new log.
    """
    dates = frame[date_column]
    present = dates.notna().to_numpy()
    if present.sum() < 2 * MIN_BLOCK_ROWS or dates.nunique() < MIN_LOG_TIMES:
        return frame, None, ""
    positions = np.flatnonzero(present)
    starts = _restarts(dates.to_numpy()[present].astype("int64"))
    if len(starts) < 2:
        return frame, None, _side_by_side_note(frame, date_column)
    if len(starts) > MAX_BLOCKS:
        return frame, None, ""
    if _labels_series(frame.iloc[positions], date_column, np.asarray(starts)):
        return frame, None, ""
    blocks = np.searchsorted(positions[starts], np.arange(len(frame)), side="right")
    blocks = np.maximum(blocks, 1)
    name = _block_name(frame)
    labelled = frame.assign(**{name: [f"Block {block}" for block in blocks]})
    pairs = pd.DataFrame({"date": dates, "block": blocks}).dropna().drop_duplicates()
    shared = int((pairs.groupby("date").size() > 1).sum())
    described = []
    for block in range(1, len(starts) + 1):
        rows = np.flatnonzero(blocks == block)
        covered = dates.iloc[rows].dropna()
        described.append(
            f"Block {block} (rows {rows[0] + 1:,} to {rows[-1] + 1:,}, "
            f"{format_moment(covered.min())} to {format_moment(covered.max())})"
        )
    note = (
        f"{date_column} starts again part-way through the file, so it holds {len(starts)} logs "
        f"one after another: {', '.join(described[:-1])} and {described[-1]}."
    )
    if shared:
        note += f" {shared:,} timestamps appear in more than one of them."
    note += f" They are kept apart in a {name} column, so each can be read as its own series."
    return labelled, name, note


# Share of timestamps that must repeat the same number of times for a file to
# read as several series recorded side by side.
SIDE_BY_SIDE_SHARE = 0.8


def _side_by_side_note(frame: pd.DataFrame, date_column: str) -> str:
    """Note a log whose timestamps nearly all repeat k times with nothing telling the repeats apart."""
    dates = frame[date_column].dropna()
    if _timed(dates).mean() < 0.5:
        return ""
    counts = dates.value_counts()
    repeats = int(counts.mode().iloc[0]) if not counts.empty else 1
    if not 2 <= repeats <= MAX_BLOCKS or (counts == repeats).mean() < SIDE_BY_SIDE_SHARE:
        return ""
    if frame.loc[dates.index].duplicated(keep=False).mean() > 0.5:
        return ""
    # A label naming the series has about as many values as there are repeats; a
    # reading tells the repeats apart only by differing.
    for column in frame.columns:
        if column == date_column or not 2 <= frame[column].nunique(dropna=True) <= 2 * repeats:
            continue
        if not frame.loc[dates.index, [date_column, column]].duplicated().any():
            return ""
    return (
        f"Most {date_column} timestamps appear {repeats} times with different readings, and no "
        f"column says which reading is which, so the file may hold {repeats} series recorded "
        "side by side. Every figure pools them."
    )


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
            f"- **{row['Column']}**: {row['Type']}; {row['Missing %']:.2f}% missing; "
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
