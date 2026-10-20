"""The public API: one call in, one object out.

    import ada
    report = ada.analyze("orders.csv")
    print(report)

Everything else is a property or a method on what ``analyze`` returned. The two
names worth learning are ``analyze`` and :class:`Refusal`.

A refusal is not an error. ADA declines a well-formed request it cannot answer
honestly -- a forecast on four periods, a question the parser cannot represent --
and says so with a code your code can branch on, rather than returning a
plausible number. An error means the input could not be used at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import pandas as pd

from .aggregation import TrendSeries, build_trend, segment_frame
from .anomalies import MIN_PERIODS as ANOMALY_MIN_PERIODS
from .anomalies import Anomaly, detect_anomalies
from .business_insights import BusinessBrief, analyze_business, build_business_report
from .file_io import list_sample_datasets as _list_sample_datasets
from .file_io import read_tabular_file
from .forecasting import Forecast, build_forecast
from .nlq import QueryAnswer, answer_question, suggested_questions, unsupported_phrasing
from .pipeline import apply_focus, apply_role_selection, focus_options, prepare_analysis
from .schema import ColumnRoles

#: Rows analyzed by default. Older rows are dropped first, and the count that
#: did not make it is on :attr:`Analysis.rows_not_analyzed`.
DEFAULT_ROW_LIMIT = 250_000

#: The largest file the hosted app accepts. The library does not enforce it --
#: it is here so a caller deciding whether to chunk can read the number.
MAX_FILE_BYTES = 25 * 1024 * 1024

#: The shape of what ``to_dict`` returns. Bumped when a key changes meaning.
SCHEMA_VERSION = 1

#: Every refusal code ADA emits today. New codes arrive in minor releases and
#: an existing code never changes meaning, so branch with a default case.
#: ``"unknown"`` is reserved for a code a newer engine returns to older code.
REFUSAL_CODES = (
    "no_date_column",
    "no_dimension_column",
    "no_measure_column",
    "not_enough_periods",
    "thin_history",
    "unsupported_phrasing",
    "unrepresentable_question",
    "unknown_segment_value",
    "unknown",
)


class AdaError(ValueError):
    """The input could not be used. Subclasses ValueError, which the engine already raises."""


class UnreadableData(AdaError):
    """The bytes are not a table ADA can read."""


class NothingToAnalyze(AdaError):
    """The table is empty, or has no columns."""


@dataclass(frozen=True)
class Refusal:
    """ADA declining to answer, with a reason your code can read.

    Falsy, so ``if result:`` reads naturally and a refusal cannot be mistaken
    for a result by accident::

        forecast = report.forecast()
        if not forecast:
            print(forecast.message)      # "8 periods are needed; this file has 4."
        else:
            print(forecast.method)
    """

    code: str
    message: str
    #: What the request would have needed, when that can be said plainly.
    needed: str | None = None
    #: What it actually had.
    had: str | None = None

    def __bool__(self) -> bool:
        return False

    def __str__(self) -> str:
        return self.message


def _frame_from(data: Any, sheet: str | None) -> tuple[pd.DataFrame, str]:
    """A DataFrame and a name for it, from a frame, a path or raw bytes."""
    if isinstance(data, pd.DataFrame):
        return data, "DataFrame"
    if isinstance(data, bytes | bytearray):
        return read_tabular_file(bytes(data), "upload.csv", sheet), "upload"
    if isinstance(data, str | Path):
        path = Path(data)
        if not path.is_file():
            raise UnreadableData(f"No such file: {path}")
        try:
            return read_tabular_file(path.read_bytes(), path.name, sheet), path.name
        except ValueError as error:  # file_io raises bare ValueError
            raise UnreadableData(str(error)) from error
    raise UnreadableData(
        f"analyze() takes a DataFrame, a path or bytes, not {type(data).__name__}."
    )


def analyze(
    data: pd.DataFrame | str | Path | bytes,
    *,
    date: str | None = None,
    measure: str | None = None,
    dimension: str | None = None,
    frequency: str | None = None,
    row_limit: int = DEFAULT_ROW_LIMIT,
    sheet: str | None = None,
    source_name: str | None = None,
) -> Analysis:
    """Read a table and work out what is in it.

    ``data`` may be a DataFrame, a path to a CSV or Excel file, or raw bytes.

    ADA detects which column is the date, the measure, the segment and the
    identifier. Pass ``date``, ``measure`` or ``dimension`` to override it --
    ``ada schema FILE`` on the command line prints what it chose.

    ``frequency`` pins the period grain to one of ``D W M Q Y``. Leaving it
    unset lets ADA pick from the recording cadence, which is right for
    exploring a file once and wrong for anything scheduled: a weekly table can
    roll up to months, and anomaly detection needs
    """ + f"{ANOMALY_MIN_PERIODS}" + """ periods before it will flag anything.
    Set it explicitly for a job that runs more than once.

    Raises :class:`AdaError` if the input cannot be used at all.
    """
    frame, detected_name = _frame_from(data, sheet)
    try:
        prepared = prepare_analysis(frame, row_limit=row_limit)
    except ValueError as error:
        raise NothingToAnalyze(str(error)) from error

    roles = prepared.detected_roles
    if date or measure or dimension:
        roles = apply_role_selection(
            roles,
            date=date if date is not None else (roles.date or ""),
            measure=measure if measure is not None else (roles.measure or ""),
            dimension=dimension if dimension is not None else (roles.dimension or ""),
        )
    return Analysis(
        frame=prepared.dataframe,
        roles=roles,
        cleaning=prepared.cleaning_report,
        rows_not_analyzed=prepared.truncated_rows,
        period=(prepared.analyzed_from, prepared.analyzed_to),
        frequency=frequency,
        source_name=source_name or detected_name,
    )


def read_file(contents: bytes, filename: str, *, sheet: str | None = None) -> pd.DataFrame:
    """Read uploaded bytes into a DataFrame, for callers holding an upload.

    Handles what real exports do: a BOM, UTF-16, latin-1, cp1252, semicolon and
    tab delimiters, a report title above the header, ragged rows.
    """
    try:
        return read_tabular_file(contents, filename, sheet)
    except ValueError as error:
        raise UnreadableData(str(error)) from error


def samples() -> dict[str, Path]:
    """The datasets bundled with the package, by name."""
    return _list_sample_datasets()


def sample(name: str = "Ecommerce Orders") -> pd.DataFrame:
    """One of the bundled datasets, for trying ADA with no data of your own."""
    available = _list_sample_datasets()
    chosen = available.get(name)
    if chosen is None:
        wanted = name.replace("-", " ").replace("_", " ").casefold()
        chosen = next((p for n, p in available.items() if n.casefold() == wanted), None)
    if chosen is None:
        raise AdaError(f"No sample named {name!r}. Available: {', '.join(sorted(available))}")
    return pd.read_csv(chosen)


@dataclass(frozen=True)
class Analysis:
    """What ADA found. Build one with :func:`analyze`, never directly."""

    frame: pd.DataFrame
    roles: ColumnRoles
    cleaning: Any
    #: Rows older than the row limit, which were not analyzed. Rows removed as
    #: unusable are counted separately on :attr:`cleaning`.
    rows_not_analyzed: int
    period: tuple[pd.Timestamp | None, pd.Timestamp | None]
    frequency: str | None = None
    source_name: str = "DataFrame"
    _trends: dict = field(default_factory=dict, repr=False, compare=False)

    @property
    def rows_analyzed(self) -> int:
        return len(self.frame)

    @cached_property
    def _brief(self) -> BusinessBrief:
        return analyze_business(self.frame, self.roles)

    @property
    def headline(self) -> str:
        return self._brief.headline

    @property
    def summary(self) -> str:
        return self._brief.summary

    @property
    def evidence(self) -> tuple:
        """Every finding, each carrying the ``calculation`` that produced it."""
        return tuple(self._brief.evidence)

    @property
    def kpis(self) -> tuple:
        return tuple(self._brief.kpis)

    @property
    def recommendations(self) -> tuple:
        """Labelled interpretation, never cause."""
        return tuple(self._brief.recommendations)

    @property
    def suggestions(self) -> tuple[str, ...]:
        """Questions :meth:`ask` can answer on this table. Not a guess: these
        are generated from the detected schema and the parser's own grammar."""
        return tuple(suggested_questions(self.frame, self.roles))

    def trend(self, frequency: str | None = None) -> TrendSeries | Refusal:
        """The measure over time. ``.notes`` says what had to be assumed."""
        if not self.roles.date:
            return Refusal("no_date_column", "This table has no date column, so there is no timeline.")
        grain = frequency or self.frequency
        if grain not in self._trends:
            self._trends[grain] = build_trend(self.frame, self.roles, frequency=grain)
        series = self._trends[grain]
        if series.frame.empty:
            return Refusal("not_enough_periods", "No periods could be built from this table.")
        return series

    @property
    def anomalies(self) -> tuple[Anomaly, ...] | Refusal:
        """Periods outside a calibrated band.

        An empty tuple means nothing left the band. A :class:`Refusal` means
        ADA could not look -- no timeline, or too little history. In a job that
        runs on a schedule those are different outcomes and must not be
        confused, so this never flattens one into the other.

        Both are falsy, so test the type rather than the truth value::

            found = report.anomalies
            if isinstance(found, ada.Refusal):
                ...   # could not look, and found.message says why
            elif not found:
                ...   # looked, nothing outside the band
        """
        series = self.trend()
        if isinstance(series, Refusal):
            return series
        periods = len(series.frame)
        if periods < ANOMALY_MIN_PERIODS:
            return Refusal(
                "not_enough_periods",
                f"{ANOMALY_MIN_PERIODS} periods are needed to calibrate a band; "
                f"this table has {periods}.",
                needed=f"{ANOMALY_MIN_PERIODS} periods",
                had=f"{periods} periods",
            )
        return detect_anomalies(series.frame)

    def forecast(self, horizon: int = 6) -> Forecast | Refusal:
        """A guarded baseline. Check ``.backtest`` before quoting it."""
        series = self.trend()
        if isinstance(series, Refusal):
            return series
        result = build_forecast(series.frame, horizon=horizon)
        if result is None:
            periods = len(series.frame)
            return Refusal(
                "thin_history",
                f"A forecast needs more history than this; the table has {periods} periods.",
                needed="more periods",
                had=f"{periods} periods",
            )
        return result

    def segments(self, limit: int = 12) -> pd.DataFrame | Refusal:
        """The breakdown by segment, as a frame ready for your own chart."""
        if not self.roles.dimension:
            return Refusal("no_dimension_column", "This table has no segment column to group by.")
        return segment_frame(self.frame, self.roles, limit=limit)

    def ask(self, question: str) -> QueryAnswer | Refusal:
        """Answer a plain-English question locally with pandas.

        Refuses rather than answering a question it cannot represent. The two
        codes are worth telling apart: ``unsupported_phrasing`` means the
        grammar has no shape for it, ``unrepresentable_question`` means the
        shape exists but this table cannot satisfy it.
        """
        if unsupported_phrasing(question):
            return Refusal(
                "unsupported_phrasing",
                "ADA has no way to represent that question, so it will not guess at a near one.",
            )
        answer = answer_question(question, self.frame, self.roles)
        if answer is None:
            return Refusal(
                "unrepresentable_question",
                "ADA could not build a plan for that question against this table.",
            )
        return answer

    def focus(self, value: str) -> Analysis | Refusal:
        """Narrow to one segment. The analysis regroups by the next dimension."""
        if not self.roles.dimension:
            return Refusal("no_dimension_column", "This table has no segment column to focus on.")
        if value not in focus_options(self.frame, self.roles):
            return Refusal("unknown_segment_value", f"No {self.roles.dimension} called {value!r}.")
        frame, roles = apply_focus(self.frame, self.roles, value)
        return Analysis(
            frame=frame,
            roles=roles,
            cleaning=self.cleaning,
            rows_not_analyzed=self.rows_not_analyzed,
            period=self.period,
            frequency=self.frequency,
            source_name=f"{self.source_name} · {value}",
        )

    def to_markdown(self) -> str:
        """The executive brief, the same one the app downloads."""
        return build_business_report(self.frame, self._brief, source_name=self.source_name)

    def to_dict(self) -> dict:
        """A JSON-safe view, for storing findings or shipping them onward.

        ``schema_version`` is the contract. The prose in ``headline``,
        ``summary`` and each ``statement`` is written for people and may be
        reworded in any release; ``calculation`` and the numbers are the part
        worth asserting on.
        """
        start, end = self.period
        return {
            "schema_version": SCHEMA_VERSION,
            "source": self.source_name,
            "rows_analyzed": self.rows_analyzed,
            "rows_not_analyzed": self.rows_not_analyzed,
            "period": {
                "from": start.isoformat() if start is not None else None,
                "to": end.isoformat() if end is not None else None,
            },
            "roles": {
                "date": self.roles.date,
                "measure": self.roles.measure,
                "dimension": self.roles.dimension,
                "identifier": self.roles.identifier,
            },
            "headline": self.headline,
            "summary": self.summary,
            "cleaning_notes": list(self.cleaning.notes),
            "evidence": [
                {
                    "kind": item.kind,
                    "title": item.title,
                    "value": item.value,
                    "statement": item.statement,
                    "calculation": item.calculation,
                }
                for item in self.evidence
            ],
            "recommendations": [
                {
                    "priority": item.priority,
                    "title": item.title,
                    "action": item.action,
                    "rationale": item.rationale,
                }
                for item in self.recommendations
            ],
            "kpis": [
                {"label": item.label, "value": item.value, "context": item.context}
                for item in self.kpis
            ],
        }

    def __str__(self) -> str:
        start, end = self.period
        dated = start is not None and end is not None
        span = f"{start:%d %b %Y} to {end:%d %b %Y}" if dated else "no date range"
        lines = [
            f"{self.source_name} — {self.rows_analyzed:,} rows, {span}",
            f"  date={self.roles.date}  measure={self.roles.measure}  segment={self.roles.dimension}",
            "",
            self.headline,
            self.summary,
        ]
        if self.rows_not_analyzed:
            lines.append(
                f"\n{self.rows_not_analyzed:,} older rows were outside the row limit "
                "and not analyzed."
            )
        if self.evidence:
            lines.append("")
            for item in self.evidence:
                lines.append(f"  {item.value}  {item.title}")
                lines.append(f"      {item.statement}")
                lines.append(f"      calc: {item.calculation}")
        return "\n".join(lines)
