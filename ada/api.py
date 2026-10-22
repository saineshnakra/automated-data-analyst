"""Public API.

    import ada
    report = ada.analyze("measurements.csv")
    print(report)

``analyze`` returns an :class:`Analysis`; everything else is a property or
method on it. When ADA cannot answer a request (a forecast from four periods,
a question it can't parse) it returns a falsy :class:`Refusal` with a code
instead of a number. Input it cannot read at all raises :class:`AdaError`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import cached_property
from pathlib import Path
from typing import Any

import pandas as pd

from .aggregation import TrendSeries, build_trend, fixed_design, group_coverage, segment_frame
from .analysis import column_profile
from .anomalies import MIN_PERIODS as ANOMALY_MIN_PERIODS
from .anomalies import Anomaly, detect_anomalies, finer_trend
from .business_insights import BusinessBrief, analyze_business, build_business_report
from .file_io import list_sample_datasets as _list_sample_datasets
from .file_io import read_tabular_file
from .forecasting import Forecast, build_forecast
from .formatting import format_moment, is_undated, show_undated
from .metrics import AGGREGATION_CHOICES, metric_for
from .nlq import (
    QueryAnswer,
    WholeTable,
    answer_question,
    rephrase_hint,
    suggested_questions,
    unsupported_phrasing,
)
from .pipeline import apply_focus, focus_options, prepare_analysis
from .quality import quality_frame
from .quality import scan as scan_quality
from .schema import ColumnRoles

#: Rows analyzed by default. Over it, a table with a date keeps its newest
#: rows and one without a date is sampled at random; see
#: :attr:`Analysis.rows_not_analyzed` and :attr:`Analysis.rows_note`.
DEFAULT_ROW_LIMIT = 1_000_000

#: Upload limit of the hosted app. Not enforced by the library.
MAX_FILE_BYTES = 25 * 1024 * 1024

#: Version of the ``to_dict`` layout. Bumped when a key changes meaning.
SCHEMA_VERSION = 1

#: Refusal codes. New codes may be added in minor releases and existing ones
#: keep their meaning, so give any branching on them a default case.
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
    """The input could not be used. A subclass of ValueError."""


class UnreadableData(AdaError):
    """The input is not a table ADA can read."""


class NothingToAnalyze(AdaError):
    """The table is empty, or has no columns."""


@dataclass(frozen=True)
class Refusal:
    """ADA declining to answer, with a machine-readable ``code``.

    Refusals are falsy::

        forecast = report.forecast()
        if not forecast:
            print(forecast.message)      # "8 periods are needed; this file has 4."
        else:
            print(forecast.method)
    """

    code: str
    message: str
    #: What the request needed, when that can be stated (e.g. "8 periods").
    needed: str | None = None
    #: What the data had.
    had: str | None = None

    def __bool__(self) -> bool:
        return False

    def __str__(self) -> str:
        return self.message


def _frame_from(data: Any, sheet: str | None) -> tuple[pd.DataFrame, str]:
    """Return a DataFrame and a display name for a frame, path or raw bytes."""
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
        except ValueError as error:
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
    aggregation: str | None = None,
    row_limit: int = DEFAULT_ROW_LIMIT,
    sheet: str | None = None,
    source_name: str | None = None,
) -> Analysis:
    """Read a table, detect its columns and return an :class:`Analysis`.

    ``data`` may be a DataFrame, a path to a CSV, TSV or Excel file, or raw bytes.

    ADA picks a date column, a measure, a grouping column and an identifier
    on its own; ``date``, ``measure`` and ``dimension`` override that choice
    (``ada schema FILE`` prints what was picked). A ``date`` column of whole
    years is read as 1 January of each year. ``frequency`` fixes the
    period grain to one of ``s min h D W M Q Y``; without it the grain follows how
    often the data was recorded. Fix it for scheduled jobs so the grain
    doesn't change between runs.

    ``aggregation`` says how the measure combines across rows: ``"sum"`` for a
    total, ``"mean"`` for an average. Without it, money and counts are summed
    and anything else is averaged, judged from the column name.

    Raises :class:`AdaError` if the input can't be used, including an override
    that names a column the table doesn't have.
    """
    if aggregation is not None and aggregation not in AGGREGATION_CHOICES:
        raise AdaError(
            f"aggregation must be 'sum' (a total) or 'mean' (an average), not {aggregation!r}."
        )
    frame, detected_name = _frame_from(data, sheet)
    try:
        prepared = prepare_analysis(frame, row_limit=row_limit, date=date)
    except ValueError as error:
        raise NothingToAnalyze(str(error)) from error

    overrides = {
        role: column
        for role, column in (("date", date), ("measure", measure), ("dimension", dimension))
        if column is not None
    }
    cleaned = prepared.dataframe
    missing = [column for column in overrides.values() if column not in cleaned.columns]
    if missing:
        available = ", ".join(map(str, cleaned.columns))
        raise AdaError(f"No column named {missing[0]!r}. Columns: {available}")
    if date is not None and not pd.api.types.is_datetime64_any_dtype(cleaned[date]):
        raise AdaError(f"{date!r} could not be read as dates, so it can't be the date column.")
    if measure is not None and not pd.api.types.is_numeric_dtype(cleaned[measure]):
        raise AdaError(f"{measure!r} is not numeric, so it can't be the measure.")
    roles = replace(prepared.detected_roles, **overrides, aggregation=aggregation)
    if measure is not None:
        # The runners-up were for ADA's own pick; a chosen measure has none. A
        # column chosen as the measure is a quantity, even if it looked like codes.
        roles = replace(
            roles,
            measure_alternatives=(),
            factors=tuple(column for column in roles.factors if column != measure),
            flags=tuple(column for column in roles.flags if column != measure),
            dimensions=tuple(
                column for column in roles.dimensions if column != measure or column not in roles.factors
            ),
        )
    name = source_name or detected_name
    # Cleaning has already set clear codes such as 999.9 or -100000 to missing
    # in every numeric column, with a note for each.
    cleaning = prepared.cleaning_report
    frame = prepared.dataframe
    whole = None
    if roles.focus and dimension is None:
        # A long table's rows measure different things; the brief covers one of them.
        # Questions about another one are answered from every row (see nlq.WholeTable).
        whole = WholeTable(frame, replace(roles, focus=None), roles.dimension, roles.focus)
        name = f"{name} · {roles.focus}"
        frame, roles = apply_focus(frame, roles, roles.focus)
    return Analysis(
        frame=frame,
        roles=replace(roles, focus=None),
        cleaning=cleaning,
        rows_not_analyzed=prepared.truncated_rows,
        period=(prepared.analyzed_from, prepared.analyzed_to),
        frequency=frequency,
        source_name=name,
        rows_note=prepared.rows_note,
        whole=whole,
    )


def read_file(contents: bytes, filename: str, *, sheet: str | None = None) -> pd.DataFrame:
    """Read CSV, TSV or Excel bytes into a DataFrame.

    Handles a BOM, UTF-16, latin-1 and cp1252, semicolon and tab delimiters,
    a title line above the header, and ragged rows.
    """
    try:
        return read_tabular_file(contents, filename, sheet)
    except ValueError as error:
        raise UnreadableData(str(error)) from error


def samples() -> dict[str, Path]:
    """Paths of the bundled sample datasets, by name."""
    return _list_sample_datasets()


def sample(name: str = "Ecommerce Orders") -> pd.DataFrame:
    """Load a bundled sample dataset by name ("Lab Measurements", "lab-measurements")."""
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
    """Result of :func:`analyze`. Not meant to be constructed directly."""

    frame: pd.DataFrame
    roles: ColumnRoles
    cleaning: Any
    #: Rows left out by the row limit. Rows removed during cleaning are
    #: reported on :attr:`cleaning`.
    rows_not_analyzed: int
    period: tuple[pd.Timestamp | None, pd.Timestamp | None]
    frequency: str | None = None
    source_name: str = "DataFrame"
    #: Which rows were analyzed when some were left out ("a random sample of
    #: 1,000,000 of 3,108,545 rows"), or "" when every row was.
    rows_note: str = ""
    #: Every row of a long table, before the brief's focus on one parameter; Ask uses it.
    whole: WholeTable | None = field(default=None, repr=False, compare=False)
    _trends: dict = field(default_factory=dict, repr=False, compare=False)

    @property
    def rows_analyzed(self) -> int:
        return len(self.frame)

    @cached_property
    def _brief(self) -> BusinessBrief:
        return analyze_business(
            self.frame, self.roles, frequency=self.frequency, blanked=self.cleaning.blanked_codes
        )

    @property
    def headline(self) -> str:
        return self._brief.headline

    @property
    def summary(self) -> str:
        return self._brief.summary

    @property
    def evidence(self) -> tuple:
        """Findings, each with the ``calculation`` that produced it."""
        return tuple(self._brief.evidence)

    @property
    def kpis(self) -> tuple:
        return tuple(self._brief.kpis)

    @property
    def recommendations(self) -> tuple:
        """Suggested next steps. These are interpretation, not causal claims."""
        return tuple(self._brief.recommendations)

    @property
    def suggestions(self) -> tuple[str, ...]:
        """Example questions that :meth:`ask` can answer for this table."""
        return tuple(suggested_questions(self.frame, self.roles))

    def profile(self) -> pd.DataFrame:
        """One row per column: type, missing values, distinct values and an example."""
        return column_profile(self.frame)

    def quality(self) -> pd.DataFrame:
        """Every issue the data checks found, one row each: issues about values
        before issues about gaps, each ranked by the rows they cover.

        Columns: ``Check``, ``Columns``, ``Rows`` (cells or rows affected),
        ``Share %``, ``Detail`` (a plain sentence), ``Rule`` (what fired) and
        ``Examples``. Clear missing-value codes (-999, 999.9) were set to
        missing during cleaning and are listed here as such; nothing else is
        changed by the checks. An empty frame means nothing was flagged. See
        ``ada/quality.py`` for the rules.
        """
        return quality_frame(scan_quality(self.frame, self.roles, blanked=self.cleaning.blanked_codes))

    def trend(self, frequency: str | None = None) -> TrendSeries | Refusal:
        """The measure per period. ``.notes`` lists any adjustments made to the periods."""
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
        """Periods outside the expected range of the trend.

        An empty tuple means nothing was flagged. A :class:`Refusal` means ADA
        could not check (no date column, or too few periods even at a finer
        grain than the trend's). Both are falsy, so check the type::

            found = report.anomalies
            if isinstance(found, ada.Refusal):
                ...   # could not check; found.message says why
            elif not found:
                ...   # checked, nothing flagged
        """
        series = self.trend()
        if isinstance(series, Refusal):
            return series
        if (refusal := self._design_refusal(series)) is not None:
            return refusal
        # Too few periods at this grain may still be enough at a finer one.
        series = finer_trend(self.frame, self.roles, series) or series
        periods = len(series.frame)
        if periods < ANOMALY_MIN_PERIODS:
            return Refusal(
                "not_enough_periods",
                f"{ANOMALY_MIN_PERIODS} periods are needed to calibrate a band; "
                f"this table has {periods}.",
                needed=f"{ANOMALY_MIN_PERIODS} periods",
                had=f"{periods} periods",
            )
        return detect_anomalies(
            series.frame,
            floor_at_zero=metric_for(self.roles.measure, self.roles).floors_at_zero,
            grain=series.frequency,
        )

    def forecast(self, horizon: int = 6) -> Forecast | Refusal:
        """Trend-based forecast with a backtest. Check ``.backtest`` before using it."""
        series = self.trend()
        if isinstance(series, Refusal):
            return series
        if (refusal := self._design_refusal(series)) is not None:
            return refusal
        result = build_forecast(
            series.frame,
            horizon=horizon,
            floor_at_zero=metric_for(self.roles.measure, self.roles).floors_at_zero,
            grain=series.frequency,
        )
        if result is None:
            periods = len(series.frame)
            return Refusal(
                "thin_history",
                f"A forecast needs more history than this; the table has {periods} periods.",
                needed="more periods",
                had=f"{periods} periods",
            )
        return result

    def _design_refusal(self, series: TrendSeries) -> Refusal | None:
        """A refusal when rows are counted and the design fixes how many each period holds."""
        design = fixed_design(self.frame, self.roles, series.frequency)
        if design is None:
            return None
        return Refusal("no_measure_column", f"This table has no measure to follow. {design.sentence}")

    def segments(self, limit: int = 12) -> pd.DataFrame | Refusal:
        """The measure broken down by the grouping column, as a DataFrame.

        With a date column, From and To give each group's first and last date
        with a value, since groups ranked side by side may cover different years.
        A group with no values for the measure has a missing Value, not zero.
        """
        if not self.roles.dimension:
            return Refusal("no_dimension_column", "This table has no segment column to group by.")
        frame = segment_frame(self.frame, self.roles, limit=limit)
        if self.roles.date:
            # Times with no calendar date show as the time of day, not on their stand-in date.
            frame = frame.join(show_undated(group_coverage(self.frame, self.roles)), on="Segment")
        return frame

    def ask(self, question: str) -> QueryAnswer | Refusal:
        """Answer a plain-English question with pandas, locally.

        Returns a :class:`Refusal` with code ``unsupported_phrasing`` when the
        question uses a form the parser doesn't support (e.g. "revenue > 200"),
        or ``unrepresentable_question`` when it can't be mapped onto this table.
        The second names the column the question may have meant, when there is one.
        """
        if unsupported_phrasing(question):
            return Refusal(
                "unsupported_phrasing",
                "ADA has no way to represent that question, so it will not guess at a near one.",
            )
        answer = answer_question(
            question, self.frame, self.roles, rows_not_analyzed=self.rows_not_analyzed, whole=self.whole
        )
        if answer is None:
            message = "ADA could not build a plan for that question against this table."
            hint = rephrase_hint(
                question, self.frame, self.roles, rows_not_analyzed=self.rows_not_analyzed, whole=self.whole
            )
            return Refusal("unrepresentable_question", f"{message} {hint}" if hint else message)
        return answer

    def focus(self, value: str) -> Analysis | Refusal:
        """Re-run the analysis on one value of the grouping column."""
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
            rows_note=self.rows_note,
        )

    def to_markdown(self) -> str:
        """The Markdown report, the same one the app offers as a download."""
        return build_business_report(self.frame, self._brief, source_name=self.source_name)

    def to_dict(self) -> dict:
        """JSON-serialisable summary of the analysis.

        The keys are versioned by ``schema_version``. Prose fields (``headline``,
        ``summary``, ``statement``) may be reworded between releases.
        """
        start, end = self.period
        return {
            "schema_version": SCHEMA_VERSION,
            "source": self.source_name,
            "rows_analyzed": self.rows_analyzed,
            "rows_not_analyzed": self.rows_not_analyzed,
            "rows_note": self.rows_note,
            "period": {
                "from": _moment(start),
                "to": _moment(end),
            },
            "roles": {
                "date": self.roles.date,
                "measure": self.roles.measure,
                "dimension": self.roles.dimension,
                "identifier": self.roles.identifier,
                # How the measure was combined: sum, mean, or count when there is none.
                "aggregation": metric_for(self.roles.measure, self.roles).aggregation,
            },
            "headline": self.headline,
            "summary": self.summary,
            "cleaning_notes": list(self.cleaning.notes),
            "data_checks": self.quality().to_dict(orient="records"),
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
        span = f"{format_moment(start)} to {format_moment(end)}" if dated else "no date range"
        if dated and is_undated(start):
            span += " (times with no calendar date)"
        roles = self.roles
        combined = f" ({metric_for(roles.measure, roles).aggregation})" if roles.measure else ""
        lines = [
            f"{self.source_name} — {self.rows_analyzed:,} rows, {span}",
            f"  date={roles.date}  measure={roles.measure}{combined}  segment={roles.dimension}",
            "",
            self.headline,
        ]
        if self.summary and self.summary != self.headline:
            lines.append(self.summary)
        if self.rows_note:
            lines.append(f"\nAnalyzed {self.rows_note} (row limit).")
        if self.evidence:
            lines.append("")
            for item in self.evidence:
                lines.append(f"  {item.value}  {item.title}")
                lines.append(f"      {item.statement}")
                lines.append(f"      calc: {item.calculation}")
        return "\n".join(lines)


def _moment(stamp: pd.Timestamp | None) -> str | None:
    """ISO text for a timestamp, or the time alone ("15:06:28") when it has no calendar date."""
    if stamp is None:
        return None
    return format_moment(stamp) if is_undated(stamp) else stamp.isoformat()
