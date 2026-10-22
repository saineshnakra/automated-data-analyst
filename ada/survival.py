"""Time-to-event data: find a follow-up time and its event flag, and compare survival between groups.

A follow-up time beside a 0/1 event column is censored where the event is 0:
those subjects were event-free when last seen, so their time is a lower bound.
Averaging the time, or comparing it between those with and without the event,
is circular: subjects who died were bound to have shorter follow-up. Survival
is instead estimated by Kaplan-Meier (``statsmodels`` ``SurvfuncRight``),
compared between groups by the log-rank test (``survdiff``) and sized by a Cox
proportional-hazards model (``PHReg``).
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.duration.survfunc import SurvfuncRight, survdiff

from .compare import format_p, holm_adjust
from .formatting import format_percentage, group_label, list_names
from .schema import ColumnRoles

# Parts of a column name that mark a follow-up time, matched within the name
# with its punctuation removed ("days_to_death", "OS.time", "futime"), and
# whole words that do ("os", "pfs", "months").
TIME_PARTS = ("surviv", "time", "followup", "futime", "duration", "daysto", "monthsto", "yearsto")
TIME_WORDS = frozenset({
    "os", "pfs", "dfs", "efs", "rfs", "ttr", "tte", "days", "months", "weeks", "years", "surv",
})
# Parts of a column name that mark the event a follow-up time ends in.
EVENT_PARTS = (
    "event", "death", "dead", "died", "deceased", "mortal", "vital", "status", "censor", "relapse",
    "recur", "progress", "fail",
)
# Stems that all mean death, so "survival" pairs with "eventdeath" or "vital_status".
DEATH_STEMS = frozenset({
    "", "death", "dead", "died", "deceased", "mortality", "vital", "surv", "os", "overall",
})
# Parts of a time or event name that make the event a death, so the result is "survival".
DEATH_PARTS = ("surviv", "death", "dead", "died", "deceas", "mortal", "vital")
# Words in a text value that mark it as the event rather than censoring ("Dead", "Yes").
EVENT_VALUE_PARTS = ("dead", "died", "deceas", "death", "event", "yes", "true", "relaps", "recur", "progress",
                     "fail", "churn")
# The parts of a time column's name taken out to find what it times.
TIME_STEM_PARTS = (
    "survival", "surv", "followup", "futime", "duration", "time", "daysto", "monthsto", "yearsto",
)
# Rows and events a pair needs before survival is estimated.
MIN_SUBJECTS = 20
MIN_EVENTS = 5
# Groupings compared by the log-rank test: few levels, each with enough subjects.
MAX_GROUP_LEVELS = 8
MIN_GROUP_SUBJECTS = 5
GROUPS_NAMED = 3
# Groupings in the Cox model, the clearest first.
MAX_COX_TERMS = 3
ALPHA = 0.05
# The landmark is the largest of these, times a power of ten, within the
# central follow-up: 10 years for a record whose follow-up reaches about 10.5.
NICE_STEPS = (1.0, 2.0, 2.5, 5.0)
LANDMARK_QUANTILE = 0.75


def _squashed(name: str) -> str:
    return re.sub(r"[^a-z]", "", str(name).casefold())


def _words(name: str) -> set[str]:
    return set(re.findall(r"[a-z]+", str(name).casefold()))


def is_time_name(name: str) -> bool:
    """Whether a column name marks a follow-up time ("survival", "OS_months", "days_to_death")."""
    squashed = _squashed(name)
    return any(part in squashed for part in TIME_PARTS) or bool(_words(name) & TIME_WORDS)


def is_event_name(name: str) -> bool:
    """Whether a column name marks the event a follow-up ends in ("eventdeath", "vital_status")."""
    squashed = _squashed(name)
    return any(part in squashed for part in EVENT_PARTS)


def _stem_of_time(name: str) -> str:
    """A time column's name without its time words: "timerecurrence" gives "recurrence"."""
    squashed = _squashed(name)
    for part in TIME_STEM_PARTS:
        squashed = squashed.replace(part, "")
    words = _words(name)
    for word in ("days", "months", "weeks", "years"):
        if word in words:
            squashed = squashed.replace(word, "")
    return squashed


def _stem_of_event(name: str) -> str:
    """An event column's name without its event words: "eventdeath" gives "death", "os_status" "os"."""
    squashed = _squashed(name)
    for part in ("event", "status", "censored", "censor", "indicator", "flag"):
        squashed = squashed.replace(part, "")
    return squashed


def _pair_score(time: str, event: str) -> int:
    """How well a time and an event column belong together: 2 for a shared stem, 1 for death, 0 for none."""
    time_stem, event_stem = _stem_of_time(time), _stem_of_event(event)
    if time_stem and time_stem == event_stem:
        return 2
    if time_stem in DEATH_STEMS and event_stem in DEATH_STEMS:
        return 1
    return 0


@dataclass(frozen=True)
class SurvivalPair:
    """A follow-up time and the 0/1 column saying whether it ended in the event."""

    time: str
    event: str
    #: The event column's value that marks the event (1, or True), and the one that marks censoring.
    event_value: object
    censored_value: object
    #: True when the column marks censoring rather than the event ("censored" = 1 is no event).
    inverted: bool = False

    @property
    def about_death(self) -> bool:
        """Whether the event is a death, so the time to it is survival rather than a time to an event."""
        names = _squashed(self.time) + _squashed(self.event)
        return any(part in names for part in DEATH_PARTS) or bool(_words(self.time) & {"os"})

    def status(self, dataframe: pd.DataFrame) -> pd.Series:
        """1 where the follow-up ended in the event, 0 where it was censored, NaN where unknown."""
        values = dataframe[self.event]
        return (values == self.event_value).astype(float).where(values.notna())


def survival_pair(dataframe: pd.DataFrame, roles: ColumnRoles) -> SurvivalPair | None:
    """The follow-up time and event column of a survival table, or None.

    A time is a non-negative numeric column named like a follow-up, with at
    least 10 distinct values; an event is a two-valued column named like an
    event, death or status. They pair when their names share a stem
    ("os_time", "os_status") or both refer to death ("survival",
    "eventdeath"). The pair needs MIN_SUBJECTS rows and MIN_EVENTS events.
    """
    times = []
    for column in dataframe.columns:
        if column == roles.date or not is_time_name(column):
            continue
        values = dataframe[column]
        if not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
            continue
        present = values.dropna()
        if present.nunique() >= 10 and float(present.min()) >= 0:
            times.append(column)
    events = {}
    for column in dataframe.columns:
        if column not in times and is_event_name(column) and (coding := _event_coding(dataframe[column])):
            events[column] = coding
    best: tuple[int, str, str] | None = None
    for time in times:
        for event in events:
            score = _pair_score(time, event)
            if score and (best is None or score > best[0]):
                best = (score, time, event)
    if best is None:
        return None
    _, time, event = best
    happened, censored = events[event]
    inverted = "censor" in _squashed(event)
    if inverted:
        happened, censored = censored, happened
    pair = SurvivalPair(time, event, happened, censored, inverted)
    status = pair.status(dataframe)
    usable = dataframe[time].notna() & status.notna()
    if usable.sum() < MIN_SUBJECTS or status[usable].sum() < MIN_EVENTS:
        return None
    return pair


def _event_coding(values: pd.Series) -> tuple[object, object] | None:
    """The (event, censored) values of a two-valued column, or None when it can't be told which is which.

    Numbers and yes/no values put the event on the larger (1, True, or 2 in
    a 1/2 status); text needs one value that names an event ("Dead", "Yes")
    and one that does not.
    """
    levels = values.dropna().unique().tolist()
    if len(levels) != 2:
        return None
    if all(isinstance(level, bool | int | float | np.integer | np.floating | np.bool_) for level in levels):
        low, high = sorted(levels)
        return high, low
    marked = [any(part in str(level).casefold() for part in EVENT_VALUE_PARTS) for level in levels]
    if sum(marked) != 1:
        return None
    return (levels[0], levels[1]) if marked[0] else (levels[1], levels[0])


def landmark(times: np.ndarray) -> float:
    """A round follow-up time to quote survival at, within the central follow-up."""
    reach = float(np.quantile(times, LANDMARK_QUANTILE))
    if reach <= 0:
        return float(np.max(times))
    scale = 10.0 ** np.floor(np.log10(reach))
    steps = [step * scale for step in (*NICE_STEPS, 10.0) if step * scale <= reach]
    return float(max(steps))


def survival_at(time: np.ndarray, status: np.ndarray, moment: float) -> float:
    """The Kaplan-Meier survival at ``moment``, or NaN beyond the last follow-up."""
    if moment > float(np.max(time)):
        return float("nan")
    fitted = SurvfuncRight(time, status)
    reached = fitted.surv_times <= moment
    return float(fitted.surv_prob[reached][-1]) if reached.any() else 1.0


@dataclass(frozen=True)
class GroupSurvival:
    """Survival compared across one grouping by the log-rank test."""

    grouping: str
    statistic: float
    df: int
    p_value: float
    adjusted_p: float
    #: Kaplan-Meier survival at the landmark per level, in the grouping's order.
    at_landmark: tuple[tuple[object, float, int], ...]


@dataclass(frozen=True)
class HazardRatio:
    level: object
    reference: object
    ratio: float
    low: float
    high: float
    p_value: float


@dataclass(frozen=True)
class SurvivalSummary:
    pair: SurvivalPair
    subjects: int
    events: int
    median: float
    landmark: float
    overall: float
    groups: tuple[GroupSurvival, ...]
    #: The Cox model's hazard ratios for the clearest grouping, and what it was adjusted for.
    hazards: tuple[HazardRatio, ...] = ()
    adjusted_for: tuple[str, ...] = ()


def _candidate_groupings(dataframe: pd.DataFrame, roles: ColumnRoles, pair: SurvivalPair) -> list[str]:
    """Groupings with few levels, each holding enough subjects, other than the pair itself."""
    names = dict.fromkeys([*roles.dimensions, *roles.factors])
    chosen = []
    for column in names:
        if column in (pair.time, pair.event) or column not in dataframe.columns or is_event_name(column):
            continue
        counts = dataframe[column].value_counts(dropna=True)
        if 2 <= len(counts) <= MAX_GROUP_LEVELS and counts.min() >= MIN_GROUP_SUBJECTS:
            chosen.append(column)
    return chosen


def _levels(values: pd.Series) -> list:
    levels = values.dropna().unique().tolist()
    try:
        return sorted(levels)
    except TypeError:
        return sorted(levels, key=str)


def _cox(
    dataframe: pd.DataFrame, pair: SurvivalPair, terms: list[str], status: pd.Series
) -> tuple[HazardRatio, ...]:
    """Hazard ratios of the first term's levels against its first level, adjusted for the other terms.

    Returns () when the model does not converge to finite estimates.
    """
    working = dataframe[[pair.time, *terms]].assign(__status=status).dropna()
    if len(working) < MIN_SUBJECTS:
        return ()
    blocks = []
    for term in terms:
        levels = _levels(working[term])
        coded = pd.get_dummies(pd.Categorical(working[term], categories=levels), prefix=term, prefix_sep="=")
        blocks.append(coded.iloc[:, 1:].astype(float))
    exog = pd.concat(blocks, axis=1)
    exog.index = working.index
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            endog = working[pair.time].to_numpy(dtype=float)
            fitted = PHReg(endog, exog.to_numpy(), status=working["__status"].to_numpy()).fit()
    except (np.linalg.LinAlgError, ValueError):
        return ()
    params, errors = np.asarray(fitted.params), np.asarray(fitted.bse)
    if not (np.all(np.isfinite(params)) and np.all(np.isfinite(errors))) or np.any(np.abs(params) > 10):
        return ()
    first = terms[0]
    levels = _levels(working[first])
    ratios = []
    for index, level in enumerate(levels[1:]):
        estimate, error = params[index], errors[index]
        ratios.append(HazardRatio(
            level=level,
            reference=levels[0],
            ratio=float(np.exp(estimate)),
            low=float(np.exp(estimate - 1.96 * error)),
            high=float(np.exp(estimate + 1.96 * error)),
            p_value=float(np.asarray(fitted.pvalues)[index]),
        ))
    return tuple(ratios)


def summarize_survival(dataframe: pd.DataFrame, roles: ColumnRoles, pair: SurvivalPair) -> SurvivalSummary:
    """Kaplan-Meier survival overall and per grouping, Holm-adjusted log-rank tests, and a Cox model."""
    status = pair.status(dataframe)
    usable = dataframe[pair.time].notna() & status.notna()
    frame = dataframe.loc[usable]
    time = frame[pair.time].to_numpy(dtype=float)
    events = status[usable].to_numpy(dtype=float)
    overall = SurvfuncRight(time, events)
    median = float(overall.quantile(0.5))
    moment = landmark(time)
    tests = []
    for grouping in _candidate_groupings(frame, roles, pair):
        labelled = frame[grouping].notna().to_numpy()
        groups = frame[grouping].to_numpy()[labelled]
        try:
            statistic, p_value = survdiff(time[labelled], events[labelled], groups)
        except (ValueError, np.linalg.LinAlgError, ZeroDivisionError):
            continue
        if not np.isfinite(p_value):
            continue
        at_landmark = []
        kept_time, kept_events = time[labelled], events[labelled]
        for level in _levels(frame[grouping]):
            chosen = groups == level
            share = survival_at(kept_time[chosen], kept_events[chosen], moment)
            at_landmark.append((level, share, int(chosen.sum())))
        tests.append((grouping, float(statistic), len(at_landmark) - 1, float(p_value), tuple(at_landmark)))
    adjusted = holm_adjust([test[3] for test in tests]) if tests else []
    groups = sorted(
        (
            GroupSurvival(*test[:4], float(adjusted_p), test[4])
            for test, adjusted_p in zip(tests, adjusted, strict=True)
        ),
        key=lambda item: (item.adjusted_p, -item.statistic),
    )
    clear = [item.grouping for item in groups if item.adjusted_p < ALPHA][:MAX_COX_TERMS]
    hazards = _cox(frame, pair, clear, status[usable]) if clear else ()
    return SurvivalSummary(
        pair=pair,
        subjects=int(usable.sum()),
        events=int(events.sum()),
        median=median,
        landmark=moment,
        overall=survival_at(time, events, moment),
        groups=tuple(groups),
        hazards=hazards,
        adjusted_for=tuple(clear[1:]) if hazards else (),
    )


def _number(value: float) -> str:
    return f"{value:,.3g}" if abs(value) < 1_000 else f"{value:,.0f}"


def describe_survival(summary: SurvivalSummary) -> tuple[str, str]:
    """The survival card's statement and its one-sentence lead."""
    pair = summary.pair
    censored = summary.subjects - summary.events
    event = group_label(pair.event, pair.event_value)
    statement = (
        f"{pair.time} is follow-up time, censored where {pair.event} = {pair.censored_value}: "
        f"{summary.events:,} of {summary.subjects:,} subjects had the event ({event}) and {censored:,} were "
        f"still event-free when last seen, so averaging {pair.time} or comparing it across {pair.event} "
        "would only restate who was followed longer."
    )
    what = "survival" if pair.about_death else f"time to {event}"
    median = (
        f"median {what} {_number(summary.median)}" if np.isfinite(summary.median)
        else f"median {what} not reached"
    )
    moment = _number(summary.landmark)
    overall = format_percentage(summary.overall * 100) if np.isfinite(summary.overall) else "—"
    statement += f" Kaplan-Meier: {median}; {overall} were still event-free at {pair.time} = {moment}."
    lead = (
        f"By Kaplan-Meier, {overall} were still event-free at {pair.time} = {moment} "
        f"({summary.events:,} events in {summary.subjects:,})."
    )
    clear = [item for item in summary.groups if item.adjusted_p < ALPHA]
    if clear:
        first = clear[0]
        shown = [
            f"{format_percentage(share * 100)} for {group_label(first.grouping, level)}"
            for level, share, _ in first.at_landmark
            if np.isfinite(share)
        ]
        subject = f"{what[0].upper()}{what[1:]} differs by {first.grouping}"
        statement += (
            f" {subject} (log-rank χ² = {first.statistic:.1f}, df = {first.df}, Holm-adjusted "
            f"{format_p(first.adjusted_p)}): at {pair.time} = {moment}, {list_names(shown, len(shown))}."
        )
        lead = (
            f"{subject} (log-rank {format_p(first.adjusted_p)}): at {pair.time} = {moment}, "
            f"{list_names(shown, len(shown))}."
        )
        others = [f"{item.grouping} ({format_p(item.adjusted_p)})" for item in clear[1:GROUPS_NAMED]]
        if others:
            statement += f" Also clear: {list_names(others, len(others))}."
        if summary.hazards:
            ratios = [
                f"{group_label(first.grouping, ratio.level)} against "
                f"{group_label(first.grouping, ratio.reference)} {ratio.ratio:.2g} "
                f"(95% CI {ratio.low:.2g} to {ratio.high:.2g})"
                for ratio in summary.hazards
            ]
            adjusted = ""
            if summary.adjusted_for:
                adjusted = f", adjusted for {list_names(list(summary.adjusted_for))}"
            statement += f" Cox hazard ratios{adjusted}: {'; '.join(ratios)}."
    elif summary.groups:
        statement += (
            f" No grouping separates survival clearly: the log-rank tests across "
            f"{list_names([item.grouping for item in summary.groups])} give Holm-adjusted p of "
            f"{format_p(summary.groups[0].adjusted_p).removeprefix('p ').removeprefix('= ')} at best."
        )
    return statement, lead
