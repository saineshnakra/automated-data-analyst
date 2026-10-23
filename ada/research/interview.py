"""``ada init``: write a first ``study.yml`` from the data and a few questions.

ADA reads the file, guesses each column's role, unit and type, and proposes an
answer to every question; the researcher presses Enter to accept or types a
better one. With ``--yes`` every guess is accepted, every analysis is marked
exploratory, since only the researcher can say what was planned, and the design
is marked unconfirmed. Analyses ``ada build`` would refuse are left out, with the
reason. Either way the result is a plain, commented YAML file meant to be read
and edited.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

from ..analysis import clean_dataframe
from ..file_io import read_tabular_file
from ..schema import (
    _UNIT_ANNOTATION,
    detect_roles,
    is_design_name,
    is_entity_name,
    is_subject_name,
    looks_like_identifier,
    looks_like_subject,
)
from .analyses import refusal
from .processing import _as_labels, read
from .study import DESIGNS, SHARING, StudyError, parse, sha256_of

#: Numbers often used to mean "missing" in lab and survey files.
SENTINELS = (-9, -99, -999, -9999, 99, 999, 9999, -1)
#: Words in a column name that mark a time point rather than a date.
TIME_WORDS = {
    "time",
    "day",
    "days",
    "week",
    "weeks",
    "visit",
    "timepoint",
    "hour",
    "hours",
    "month",
    "wave",
    "session",
    "period",
    "minute",
    "minutes",
    "cycle",
    "year",
    "years",
}
#: Words that mark a cluster: what several subjects or samples share.
CLUSTER_WORDS = {
    "site",
    "centre",
    "center",
    "clinic",
    "hospital",
    "school",
    "litter",
    "cage",
    "animal",
    "donor",
    "farm",
    "village",
    "ward",
    "tank",
    "pond",
    "household",
    "practice",
    "colony",
}
#: Words that mark columns that may identify a person and should not be shared.
PERSONAL_WORDS = {
    "name",
    "surname",
    "email",
    "phone",
    "address",
    "postcode",
    "zip",
    "dob",
    "birth",
    "ssn",
    "mrn",
    "nhs",
    "initials",
}
#: Units recognised in the brackets that end a column name, lowercased, each with its prefix: "mg",
#: "µmol", "kpa". Combinations such as "mg/dL", "kg/m2" or "mL per min" are built from these.
#: Letters that are as often a label as a unit ("a", "c", "k", "t", "y") are left out.
UNITS = frozenset(
    {
        *("g", "kg", "mg", "µg", "ug", "mcg", "ng", "pg", "lb", "lbs", "oz"),
        *("m", "cm", "mm", "km", "µm", "um", "nm", "in", "inch", "inches", "ft", "mi"),
        *("l", "ml", "µl", "ul", "dl", "cl", "cc", "gal"),
        *("s", "sec", "secs", "second", "seconds", "ms", "µs", "us", "min", "mins", "minute", "minutes"),
        *("h", "hr", "hrs", "hour", "hours", "d", "day", "days", "wk", "wks", "week", "weeks"),
        *("mo", "month", "months", "yr", "yrs", "year", "years"),
        *("°", "°c", "°f", "degc", "degf", "deg", "degree", "degrees", "rad"),
        *("pa", "kpa", "mpa", "hpa", "gpa", "bar", "mbar", "atm", "mmhg", "cmh2o", "psi", "torr"),
        *("j", "kj", "mj", "cal", "kcal", "wh", "kwh", "ev", "w", "kw", "mw"),
        *("v", "mv", "µv", "uv", "ma", "µa", "ohm", "ohms", "ω", "kω"),
        *("hz", "khz", "mhz", "ghz", "bpm", "rpm"),
        *("mol", "mmol", "µmol", "umol", "nmol", "pmol", "meq", "iu", "u", "ppm", "ppb"),
        *("db", "dba", "lux", "lx", "ha", "acre", "acres"),
        *("cells", "copies", "cfu", "beats", "breaths"),
        *("%", "‰", "usd", "eur", "gbp", "$", "€", "£"),
    }
)
MAX_GROUP_LEVELS = 12
JOURNALS = ("default", "plos", "nature", "cell", "science", "elife")


@dataclass
class Guess:
    """What ADA proposes for each question, before the researcher answers."""

    data_path: str
    sheet: str | None
    sha256: str
    columns: list[str]
    frame: pd.DataFrame
    title: str = ""
    observation: str = ""
    identifiers: list[str] = field(default_factory=list)
    outcomes: list[str] = field(default_factory=list)
    groups: list[str] = field(default_factory=list)
    time: str | None = None
    covariates: list[str] = field(default_factory=list)
    units: dict[str, str] = field(default_factory=dict)
    types: dict[str, str] = field(default_factory=dict)
    missing_codes: list = field(default_factory=list)
    design: str = "independent"
    #: Whether the researcher answered the design question; with ``--yes`` the design is only a guess.
    confirmed: bool = False
    subject: str | None = None
    cluster: str | None = None
    analyses: list[dict] = field(default_factory=list)
    #: Proposed analyses left out because ``ada build`` would refuse them: (question, reason).
    dropped: list[tuple[str, str]] = field(default_factory=list)
    planned: list[str] = field(default_factory=list)
    sharing: str = "processed"
    withhold: list[str] = field(default_factory=list)
    authors: list[str] = field(default_factory=list)
    license: str = ""
    journal: str = "default"
    notes: list[str] = field(default_factory=list)
    #: The folder study.yml is written to; ``data_path`` is relative to it.
    root: Path = Path(".")


def _words(name: str) -> set[str]:
    return set(re.findall(r"[a-z]+", str(name).lower()))


def _unit_of(column: str) -> str:
    """The unit in the brackets that end a column name, as written: "mmHg" in "Systolic BP (mmHg)".

    Only known units (see UNITS) and their combinations count, such as "mg/dL", "kg/m²" or "mL/min/1.73m2".
    Other words in brackets say what kind of value a column holds, not its unit: "Abs" in "Jitter(Abs)" is
    absolute jitter, and "(0-10)" is a scale. Those leave the unit blank.
    """
    match = _UNIT_ANNOTATION.search(str(column))
    if not match:
        return ""
    text = " ".join(match.group(1).split())
    parts = [part for part in re.split(r"[/·*\s]+", text.lower().replace("μ", "µ")) if part and part != "per"]
    # A power ("m2", "m^2", "m²", "s-1") and a number in front ("1.73m2") belong to the unit.
    atoms = [re.sub(r"^[0-9.]+|\^?-?[0-9²³]+$", "", part) for part in parts]
    return text if atoms and all(atom in UNITS for atom in atoms) else ""


def _type_of(series: pd.Series, name: str, identifiers: list[str]) -> str:
    present = series.dropna()
    if name in identifiers:
        return "text" if not is_numeric_dtype(series) else "categorical"
    if is_datetime64_any_dtype(series):
        return "date"
    if is_numeric_dtype(series):
        distinct = set(present.unique())
        if distinct and distinct <= {0, 1}:
            return "binary"
        return "numeric"
    levels = present.astype(str).nunique()
    if levels <= max(MAX_GROUP_LEVELS * 4, 1) or levels <= 0.5 * max(len(present), 1):
        return "categorical"
    return "text"


def _sentinels(frame: pd.DataFrame, numeric: list[str]) -> list[int]:
    """Sentinel codes that occur in numeric columns far outside the column's other values."""
    found = []
    for code in SENTINELS:
        for column in numeric:
            values = frame[column].dropna()
            hits = values == code
            if not hits.any() or hits.all():
                continue
            others = values[~hits]
            low, high = others.quantile(0.01), others.quantile(0.99)
            spread = max(high - low, 1e-9)
            if code < low - spread or code > high + spread or (code < 0 <= others.min()):
                found.append(code)
                break
    return found


def guess(data_path: Path, root: Path, sheet: str | None = None) -> Guess:
    """Read the data and propose an answer to every question."""
    raw = read_tabular_file(data_path.read_bytes(), data_path.name, sheet)
    frame, _ = clean_dataframe(raw)
    roles = detect_roles(frame)
    columns = [str(column) for column in frame.columns]
    # Relative even when the data sits outside the study folder ("../data/x.csv"), so the
    # study rebuilds the same on another researcher's computer.
    relative = Path(os.path.relpath(data_path.resolve(), root.resolve())).as_posix()
    result = Guess(
        data_path=relative,
        sheet=sheet,
        sha256=sha256_of(data_path),
        columns=columns,
        frame=frame,
        title=data_path.stem.replace("_", " ").replace("-", " ").strip().capitalize(),
        root=root,
    )

    identifiers = [
        column
        for column in columns
        if looks_like_identifier(column, frame[column])
        or (is_entity_name(column) and not is_numeric_dtype(frame[column]))
        # Number codes of a unit measured more than once: subject# 1 to 42 over 5,875 recordings.
        # Numbered sites and other clusters are found below.
        or (looks_like_subject(column, frame[column]) and not _words(column) & CLUSTER_WORDS)
    ]
    if roles.identifier and roles.identifier not in identifiers:
        identifiers.insert(0, roles.identifier)
    # A column with one value tells no rows apart, such as the station in one station's file.
    identifiers = [column for column in identifiers if frame[column].nunique() > 1]
    clusters = [
        column
        for column in columns
        if _words(column) & CLUSTER_WORDS
        and column not in identifiers
        and 1 < frame[column].nunique() < len(frame) / 2
    ]
    result.identifiers = identifiers + [column for column in clusters if column not in identifiers]

    times = [
        column
        for column in columns
        if _words(column) & TIME_WORDS
        and column not in result.identifiers
        and frame[column].nunique() <= max(30, len(frame) // 5)
    ]
    result.time = times[0] if times else None

    non_readings = set(roles.non_readings) | set(roles.factors)
    numeric = [
        column for column in roles.numeric if column not in result.identifiers and column != result.time
    ]
    candidates = [column for column in numeric if column not in non_readings]
    outcome = roles.measure if roles.measure in candidates else (candidates[0] if candidates else None)
    result.outcomes = [outcome] if outcome else []

    groups = []
    for column in [roles.dimension, *roles.dimensions, *roles.factors]:
        if (
            column
            and column not in groups
            and column not in result.identifiers
            and column != result.time
            and 1 < frame[column].nunique() <= MAX_GROUP_LEVELS
            and not is_design_name(column)
        ):
            groups.append(column)
    result.groups = groups[:1]
    blocks = [
        column
        for column in columns
        if is_design_name(column)
        and column not in result.identifiers
        and 1 < frame[column].nunique() <= MAX_GROUP_LEVELS
        and column not in result.groups
    ]
    result.covariates = [column for column in candidates if column not in result.outcomes][:4] + [
        column for column in blocks if column not in candidates
    ]

    result.types = {column: _type_of(frame[column], column, result.identifiers) for column in columns}
    for column in columns:
        unit = _unit_of(column)
        if unit:
            result.units[column] = unit
        elif "%" in column or "percent" in _words(column):
            result.units[column] = "%"
    result.missing_codes = _sentinels(frame, [c for c in columns if result.types[c] == "numeric"])

    subjects = [column for column in result.identifiers if column not in clusters]
    # A unit measured more than once, not a key with the odd value repeated or a visit number.
    repeated = [column for column in subjects if looks_like_subject(column, frame[column])]
    if repeated:
        result.subject = repeated[0]
        per_subject = frame.groupby(result.subject, observed=True).size()
        if result.time and frame[result.time].nunique() == 2 and not result.groups:
            # Two time points per subject, such as before and after: a paired comparison of the two.
            result.groups, result.time = [result.time], None
        condition = result.groups[0] if result.groups else None
        if condition and frame[condition].nunique() == 2 and per_subject.max() == 2 and not result.time:
            result.design = "paired"
        else:
            result.design = "repeated"
    elif clusters:
        result.design = "clustered"
        result.cluster = clusters[0]
    if result.design != "independent":
        unit = result.subject or result.cluster
        result.notes.append(f"Guessed a {result.design} design from {unit} ({_repeats(frame, unit)}).")

    result.withhold = [column for column in columns if _words(column) & PERSONAL_WORDS]
    result.observation = _observation(result)
    result.analyses = propose_analyses(result)
    return result


def _observation(found: Guess) -> str:
    if found.subject and found.time:
        return f"one {found.subject} at one {found.time}"
    if found.subject:
        return f"one measurement of one {found.subject}"
    if found.identifiers:
        return f"one {found.identifiers[0]}"
    return "one observation"


def _repeats(frame: pd.DataFrame, column: str) -> str:
    """How a column's values repeat: "42 values over 5,875 rows, 101 to 168 rows each"."""
    counts = frame[column].value_counts()
    low, high = int(counts.min()), int(counts.max())
    each = f"{low:,} rows each" if low == high else f"{low:,} to {high:,} rows each"
    return f"{len(counts):,} values over {int(counts.sum()):,} rows, {each}"


def _codes(series: pd.Series) -> bool:
    """Whether a numeric column holds a few whole-number codes, such as cultivar 1, 2 and 3."""
    present = series.dropna()
    return (
        is_numeric_dtype(present)
        and bool((present % 1 == 0).all())
        and 1 < present.nunique() <= MAX_GROUP_LEVELS
    )


def _kind(found: Guess, column: str) -> str:
    """The type written for a column. A group of number codes is categorical: `compare:` needs it."""
    kind = found.types.get(column, "text")
    if kind == "numeric" and column in found.groups and _codes(found.frame[column]):
        return "categorical"
    return kind


def _levels(found: Guess, column: str) -> list[str]:
    """A group's levels: number codes in their order (the lowest is the reference), others as they come."""
    present = found.frame[column].dropna()
    if is_numeric_dtype(present):
        return list(_as_labels(pd.Series(sorted(present.unique()))))
    return list(dict.fromkeys(present.astype(str)))


def propose_analyses(found: Guess) -> list[dict]:
    """One analysis per outcome and group, time or covariate, for the researcher to mark as planned or not."""
    proposals = []
    for outcome in found.outcomes:
        for group in found.groups:
            proposals.append(
                {
                    "id": _slug(f"{outcome} by {group}"),
                    "outcome": outcome,
                    "compare": group,
                    "question": f"Does {outcome} differ by {group}?",
                }
            )
        if found.time and found.design == "repeated":
            entry = {
                "id": _slug(f"{outcome} over {found.time}"),
                "outcome": outcome,
                "change": found.time,
                "question": f"How does {outcome} change over {found.time}?",
            }
            if found.groups:
                entry["by"] = found.groups[0]
                entry["question"] = (
                    f"Does the change in {outcome} over {found.time} differ by {found.groups[0]}?"
                )
            proposals.append(entry)
        numeric = [name for name in found.covariates if found.types.get(name) == "numeric"]
        for covariate in numeric[:2]:
            proposals.append(
                {
                    "id": _slug(f"{outcome} and {covariate}"),
                    "outcome": outcome,
                    "relate": covariate,
                    "question": f"Is {outcome} related to {covariate}?",
                }
            )
    return proposals


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:48]


class _Asker:
    """Asks each question with ADA's guess as the default; Enter accepts it."""

    def __init__(
        self, ask: Callable[[str], str], say: Callable[[str], None], interactive: bool, columns: list[str]
    ):
        self.ask, self.say, self.interactive, self.columns = ask, say, interactive, columns

    def text(self, question: str, default: str = "") -> str:
        if not self.interactive:
            return default
        shown = f" [{default}]" if default else ""
        answer = self.ask(f"{question}{shown}: ").strip()
        return answer or default

    def columns_of(
        self, question: str, default: list[str], allow_none: bool = True, single: bool = False
    ) -> list[str]:
        hint = "a column name" if single else "comma-separated column names"
        while True:
            answer = self.text(
                f"{question} ({hint}" + (", or none" if allow_none else "") + ")",
                ", ".join(default) or ("none" if allow_none else ""),
            )
            if answer.lower() in ("none", "-", ""):
                return []
            names = (
                [answer.strip()] if single else [name.strip() for name in answer.split(",") if name.strip()]
            )
            unknown = [name for name in names if name not in self.columns]
            if not unknown or not self.interactive:
                return [name for name in names if name in self.columns]
            self.say(f"  Not columns of the file: {', '.join(unknown)}. Columns: {', '.join(self.columns)}")

    def choice(self, question: str, options, default: str) -> str:
        while True:
            answer = self.text(f"{question} ({' / '.join(options)})", default)
            if answer in options or not self.interactive:
                return answer if answer in options else default
            self.say(f"  Please answer one of: {', '.join(options)}")


def interview(
    found: Guess,
    ask: Callable[[str], str] = input,
    say: Callable[[str], None] = print,
    interactive: bool = True,
) -> Guess:
    """Ask each question, with ADA's guess as the default answer."""
    asker = _Asker(ask, say, interactive, found.columns)
    if interactive:
        say(f"\nADA read {found.data_path}: {len(found.frame):,} rows, {len(found.columns)} columns.")
        say(f"Columns: {', '.join(found.columns)}")
        say("Press Enter to accept the suggestion in brackets.\n")
    found.title = asker.text("Study title", found.title)
    found.observation = asker.text("What does one row represent", found.observation)
    found.identifiers = asker.columns_of(
        "Identifier columns (participant, specimen, site or experiment)", found.identifiers
    )
    found.outcomes = asker.columns_of(
        "Outcomes, the measurements the study is about", found.outcomes, allow_none=False
    )
    found.groups = asker.columns_of("Groups or conditions compared", found.groups)
    times = asker.columns_of("Time point column", [found.time] if found.time else [], single=True)
    found.time = times[0] if times else None
    found.covariates = asker.columns_of(
        "Covariates to adjust for or relate to the outcomes",
        [c for c in found.covariates if c not in found.outcomes],
    )
    for column in [*found.outcomes, *found.covariates]:
        if found.types.get(column) == "numeric":
            unit = asker.text(f"Unit of {column} (blank if none)", found.units.get(column, ""))
            if unit:
                found.units[column] = unit
            else:
                found.units.pop(column, None)
    codes = asker.text(
        "Values that mean 'missing' (comma-separated, or none)",
        ", ".join(map(str, found.missing_codes)) or "none",
    )
    found.missing_codes = (
        []
        if codes.lower() == "none"
        else [_number_or_text(code) for code in codes.split(",") if code.strip()]
    )
    unit = found.subject or found.cluster
    if interactive and found.design != "independent" and unit:
        # Rows that share a subject or cluster are not independent; answering independent must be a choice.
        alone = (
            "each row is a different subject"
            if found.subject
            else f"rows of the same {unit} are no more alike than any two rows"
        )
        say(
            f"\n{unit} repeats: {_repeats(found.frame, unit)}. Rows of the same {unit} are not independent, "
            f"so ADA suggests a {found.design} design. Answer independent only if {alone}."
        )
    elif interactive:
        # A few repeated values are more likely a key's duplicates than a subject; the researcher knows.
        for column in found.identifiers:
            doubled = int(found.frame[column].duplicated(keep=False).sum())
            if is_subject_name(column) and doubled:
                say(
                    f"\n{column} repeats on {doubled:,} of {len(found.frame):,} rows. If those rows measure "
                    f"the same {column} more than once, answer repeated."
                )
    found.design = asker.choice("Design", list(DESIGNS), found.design)
    found.confirmed = interactive
    # Columns with one value were left out of the identifiers, but the researcher may have named one.
    several = [column for column in found.identifiers if found.frame[column].nunique() > 1]
    if found.design in ("paired", "repeated"):
        found.subject = _unit_column(
            asker,
            found.frame,
            "Column naming the subject measured more than once",
            [found.subject] if found.subject else several[:1],
        )
    if found.design == "clustered":
        found.cluster = _unit_column(
            asker,
            found.frame,
            "Column naming the cluster each row belongs to",
            [found.cluster] if found.cluster else several[-1:],
        )
    found.analyses = propose_analyses(found)
    _leave_out_refused(found, say)
    if interactive and found.analyses:
        say("\nAnalyses ADA can run for this design:")
        for number, analysis in enumerate(found.analyses, 1):
            say(f"  {number}. {analysis['question']}")
        picked = asker.text("Which were planned before you saw the data? (numbers, or none)", "none")
        numbers = {int(item) for item in re.findall(r"\d+", picked)}
        found.planned = [
            analysis["id"] for number, analysis in enumerate(found.analyses, 1) if number in numbers
        ]
        keep = asker.text("Keep the others as exploratory analyses? (y/n)", "y")
        if keep.lower().startswith("n"):
            found.analyses = [analysis for analysis in found.analyses if analysis["id"] in found.planned]
    found.sharing = asker.choice("Which data can be made public", list(SHARING), found.sharing)
    found.withhold = asker.columns_of("Columns that must never be shared", found.withhold)
    authors = asker.text("Authors (comma-separated)", ", ".join(found.authors))
    found.authors = [name.strip() for name in authors.split(",") if name.strip()]
    found.license = asker.text(
        "Licence for the shared data, such as CC-BY-4.0 (blank to decide later)", found.license
    )
    found.journal = asker.choice("Journal preset for figures", list(JOURNALS), found.journal)
    return found


def _unit_column(asker: _Asker, frame: pd.DataFrame, question: str, default: list[str]) -> str | None:
    """Ask for the subject or cluster column. One with a single value tells no rows apart, so ask again."""
    while True:
        chosen = asker.columns_of(question, default, allow_none=False, single=True)
        if not chosen or frame[chosen[0]].nunique() > 1 or not asker.interactive:
            return chosen[0] if chosen else None
        asker.say(f"  {chosen[0]} has a single value, so it cannot tell one subject or cluster from another.")


def _leave_out_refused(found: Guess, say: Callable[[str], None]) -> None:
    """Leave out each proposed analysis that `ada build` would refuse, saying which and why.

    The study is read and processed as `ada build` will read it, and each analysis goes through the
    build's own checks before fitting (``analyses.refusal``). A study file with a problem of its own
    is left for `ada build` to report.
    """
    import yaml  # noqa: PLC0415

    found.dropped = []
    if not found.analyses:
        return
    try:
        study = parse(yaml.safe_load(render(found)), (found.root / "study.yml").resolve())
        data = read(study).data
    except StudyError:
        return
    kept = []
    for analysis, spec in zip(found.analyses, study.analyses, strict=True):
        reason = refusal(study, data, spec)
        if reason:
            found.dropped.append((analysis["question"], reason))
        else:
            kept.append(analysis)
    found.analyses = kept
    if found.dropped:
        count = len(found.dropped)
        noun = "analysis" if count == 1 else "analyses"
        say(f"\nLeft out {count} proposed {noun} that `ada build` would not run:")
        for question, reason in found.dropped:
            say(f"  - {question} {reason}")


def _number_or_text(text: str):
    text = text.strip()
    try:
        number = float(text)
    except ValueError:
        return text
    return int(number) if number.is_integer() else number


def _yaml(value) -> str:
    """One YAML value on one line, quoted when it needs to be."""
    import yaml  # noqa: PLC0415

    if isinstance(value, list | tuple):
        return "[" + ", ".join(_yaml(item) for item in value) + "]"
    text = yaml.safe_dump(value, default_flow_style=True, allow_unicode=True, width=10_000).strip()
    return (
        text[:-4].strip()
        if text.endswith("\n...") or text.endswith(" ...")
        else text.removesuffix("...").strip()
    )


def _role(found: Guess, column: str) -> str:
    if column in found.outcomes:
        return "outcome"
    if column in found.groups:
        return "group"
    if column == found.time:
        return "time"
    if column in found.covariates:
        return "covariate"
    if column in found.identifiers or column in (found.subject, found.cluster):
        return "identifier"
    return "other"


def render(found: Guess) -> str:
    """The study file as commented YAML."""
    lines = [
        "# study.yml: what ADA needs to know to build this study's assets.",
        "# `ada build` reads it; ADA never rewrites it. Edit anything, then run `ada build` again.",
        "# Each setting is explained in the ADA docs under docs/research.md.",
        "",
        "study:",
        f"  title: {_yaml(found.title)}",
        "  authors:" + ("" if found.authors else " []"),
    ]
    lines += [f"    - name: {_yaml(name)}" for name in found.authors]
    lines += [
        f"  license: {_yaml(found.license)}      # for the shared data, such as CC-BY-4.0; you choose it",
        '  code_license: ""                   # for any code you share, such as MIT',
        "  keywords: []",
        "",
        "data:",
        f"  path: {_yaml(found.data_path)}",
    ]
    if found.sheet:
        lines.append(f"  sheet: {_yaml(found.sheet)}")
    lines += [
        f"  sha256: {found.sha256}",
        "  # ADA stops if the file's sha256 changes; update it only when you change the file on purpose.",
        f"  observation: {_yaml(found.observation)}   # what one row represents",
        f"  missing_codes: {_yaml(found.missing_codes)}   # values that mean 'missing' in every column",
        "",
        "design:",
        f"  type: {found.design}   # " + " | ".join(DESIGNS),
        "  confirmed: true   # you answered the design question"
        if found.confirmed
        else "  confirmed: false   # ADA guessed the design from the data; check it, then set this to true",
    ]
    if found.subject:
        lines.append(f"  subject: {_yaml(found.subject)}   # who or what was measured more than once")
    if found.cluster:
        lines.append(
            f"  cluster: {_yaml(found.cluster)}   # what several rows share, such as a site or an animal"
        )
    if found.time:
        lines.append(f"  time: {_yaml(found.time)}")
    lines += [
        "",
        "# One entry per column. role: outcome | group | time | covariate | identifier | other",
        "# type: numeric | categorical | binary | date | text. For categorical variables the first of",
        "# `levels` is the reference the others are compared with. `range` is checked, never enforced.",
        "variables:",
    ]
    for column in found.columns:
        kind = _kind(found, column)
        lines.append(f"  {_yaml(column)}:")
        lines.append(f"    role: {_role(found, column)}")
        lines.append(f"    type: {kind}")
        if found.units.get(column):
            lines.append(f"    unit: {_yaml(found.units[column])}")
        if kind in ("categorical", "binary") and _role(found, column) == "group":
            levels = _levels(found, column)
            if len(levels) <= MAX_GROUP_LEVELS:
                lines.append(f"    levels: {_yaml(levels)}")
        lines.append('    description: ""')
    lines += [
        "",
        "# Exclusions and changes, applied in order. Each needs a reason, and each is reported with",
        "# the rows before and after. Examples:",
        "#   - exclude:",
        "#       where: {column: viability, below: 0}",
        "#     reason: Readings below zero are instrument errors.",
        "#   - exclude:",
        "#       ids: [P07]",
        "#     reason: Withdrew consent.",
        "#   - transform: {column: dose, to: log_dose, apply: log10}",
        "#     reason: Doses span four orders of magnitude.",
        "processing: []",
        "",
        "decisions:",
        "  missing_data: complete-case   # each analysis uses the rows with every value it needs",
        "  outliers: keep                # keep | flag (flag lists extreme values in the audit; both keep)",
        "  multiple_comparisons: holm    # holm | bonferroni | fdr_bh | none, within each family",
        "  alpha: 0.05",
        "  confidence: 0.95",
        "",
        "# Only these analyses are run. planned: true means decided before seeing the data; everything",
        "# else is reported as exploratory. Analyses in the same family are adjusted together.",
    ]
    if found.dropped:
        lines.append("# Left out because `ada build` would not run them:")
        lines += [f"#   - {question} {' '.join(reason.split())}" for question, reason in found.dropped]
    lines.append("analyses:" + ("" if found.analyses else " []"))
    for analysis in found.analyses:
        planned = analysis["id"] in found.planned
        lines.append(f"  - id: {analysis['id']}")
        lines.append(f"    question: {_yaml(analysis['question'])}")
        lines.append(f"    outcome: {_yaml(analysis['outcome'])}")
        for key in ("compare", "relate", "change", "by"):
            if analysis.get(key):
                lines.append(f"    {key}: {_yaml(analysis[key])}")
        lines.append("    covariates: []")
        lines.append(f"    planned: {'true' if planned else 'false'}")
        lines.append(f"    family: {'primary' if planned else 'exploratory'}")
    lines += [
        "",
        "# Figures to draw. Without this list ADA draws one per analysis.",
        "# type: groups | paired | longitudinal | scatter | missingness | diagnostics | panel",
        "figures: []",
        "",
        "output:",
        "  dir: paper-assets",
        f"  journal: {found.journal}   # " + " | ".join(JOURNALS),
        "  tables: [csv, docx, tex]",
        "  figures: [pdf, svg, png]   # add tiff or eps if the journal asks for them",
        "",
        "sharing:",
        f"  data: {found.sharing}   # " + " | ".join(SHARING),
        f"  withhold: {_yaml(found.withhold)}   # columns that never leave this computer",
        "  row_level: true   # whether files listing individual rows may be shared",
        "",
        "# Your own Python files whose @research.table and @research.figure functions join the assets.",
        "custom: []",
        "",
    ]
    return "\n".join(lines)


def init(
    data: str | Path,
    *,
    study_path: str | Path = "study.yml",
    sheet: str | None = None,
    interactive: bool = True,
    ask: Callable[[str], str] = input,
    say: Callable[[str], None] = print,
    overwrite: bool = False,
) -> Path:
    """Write ``study.yml`` for ``data``, asking each question unless ``interactive`` is False."""
    data_path = Path(data)
    if not data_path.is_file():
        raise FileNotFoundError(f"No such file: {data_path}")
    target = Path(study_path)
    if target.is_dir():
        target = target / "study.yml"
    if target.exists() and not overwrite:
        raise FileExistsError(f"{target} already exists. Edit it, or pass --force to start again.")
    found = guess(data_path, target.resolve().parent, sheet)
    if not interactive:
        for note in found.notes:
            say(f"{note} It is marked unconfirmed in study.yml until you check it.")
    found = interview(found, ask=ask, say=say, interactive=interactive)
    target.write_text(render(found), encoding="utf-8")
    return target
