"""Read a folder of tables: what each file holds, how the files join, and what to ask before joining.

    import ada
    folder = ada.scan("data/")
    folder.files       # one FileProfile per table
    folder.links       # proposed joins, with how well the keys match both ways
    folder.questions   # what a person needs to confirm, grouped by topic
    folder.to_dict()   # all of it as JSON-safe data

A link is proposed from the values, not the names alone: a column whose values
are found among the values of a key in another file (one value per row there).
Names decide which columns are tried and how sure the link is, and whole
numbers, which overlap by chance, link only under matching names. Every link
reports its coverage both ways, its cardinality and the rows it leaves
unmatched, and a link is marked ``needs_confirmation`` when its names differ,
its keys don't all match, or the key side repeats. Nothing is joined: a scan
proposes, and a person confirms.

Answers a person gives are kept in a small YAML file (``answers``) and read
back on the next scan, so a confirmed link or row meaning is asked once.
"""

from __future__ import annotations

import itertools
import re
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype

from .file_io import SUPPORTED_SUFFIXES
from .formatting import counted, normalized_name
from .schema import IDENTIFIER_NAME_WORDS, is_identifier_name, is_year_name
from .sources import Source

#: Version of the ``FolderScan.to_dict`` layout. Bumped when a key changes meaning.
SCAN_SCHEMA_VERSION = 1

# A key needs this many distinct values before values alone can propose a link.
MIN_KEY_VALUES = 20
# Share of a column's distinct values that must be found in the other file's key.
MIN_COVERAGE = 0.9
# With matching names, a lower share still proposes a link, flagged to confirm.
MIN_NAMED_COVERAGE = 0.5
# Columns combined into a composite key, at most.
MAX_KEY_PARTS = 3
# Columns named like a row index ("rownames", "Unnamed: 0") are never keys.
_ROW_INDEX = re.compile(r"^(?:unnamed: ?\d+|rownames|index|row|row_?id|x)$", re.IGNORECASE)
# Columns of a codebook: what each column is called, what it means, its unit.
CODEBOOK_NAME_WORDS = frozenset({"column", "variable", "field", "name", "indicator", "code"})
CODEBOOK_TEXT_WORDS = frozenset({"description", "definition", "label", "title", "meaning"})
CODEBOOK_UNIT_WORDS = frozenset({"unit", "units"})


@dataclass(frozen=True)
class FileProfile:
    """One table in the folder: its size, how ADA reads it, and why it could not be read."""

    name: str
    source: Source | None = None
    rows: int = 0
    columns: tuple[str, ...] = ()
    row_noun: str | None = None
    identifier: str | None = None
    unit: str | None = None
    measure: str | None = None
    date: str | None = None
    grouping: str | None = None
    #: Units a codebook in the folder gives for this file's columns.
    units: tuple[tuple[str, str], ...] = ()
    #: The file this one describes, when it is a codebook.
    describes: str | None = None
    #: The fewest columns that tell every row apart, when up to three do.
    row_key: tuple[str, ...] | None = None
    notes: tuple[str, ...] = ()
    problem: str | None = None

    @property
    def stem(self) -> str:
        return Path(self.name).stem

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "source": None if self.source is None else self.source.to_dict(),
            "rows": self.rows,
            "columns": list(self.columns),
            "row_noun": self.row_noun,
            "identifier": self.identifier,
            "unit": self.unit,
            "measure": self.measure,
            "date": self.date,
            "grouping": self.grouping,
            "units": {column: unit for column, unit in self.units},
            "describes": self.describes,
            "row_key": None if self.row_key is None else list(self.row_key),
            "notes": list(self.notes),
            "problem": self.problem,
        }


@dataclass(frozen=True)
class Link:
    """A proposed join from columns of one file to a key of another, with how well they match."""

    from_file: str
    from_columns: tuple[str, ...]
    to_file: str
    to_columns: tuple[str, ...]
    kind: str
    #: Share of the from side's distinct key values found on the to side, and the other way round.
    coverage: float
    coverage_back: float
    unmatched_rows: int
    unmatched_values: int
    #: Most rows the to side holds for one key value: above 1, a join repeats the from rows.
    fan_out: int
    named: bool
    needs_confirmation: bool
    reasons: tuple[str, ...] = ()
    confirmed: bool | None = None

    def label(self, side: str) -> str:
        if side == "from":
            file, columns = self.from_file, self.from_columns
        else:
            file, columns = self.to_file, self.to_columns
        stem = Path(file).stem
        return f"{stem}.{columns[0]}" if len(columns) == 1 else f"{stem}.({', '.join(columns)})"

    @property
    def key(self) -> str:
        return f"link:{self.label('from')}->{self.label('to')}"

    def to_dict(self) -> dict:
        return {
            "from": self.label("from"),
            "to": self.label("to"),
            "from_file": self.from_file,
            "from_columns": list(self.from_columns),
            "to_file": self.to_file,
            "to_columns": list(self.to_columns),
            "kind": self.kind,
            "coverage": self.coverage,
            "coverage_back": self.coverage_back,
            "unmatched_rows": self.unmatched_rows,
            "unmatched_values": self.unmatched_values,
            "fan_out": self.fan_out,
            "named": self.named,
            "needs_confirmation": self.needs_confirmation,
            "reasons": list(self.reasons),
            "confirmed": self.confirmed,
        }


@dataclass(frozen=True)
class Question:
    """Something a person needs to confirm, with ADA's proposed answer and any answer already given."""

    topic: str
    key: str
    text: str
    proposed: str | None = None
    answer: str | None = None

    def to_dict(self) -> dict:
        return {"topic": self.topic, "key": self.key, "text": self.text, "proposed": self.proposed,
                "answer": self.answer}


# Topics in the order a person is asked about them.
TOPICS = ("rows", "units", "repeated", "links", "intent")


@dataclass(frozen=True)
class FolderScan:
    """The result of ``scan``: the files, the proposed links and the questions, by topic."""

    folder: str
    files: tuple[FileProfile, ...]
    links: tuple[Link, ...]
    questions: tuple[Question, ...]
    warnings: tuple[str, ...] = field(default=())

    def file(self, name: str) -> FileProfile | None:
        return next((profile for profile in self.files if profile.name == name), None)

    def open_questions(self) -> tuple[Question, ...]:
        """The questions no answer covers yet."""
        return tuple(question for question in self.questions if question.answer is None)

    def to_dict(self) -> dict:
        return {
            "schema_version": SCAN_SCHEMA_VERSION,
            "folder": self.folder,
            "files": [profile.to_dict() for profile in self.files],
            "links": [link.to_dict() for link in self.links],
            "questions": {
                topic: [question.to_dict() for question in self.questions if question.topic == topic]
                for topic in TOPICS
            },
            "warnings": list(self.warnings),
        }

    def answers_template(self) -> str:
        """A YAML file of every question with ADA's proposed answer, for a person to edit and keep."""
        lines = ["# Answers to ADA's questions about this folder. Edit the answers, keep the keys,",
                 "# and pass the file back with `ada scan FOLDER --answers FILE`.", ""]
        for topic in TOPICS:
            asked = [question for question in self.questions if question.topic == topic]
            if not asked:
                continue
            lines.append(f"# {topic}")
            for question in asked:
                lines.append(f"# {question.text}")
                value = question.answer if question.answer is not None else (question.proposed or "")
                lines.append(f"{_yaml_key(question.key)}: {_yaml_value(value)}")
            lines.append("")
        return "\n".join(lines)

    def __str__(self) -> str:
        files, links = counted(len(self.files), "file"), counted(len(self.links), "proposed link")
        out = [f"{self.folder}: {files}, {links}"]
        for profile in self.files:
            if profile.problem:
                out.append(f"  {profile.name}: not read ({profile.problem})")
                continue
            if profile.row_noun:
                what = f"one row per {singular(profile.row_noun)}"
            elif profile.row_key:
                what = f"one row per {', '.join(profile.row_key)}"
            else:
                what = "rows of unknown meaning"
            extra = f", describes {profile.describes}" if profile.describes else ""
            size = f"{counted(profile.rows, 'row')} x {counted(len(profile.columns), 'column')}"
            out.append(f"  {profile.name}: {size}, {what}{extra}")
        for link in self.links:
            flag = " (confirm)" if link.needs_confirmation and link.confirmed is None else ""
            unmatched = counted(link.unmatched_rows, "row")
            out.append(f"  link {link.label('from')} -> {link.label('to')}: {link.kind}, "
                       f"{link.coverage:.1%} of keys found, {unmatched} unmatched{flag}")
        open_ones = self.open_questions()
        if open_ones:
            out.append(f"  {counted(len(open_ones), 'question')} to answer:")
            out.extend(f"    [{question.topic}] {question.text}" for question in open_ones)
        out.extend(f"  warning: {warning}" for warning in self.warnings)
        return "\n".join(out)


def _yaml_key(key: str) -> str:
    return f'"{key}"'


def _yaml_value(value: str) -> str:
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def scan(folder: str | Path, *, answers: str | Path | dict | None = None) -> FolderScan:
    """Profile every table in ``folder``, propose links between them and list the questions to confirm.

    ``answers`` (a YAML file or a dict of question key to answer) records what a
    person already confirmed; a link answered "no" is dropped, "yes" confirms it.
    """
    root = Path(folder)
    if not root.is_dir():
        raise NotADirectoryError(f"{folder} is not a folder.")
    given = _load_answers(answers)
    paths = sorted(
        path for path in root.iterdir()
        if path.is_file() and not path.name.startswith(".") and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    profiles: list[FileProfile] = []
    frames: dict[str, pd.DataFrame] = {}
    for path in paths:
        profile, frame = _profile(path)
        profiles.append(profile)
        if frame is not None:
            frames[path.name] = frame
    profiles = _read_codebooks(profiles, frames)
    keys = {
        profile.name: _Keys(profile, frames[profile.name])
        for profile in profiles if profile.name in frames and not profile.describes
    }
    for held in keys.values():
        held.profile = replace(held.profile, row_key=held.row_key())
    profiles = [keys[profile.name].profile if profile.name in keys else profile for profile in profiles]
    links = _links(keys)
    links = [_answered_link(link, given) for link in links]
    links = [link for link in links if link.confirmed is not False]
    warnings = _warnings(profiles, links, frames)
    questions = _questions(profiles, links, frames, given)
    return FolderScan(str(folder), tuple(profiles), tuple(links), tuple(questions), tuple(warnings))


def _load_answers(answers: str | Path | dict | None) -> dict[str, str]:
    if answers is None:
        return {}
    if isinstance(answers, dict):
        return {str(key): str(value) for key, value in answers.items()}
    import yaml  # noqa: PLC0415 - only needed when answers are read

    path = Path(answers)
    if not path.is_file():
        return {}
    loaded = yaml.safe_load(path.read_text()) or {}
    return {str(key): "" if value is None else str(value) for key, value in loaded.items()}


def _profile(path: Path) -> tuple[FileProfile, pd.DataFrame | None]:
    from .api import AdaError, analyze  # noqa: PLC0415
    from .compare import entity_column  # noqa: PLC0415

    try:
        analysis = analyze(path, row_limit=sys.maxsize)
    except AdaError as error:
        return FileProfile(name=path.name, problem=str(error)), None
    frame, roles = analysis.frame, analysis.roles
    unit = entity_column(frame, roles)
    return FileProfile(
        name=path.name,
        source=analysis.source,
        rows=len(frame),
        columns=tuple(str(column) for column in frame.columns),
        row_noun=roles.row_noun,
        identifier=roles.identifier,
        unit=unit,
        measure=roles.measure,
        date=roles.date,
        grouping=roles.dimension,
        notes=tuple(roles.notes[:5]),
    ), frame


def _words(name: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", normalized_name(name)))


def _codebook_columns(frame: pd.DataFrame) -> tuple[str, str | None, str | None] | None:
    """The name, description and unit columns of a codebook, or None for an ordinary table."""
    named = {column: _words(str(column)) for column in frame.columns}
    name = next((column for column, words in named.items() if words & CODEBOOK_NAME_WORDS
                 and not is_numeric_dtype(frame[column])), None)
    text = next((column for column, words in named.items() if words & CODEBOOK_TEXT_WORDS), None)
    unit = next((column for column, words in named.items() if words & CODEBOOK_UNIT_WORDS), None)
    return (name, text, unit) if name and (text or unit) else None


def _read_codebooks(profiles: list[FileProfile], frames: dict[str, pd.DataFrame]) -> list[FileProfile]:
    """Mark files that describe another file's columns, and give that file the units they state.

    A codebook has a column of names and a column of descriptions or units,
    and half or more of another file's columns appear among its names.
    """
    found = dict((profile.name, profile) for profile in profiles)
    for profile in profiles:
        frame = frames.get(profile.name)
        if frame is None:
            continue
        columns = _codebook_columns(frame)
        if columns is None:
            continue
        name, _, unit = columns
        listed = set(frame[name].dropna().astype(str))
        target = max(
            (other for other in profiles if other.name != profile.name and other.columns),
            key=lambda other: len(listed & set(other.columns)) / len(other.columns),
            default=None,
        )
        if target is None or len(listed & set(target.columns)) < 0.5 * len(target.columns):
            continue
        found[profile.name] = replace(found[profile.name], describes=target.name)
        if unit:
            units = frame[[name, unit]].dropna()
            stated = tuple(
                (str(column), str(value)) for column, value in units.itertuples(index=False)
                if str(column) in target.columns and str(value).strip()
            )
            found[target.name] = replace(found[target.name], units=stated)
    return [found[profile.name] for profile in profiles]


def _key_values(series: pd.Series) -> pd.Series | None:
    """A column's values as text keys, or None when it can't be a key (decimals, flags, a single value)."""
    values = series.dropna()
    if values.empty or is_bool_dtype(values) or values.nunique() < 2:
        return None
    if is_datetime64_any_dtype(values):
        return values.dt.strftime("%Y-%m-%d %H:%M:%S")
    if is_numeric_dtype(values):
        numbers = values.to_numpy(dtype=float)
        if not np.all(numbers == np.floor(numbers)):
            return None
        return pd.Series(numbers.astype(np.int64).astype(str), index=values.index)
    return values.astype(str).str.strip()


def _keyish(name: str, series: pd.Series, profile: FileProfile) -> bool:
    """Whether a column could be a key: text, a whole number named like an id or a year, a date."""
    if _ROW_INDEX.match(str(name)):
        return False
    if is_numeric_dtype(series) and not is_bool_dtype(series):
        return (is_identifier_name(str(name)) or is_year_name(str(name)) or str(name) == profile.identifier
                or str(name).lower().endswith("id") or bool(_words(str(name)) & IDENTIFIER_NAME_WORDS))
    return True


class _Keys:
    """One table's candidate key columns, each read once as text keys with its distinct values and counts."""

    def __init__(self, profile: FileProfile, frame: pd.DataFrame):
        self.profile, self.frame = profile, frame
        self.columns = [column for column in frame.columns if _keyish(column, frame[column], profile)]
        self._cache: dict[tuple[str, ...], tuple[pd.Series, set, pd.Series] | None] = {}

    def get(self, columns: tuple[str, ...]) -> tuple[pd.Series, set, pd.Series] | None:
        """The key series, its distinct values and how often each occurs, for one or more columns."""
        if columns not in self._cache:
            parts = [_key_values(self.frame[column]) for column in columns]
            if any(part is None for part in parts):
                self._cache[columns] = None
            else:
                joined = pd.concat(parts, axis=1).dropna()
                series = joined.agg("\x1f".join, axis=1) if len(columns) > 1 else joined.iloc[:, 0]
                self._cache[columns] = (series, set(series.unique()), series.value_counts())
        return self._cache[columns]

    def unique(self, columns: tuple[str, ...], *, every_row: bool = False) -> bool:
        """Whether the columns tell the rows that have them all apart, checked cheaply first.

        With ``every_row`` no row may lack one of them, as a row key needs; a join
        key only has to tell apart the rows that can join.
        """
        block = self.frame[list(columns)]
        complete = block.dropna()
        if every_row and len(complete) < len(block):
            return False
        distinct = [complete[column].nunique() for column in columns]
        if np.prod(np.asarray(distinct, dtype=float)) < len(complete):
            return False
        return not complete.duplicated().any()

    def row_key(self) -> tuple[str, ...] | None:
        """The fewest key columns (up to MAX_KEY_PARTS) that tell every row apart, or None."""
        for size in range(1, MAX_KEY_PARTS + 1):
            for columns in itertools.combinations(self.columns, size):
                if self.unique(columns, every_row=True):
                    return columns
        return None


def _names_match(first: str, second: str, to_profile: FileProfile) -> bool:
    """Whether two column names say they hold the same key.

    The same name ("playerID"), the same words but one ("customer_zip_code_prefix"
    and "geolocation_zip_code_prefix"), or a bare "id" in the file the other
    column is named for ("customer_id" beside customers.csv's "id").
    """
    if normalized_name(first) == normalized_name(second):
        return True
    a, b = _words(first), _words(second)
    if len(a & b) >= 2 and len(a ^ b) <= 2:
        return True
    marks = {"id", "key", "code"}
    return b <= marks and bool(a & _words(to_profile.stem)) and bool(a & marks)


def _plain_numbers(values: set) -> bool:
    """Whether every key value is a whole number in digits, as row numbers, counters and positions are.

    Such values overlap by chance (grid positions 1 to 34 sit inside any id
    column counting past 34), so they link only under matching names.
    """
    return all(value.isdigit() for value in values)


def _key_named(columns: tuple[str, ...]) -> bool:
    """Whether a name holds a key word (id, code, zip, number), unlike a shared label such as nationality."""
    return any(_words(str(column)) & IDENTIFIER_NAME_WORDS for column in columns)


def _link(source: _Keys, from_columns: tuple[str, ...], target: _Keys, to_columns: tuple[str, ...],
          named: bool) -> Link | None:
    """Measure one candidate link; None when too little of it matches."""
    left, right = source.get(from_columns), target.get(to_columns)
    if left is None or right is None:
        return None
    (left_keys, left_values, left_counts), (_, right_values, right_counts) = left, right
    if not named and (len(right_values) < MIN_KEY_VALUES or _plain_numbers(left_values)):
        return None
    shared = left_values & right_values
    coverage = len(shared) / len(left_values)
    if coverage < (MIN_NAMED_COVERAGE if named else MIN_COVERAGE):
        return None
    right_unique = int(right_counts.max()) == 1
    if not right_unique and (not named or len(right_values) < MIN_KEY_VALUES or not _key_named(from_columns)):
        return None
    left_unique = int(left_counts.max()) == 1
    kind = {(True, True): "one-to-one", (False, True): "many-to-one", (True, False): "one-to-many",
            (False, False): "many-to-many"}[(left_unique, right_unique)]
    unmatched = ~left_keys.isin(right_values)
    from_profile, to_profile = source.profile, target.profile
    reasons = []
    if not named:
        reasons.append("the names differ; the link rests on the values alone")
    if coverage < 1:
        missed = len(left_values) - len(shared)
        reasons.append(f"{missed:,} {_labelled(from_profile, from_columns)} "
                       f"{'value' if missed == 1 else 'values'} ({counted(int(unmatched.sum()), 'row')}) "
                       f"{'has' if missed == 1 else 'have'} no match in {to_profile.name}")
    fan_out = int(right_counts.max())
    if not right_unique:
        reasons.append(f"{_labelled(to_profile, to_columns)} repeats (up to {fan_out:,} rows for one value), "
                       "so joining on it repeats rows")
    return Link(
        from_file=from_profile.name, from_columns=from_columns,
        to_file=to_profile.name, to_columns=to_columns,
        kind=kind, coverage=coverage, coverage_back=len(shared) / len(right_values),
        unmatched_rows=int(unmatched.sum()), unmatched_values=len(left_values) - len(shared),
        fan_out=fan_out, named=named, needs_confirmation=bool(reasons), reasons=tuple(reasons),
    )


def _labelled(profile: FileProfile, columns: tuple[str, ...]) -> str:
    return f"{profile.stem}.{columns[0]}" if len(columns) == 1 else f"{profile.stem}.({', '.join(columns)})"


KEYED = ("many-to-one", "one-to-one")


def _links(keys: dict[str, _Keys]) -> list[Link]:
    """Every link between two tables, the best one for each from-column, found from the values."""
    found: list[Link] = []
    for source, target in itertools.permutations(keys.values(), 2):
        single = [
            link
            for a in source.columns for b in target.columns
            if (link := _link(source, (a,), target, (b,), _names_match(str(a), str(b), target.profile)))
        ]
        found.extend(_best_per_column(single))
        if not any(link.kind in KEYED for link in single):
            composite = _composite(source, target)
            if composite is not None:
                found.append(composite)
    return _without_redundant(_one_direction(found))


def _composite(source: _Keys, target: _Keys) -> Link | None:
    """The smallest set of columns named alike in both files that is a key of the second file."""
    shared = [column for column in source.columns if column in target.columns]
    for size in range(2, MAX_KEY_PARTS + 1):
        for columns in itertools.combinations(shared, size):
            if not target.unique(columns):
                continue
            link = _link(source, columns, target, columns, True)
            if link is not None and link.kind in KEYED:
                return link
    return None


def _best_per_column(links: list[Link]) -> list[Link]:
    """For each from-column, the link with the best match: a key on the other side, named alike, covered."""
    best: dict[tuple, Link] = {}
    for link in links:
        rank = (link.kind in KEYED, link.named, link.coverage)
        held = best.get(link.from_columns)
        if held is None or rank > (held.kind in KEYED, held.named, held.coverage):
            best[link.from_columns] = link
    return list(best.values())


def _one_direction(links: list[Link]) -> list[Link]:
    """Drop the reverse of a link already kept: a one-to-many is the many-to-one read backwards."""
    kept: list[Link] = []
    seen: set[frozenset] = set()
    order = {"many-to-one": 0, "one-to-one": 1, "many-to-many": 2, "one-to-many": 3}
    for link in sorted(links, key=lambda item: (order[item.kind], item.from_file, item.from_columns)):
        pair = frozenset({(link.from_file, link.from_columns), (link.to_file, link.to_columns)})
        if pair in seen or link.kind == "one-to-many":
            continue
        seen.add(pair)
        kept.append(link)
    return kept


def _without_redundant(links: list[Link]) -> list[Link]:
    """Drop a many-to-many link whose columns already belong to a key link of either file.

    Batting.teamID beside Teams.teamID repeats on both sides because the key is
    (yearID, teamID); two tables that each link to People by playerID need no
    link to each other.
    """
    keyed: set[tuple[str, str]] = set()
    for link in links:
        if link.kind in KEYED:
            keyed |= {(link.from_file, column) for column in link.from_columns}
            keyed |= {(link.to_file, column) for column in link.to_columns}
    return [
        link for link in links
        if link.kind in KEYED
        or not ({(link.from_file, column) for column in link.from_columns} & keyed
                or {(link.to_file, column) for column in link.to_columns} & keyed)
    ]


def _answered_link(link: Link, given: dict[str, str]) -> Link:
    answer = given.get(link.key)
    if answer is None:
        return link
    word = answer.strip().lower()
    if word in ("yes", "y", "true", "confirmed"):
        return replace(link, confirmed=True)
    if word in ("no", "n", "false", "rejected"):
        return replace(link, confirmed=False)
    return link


def _warnings(profiles: list[FileProfile], links: list[Link], frames: dict[str, pd.DataFrame]) -> list[str]:
    """Things to know before joining: tables of the same length with nothing to join them by."""
    warnings = []
    linked = {link.from_file for link in links} | {link.to_file for link in links}
    tables = [profile for profile in profiles if profile.name in frames and not profile.describes]
    for first, second in itertools.combinations(tables, 2):
        unlinked = first.name not in linked and second.name not in linked
        if first.rows == second.rows and first.rows and unlinked:
            warnings.append(
                f"{first.name} and {second.name} have the same number of rows ({first.rows:,}) and no shared "
                "key: they may describe the same records, but joining them by row order is only safe if a "
                "person confirms the rows are in the same order."
            )
    return warnings


# Plurals whose singular is not the word less its "s".
_SINGULARS = {"people": "person", "children": "child", "mice": "mouse", "men": "man", "women": "woman",
              "geese": "goose", "teeth": "tooth", "feet": "foot"}


def singular(noun: str) -> str:
    """"person" for "people", "salary" for "salaries", "status" for "statuses", "flight" for "flights"."""
    if noun in _SINGULARS:
        return _SINGULARS[noun]
    if noun.endswith("ies") and len(noun) > 4:
        return noun[:-3] + "y"
    if re.search(r"(?:ss|x|ch|sh|us)es$", noun):
        return noun[:-2]
    return noun[:-1] if noun.endswith("s") and not noun.endswith("ss") else noun


def _questions(profiles: list[FileProfile], links: list[Link], frames: dict[str, pd.DataFrame],
               given: dict[str, str]) -> list[Question]:
    """The questions, grouped by topic, each with ADA's proposed answer and any answer given."""
    asked: list[Question] = []

    def ask(topic: str, key: str, text: str, proposed: str | None) -> None:
        asked.append(Question(topic, key, text, proposed, given.get(key)))

    for profile in profiles:
        if profile.problem or profile.describes:
            continue
        frame = frames[profile.name]
        if profile.row_noun:
            proposed = f"one {singular(profile.row_noun)} (from the file's name)"
        elif profile.row_key:
            proposed = f"one {' and '.join(profile.row_key)} (no two rows share them)"
        else:
            proposed = None
        ask("rows", f"rows:{profile.name}", f"What does one row of {profile.name} stand for?", proposed)
        measure = profile.measure
        if measure and _needs_unit(measure, frame[measure], dict(profile.units)):
            ask("units", f"units:{profile.name}:{measure}",
                f"What unit is {measure} in {profile.name} recorded in?", None)
        if profile.unit and profile.unit in frame.columns:
            per_unit = frame[profile.unit].value_counts()
            # A key most of whose values appear once has duplicates, not repeated measurements.
            if per_unit.median() >= 2 and len(per_unit) >= MIN_KEY_VALUES:
                ask("repeated", f"repeated:{profile.name}:{profile.unit}",
                    f"{profile.unit} has several rows each in {profile.name} "
                    f"(up to {int(per_unit.max()):,}). "
                    f"Are they repeated measurements of the same {profile.unit}?",
                    f"yes: each {profile.unit} counts once in tests and correlations")
    for link in links:
        if link.needs_confirmation and link.confirmed is None:
            ask("links", link.key,
                f"Join {link.label('from')} to {link.label('to')} ({link.kind})? "
                + "; ".join(link.reasons) + ".",
                "yes" if link.coverage >= MIN_COVERAGE and link.kind in KEYED else None)
    ask("intent", "intent", "What should the analysis find out, and about which rows?", None)
    return asked


def _needs_unit(column: str, values: pd.Series, stated: dict[str, str]) -> bool:
    """Whether to ask for a measure's unit: a reading with decimals whose unit nothing in the folder gives.

    Whole numbers are mostly counts, which have no unit to ask about.
    """
    if column in stated or _unit_named(column) or _ROW_INDEX.match(str(column)):
        return False
    if not is_numeric_dtype(values):
        return False
    numbers = values.dropna().to_numpy(dtype=float)
    return bool(numbers.size) and not np.all(numbers == np.floor(numbers))


def _unit_named(column: str) -> bool:
    """Whether the name states a unit: "body_mass_g", "Rainfall (mm)", "Delta 13 C (o/oo)", "Share %"."""
    from .claims import column_unit  # noqa: PLC0415

    bracketed = re.search(r"[(\[][^)\]]+[)\]]\s*$", str(column)) is not None
    return column_unit(column) is not None or "%" in str(column) or bracketed


__all__ = ["SCAN_SCHEMA_VERSION", "FileProfile", "FolderScan", "Link", "Question", "scan", "singular"]
