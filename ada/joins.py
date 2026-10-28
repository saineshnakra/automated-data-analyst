"""Answer a question about a folder: from the file it fits, or from two files joined along a link.

A question or claim about a folder is first tried on each file. When no file
holds every column and value it names (a player's name in one file, the
home runs in another), two files are joined along a link ``ada.folder``
proposes and it is tried on the joined table.

Only links where each row of the first file finds at most one row of the
second (many-to-one, one-to-one) are joined, so a join never repeats rows.
It is a left join: every row of the first file stays, and a row whose key has
no match gets empty columns from the second. When the first file has a
date or year of its own, the second file's dates and date parts (a birth
date, a birth year) are left out, so a year in the question still means the
first file's; a first file with none takes them (a race's date for a result). Which links may be used is the
decision ``join_links``:

- ``sure`` (the default): links that need no confirmation, and links the
  answers file confirms
- ``all``: every many-to-one or one-to-one link the scan proposes
- ``none``: no joins
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype

from .decisions import Decision, checked_choices, decide
from .file_io import SUPPORTED_SUFFIXES
from .folder import KEYED, FolderScan, Link, load_answers, scan
from .formatting import counted, group_label, list_names, normalized_name
from .schema import finer_part_name, is_month_name, is_year_name
from .sources import Source


@dataclass(frozen=True)
class JoinedTable:
    """Files of a folder joined along links: every row of the first file, with the others' columns."""

    links: tuple[Link, ...]
    frame: pd.DataFrame

    @property
    def from_file(self) -> str:
        return self.links[0].from_file

    @property
    def name(self) -> str:
        return f"{self.from_file} joined to {list_names([link.to_file for link in self.links], limit=6)}"

    def describe(self) -> str:
        parts = []
        for link in self.links:
            unmatched = counted(link.unmatched_rows, "row")
            parts.append(f"{link.to_file} on {link.label('from')} = {link.label('to')} ({link.kind}; "
                         f"{link.coverage:.1%} of keys found, {unmatched} unmatched)")
        return f"Joined {self.from_file} to {'; and '.join(parts)}."

    def source(self, folder: Path) -> Source:
        return Source(name=self.name, path=str(folder), rows=len(self.frame), columns=self.frame.shape[1])


def files_in(folder: Path) -> list[Path]:
    """The readable tables of ``folder``, in name order."""
    return sorted(
        item for item in folder.iterdir()
        if item.is_file() and not item.name.startswith(".") and item.suffix.lower() in SUPPORTED_SUFFIXES
    )


def usable(link: Link, policy: str) -> bool:
    """Whether ``join_links`` set to ``policy`` lets a join run along ``link``."""
    if link.kind not in KEYED or policy == "none":
        return False
    return policy == "all" or link.confirmed is True or not link.needs_confirmation


def join_decision(choices: Mapping[str, str]) -> Decision:
    return decide(
        "join_links", choices,
        "Join two files along the links ADA is sure of, along every link it proposes, or not at all?",
        "A link that needs confirmation (names that differ, keys that do not all match) may join rows "
        "that do not belong together.",
    )


def _join_keys(series: pd.Series) -> pd.Series:
    """A key column as text, as the scan compares keys: 7.0 and 7 are both "7"."""
    if is_datetime64_any_dtype(series):
        return series.dt.strftime("%Y-%m-%d %H:%M:%S")
    if is_numeric_dtype(series) and not is_bool_dtype(series):
        numbers = series.to_numpy(dtype=float)
        whole = np.isfinite(numbers) & (numbers == np.floor(numbers))
        integers = np.nan_to_num(numbers).astype(np.int64).astype(str)
        text = pd.Series(np.where(whole, integers, series.astype(str)), index=series.index)
        return text.where(series.notna())
    return series.astype("string").str.strip().astype(object).where(series.notna())


def _dated(name: str, values: pd.Series) -> bool:
    """Whether a column holds dates or a part of them: birthDate, birthYear, deathDay.

    Times read onto a single day ("1:27.452", a fastest lap) are durations or times of day, not dates.
    """
    if is_datetime64_any_dtype(values):
        return values.dropna().dt.normalize().nunique() > 1
    return is_year_name(name) or is_month_name(name) or finer_part_name(name) is not None


# Words that mark a person's birth or death date: it describes the person, not the row's event.
LIFE_WORDS = frozenset({"birth", "born", "dob", "death", "died", "deceased"})


def _life_date(name: str) -> bool:
    return bool(LIFE_WORDS & set(normalized_name(name).split()))


def _join(left: pd.DataFrame, right: pd.DataFrame, link: Link, names: dict[str, str] | None = None
          ) -> tuple[pd.DataFrame, dict[str, str]] | None:
    """``left`` with ``right``'s other columns, matched on the link's keys; None if a row would repeat.

    ``names`` gives the names the link's first-side columns carry in
    ``left`` when a join renamed them. Returns the joined table and the new
    names of ``right``'s columns that clashed with ``left``'s.
    """
    names = names or {}
    keys = [f"__key{index}" for index in range(len(link.to_columns))]
    right = right.assign(**{key: _join_keys(right[column]) for key, column in zip(keys, link.to_columns,
                                                                                    strict=True)})
    right = right.dropna(subset=keys)
    if right.duplicated(subset=keys).any():
        return None
    stem = Path(link.to_file).stem
    # A second file's dates are left out when the first has its own, so "in 2001" stays the first file's
    # year (a batting season, not a birth year); a first file with none takes them (a result's race date).
    # A birth or death date is never taken: "in 2019" asks about the rows' events, not about people's lives.
    dated = any(_dated(str(column), left[column]) and not _life_date(str(column)) for column in left.columns)
    extra = [
        column for column in right.columns
        if column not in keys and column not in link.to_columns
        and not ((dated or _life_date(str(column))) and _dated(str(column), right[column]))
    ]
    clash = {column: f"{column} ({stem})" for column in extra if column in left.columns}
    right = right[[*keys, *extra]].rename(columns=clash)
    left_keys = [names.get(column, column) for column in link.from_columns]
    if any(column not in left.columns for column in left_keys):
        return None
    left = left.assign(**{key: _join_keys(left[column]) for key, column in zip(keys, left_keys, strict=True)})
    joined = left.merge(right, how="left", on=keys, validate="many_to_one").drop(columns=keys)
    return (joined, clash) if len(joined) == len(left) else None


def joined_tables(folder: Path, found: FolderScan, policy: str, wide: bool = False) -> list[JoinedTable]:
    """Files joined along the links ``policy`` allows, in the scan's order.

    By default each file is joined to one other along one link. With
    ``wide``, each file that links to two or more others is joined to all of
    them, and to the files those link to in turn (results to races, and races
    to circuits), so a question can use columns from three or more files.
    """
    from .api import AdaError, analyze_file  # noqa: PLC0415

    frames: dict[str, pd.DataFrame | None] = {}

    def frame(name: str) -> pd.DataFrame | None:
        if name not in frames:
            try:
                frames[name] = analyze_file(folder / name).frame
            except AdaError:
                frames[name] = None
        return frames[name]

    links = [link for link in found.links if usable(link, policy)]
    tables = []
    if not wide:
        for link in links:
            left, right = frame(link.from_file), frame(link.to_file)
            if left is None or right is None:
                continue
            joined = _join(left, right, link)
            if joined is not None:
                tables.append(JoinedTable((link,), joined[0]))
        return tables
    outgoing: dict[str, list[Link]] = {}
    for link in links:
        outgoing.setdefault(link.from_file, []).append(link)
    for start, first_links in outgoing.items():
        table = frame(start)
        if table is None:
            continue
        used: list[Link] = []
        names: dict[str, dict[str, str]] = {start: {}}
        for hop in (first_links, None):
            # The second hop follows the links of every file the first hop brought in.
            candidates = hop if hop is not None else [link for file in list(names) if file != start
                                                      for link in outgoing.get(file, [])]
            for link in candidates:
                right = frame(link.to_file)
                if link.to_file in names or right is None:
                    continue
                joined = _join(table, right, link, names[link.from_file])
                if joined is None:
                    continue
                table, clash = joined
                names[link.to_file] = clash
                used.append(link)
        if len(used) >= 2:
            tables.append(JoinedTable(tuple(used), table))
    return tables


def folder_choices(
    answers: Mapping[str, str], name: str | None, choices: Mapping[str, str]
) -> dict[str, str]:
    """The choices for one file: the answers file's, for every file and for this one, then ``choices``."""
    given: dict[str, str] = {}
    specific: dict[str, str] = {}
    for question, answer in answers.items():
        parts = str(question).split(":")
        if parts[0] != "decide" or not str(answer).strip() or len(parts) < 2:
            continue
        if len(parts) == 2:
            given[parts[1]] = str(answer).strip()
        elif ":".join(parts[1:-1]) == name:
            specific[parts[-1]] = str(answer).strip()
    return checked_choices({**given, **specific, **choices})


def ask_folder(
    folder: str | Path,
    question: str,
    *,
    decisions: Mapping[str, str] | None = None,
    answers: str | Path | Mapping[str, str] | None = None,
):
    """Answer ``question`` about the tables in ``folder``.

    The question is asked of each file. One file that answers it gives the
    answer, and its first assumption says so; several that give different
    numbers give a ``Refusal`` naming them. When none answers it, two files
    joined along a link ``join_links`` allows are tried the same way, and the
    answer lists that decision and says which join it used.
    """
    from .api import AdaError, Refusal, analyze, analyze_file  # noqa: PLC0415

    root = Path(folder)
    given = load_answers(answers)
    choices = checked_choices(decisions)
    files = files_in(root)
    if not files:
        return Refusal("unrepresentable_question", f"{folder} holds no table ADA can read.")
    fitted = []
    for item in files:
        try:
            analysis = analyze_file(item)
        except AdaError:
            continue
        answer = analysis.ask(question, decisions=folder_choices(given, item.name, choices))
        if not isinstance(answer, Refusal):
            fitted.append((item.name, answer))
    policy = folder_choices(given, None, choices).get("join_links", "sure")
    if fitted:
        answer = _one_answer(root, fitted, "file", question)
        if isinstance(answer, Refusal):
            return answer
        return _with_names(root, answer, _name(fitted[0][0]), given, policy)
    found = scan(root, answers=given)
    joined: list[JoinedTable] = []
    for wide in (False, True):
        # Two files first; three or more only when no pair holds what the question names.
        tables = joined_tables(root, found, policy, wide=wide)
        joined += tables
        for table in tables:
            analysis = analyze(table.frame, row_limit=sys.maxsize)
            answer = analysis.ask(question, decisions=folder_choices(given, table.from_file, choices))
            if not isinstance(answer, Refusal):
                fitted.append((table, answer))
        if fitted:
            break
    if not fitted:
        tried = f" or in the {counted(len(joined), 'join')} ADA tried" if joined else ""
        return Refusal(
            "unrepresentable_question",
            f"No file in {root.name}{tried} holds what the question names. Name the file, or "
            "confirm the link that joins the files it needs (ada scan --template).",
        )
    return _one_answer(root, fitted, "join", question, join_decision(folder_choices(given, None, choices)))


def _one_answer(root: Path, fitted: list, kind: str, question: str, decision: Decision | None = None):
    """The answer when the files or joins that answered agree, or a Refusal naming them when they do not."""
    from .api import Refusal  # noqa: PLC0415

    values = {None if answer.value is None else round(answer.value, 12) for _, answer in fitted}
    if len(fitted) > 1 and (len(values) > 1 or None in values):
        shown = "; ".join(
            f"{_name(found)} gives {answer.value:,.10g}" if answer.value is not None else _name(found)
            for found, answer in fitted
        )
        return Refusal(
            "unrepresentable_question",
            f"The question fits more than one {kind} in {root.name}, with different answers: {shown}. "
            "Name the file to ask about.",
        )
    found, answer = fitted[0]
    if kind == "join":
        note = found.describe()
        source = found.source(root)
    else:
        names = [_name(item) for item, _ in fitted]
        note = (f"Answered from {names[0]}, the only file in {root.name} whose columns fit the question."
                if len(names) == 1 else f"{', '.join(names)} all give this answer; it is from {names[0]}.")
        source = answer.source
    scope = answer.scope
    if scope is not None:
        scope = replace(scope, readings=(note, *scope.readings))
    decisions = (decision, *answer.decisions) if decision is not None else answer.decisions
    return replace(answer, answer=f"{answer.answer} {note}", scope=scope, source=source, decisions=decisions,
                   question=question)


# The last word of a column that says what a code stands for: "name", "country_name", "title".
NAME_WORDS = frozenset({"name", "title"})


def _names_codes(column: str, values: pd.Series) -> bool:
    """Whether ``column`` names each row: text, its last word a name word, nearly every value distinct."""
    words = normalized_name(column).split()
    present = values.dropna()
    return (bool(words) and words[-1] in NAME_WORDS and not is_numeric_dtype(values) and not present.empty
            and present.nunique() >= 0.9 * len(present))


def _with_names(root: Path, answer, file: str, given: Mapping[str, str], policy: str):
    """``answer`` with the name beside each code it groups by, from the file those codes link to.

    An answer grouped by iso_country shows "US" and "CN". When the folder's
    scan links iso_country to the key of another file that has a column of
    names (countries.csv: code, name), the table gains that column and the
    answer says what its first codes stand for. Names change no number, so a
    link that needs confirmation is used too, and the answer says what the
    link rests on; a link the answers file rejects, or ``join_links`` set to
    ``none``, is not.
    """
    from .api import AdaError, analyze_file  # noqa: PLC0415

    column, table = answer.plan.dimension, answer.table
    if (policy == "none" or column is None or answer.plan.second_dimension is not None or table is None
            or column not in table.columns or table.empty):
        return answer
    for link in scan(root, answers=given).links:
        if (link.from_file != file or link.from_columns != (column,) or link.kind not in KEYED
                or link.confirmed is False or link.coverage < 1):
            continue
        try:
            other = analyze_file(root / link.to_file).frame
        except AdaError:
            continue
        key = link.to_columns[0]
        names = [name for name in other.columns if name != key and _names_codes(str(name), other[name])]
        names.sort(key=lambda name: len(normalized_name(str(name)).split()))
        if key not in other.columns or not names:
            continue
        keys = _join_keys(other[key])
        lookup = pd.Series(other[names[0]].to_numpy(), index=keys.to_numpy())[keys.notna().to_numpy()]
        if lookup.index.duplicated().any():
            continue
        shown = _join_keys(table[column]).map(lookup)
        if shown.isna().iloc[0] or (shown.astype(str) == table[column].astype(str)).all():
            continue
        named = table.copy()
        named.insert(named.columns.get_loc(column) + 1, f"{names[0]} ({Path(link.to_file).stem})",
                     shown.to_numpy())
        pairs = [f"{group_label(column, code)} is {name}"
                 for code, name in zip(table[column], shown, strict=True) if pd.notna(name)][:3]
        unsure = link.needs_confirmation and link.confirmed is not True and link.reasons
        rests = f"; {link.reasons[0]}" if unsure else ""
        note = (f"In {link.to_file}, {list_names(pairs)} (matched on {link.label('from')} = "
                f"{link.label('to')}{rests}).")
        scope = answer.scope if answer.scope is None else replace(answer.scope,
                                                                  readings=(*answer.scope.readings, note))
        return replace(answer, answer=f"{answer.answer} {note}", table=named, scope=scope)
    return answer


def _name(found) -> str:
    return found.name if isinstance(found, JoinedTable) else str(found)


__all__ = ["JoinedTable", "ask_folder", "folder_choices", "join_decision", "joined_tables", "usable"]
