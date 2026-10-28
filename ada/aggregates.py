"""Rows that add up other rows: a World or Total row, and regions or income groups among countries.

Published tables often list such rows beside the rows they add up: World,
Africa and High-income countries among OWID's countries, a Total at the
bottom of a spreadsheet. Counting them with the rest counts some values
twice. ``aggregate_rows`` finds them by their labels, so an answer can leave
them out and say which it left out; whether to is a decision (see
``ada.decisions``).

- In a column that names places (country, entity, region, location and the
  like), a row is an aggregate when its label is a total (World, Total), a
  continent or region (Africa, South Asia, Europe & Central Asia), or a group
  (income, countries, OECD, excluding). A column of codes (USA, OWID_WRL) is
  not read as names, so the code IDA is not taken for a group.
- In any other text column, only a total label counts (Total, Grand total),
  and only when those rows add up to the other rows in some numeric column.

Aggregates must be fewer than half of a column's labels and leave at least
two others, so a column of regions alone (gapminder's continent) has none.
"""

from __future__ import annotations

import re
import weakref
from dataclasses import dataclass

import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype

from .formatting import list_names, normalized_name
from .numeric import to_numbers

#: Words in a column's name that say it names places.
PLACE_WORDS = frozenset({
    "country", "countries", "entity", "nation", "region", "area", "location", "economy", "territory",
    "geo", "geography", "state", "province", "county", "place", "continent",
})
#: Labels that stand for every other row.
TOTAL_LABELS = frozenset({
    "total", "totals", "grand total", "sum", "overall", "all", "world", "global", "all countries",
    "world total", "total world", "whole world",
})
#: Continents and regions, as UN, World Bank, FAO and OWID tables name them, after ``_label_key``.
REGION_NAMES = frozenset({
    "africa", "asia", "europe", "oceania", "americas", "north america", "south america", "latin america",
    "northern africa", "north africa", "eastern africa", "middle africa", "southern africa", "western africa",
    "sub-saharan africa", "caribbean", "central america", "northern america",
    "latin america and the caribbean", "latin america and caribbean", "central asia", "eastern asia",
    "east asia", "south-eastern asia", "southeastern asia", "south eastern asia", "southern asia",
    "south asia", "western asia", "eastern europe", "northern europe", "southern europe", "western europe",
    "central europe", "australia and new zealand", "melanesia", "polynesia", "middle east",
    "middle east and north africa", "east asia and pacific", "europe and central asia",
    "africa eastern and southern", "africa western and central", "central europe and the baltics",
    "european union", "euro area", "eu-27", "eu-28", "eu27", "eu28",
})
#: Words that mark a group of places: "High income", "Least developed countries", "Asia (excl. China)".
GROUP_PATTERN = re.compile(
    r"\b(?:income|countries|excl|excluding|including|oecd|ida|ibrd|hipc|dividend|small states|developing"
    r"|developed|fragile|world|total)\b|\(region\)"
)
#: A label that is a code rather than a name: "USA", "OWID_WRL", "IDA".
CODE_PATTERN = re.compile(r"[A-Z0-9_]{2,8}")
#: The share of a column's labels that are codes for it to be a column of codes, not of names.
CODE_SHARE = 0.8
#: Most of a column's labels must be parts, not aggregates.
MAX_AGGREGATE_SHARE = 0.5
#: How close a total row's sum must come to the other rows' sum, as a fraction of it.
SUM_TOLERANCE = 0.01


@dataclass(frozen=True)
class AggregateRows:
    """The rows of ``column`` that add up others: every such label, and those that stand for all rows."""

    column: str
    labels: tuple[str, ...]
    totals: tuple[str, ...] = ()

    def mask(self, frame: pd.DataFrame) -> pd.Series:
        """The rows of ``frame`` that are aggregates."""
        return frame[self.column].astype(str).isin(self.labels) & frame[self.column].notna()

    def named(self, labels: list[str] | tuple[str, ...] | None = None) -> str:
        """Some of the labels for a sentence, totals first: "World, Africa, Asia and 45 others"."""
        labels = list(self.labels if labels is None else labels)
        ordered = [label for label in labels if label in self.totals] + [
            label for label in labels if label not in self.totals
        ]
        return list_names(ordered)


def _label_key(label: str) -> str:
    """A label as the name lists hold it: "Asia, Central" is "central asia", "Africa (GCP)" is "africa"."""
    text = re.sub(r"\([^)]*\)", " ", label.casefold().replace("&", " and "))
    if text.count(",") == 1:
        before, after = (part.strip() for part in text.split(","))
        if len(after.split()) == 1:
            text = f"{after} {before}"
    return " ".join(text.split())


def distinct_parts(labels: list[str]) -> dict[str, str]:
    """Each label without the words every label starts or ends with, after a comma: "Age of mother, all ages"
    is "all ages", "Alberta, place of residence of mother" is "Alberta"."""
    parts = {label: label for label in labels}
    if len(labels) < 2:
        return parts
    heads = {label.split(",", 1)[0] for label in labels if "," in label}
    tails = {label.rsplit(",", 1)[1] for label in labels if "," in label}
    if len(heads) == 1 and all("," in label for label in labels):
        parts = {label: label.split(",", 1)[1].strip() for label in labels}
    elif len(tails) == 1 and all("," in label for label in labels):
        parts = {label: label.rsplit(",", 1)[0].strip() for label in labels}
    return parts


#: How near the rows of a label with no total word must come to the rows of the others, as a fraction.
NAMELESS_TOLERANCE = 0.02


def _sum_of_others(frame: pd.DataFrame, column: object, labels: list[str]) -> str | None:
    """The label whose counts are the other labels' added up, cell by cell, though no word says it is a total:
    "Canada" over the provinces. Only the label with the largest counts is tried, on columns of counts."""
    if len(frame) > MAX_ALIGNED_ROWS:
        return None
    texts = frame[column].astype(str)
    for other in frame.columns:
        values = frame[other]
        if other == column or is_bool_dtype(values) or not is_numeric_dtype(values) or values.nunique() <= 20:
            continue
        numbers = to_numbers(values)
        if (numbers < 0).any():
            continue
        largest = numbers.groupby(texts).sum().idxmax()
        if _covers_where_given(frame, column, largest, other, numbers, tolerance=NAMELESS_TOLERANCE):
            return str(largest)
    return None


def _is_place_column(column: object) -> bool:
    return bool(set(normalized_name(str(column)).split()) & PLACE_WORDS)


def _is_aggregate(label: str, place: bool) -> bool:
    key = _label_key(label)
    if key in TOTAL_LABELS:
        return True
    return place and (key in REGION_NAMES or bool(GROUP_PATTERN.search(label.casefold())))


def _adds_up(frame: pd.DataFrame, column: object, labels: list[str]) -> bool:
    """Whether the rows labelled ``labels`` sum to the other rows in some numeric column."""
    rows = frame[column].astype(str).isin(labels) & frame[column].notna()
    for other in frame.columns:
        if other == column or is_bool_dtype(frame[other]) or not is_numeric_dtype(frame[other]):
            continue
        values = to_numbers(frame[other])
        if not values[rows].notna().any():
            continue
        total, parts = float(values[rows].sum()), float(values[~rows].sum())
        if parts and abs(total - parts) <= SUM_TOLERANCE * abs(parts):
            return True
    return False


def _found_in(frame: pd.DataFrame, column: object) -> AggregateRows | None:
    series = frame[column]
    if is_numeric_dtype(series) or is_datetime64_any_dtype(series) or is_bool_dtype(series):
        return None
    labels = [str(label) for label in pd.unique(series.dropna().astype(str))]
    if len(labels) < 3:
        return None
    codes = sum(bool(CODE_PATTERN.fullmatch(label)) for label in labels) >= CODE_SHARE * len(labels)
    place = _is_place_column(column) and not codes
    found = [label for label in labels if _is_aggregate(label, place)]
    if not found or len(labels) - len(found) < 2 or len(found) > MAX_AGGREGATE_SHARE * len(labels):
        return None
    if not place and not _adds_up(frame, column, found):
        return None
    totals = tuple(label for label in found if _label_key(label) in TOTAL_LABELS)
    return AggregateRows(str(column), tuple(found), totals)


#: A level that stands for every other level of its column in a cross-tabulated table: "All Ages",
#: "Both sexes", "Persons", "All races", "Total"; "All Other Transport" is one level among the rest.
TOTAL_LEVEL = re.compile(
    r"(?:all(?!\sothers?\b)|total)(?:\s.+)?|both sexes|persons|everyone|.+\stotal|overall"
)
#: Fewest and most labels a column of levels with a total among them may hold.
MIN_LEVELS, MAX_LEVELS = 3, 60
#: How far the rows of a total level may exceed the rows of the other levels in some count, as a
#: fraction: groups no level names ("other races") add to the total but to no level.
LEVEL_EXCESS = 0.5


@dataclass(frozen=True)
class TotalLevel:
    """A column of a cross-tabulated table and its level that stands for all its others."""

    column: str
    label: str


def _covers(frame: pd.DataFrame, column: object, label: str) -> bool:
    """Whether the rows of ``label`` add up to at least the rows of the other levels of ``column`` in some
    numeric column, and to no more than half as much again.

    When some counts are blank (a total withheld where its parts are given),
    only the cells whose total and every part are given are compared.
    """
    rows = frame[column].astype(str).eq(label) & frame[column].notna()
    for other in frame.columns:
        if other == column or is_bool_dtype(frame[other]) or not is_numeric_dtype(frame[other]):
            continue
        values = to_numbers(frame[other])
        total, parts = float(values[rows].sum()), float(values[~rows & frame[column].notna()].sum())
        if parts > 0 and parts * (1 - SUM_TOLERANCE) <= total <= parts * (1 + LEVEL_EXCESS):
            return True
        if values.isna().any() and _covers_where_given(frame, column, label, other, values):
            return True
    return False


#: Most rows a table may have for the cell-by-cell comparison of a total with its parts.
MAX_ALIGNED_ROWS = 200_000


def _covers_where_given(frame: pd.DataFrame, column: object, label: str, other: object, values: pd.Series,
                        tolerance: float | None = None) -> bool:
    """``_covers`` over the cells whose total and every part are given, matched on the other columns.

    When no cell gives every part (a territory split in two partway through
    the years), each cell's total is compared with the parts it gives. With
    ``tolerance`` the total must come within that fraction of its parts
    either way, in nine cells of ten.
    """
    if len(frame) > MAX_ALIGNED_ROWS:
        return False
    # A code beside the column (C02076V03371 for Age Group, a DGUID for GEO) names the same level, so it
    # marks no cell.
    keys = [key for key in frame.columns if key not in (column, other)
            and not (is_numeric_dtype(frame[key]) and frame[key].nunique() > len(frame) // 2)
            and not (frame[key].nunique() > 1
                     and frame.dropna(subset=[key]).groupby(key)[column].nunique().max() == 1)]
    if not keys:
        return False
    cells = frame[keys].astype(str).agg("\x1f".join, axis=1)
    level = frame[column].astype(str)
    table = pd.DataFrame({"cell": cells, "level": level, "value": values}).dropna(subset=["value"])
    if table.duplicated(["cell", "level"]).any():
        return False
    wide = table.pivot(index="cell", columns="level", values="value")
    if label not in wide.columns or wide.shape[1] < 2:
        return False
    given = wide.loc[wide[label].notna()]
    others = given.drop(columns=[label])
    if tolerance is not None or not others.notna().all(axis=1).any():
        # A cell of zeros (no births under 15 in a territory) says nothing either way.
        usable = (others.notna().sum(axis=1) >= 2) & (others.sum(axis=1) > 0)
        totals, parts = given.loc[usable, label], others.loc[usable].sum(axis=1)
        if len(totals) < 3:
            return False
        low, high = (1 - tolerance, 1 + tolerance) if tolerance is not None else (1 - SUM_TOLERANCE,
                                                                                  1 + LEVEL_EXCESS)
        within = (totals / parts).between(low, high)
        if bool(within.mean() >= 0.9):
            return True
        # Counts beside percentages of them ("Number of live births", "Percentage of live births"): the
        # counts may add up where the percentages do not.
        cell_keys = frame.loc[table.index, keys].set_axis(table["cell"]).groupby(level=0).first()
        for key in keys:
            if not 2 <= frame[key].nunique() <= 5:
                continue
            groups = cell_keys.loc[within.index, key].astype(str)
            for _, members in within.groupby(groups):
                if len(members) >= 3 and bool(members.mean() >= 0.9):
                    return True
        return False
    complete = given.loc[others.notna().all(axis=1)]
    total = float(complete[label].sum())
    parts = float(complete.drop(columns=[label]).to_numpy().sum())
    return parts > 0 and parts * (1 - SUM_TOLERANCE) <= total <= parts * (1 + LEVEL_EXCESS)


#: Levels a column of geographic levels may name: the whole, and the parts it is split into.
WHOLE_LEVELS = frozenset({"country", "national", "nation", "nationwide", "world", "global", "total", "all"})
PART_LEVELS = frozenset({
    "area", "areas", "county", "counties", "state", "states", "region", "regions", "regional", "district",
    "districts", "municipality", "municipalities", "municipal", "province", "provinces", "provincial", "city",
    "cities", "local", "zone", "zones", "borough", "boroughs", "ward", "wards", "parish", "parishes",
    "division", "divisions", "subregion", "subregions", "neighbourhood", "neighborhood", "local authority",
    "council", "health board",
})
#: How close the whole's rows must come to one level of parts in some count, as a fraction of it.
LEVEL_TOLERANCE = 0.02


def _level_whole(frame: pd.DataFrame, column: object, labels: list[str]) -> str | None:
    """The label of a column of geographic levels ("country" of area, county and country) whose rows add
    up to the rows of another level in some count, or None."""
    keys = {label: _label_key(label) for label in labels}
    wholes = [label for label, key in keys.items() if key in WHOLE_LEVELS]
    if len(wholes) != 1 or not all(key in WHOLE_LEVELS | PART_LEVELS for key in keys.values()):
        return None
    texts = frame[column].astype(str)
    for other in frame.columns:
        if other == column or is_bool_dtype(frame[other]) or not is_numeric_dtype(frame[other]):
            continue
        values = to_numbers(frame[other])
        whole = float(values[texts.eq(wholes[0])].sum())
        for label in labels:
            part = float(values[texts.eq(label)].sum()) if label != wholes[0] else 0.0
            if part > 0 and abs(whole - part) <= LEVEL_TOLERANCE * part:
                return wholes[0]
    return None


def total_levels(frame: pd.DataFrame) -> tuple[TotalLevel, ...]:
    """The columns of a cross-tabulated table that each hold one level standing for all the others.

    A published table of counts often gives every count under each level of
    a column and again under its total ("Both sexes", "All Ages", "All
    races"), so adding up every row counts each death once per level. A column
    qualifies when one of its 3 to 60 labels reads as a total and its rows add
    up to at least the other levels' rows in some count. Empty unless two or
    more columns qualify: one total row alone is ``aggregate_rows``' to find.

    A column of geographic levels (area, county and country, each row at one
    of them) qualifies alone, when the whole's rows add up to the rows of one
    level of parts in some count: each fish is then counted once per level.
    """
    cached = _LEVELS.get(id(frame))
    if cached and cached[0]() is frame:
        return cached[1]
    found, wholes = [], []
    for column in frame.columns:
        series = frame[column]
        if is_numeric_dtype(series) or is_datetime64_any_dtype(series) or is_bool_dtype(series):
            continue
        labels = [str(label) for label in pd.unique(series.dropna().astype(str))]
        if 2 <= len(labels) <= MIN_LEVELS * 2 and (whole := _level_whole(frame, column, labels)):
            wholes.append(TotalLevel(str(column), whole))
            continue
        if not MIN_LEVELS <= len(labels) <= MAX_LEVELS:
            continue
        # "Age of mother, all ages": the words every label shares say nothing of the level.
        keys = distinct_parts(labels)
        totals = [label for label in labels if TOTAL_LEVEL.fullmatch(_label_key(keys[label]))]
        if len(totals) == 1 and _covers(frame, column, totals[0]):
            found.append(TotalLevel(str(column), totals[0]))
        elif not totals and (whole := _sum_of_others(frame, column, labels)):
            found.append(TotalLevel(str(column), whole))
    levels = (*(found if len(found) >= 2 else ()), *wholes)
    if len(_LEVELS) >= 8:
        _LEVELS.clear()
    _LEVELS[id(frame)] = (weakref.ref(frame), levels)
    return levels


# The total levels of the last few tables looked at, found once per table.
_LEVELS: dict[int, tuple[weakref.ref, tuple[TotalLevel, ...]]] = {}


# The aggregate rows of the last few tables looked at, found once per table.
_FOUND: dict[int, tuple[weakref.ref, AggregateRows | None]] = {}


def aggregate_rows(frame: pd.DataFrame) -> AggregateRows | None:
    """The rows of ``frame`` that add up others, in the column with the most of them, or None.

    The column with the most aggregate labels wins, then the first.
    """
    cached = _FOUND.get(id(frame))
    if cached and cached[0]() is frame:
        return cached[1]
    best: AggregateRows | None = None
    for column in frame.columns:
        found = _found_in(frame, column)
        if found is None:
            continue
        if best is None or len(found.labels) > len(best.labels):
            best = found
    if len(_FOUND) >= 8:
        _FOUND.clear()
    _FOUND[id(frame)] = (weakref.ref(frame), best)
    return best


__all__ = ["AggregateRows", "TotalLevel", "aggregate_rows", "total_levels"]
