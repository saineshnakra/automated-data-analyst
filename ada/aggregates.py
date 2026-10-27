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


__all__ = ["AggregateRows", "aggregate_rows"]
