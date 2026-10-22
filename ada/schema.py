"""Infer the business role of each column: measure, date, dimension or identifier.

Roles are guessed from names, types and cardinality together, since none is
reliable alone: "Order Number" is numeric but not a measure, and "Value" may be either.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype

from .formatting import normalized_name


@dataclass(frozen=True)
class ColumnRoles:
    date: str | None
    measure: str | None
    dimension: str | None
    identifier: str | None
    numeric: tuple[str, ...]
    dimensions: tuple[str, ...]
    #: "sum" or "mean" when the user has said how the measure combines; None
    #: follows its name. Read it through ``metrics.metric_for``.
    aggregation: str | None = None


MEASURE_KEYWORDS = {
    "revenue": 14,
    "mrr": 14,
    "arr": 14,
    "sales": 14,
    "gmv": 14,
    "profit": 13,
    "margin": 12,
    "amount": 11,
    "value": 10,
    "income": 10,
    "spend": 9,
    "cost": 8,
    "expense": 8,
    "price": 7,
    "turnover": 14,
    "total": 9,
    "fee": 7,
    "fare": 7,
    "quantity": 6,
    "units": 6,
    "orders": 6,
    "volume": 6,
    "balance": 6,
    "count": 6,
    "abundance": 6,
    "yield": 6,
    "concentration": 5,
    "score": 4,
}

DIMENSION_KEYWORDS = {
    "product": 12,
    "category": 12,
    "segment": 12,
    "region": 11,
    "country": 10,
    "state": 9,
    "city": 9,
    "channel": 10,
    "customer": 8,
    "client": 8,
    "team": 7,
    "department": 7,
    "status": 6,
    "type": 5,
}

# What a keyword earlier in the name is worth against one at the end.
MODIFIER_WEIGHT = 2 / 3

IDENTIFIER_TOKENS = ("id", "uuid", "key", "code", "number", "invoice", "order")
TIME_PART_TOKENS = ("year", "month", "week", "day", "hour", "minute", "quarter")
# Experimental-design columns are numbered but hold labels, not quantities. Plate
# rows and columns lay out where a sample sat, not what was measured.
DESIGN_TOKENS = frozenset({
    "replicate", "rep", "batch", "plate", "well", "run", "trial", "block", "lane",
    "row", "column", "col",
})
# Conditions recorded beside an outcome, and rarely the outcome itself. Any word
# of the name counts, so "air_temp_c" matches.
COVARIATE_TOKENS = frozenset({"temp", "temperature", "humidity", "wind", "air"})
# Coordinates are continuous numbers, but never a quantity to total or trend.
COORDINATE_TOKENS = frozenset({"latitude", "longitude", "lat", "lon", "lng"})

# Head words that mark a column as a key rather than a quantity. Such columns stay
# text through reading and cleaning, so an account number 00042 keeps its zeros.
# Whole-word match, so "Paid Amount" is not an id.
IDENTIFIER_NAME_WORDS = frozenset({
    "id", "ids", "code", "codes", "zip", "zipcode", "postal", "postcode", "phone", "telephone",
    "mobile", "tel", "fax", "sku", "number", "no", "#", "key", "uuid",
})

# Head words that name the unit a row belongs to rather than a quantity: subject 12
# is not more of anything than subject 3. These columns repeat (three readings per
# participant), so they fail the uniqueness test that identifiers must pass.
ENTITY_NAME_WORDS = frozenset({
    "subject", "participant", "patient", "animal", "mouse", "rat", "donor", "individual",
    "specimen",
}) | IDENTIFIER_NAME_WORDS

# Head words of columns holding calendar years and months. A table with no date
# column but a year (and perhaps a month) is still a timeline; cleaning builds
# its dates from these.
YEAR_NAME_WORDS = frozenset({"year", "years", "yr"})
MONTH_NAME_WORDS = frozenset({"month", "months", "mon"})


# A trailing unit or annotation is not the head noun: "Revenue (USD)" is scored on "Revenue".
_ANNOTATION = re.compile(r"\s*[(\[][^)\]]*[)\]]\s*$")


DATE_NAME_TOKENS = ("date", "time", "timestamp", "created", "updated")


EVENT_DATE_TOKENS = ("order", "transaction", "sale", "invoice", "posting", "created", "booking")
FOLLOW_UP_DATE_TOKENS = ("ship", "delivery", "delivered", "due", "updated", "modified", "closed")


def _date_preference(name: str) -> int:
    words = normalized_name(name).split()
    if any(token in words for token in EVENT_DATE_TOKENS):
        return 2
    if any(token in words for token in FOLLOW_UP_DATE_TOKENS):
        return 0
    return 1


def is_identifier_name(name: str) -> bool:
    """Whether the name marks a key column: its last word is in IDENTIFIER_NAME_WORDS.

    The reader and the cleaner both use this, so they agree on which columns
    (such as "Order No") stay text.
    """
    words = _head_words(name)
    return bool(words) and words[-1] in IDENTIFIER_NAME_WORDS


def is_design_name(name: str) -> bool:
    """Whether the name marks an experimental-design label such as "Replicate" or "Plate".

    "Position" counts only after a design word ("Well Position"), since on its
    own it is as likely a player's or an employee's role.
    """
    words = _head_words(name)
    if words[-2:-1] and words[-1] == "position":
        return words[-2] in DESIGN_TOKENS
    return bool(words) and words[-1] in DESIGN_TOKENS


def is_entity_name(name: str) -> bool:
    """Whether the name labels the unit a row belongs to, such as "Subject" or "Patient ID"."""
    words = _head_words(name)
    return bool(words) and words[-1] in ENTITY_NAME_WORDS


def is_year_name(name: str) -> bool:
    """Whether the name marks a column of calendar years, such as "Year" or "survey_yr"."""
    words = _head_words(name)
    return bool(words) and words[-1] in YEAR_NAME_WORDS


def is_month_name(name: str) -> bool:
    """Whether the name marks a column of calendar months, such as "Month"."""
    words = _head_words(name)
    return bool(words) and words[-1] in MONTH_NAME_WORDS


def _named_like_a_date(name: str) -> bool:
    return any(token in normalized_name(name).split() for token in DATE_NAME_TOKENS)


def _head_words(name: str) -> list[str]:
    """Words of a column name, with any trailing annotation and punctuation removed."""
    words = (word.strip(".,:;") for word in normalized_name(_ANNOTATION.sub("", str(name))).split())
    return [word for word in words if word]


def _keyword_score(name: str, keywords: dict[str, int]) -> int:
    """Score a column name against keywords, weighting its last word most.

    The last word says what the column is; earlier words only qualify it.
    "Product Category" scores as a category; "Product Container" gets only the
    reduced MODIFIER_WEIGHT score for "product".
    """
    words = _head_words(name)
    if not words:
        return 0
    head_score = keywords.get(words[-1], 0)
    anywhere = (score for token, score in keywords.items() if token in " ".join(words))
    return max(head_score, int(max(anywhere, default=0) * MODIFIER_WEIGHT))


def looks_like_identifier(name: str, series: pd.Series) -> bool:
    if is_datetime64_any_dtype(series):
        # "Order Date" has an identifier token and unique values, but is not a row key.
        return False
    words = _head_words(name)
    if words and words[-1] in MEASURE_KEYWORDS:
        # The last word decides, as in _keyword_score: "Invoice Amount" is an amount.
        return False
    token_match = any(token in words for token in IDENTIFIER_TOKENS)
    unique_ratio = series.nunique(dropna=True) / max(int(series.notna().sum()), 1)
    return token_match and unique_ratio >= 0.8


def detect_roles(dataframe: pd.DataFrame) -> ColumnRoles:
    """Infer likely business roles from names, types, and cardinality."""
    numeric = dataframe.select_dtypes(include=np.number).columns.tolist()
    date_columns = [
        column for column in dataframe.columns if is_datetime64_any_dtype(dataframe[column])
    ]

    # Sorted by name first because max() keeps the first of equal elements, so a
    # tie goes to the alphabetically first name regardless of column order.
    date = max(
        sorted(date_columns, key=lambda column: (normalized_name(column), str(column))),
        key=lambda column: (
            1 if any(token in normalized_name(column) for token in ("date", "time", "created")) else 0,
            # The event that created the row outranks later ones: order date over ship date.
            _date_preference(column),
            int(dataframe[column].notna().sum()),
        ),
        default=None,
    )

    measure_candidates: list[tuple[int, float, float, str]] = []
    for column in numeric:
        series = dataframe[column]
        name = normalized_name(column)
        score = _keyword_score(column, MEASURE_KEYWORDS)
        if looks_like_identifier(column, series):
            score -= 20
        if any(token == name or name.endswith(f" {token}") for token in TIME_PART_TOKENS):
            score -= 15
        if is_design_name(column):
            score -= 15
        words = set(_head_words(column))
        if words & COORDINATE_TOKENS:
            score -= 15
        if words & COVARIATE_TOKENS:
            # Light, so a covariate still wins when nothing else is measured.
            score -= 3
        non_null_ratio = float(series.notna().mean())
        # Amounts carry fractions; postal codes and counters do not. A weak
        # signal, so it only breaks ties.
        continuous = float(
            not is_bool_dtype(series) and bool((series.dropna() % 1 != 0).any())
        )
        measure_candidates.append((score, continuous, non_null_ratio, column))

    measure = None
    if measure_candidates:
        # Name is the final tiebreak, so column order never changes the measure.
        measure = min(
            measure_candidates,
            key=lambda candidate: (-candidate[0], -candidate[1], -candidate[2], candidate[3]),
        )[3]

    dimensions: list[str] = []
    dimension_candidates: list[tuple[int, int, str]] = []
    for column in dataframe.columns:
        if column == date or column in numeric:
            continue
        if _named_like_a_date(column):
            # Date-named columns are never segments, even when they didn't parse as dates.
            continue
        series = dataframe[column]
        unique = int(series.nunique(dropna=True))
        non_null = int(series.notna().sum())
        if unique < 2 or unique > 100 or unique / max(non_null, 1) > 0.65:
            continue
        dimensions.append(column)
        score = _keyword_score(column, DIMENSION_KEYWORDS)
        if is_design_name(column):
            # A plate row or run label groups by layout, not by anything compared,
            # so it is picked only when no other column qualifies.
            score -= 15
        preferred_size = -abs(unique - 10)
        dimension_candidates.append((score, preferred_size, column))

    dimension = (
        max(dimension_candidates, key=lambda candidate: (candidate[0], candidate[1]))[2]
        if dimension_candidates
        else None
    )

    identifier_candidates = [
        column
        for column in dataframe.columns
        if looks_like_identifier(column, dataframe[column])
    ]
    identifier = identifier_candidates[0] if identifier_candidates else None

    return ColumnRoles(
        date=date,
        measure=measure,
        dimension=dimension,
        identifier=identifier,
        numeric=tuple(numeric),
        dimensions=tuple(dimensions),
    )
