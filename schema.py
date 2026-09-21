"""Working out what the columns of an unfamiliar file actually mean.

ADA has to guess a business schema before it can say anything: which column
is the outcome worth tracking, which one places it in time, which one splits
it into parts a reader recognises, and which one merely identifies a row.
The guesses come from names, types, and cardinality together, because none
of the three is reliable on its own -- "Order Number" is numeric but is not
a measure, and a column called "Value" may be either.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype

from formatting import normalized_name


@dataclass(frozen=True)
class ColumnRoles:
    date: str | None
    measure: str | None
    dimension: str | None
    identifier: str | None
    numeric: tuple[str, ...]
    dimensions: tuple[str, ...]


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

# Head words that say a column holds a key rather than a quantity. The reader
# keeps such a column as text and the cleaner leaves it alone, so an account
# number written 00042 stays 00042 from upload to chart. Matched as a whole
# word: "Paid Amount" is not an id because it contains one.
IDENTIFIER_NAME_WORDS = frozenset(
    {"id", "ids", "code", "codes", "zip", "postal", "phone", "sku", "number", "no", "key", "uuid"}
)


# A trailing unit or annotation is not the head noun. "Revenue (USD)" is
# revenue; scoring it on "usd" let an unrelated column take the measure role.
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
    """Whether a column's name says it holds a key: its last word is one of
    IDENTIFIER_NAME_WORDS, or the whole name is.

    This is the one place that decides, so the reader that keeps "Order No"
    as text and the cleaner that must not turn it back into a number cannot
    disagree about which columns those are.
    """
    words = _head_words(name)
    return bool(words) and words[-1] in IDENTIFIER_NAME_WORDS


def _named_like_a_date(name: str) -> bool:
    return any(token in normalized_name(name).split() for token in DATE_NAME_TOKENS)


def _head_words(name: str) -> list[str]:
    """Words of a column name, with any trailing annotation removed."""
    return normalized_name(_ANNOTATION.sub("", str(name))).split()


def _keyword_score(name: str, keywords: dict[str, int]) -> int:
    """Score a column name, trusting its last word most.

    The last word says what a column *is*; earlier words only qualify it.
    "Product Category" is a category, but "Product Container" is a container --
    a packaging attribute that scored just as high as the real product
    dimension while both merely contained the word "product".
    """
    words = _head_words(name)
    if not words:
        return 0
    head_score = keywords.get(words[-1], 0)
    anywhere = (score for token, score in keywords.items() if token in " ".join(words))
    return max(head_score, int(max(anywhere, default=0) * MODIFIER_WEIGHT))


def looks_like_identifier(name: str, series: pd.Series) -> bool:
    if is_datetime64_any_dtype(series):
        # "Order Date" carries an identifier token and its values are as unique
        # as any key, but a date names a moment, not a row.
        return False
    words = _head_words(name)
    if words and words[-1] in MEASURE_KEYWORDS:
        # The last word decides here for the same reason it decides a measure
        # score: "Invoice Amount" is an amount that happens to sit beside an
        # invoice, and reading it as a key costs the file its real metric.
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

    # Candidates are sorted by their full name before ranking. max() returns
    # the first of equal elements, so a tie between "Transaction Date A" and
    # "Transaction Date B" resolves to the alphabetically first name whatever
    # order the file wrote them in. An earlier tiebreak compared only the
    # first eight bytes, which is not enough to separate two long names that
    # share a prefix.
    date = max(
        sorted(date_columns, key=lambda column: (normalized_name(column), str(column))),
        key=lambda column: (
            1 if any(token in normalized_name(column) for token in ("date", "time", "created")) else 0,
            # The event that produced the row outranks what happened to it
            # afterwards: an order dates a sale, a shipment dates a delivery.
            _date_preference(column),
            int(dataframe[column].notna().sum()),
        ),
        default=None,
    )

    measure_candidates: list[tuple[int, float, str]] = []
    for column in numeric:
        series = dataframe[column]
        name = normalized_name(column)
        score = _keyword_score(column, MEASURE_KEYWORDS)
        if looks_like_identifier(column, series):
            score -= 20
        if any(token == name or name.endswith(f" {token}") for token in TIME_PART_TOKENS):
            score -= 15
        non_null_ratio = float(series.notna().mean())
        # Amounts carry fractions; postal codes and counters do not. It is a
        # weak signal, so it only breaks ties -- but it breaks them on the
        # data, where the fallback was breaking them on column order.
        continuous = float(
            not is_bool_dtype(series) and bool((series.dropna() % 1 != 0).any())
        )
        measure_candidates.append((score, continuous, non_null_ratio, column))

    measure = None
    if measure_candidates:
        # The column name is the final tiebreak so that two exports of one
        # table, written in different column orders, detect the same measure.
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
            # ADA tried to parse this as a date and could not. Promoting it to
            # "business segment" produced evidence reading "2024-06-11 is the
            # largest date", which is not a finding about the business.
            continue
        series = dataframe[column]
        unique = int(series.nunique(dropna=True))
        non_null = int(series.notna().sum())
        if unique < 2 or unique > 100 or unique / max(non_null, 1) > 0.65:
            continue
        dimensions.append(column)
        score = _keyword_score(column, DIMENSION_KEYWORDS)
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


def override_roles(
    roles: ColumnRoles,
    *,
    date: str | None = None,
    measure: str | None = None,
    dimension: str | None = None,
) -> ColumnRoles:
    return ColumnRoles(
        date=date if date is not None else roles.date,
        measure=measure if measure is not None else roles.measure,
        dimension=dimension if dimension is not None else roles.dimension,
        identifier=roles.identifier,
        numeric=roles.numeric,
        dimensions=roles.dimensions,
    )
