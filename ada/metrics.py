"""Resolve how a measure aggregates and whether a rise in it is good news.

Every surface asks here instead of re-reading the column name, so they agree that
rates and measured levels (temperature, pH, concentration) are averaged, not
summed, and have no meaningful shares or waterfall.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .formatting import is_currency, is_percentage, normalized_name, percentage_outranks_currency

# Words that make a rise bad news when they lead or end a measure's name.
ADVERSE_MEASURE_TOKENS = frozenset({
    "cost", "costs", "expense", "expenses", "spend", "churn", "refund", "refunds",
    "returns", "loss", "losses", "overdue", "delay", "delays", "hours", "tickets",
    "complaints", "defects", "errors", "bounce", "debt", "outstanding",
})
# Measurements that are averaged rather than summed. Adding up temperatures or
# concentrations across rows doesn't give a meaningful quantity.
LEVEL_WORDS = frozenset({
    "temperature", "temp", "ph", "absorbance", "fluorescence", "intensity", "concentration",
    "conc", "density", "viscosity", "pressure", "humidity", "salinity", "conductivity",
    "turbidity", "ratio", "index", "score", "rating", "level", "mean", "average", "avg",
    "median", "age", "bmi", "height", "length", "width", "depth", "diameter", "velocity",
    "speed", "voltage", "elevation", "altitude", "latitude", "longitude",
})
# Unit annotations that mark a level: a concentration or density (mg/L, cells/mL,
# g/cm3), degrees, ppm/ppb, and molar units (mM, µM). Currency names never match.
_LEVEL_UNIT = re.compile(
    r"[(\[][^)\]]*(?:/\s*(?:[mµudn]?[lL]|[mµunk]?g|c?m[23²³])\b|°|\bpp[mbt]\b|\b[mµun]M\b)[^)\]]*[)\]]"
)
# Optical density readings: "OD", "OD600".
_OPTICAL_DENSITY = re.compile(r"^od\d*$")
# Any of these anywhere in the name makes a rise good news: "Cost Savings".
FAVOURABLE_OVERRIDES = frozenset({"savings", "saved", "recovered", "avoided"})
# Words that start a qualifier. "Revenue After Returns" is judged on "Revenue",
# and so is "Revenue Net of Returns".
QUALIFIER_WORDS = frozenset({"after", "before", "less", "minus", "excluding", "excl", "without"})


def increase_is_welcome(measure: str | None) -> bool:
    """Whether a rise in this measure is good news.

    The name is cut at the first qualifier word, then its first and last words
    decide: "Refund Amount" and "Sales Returns" are bad news when they rise,
    "Revenue After Returns" is not.
    """
    if not measure:
        return True
    words = normalized_name(measure).split()
    if any(word in FAVOURABLE_OVERRIDES for word in words):
        return True
    for position, word in enumerate(words):
        net_of = word == "net" and words[position + 1 : position + 2] == ["of"]
        if position > 0 and (word in QUALIFIER_WORDS or net_of):
            words = words[:position]
            break
    if not words:
        return True
    return words[0] not in ADVERSE_MEASURE_TOKENS and words[-1] not in ADVERSE_MEASURE_TOKENS


@dataclass(frozen=True)
class MetricSpec:
    """How a measure combines, and what may be said about it."""

    name: str | None
    aggregation: str  # sum | mean | count
    unit: str  # amount | rate | level | count
    increase_is_welcome: bool

    @property
    def additive(self) -> bool:
        """Whether parts add up to the whole.

        Shares of a total, concentration and segment waterfalls are valid only
        when this is true; it is false for rates and levels, which are averaged.
        """
        return self.aggregation != "mean"

    @property
    def combines_as(self) -> str:
        return {"sum": "total", "mean": "average", "count": "count"}[self.aggregation]


def is_level_name(name: str) -> bool:
    """Whether the column name describes a measured level, such as "Water Temp (°C)"."""
    words = set(normalized_name(name).split()) | set(re.findall(r"[a-z0-9]+", str(name).lower()))
    if words & LEVEL_WORDS or any(_OPTICAL_DENSITY.match(word) for word in words):
        return True
    return not is_currency(name) and bool(_LEVEL_UNIT.search(str(name)))


def resolve_metric(name: str | None) -> MetricSpec:
    """Decide a measure's aggregation, unit and direction from its column name.

    Percentage-named columns are averaged as rates and measured levels
    (see ``is_level_name``) as levels; other columns are summed as amounts.
    No name means a row count.
    """
    if not name:
        return MetricSpec(name=None, aggregation="count", unit="count", increase_is_welcome=True)
    if is_percentage(name) and percentage_outranks_currency(name):
        return MetricSpec(
            name=name, aggregation="mean", unit="rate", increase_is_welcome=increase_is_welcome(name)
        )
    if is_level_name(name):
        return MetricSpec(
            name=name, aggregation="mean", unit="level", increase_is_welcome=increase_is_welcome(name)
        )
    return MetricSpec(
        name=name, aggregation="sum", unit="amount", increase_is_welcome=increase_is_welcome(name)
    )
