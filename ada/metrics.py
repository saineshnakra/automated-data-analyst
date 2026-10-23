"""Resolve how a measure aggregates and whether a rise in it is good news.

Every surface asks here instead of re-reading the column name, so they agree on
what a measure is. Only money and counts of things are summed: a reading such
as a temperature, a concentration or a price is averaged, since adding it up
across rows gives no meaningful quantity, and an unfamiliar name is averaged
too, because the mean of an amount still reads sensibly while the sum of a
reading does not. Among lab readings "total" means all forms of a substance
(total nitrogen, total phosphorus), so a name that is an amount only by that
word is averaged there.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .formatting import (
    CURRENCY_CODES,
    is_currency,
    is_percentage,
    normalized_name,
    percentage_outranks_currency,
)

if TYPE_CHECKING:  # Only for annotations; schema needs nothing from here.
    from .schema import ColumnRoles

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
# Amounts of money and counts of things, whose rows add up to a total.
ADDITIVE_WORDS = frozenset({
    "amount", "amounts", "total", "totals", "cost", "costs", "spend", "spending",
    "expense", "expenses", "expenditure", "expenditures", "profit", "profits", "income",
    "earnings", "fee", "fees", "payment", "payments", "fare", "fares", "tip", "tips",
    "tax", "taxes", "donation", "donations", "quantity", "quantities", "qty", "units",
    "count", "counts", "orders", "visits", "sessions", "clicks", "impressions", "views",
    "pageviews", "downloads", "installs", "signups", "registrations", "subscriptions",
    "tickets", "calls", "messages", "emails", "cases", "infections", "deaths", "births",
    "admissions", "discharges", "patients", "passengers", "trips", "flights", "customers",
    "users", "accounts", "transactions", "items", "bookings", "reservations", "leads",
    "conversions", "purchases", "applications", "enrollments", "responses", "shipments",
    "deliveries", "incidents", "defects", "errors", "complaints", "refunds", "returns",
    "cells", "colonies", "individuals", "specimens", "sightings", "abundance", "population",
    "visitors", "arrivals", "departures",
})
# Levels whose scale carries on below zero, so a forecast heading down may pass it
# even when the history never did: a temperature in °C or °F, a height relative to
# sea level. Rainfall, concentrations and most other readings stop at zero.
SIGNED_LEVEL_WORDS = frozenset({
    "temperature", "temp", "celsius", "fahrenheit", "elevation", "altitude",
})
# Head words of a per-unit price, which is averaged although it is money.
PRICE_WORDS = frozenset({"price", "prices"})
# Money beyond the currency words, which adds up even across subjects.
MONEY_WORDS = frozenset({
    "fee", "fees", "payment", "payments", "fare", "fares", "tip", "tips", "tax", "taxes",
    "donation", "donations", "earnings", "expenditure", "expenditures", "spending", "costs",
    "expenses", "profits",
})
# Words of a row key naming one subject or sample per row. A participant's total
# score is compared between participants, never added up across them; an order
# number names a sale, whose amounts do add up.
SUBJECT_WORDS = frozenset({
    "subject", "participant", "patient", "animal", "mouse", "rat", "donor", "individual",
    "specimen", "sample", "respondent", "infant", "child", "person",
})
# Words that make a name a total, which is money or a count only beside other money or counts.
TOTAL_WORDS = frozenset({"total", "totals"})
# How a measure may be told to combine, overriding what its name implies.
AGGREGATION_CHOICES = ("sum", "mean")
# Unit annotations that mark a level: a concentration or density (mg/L, cells/mL,
# g/cm3), degrees, ppm/ppb, and molar units (mM, µM). Currency names never match.
_LEVEL_UNIT = re.compile(
    r"[(\[][^)\]]*(?:/\s*(?:[mµudn]?[lL]|[mµunk]?g|c?m[23²³])\b|°|\bpp[mbt]\b|\b[mµun]M\b)[^)\]]*[)\]]"
)
# A trailing unit or annotation is not part of the name proper: "Price (USD)" is a price.
_ANNOTATION = re.compile(r"\s*[(\[][^)\]]*[)\]]\s*$")
# Optical density readings: "OD", "OD600".
_OPTICAL_DENSITY = re.compile(r"^od\d*$")
# Any of these anywhere in the name makes a rise good news: "Cost Savings".
FAVOURABLE_OVERRIDES = frozenset({"savings", "saved", "recovered", "avoided"})
# Words that start a qualifier. "Revenue After Returns" is judged on "Revenue",
# and so is "Revenue Net of Returns".
QUALIFIER_WORDS = frozenset({"after", "before", "less", "minus", "excluding", "excl", "without"})


def _named_direction(measure: str | None) -> bool | None:
    """What the name says about a rise: True for good news, False for bad, None for neither.

    The name is cut at the first qualifier word, then its first and last words
    decide: "Refund Amount" and "Sales Returns" are bad news when they rise,
    "Revenue After Returns" says nothing either way.
    """
    if not measure:
        return None
    words = normalized_name(measure).split()
    if any(word in FAVOURABLE_OVERRIDES for word in words):
        return True
    for position, word in enumerate(words):
        net_of = word == "net" and words[position + 1 : position + 2] == ["of"]
        if position > 0 and (word in QUALIFIER_WORDS or net_of):
            words = words[:position]
            break
    if words and (words[0] in ADVERSE_MEASURE_TOKENS or words[-1] in ADVERSE_MEASURE_TOKENS):
        return False
    return None


def increase_is_welcome(measure: str | None) -> bool:
    """Whether a rise in this measure is good news.

    True unless the name marks the measure as adverse (see ``_named_direction``),
    so more revenue or more units read as good news by default.
    """
    return _named_direction(measure) is not False


@dataclass(frozen=True)
class MetricSpec:
    """How a measure combines, and what may be said about it."""

    name: str | None
    aggregation: str  # sum | mean | count
    unit: str  # amount | rate | level | count
    increase_is_welcome: bool
    #: False for a level whose name says nothing about which way is good: a rise
    #: in CO2 or rainfall is reported, not praised or flagged.
    direction_known: bool = True

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

    @property
    def floors_at_zero(self) -> bool:
        """Whether a forecast may stop at zero when the history never goes below it.

        Amounts, counts, rates and most readings (rainfall, a concentration)
        cannot be negative. A temperature or an elevation can: one falling past
        0 °C keeps falling.
        """
        return not is_signed_level_name(self.name)


def is_signed_level_name(name: str | None) -> bool:
    """Whether the name is a reading whose scale goes below zero, such as "Water Temp (°C)"."""
    if not name:
        return False
    words = set(normalized_name(name).split()) | set(re.findall(r"[a-z0-9]+", str(name).lower()))
    return bool(words & SIGNED_LEVEL_WORDS) or "°" in str(name)


def is_level_name(name: str) -> bool:
    """Whether the column name describes a measured level, such as "Water Temp (°C)"."""
    words = set(normalized_name(name).split()) | set(re.findall(r"[a-z0-9]+", str(name).lower()))
    if words & LEVEL_WORDS or any(_OPTICAL_DENSITY.match(word) for word in words):
        return True
    return not is_currency(name) and bool(_LEVEL_UNIT.search(str(name)))


def _name_words(name: str) -> list[str]:
    """Whole words of the name, without a trailing annotation or currency code."""
    words = re.findall(r"[a-z0-9]+", normalized_name(_ANNOTATION.sub("", str(name))))
    while len(words) > 1 and words[-1] in CURRENCY_CODES:
        words.pop()
    return words


def is_additive_name(name: str) -> bool:
    """Whether the name marks an amount of money or a count of things.

    A per-unit figure is not one, even when it is money: "Unit Price",
    "Revenue per User" and "Spending per Capita" are averaged. A price that is
    also called a total ("Total Price") is still summed.
    """
    words = _name_words(name)
    if not words or "per" in words:
        return False
    named = bool(set(words) & ADDITIVE_WORDS)
    if words[-1] in PRICE_WORDS:
        return named
    return named or is_currency(name)


def resolve_metric(name: str | None, aggregation: str | None = None) -> MetricSpec:
    """Decide a measure's aggregation, unit and direction from its column name.

    Percentage-named columns are averaged as rates and measured levels
    (see ``is_level_name``) as levels. Money and counts (see
    ``is_additive_name``) are summed as amounts, and anything else is averaged
    as a level. No name means a row count.

    ``aggregation`` ("sum" or "mean") overrides the choice for a name that
    misleads. The unit follows it: a column told to sum is an amount, and one
    told to average is a level unless it is already a rate.
    """
    if not name:
        return MetricSpec(name=None, aggregation="count", unit="count", increase_is_welcome=True)
    if is_percentage(name) and percentage_outranks_currency(name):
        combined, unit = "mean", "rate"
    elif is_level_name(name):
        combined, unit = "mean", "level"
    elif is_additive_name(name):
        combined, unit = "sum", "amount"
    else:
        combined, unit = "mean", "level"
    if aggregation is not None and aggregation != combined:
        combined, unit = aggregation, "amount" if aggregation == "sum" else "level"
    named = _named_direction(name)
    return MetricSpec(
        name=name,
        aggregation=combined,
        unit=unit,
        increase_is_welcome=named is not False,
        direction_known=unit != "level" or named is not None,
    )


def rows_are_subjects(roles: ColumnRoles | None) -> bool:
    """Whether each row is one subject or sample: the row key is named for one ("ParticipantId")."""
    if roles is None or not roles.identifier:
        return False
    return bool(set(normalized_name(roles.identifier).split()) & SUBJECT_WORDS)


def _names_money(name: str) -> bool:
    return is_currency(name) or bool(set(_name_words(name)) & MONEY_WORDS)


def _total_only(name: str) -> bool:
    """Whether the name is an amount only because it says "total": "nitrates_total", "density_Total"."""
    words = set(_name_words(name))
    named = words & (ADDITIVE_WORDS - TOTAL_WORDS) or words & PRICE_WORDS or _names_money(name)
    return bool(words & TOTAL_WORDS) and not named


def _among_readings(column: str, roles: ColumnRoles) -> bool:
    """Whether the column's numeric neighbours are readings, with no money or count among them.

    Two at least are needed: a table of a date, a store and a "Total" is a
    sales total.
    """
    others = [
        other for other in roles.numeric
        if other != column and other not in roles.factors and other not in roles.non_readings
    ]
    counted = [other for other in others if is_additive_name(other) and not _total_only(other)]
    return len(others) >= 2 and not counted


def metric_for(column: str | None, roles: ColumnRoles | None = None) -> MetricSpec:
    """How ``column`` combines, with the roles' override applied when it is their measure.

    Callers that hold the roles ask here rather than ``resolve_metric``, so an
    override set in the app, the CLI or ``ada.analyze`` reaches the trend, the
    breakdown, the brief, the charts and the answers alike. A 0/1 flag (see
    ``ColumnRoles.flags``) averages to the share of rows that are 1, so it is a
    rate whatever its name, unless the override says to count the 1s. When
    each row is one subject or sample (see ``rows_are_subjects``), a count or
    a total that is not money is averaged too: summing each infant's looking
    time across infants, or splitting it into shares, mostly counts infants.
    So is a name that is an amount only because it says "total" when every
    other numeric column is a reading (see ``_among_readings``): nitrate summed
    over water samples is no quantity at all.
    """
    override = roles.aggregation if roles is not None and column and column == roles.measure else None
    if roles is not None and column in roles.flags and override != "sum":
        named = _named_direction(column)
        return MetricSpec(
            name=column,
            aggregation="mean",
            unit="rate",
            increase_is_welcome=named is not False,
            direction_known=named is not None,
        )
    spec = resolve_metric(column, override)
    per_subject = override is None and rows_are_subjects(roles) and not _names_money(column)
    reading = (
        override is None and roles is not None and bool(column) and _total_only(column)
        and _among_readings(column, roles)
    )
    if spec.aggregation == "sum" and (per_subject or reading):
        return resolve_metric(column, "mean")
    return spec
