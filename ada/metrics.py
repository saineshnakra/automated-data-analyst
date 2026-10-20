"""Resolve how a measure aggregates and whether a rise in it is good news.

Every surface asks here instead of re-reading the column name, so they agree that
a rate is averaged, not summed, and has no meaningful shares or waterfall.
"""

from __future__ import annotations

from dataclasses import dataclass

from .formatting import is_percentage, normalized_name, percentage_outranks_currency

# Words that make a rise bad news when they lead or end a measure's name.
ADVERSE_MEASURE_TOKENS = frozenset({
    "cost", "costs", "expense", "expenses", "spend", "churn", "refund", "refunds",
    "returns", "loss", "losses", "overdue", "delay", "delays", "hours", "tickets",
    "complaints", "defects", "errors", "bounce", "debt", "outstanding",
})
# Any of these anywhere in the name makes a rise good news: "Cost Savings".
FAVOURABLE_OVERRIDES = frozenset({"savings", "saved", "recovered", "avoided"})
# Words that start a qualifier. "Revenue After Returns" is judged on "Revenue".
QUALIFIER_WORDS = frozenset({"after", "before", "less", "minus", "excluding", "excl"})


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
        if word in QUALIFIER_WORDS and position > 0:
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
    unit: str  # amount | rate | count
    increase_is_welcome: bool

    @property
    def additive(self) -> bool:
        """Whether parts add up to the whole.

        Shares of a total, concentration and segment waterfalls are valid only
        when this is true; it is false for a rate, which is averaged.
        """
        return self.aggregation != "mean"

    @property
    def combines_as(self) -> str:
        return {"sum": "total", "mean": "average", "count": "count"}[self.aggregation]


def resolve_metric(name: str | None) -> MetricSpec:
    """Decide a measure's aggregation, unit and direction from its column name.

    Percentage-named columns are averaged as rates, others summed as amounts;
    no name means a row count.
    """
    if not name:
        return MetricSpec(name=None, aggregation="count", unit="count", increase_is_welcome=True)
    if is_percentage(name) and percentage_outranks_currency(name):
        return MetricSpec(
            name=name, aggregation="mean", unit="rate", increase_is_welcome=increase_is_welcome(name)
        )
    return MetricSpec(
        name=name, aggregation="sum", unit="amount", increase_is_welcome=increase_is_welcome(name)
    )
