"""What a measure is, decided once and consumed everywhere.

The rate change fixed the headline, then the trend, then chat, and each
surface disagreed with the next about whether a rate adds up. It does not:
two months of conversion rate do not sum to anything, a segment's share of
"total conversion rate" is not a share of anything, and a waterfall of
changes in segment averages does not reconcile to the change in the overall
average. So the decision is made here, once, and every surface asks for it
rather than inferring it again from the column name.
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

        Only then do shares of a total, a concentration index or a waterfall
        of segment changes mean anything. A rate is an average of averages.
        """
        return self.aggregation != "mean"

    @property
    def combines_as(self) -> str:
        return {"sum": "total", "mean": "average", "count": "count"}[self.aggregation]


def resolve_metric(name: str | None) -> MetricSpec:
    """The one place a column name becomes a decision about arithmetic."""
    if not name:
        return MetricSpec(name=None, aggregation="count", unit="count", increase_is_welcome=True)
    if is_percentage(name) and percentage_outranks_currency(name):
        return MetricSpec(
            name=name, aggregation="mean", unit="rate", increase_is_welcome=increase_is_welcome(name)
        )
    return MetricSpec(
        name=name, aggregation="sum", unit="amount", increase_is_welcome=increase_is_welcome(name)
    )
