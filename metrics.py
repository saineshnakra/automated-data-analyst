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

from formatting import is_percentage, normalized_name, percentage_outranks_currency

# A measure where going up is bad. Whole-word: "Cost" and "Refund Amount"
# count, "Cost Savings" does not.
ADVERSE_MEASURE_TOKENS = frozenset({
    "cost", "costs", "expense", "expenses", "spend", "churn", "refund", "refunds",
    "returns", "loss", "losses", "overdue", "delay", "delays", "hours", "tickets",
    "complaints", "defects", "errors", "bounce", "debt", "outstanding",
})
# A measure whose first word settles it as good news whatever follows:
# "Revenue After Returns" is revenue.
FAVOURABLE_MEASURE_TOKENS = frozenset({
    "revenue", "sales", "profit", "income", "margin", "bookings", "mrr", "arr", "gmv",
    "orders", "units", "conversion", "conversions", "retention", "signups", "leads",
})
FAVOURABLE_OVERRIDES = frozenset({"savings", "saved", "recovered", "avoided"})


def increase_is_welcome(measure: str | None) -> bool:
    """Whether a rise in this measure is good news.

    The first word decides when it is itself a known measure -- "Revenue After
    Returns" is revenue, "Refund Amount" is a refund. Otherwise the head noun
    decides: "Customer Acquisition Cost" is a cost. A word in the middle is a
    qualifier and does not flip the reading.
    """
    if not measure:
        return True
    words = normalized_name(measure).split()
    if not words or any(word in FAVOURABLE_OVERRIDES for word in words):
        return True
    first, last = words[0], words[-1]
    if first in FAVOURABLE_MEASURE_TOKENS:
        return True
    if first in ADVERSE_MEASURE_TOKENS:
        return False
    return last not in ADVERSE_MEASURE_TOKENS


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
