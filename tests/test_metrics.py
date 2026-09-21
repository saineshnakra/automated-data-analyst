"""The one place a metric says how it combines and which way is good news.

Nothing reads this yet: it is the groundwork the rate fixes are built on, and
it is separate so the rule itself can be reviewed before the eight surfaces
that will follow it.
"""

from __future__ import annotations

import unittest

from metrics import increase_is_welcome, resolve_metric


class MetricSpecTests(unittest.TestCase):
    def test_a_rate_averages_and_an_amount_totals(self):
        rate = resolve_metric("Conversion Rate")
        self.assertEqual(rate.aggregation, "mean")
        self.assertEqual(rate.unit, "rate")
        self.assertFalse(rate.additive)
        self.assertEqual(rate.combines_as, "average")

        amount = resolve_metric("Revenue")
        self.assertEqual(amount.aggregation, "sum")
        self.assertEqual(amount.unit, "amount")
        self.assertTrue(amount.additive)
        self.assertEqual(amount.combines_as, "total")

    def test_no_measure_counts_rows(self):
        counted = resolve_metric(None)
        self.assertEqual(counted.aggregation, "count")
        self.assertEqual(counted.combines_as, "count")

    def test_direction_of_good_is_read_from_the_whole_name(self):
        # "Cost Savings" is not a cost, and "Revenue After Returns" is not a
        # return. A substring test on either would get both backwards.
        self.assertTrue(increase_is_welcome("Revenue After Returns"))
        self.assertTrue(increase_is_welcome("Cost Savings"))
        self.assertTrue(increase_is_welcome("Units"))
        self.assertFalse(increase_is_welcome("Cost"))
        self.assertFalse(increase_is_welcome("Refund Amount"))

    def test_an_unnamed_metric_is_not_assumed_to_be_bad_news(self):
        self.assertTrue(increase_is_welcome(None))


if __name__ == "__main__":
    unittest.main()
