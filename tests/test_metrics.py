"""The one place a metric says how it combines and which way is good news.

Nothing reads this yet: it is the groundwork the rate fixes are built on, and
it is separate so the rule itself can be reviewed before the eight surfaces
that will follow it.
"""

from __future__ import annotations

import unittest

from ada.metrics import increase_is_welcome, resolve_metric


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
        for name in ("Revenue", "Units", "Cost Savings", "Revenue After Returns", "Defects Avoided"):
            with self.subTest(name=name):
                self.assertTrue(increase_is_welcome(name))
        for name in ("Cost", "Refund Amount", "Sales Returns", "Customer Acquisition Cost", "Churn Rate"):
            with self.subTest(name=name):
                self.assertFalse(increase_is_welcome(name))

    def test_an_unnamed_metric_is_not_assumed_to_be_bad_news(self):
        self.assertTrue(increase_is_welcome(None))


if __name__ == "__main__":
    unittest.main()
