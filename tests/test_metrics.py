from __future__ import annotations

import unittest

from ada.metrics import increase_is_welcome, is_level_name, resolve_metric


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


class LevelMeasureTests(unittest.TestCase):
    def test_measured_levels_are_averaged(self):
        for name in ("Water Temp (°C)", "pH", "OD600", "Concentration (mg/L)", "Lactate (mM)",
                     "Nitrate (ppm)", "CSAT Score", "Body Weight"):
            with self.subTest(name=name):
                spec = resolve_metric(name)
                self.assertEqual(spec.unit, "level")
                self.assertEqual(spec.aggregation, "mean")
                self.assertFalse(spec.additive)

    def test_amounts_and_counts_are_still_summed(self):
        for name in ("Revenue", "Units", "Live Cells", "Resolution Hours", "Amount (USD)"):
            with self.subTest(name=name):
                self.assertEqual(resolve_metric(name).aggregation, "sum")

    def test_is_level_name_reads_units(self):
        self.assertTrue(is_level_name("Glucose (mmol/L)"))
        self.assertTrue(is_level_name("Speed (km/h)"))
        self.assertFalse(is_level_name("Distance (m)"))


if __name__ == "__main__":
    unittest.main()
