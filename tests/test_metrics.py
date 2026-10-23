from __future__ import annotations

import unittest

from ada.metrics import increase_is_welcome, is_additive_name, is_level_name, metric_for, resolve_metric
from ada.schema import ColumnRoles


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

    def test_only_readings_whose_scale_goes_below_zero_forecast_past_it(self):
        for name in ("Revenue", "Churn Rate", None, "Rainfall (mm)", "Concentration (mg/L)", "CO2"):
            with self.subTest(name=name):
                self.assertTrue(resolve_metric(name).floors_at_zero)
        for name in ("Water Temp (°C)", "air_temp", "Elevation (m)", "Reading (°F)"):
            with self.subTest(name=name):
                self.assertFalse(resolve_metric(name).floors_at_zero)

    def test_direction_of_good_is_read_from_the_whole_name(self):
        for name in ("Revenue", "Units", "Cost Savings", "Revenue After Returns", "Defects Avoided",
                     "Revenue Net of Returns", "Revenue Without Refunds", "Net Revenue"):
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
                     "Nitrate (ppm)", "CSAT Score", "Cell Density (cells/mL)"):
            with self.subTest(name=name):
                spec = resolve_metric(name)
                self.assertEqual(spec.unit, "level")
                self.assertEqual(spec.aggregation, "mean")
                self.assertFalse(spec.additive)

    def test_amounts_and_counts_are_still_summed(self):
        for name in ("Revenue", "Units", "Live Cells", "Amount (USD)", "Margin Amount",
                     "Revenue ($M)", "Population (M)", "Revenue (USD/month)", "Weekly Sales (units/wk)",
                     "Orders", "Patient Visits", "Colony Count", "Individuals", "Spending_USD"):
            with self.subTest(name=name):
                spec = resolve_metric(name)
                self.assertEqual(spec.aggregation, "sum")
                self.assertEqual(spec.unit, "amount")

    def test_a_name_that_is_neither_money_nor_a_count_is_averaged(self):
        # Averaging an amount still gives a sensible figure per row; summing a
        # reading (ppm, mpg, a BOLD signal) gives none, so the unknown case averages.
        for name in ("CO2", "signal", "mpg", "pulse", "GFP (RFU)", "Abs 450nm",
                     "Resolution Hours", "Reopened", "Shipping Weight", "Discount"):
            with self.subTest(name=name):
                spec = resolve_metric(name)
                self.assertEqual(spec.aggregation, "mean")
                self.assertEqual(spec.unit, "level")
                self.assertFalse(spec.additive)

    def test_a_per_unit_price_is_averaged_though_it_is_money(self):
        for name in ("Price", "Unit Price", "Close Price (USD)", "Price_USD", "Revenue per User",
                     "Spending per Capita", "Cost per Click"):
            with self.subTest(name=name):
                self.assertEqual(resolve_metric(name).aggregation, "mean")
                self.assertFalse(is_additive_name(name))
        self.assertEqual(resolve_metric("Total Price").aggregation, "sum")

    def test_additive_words_match_whole_words_only(self):
        self.assertTrue(is_additive_name("Cell Count"))
        self.assertFalse(is_additive_name("Discount"))
        self.assertFalse(is_additive_name("Unitary Charge"))

    def test_is_level_name_reads_units(self):
        self.assertTrue(is_level_name("Glucose (mmol/L)"))
        self.assertTrue(is_level_name("Speed (km/h)"))
        self.assertFalse(is_level_name("Distance (m)"))


class DirectionTests(unittest.TestCase):
    def test_a_level_whose_name_says_nothing_has_no_good_direction(self):
        for name in ("CO2", "Water Temp (°C)", "Rainfall", "pH"):
            with self.subTest(name=name):
                self.assertFalse(resolve_metric(name).direction_known)

    def test_a_level_with_a_direction_word_keeps_it(self):
        hours = resolve_metric("Resolution Hours")
        self.assertTrue(hours.direction_known)
        self.assertFalse(hours.increase_is_welcome)

    def test_amounts_and_rates_always_have_a_direction(self):
        for name in ("Revenue", "Units", "Churn Rate", "Viability %"):
            with self.subTest(name=name):
                self.assertTrue(resolve_metric(name).direction_known)
        self.assertTrue(resolve_metric(None).direction_known)


class AggregationOverrideTests(unittest.TestCase):
    def roles(self, measure, aggregation):
        return ColumnRoles(
            date=None, measure=measure, dimension=None, identifier=None,
            numeric=("Rainfall (mm)", "Revenue"), dimensions=(), aggregation=aggregation,
        )

    def test_summing_a_level_makes_it_an_amount(self):
        spec = resolve_metric("Rainfall (mm)", "sum")
        self.assertEqual((spec.aggregation, spec.unit), ("sum", "amount"))
        self.assertTrue(spec.additive)
        self.assertTrue(spec.direction_known)

    def test_averaging_an_amount_makes_it_a_level(self):
        spec = resolve_metric("Revenue", "mean")
        self.assertEqual((spec.aggregation, spec.unit), ("mean", "level"))
        self.assertTrue(spec.floors_at_zero)

    def test_an_override_that_agrees_with_the_name_changes_nothing(self):
        self.assertEqual(resolve_metric("Churn Rate", "mean"), resolve_metric("Churn Rate"))
        self.assertEqual(resolve_metric("Revenue", "sum"), resolve_metric("Revenue"))

    def test_the_override_applies_to_the_measure_only(self):
        roles = self.roles("Rainfall (mm)", "sum")
        self.assertEqual(metric_for("Rainfall (mm)", roles).aggregation, "sum")
        self.assertEqual(metric_for("Revenue", roles).aggregation, "sum")
        self.assertEqual(metric_for("Revenue", self.roles("Rainfall (mm)", "mean")).aggregation, "sum")
        self.assertEqual(metric_for(None, roles).aggregation, "count")

    def test_without_roles_the_name_decides(self):
        self.assertEqual(metric_for("Rainfall (mm)"), resolve_metric("Rainfall (mm)"))


class FlagMeasureTests(unittest.TestCase):
    def roles(self, aggregation=None):
        return ColumnRoles(
            date=None, measure="Reopened Count", dimension=None, identifier=None,
            numeric=("Reopened Count",), dimensions=("Reopened Count",), aggregation=aggregation,
            factors=("Reopened Count",), flags=("Reopened Count",),
        )

    def test_a_flag_averages_to_a_share_whatever_its_name(self):
        spec = metric_for("Reopened Count", self.roles())
        self.assertEqual((spec.aggregation, spec.unit), ("mean", "rate"))
        self.assertFalse(spec.direction_known)

    def test_an_override_can_still_count_the_ones(self):
        self.assertEqual(metric_for("Reopened Count", self.roles("sum")).aggregation, "sum")


class PerSubjectTests(unittest.TestCase):
    def roles(self, identifier, aggregation=None):
        return ColumnRoles(
            date=None, measure="CorrectTotal", dimension="Condition", identifier=identifier,
            numeric=("CorrectTotal", "Spend"), dimensions=("Condition",), aggregation=aggregation,
        )

    def test_a_total_per_participant_is_averaged_not_summed(self):
        spec = metric_for("CorrectTotal", self.roles("ParticipantId"))

        self.assertEqual(spec.aggregation, "mean")
        self.assertFalse(spec.additive)

    def test_money_per_subject_still_adds_up(self):
        self.assertEqual(metric_for("Spend", self.roles("Patient ID")).aggregation, "sum")

    def test_amounts_on_order_rows_still_add_up(self):
        self.assertEqual(metric_for("CorrectTotal", self.roles("Order ID")).aggregation, "sum")
        self.assertEqual(metric_for("CorrectTotal", self.roles(None)).aggregation, "sum")

    def test_an_override_to_sum_is_kept(self):
        self.assertEqual(metric_for("CorrectTotal", self.roles("ParticipantId", "sum")).aggregation, "sum")


class LabTotalTests(unittest.TestCase):
    def roles(self, *numeric):
        return ColumnRoles(
            date=None, measure=numeric[0], dimension=None, identifier=None, numeric=numeric, dimensions=(),
        )

    def test_a_total_among_lab_readings_is_averaged(self):
        spec = metric_for("nitrates_total", self.roles("nitrates_total", "calcium", "chloride"))

        self.assertEqual(spec.aggregation, "mean")

    def test_a_total_beside_money_or_counts_still_adds_up(self):
        self.assertEqual(metric_for("Total", self.roles("Total", "Quantity", "Discount")).aggregation, "sum")
        self.assertEqual(metric_for("Order Total", self.roles("Order Total")).aggregation, "sum")

    def test_a_total_named_for_a_count_still_adds_up(self):
        spec = metric_for("Total Cases", self.roles("Total Cases", "temperature", "humidity"))

        self.assertEqual(spec.aggregation, "sum")


if __name__ == "__main__":
    unittest.main()
