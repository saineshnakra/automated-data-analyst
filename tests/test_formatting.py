from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from ada.formatting import (
    format_change,
    format_number,
    format_rate_change,
    normalized_name,
    rate_scale,
    sentence_case,
)


class RateScaleTests(unittest.TestCase):
    def test_a_column_of_fractions_is_read_as_fractions(self):
        self.assertEqual(rate_scale(pd.Series([0.1, 0.25, 0.4])), "fraction")

    def test_a_column_of_points_is_read_as_points(self):
        # Read as fractions these would be 1,250% and more.
        self.assertEqual(rate_scale(pd.Series([12.5, 30.0, 47.2])), "points")

    def test_no_column_falls_back_to_the_value_itself(self):
        self.assertEqual(rate_scale(None, 0.25), "fraction")
        self.assertEqual(rate_scale(None, 25.0), "points")


class RateChangeTests(unittest.TestCase):
    def test_a_fraction_column_scales_up_to_points(self):
        rates = pd.Series([0.2, 0.4])
        self.assertEqual(format_rate_change(0.2, rates), "20.0 percentage points")

    def test_a_points_column_is_already_points(self):
        rates = pd.Series([20.0, 40.0])
        self.assertEqual(format_rate_change(20.0, rates), "20.0 percentage points")

    def test_the_size_is_reported_without_its_sign(self):
        # The caller words the direction ("fell by"), so only the magnitude is returned.
        rates = pd.Series([0.2, 0.4])
        self.assertEqual(format_rate_change(-0.03, rates), "3.0 percentage points")

    def test_a_relative_reading_never_appears(self):
        # 0.2 to 0.4 is +100% relative; the output must be in points only.
        self.assertNotIn("100", format_rate_change(0.2, pd.Series([0.2, 0.4])))

    def test_an_unmeasurable_change_says_so_rather_than_printing_nan(self):
        for bad in (float("nan"), np.inf, -np.inf):
            with self.subTest(delta=bad):
                self.assertEqual(format_rate_change(bad, None), "an unmeasurable amount")


class FormatChangeTests(unittest.TestCase):
    def test_each_unit_changes_in_its_own_terms(self):
        self.assertEqual(format_change(200, 250, "Revenue", "amount"), "25.0%")
        self.assertEqual(format_change(0.2, 0.25, "Churn Rate", "rate"), "5.0 percentage points")
        self.assertEqual(format_change(15.2, 13.8, "Water Temp (°C)", "level", signed=True), "-1.40")

    def test_a_relative_change_from_zero_is_undefined(self):
        self.assertIsNone(format_change(0, 10, "Revenue", "amount"))
        self.assertEqual(format_change(0, 1.5, "pH", "level"), "1.50")

    def test_non_finite_values_have_no_change(self):
        self.assertIsNone(format_change(np.nan, 1.0, "pH", "level"))


class NumberDisplayTests(unittest.TestCase):
    def test_small_values_keep_three_significant_figures(self):
        self.assertEqual(format_number(0.0012, "Concentration"), "0.0012")
        self.assertEqual(format_number(0.123456, "OD600"), "0.123")
        self.assertEqual(format_number(1.4, "OD600"), "1.40")

    def test_camel_case_splits_words_but_not_units(self):
        self.assertEqual(normalized_name("netRevenue"), "net revenue")
        self.assertEqual(normalized_name("customerID"), "customer id")
        self.assertEqual(normalized_name("pH"), "ph")

    def test_sentence_case_keeps_acronyms_and_units(self):
        self.assertEqual(sentence_case("Water Temp (°C)"), "water temp (°C)")
        self.assertEqual(sentence_case("SKU"), "SKU")
        self.assertEqual(sentence_case("pH"), "pH")


if __name__ == "__main__":
    unittest.main()
