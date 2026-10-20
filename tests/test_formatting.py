from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from ada.formatting import (
    format_change,
    format_number,
    format_rate_change,
    format_values_apart,
    group_label,
    is_money,
    list_names,
    normalized_name,
    rate_scale,
    sentence_case,
    with_aggregate,
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

    def test_sentence_case_keeps_chemical_symbols_and_short_codes(self):
        for name in ("Ca", "Na", "Mg", "CO", "NO2", "PM2.5", "Gp"):
            with self.subTest(name=name):
                self.assertEqual(sentence_case(name), name)
        self.assertEqual(sentence_case("Average Ca"), "average Ca")
        self.assertEqual(sentence_case("Co2 Level"), "Co2 level")

    def test_readings_keep_their_digits(self):
        # "1.1K" would hide that a heart rate of 1,111 is impossible.
        self.assertEqual(format_number(1_111, "Heart rate"), "1,111")
        self.assertEqual(format_number(1_944.66, "Looking Time"), "1,944.7")
        self.assertEqual(format_number(55_034.3, "result"), "55,034")
        self.assertEqual(format_number(101_325, "Pressure (Pa)"), "101,325")

    def test_only_money_and_large_counts_are_compacted(self):
        self.assertEqual(format_number(1_250_000, "Revenue"), "$1.2M")
        self.assertEqual(format_number(12_000, "Units"), "12.0K")
        self.assertEqual(format_number(9_999, "Units"), "9,999")
        self.assertEqual(format_number(212_019), "212.0K")
        # A summed reading with fractions is not a count of anything.
        self.assertEqual(
            format_number(15_489_238.4, "Total Rain", column_values=pd.Series([0.2, 184.7])), "15,489,238"
        )
        self.assertEqual(format_number(12_000, "Units", compact=False), "12,000")

    def test_very_small_and_very_large_readings_use_scientific_notation(self):
        self.assertEqual(format_number(3.624e11, "1001"), "3.624e+11")
        self.assertEqual(format_number(-1.098e10, "1001"), "-1.098e+10")
        self.assertEqual(format_number(0.000123, "Concentration"), "1.23e-04")
        self.assertEqual(format_number(0, "Concentration"), "0")
        # Money keeps its scale words.
        self.assertEqual(format_number(1.5e12, "Revenue"), "$1.5T")


class MoneyTests(unittest.TestCase):
    def test_a_coded_band_is_not_money(self):
        bands = pd.Series([1.0, 2.0, 5.0, 14.0, 15.0])

        self.assertFalse(is_money("income", bands))
        self.assertEqual(format_number(8.11, "income", column_values=bands), "8.11")

    def test_amounts_are_money(self):
        self.assertTrue(is_money("income", pd.Series([1_250.0, 30_000.0, 52_000.0])))
        self.assertTrue(is_money("Price", pd.Series([2.5, 4.99])))
        self.assertEqual(format_number(8.11, "Price", column_values=pd.Series([2.5, 8.11])), "$8.11")

    def test_without_the_column_the_name_decides(self):
        self.assertTrue(is_money("income"))
        self.assertFalse(is_money("Heart rate", pd.Series([60.0, 80.0])))


class LabelTests(unittest.TestCase):
    def test_an_aggregate_word_is_not_repeated(self):
        self.assertEqual(with_aggregate("Total", "Total Rain"), "Total Rain")
        self.assertEqual(with_aggregate("Total", "CorrectTotal"), "CorrectTotal")
        self.assertEqual(with_aggregate("Average", "Mean Temp"), "Mean Temp")
        self.assertEqual(with_aggregate("Maximum", "Max Wind Speed"), "Max Wind Speed")

    def test_a_different_aggregate_is_kept(self):
        self.assertEqual(with_aggregate("Total", "Revenue"), "Total Revenue")
        self.assertEqual(with_aggregate("Average", "Total Rain"), "Average Total Rain")

    def test_names_are_listed_for_a_sentence(self):
        self.assertEqual(list_names(["A"]), "A")
        self.assertEqual(list_names(["A", "B"]), "A and B")
        self.assertEqual(list_names(["A", "B", "C"]), "A, B and C")
        self.assertEqual(list_names(["A", "B", "C", "D"]), "A, B, C and 1 other")



class ChangeEndsTests(unittest.TestCase):
    def test_ends_that_round_alike_are_written_out(self):
        self.assertEqual(format_number(3_004.25), format_number(3_004.21))
        self.assertEqual(format_values_apart(3_004.25, 3_004.21), ("3,004.25", "3,004.21"))
        self.assertEqual(format_values_apart(15_700, 15_740, column="Units"), ("15,700", "15,740"))
        self.assertEqual(
            format_values_apart(2_700_100, 2_700_400, column="Revenue"), ("$2,700,100.00", "$2,700,400.00")
        )
        self.assertEqual(format_values_apart(0.2501, 0.2504, column="Conversion Rate"), ("25.01%", "25.04%"))
        self.assertEqual(format_values_apart(12.3401, 12.3412, column="pH"), ("12.340", "12.341"))

    def test_readings_in_scientific_notation_are_kept_apart(self):
        self.assertEqual(
            format_values_apart(3.62412e11, 3.62438e11, column="1001"), ("3.6241e+11", "3.6244e+11")
        )

    def test_ends_that_already_differ_keep_the_usual_rendering(self):
        self.assertEqual(format_values_apart(36_800, 41_300, column="Revenue"), ("$36.8K", "$41.3K"))
        self.assertEqual(format_values_apart(0.0012, 0.0015), ("0.0012", "0.0015"))

    def test_a_value_just_outside_a_range_never_prints_as_its_bound(self):
        self.assertEqual(
            format_values_apart(13.336, 13.341, 15.5, column="Resolution Hours"),
            ("13.336", "13.341", "15.500"),
        )

    def test_equal_ends_print_alike(self):
        self.assertEqual(format_values_apart(5.0, 5.0, column="pH"), ("5", "5"))
        self.assertEqual(format_values_apart(12_000.0, 12_000.0, column="Units"), ("12.0K", "12.0K"))


if __name__ == "__main__":
    unittest.main()


class GroupLabelTests(unittest.TestCase):
    def test_a_numeric_group_is_named_with_its_column(self):
        self.assertEqual(group_label("Status", 2), "Status 2")
        self.assertEqual(group_label("Dose", 2.5), "Dose 2.5")
        self.assertEqual(group_label("Dose", np.float64(10.0)), "Dose 10")
        self.assertEqual(group_label("Treated", True), "Treated 1")

    def test_a_text_group_is_left_as_it_is(self):
        self.assertEqual(group_label("Site", "North Marsh"), "North Marsh")
