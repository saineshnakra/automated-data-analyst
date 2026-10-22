from __future__ import annotations

import unittest

import pandas as pd

from ada.schema import (
    IDENTIFIER_NAME_WORDS,
    MEASURE_KEYWORDS,
    detect_roles,
    is_design_name,
    is_entity_name,
    is_identifier_name,
)


class IdentifierNameTests(unittest.TestCase):
    def test_the_last_word_decides(self):
        for name in ("Order No", "Customer ID", "Product Code", "Postal Code", "Store Number"):
            with self.subTest(name=name):
                self.assertTrue(is_identifier_name(name))

    def test_a_name_that_is_only_the_word_counts(self):
        for word in ("ID", "Code", "SKU", "UUID"):
            with self.subTest(word=word):
                self.assertTrue(is_identifier_name(word))

    def test_a_measure_that_merely_contains_the_word_does_not(self):
        # Only the last word counts, so "Identified Revenue" is a measure.
        for name in ("Identified Revenue", "Code Coverage", "Number of Orders", "Keyboard Sales"):
            with self.subTest(name=name):
                self.assertFalse(is_identifier_name(name))

    def test_an_ordinary_measure_is_not_a_key(self):
        for name in ("Revenue", "Units Sold", "Conversion Rate", "Cost"):
            with self.subTest(name=name):
                self.assertFalse(is_identifier_name(name))

    def test_case_and_separators_do_not_matter(self):
        for name in ("order_no", "ORDER NO", "Order-No", "  order   no  "):
            with self.subTest(name=name):
                self.assertTrue(is_identifier_name(name))

    def test_an_empty_name_is_not_a_key(self):
        for name in ("", "   ", "_"):
            with self.subTest(name=name):
                self.assertFalse(is_identifier_name(name))

    def test_the_word_list_is_reachable_for_callers_that_need_it(self):
        self.assertIn("id", IDENTIFIER_NAME_WORDS)
        self.assertIn("sku", IDENTIFIER_NAME_WORDS)
        self.assertNotIn("revenue", IDENTIFIER_NAME_WORDS)


class RoleDeterminismTests(unittest.TestCase):
    def test_the_event_date_wins_whatever_order_the_columns_arrive_in(self):
        months = pd.date_range("2024-01-01", periods=10, freq="MS")
        frame = pd.DataFrame(
            {"Order Date": months, "Ship Date": months + pd.Timedelta(days=3), "Revenue": range(10)}
        )

        self.assertEqual(detect_roles(frame).date, "Order Date")
        self.assertEqual(detect_roles(frame[["Ship Date", "Order Date", "Revenue"]]).date, "Order Date")


class DesignColumnTests(unittest.TestCase):
    def test_a_replicate_number_is_not_picked_as_the_measure(self):
        frame = pd.DataFrame({
            "Treatment": ["Control", "Drug"] * 6,
            "Replicate": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
            "Yield": [4.1, 3.2, 4.0, 3.1, 4.3, 3.3, 3.9, 3.0, 4.2, 3.4, 4.0, 3.2],
        })
        self.assertEqual(detect_roles(frame).measure, "Yield")
        self.assertTrue(is_design_name("Replicate"))
        self.assertTrue(is_design_name("Plate"))
        self.assertFalse(is_design_name("Run Date"))


class PlateLayoutTests(unittest.TestCase):
    def setUp(self):
        rows = [letter for letter in "ABCDEFGH" for _ in range(12)]
        self.frame = pd.DataFrame({
            "Row": rows,
            "Column": list(range(1, 13)) * 8,
            "Sample": ["Drug X", "Drug Y", "Control", "Blank"] * 24,
            "Abs 450nm": [round(0.1 + 0.01 * index, 3) for index in range(96)],
        })

    def test_a_plate_row_is_not_the_grouping_when_a_sample_column_exists(self):
        self.assertEqual(detect_roles(self.frame).dimension, "Sample")

    def test_a_plate_row_is_still_used_when_nothing_else_qualifies(self):
        self.assertEqual(detect_roles(self.frame.drop(columns=["Sample"])).dimension, "Row")

    def test_layout_names_are_design_names(self):
        for name in ("Row", "Column", "col", "Well Position", "Plate Row"):
            with self.subTest(name=name):
                self.assertTrue(is_design_name(name))

    def test_a_position_on_its_own_is_not_a_layout(self):
        self.assertFalse(is_design_name("Position"))
        self.assertFalse(is_design_name("Player Position"))


class ResearchMeasureTests(unittest.TestCase):
    def test_an_outcome_beats_the_temperature_it_was_measured_at(self):
        # Readings below the detection limit leave the outcome less complete.
        frame = pd.DataFrame({
            "Site": ["River A", "River B"] * 6,
            "Concentration (mg/L)": [5.0, 0.7, None, 8.2, 4.1, None] * 2,
            "Temp (°C)": [11.7, 12.1, 12.5, 12.0, 11.9, 12.3] * 2,
        })
        self.assertEqual(detect_roles(frame).measure, "Concentration (mg/L)")

    def test_a_count_beats_air_temperature_even_without_fractions(self):
        frame = pd.DataFrame({
            "count": [0, 3, 7, 2, 5, 1] * 2,
            "air_temp_c": [10.7, 11.2, 15.0, 10.8, 13.1, 12.4] * 2,
            "humidity": [55.5, 60.1, 48.2, 52.0, 58.3, 61.7] * 2,
        })
        self.assertEqual(detect_roles(frame).measure, "count")

    def test_a_covariate_is_still_the_measure_when_it_is_all_there_is(self):
        frame = pd.DataFrame({"Station": ["N", "S"] * 6, "Wind Speed": [3.5, 4.1, 2.2] * 4})
        self.assertEqual(detect_roles(frame).measure, "Wind Speed")

    def test_coordinates_are_never_the_measure(self):
        frame = pd.DataFrame({
            "Latitude": [51.501, 51.502, 51.503] * 4,
            "lon": [-0.141, -0.142, -0.143] * 4,
            "Depth": [3, 5, 8] * 4,
        })
        self.assertEqual(detect_roles(frame).measure, "Depth")

    def test_research_outcome_words_score_below_money(self):
        for word in ("count", "abundance", "concentration", "yield"):
            with self.subTest(word=word):
                self.assertGreaterEqual(MEASURE_KEYWORDS[word], 5)
                self.assertLess(MEASURE_KEYWORDS[word], MEASURE_KEYWORDS["price"])


class EntityNameTests(unittest.TestCase):
    def test_the_unit_a_row_belongs_to_is_an_entity(self):
        for name in ("subject", "Participant", "Patient ID", "mouse", "Donor", "specimen_no"):
            with self.subTest(name=name):
                self.assertTrue(is_entity_name(name))

    def test_a_quantity_measured_on_the_unit_is_not(self):
        for name in ("Animal Weight", "Patient Age", "score", "Heart Rate"):
            with self.subTest(name=name):
                self.assertFalse(is_entity_name(name))


class IdentifierNamePunctuationTests(unittest.TestCase):
    def test_key_names_with_punctuation_or_symbols(self):
        for name in ("Phone No.", "Account #", "Zipcode", "Postcode", "Telephone"):
            with self.subTest(name=name):
                self.assertTrue(is_identifier_name(name))
        for name in ("Width", "Humidity", "Paid Amount"):
            with self.subTest(name=name):
                self.assertFalse(is_identifier_name(name))


if __name__ == "__main__":
    unittest.main()
