from __future__ import annotations

import unittest

import pandas as pd

from ada.schema import IDENTIFIER_NAME_WORDS, detect_roles, is_identifier_name


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


if __name__ == "__main__":
    unittest.main()
