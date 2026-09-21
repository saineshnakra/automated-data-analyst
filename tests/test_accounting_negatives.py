"""A refund written the way accounting software writes it must still read as a refund."""

import unittest

import pandas as pd

from analysis import _numeric_from_text, _read_formatted_number


class AccountingNegativeTests(unittest.TestCase):

    def test_currency_outside_the_parentheses(self):
        """"$(100)" is a hundred owed. Refusing it drops the cell entirely."""
        self.assertEqual(_read_formatted_number("$(100)"), -100.0)

    def test_sign_after_the_currency_symbol(self):
        """"$-100" is how a good many exports write a negative amount."""
        self.assertEqual(_read_formatted_number("$-100"), -100.0)

    def test_parentheses_win_over_a_leading_plus(self):
        """Parentheses say owed. A "+" in front does not take that back."""
        self.assertEqual(_read_formatted_number("+(100)"), -100.0)

    def test_markers_in_any_order_still_read_as_owed(self):
        for text in ("-$100", "($100)", "-(100)", "(-100)", "(1,234.50)"):
            with self.subTest(text=text):
                self.assertLess(_read_formatted_number(text), 0)

    def test_a_refund_column_is_not_silently_lost(self):
        """The bug that matters: an unreadable cell becomes NaN and the money vanishes."""
        column = pd.Series(["$1,000.00", "$(250.00)", "$-125.50", "$375.25"])
        read = _numeric_from_text(column)
        self.assertEqual(read.notna().sum(), 4)
        self.assertAlmostEqual(float(read.sum()), 999.75)

    def test_a_doubled_marker_is_still_refused(self):
        for text in ("((100))", "--100", "$$100", "(100", "1,2,3", "12 34"):
            with self.subTest(text=text):
                self.assertIsNone(_read_formatted_number(text))

    def test_ordinary_numbers_are_unchanged(self):
        self.assertEqual(_read_formatted_number("100"), 100.0)
        self.assertEqual(_read_formatted_number("1,234"), 1234.0)
        self.assertEqual(_read_formatted_number("1234,50"), 1234.5)
        self.assertEqual(_read_formatted_number("1 234,50"), 1234.5)
        self.assertEqual(_read_formatted_number("12.5"), 12.5)


if __name__ == "__main__":
    unittest.main()
