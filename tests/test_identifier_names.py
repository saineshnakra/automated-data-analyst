"""One rule for "this column's name says it holds a key".

Three places need the answer and they must not disagree: the reader, which
keeps "Order No" as text rather than parsing it into a float; the cleaner,
which must not turn it back into a number afterwards; and the parser, which
must not offer it as a measure to average. When they each carried their own
list, a column was a key to one of them and a number to another - which is how
"Invoice No" ended up rendered as 1.00047e+11.

Nothing else changes here: this is the rule on its own, before the three
callers are pointed at it.
"""

from __future__ import annotations

import unittest

from schema import IDENTIFIER_NAME_WORDS, is_identifier_name


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
        # The head noun is what the column holds. "Identified Revenue" is
        # revenue, and a substring test would have made it a key.
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
        # The parser matches identifier-ish words in a question, so it reads
        # the same set rather than restating it.
        self.assertIn("id", IDENTIFIER_NAME_WORDS)
        self.assertIn("sku", IDENTIFIER_NAME_WORDS)
        self.assertNotIn("revenue", IDENTIFIER_NAME_WORDS)


if __name__ == "__main__":
    unittest.main()
