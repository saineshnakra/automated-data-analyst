"""A report title above the table must not be defeated by a thousands separator."""

import unittest

from file_io import read_tabular_file

TITLE = "Quarterly revenue export\n"
HEADER = "Date,Category,Revenue\n"
ROWS_PLAIN = "2024-01-01,Apparel,1234.50\n2024-01-02,Home,250.00\n2024-02-01,Apparel,99\n"
ROWS_QUOTED = '2024-01-01,Apparel,"1,234.50"\n2024-01-02,Home,250.00\n2024-02-01,Apparel,99\n'


class TitleRowTests(unittest.TestCase):

    def read(self, text):
        return read_tabular_file(text.encode("utf-8"), "export.csv")

    def test_title_above_a_plain_table(self):
        frame = self.read(TITLE + HEADER + ROWS_PLAIN)
        self.assertEqual(list(frame.columns), ["Date", "Category", "Revenue"])
        self.assertEqual(len(frame), 3)

    def test_title_above_a_table_holding_a_thousands_separator(self):
        """The row's quoted "1,234.50" carries a comma that is not a delimiter."""
        frame = self.read(TITLE + HEADER + ROWS_QUOTED)
        self.assertEqual(list(frame.columns), ["Date", "Category", "Revenue"])
        self.assertEqual(len(frame), 3)

    def test_a_quoted_comma_in_the_header_itself(self):
        header = 'Date,"Revenue, net",Category\n'
        rows = '2024-01-01,100,Apparel\n2024-01-02,200,Home\n2024-02-01,300,Apparel\n'
        frame = self.read(TITLE + header + rows)
        self.assertEqual(list(frame.columns), ["Date", "Revenue, net", "Category"])

    def test_a_genuinely_single_column_file_is_left_alone(self):
        """Sentences are not a table, and re-reading them as one would shred them."""
        text = "Notes\nthe first note\nthe second note\nthe third note\n"
        frame = self.read(text)
        self.assertEqual(list(frame.columns), ["Notes"])
        self.assertEqual(len(frame), 3)


if __name__ == "__main__":
    unittest.main()
