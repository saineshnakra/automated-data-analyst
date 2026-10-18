"""One cell reading "inf" owned every total built on its column."""

import unittest

import numpy as np
import pandas as pd

from analysis import clean_dataframe


class NonFiniteValuesTests(unittest.TestCase):

    def test_an_infinity_does_not_own_the_total(self):
        frame = pd.DataFrame({
            "Date": pd.to_datetime(["2025-01-05", "2025-02-05", "2025-03-05"]),
            "Revenue": [1.0, np.inf, 3.0],
        })
        cleaned, _ = clean_dataframe(frame)
        self.assertEqual(float(cleaned["Revenue"].sum()), 4.0)
        self.assertTrue(pd.isna(cleaned["Revenue"].iloc[1]))

    def test_negative_infinity_too(self):
        frame = pd.DataFrame({"Revenue": [1.0, -np.inf, 3.0], "Team": list("abc")})
        cleaned, _ = clean_dataframe(frame)
        self.assertEqual(float(cleaned["Revenue"].sum()), 4.0)

    def test_the_text_inf_read_as_a_number_is_caught_too(self):
        """pandas parses the string "inf" into a float infinity."""
        frame = pd.DataFrame({"Revenue": ["1.0", "inf", "3.0"], "Team": list("abc")})
        cleaned, _ = clean_dataframe(frame)
        self.assertEqual(float(cleaned["Revenue"].sum()), 4.0)

    def test_the_reader_is_told(self):
        frame = pd.DataFrame({"Revenue": [1.0, np.inf, 3.0], "Team": list("abc")})
        _, report = clean_dataframe(frame)
        self.assertEqual(report.non_finite_cells, 1)
        self.assertTrue(any("Revenue" in note for note in report.notes), report.notes)

    def test_ordinary_numbers_are_untouched(self):
        frame = pd.DataFrame({"Revenue": [1.0, 2.0, 3.0], "Team": list("abc")})
        cleaned, report = clean_dataframe(frame)
        self.assertEqual(float(cleaned["Revenue"].sum()), 6.0)
        self.assertEqual(report.non_finite_cells, 0)


if __name__ == "__main__":
    unittest.main()
