"""A change in a rate is a number of percentage points.

20% to 40% is twenty points, or a hundred percent relative. Both readings are
defensible and they disagree by a factor of five, so "increased by 20%" is
ambiguous at best and wrong on the relative reading. The unit gets spelt out.

The other half is scale. Some files store a rate as 0.25 and some as 25, and
the same delta means different things in each: a jump of 0.2 is twenty points
in the first file and a fifth of a point in the second. rate_scale decides once
per column rather than once per value, so a column cannot be read one way in
the headline and another in the table.

Nothing calls these yet - the surfaces that will are their own pull requests.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from formatting import format_rate_change, rate_scale


class RateScaleTests(unittest.TestCase):
    def test_a_column_of_fractions_is_read_as_fractions(self):
        self.assertEqual(rate_scale(pd.Series([0.1, 0.25, 0.4])), "fraction")

    def test_a_column_of_points_is_read_as_points(self):
        # 12.5 as a fraction would be 1,250% - which no conversion rate is.
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
        # The direction is the sentence's job; this is the magnitude, so a
        # caller writing "fell by ..." does not produce "fell by -3.0".
        rates = pd.Series([0.2, 0.4])
        self.assertEqual(format_rate_change(-0.03, rates), "3.0 percentage points")

    def test_a_relative_reading_never_appears(self):
        # 20 -> 40 is a hundred percent relative. That number must not be what
        # comes out of a function about points.
        self.assertNotIn("100", format_rate_change(0.2, pd.Series([0.2, 0.4])))

    def test_an_unmeasurable_change_says_so_rather_than_printing_nan(self):
        for bad in (float("nan"), np.inf, -np.inf):
            with self.subTest(delta=bad):
                self.assertEqual(format_rate_change(bad, None), "an unmeasurable amount")


if __name__ == "__main__":
    unittest.main()
