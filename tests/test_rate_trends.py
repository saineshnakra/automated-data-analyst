"""A rate is averaged over a period, and its empty periods stay empty.

Both halves are the same mistake in two directions. Summing a conversion rate
over a month answers a question nobody asked - four records of 20% do not make
80%. And filling a month that has no records with 0.0 states that everybody who
visited failed to convert, when in fact nobody visited.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from aggregation import build_trend
from schema import detect_roles


class RateTrendTests(unittest.TestCase):
    def setUp(self):
        # Four records a month: a sum and a mean differ by a factor of four.
        rows = []
        for month in ("2025-01-31", "2025-02-28", "2025-03-31"):
            rows.extend((month, 0.2) for _ in range(4))
        self.frame = pd.DataFrame(rows, columns=["Date", "Conversion Rate"])
        self.frame["Date"] = pd.to_datetime(self.frame["Date"])

    def test_a_rate_is_the_period_average_not_the_period_total(self):
        series = build_trend(self.frame, detect_roles(self.frame))
        np.testing.assert_allclose(series.frame["Value"].to_numpy(), [0.2, 0.2, 0.2])

    def test_an_amount_is_still_the_period_total(self):
        amounts = self.frame.rename(columns={"Conversion Rate": "Revenue"})
        amounts["Revenue"] = 100.0
        series = build_trend(amounts, detect_roles(amounts))
        np.testing.assert_allclose(series.frame["Value"].to_numpy(), [400.0, 400.0, 400.0])


class EmptyPeriodTests(unittest.TestCase):
    def monthly(self, measure, *, skip):
        """Twelve month-ends with one month absent from the file."""
        months = [m for i, m in enumerate(pd.date_range("2025-01-31", periods=12, freq="ME"))
                  if i != skip]
        value = 0.2 if measure == "Conversion Rate" else 100.0
        frame = pd.DataFrame({"Date": months, measure: [value] * len(months)})
        return build_trend(frame, detect_roles(frame))

    def test_a_missing_day_of_a_rate_is_empty_not_zero(self):
        series = self.monthly("Conversion Rate", skip=4)
        gap = series.frame.loc[series.frame["Period"] == pd.Timestamp("2025-05-01"), "Value"]
        self.assertEqual(len(gap), 1)
        self.assertTrue(np.isnan(float(gap.iloc[0])), "a rate nobody measured is not 0%")
        self.assertFalse(series.filled_as_zero)

    def test_a_missing_day_of_an_amount_is_a_day_that_sold_nothing(self):
        series = self.monthly("Revenue", skip=4)
        gap = series.frame.loc[series.frame["Period"] == pd.Timestamp("2025-05-01"), "Value"]
        self.assertEqual(float(gap.iloc[0]), 0.0)
        self.assertTrue(series.filled_as_zero)

    def test_the_caption_says_which_of_the_two_happened(self):
        self.assertIn("left empty", " ".join(self.monthly("Conversion Rate", skip=4).notes))
        self.assertIn("counted as zero", " ".join(self.monthly("Revenue", skip=4).notes))

    def test_a_period_whose_values_are_all_missing_stays_missing(self):
        frame = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2025-01-31", "2025-02-28", "2025-03-31"]),
                "Revenue": [100.0, np.nan, 100.0],
            }
        )
        series = build_trend(frame, detect_roles(frame))
        february = series.frame.loc[series.frame["Period"] == pd.Timestamp("2025-02-01"), "Value"]
        self.assertTrue(np.isnan(float(february.iloc[0])), "no reading is not a reading of zero")

    def test_two_readings_are_not_spread_into_invented_periods(self):
        frame = pd.DataFrame(
            {"Month": pd.to_datetime(["2024-01-31", "2024-02-29"]), "Revenue": [100.0, 120.0]}
        )
        series = build_trend(frame, detect_roles(frame))
        self.assertEqual(len(series.frame), 2)
        self.assertEqual(series.filled_periods, 0)


if __name__ == "__main__":
    unittest.main()
