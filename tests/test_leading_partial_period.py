"""A first period the data only starts part-way into must not become the baseline."""

import unittest

import pandas as pd

from aggregation import build_trend
from schema import ColumnRoles


def roles(**kw):
    base = dict(date=None, measure=None, dimension=None, identifier=None, numeric=(), dimensions=())
    base.update(kw)
    return ColumnRoles(**base)


def weekly_frame_starting_on(start: str, weeks: int = 12) -> pd.DataFrame:
    """Daily rows of a steady 100 a day, beginning mid-week."""
    days = pd.date_range(start, periods=weeks * 7, freq="D")
    return pd.DataFrame({"Date": days, "Revenue": [100.0] * len(days)})


class LeadingPartialPeriodTests(unittest.TestCase):

    def setUp(self):
        self.roles = roles(date="Date", measure="Revenue", numeric=("Revenue",))

    def test_a_short_first_week_is_excluded(self):
        """Starting on a Saturday leaves two days of trade in week one."""
        frame = weekly_frame_starting_on("2024-06-01")  # a Saturday
        series = build_trend(frame, self.roles, frequency="W")
        first = float(series.frame.iloc[0]["Value"])
        self.assertEqual(first, 700.0, "the two-day opening week is still the baseline")

    def test_the_reader_is_told_it_was_excluded(self):
        frame = weekly_frame_starting_on("2024-06-01")
        series = build_trend(frame, self.roles, frequency="W")
        self.assertTrue(
            any("2 of 7 days" in note for note in series.notes),
            f"no note explains the dropped opening period: {series.notes}",
        )

    def test_a_complete_first_period_is_left_alone(self):
        """A file that begins on a Monday has a whole first week."""
        frame = weekly_frame_starting_on("2024-06-03")  # a Monday
        series = build_trend(frame, self.roles, frequency="W")
        self.assertEqual(float(series.frame.iloc[0]["Value"]), 700.0)
        self.assertEqual(len(series.frame), 12)

    def test_monthly_data_starting_on_the_first_is_untouched(self):
        days = pd.date_range("2024-01-01", periods=365, freq="D")
        frame = pd.DataFrame({"Date": days, "Revenue": [100.0] * len(days)})
        series = build_trend(frame, self.roles, frequency="M")
        self.assertEqual(float(series.frame.iloc[0]["Value"]), 3100.0)

    def test_too_few_periods_to_judge_leaves_it_alone(self):
        """With no run of periods to compare against, nothing is dropped."""
        frame = weekly_frame_starting_on("2024-06-01", weeks=2)
        series = build_trend(frame, self.roles, frequency="W")
        self.assertEqual(float(series.frame.iloc[0]["Value"]), 200.0)


if __name__ == "__main__":
    unittest.main()
