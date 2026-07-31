import unittest

import numpy as np
import pandas as pd

from business_insights import build_trend, preferred_frequency, trend_frame
from schema import detect_roles


def daily_sales(start, end, *, amount=100.0, seed=5):
    """One row per day between the two dates, so every period is well covered."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    return pd.DataFrame(
        {
            "Date": dates,
            "Revenue": amount + rng.normal(0, 4, len(dates)),
            "Region": ["West", "East"] * (len(dates) // 2) + ["West"] * (len(dates) % 2),
        }
    )


class GrainSelectionTests(unittest.TestCase):
    def test_monthly_readings_are_not_charted_as_weeks(self):
        dates = pd.to_datetime(["2025-01-15", "2025-02-15", "2025-03-15", "2025-04-15", "2025-05-15"])

        self.assertEqual(preferred_frequency(pd.Series(dates)), "M")

    def test_dense_daily_data_over_a_short_span_stays_weekly(self):
        frame = daily_sales("2025-01-01", "2025-03-15")

        self.assertEqual(preferred_frequency(frame["Date"]), "W")

    def test_two_years_of_daily_data_rolls_up_to_months(self):
        frame = daily_sales("2024-01-01", "2025-12-31")

        self.assertEqual(preferred_frequency(frame["Date"]), "M")


class EmptyPeriodTests(unittest.TestCase):
    def test_a_month_with_no_rows_counts_as_zero(self):
        frame = pd.concat(
            [daily_sales("2024-01-01", "2024-04-30"), daily_sales("2024-06-01", "2024-12-31")]
        )
        roles = detect_roles(frame)

        series = build_trend(frame, roles)

        self.assertEqual(series.frequency, "M")
        self.assertEqual(series.filled_periods, 1)
        self.assertEqual(len(series.frame), 12)
        may = series.frame[series.frame["Period"] == pd.Timestamp("2024-05-01")]
        self.assertEqual(float(may["Value"].iloc[0]), 0.0)

    def test_the_timeline_stays_evenly_spaced_after_filling(self):
        frame = pd.concat(
            [daily_sales("2024-01-01", "2024-03-31"), daily_sales("2024-07-01", "2024-12-31")]
        )

        series = build_trend(frame, detect_roles(frame))
        gaps = series.frame["Period"].diff().dropna().dt.days.unique()

        self.assertTrue(all(28 <= gap <= 31 for gap in gaps))
        self.assertEqual(series.filled_periods, 3)

    def test_filling_is_reported_rather_than_silent(self):
        frame = pd.concat(
            [daily_sales("2024-01-01", "2024-04-30"), daily_sales("2024-06-01", "2024-12-31")]
        )

        notes = build_trend(frame, detect_roles(frame)).notes

        self.assertTrue(any("counted as zero" in note for note in notes))

    def test_a_complete_timeline_reports_nothing(self):
        frame = daily_sales("2024-01-01", "2025-12-31")

        series = build_trend(frame, detect_roles(frame))

        self.assertEqual(series.filled_periods, 0)
        self.assertEqual(series.notes, ())


class PartialPeriodTests(unittest.TestCase):
    def test_an_extract_cut_mid_month_drops_the_stub(self):
        frame = daily_sales("2024-01-01", "2025-07-12")

        series = build_trend(frame, detect_roles(frame))

        self.assertEqual(series.partial_period, pd.Timestamp("2025-07-01"))
        self.assertEqual(series.partial_coverage, "12 of 31 days")
        self.assertNotIn(pd.Timestamp("2025-07-01"), set(series.frame["Period"]))
        self.assertEqual(series.frame["Period"].max(), pd.Timestamp("2025-06-01"))

    def test_the_exclusion_is_stated_plainly(self):
        frame = daily_sales("2024-01-01", "2025-07-12")

        notes = build_trend(frame, detect_roles(frame)).notes

        self.assertTrue(any("still in progress" in note for note in notes))
        self.assertTrue(any("Jul 2025" in note for note in notes))

    def test_a_stub_month_can_no_longer_read_as_a_collapse(self):
        """The half-month total is roughly half; excluding it removes a fake -60%."""
        frame = daily_sales("2024-01-01", "2025-07-12")
        roles = detect_roles(frame)

        trend = build_trend(frame, roles).frame
        change = (trend["Value"].iloc[-1] - trend["Value"].iloc[-2]) / trend["Value"].iloc[-2] * 100

        self.assertLess(abs(change), 10)

    def test_a_month_that_simply_ends_early_is_kept(self):
        """No orders in the last three days is not the same as a truncated extract."""
        frame = daily_sales("2024-01-01", "2025-06-28")

        series = build_trend(frame, detect_roles(frame))

        self.assertIsNone(series.partial_period)
        self.assertEqual(series.frame["Period"].max(), pd.Timestamp("2025-06-01"))

    def test_sparse_month_end_reporting_is_never_called_partial(self):
        """Monthly snapshots all land on the 1st; that is the pattern, not a stub."""
        dates = pd.date_range("2023-01-01", periods=18, freq="MS")
        frame = pd.DataFrame({"Date": dates, "Revenue": np.linspace(100, 200, 18)})

        series = build_trend(frame, detect_roles(frame))

        self.assertIsNone(series.partial_period)
        self.assertEqual(len(series.frame), 18)

    def test_a_trailing_partial_week_is_caught_too(self):
        frame = daily_sales("2025-01-01", "2025-03-12")  # weekly grain, cut on a Wednesday

        series = build_trend(frame, detect_roles(frame))

        self.assertEqual(series.frequency, "W")
        self.assertEqual(series.partial_period, pd.Timestamp("2025-03-10"))

    def test_a_history_too_short_to_judge_keeps_every_period(self):
        frame = daily_sales("2025-01-01", "2025-01-15")

        series = build_trend(frame, detect_roles(frame))

        self.assertIsNone(series.partial_period)
        self.assertEqual(len(series.frame), 3)

    def test_trend_frame_still_returns_just_the_periods(self):
        frame = daily_sales("2024-01-01", "2025-07-12")
        roles = detect_roles(frame)

        self.assertEqual(list(trend_frame(frame, roles).columns), ["Period", "Value"])
        self.assertTrue(trend_frame(frame, roles).equals(build_trend(frame, roles).frame))

    def test_a_dataframe_without_a_date_yields_an_empty_series(self):
        frame = pd.DataFrame({"Revenue": [1.0, 2.0], "Region": ["West", "East"]})

        series = build_trend(frame, detect_roles(frame))

        self.assertTrue(series.frame.empty)
        self.assertEqual(series.notes, ())


if __name__ == "__main__":
    unittest.main()
