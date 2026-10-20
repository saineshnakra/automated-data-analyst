import unittest

import numpy as np
import pandas as pd

from ada.aggregation import build_trend, measure_aggregation, preferred_frequency, segment_frame, trend_frame
from ada.schema import ColumnRoles, detect_roles

MONTHS = pd.date_range("2024-01-01", periods=12, freq="MS")


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

    def test_annual_figures_are_years_not_zero_filled_quarters(self):
        frame = pd.DataFrame(
            {"Year": pd.to_datetime(["2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01"]),
             "Revenue": [100.0, 120.0, 150.0, 170.0]}
        )

        series = build_trend(frame, detect_roles(frame))

        self.assertEqual(series.frequency, "Y")
        self.assertEqual(series.filled_periods, 0)
        self.assertEqual(len(series.frame), 4)


class RateMeasureTests(unittest.TestCase):
    def setUp(self):
        # Four records a month: a sum and a mean differ by a factor of four.
        rows = []
        for month in ("2025-01-31", "2025-02-28", "2025-03-31"):
            rows.extend((month, 0.2) for _ in range(4))
        self.frame = pd.DataFrame(rows, columns=["Date", "Conversion Rate"])
        self.frame["Date"] = pd.to_datetime(self.frame["Date"])

    def test_measure_aggregation_averages_rates_and_sums_amounts(self):
        self.assertEqual(measure_aggregation("Conversion Rate"), "mean")
        self.assertEqual(measure_aggregation("Revenue"), "sum")
        self.assertEqual(measure_aggregation("Margin Amount"), "sum")

    def test_a_rate_is_the_period_average_not_the_period_total(self):
        series = build_trend(self.frame, detect_roles(self.frame))
        np.testing.assert_allclose(series.frame["Value"].to_numpy(), [0.2, 0.2, 0.2])

    def test_an_amount_is_still_the_period_total(self):
        amounts = self.frame.rename(columns={"Conversion Rate": "Revenue"})
        amounts["Revenue"] = 100.0
        series = build_trend(amounts, detect_roles(amounts))
        np.testing.assert_allclose(series.frame["Value"].to_numpy(), [400.0, 400.0, 400.0])

    def test_a_rate_trend_is_one_of_period_means(self):
        frame = pd.DataFrame(
            {"Date": MONTHS[:4], "Conversion Rate": [0.1, 0.2, 0.3, 0.4], "Channel": ["a", "b"] * 2}
        )

        series = build_trend(frame, detect_roles(frame))

        self.assertAlmostEqual(float(series.frame["Value"].iloc[0]), 0.1)


class EmptyPeriodTests(unittest.TestCase):
    def monthly(self, measure, *, skip):
        """Twelve month-ends with one month absent from the file."""
        months = [m for i, m in enumerate(pd.date_range("2025-01-31", periods=12, freq="ME"))
                  if i != skip]
        value = 0.2 if measure == "Conversion Rate" else 100.0
        frame = pd.DataFrame({"Date": months, measure: [value] * len(months)})
        return build_trend(frame, detect_roles(frame))

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

    def test_a_missing_month_of_a_rate_is_empty_not_zero(self):
        series = self.monthly("Conversion Rate", skip=4)
        gap = series.frame.loc[series.frame["Period"] == pd.Timestamp("2025-05-01"), "Value"]
        self.assertEqual(len(gap), 1)
        self.assertTrue(np.isnan(float(gap.iloc[0])), "an unmeasured period of a rate stays empty, not 0%")
        self.assertFalse(series.filled_as_zero)

    def test_a_missing_month_of_an_amount_counts_as_zero(self):
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
        self.assertTrue(np.isnan(float(february.iloc[0])), "a period with no rows stays empty")

    def test_two_readings_are_not_spread_into_invented_periods(self):
        frame = pd.DataFrame(
            {"Month": pd.to_datetime(["2024-01-31", "2024-02-29"]), "Revenue": [100.0, 120.0]}
        )
        series = build_trend(frame, detect_roles(frame))
        self.assertEqual(len(series.frame), 2)
        self.assertEqual(series.filled_periods, 0)


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

    def test_a_stub_month_does_not_read_as_a_collapse(self):
        """July holds 12 of 31 days, so keeping it would show a drop of about 60%."""
        frame = daily_sales("2024-01-01", "2025-07-12")
        roles = detect_roles(frame)

        trend = build_trend(frame, roles).frame
        change = (trend["Value"].iloc[-1] - trend["Value"].iloc[-2]) / trend["Value"].iloc[-2] * 100

        self.assertLess(abs(change), 10)

    def test_a_month_that_simply_ends_early_is_kept(self):
        """The data stops on 28 June, two days before the month ends."""
        frame = daily_sales("2024-01-01", "2025-06-28")

        series = build_trend(frame, detect_roles(frame))

        self.assertIsNone(series.partial_period)
        self.assertEqual(series.frame["Period"].max(), pd.Timestamp("2025-06-01"))

    def test_sparse_month_end_reporting_is_never_called_partial(self):
        """One row per month dated the 1st, so every month has a single day of data."""
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


class SegmentDenominatorTests(unittest.TestCase):
    def test_blank_segments_stay_in_the_total(self):
        frame = pd.DataFrame(
            {"Date": MONTHS, "Region": ["West", "East"] + [None] * 10,
             "Revenue": [100.0, 200.0] + [90.0] * 10}
        )
        roles = ColumnRoles(date="Date", measure="Revenue", dimension="Region", identifier=None,
                            numeric=("Revenue",), dimensions=("Region",))

        segments = segment_frame(frame, roles)

        self.assertAlmostEqual(float(segments["Value"].sum()), float(frame["Revenue"].sum()))


if __name__ == "__main__":
    unittest.main()
