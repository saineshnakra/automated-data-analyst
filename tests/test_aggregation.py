import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from ada.aggregation import (
    build_trend,
    group_coverage,
    heatmap_frame,
    matched_period_change,
    preferred_frequency,
    seasonal_context,
    segment_frame,
    segment_period_change,
    trend_frame,
)
from ada.anomalies import detect_anomalies
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
            rows.extend((month, rate) for rate in (0.1, 0.3, 0.2, 0.2))
        self.frame = pd.DataFrame(rows, columns=["Date", "Conversion Rate"])
        self.frame["Date"] = pd.to_datetime(self.frame["Date"])

    def test_a_rate_is_the_period_average_not_the_period_total(self):
        series = build_trend(self.frame, detect_roles(self.frame))
        np.testing.assert_allclose(series.frame["Value"].to_numpy(), [0.2, 0.2, 0.2])

    def test_an_amount_is_still_the_period_total(self):
        amounts = self.frame.rename(columns={"Conversion Rate": "Revenue"})
        amounts["Revenue"] = [50.0, 150.0, 100.0, 100.0] * 3
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
        frame = pd.DataFrame({"Date": months, measure: [value * (1 + i % 2) for i in range(len(months))]})
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
                "Revenue": [100.0, np.nan, 120.0],
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

    def test_missing_months_of_an_averaged_reading_are_gaps_not_zeros(self):
        months = pd.date_range("2023-01-01", periods=24, freq="MS")
        kept = months.delete([5, 6, 15])
        steps = np.arange(len(kept))
        readings = 400 + 0.2 * steps + 0.3 * (-1.0) ** steps
        frame = pd.DataFrame({"Date": kept, "CO2": readings})

        series = build_trend(frame, detect_roles(frame), frequency="M")

        self.assertEqual(series.filled_periods, 3)
        self.assertFalse(series.filled_as_zero)
        gaps = series.frame.loc[series.frame["Value"].isna(), "Period"]
        self.assertEqual(list(gaps), [months[5], months[6], months[15]])
        self.assertEqual((series.frame["Value"] == 0).sum(), 0)
        # A zero would sit far below the trendline; a gap is simply not checked.
        self.assertEqual(detect_anomalies(series.frame), ())


class AggregationOverrideTests(unittest.TestCase):
    def setUp(self):
        # Two gauges a month, reading 10 and 20 mm: a mean of 15, a total of 30.
        months = pd.date_range("2024-01-01", periods=6, freq="MS").delete(3)
        self.frame = pd.DataFrame(
            {
                "Date": np.repeat(months, 2),
                "Rainfall (mm)": [10.0, 20.0] * len(months),
                "Gauge": ["North", "South"] * len(months),
            }
        )
        self.detected = detect_roles(self.frame)

    def test_the_name_alone_averages_the_reading(self):
        series = build_trend(self.frame, self.detected, frequency="M")
        np.testing.assert_allclose(series.frame["Value"].dropna(), [15.0] * 5)
        self.assertTrue(np.isnan(series.frame["Value"].iloc[3]))

    def test_a_sum_override_totals_each_period_and_counts_empty_ones_as_zero(self):
        roles = replace(self.detected, aggregation="sum")

        series = build_trend(self.frame, roles, frequency="M")

        np.testing.assert_allclose(series.frame["Value"], [30.0, 30.0, 30.0, 0.0, 30.0, 30.0])
        self.assertTrue(series.filled_as_zero)

    def test_segments_follow_the_override(self):
        summed = segment_frame(self.frame, replace(self.detected, aggregation="sum"))
        averaged = segment_frame(self.frame, self.detected)

        self.assertEqual(summed.set_index("Segment")["Value"].to_dict(), {"South": 100.0, "North": 50.0})
        self.assertEqual(averaged.set_index("Segment")["Value"].to_dict(), {"South": 20.0, "North": 10.0})


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


class HeatmapTests(unittest.TestCase):
    def test_an_averaged_measure_leaves_empty_cells_blank(self):
        frame = pd.DataFrame({
            "Date": pd.to_datetime(["2024-01-05", "2024-02-05", "2024-01-06"]),
            "Site": ["North", "North", "South"],
            "pH": [7.1, 7.3, 6.9],
        })
        roles = ColumnRoles(date="Date", measure="pH", dimension="Site", identifier=None,
                            numeric=("pH",), dimensions=("Site",))
        grid = heatmap_frame(frame, roles, frequency="M")
        self.assertTrue(np.isnan(grid.loc["South", pd.Timestamp("2024-02-01")]))
        self.assertAlmostEqual(grid.loc["North", pd.Timestamp("2024-01-01")], 7.1)

    def test_segment_frame_can_keep_every_segment(self):
        frame = pd.DataFrame({"Store": [f"S{i}" for i in range(150)], "Revenue": [1.0] * 150})
        roles = ColumnRoles(date=None, measure="Revenue", dimension="Store", identifier=None,
                            numeric=("Revenue",), dimensions=("Store",))
        self.assertEqual(len(segment_frame(frame, roles, limit=None)), 150)
        self.assertEqual(len(segment_frame(frame, roles)), 12)


def sensor_log(start, periods, freq, *, seed=3):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "Time": pd.date_range(start, periods=periods, freq=freq),
        "Reading": 100 + rng.normal(0, 1, periods),
    })


class SubDayGrainTests(unittest.TestCase):
    def test_a_fifty_minute_log_a_second_apart_is_charted_by_the_minute(self):
        frame = sensor_log("2024-05-01 15:06:28", 2974, "s")

        series = build_trend(frame, detect_roles(frame))

        self.assertEqual(series.frequency, "min")
        self.assertGreaterEqual(len(series.frame), 49)

    def test_a_few_days_of_hourly_readings_get_an_hourly_trend(self):
        frame = sensor_log("2024-05-01", 24 * 4, "h")

        self.assertEqual(preferred_frequency(frame["Time"]), "h")

    def test_three_weeks_of_hourly_readings_are_charted_by_the_day(self):
        frame = sensor_log("2024-05-01", 24 * 21, "h")

        self.assertEqual(preferred_frequency(frame["Time"]), "D")

    def test_a_few_minutes_of_readings_are_charted_by_the_second(self):
        frame = sensor_log("2024-05-01 10:00", 300, "s")

        self.assertEqual(preferred_frequency(frame["Time"]), "s")

    def test_dates_without_a_time_of_day_keep_weeks_as_the_finest_grain(self):
        dates = pd.Series(pd.date_range("2025-01-01", periods=20, freq="D"))

        self.assertEqual(preferred_frequency(dates), "W")

    def test_a_minute_cut_short_reports_seconds(self):
        frame = sensor_log("2024-05-01 15:00:00", 60 * 20 + 2, "s")

        series = build_trend(frame, detect_roles(frame))

        self.assertEqual(series.partial_coverage, "2 of 60 seconds")
        self.assertIn("15:20 is still in progress", series.notes[0])

    def test_sub_day_periods_feed_the_anomaly_check(self):
        frame = sensor_log("2024-05-01 15:00", 3000, "s")
        frame.loc[2400:2459, "Reading"] += 50

        series = build_trend(frame, detect_roles(frame))
        flagged = detect_anomalies(series.frame)

        self.assertEqual([anomaly.period for anomaly in flagged], [pd.Timestamp("2024-05-01 15:40")])


class ShortPeriodNoteTests(unittest.TestCase):
    def test_a_week_ending_on_saturday_covers_six_days(self):
        """Hourly readings to Saturday 23:59: the last week has 6 of its 7 days."""
        stamps = pd.date_range("2024-09-02", "2025-01-04 23:00", freq="h")
        frame = pd.DataFrame({"Time": stamps, "CO": 400.0 + np.arange(len(stamps)) % 5})

        series = build_trend(frame, detect_roles(frame), frequency="W")

        self.assertEqual(series.short_coverage, "6 of 7 days")

    def test_a_period_covered_to_its_last_day_is_never_called_short(self):
        stamps = pd.date_range("2024-09-02", "2025-01-05 01:00", freq="h")
        frame = pd.DataFrame({"Time": stamps, "CO": 400.0 + np.arange(len(stamps)) % 5})

        series = build_trend(frame, detect_roles(frame), frequency="W")

        self.assertEqual(series.short_coverage, "")
        self.assertFalse(any("7 of 7" in note for note in series.notes))


def station_rows(station, months, value):
    return pd.DataFrame({"Date": months, "Station": station, "Rain (mm)": value})


class GroupsAcrossPeriodsTests(unittest.TestCase):
    def setUp(self):
        months = pd.date_range("2024-01-01", periods=6, freq="MS")
        self.frame = pd.concat([
            station_rows("North", months, 10.0),
            station_rows("South", months[:5], 50.0),
            station_rows("East", months[5:], 90.0),
        ], ignore_index=True)
        self.roles = ColumnRoles(
            date="Date", measure="Rain (mm)", dimension="Station", identifier=None,
            numeric=("Rain (mm)",), dimensions=("Station",),
        )

    def test_the_latest_change_is_measured_on_groups_in_both_periods(self):
        series = build_trend(self.frame, self.roles)

        matched = matched_period_change(self.frame, self.roles, series)

        self.assertEqual((matched.previous, matched.current, matched.groups), (10.0, 10.0, 1))
        self.assertEqual(matched.stopped, ("South",))
        self.assertEqual(matched.started, ("East",))

    def test_the_same_groups_in_both_periods_need_no_matching(self):
        frame = self.frame[self.frame["Station"] == "North"]

        self.assertIsNone(matched_period_change(frame, self.roles, build_trend(frame, self.roles)))

    def test_a_group_that_stops_reporting_is_not_a_change(self):
        grouped, _, _ = segment_period_change(self.frame, self.roles)

        self.assertEqual(list(grouped.index), ["North"])
        self.assertEqual(set(grouped.attrs["left_out"]), {"South", "East"})

    def test_a_group_with_no_values_has_no_total(self):
        frame = pd.concat([self.frame, station_rows("Foster", self.frame["Date"].unique(), np.nan)])
        roles = replace(self.roles, aggregation="sum")

        segments = segment_frame(frame, roles, limit=None).set_index("Segment")["Value"]

        self.assertTrue(np.isnan(segments["Foster"]))
        self.assertEqual(segments["North"], 60.0)

    def test_each_group_reports_its_own_dates(self):
        coverage = group_coverage(self.frame, self.roles)

        self.assertEqual(coverage.loc["East", "From"], pd.Timestamp("2024-06-01"))
        self.assertEqual(coverage.loc["South", "To"], pd.Timestamp("2024-05-01"))


def seasonal_months(years=3, *, start="2021-01-01"):
    months = pd.date_range(start, periods=12 * years, freq="MS")
    rng = np.random.default_rng(11)
    cycle = 10 * np.cos(2 * np.pi * (months.month - 1) / 12)
    return pd.DataFrame({"Date": months, "Temperature": 15 + cycle + rng.normal(0, 0.5, len(months))})


class SeasonalContextTests(unittest.TestCase):
    def test_the_latest_month_is_set_beside_the_same_month_a_year_earlier(self):
        frame = seasonal_months(years=2)
        series = build_trend(frame, detect_roles(frame))

        context = seasonal_context(series, relative=False)

        self.assertEqual(series.frequency, "M")
        self.assertEqual(context.year_ago, pd.Timestamp("2021-12-01"))
        self.assertEqual(len(context.usual_changes), 1)
        self.assertGreater(context.usual_changes[0], 0)

    def test_quarters_are_compared_with_the_same_quarter(self):
        frame = seasonal_months(years=4)
        series = build_trend(frame, detect_roles(frame))

        context = seasonal_context(series, relative=False)

        self.assertEqual(series.frequency, "Q")
        self.assertEqual(context.year_ago, pd.Timestamp("2023-10-01"))
        self.assertEqual(len(context.usual_changes), 3)

    def test_a_series_without_a_yearly_pattern_has_no_context(self):
        frame = seasonal_months(years=2)
        frame["Temperature"] = 15 + np.random.default_rng(2).normal(0, 0.5, len(frame))

        self.assertIsNone(seasonal_context(build_trend(frame, detect_roles(frame)), relative=False))


if __name__ == "__main__":
    unittest.main()
